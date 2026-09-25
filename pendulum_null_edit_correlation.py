"""Null-edit reconstruction and reconstruction/counterfactual-error correlation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms

from data.pendulum_dataset import PENDULUM_ATTRS, PendulumDataset
from evaluation.evaluate_pendulum import PAIEditor, load_scm, raw_to_ca, scm_target
from evaluation.reevaluate_pendulum_saved_ca_regressors import load_predictors


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--ca-root", required=True)
    p.add_argument("--ca-scm", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--joint-config", required=True)
    p.add_argument("--joint-root", required=True)
    p.add_argument("--cf-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def bootstrap_spearman(x, y, draws, seed=42):
    rng = np.random.default_rng(seed)
    values = []
    n = len(x)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        value = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(value):
            values.append(value)
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def main():
    args = args_parser()
    device = torch.device("cuda")
    cf_root = Path(args.cf_root)
    selection = json.loads((cf_root / "selection.json").read_text())
    ids = [int(x) for x in selection["test_indices"]]
    test = PendulumDataset(args.data_root, "test", image_size=256)
    train = PendulumDataset(args.data_root, "train", image_size=256)
    editor = PAIEditor(args.joint_config, args.joint_root, device, args.steps,
                       "joint_label", guidance_scale=1.0)
    predictors = load_predictors(args.ca_root, args.checkpoint_dir, device)
    scm = load_scm(args.ca_root, args.ca_scm, device)

    out = Path(args.output_dir)
    image_dir = out / "null_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    tf96 = transforms.Compose([
        transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    tf128 = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print(f"LPIPS unavailable ({exc}); continuing with pixel MAE")
        lpips_model = None

    null_pixel = np.zeros(len(ids), dtype=np.float64)
    null_lpips = np.full(len(ids), np.nan, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(ids), args.batch_size):
            batch_ids = ids[start:start + args.batch_size]
            items = [test[i] for i in batch_ids]
            paths = [x["path"] for x in items]
            source_unit = torch.stack([x["factors"] for x in items]).to(device)
            recon = editor.edit(paths, source_unit, source_unit)
            source = torch.stack([tf128(Image.open(p).convert("RGB")) for p in paths]).to(device)
            generated = torch.stack([tf128(x) for x in recon]).to(device)
            null_pixel[start:start + len(batch_ids)] = (
                generated - source).abs().mean((1, 2, 3)).cpu().numpy()
            if lpips_model is not None:
                null_lpips[start:start + len(batch_ids)] = lpips_model(
                    source * 2 - 1, generated * 2 - 1).reshape(-1).cpu().numpy()
            for sample_id, image in zip(batch_ids, recon):
                image.save(image_dir / f"{sample_id:05d}.png")

    all_mean = []
    all_diag = []
    all_non_target = []
    all_null_pixel = []
    all_null_lpips = []
    per_do = {}
    with torch.no_grad():
        for do_idx, do_name in enumerate(PENDULUM_ATTRS):
            train_ids = selection["intervention_train_indices"][str(do_idx)]
            rows = []
            for start in range(0, len(ids), 128):
                batch_ids = ids[start:start + 128]
                source_raw = torch.stack([test[i]["raw_factors"] for i in batch_ids]).to(device)
                intervention_raw = source_raw.clone()
                chosen = train_ids[start:start + len(batch_ids)]
                intervention_raw[:, do_idx] = torch.stack([
                    train[int(i)]["raw_factors"][do_idx] for i in chosen]).to(device)
                _, target_raw, _ = scm_target(scm, source_raw, do_idx, intervention_raw)
                target = raw_to_ca(target_raw)
                images = torch.stack([
                    tf96(Image.open(cf_root / "images" / do_name / f"{i:05d}.png").convert("RGB"))
                    for i in batch_ids]).to(device)
                prediction = torch.cat([
                    predictors[name][0](images) for name in PENDULUM_ATTRS], dim=1)
                rows.append((prediction - target).abs().cpu().numpy())
            error = np.concatenate(rows)
            mean_error = error.mean(1)
            diag_error = error[:, do_idx]
            non_target = np.delete(error, do_idx, axis=1).mean(1)
            per_do[do_name] = {
                "mean_four_attribute_mae": float(mean_error.mean()),
                "target_mae": float(diag_error.mean()),
                "non_target_mae": float(non_target.mean()),
            }
            all_mean.extend(mean_error)
            all_diag.extend(diag_error)
            all_non_target.extend(non_target)
            all_null_pixel.extend(null_pixel)
            all_null_lpips.extend(null_lpips)

    all_mean = np.asarray(all_mean)
    all_diag = np.asarray(all_diag)
    all_non_target = np.asarray(all_non_target)
    all_null_pixel = np.asarray(all_null_pixel)
    all_null_lpips = np.asarray(all_null_lpips)
    metrics = {"n_sources": len(ids), "n_counterfactuals": len(all_mean),
               "ddim_steps": args.steps, "per_intervention": per_do, "correlations": {}}
    for recon_name, recon in [("pixel_mae", all_null_pixel), ("lpips", all_null_lpips)]:
        valid = np.isfinite(recon)
        if not valid.any():
            continue
        for error_name, error in [("four_attribute_mae", all_mean),
                                  ("target_mae", all_diag),
                                  ("non_target_mae", all_non_target)]:
            rho = float(spearmanr(recon[valid], error[valid]).statistic)
            metrics["correlations"][f"{recon_name}_vs_{error_name}"] = {
                "spearman_rho": rho,
                "bootstrap_95_ci": bootstrap_spearman(
                    recon[valid], error[valid], args.bootstrap),
            }
    np.savez_compressed(out / "per_sample_errors.npz", null_pixel_mae=null_pixel,
                        null_lpips=null_lpips, cf_four_attribute_mae=all_mean,
                        cf_target_mae=all_diag, cf_non_target_mae=all_non_target)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
