"""Test whether label-only learned one Pendulum factor with/without inversion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from data.pendulum_dataset import PENDULUM_ATTRS, PENDULUM_MINMAX, PendulumDataset, denormalize_pendulum
from evaluation.evaluate_pendulum import PAIEditor, load_regressor, pil_to_tensor, tensor_to_pil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--regressor", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attribute", choices=PENDULUM_ATTRS, default="shadow_position")
    return p.parse_args()


def make_grid(images, predictions, sources, targets, output, title):
    cell, label_h = 176, 52
    rows, cols = len(images), len(targets)
    canvas = Image.new("RGB", (cell * cols, 28 + rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill="black")
    for r in range(rows):
        for c, target in enumerate(targets):
            x, y = c * cell, 28 + r * (cell + label_h)
            canvas.paste(images[r][c].resize((cell, cell)), (x, y))
            draw.multiline_text(
                (x + 3, y + cell + 2),
                f"target={target:.1f} pred={predictions[r][c]:.2f}\nsource={sources[r]:.2f}",
                fill="black",
            )
    canvas.save(output)


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    ds = PendulumDataset(args.data_root, "test", image_size=256)
    ids = list(range(args.samples))
    items = [ds[i] for i in ids]
    paths = [x["path"] for x in items]
    source_unit = torch.stack([x["factors"] for x in items]).to(device)
    source_raw = torch.stack([x["raw_factors"] for x in items]).to(device)
    attr_idx = PENDULUM_ATTRS.index(args.attribute)
    lo, hi = map(float, PENDULUM_MINMAX[attr_idx])
    targets = np.linspace(lo, hi, 5).tolist()

    editor = PAIEditor(args.config, args.checkpoint_root, device, args.steps,
                       "joint_label", args.guidance_scale)
    inner, pipe = editor.inner, editor.inner.pipe
    regressor = load_regressor(args.regressor, device)

    # Abduct each source once. Every target is decoded from the same inverted latent.
    x = torch.stack([inner.image_tfm(Image.open(p).convert("RGB")) for p in paths]).to(
        device, dtype=pipe.vae.dtype
    )
    lat0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pe_src, neg = inner.condition_embeds(source_unit)
    inverted = inner.invert(lat0, pe_src, neg)

    # Use one fixed random latent per row for the direct-generation comparison.
    gen = torch.Generator(device=device).manual_seed(args.seed)
    random_latent = torch.randn(inverted.shape, generator=gen, device=device, dtype=inverted.dtype)
    random_latent = random_latent * getattr(inner.scheduler, "init_noise_sigma", 1.0)

    direct_rows = [[] for _ in ids]
    inversion_rows = [[] for _ in ids]
    direct_preds = [[] for _ in ids]
    inversion_preds = [[] for _ in ids]
    records = []
    for target in targets:
        target_unit = source_unit.clone()
        target_unit[:, attr_idx] = float((target - lo) / (hi - lo))
        pe_target, _ = inner.condition_embeds(target_unit)
        direct = tensor_to_pil(inner.decode64(inner.sample(random_latent.clone(), pe_target, neg)))
        edited = tensor_to_pil(inner.decode64(inner.sample(inverted.clone(), pe_target, neg)))
        with torch.no_grad():
            pred_direct = denormalize_pendulum(regressor(pil_to_tensor(direct).to(device)))[:, attr_idx].cpu()
            pred_edited = denormalize_pendulum(regressor(pil_to_tensor(edited).to(device)))[:, attr_idx].cpu()
        for r in range(len(ids)):
            direct_rows[r].append(direct[r]); inversion_rows[r].append(edited[r])
            direct_preds[r].append(float(pred_direct[r])); inversion_preds[r].append(float(pred_edited[r]))
        records.append({"target": target, "direct_predictions": pred_direct.tolist(),
                        "inversion_predictions": pred_edited.tolist()})

    source_values = source_raw[:, attr_idx].cpu().tolist()
    make_grid(direct_rows, direct_preds, source_values, targets, Path(args.output_dir) / "direct_generation.png",
              f"{args.attribute}; fixed random latent; label-only; CFG={args.guidance_scale}")
    make_grid(inversion_rows, inversion_preds, source_values, targets, Path(args.output_dir) / "ddim_inversion_edit.png",
              f"{args.attribute}; fixed DDIM-inverted source latent; label-only; CFG={args.guidance_scale}")

    target_np = np.asarray(targets)
    summary = {"attribute": args.attribute, "targets": targets, "source_indices": ids,
               "source_values": source_values,
               "guidance_scale": args.guidance_scale, "records": records, "per_sample": []}
    for r in range(len(ids)):
        d = np.asarray(direct_preds[r]); e = np.asarray(inversion_preds[r])
        summary["per_sample"].append({
            "index": ids[r], "direct_predictions": d.tolist(), "inversion_predictions": e.tolist(),
            "direct_slope": float(np.polyfit(target_np, d, 1)[0]),
            "inversion_slope": float(np.polyfit(target_np, e, 1)[0]),
            "direct_range": float(d.max() - d.min()), "inversion_range": float(e.max() - e.min()),
        })
    summary["mean_direct_slope"] = float(np.mean([x["direct_slope"] for x in summary["per_sample"]]))
    summary["mean_inversion_slope"] = float(np.mean([x["inversion_slope"] for x in summary["per_sample"]]))
    summary["mean_direct_range"] = float(np.mean([x["direct_range"] for x in summary["per_sample"]]))
    summary["mean_inversion_range"] = float(np.mean([x["inversion_range"] for x in summary["per_sample"]]))
    (Path(args.output_dir) / "response.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("mean_")}, indent=2))


if __name__ == "__main__":
    main()
