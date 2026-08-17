from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset  # noqa: E402
from evaluation.evaluate_latent_label_film_celeba import (  # noqa: E402
    COMPLEX_ATTRS,
    FilenameSubset,
    LabelFilmEditor,
    _labels_pm1,
    _load_yaml,
)
from models.full_label_scm import FullLabelSCM  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/latent_lora_celeba_ca_pipeline_label_film.yaml")
    p.add_argument("--selection-json", required=True)
    p.add_argument(
        "--scm-checkpoint",
        default="/scratch/gilbreth/yu1331/ckpts/celeba/full_label_scm/causal_adapter_graph/checkpoints/full_label_scm_best.pt",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-each", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def load_indices(path: str, n_each: int):
    with open(path) as f:
        raw = json.load(f)
    return [int(x) for x in raw["male_indices"][:n_each]] + [
        int(x) for x in raw["female_indices"][:n_each]
    ]


def make_cf(scm, c_orig, male_idx):
    target = -c_orig[:, male_idx]
    return torch.cat(
        [
            scm.intervene_pm1(c_orig[i : i + 1], male_idx, target[i].item())
            for i in range(c_orig.shape[0])
        ],
        dim=0,
    )


def pil_from_tensor(x):
    x = (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def label_str(vals):
    return " ".join(f"{k[0]}:{'+' if float(v) > 0 else '-'}" for k, v in zip(COMPLEX_ATTRS, vals))


@torch.no_grad()
def decode256(editor: LabelFilmEditor, latents):
    img = editor.pipe.vae.decode(latents / editor.pipe.vae.config.scaling_factor).sample
    return (img / 2 + 0.5).clamp(0, 1).float().cpu()


@torch.no_grad()
def generate_pair(editor: LabelFilmEditor, c_orig, c_cf, generator):
    bsz = c_orig.shape[0]
    latent_shape = (
        bsz,
        editor.pipe.unet.config.in_channels,
        int(editor.cfg.get("image_size", 256)) // editor.pipe.vae_scale_factor,
        int(editor.cfg.get("image_size", 256)) // editor.pipe.vae_scale_factor,
    )
    latents = torch.randn(
        latent_shape,
        generator=generator,
        device=editor.device,
        dtype=editor.pipe.unet.dtype,
    )
    latents = latents * editor.scheduler.init_noise_sigma
    pe, ne = editor._prompt_embeds(bsz)

    t = editor.scheduler.timesteps[len(editor.scheduler.timesteps) // 2]
    eps_orig = editor._noise_pred(latents, t, c_orig.to(editor.device), pe, ne).float()
    eps_cf = editor._noise_pred(latents, t, c_cf.to(editor.device), pe, ne).float()
    eps_delta = (eps_orig - eps_cf).abs().flatten(1)
    eps_base = eps_orig.abs().flatten(1).mean(dim=1).clamp_min(1e-8)

    out_orig = editor.sample(latents.clone(), c_orig.to(editor.device), pe, ne)
    out_cf = editor.sample(latents.clone(), c_cf.to(editor.device), pe, ne)
    img_orig = decode256(editor, out_orig)
    img_cf = decode256(editor, out_cf)
    pix_mse = ((img_orig - img_cf) ** 2).mean(dim=(1, 2, 3))
    return img_orig, img_cf, {
        "noise_abs_delta_mean": eps_delta.mean(dim=1).detach().cpu().tolist(),
        "noise_abs_delta_max": eps_delta.max(dim=1).values.detach().cpu().tolist(),
        "noise_relative_delta_mean": (eps_delta.mean(dim=1) / eps_base).detach().cpu().tolist(),
        "generated_pair_pixel_mse": pix_mse.detach().cpu().tolist(),
    }


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = _load_yaml(cfg_path)
    run_root = Path(cfg["output_dir"]) / str(cfg.get("run_name"))
    lora_dir = run_root / "lora"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = load_indices(args.selection_json, args.num_each)
    ds = CelebADataset(
        root=cfg["data_root"],
        split="val",
        factor_cols=list(CELEBA_ATTR_ORDER),
        image_size=64,
    )
    loader = DataLoader(
        FilenameSubset(ds, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scm = FullLabelSCM().to(device).eval()
    ck = torch.load(args.scm_checkpoint, map_location=device)
    scm.load_state_dict(ck.get("model_state_dict", ck))
    editor = LabelFilmEditor(cfg, lora_dir, device, args.steps, args.guidance_scale)
    male_idx = COMPLEX_ATTRS.index("Male")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    rows = []
    items = []
    cursor = 0
    for batch in loader:
        c_orig = _labels_pm1(batch, device)
        c_cf = make_cf(scm, c_orig, male_idx)
        img_orig, img_cf, stats = generate_pair(editor, c_orig, c_cf, generator)
        for j, fn in enumerate(batch["filename"]):
            left = pil_from_tensor(img_orig[j])
            right = pil_from_tensor(img_cf[j])
            row = Image.new("RGB", (512, 292), "white")
            row.paste(left, (0, 0))
            row.paste(right, (256, 0))
            d = ImageDraw.Draw(row)
            src = c_orig[j].detach().cpu().tolist()
            tgt = c_cf[j].detach().cpu().tolist()
            d.text(
                (4, 258),
                f"{cursor:02d} idx={indices[cursor]} file={fn}\norig {label_str(src)}\ncf   {label_str(tgt)}",
                fill=(0, 0, 0),
            )
            rows.append(row)
            item = {
                "slot": cursor,
                "index": indices[cursor],
                "filename": str(fn),
                "orig_pm1": src,
                "cf_pm1": tgt,
            }
            for k, vals in stats.items():
                item[k] = vals[j]
            items.append(item)
            cursor += 1

    cols = 4
    w, h = rows[0].size
    grid = Image.new("RGB", (cols * w, ((len(rows) + cols - 1) // cols) * h), "white")
    for i, row in enumerate(rows):
        grid.paste(row, ((i % cols) * w, (i // cols) * h))
    grid_path = out_dir / "conditioning_noise_orig_vs_cf_grid.png"
    grid.save(grid_path)

    summary = {
        "num_samples": len(items),
        "seed": args.seed,
        "selection_json": args.selection_json,
        "grid": str(grid_path),
        "mean_noise_abs_delta": float(
            sum(x["noise_abs_delta_mean"] for x in items) / max(len(items), 1)
        ),
        "mean_noise_relative_delta": float(
            sum(x["noise_relative_delta_mean"] for x in items) / max(len(items), 1)
        ),
        "mean_generated_pair_pixel_mse": float(
            sum(x["generated_pair_pixel_mse"] for x in items) / max(len(items), 1)
        ),
        "items": items,
    }
    with open(out_dir / "conditioning_diagnostic.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()
