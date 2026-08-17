from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms

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
    p.add_argument("--num-each", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
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


@torch.no_grad()
def edit256(editor: LabelFilmEditor, data_root: Path, filenames, c_orig, c_cf):
    lat0 = editor.encode_images(data_root, filenames)
    pe, ne = editor._prompt_embeds(lat0.shape[0])
    lat_t = editor.invert(lat0, c_orig.to(editor.device), pe, ne)
    lat_cf = editor.sample(lat_t, c_cf.to(editor.device), pe, ne)
    img = editor.pipe.vae.decode(lat_cf / editor.pipe.vae.config.scaling_factor).sample
    return (img / 2 + 0.5).clamp(0, 1).float().cpu()


def pil_from_tensor(x):
    x = (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def label_str(vals):
    parts = []
    for k, v in zip(COMPLEX_ATTRS, vals):
        parts.append(f"{k[0]}:{'+' if float(v) > 0 else '-'}")
    return " ".join(parts)


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = _load_yaml(cfg_path)
    data_root = Path(cfg["data_root"])
    run_root = Path(cfg["output_dir"]) / str(cfg.get("run_name"))
    lora_dir = run_root / "lora"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs").mkdir(exist_ok=True)

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

    orig_tfm = transforms.Compose(
        [
            transforms.CenterCrop(150),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    rows = []
    meta = []
    cursor = 0
    for batch in loader:
        filenames = batch["filename"]
        c_orig = _labels_pm1(batch, device)
        c_cf = make_cf(scm, c_orig, male_idx)
        fake = edit256(editor, data_root, filenames, c_orig, c_cf)
        for j, fn in enumerate(filenames):
            orig = orig_tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB"))
            orig_pil = pil_from_tensor(orig)
            fake_pil = pil_from_tensor(fake[j])
            pair = Image.new("RGB", (512, 292), "white")
            pair.paste(orig_pil, (0, 0))
            pair.paste(fake_pil, (256, 0))
            d = ImageDraw.Draw(pair)
            src = c_orig[j].detach().cpu().tolist()
            tgt = c_cf[j].detach().cpu().tolist()
            caption = (
                f"{cursor:02d} idx={indices[cursor]} file={fn}\n"
                f"orig {label_str(src)}\n"
                f"cf   {label_str(tgt)}"
            )
            d.text((4, 258), caption, fill=(0, 0, 0))
            pair_path = out_dir / "pairs" / f"{cursor:02d}_{Path(str(fn)).stem}_idx{indices[cursor]}.png"
            pair.save(pair_path)
            rows.append(pair)
            meta.append(
                {
                    "slot": cursor,
                    "index": indices[cursor],
                    "filename": str(fn),
                    "orig_pm1": src,
                    "cf_pm1": tgt,
                    "pair": str(pair_path),
                }
            )
            cursor += 1

    cols = 4
    w, h = rows[0].size
    grid = Image.new("RGB", (cols * w, ((len(rows) + cols - 1) // cols) * h), "white")
    for i, im in enumerate(rows):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    grid_path = out_dir / "grid_32_orig_cf.png"
    grid.save(grid_path)
    with open(out_dir / "grid_32_metadata.json", "w") as f:
        json.dump({"indices": indices, "items": meta, "grid": str(grid_path)}, f, indent=2)
    print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()
