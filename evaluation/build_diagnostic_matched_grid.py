"""Build compact, annotated Pendulum grids from already generated images.

Rows are ranked by the absolute difference in target-attribute error between
two methods.  This makes the visual diagnostic correspond exactly to the MAE
reported by evaluate_pendulum.py, without rerunning diffusion sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from data.pendulum_dataset import PENDULUM_ATTRS, PendulumDataset, denormalize_pendulum
from evaluation.evaluate_pendulum import load_regressor, load_scm, scm_target


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--ca-root", required=True)
    p.add_argument("--ca-scm", required=True)
    p.add_argument("--regressor", required=True)
    p.add_argument("--method", action="append", nargs=2, metavar=("NAME", "EVAL_ROOT"), required=True)
    p.add_argument("--reference-a", default="ca")
    p.add_argument("--reference-b", default="joint_label")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--counterexamples", type=int, default=2)
    return p.parse_args()


def predict(model, paths, device, batch=64):
    tfm = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    out = []
    for start in range(0, len(paths), batch):
        x = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths[start:start + batch]]).to(device)
        with torch.no_grad():
            out.append(denormalize_pendulum(model(x)).cpu())
    return torch.cat(out)


def vec(v):
    return "[" + ", ".join(f"{float(x):.2f}" for x in v) + "]"


def main():
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = {name: Path(root) for name, root in args.method}
    if args.reference_a not in roots or args.reference_b not in roots:
        raise ValueError("Both ranking references must be supplied with --method")
    selection = json.loads((roots[args.reference_a] / "selection.json").read_text())
    for name, root in roots.items():
        other = json.loads((root / "selection.json").read_text())
        if other != selection:
            raise RuntimeError(f"Selection mismatch for {name}; refusing an unmatched grid")

    test = PendulumDataset(args.data_root, "test", image_size=256)
    train = PendulumDataset(args.data_root, "train", image_size=256)
    scm = load_scm(args.ca_root, args.ca_scm, device)
    reg = load_regressor(args.regressor, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}

    ids = selection["test_indices"]
    source_raw = torch.stack([test[i]["raw_factors"] for i in ids]).to(device)
    for do_idx, do_name in enumerate(PENDULUM_ATTRS):
        intervention_raw = source_raw.clone()
        train_ids = selection["intervention_train_indices"][str(do_idx)] if str(do_idx) in selection["intervention_train_indices"] else selection["intervention_train_indices"][do_idx]
        intervention_raw[:, do_idx] = torch.stack([train[i]["raw_factors"][do_idx] for i in train_ids]).to(device)
        _, target_raw, _ = scm_target(scm, source_raw, do_idx, intervention_raw)
        target_raw = target_raw.cpu()
        image_paths = {
            name: [root / "images" / do_name / f"{i:05d}.png" for i in ids]
            for name, root in roots.items()
        }
        preds = {name: predict(reg, paths, device) for name, paths in image_paths.items()}
        errors = {name: (pred[:, do_idx] - target_raw[:, do_idx]).abs() for name, pred in preds.items()}
        gap = errors[args.reference_b] - errors[args.reference_a]
        worse_b = torch.argsort(gap, descending=True)[:args.top].tolist()
        better_b = torch.argsort(gap)[:args.counterexamples].tolist()
        selected = worse_b + [i for i in better_b if i not in worse_b]

        cell, text_h = 176, 94
        columns = ["source"] + list(roots)
        canvas = Image.new("RGB", (cell * len(columns), (cell + text_h) * len(selected) + 30), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 7), f"do({do_name}); ranked by target-error gap {args.reference_b} - {args.reference_a}", fill="black")
        rows = []
        for row, pos in enumerate(selected):
            y = 30 + row * (cell + text_h)
            src = Image.open(test[ids[pos]]["path"]).convert("RGB")
            images = [src] + [Image.open(image_paths[name][pos]).convert("RGB") for name in roots]
            for col, (label, image) in enumerate(zip(columns, images)):
                x = col * cell
                canvas.paste(image.resize((cell, cell)), (x, y))
                draw.text((x + 3, y + 3), label, fill="black", stroke_width=2, stroke_fill="white")
                if label == "source":
                    note = f"src {vec(source_raw[pos].cpu())}\ndo={float(intervention_raw[pos, do_idx]):.2f}\ntgt {vec(target_raw[pos])}"
                else:
                    note = f"pred {vec(preds[label][pos])}\ntgt err={float(errors[label][pos]):.3f}"
                draw.multiline_text((x + 3, y + cell + 3), note, fill="black", spacing=2)
            rows.append({
                "test_index": ids[pos], "source": source_raw[pos].cpu().tolist(),
                "do_value": float(intervention_raw[pos, do_idx]), "target": target_raw[pos].tolist(),
                "predictions": {name: preds[name][pos].tolist() for name in roots},
                "target_errors": {name: float(errors[name][pos]) for name in roots},
                "ranking_gap": float(gap[pos]),
            })
        canvas.save(output / f"diagnostic_{do_name}.png")
        manifest[do_name] = rows
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
