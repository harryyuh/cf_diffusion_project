"""Generate CelebA Male<->Female grids with official Causal-Adapter on an existing selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]


def _add_ca_paths(ca_root: Path) -> None:
    for p in (ca_root / "diffusers" / "src", ca_root / "notebook_benchmarks", ca_root):
        sp = str(p.resolve())
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)


def _patch_safety_checker_compat() -> None:
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    except Exception:
        return
    if not hasattr(StableDiffusionSafetyChecker, "all_tied_weights_keys"):
        StableDiffusionSafetyChecker.all_tied_weights_keys = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--causal-adapter-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter")
    p.add_argument("--base-model-path", default="lambdalabs/miniSD-diffusers")
    p.add_argument("--controlnet-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/controlnet/controlnet-steps-200000.safetensors")
    p.add_argument("--text-embedding-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/controlnet/learned_embeds-steps-200000.safetensors")
    p.add_argument("--scm-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt")
    p.add_argument("--selection-json", required=True)
    p.add_argument("--celeba-root", default="/scratch/gilbreth/yu1331/datasets/celebA")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--invert-guidance-scale", type=float, default=1.0)
    p.add_argument("--editing", choices=["p2p", "standard"], default="p2p")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    return p.parse_args()


def _load_selection(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_attrs(root: Path) -> Dict[str, Dict[str, float]]:
    attr_path = root / "list_attr_celeba.txt"
    lines = attr_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    cols = lines[1].split()
    out: Dict[str, Dict[str, float]] = {}
    for line in lines[2:]:
        parts = line.split()
        if len(parts) != len(cols) + 1:
            continue
        out[parts[0]] = {c: float(v) for c, v in zip(cols, parts[1:])}
    return out


def _label01(attrs_by_file: Dict[str, Dict[str, float]], fname: str) -> torch.Tensor:
    vals = [attrs_by_file[fname][k] for k in ["Young", "Male", "No_Beard", "Bald"]]
    vals = [1.0 if v > 0 else 0.0 for v in vals]
    return torch.tensor(vals, dtype=torch.float32).view(1, 4)


def _target_for(label01: torch.Tensor, male_value: float) -> torch.Tensor:
    target = label01.clone()
    target[:, 1] = float(male_value)
    return target


def _to_tile(img: Image.Image, size: int = 64) -> torch.Tensor:
    return transforms.ToTensor()(img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC))


def _save_grid(tiles: List[torch.Tensor], path: Path) -> None:
    from utils.visualization import save_image_grid

    grid = torch.stack(tiles, dim=0)
    save_image_grid(grid, nrow=2, path=path, cmap=None, dpi=100.0)


def main() -> None:
    args = parse_args()
    ca_root = Path(args.causal_adapter_root)
    _add_ca_paths(ca_root)
    _patch_safety_checker_compat()

    from inference_utils import build_transforms, load_causal_adapter  # type: ignore
    from causal_modules.ddim_modules import P2P_editing, ddim_editing  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    assets = load_causal_adapter(
        "celeA_complex",
        base_model_path=args.base_model_path,
        controlnet_path=args.controlnet_path,
        text_embedding_path=args.text_embedding_path,
        scm_path=args.scm_path or None,
        device=device,
        torch_dtype=dtype,
    )
    image_tfm, original_tfm, _ = build_transforms("celeA_complex", size=256)

    sel = _load_selection(Path(args.selection_json))
    root = Path(args.celeba_root)
    attrs = _read_attrs(root)
    img_dir = root / "img_align_celeba"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blend_presets = {
        1: dict(blend_params={"start_blend": 0.0, "th": (0.3, 0.5)}, cross_replace_steps=0.2, self_replace_steps=0.2),
    }

    def run_grid(tag: str, filenames: List[str], male_target: float) -> None:
        tiles: List[torch.Tensor] = []
        for fname in filenames:
            pil = Image.open(img_dir / fname).convert("RGB")
            source_preview = original_tfm(pil.copy())
            image_t = image_tfm(pil.copy()).unsqueeze(0).to(device=device, dtype=dtype)
            label = _label01(attrs, fname).to(device=device, dtype=dtype)
            target_labels = _target_for(label, male_target).unsqueeze(2)
            if args.editing == "p2p":
                cfg = blend_presets[1]
                images, *_ = P2P_editing(
                    assets.pipe,
                    image_t,
                    label,
                    assets.presudo_token_ids,
                    assets.prompt,
                    assets.presudo_list,
                    num_steps=args.num_steps,
                    invert_guidance_scale=args.invert_guidance_scale,
                    set_guidance_scale=args.guidance_scale,
                    intervention_indx=1,
                    intervention_values=torch.tensor(float(male_target), device=device, dtype=dtype),
                    return_PIL=True,
                    blend_word=True,
                    blend_params=cfg["blend_params"],
                    disentangle=False,
                    cross_replace_steps=cfg["cross_replace_steps"],
                    self_replace_steps=cfg["self_replace_steps"],
                    DSCM_labels=None,
                )
                cf_pil = images[-1]
            else:
                images, *_ = ddim_editing(
                    assets.pipe,
                    image_t,
                    label,
                    assets.presudo_token_ids,
                    assets.prompt,
                    num_steps=args.num_steps,
                    invert_guidance_scale=args.invert_guidance_scale,
                    set_guidance_scale=args.guidance_scale,
                    intervention_indx=None,
                    intervention_values=None,
                    return_PIL=True,
                    DSCM_labels=target_labels,
                )
                cf_pil = images[-1]
            tiles.append(_to_tile(source_preview))
            tiles.append(_to_tile(cf_pil))
        _save_grid(tiles, out_dir / f"grid_{tag}.png")

    run_grid("male_to_female", list(sel["male_filenames"]), 0.0)
    run_grid("female_to_male", list(sel["female_filenames"]), 1.0)

    with open(out_dir / "selection.json", "w") as f:
        json.dump(sel, f, indent=2)
    meta = {
        "schema_version": 1,
        "script": "generate_causal_adapter_celeba_counterfactual_male.py",
        "selection_json": str(Path(args.selection_json).resolve()),
        "generation": {
            "editing": args.editing,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "invert_guidance_scale": args.invert_guidance_scale,
            "grid_files": ["grid_male_to_female.png", "grid_female_to_male.png"],
        },
        "model": {
            "base_model_path": args.base_model_path,
            "controlnet_path": args.controlnet_path,
            "text_embedding_path": args.text_embedding_path,
            "scm_path": args.scm_path,
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote Causal-Adapter grids under {out_dir}")


if __name__ == "__main__":
    main()
