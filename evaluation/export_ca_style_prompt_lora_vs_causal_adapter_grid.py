"""Export CA-style orig/counterfactual grids for Causal-Adapter and prompt-LoRA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset, expand_env_in_cfg  # noqa: E402
from evaluation.evaluate_latent_prompt_lora_celeba import (  # noqa: E402
    COMPLEX_ATTRS,
    FilenameSubset,
    PromptLoraEditor,
    _causal_adapter_intervene_pm1,
    _labels_pm1,
    _load_causal_adapter_scm,
    _prompts_from_pm1,
)
from utils.visualization import save_image_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/latent_lora_celeba_ca_pipeline_four_label_prompt_balanced.yaml")
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--selection-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--intervention-attr", choices=COMPLEX_ATTRS, required=True)
    p.add_argument("--num-each", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--invert-guidance-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--scheduler-mode", choices=["pretrained", "ca"], default="pretrained")
    p.add_argument("--vae-latent-mode", choices=["sample", "mean"], default="sample")
    p.add_argument("--editing", choices=["standard", "ptp"], default="standard")
    p.add_argument("--condition-mode", choices=["prompt", "pai", "joint_label"], default="prompt")
    p.add_argument("--pai-checkpoint", default="")
    p.add_argument("--inversion", choices=["ddim", "direct"], default="ddim")
    p.add_argument("--ptp-cross-replace-steps", type=float, default=0.2)
    p.add_argument("--ptp-self-replace-steps", type=float, default=0.2)
    p.add_argument("--ptp-controller", choices=["simple", "ca", "ca_refine"], default="simple")
    p.add_argument("--ptp-local-blend", action="store_true")
    p.add_argument("--ptp-blend-th", type=float, nargs=2, default=(0.3, 0.3))
    p.add_argument("--ptp-start-blend", type=float, default=0.0)
    p.add_argument("--ptp-substruct-attr-indices", type=int, nargs="*", default=())
    p.add_argument("--causal-adapter-root", default="/home/yu1331/cf-diffusion-celeba/external/Causal-Adapter")
    p.add_argument("--ca-base-model-path", default="lambdalabs/miniSD-diffusers")
    p.add_argument("--ca-controlnet-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/controlnet/controlnet-steps-200000.safetensors")
    p.add_argument("--ca-text-embedding-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/controlnet/learned_embeds-steps-200000.safetensors")
    p.add_argument("--ca-scm-path", default="/scratch/gilbreth/yu1331/models/Causal-Adapter/celeba/scm/best_model.pt")
    p.add_argument("--ca-steps", type=int, default=50)
    p.add_argument("--ca-guidance-scale", type=float, default=3.0)
    p.add_argument("--ca-invert-guidance-scale", type=float, default=1.0)
    p.add_argument("--ca-editing", choices=["ptp", "standard"], default="ptp")
    p.add_argument("--export-ptp-masks", action="store_true")
    p.add_argument("--ptp-mask-steps", type=int, nargs="+", default=(0, 4, 9, 24, 49))
    p.add_argument("--skip-ca-grid", action="store_true")
    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def _load_selection(path: Path, n_each: int) -> Dict[str, List[Any]]:
    raw = json.loads(path.read_text())
    return {
        "indices": [int(x) for x in raw["male_indices"][:n_each]] + [int(x) for x in raw["female_indices"][:n_each]],
        "filenames": [str(x) for x in raw["male_filenames"][:n_each]] + [str(x) for x in raw["female_filenames"][:n_each]],
    }


def _read_attrs(root: Path) -> Dict[str, Dict[str, int]]:
    lines = (root / "list_attr_celeba.txt").read_text().strip().splitlines()
    cols = lines[1].split()
    out: Dict[str, Dict[str, int]] = {}
    for line in lines[2:]:
        parts = line.split()
        if len(parts) != 1 + len(cols):
            continue
        out[parts[0]] = {c: int(v) for c, v in zip(cols, parts[1:])}
    return out


def _add_ca_paths(ca_root: Path) -> None:
    bench_root = ca_root / "counterfactual-benchmark" / "counterfactual_benchmark"
    for p in (ca_root / "diffusers" / "src", ca_root / "notebook_benchmarks", ca_root, bench_root):
        sp = str(p.resolve())
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)
    for name in list(sys.modules):
        if name == "diffusers" or name.startswith("diffusers."):
            del sys.modules[name]


def _patch_safety_checker_compat() -> None:
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    except Exception:
        return
    if not hasattr(StableDiffusionSafetyChecker, "all_tied_weights_keys"):
        StableDiffusionSafetyChecker.all_tied_weights_keys = {}


def _label01(attrs: Dict[str, Dict[str, int]], fname: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    row = attrs[fname]
    return torch.tensor([[1.0 if row[k] > 0 else 0.0 for k in COMPLEX_ATTRS]], device=device, dtype=dtype)


def _to_tile(img: Image.Image, size: int = 64) -> torch.Tensor:
    return transforms.ToTensor()(img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC))


@torch.no_grad()
def _prompt_lora_tiles(
    editor: PromptLoraEditor,
    data_root: Path,
    loader: DataLoader,
    ca_scm,
    intervention_idx: int,
    editing: str,
    inversion: str,
    ptp_cross_replace_steps: float,
    ptp_self_replace_steps: float,
    mask_batches=None,
    source_tiles=None,
) -> List[torch.Tensor]:
    tiles: List[torch.Tensor] = []
    orig_tfm = transforms.Compose(
        [
            transforms.CenterCrop(150),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    for batch in loader:
        filenames = batch["filename"]
        c_orig = _labels_pm1(batch, editor.device)
        c_cf = _causal_adapter_intervene_pm1(ca_scm, c_orig, intervention_idx)
        fake = editor.edit(
            data_root,
            filenames,
            c_orig,
            c_cf,
            editing=editing,
            inversion=inversion,
            ptp_cross_replace_steps=ptp_cross_replace_steps,
            ptp_self_replace_steps=ptp_self_replace_steps,
        ).float().cpu()
        if mask_batches is not None:
            for step, masks in editor.last_ptp_masks.items():
                mask_batches.setdefault(step, []).append(masks)
        for j, fn in enumerate(filenames):
            orig = orig_tfm(Image.open(data_root / "img_align_celeba" / str(fn)).convert("RGB"))
            orig64 = torch.nn.functional.interpolate(
                orig.unsqueeze(0), size=(64, 64), mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0, 1)
            tiles.append(orig64)
            if source_tiles is not None:
                source_tiles.append(orig64)
            tiles.append(fake[j].clamp(0, 1))
    return tiles


@torch.no_grad()
def _causal_adapter_tiles(args: argparse.Namespace, data_root: Path, filenames: List[str], intervention_idx: int) -> List[torch.Tensor]:
    _add_ca_paths(Path(args.causal_adapter_root))
    _patch_safety_checker_compat()
    from inference_utils import build_transforms, load_causal_adapter  # type: ignore
    from causal_modules.ddim_modules import P2P_editing, ddim_editing  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    assets = load_causal_adapter(
        "celeA_complex",
        base_model_path=args.ca_base_model_path,
        controlnet_path=args.ca_controlnet_path,
        text_embedding_path=args.ca_text_embedding_path,
        scm_path=args.ca_scm_path,
        device=device,
        torch_dtype=dtype,
    )
    image_tfm, original_tfm, _ = build_transforms("celeA_complex", size=256)
    attrs = _read_attrs(data_root)
    img_dir = data_root / "img_align_celeba"
    tiles: List[torch.Tensor] = []
    blend_params = {"start_blend": 0.0, "th": tuple(args.ptp_blend_th)}
    for fname in filenames:
        pil = Image.open(img_dir / fname).convert("RGB")
        source_preview = original_tfm(pil.copy())
        image_t = image_tfm(pil.copy()).unsqueeze(0).to(device=device, dtype=dtype)
        label = _label01(attrs, fname, device, dtype)
        intervention_value = 1.0 - label[:, intervention_idx]
        if args.ca_editing == "ptp":
            images, *_ = P2P_editing(
                assets.pipe,
                image_t,
                label,
                assets.presudo_token_ids,
                assets.prompt,
                assets.presudo_list,
                num_steps=args.ca_steps,
                invert_guidance_scale=args.ca_invert_guidance_scale,
                set_guidance_scale=args.ca_guidance_scale,
                intervention_indx=intervention_idx,
                intervention_values=intervention_value.squeeze(0),
                return_PIL=True,
                blend_word=True,
                blend_params=blend_params,
                disentangle=False,
                cross_replace_steps=args.ptp_cross_replace_steps,
                self_replace_steps=args.ptp_self_replace_steps,
                DSCM_labels=None,
            )
        else:
            images, *_ = ddim_editing(
                assets.pipe,
                image_t,
                label,
                assets.presudo_token_ids,
                assets.prompt,
                num_steps=args.ca_steps,
                invert_guidance_scale=args.ca_invert_guidance_scale,
                set_guidance_scale=args.ca_guidance_scale,
                intervention_indx=intervention_idx,
                intervention_values=intervention_value.squeeze(0),
                return_PIL=True,
                disentangle=False,
                DSCM_labels=None,
            )
        tiles.append(_to_tile(source_preview))
        tiles.append(_to_tile(images[-1]))
    return tiles


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = _load_yaml(cfg_path)
    data_root = Path(cfg["data_root"])
    selection = _load_selection(Path(args.selection_json), args.num_each)
    intervention_idx = COMPLEX_ATTRS.index(args.intervention_attr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CelebADataset(root=str(data_root), split="val", factor_cols=list(CELEBA_ATTR_ORDER), image_size=64)
    loader = DataLoader(
        FilenameSubset(ds, selection["indices"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    ca_scm = _load_causal_adapter_scm(Path(args.causal_adapter_root), Path(args.ca_scm_path), device)
    editor = PromptLoraEditor(
        cfg,
        Path(args.lora_dir),
        device,
        args.steps,
        args.guidance_scale,
        invert_guidance_scale=args.invert_guidance_scale,
        negative_prompt=args.negative_prompt,
        scheduler_mode=args.scheduler_mode,
        vae_latent_mode=args.vae_latent_mode,
        condition_mode=args.condition_mode,
        pai_checkpoint=Path(args.pai_checkpoint) if args.pai_checkpoint else None,
        ptp_controller=args.ptp_controller,
        ptp_local_blend=args.ptp_local_blend,
        ptp_blend_th=args.ptp_blend_th,
        ptp_start_blend=args.ptp_start_blend,
        ptp_substruct_attr_indices=args.ptp_substruct_attr_indices,
        capture_ptp_masks=args.export_ptp_masks,
        ptp_mask_steps=args.ptp_mask_steps,
    )

    out_dir = Path(args.out_dir)
    ours_dir = out_dir / "ours" / f"grids_do_{args.intervention_attr}"
    ca_dir = out_dir / "causal_adapter" / f"grids_do_{args.intervention_attr}"
    mask_batches = {} if args.export_ptp_masks else None
    source_tiles = [] if args.export_ptp_masks else None
    ours_tiles = _prompt_lora_tiles(
        editor,
        data_root,
        loader,
        ca_scm,
        intervention_idx,
        args.editing,
        args.inversion,
        args.ptp_cross_replace_steps,
        args.ptp_self_replace_steps,
        mask_batches=mask_batches,
        source_tiles=source_tiles,
    )
    save_image_grid(torch.stack(ours_tiles), nrow=2, path=ours_dir / "grid_32_orig_cf.png", cmap=None, dpi=100.0)

    if args.export_ptp_masks:
        mask_dir = out_dir / "ours" / f"masks_do_{args.intervention_attr}"
        sources = torch.stack(source_tiles)
        red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
        for step in sorted(mask_batches):
            masks = torch.cat(mask_batches[step], dim=0)
            masks64 = torch.nn.functional.interpolate(masks, size=(64, 64), mode="nearest")
            mask_rgb = masks64.repeat(1, 3, 1, 1)
            overlays = sources * (1.0 - 0.4 * masks64) + red * (0.4 * masks64)
            save_image_grid(
                mask_rgb,
                nrow=8,
                path=mask_dir / f"mask_step_{step:02d}.png",
                cmap=None,
                dpi=100.0,
            )
            save_image_grid(
                overlays.clamp(0, 1),
                nrow=8,
                path=mask_dir / f"overlay_step_{step:02d}.png",
                cmap=None,
                dpi=100.0,
            )

    if not args.skip_ca_grid:
        ca_tiles = _causal_adapter_tiles(args, data_root, selection["filenames"], intervention_idx)
        save_image_grid(torch.stack(ca_tiles), nrow=2, path=ca_dir / "grid_32_orig_cf.png", cmap=None, dpi=100.0)

    meta = {
        "intervention_attr": args.intervention_attr,
        "selection_json": args.selection_json,
        "indices": selection["indices"],
        "filenames": selection["filenames"],
        "style": "causal_adapter_vs_parent_only_plain_orig_cf_grid_nrow2",
        "ours_editing": args.editing,
        "ours_condition_mode": args.condition_mode,
        "ours_inversion": args.inversion,
        "ours_invert_guidance_scale": args.invert_guidance_scale,
        "ours_generation_guidance_scale": args.guidance_scale,
        "ours_negative_prompt": args.negative_prompt,
        "ours_scheduler_mode": args.scheduler_mode,
        "ours_vae_latent_mode": args.vae_latent_mode,
        "ptp": {
            "cross_replace_steps": args.ptp_cross_replace_steps,
            "self_replace_steps": args.ptp_self_replace_steps,
            "controller": args.ptp_controller,
            "local_blend": args.ptp_local_blend,
            "blend_th": list(args.ptp_blend_th),
            "substruct_attr_indices": list(args.ptp_substruct_attr_indices),
        }
        if args.editing == "ptp"
        else None,
        "ours_grid": str(ours_dir / "grid_32_orig_cf.png"),
        "causal_adapter_grid": str(ca_dir / "grid_32_orig_cf.png"),
        "causal_adapter_editing": args.ca_editing,
    }
    metadata_dirs = [ours_dir] if args.skip_ca_grid else [ours_dir, ca_dir]
    for d in metadata_dirs:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "grid_32_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
