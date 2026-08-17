"""CelebA Male↔Female counterfactual grids via DDIM encode / decode (image space)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from data.celeba_dataset import CelebADataset, CELEBA_ATTR_ORDER, expand_env_in_cfg
from models.diffusion_unet_factory import build_conditional_diffusion_unet
from models.vae_factory import build_vae_from_train_cfg
from utils.checkpoint import load_checkpoint
from utils.diffusion_utils import DiffusionConfig, GaussianDiffusion
from utils.visualization import save_image_grid

SELECTION_SCHEMA_VERSION = 1


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def _diffusion_ckpt_subdir(cfg: Dict[str, Any]) -> str:
    if cfg.get("checkpoint_subdir"):
        return str(cfg["checkpoint_subdir"])
    use_vae = bool(cfg.get("use_vae_condition", True))
    if not use_vae:
        return "parent_only"
    if bool(cfg.get("condition_include_z_rest", True)):
        return "with_vae_condition"
    return "with_vae_parent_cond_only"


def _parent_matrix(batch: Dict[str, Any], keys: List[str], device: torch.device) -> torch.Tensor:
    return torch.stack([batch[k].to(device).float().squeeze(-1) for k in keys], dim=1)


@torch.no_grad()
def _ddim_cf(
    vae: Optional[nn.Module],
    unet: nn.Module,
    diffusion: GaussianDiffusion,
    x: torch.Tensor,
    f_obs: torch.Tensor,
    f_tgt: torch.Tensor,
    n_parent_dims: int,
    start_t: int,
    device: torch.device,
    *,
    include_z_rest: bool,
) -> torch.Tensor:
    if f_obs.dim() == 1:
        f_obs = f_obs.unsqueeze(1)
    if f_tgt.dim() == 1:
        f_tgt = f_tgt.unsqueeze(1)
    if include_z_rest:
        assert vae is not None
        mu, _ = vae.encode(x)
        z_rest = mu[:, n_parent_dims:]
        cond_enc = torch.cat([z_rest, f_obs], dim=1)
        cond_dec = torch.cat([z_rest, f_tgt], dim=1)
    else:
        cond_enc = f_obs
        cond_dec = f_tgt
    x_t = diffusion.ddim_encode_to_xt(unet, x, start_t, cond_enc, device)
    out = diffusion.p_sample_loop_from_xt(unet, x_t, start_t, cond_dec, device)
    return out.clamp(0.0, 1.0)


def _load_models(cfg: Dict[str, Any], device: torch.device, diffusion_ckpt: Optional[Path]) -> Tuple:
    use_vae = bool(cfg.get("use_vae_condition", True))
    include_z_rest = bool(cfg.get("condition_include_z_rest", True)) if use_vae else False
    vae: Optional[nn.Module] = None
    n_parent_dims = 0
    rest_dim = 0
    if use_vae:
        vae_cfg_path = Path(cfg.get("vae_config", "configs/vae_celeba.yaml"))
        if not vae_cfg_path.is_absolute():
            vae_cfg_path = Path(__file__).resolve().parents[1] / vae_cfg_path
        with open(vae_cfg_path, "r") as f:
            vae_cfg = yaml.safe_load(f)
        expand_env_in_cfg(vae_cfg)
        n_parent_dims = int(vae_cfg.get("n_parent_dims", 0))
        rest_dim = int(vae_cfg["latent_dim"]) - n_parent_dims
        vae = build_vae_from_train_cfg(vae_cfg)
        vck = Path(cfg["vae_checkpoint"])
        load_checkpoint(vck, model=vae, map_location="cpu")
        vae = vae.to(device).eval()
        for p in vae.parameters():
            p.requires_grad = False

    parent_keys: List[str] = list(cfg.get("parent_keys") or ["Male"])
    parent_dim = len(parent_keys)
    if use_vae and include_z_rest:
        cond_dim = rest_dim + parent_dim
    else:
        cond_dim = parent_dim

    unet = build_conditional_diffusion_unet(cfg, cond_dim).to(device)
    subdir = _diffusion_ckpt_subdir(cfg)
    base_out = Path(cfg["output_dir"])
    if diffusion_ckpt is None:
        diffusion_ckpt = base_out / subdir / "checkpoints" / "diffusion_best.pt"
    if not diffusion_ckpt.is_file():
        alt = diffusion_ckpt.parent / "diffusion_last.pt"
        diffusion_ckpt = alt if alt.is_file() else diffusion_ckpt
    ck = torch.load(diffusion_ckpt, map_location=device)
    unet.load_state_dict(ck["unet_state_dict"])
    unet.eval()
    ckpt_epoch = ck.get("epoch")
    ckpt_train_loss = ck.get("train_loss")

    diff = GaussianDiffusion(
        DiffusionConfig(
            timesteps=int(cfg.get("timesteps", 1000)),
            beta_start=float(cfg.get("beta_start", 1e-4)),
            beta_end=float(cfg.get("beta_end", 0.02)),
            ddim_eta=float(cfg.get("ddim_eta", 0.0)),
        )
    )
    for a in ("betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
        setattr(diff, a, getattr(diff, a).to(device))
    return (
        vae,
        unet,
        diff,
        n_parent_dims,
        include_z_rest,
        str(diffusion_ckpt),
        subdir,
        ckpt_epoch,
        ckpt_train_loss,
    )


def _resolve_data_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["data_root"]).expanduser().resolve()


def _default_male_female_indices(ds: CelebADataset, parent_key: str, n_each: int) -> Tuple[List[int], List[int]]:
    male_idx = [i for i in range(len(ds)) if float(ds._attrs.iloc[i][parent_key]) > 0]
    female_idx = [i for i in range(len(ds)) if float(ds._attrs.iloc[i][parent_key]) < 0]
    n = int(n_each)
    return male_idx[:n], female_idx[:n]


def _selection_record(
    *,
    data_root: Path,
    split: str,
    image_size: int,
    factor_cols: List[str],
    parent_keys: List[str],
    n_each: int,
    male_indices: List[int],
    female_indices: List[int],
    ds: CelebADataset,
) -> Dict[str, Any]:
    male_names = [ds._filenames[i] for i in male_indices]
    female_names = [ds._filenames[i] for i in female_indices]
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "data_root": str(data_root),
        "split": split,
        "image_size": int(image_size),
        "celeba_attr_cols": list(factor_cols),
        "parent_keys": list(parent_keys),
        "n_each": int(n_each),
        "male_indices": [int(x) for x in male_indices],
        "female_indices": [int(x) for x in female_indices],
        "male_filenames": male_names,
        "female_filenames": female_names,
    }


def _validate_indices_for_male_female(
    ds: CelebADataset, parent_key: str, male_indices: List[int], female_indices: List[int]
) -> None:
    for tag, idxs, want_pos in (("male", male_indices, True), ("female", female_indices, False)):
        for i in idxs:
            if i < 0 or i >= len(ds):
                raise ValueError(f"selection {tag}_indices out of range: {i} (len={len(ds)})")
            v = float(ds._attrs.iloc[i][parent_key])
            if want_pos and not (v > 0):
                raise ValueError(f"selection index {i} expected Male>0 for {tag}, got {v}")
            if not want_pos and not (v < 0):
                raise ValueError(f"selection index {i} expected Male<0 for {tag}, got {v}")


def _load_selection_json(path: Path, ds: CelebADataset, cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[int], List[int], Dict[str, Any]]:
    with open(path, "r") as f:
        raw = json.load(f)
    if "male_indices" not in raw or "female_indices" not in raw:
        raise ValueError(f"{path}: JSON must contain male_indices and female_indices")
    male_indices = [int(x) for x in raw["male_indices"]]
    female_indices = [int(x) for x in raw["female_indices"]]
    parent_keys: List[str] = list(cfg.get("parent_keys") or ["Male"])
    pk = parent_keys[0]
    _validate_indices_for_male_female(ds, pk, male_indices, female_indices)

    want_root = _resolve_data_root(cfg)
    if raw.get("data_root"):
        got = Path(str(raw["data_root"])).expanduser().resolve()
        if got != want_root and os.path.realpath(str(got)) != os.path.realpath(str(want_root)):
            raise ValueError(f"selection data_root mismatch: file has {got}, config has {want_root}")
    if raw.get("split") and str(raw["split"]) != str(args.split):
        raise ValueError(f"selection split mismatch: file has {raw['split']}, args have {args.split}")
    if raw.get("image_size") is not None and int(raw["image_size"]) != int(cfg.get("image_size", 64)):
        raise ValueError("selection image_size mismatch vs config")
    factor_cols: List[str] = list(CELEBA_ATTR_ORDER) if cfg.get("use_all_celeba_attrs") else list(cfg["celeba_attr_cols"])
    if raw.get("celeba_attr_cols") is not None and list(raw["celeba_attr_cols"]) != factor_cols:
        raise ValueError("selection celeba_attr_cols mismatch vs config (column set/order must match dataset)")
    if raw.get("parent_keys") is not None and list(raw["parent_keys"]) != parent_keys:
        raise ValueError("selection parent_keys mismatch vs config")

    if raw.get("male_filenames"):
        for i, exp in zip(male_indices, raw["male_filenames"]):
            if ds._filenames[i] != exp:
                raise ValueError(f"male index {i}: expected filename {exp}, dataset has {ds._filenames[i]}")
    if raw.get("female_filenames"):
        for i, exp in zip(female_indices, raw["female_filenames"]):
            if ds._filenames[i] != exp:
                raise ValueError(f"female index {i}: expected filename {exp}, dataset has {ds._filenames[i]}")

    sel = _selection_record(
        data_root=want_root,
        split=str(args.split),
        image_size=int(cfg.get("image_size", 64)),
        factor_cols=factor_cols,
        parent_keys=parent_keys,
        n_each=len(male_indices),
        male_indices=male_indices,
        female_indices=female_indices,
        ds=ds,
    )
    return male_indices, female_indices, sel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--diffusion-checkpoint", type=str, default="")
    p.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    p.add_argument("--n-each", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--start-t", type=int, default=300)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--selection-json",
        type=str,
        default="",
        help="Reuse exact male/female dataset indices + filenames from a prior run (same data_root/split/attrs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[1] / cfg_path
    cfg = _load_yaml(cfg_path)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    diff_ckpt = Path(args.diffusion_checkpoint) if args.diffusion_checkpoint.strip() else None
    vae, unet, diffusion, n_parent_dims, include_z_rest, ckpt_used, subdir, ckpt_epoch, ckpt_train_loss = (
        _load_models(cfg, device, diff_ckpt)
    )

    parent_keys: List[str] = list(cfg.get("parent_keys") or ["Male"])
    if cfg.get("use_all_celeba_attrs"):
        factor_cols = list(CELEBA_ATTR_ORDER)
    else:
        factor_cols = list(cfg["celeba_attr_cols"])
    image_size = int(cfg.get("image_size", 64))
    ds = CelebADataset(
        root=cfg["data_root"],
        split=args.split,
        factor_cols=factor_cols,
        image_size=image_size,
    )
    root_resolved = _resolve_data_root(cfg)

    if args.selection_json.strip():
        sel_path = Path(args.selection_json)
        if not sel_path.is_file():
            raise FileNotFoundError(f"--selection-json not found: {sel_path}")
        male_idx, female_idx, selection = _load_selection_json(sel_path, ds, cfg, args)
        if len(male_idx) != int(args.n_each) or len(female_idx) != int(args.n_each):
            raise ValueError(
                f"--n-each ({args.n_each}) must match selection lengths "
                f"(male={len(male_idx)}, female={len(female_idx)}); fix N_EACH or regenerate selection."
            )
    else:
        male_idx, female_idx = _default_male_female_indices(ds, parent_keys[0], int(args.n_each))
        selection = _selection_record(
            data_root=root_resolved,
            split=str(args.split),
            image_size=image_size,
            factor_cols=factor_cols,
            parent_keys=parent_keys,
            n_each=int(args.n_each),
            male_indices=male_idx,
            female_indices=female_idx,
            ds=ds,
        )
        _validate_indices_for_male_female(ds, parent_keys[0], male_idx, female_idx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_dpi = 100.0

    def run_grid(name: str, indices: List[int], flip_to: float) -> None:
        if not indices:
            return
        subset = Subset(ds, indices)
        loader = DataLoader(subset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
        tiles: List[torch.Tensor] = []
        for batch in loader:
            x = batch["image"].to(device)
            f = _parent_matrix(batch, parent_keys, device)
            f_tgt = torch.full_like(f, float(flip_to))
            x_cf = _ddim_cf(
                vae,
                unet,
                diffusion,
                x,
                f,
                f_tgt,
                n_parent_dims,
                int(args.start_t),
                device,
                include_z_rest=include_z_rest,
            )
            for bi in range(x.size(0)):
                tiles.append(x[bi])
                tiles.append(x_cf[bi])
        if not tiles:
            return
        grid = torch.stack(tiles, dim=0)
        save_image_grid(grid, nrow=2, path=out_dir / f"grid_{name}.png", cmap=None, dpi=grid_dpi)

    run_grid("male_to_female", male_idx, -1.0)
    run_grid("female_to_male", female_idx, 1.0)

    with open(out_dir / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    meta: Dict[str, Any] = {
        "schema_version": 1,
        "script": "generate_celeba_counterfactual_male.py",
        "selection": selection,
        "selection_json": str(Path(args.selection_json).resolve())
        if args.selection_json.strip()
        else "",
        "generation": {
            "start_t": int(args.start_t),
            "batch_size": int(args.batch_size),
            "grid_png_dpi": grid_dpi,
            "grid_files": ["grid_male_to_female.png", "grid_female_to_male.png"],
        },
        "model": {
            "config": str(cfg_path),
            "diffusion_checkpoint": ckpt_used,
            "diffusion_ckpt_subdir": subdir,
            "diffusion_ckpt_epoch": ckpt_epoch,
            "diffusion_ckpt_train_loss": ckpt_train_loss,
            "use_vae_condition": bool(cfg.get("use_vae_condition", True)),
            "condition_include_z_rest": bool(include_z_rest),
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote grids under {out_dir}")


if __name__ == "__main__":
    main()
