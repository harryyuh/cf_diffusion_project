"""CelebA Male<->Female counterfactuals with full-label SCM targets + DDIM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.celeba_dataset import CelebADataset, expand_env_in_cfg
from inference.generate_celeba_counterfactual_male import _ddim_cf, _load_models
from models.full_label_scm import FULL_LABEL_KEYS, FullLabelSCM
from utils.visualization import save_image_grid


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/diffusion_celeba_full_label.yaml")
    p.add_argument("--diffusion-checkpoint", default="")
    p.add_argument("--scm-checkpoint", default="/scratch/gilbreth/yu1331/ckpts/celeba/full_label_scm/causal_adapter_graph/checkpoints/full_label_scm_best.pt")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--selection-json", default="")
    p.add_argument("--n-each", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--start-t", type=int, default=300)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def _labels(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack([batch[k].to(device).float().squeeze(-1) for k in FULL_LABEL_KEYS], dim=1)


def _default_indices(ds: CelebADataset, n_each: int) -> tuple[List[int], List[int]]:
    male = [i for i in range(len(ds)) if float(ds._attrs.iloc[i]["Male"]) > 0]
    female = [i for i in range(len(ds)) if float(ds._attrs.iloc[i]["Male"]) < 0]
    return male[:n_each], female[:n_each]


def _load_indices(path: str, n_each: int) -> tuple[List[int], List[int], Dict[str, Any]]:
    if not path:
        return [], [], {}
    with open(path, "r") as f:
        raw = json.load(f)
    male = [int(x) for x in raw["male_indices"]]
    female = [int(x) for x in raw["female_indices"]]
    if len(male) != n_each or len(female) != n_each:
        raise ValueError(f"--n-each={n_each} does not match selection lengths {len(male)} / {len(female)}")
    return male, female, raw


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    cfg = _load_yaml(cfg_path)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    diff_ckpt = Path(args.diffusion_checkpoint) if args.diffusion_checkpoint.strip() else None
    vae, unet, diffusion, n_parent_dims, include_z_rest, ckpt_used, subdir, ckpt_epoch, _, ckpt_loss = _load_models(
        cfg, device, diff_ckpt
    )

    scm = FullLabelSCM().to(device)
    ck = torch.load(args.scm_checkpoint, map_location=device)
    scm.load_state_dict(ck["model_state_dict"])
    scm.eval()

    ds = CelebADataset(
        root=cfg["data_root"],
        split=args.split,
        factor_cols=list(FULL_LABEL_KEYS),
        image_size=int(cfg.get("image_size", 64)),
    )
    male_idx, female_idx, raw_selection = _load_indices(args.selection_json, int(args.n_each))
    if not male_idx:
        male_idx, female_idx = _default_indices(ds, int(args.n_each))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    male_intervention_index = list(FULL_LABEL_KEYS).index("Male")

    def run_grid(tag: str, indices: List[int], target_male_pm1: float) -> None:
        loader = DataLoader(Subset(ds, indices), batch_size=int(args.batch_size), shuffle=False, num_workers=0)
        tiles: List[torch.Tensor] = []
        for batch in loader:
            x = batch["image"].to(device)
            f_obs = _labels(batch, device)
            f_tgt = scm.intervene_pm1(f_obs, male_intervention_index, target_male_pm1)
            x_cf = _ddim_cf(
                vae,
                unet,
                diffusion,
                x,
                f_obs,
                f_tgt,
                n_parent_dims,
                int(args.start_t),
                device,
                include_z_rest=include_z_rest,
            )
            for i in range(x.shape[0]):
                tiles.append(x[i])
                tiles.append(x_cf[i])
        save_image_grid(torch.stack(tiles, dim=0), nrow=2, path=out_dir / f"grid_{tag}.png", cmap=None, dpi=100.0)

    run_grid("male_to_female", male_idx, -1.0)
    run_grid("female_to_male", female_idx, 1.0)

    meta = {
        "script": "generate_celeba_counterfactual_male_full_label.py",
        "full_label_keys": list(FULL_LABEL_KEYS),
        "graph": scm.graph.detach().cpu().int().tolist(),
        "config": str(cfg_path),
        "diffusion_checkpoint": ckpt_used,
        "diffusion_checkpoint_subdir": subdir,
        "diffusion_checkpoint_epoch": ckpt_epoch,
        "diffusion_checkpoint_train_loss": ckpt_loss,
        "scm_checkpoint": str(args.scm_checkpoint),
        "selection_json": args.selection_json,
        "selection": raw_selection,
        "n_each": int(args.n_each),
        "start_t": int(args.start_t),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"DONE: wrote full-label grids under {out_dir}")


if __name__ == "__main__":
    main()
