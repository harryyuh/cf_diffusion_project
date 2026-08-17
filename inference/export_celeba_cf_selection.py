"""Export shared selection.json for CelebA Male↔Female counterfactual grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset
from inference.generate_celeba_counterfactual_male import (
    _default_male_female_indices,
    _load_yaml,
    _resolve_data_root,
    _selection_record,
    _validate_indices_for_male_female,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="configs/diffusion_celeba_parent_only.yaml",
        help="Dataset fields (data_root, celeba_attr_cols, image_size) are taken from this yaml.",
    )
    p.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    p.add_argument("--n-each", type=int, default=6)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[1] / cfg_path
    cfg = _load_yaml(cfg_path)

    parent_keys = list(cfg.get("parent_keys") or ["Male"])
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
    male_idx, female_idx = _default_male_female_indices(ds, parent_keys[0], int(args.n_each))
    _validate_indices_for_male_female(ds, parent_keys[0], male_idx, female_idx)

    selection = _selection_record(
        data_root=_resolve_data_root(cfg),
        split=str(args.split),
        image_size=image_size,
        factor_cols=factor_cols,
        parent_keys=parent_keys,
        n_each=int(args.n_each),
        male_indices=male_idx,
        female_indices=female_idx,
        ds=ds,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
