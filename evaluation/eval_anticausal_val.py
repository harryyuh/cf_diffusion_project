"""
Validation MSE / MAE for CelebA anticausal predictor (per attribute).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset, expand_env_in_cfg
from models.anticausal_predictor import ConvAnticausalConfig, ConvAnticausalPredictor
from utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def _build_model(ckpt: Dict[str, Any]) -> ConvAnticausalPredictor:
    mc = ckpt.get("model_config") or {}
    hidden_dims = mc.get("hidden_dims", (32, 64, 128))
    if isinstance(hidden_dims, list):
        hidden_dims = tuple(hidden_dims)
    cfg = ConvAnticausalConfig(
        in_channels=int(mc.get("in_channels", 3)),
        image_size=int(mc.get("image_size", 64)),
        hidden_dims=tuple(hidden_dims),
        head_hidden=int(mc.get("head_hidden", 256)),
        regression_dim=int(mc.get("regression_dim", 1)),
        num_classes=int(mc.get("num_classes", 10)),
        use_classification=bool(mc.get("use_classification", False)),
    )
    return ConvAnticausalPredictor(cfg)


def _denorm(y: torch.Tensor, y_mean: Optional[torch.Tensor], y_std: Optional[torch.Tensor]) -> torch.Tensor:
    if y_mean is None or y_std is None:
        return y
    return y * y_std.to(y.device) + y_mean.to(y.device)


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(str(cfg.get("checkpoint", "")))
    if not ckpt_path.is_file():
        out_dir = Path(cfg["output_dir"])
        ckpt_path = out_dir / "checkpoints" / "anticausal_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    reg_cols: List[str] = list(ckpt.get("regression_cols", cfg.get("regression_cols", [])))
    y_mean: Optional[torch.Tensor] = ckpt.get("y_mean")
    y_std: Optional[torch.Tensor] = ckpt.get("y_std")

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(ckpt).to(device)
    load_checkpoint(ckpt_path, model=model, map_location="cpu")
    model.eval()

    if cfg.get("use_all_celeba_attrs"):
        factor_cols = list(CELEBA_ATTR_ORDER)
    else:
        factor_cols = list(cfg["celeba_attr_cols"])
    image_size = int(cfg.get("image_size", 64))
    ds = CelebADataset(root=cfg["data_root"], split=args.split, factor_cols=factor_cols, image_size=image_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    abs_sum = np.zeros(len(reg_cols), dtype=np.float64)
    sq_sum = np.zeros(len(reg_cols), dtype=np.float64)
    n_total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            pred = model(x)["y_reg"]
            pred_dn = _denorm(pred, y_mean, y_std)
            tgt = torch.stack([batch[c].squeeze(-1) for c in reg_cols], dim=1).to(device).float()
            abs_sum += (pred_dn - tgt).abs().sum(dim=0).cpu().numpy()
            sq_sum += ((pred_dn - tgt) ** 2).sum(dim=0).cpu().numpy()
            n_total += x.size(0)

    mae = (abs_sum / max(n_total, 1)).tolist()
    rmse = np.sqrt(np.maximum(sq_sum / max(n_total, 1), 0.0)).tolist()
    out = {
        "checkpoint": str(ckpt_path.resolve()),
        "split": args.split,
        "n": n_total,
        "regression_cols": reg_cols,
        "mae_per_col": {reg_cols[i]: mae[i] for i in range(len(reg_cols))},
        "rmse_per_col": {reg_cols[i]: float(rmse[i]) for i in range(len(reg_cols))},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
