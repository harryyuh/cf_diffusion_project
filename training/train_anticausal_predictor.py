"""
Train convolutional anti-causal predictors on CelebA: h(x) -> selected attributes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset, expand_env_in_cfg
from models.anticausal_predictor import ConvAnticausalConfig, ConvAnticausalPredictor
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train anticausal conv regressor on CelebA.")
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def regression_target_mean_std(
    dataset: CelebADataset, cols: List[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    y = dataset.factor_matrix(cols)
    t = torch.from_numpy(y)
    return t.mean(dim=0), t.std(dim=0).clamp(min=1e-6)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 123))

    output_dir = Path(cfg["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("anticausal", log_dir=log_dir)
    logger.info(json.dumps(cfg, indent=2))

    reg_cols = list(cfg.get("regression_cols", ["Male", "Smiling"]))
    if cfg.get("use_all_celeba_attrs"):
        factor_cols = list(CELEBA_ATTR_ORDER)
    else:
        factor_cols = list(cfg["celeba_attr_cols"])
    image_size = int(cfg.get("image_size", 64))
    for c in reg_cols:
        if c not in factor_cols:
            raise ValueError(f"regression_cols entry {c} missing from celeba_attr_cols")

    train_ds = CelebADataset(
        root=cfg["data_root"],
        split="train",
        factor_cols=factor_cols,
        image_size=image_size,
    )
    val_ds = CelebADataset(
        root=cfg["data_root"],
        split=cfg.get("val_split", "val"),
        factor_cols=factor_cols,
        image_size=image_size,
    )

    y_mean: Optional[torch.Tensor] = None
    y_std: Optional[torch.Tensor] = None
    if bool(cfg.get("normalize_regression_targets", True)):
        y_mean, y_std = regression_target_mean_std(train_ds, reg_cols)
        logger.info(f"Regression target mean: {y_mean.tolist()} std: {y_std.tolist()}")

    y_mean_cpu = y_mean
    y_std_cpu = y_std

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        imgs = torch.stack([b["image"] for b in batch], dim=0)
        y_reg = torch.stack([torch.cat([b[c] for c in reg_cols], dim=0) for b in batch], dim=0)
        if y_mean_cpu is not None and y_std_cpu is not None:
            y_reg = (y_reg - y_mean_cpu) / y_std_cpu
        return {"image": imgs, "y_reg": y_reg}

    def to_device(b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate,
    )

    use_cls = bool(cfg.get("use_classification", False))
    if use_cls:
        raise NotImplementedError("CelebA anticausal config uses regression only in this project.")

    model_cfg = ConvAnticausalConfig(
        in_channels=cfg.get("in_channels", 3),
        image_size=image_size,
        hidden_dims=tuple(cfg.get("hidden_dims", [32, 64, 128])),
        head_hidden=cfg.get("head_hidden", 256),
        regression_dim=len(reg_cols),
        num_classes=cfg.get("num_classes", 10),
        use_classification=False,
    )
    model = ConvAnticausalPredictor(model_cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    reg_weight = float(cfg.get("reg_loss_weight", 1.0))

    best_val = float("inf")
    epochs = int(cfg.get("epochs", 50))

    for epoch in range(1, epochs + 1):
        model.train()
        tr_reg = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs} train"):
            batch = to_device(batch)
            opt.zero_grad(set_to_none=True)
            pred = model(batch["image"])
            loss = reg_weight * nn.functional.mse_loss(pred["y_reg"], batch["y_reg"])
            loss.backward()
            opt.step()
            tr_reg += float(loss.item())
            n += 1
        tr_reg /= max(n, 1)

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch)
                pred = model(batch["image"])
                vl = reg_weight * nn.functional.mse_loss(pred["y_reg"], batch["y_reg"])
                val_loss += float(vl.item())
                nv += 1
        val_loss /= max(nv, 1)

        logger.info(f"epoch {epoch} train_mse~{tr_reg:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "config": cfg,
                "model_config": model_cfg.__dict__,
                "regression_cols": reg_cols,
                "y_mean": y_mean if y_mean is not None else None,
                "y_std": y_std if y_std is not None else None,
                "val_mse": val_loss,
            }
            save_checkpoint(state, ckpt_dir, "anticausal_best.pt")
            logger.info(f"saved best to {ckpt_dir / 'anticausal_best.pt'}")


if __name__ == "__main__":
    main()
