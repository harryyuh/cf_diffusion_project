from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.celeba_dataset import CelebADataset, expand_env_in_cfg
from models.full_label_scm import FULL_LABEL_KEYS, FullLabelSCM
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def batch_labels(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack([batch[k].to(device).float().squeeze(-1) for k in FULL_LABEL_KEYS], dim=1)


@torch.no_grad()
def evaluate(model: FullLabelSCM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n = 0
    correct = torch.zeros(len(FULL_LABEL_KEYS), device=device)
    total = 0
    child = model.child_indices
    for batch in loader:
        y = batch_labels(batch, device)
        labels01 = (y > 0).float()
        logits = model.logits_from_01(labels01)
        loss = model.loss(y)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        loss_sum += float(loss.item()) * y.shape[0]
        n += y.shape[0]
        if child:
            correct[child] += (pred[:, child] == labels01[:, child]).float().sum(dim=0)
        total += y.shape[0]
    out = {"loss": loss_sum / max(n, 1)}
    for idx in child:
        out[f"acc_{FULL_LABEL_KEYS[idx]}"] = float((correct[idx] / max(total, 1)).item())
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(cfg.get("seed", 42)))

    output_dir = Path(cfg["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CelebADataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        factor_cols=list(FULL_LABEL_KEYS),
        image_size=int(cfg.get("image_size", 64)),
    )
    val_ds = CelebADataset(
        root=cfg["data_root"],
        split=cfg.get("val_split", "val"),
        factor_cols=list(FULL_LABEL_KEYS),
        image_size=int(cfg.get("image_size", 64)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 512)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    model = FullLabelSCM(hidden_dim=int(cfg.get("hidden_dim", 64))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    best_val = float("inf")
    last_path = ckpt_dir / "full_label_scm_last.pt"
    best_path = ckpt_dir / "full_label_scm_best.pt"
    start_epoch = 0
    if last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck.get("epoch", 0))
        best_val = float(ck.get("best_val_loss", best_val))
        print(f"Resuming from {last_path} at epoch {start_epoch}")

    history = []
    epochs = int(cfg.get("epochs", 50))
    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"SCM epoch {epoch + 1}/{epochs}")
        train_loss = 0.0
        n = 0
        for batch in pbar:
            y = batch_labels(batch, device)
            loss = model.loss(y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * y.shape[0]
            n += y.shape[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss /= max(n, 1)
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": train_loss, **val}
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "config": cfg,
            "full_label_keys": list(FULL_LABEL_KEYS),
            "graph": model.graph.detach().cpu().int().tolist(),
            "best_val_loss": min(best_val, val["loss"]),
        }
        torch.save(state, last_path)
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(state, best_path)
            print(f"saved best {best_path} val_loss={best_val:.6f}")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
