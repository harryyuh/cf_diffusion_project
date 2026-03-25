import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.morphomnist_dataset import MorphoMNISTDataset
from models.vae import ConvVAE, VAEConfig, beta_vae_loss
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger
from utils.seed import set_seed
from utils.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE on MorphoMNIST.")
    parser.add_argument("--config", type=str, required=True, help="Path to VAE config YAML.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Expand env vars in string values (e.g. ${SCRATCH}/datasets/global)
    if cfg:
        for k, v in cfg.items():
            if isinstance(v, str) and "$" in v:
                cfg[k] = os.path.expandvars(v)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = cfg.get("seed", 123)
    set_seed(seed)

    output_dir = Path(cfg["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_vae", log_dir=log_dir)
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    # Dataset
    train_dataset = MorphoMNISTDataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        mode=cfg.get("data_mode", "images_csv"),
        data_file=cfg.get("train_data_file"),
        metadata_file=cfg.get("train_metadata_file"),
        image_key=cfg.get("image_key", "images"),
    )
    val_dataset = MorphoMNISTDataset(
        root=cfg["data_root"],
        split=cfg.get("val_split", "val"),
        mode=cfg.get("data_mode", "images_csv"),
        data_file=cfg.get("val_data_file"),
        metadata_file=cfg.get("val_metadata_file"),
        image_key=cfg.get("image_key", "images"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    # Model
    vae_config = VAEConfig(
        in_channels=cfg.get("in_channels", 1),
        latent_dim=cfg["latent_dim"],
        hidden_dims=tuple(cfg.get("hidden_dims", [32, 64, 128])),
        image_size=cfg.get("image_size", 28),
        n_parent_dims=cfg.get("n_parent_dims", 0),
        parent_pred_hidden=cfg.get("parent_pred_hidden", 32),
        use_adversary=cfg.get("use_adversary", False),
        adv_hidden=cfg.get("adv_hidden", 32),
    )
    model = ConvVAE(vae_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    epochs = cfg.get("epochs", 100)
    beta = cfg.get("beta_vae", 1.0)
    recon_loss_type = cfg.get("recon_loss", "mse")
    parent_pred_weight = cfg.get("parent_pred_weight", 0.0)  # lambda for |phi(z1)-A|^2
    adv_weight = cfg.get("adv_weight", 0.0)  # lambda_adv: encoder gets reversed grad so z_rest cannot predict A
    father_key = cfg.get("father_key", "thickness")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_parent_sum = 0.0
        train_adv_sum = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            x = batch["image"].to(device)
            out = model(x)
            loss, components = beta_vae_loss(
                out["recon"], x, out["mu"], out["logvar"],
                beta=beta, recon_loss_type=recon_loss_type,
            )
            A = batch[father_key].to(device).float().unsqueeze(1)
            if "parent_pred" in out and parent_pred_weight > 0:
                parent_loss = F.mse_loss(out["parent_pred"], A)
                loss = loss + parent_pred_weight * parent_loss
                components["parent_loss"] = parent_loss
                train_parent_sum += parent_loss.item()
            if "z_rest_grl" in out and adv_weight > 0:
                adv_pred = model.adversary(out["z_rest_grl"])
                adv_loss = F.mse_loss(adv_pred, A)
                loss = loss + adv_weight * adv_loss
                components["adv_loss"] = adv_loss
                train_adv_sum += adv_loss.item()

            if not math.isfinite(loss.item()):
                logger.error(f"Non-finite loss (epoch {epoch+1}, batch {n_batches}): {loss.item()}. Stopping.")
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_recon_sum += components["recon_loss"].item()
            train_kl_sum += components["kl"].item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_sum / n_batches
        train_recon = train_recon_sum / n_batches
        train_kl = train_kl_sum / n_batches

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                out = model(x)
                loss, _ = beta_vae_loss(
                    out["recon"], x, out["mu"], out["logvar"],
                    beta=beta, recon_loss_type=recon_loss_type,
                )
                A_val = batch[father_key].to(device).float().unsqueeze(1)
                if "parent_pred" in out and parent_pred_weight > 0:
                    loss = loss + parent_pred_weight * F.mse_loss(out["parent_pred"], A_val)
                if "z_rest_grl" in out and adv_weight > 0:
                    adv_pred = model.adversary(out["z_rest_grl"])
                    loss = loss + adv_weight * F.mse_loss(adv_pred, A_val)
                val_loss_sum += loss.item() * x.size(0)
                val_n += x.size(0)
        val_loss = val_loss_sum / val_n if val_n else float("inf")

        log_msg = f"Epoch {epoch+1} train_loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f}"
        if parent_pred_weight > 0:
            log_msg += f" parent={train_parent_sum / n_batches:.4f}"
        if adv_weight > 0:
            log_msg += f" adv={train_adv_sum / n_batches:.4f}"
        log_msg += f" val_loss={val_loss:.4f}"
        logger.info(log_msg)

        # Save sample reconstructions every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                x_sample = sample_batch["image"].to(device)[:8]
                out_sample = model(x_sample)
            grid_input = x_sample
            grid_recon = out_sample["recon"]
            save_image_grid(grid_input, 4, sample_dir / f"epoch{epoch+1}_input.png", cmap="gray")
            save_image_grid(grid_recon, 4, sample_dir / f"epoch{epoch+1}_recon.png", cmap="gray")

        # Checkpoint
        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "val_loss": val_loss,
        }
        save_checkpoint(state, ckpt_dir, "vae_last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(state, ckpt_dir, "vae_best.pt")
            logger.info(f"Saved best checkpoint (val_loss={val_loss:.4f})")

    logger.info("VAE training finished.")


if __name__ == "__main__":
    main()