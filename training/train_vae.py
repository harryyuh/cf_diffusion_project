import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
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
    )
    model = ConvVAE(vae_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    epochs = cfg.get("epochs", 100)
    beta = cfg.get("beta_vae", 1.0)
    recon_loss_type = cfg.get("recon_loss", "mse")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            x = batch["image"].to(device)
            out = model(x)
            loss, components = beta_vae_loss(
                out["recon"], x, out["mu"], out["logvar"],
                beta=beta, recon_loss_type=recon_loss_type,
            )
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
                val_loss_sum += loss.item() * x.size(0)
                val_n += x.size(0)
        val_loss = val_loss_sum / val_n if val_n else float("inf")

        logger.info(
            f"Epoch {epoch+1} train_loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} val_loss={val_loss:.4f}"
        )

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