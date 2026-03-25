"""
Train conditional diffusion decoder.

- use_vae_condition=True (default): p_theta(x | z_rest, f), frozen VAE encoder, cond = concat(z_rest, f).
- use_vae_condition=False: p_theta(x | f) with parent A only, cond = f (no VAE).
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.morphomnist_dataset import MorphoMNISTDataset
from models.diffusion_unet import ConditionedUNet, UNetConfig
from models.vae import ConvVAE, VAEConfig
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.diffusion_utils import DiffusionConfig, GaussianDiffusion
from utils.logger import get_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional diffusion decoder.")
    parser.add_argument("--config", type=str, required=True, help="Path to diffusion config YAML.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 123))

    use_vae_condition = bool(cfg.get("use_vae_condition", True))
    # Separate checkpoint roots: base output_dir / with_vae_condition | parent_only (or checkpoint_subdir)
    base_output = Path(cfg["output_dir"])
    if cfg.get("checkpoint_subdir"):
        run_subdir = str(cfg["checkpoint_subdir"])
    else:
        run_subdir = "with_vae_condition" if use_vae_condition else "parent_only"
    output_dir = base_output / run_subdir
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_diffusion", log_dir=log_dir)
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")
    logger.info(
        f"Checkpoints under {output_dir} (use_vae_condition={use_vae_condition}, subdir={run_subdir})"
    )

    vae: Optional[ConvVAE] = None
    n_parent_dims = 0
    rest_dim = 0

    if use_vae_condition:
        # Load VAE config to determine latent_dim and n_parent_dims (first part = parent, rest = z_rest)
        vae_cfg_path = cfg.get("vae_config", "configs/vae.yaml")
        with open(vae_cfg_path, "r") as f:
            vae_cfg = yaml.safe_load(f)
        latent_dim = vae_cfg["latent_dim"]
        n_parent_dims = vae_cfg.get("n_parent_dims", 0)
        rest_dim = latent_dim - n_parent_dims
        logger.info(f"Using latent_dim={latent_dim}, n_parent_dims={n_parent_dims}, z_rest_dim={rest_dim}")

        # Frozen VAE encoder
        vae_config = VAEConfig(
            in_channels=vae_cfg.get("in_channels", 1),
            latent_dim=latent_dim,
            hidden_dims=tuple(vae_cfg.get("hidden_dims", [32, 64, 128])),
            image_size=vae_cfg.get("image_size", 28),
            n_parent_dims=n_parent_dims,
            parent_pred_hidden=vae_cfg.get("parent_pred_hidden", 32),
            use_adversary=vae_cfg.get("use_adversary", False),
            adv_hidden=vae_cfg.get("adv_hidden", 32),
        )
        vae = ConvVAE(vae_config)
        vae_ckpt = load_checkpoint(Path(cfg["vae_checkpoint"]), model=vae, map_location="cpu")
        vae = vae.to(device)
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()
        logger.info("VAE encoder loaded and frozen.")
    else:
        logger.info("use_vae_condition=False: training diffusion with parent A only (cond_dim=1). VAE not loaded.")

    # Dataset
    train_dataset = MorphoMNISTDataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        mode=cfg.get("data_mode", "images_csv"),
        data_file=cfg.get("train_data_file"),
        metadata_file=cfg.get("train_metadata_file"),
        image_key=cfg.get("image_key", "images"),
    )
    father_key = cfg.get("father_key", "thickness")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    cond_dim = rest_dim + 1 if use_vae_condition else 1
    unet_config = UNetConfig(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("unet_base_channels", 32),
        channel_mults=tuple(cfg.get("unet_channel_mults", [1, 2, 4])),
        time_emb_dim=cfg.get("time_emb_dim", 128),
        cond_dim=cond_dim,
    )
    unet = ConditionedUNet(unet_config).to(device)

    model = unet
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))

    diff_config = DiffusionConfig(
        timesteps=cfg.get("timesteps", 1000),
        beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02),
        ddim_eta=cfg.get("ddim_eta", 0.0),
    )
    diffusion = GaussianDiffusion(diff_config)
    for attr in (
        "betas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
    ):
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    epochs = cfg.get("epochs", 100)

    # Resume from last checkpoint if available
    last_ckpt_path = ckpt_dir / "diffusion_last.pt"
    start_epoch = 0
    best_loss = float("inf")
    if last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device)
        if "unet_state_dict" in ckpt:
            unet.load_state_dict(ckpt["unet_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
        if "train_loss" in ckpt:
            best_loss = ckpt["train_loss"]
        logger.info(f"Resuming diffusion training from epoch {start_epoch} using {last_ckpt_path}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            x = batch["image"].to(device)
            f = batch[father_key].to(device).float().unsqueeze(1)  # (B, 1)

            if use_vae_condition:
                assert vae is not None
                with torch.no_grad():
                    mu, logvar = vae.encode(x)
                    z = mu  # use mean for conditioning
                    z_rest = z[:, n_parent_dims:]  # (B, rest_dim)
                cond = torch.cat([z_rest, f], dim=1)  # (B, rest_dim + 1)
            else:
                cond = f  # (B, 1) parent only

            # Sample t and noise
            b = x.size(0)
            t = torch.randint(0, diffusion.timesteps, (b,), device=device).long()
            noise = torch.randn_like(x, device=device)
            x_t = diffusion.q_sample(x, t, noise)

            eps_pred = model(x_t, t, cond)
            loss = nn.functional.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        mean_loss = epoch_loss / n_batches
        logger.info(f"Epoch {epoch+1} train_loss={mean_loss:.4f}")

        state = {
            "epoch": epoch + 1,
            "unet_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "train_loss": mean_loss,
            "use_vae_condition": use_vae_condition,
        }
        save_checkpoint(state, ckpt_dir, "diffusion_last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(state, ckpt_dir, "diffusion_best.pt")
            logger.info(f"Saved best checkpoint (loss={mean_loss:.4f})")

    logger.info("Diffusion training finished.")


if __name__ == "__main__":
    main()
