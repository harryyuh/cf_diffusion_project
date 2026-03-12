"""
Train conditional diffusion decoder p_theta(x | z_rest, f).
VAE encoder is frozen; only diffusion model is trained.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.morphomnist_dataset import MorphoMNISTDataset
from models.condition_mlp import ConditionMLP, ConditionMLPConfig
from models.diffusion_unet import ConditionedUNet, UNetConfig
from models.vae import ConvVAE, VAEConfig
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.diffusion_utils import DiffusionConfig, GaussianDiffusion
from utils.latent_utils import load_latent_splits, split_latents
from utils.logger import get_logger
from utils.seed import set_seed


class ConditionalDiffusionModel(nn.Module):
    """Wrapper: condition MLP + U-Net. Forward(x_t, t, cond_vector)."""

    def __init__(
        self,
        cond_mlp: ConditionMLP,
        unet: ConditionedUNet,
    ) -> None:
        super().__init__()
        self.cond_mlp = cond_mlp
        self.unet = unet

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_emb = self.cond_mlp(cond)
        return self.unet(x_t, t, cond_emb)


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

    output_dir = Path(cfg["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_diffusion", log_dir=log_dir)
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    # Load latent split
    analysis_dir = Path(cfg["latent_analysis_dir"])
    father_dims, rest_dims = load_latent_splits(
        analysis_dir / "father_dims.json",
        analysis_dir / "rest_dims.json",
    )
    logger.info(f"Using rest_dims: {len(rest_dims)} dims")

    # Frozen VAE encoder
    vae_config = VAEConfig(
        in_channels=cfg.get("in_channels", 1),
        latent_dim=cfg["vae_latent_dim"],
        hidden_dims=tuple(cfg.get("vae_hidden_dims", [32, 64, 128])),
        image_size=cfg.get("image_size", 28),
    )
    vae = ConvVAE(vae_config)
    vae_ckpt = load_checkpoint(Path(cfg["vae_checkpoint"]), model=vae, map_location="cpu")
    vae = vae.to(device)
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()
    logger.info("VAE encoder loaded and frozen.")

    # Dataset
    train_dataset = MorphoMNISTDataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        mode=cfg.get("data_mode", "images_csv"),
        data_file=cfg.get("train_data_file"),
        metadata_file=cfg.get("train_metadata_file"),
        image_key=cfg.get("image_key", "images"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    # Condition: concat(z_rest, f). z_rest has len(rest_dims), f is 1 scalar.
    cond_input_dim = len(rest_dims) + 1
    cond_mlp_config = ConditionMLPConfig(
        input_dim=cond_input_dim,
        hidden_dim=cfg.get("cond_mlp_hidden_dim", 128),
        output_dim=cfg.get("cond_emb_dim", 128),
        dropout=cfg.get("cond_mlp_dropout", 0.0),
    )
    cond_mlp = ConditionMLP(cond_mlp_config).to(device)

    unet_config = UNetConfig(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("unet_base_channels", 32),
        channel_mults=tuple(cfg.get("unet_channel_mults", [1, 2, 4])),
        time_emb_dim=cfg.get("time_emb_dim", 128),
        cond_emb_dim=cfg.get("cond_emb_dim", 128),
    )
    unet = ConditionedUNet(unet_config).to(device)

    model = ConditionalDiffusionModel(cond_mlp, unet).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))

    diff_config = DiffusionConfig(
        timesteps=cfg.get("timesteps", 1000),
        beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02),
    )
    diffusion = GaussianDiffusion(diff_config)
    for attr in (
        "betas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
    ):
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    epochs = cfg.get("epochs", 100)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            x = batch["image"].to(device)
            f = batch["thickness"].to(device).float().unsqueeze(1)  # (B, 1)

            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = mu  # use mean for conditioning
                split = split_latents(z, father_dims, rest_dims)
                z_rest = split["z_rest"]  # (B, len(rest_dims))
            cond = torch.cat([z_rest, f], dim=1)  # (B, cond_input_dim)

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
            "cond_mlp_state_dict": cond_mlp.state_dict(),
            "unet_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "train_loss": mean_loss,
        }
        save_checkpoint(state, ckpt_dir, "diffusion_last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(state, ckpt_dir, "diffusion_best.pt")
            logger.info(f"Saved best checkpoint (loss={mean_loss:.4f})")

    logger.info("Diffusion training finished.")


if __name__ == "__main__":
    main()
