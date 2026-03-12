"""
Extract latent means z from all training images using the frozen VAE encoder.
Saves z and thickness (and optional metadata) for logistic regression.
"""
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.morphomnist_dataset import MorphoMNISTDataset
from models.vae import ConvVAE, VAEConfig
from utils.checkpoint import load_checkpoint
from utils.latent_utils import save_latent_arrays
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract latents from train set using frozen VAE encoder.")
    parser.add_argument("--config", type=str, required=True, help="Path to logistic config YAML.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 123))

    # Load VAE encoder
    vae_config = VAEConfig(
        in_channels=cfg.get("in_channels", 1),
        latent_dim=cfg["latent_dim"],
        hidden_dims=tuple(cfg.get("hidden_dims", [32, 64, 128])),
        image_size=cfg.get("image_size", 28),
    )
    vae = ConvVAE(vae_config)
    load_checkpoint(Path(cfg["vae_checkpoint"]), model=vae, map_location="cpu")
    vae = vae.to(device)
    vae.eval()

    dataset = MorphoMNISTDataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        mode=cfg.get("data_mode", "images_csv"),
        data_file=cfg.get("train_data_file"),
        metadata_file=cfg.get("train_metadata_file"),
        image_key=cfg.get("image_key", "images"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    z_list = []
    thickness_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latents"):
            x = batch["image"].to(device)
            mu, _ = vae.encode(x)
            z_list.append(mu.cpu().numpy())
            t = batch["thickness"]
            if isinstance(t, torch.Tensor):
                t = t.numpy()
            else:
                t = np.array(t)
            thickness_list.append(t)

    z = np.concatenate(z_list, axis=0)
    thickness = np.concatenate(thickness_list, axis=0)

    save_dir = Path(cfg["output_dir"]) / "latents"
    save_latent_arrays(z, thickness, save_dir, prefix="train")

    print(f"Saved z shape {z.shape}, thickness shape {thickness.shape} to {save_dir}")


if __name__ == "__main__":
    main()
