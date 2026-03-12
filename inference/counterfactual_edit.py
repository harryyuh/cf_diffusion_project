"""
Counterfactual image editing: given x and target thickness f_target,
generate x_cf ~ p_theta(x | z_rest, f_target).
Optionally save original, VAE reconstruction, and counterfactuals in a grid.
"""
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.morphomnist_dataset import MorphoMNISTDataset
from models.condition_mlp import ConditionMLP, ConditionMLPConfig
from models.diffusion_unet import ConditionedUNet, UNetConfig
from models.vae import ConvVAE, VAEConfig
from utils.checkpoint import load_checkpoint
from utils.diffusion_utils import DiffusionConfig, GaussianDiffusion
from utils.latent_utils import load_latent_splits, split_latents
from utils.seed import set_seed
from utils.visualization import save_image_grid


class ConditionalDiffusionModel(torch.nn.Module):
    """Wrapper: condition MLP + U-Net. Forward(x_t, t, cond_vector)."""

    def __init__(self, cond_mlp: ConditionMLP, unet: ConditionedUNet) -> None:
        super().__init__()
        self.cond_mlp = cond_mlp
        self.unet = unet

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_emb = self.cond_mlp(cond)
        return self.unet(x_t, t, cond_emb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual thickness editing.")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config YAML.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_models(cfg: Dict[str, Any], device: torch.device):
    """Load frozen VAE and conditional diffusion model."""
    vae_config = VAEConfig(
        in_channels=cfg.get("in_channels", 1),
        latent_dim=cfg["vae_latent_dim"],
        hidden_dims=tuple(cfg.get("vae_hidden_dims", [32, 64, 128])),
        image_size=cfg.get("image_size", 28),
    )
    vae = ConvVAE(vae_config)
    load_checkpoint(Path(cfg["vae_checkpoint"]), model=vae, map_location="cpu")
    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    cond_input_dim = cfg["cond_input_dim"]  # len(rest_dims) + 1
    cond_mlp_config = ConditionMLPConfig(
        input_dim=cond_input_dim,
        hidden_dim=cfg.get("cond_mlp_hidden_dim", 128),
        output_dim=cfg.get("cond_emb_dim", 128),
        dropout=0.0,
    )
    cond_mlp = ConditionMLP(cond_mlp_config)
    unet_config = UNetConfig(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("unet_base_channels", 32),
        channel_mults=tuple(cfg.get("unet_channel_mults", [1, 2, 4])),
        time_emb_dim=cfg.get("time_emb_dim", 128),
        cond_emb_dim=cfg.get("cond_emb_dim", 128),
    )
    unet = ConditionedUNet(unet_config)
    diffusion_model = ConditionalDiffusionModel(cond_mlp, unet)

    ckpt = torch.load(cfg["diffusion_checkpoint"], map_location="cpu")
    if "cond_mlp_state_dict" in ckpt:
        cond_mlp.load_state_dict(ckpt["cond_mlp_state_dict"])
        unet.load_state_dict(ckpt["unet_state_dict"])
    else:
        diffusion_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()

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

    return vae, diffusion_model, diffusion


def run_counterfactual(
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    x: torch.Tensor,
    z_rest: torch.Tensor,
    f_target: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate one counterfactual image for batch (single f_target per batch)."""
    b = x.size(0)
    f = torch.full((b, 1), f_target, dtype=z_rest.dtype, device=device)
    cond = torch.cat([z_rest, f], dim=1)
    with torch.no_grad():
        x_cf = diffusion.p_sample_loop(diffusion_model, x.shape, cond, device)
    return x_cf


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 123))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = Path(cfg["latent_analysis_dir"])
    father_dims, rest_dims = load_latent_splits(
        analysis_dir / "father_dims.json",
        analysis_dir / "rest_dims.json",
    )

    vae, diffusion_model, diffusion = load_models(cfg, device)

    dataset = MorphoMNISTDataset(
        root=cfg["data_root"],
        split=cfg.get("split", "test"),
        mode=cfg.get("data_mode", "images_csv"),
        data_file=cfg.get("data_file"),
        metadata_file=cfg.get("metadata_file"),
        image_key=cfg.get("image_key", "images"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    thickness_targets: List[float] = cfg.get("thickness_targets", [0.2, 0.4, 0.6, 0.8])
    max_images: Optional[int] = cfg.get("max_images")
    metadata_rows: List[Dict[str, Any]] = []

    count = 0
    for batch in tqdm(loader, desc="Counterfactual edit"):
        if max_images is not None and count >= max_images:
            break
        x = batch["image"].to(device)
        thickness_src = batch["thickness"]
        if isinstance(thickness_src, torch.Tensor):
            thickness_src = thickness_src.cpu().numpy()
        else:
            thickness_src = [float(thickness_src)] * x.size(0)

        with torch.no_grad():
            mu, logvar = vae.encode(x)
            z = mu
            split = split_latents(z, father_dims, rest_dims)
            z_rest = split["z_rest"]
            recon = vae.decode(z)

        # Generate counterfactuals for each target thickness
        cf_list = []
        for f_t in thickness_targets:
            x_cf = run_counterfactual(vae, diffusion_model, diffusion, x, z_rest, f_t, device)
            cf_list.append(x_cf)

        # Save grid: original | recon | cf_1 | cf_2 | ...
        n = x.size(0)
        for i in range(n):
            idx = count + i
            images = [x[i : i + 1], recon[i : i + 1]] + [c[i : i + 1] for c in cf_list]
            grid = torch.cat(images, dim=0)
            save_image_grid(
                grid,
                nrow=len(images),
                save_path=grid_dir / f"sample_{idx:05d}.png",
                cmap="gray",
            )
            metadata_rows.append({
                "sample_id": idx,
                "source_thickness": thickness_src[i],
                "target_thicknesses": thickness_targets,
                "grid_path": str(grid_dir / f"sample_{idx:05d}.png"),
            })

        count += n
        if max_images is not None and count >= max_images:
            break

    # Save metadata CSV
    with open(output_dir / "counterfactual_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "source_thickness", "target_thicknesses", "grid_path"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Saved {count} grids to {grid_dir}, metadata to {output_dir / 'counterfactual_metadata.csv'}")


if __name__ == "__main__":
    main()
