"""Train conditional diffusion on CelebA (IDDPM ε-prediction, DDIM at inference)."""
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

from data.celeba_dataset import CelebADataset, CELEBA_ATTR_ORDER, expand_env_in_cfg
from models.diffusion_unet_factory import build_conditional_diffusion_unet
from models.vae_factory import build_vae_from_train_cfg
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.diffusion_utils import DiffusionConfig, GaussianDiffusion
from utils.logger import get_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train conditional diffusion on CelebA.")
    p.add_argument("--config", type=str, required=True, help="Path to diffusion config YAML.")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    expand_env_in_cfg(cfg)
    return cfg


def _parent_targets(batch: Dict[str, Any], cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    keys: Optional[List[str]] = cfg.get("parent_keys")
    if keys:
        return torch.stack([batch[k].to(device).float().squeeze(-1) for k in keys], dim=1)
    k = cfg.get("father_key", "Male")
    return batch[k].to(device).float()


def _resolve_checkpoint_subdir(cfg: Dict[str, Any], use_vae: bool, include_z_rest: bool) -> str:
    if cfg.get("checkpoint_subdir"):
        return str(cfg["checkpoint_subdir"])
    if not use_vae:
        return "parent_only"
    if include_z_rest:
        return "with_vae_condition"
    return "with_vae_parent_cond_only"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 123))

    use_vae_condition = bool(cfg.get("use_vae_condition", True))
    # If True (default) when use_vae_condition: concat(z_rest, f). If False: cond is f only (gender / parent_keys).
    condition_include_z_rest = bool(cfg.get("condition_include_z_rest", True))
    if not use_vae_condition:
        condition_include_z_rest = False

    run_subdir = _resolve_checkpoint_subdir(cfg, use_vae_condition, condition_include_z_rest)
    base_output = Path(cfg["output_dir"])
    output_dir = base_output / run_subdir
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_diffusion", log_dir=log_dir)
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")
    logger.info(
        f"Checkpoints under {output_dir} (use_vae_condition={use_vae_condition}, "
        f"condition_include_z_rest={condition_include_z_rest}, subdir={run_subdir})"
    )

    vae: Optional[nn.Module] = None
    n_parent_dims = 0
    rest_dim = 0

    if use_vae_condition:
        vae_cfg_path = cfg.get("vae_config", "configs/vae_celeba.yaml")
        with open(vae_cfg_path, "r") as f:
            vae_cfg = yaml.safe_load(f)
        expand_env_in_cfg(vae_cfg)
        latent_dim = int(vae_cfg["latent_dim"])
        n_parent_dims = int(vae_cfg.get("n_parent_dims", 0))
        rest_dim = latent_dim - n_parent_dims
        logger.info(f"Using latent_dim={latent_dim}, n_parent_dims={n_parent_dims}, z_rest_dim={rest_dim}")

        vae = build_vae_from_train_cfg(vae_cfg)
        load_checkpoint(Path(cfg["vae_checkpoint"]), model=vae, map_location="cpu")
        vae = vae.to(device)
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()
        logger.info("VAE encoder loaded and frozen.")
        if not condition_include_z_rest:
            logger.info(
                "condition_include_z_rest=False: UNet is trained with parent f only in cond (no z_rest); "
                "VAE remains in checkpoint stack for inference parity with cf-diffusion-project."
            )
    else:
        _pdim = len(cfg["parent_keys"]) if cfg.get("parent_keys") else 1
        logger.info(
            f"use_vae_condition=False (parent_only): cond_dim={_pdim}; VAE not loaded."
        )

    parent_dim = len(cfg["parent_keys"]) if cfg.get("parent_keys") else 1

    if use_vae_condition and condition_include_z_rest:
        cond_dim = rest_dim + parent_dim
    else:
        cond_dim = parent_dim

    if cfg.get("use_all_celeba_attrs"):
        factor_cols = list(CELEBA_ATTR_ORDER)
    else:
        factor_cols = list(cfg["celeba_attr_cols"])
    image_size = int(cfg.get("image_size", 64))

    train_dataset = CelebADataset(
        root=cfg["data_root"],
        split=cfg.get("train_split", "train"),
        factor_cols=factor_cols,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    unet = build_conditional_diffusion_unet(cfg, cond_dim).to(device)

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
        "betas",
        "alphas_cumprod",
        "alphas_cumprod_prev",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
    ):
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    epochs = cfg.get("epochs", 100)

    last_ckpt_path = ckpt_dir / "diffusion_last.pt"
    best_ckpt_path = ckpt_dir / "diffusion_best.pt"
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
        logger.info(f"Resuming diffusion training from epoch {start_epoch} using {last_ckpt_path}")
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        if "train_loss" in best_ckpt:
            best_loss = float(best_ckpt["train_loss"])
            logger.info(f"Loaded best_loss={best_loss:.6f} from {best_ckpt_path}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            x = batch["image"].to(device)
            f = _parent_targets(batch, cfg, device)

            if use_vae_condition and condition_include_z_rest:
                assert vae is not None
                with torch.no_grad():
                    mu, _logvar = vae.encode(x)
                    z = mu
                    z_rest = z[:, n_parent_dims:]
                cond = torch.cat([z_rest, f], dim=1)
            else:
                cond = f

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
            "condition_include_z_rest": condition_include_z_rest,
        }
        save_checkpoint(state, ckpt_dir, "diffusion_last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(state, ckpt_dir, "diffusion_best.pt")
            logger.info(f"Saved best checkpoint (loss={mean_loss:.4f})")

    logger.info("Diffusion training finished.")


if __name__ == "__main__":
    main()
