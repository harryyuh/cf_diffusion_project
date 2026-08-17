"""Build :class:`~models.vae.ConvVAE` from a training YAML dict (CelebA / MorphoMNIST)."""
from __future__ import annotations

from typing import Any, Dict

from models.vae import ConvVAE, VAEConfig


def build_vae_from_train_cfg(cfg: Dict[str, Any]) -> ConvVAE:
    """Construct ``ConvVAE`` from keys written by ``configs/vae_celeba.yaml``-style files."""
    hidden = cfg.get("hidden_dims")
    if hidden is None:
        hidden = (32, 64, 128)
    else:
        hidden = tuple(int(x) for x in hidden)

    vc = VAEConfig(
        in_channels=int(cfg.get("in_channels", 3)),
        latent_dim=int(cfg["latent_dim"]),
        hidden_dims=hidden,
        image_size=int(cfg.get("image_size", 64)),
        n_parent_dims=int(cfg.get("n_parent_dims", 0)),
        parent_pred_hidden=int(cfg.get("parent_pred_hidden", 32)),
        parent_pred_dim=int(cfg.get("parent_pred_dim", 1)),
        use_adversary=bool(cfg.get("use_adversary", False)),
        adv_hidden=int(cfg.get("adv_hidden", 32)),
        dual_branch_encoder=bool(cfg.get("dual_branch_encoder", False)),
    )
    return ConvVAE(vc)
