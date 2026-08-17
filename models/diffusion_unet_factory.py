"""Construct the conditional diffusion U-Net from a train/inference YAML."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from models.diffusion_unet import ConditionedUNet, UNetConfig


def build_conditional_diffusion_unet(cfg: Dict[str, Any], cond_dim: int) -> nn.Module:
    """
    ``unet_arch`` (default ``compact``): legacy :class:`~models.diffusion_unet.ConditionedUNet``.
    ``iddpm`` / ``improved`` / ``openai``: guided-diffusion style U-Net with attention
    (:class:`~improved_diffusion.iddpm_cond_unet.IDDPMCondUNet`), same ``forward(x,t,cond)``.
    Condition is ``emb = time_embed(t) + Linear(cond_dim -> emb_dim)`` — same injection as compact.
    """
    arch = str(cfg.get("unet_arch", "compact")).lower()
    if arch in ("iddpm", "improved", "openai"):
        from improved_diffusion.iddpm_cond_unet import build_iddpm_cond_unet

        return build_iddpm_cond_unet(cfg, cond_dim)

    unet_config = UNetConfig(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("unet_base_channels", 32),
        channel_mults=tuple(cfg.get("unet_channel_mults", [1, 2, 4])),
        time_emb_dim=cfg.get("time_emb_dim", 128),
        cond_dim=cond_dim,
    )
    return ConditionedUNet(unet_config)
