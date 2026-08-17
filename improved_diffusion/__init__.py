"""Vendored OpenAI improved-diffusion building blocks + cf-specific conditional UNet."""

from improved_diffusion.iddpm_cond_unet import (
    IDDPMCondUNet,
    IDDPMCondUNetConfig,
    attention_ds_tuple,
    build_iddpm_cond_unet,
    channel_mult_for_image_size,
)

__all__ = [
    "IDDPMCondUNet",
    "IDDPMCondUNetConfig",
    "attention_ds_tuple",
    "build_iddpm_cond_unet",
    "channel_mult_for_image_size",
]
