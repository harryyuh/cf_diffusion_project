"""
IDDPM-style UNet (OpenAI improved-diffusion) with a single vector condition ``cond``:
``emb = time_embed(t) + Linear(cond_dim -> time_embed_dim)``, matching cf-diffusion-project's
:class:`models.diffusion_unet.ConditionedUNet` injection — no internal VAE encoder, no causal graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from improved_diffusion.nn_layers import (
    SiLU,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from improved_diffusion.unet_core import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
)


def channel_mult_for_image_size(image_size: int) -> Tuple[int, ...]:
    if image_size == 28:
        return (1, 2, 2)
    if image_size == 32:
        return (1, 2, 2, 2)
    if image_size == 64:
        return (1, 2, 3, 4)
    if image_size == 96:
        return (1, 2, 3, 4)
    raise ValueError(f"Unsupported image_size={image_size} for IDDPMCondUNet")


def attention_ds_tuple(image_size: int, attention_resolutions: str) -> Tuple[int, ...]:
    """Same convention as CausalDiffAE_new ``script_util.create_model``: ds = image_size // res."""
    parts = [p.strip() for p in attention_resolutions.split(",") if p.strip()]
    return tuple(image_size // int(p) for p in parts)


@dataclass
class IDDPMCondUNetConfig:
    image_size: int = 28
    in_channels: int = 1
    cond_dim: int = 514
    model_channels: int = 128
    num_res_blocks: int = 3
    attention_resolutions: str = "16,8"
    dropout: float = 0.0
    conv_resample: bool = True
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = False
    num_heads: int = 1
    num_heads_upsample: int = -1


class IDDPMCondUNet(nn.Module):
    """
    Full U-Net with self-attention (guided-diffusion style). Forward matches ``ConditionedUNet``:
    ``forward(x, timesteps, cond) -> eps``.
    """

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        cond_dim: int,
        model_channels: int = 128,
        num_res_blocks: int = 3,
        attention_resolutions: Tuple[int, ...] = (1, 3),
        channel_mult: Tuple[int, ...] = (1, 2, 2),
        dropout: float = 0.0,
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        num_heads: int = 1,
        num_heads_upsample: int = -1,
        dims: int = 2,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_dim = cond_dim

        out_channels = in_channels if out_channels is None else out_channels
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        input_block_chans: List[int] = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample)
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = emb + self.cond_proj(cond)

        hs = []
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)


def build_iddpm_cond_unet(cfg: Dict[str, Any], cond_dim: int) -> IDDPMCondUNet:
    """Build from diffusion train YAML (keys ``iddpm_*``)."""
    mc = int(cfg.get("iddpm_model_channels", 128))
    if mc % 32 != 0:
        raise ValueError(
            f"iddpm_model_channels ({mc}) must be divisible by 32 (OpenAI GroupNorm uses num_groups=32)."
        )
    image_size = int(cfg.get("image_size", 28))
    icfg = IDDPMCondUNetConfig(
        image_size=image_size,
        in_channels=int(cfg.get("in_channels", 1)),
        cond_dim=cond_dim,
        model_channels=int(cfg.get("iddpm_model_channels", 128)),
        num_res_blocks=int(cfg.get("iddpm_num_res_blocks", 3)),
        attention_resolutions=str(cfg.get("iddpm_attention_resolutions", "16,8")),
        dropout=float(cfg.get("iddpm_dropout", 0.0)),
        conv_resample=bool(cfg.get("iddpm_conv_resample", True)),
        use_checkpoint=bool(cfg.get("iddpm_use_checkpoint", False)),
        use_scale_shift_norm=bool(cfg.get("iddpm_use_scale_shift_norm", False)),
        num_heads=int(cfg.get("iddpm_num_heads", 1)),
        num_heads_upsample=int(cfg.get("iddpm_num_heads_upsample", -1)),
    )
    cm = channel_mult_for_image_size(icfg.image_size)
    att = attention_ds_tuple(icfg.image_size, icfg.attention_resolutions)
    return IDDPMCondUNet(
        image_size=icfg.image_size,
        in_channels=icfg.in_channels,
        cond_dim=cond_dim,
        model_channels=icfg.model_channels,
        num_res_blocks=icfg.num_res_blocks,
        attention_resolutions=att,
        channel_mult=cm,
        dropout=icfg.dropout,
        conv_resample=icfg.conv_resample,
        use_checkpoint=icfg.use_checkpoint,
        use_scale_shift_norm=icfg.use_scale_shift_norm,
        num_heads=icfg.num_heads,
        num_heads_upsample=icfg.num_heads_upsample,
        out_channels=int(cfg.get("in_channels", 1)),
    )
