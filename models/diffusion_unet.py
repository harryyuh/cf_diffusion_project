from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.diffusion_utils import get_timestep_embedding


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = torch.relu(h)
        h = self.conv1(h)
        # Add conditioning embedding as bias
        emb_out = self.emb_proj(emb).view(emb.size(0), -1, 1, 1)
        h = h + emb_out
        h = self.norm2(h)
        h = torch.relu(h)
        h = self.conv2(h)
        return h + self.skip(x)


@dataclass
class UNetConfig:
    in_channels: int = 1
    base_channels: int = 32
    channel_mults: tuple = (1, 2, 4)
    time_emb_dim: int = 128
    # Dimension of raw condition vector (e.g. concat(z_rest, f))
    cond_dim: int = 128


class ConditionedUNet(nn.Module):
    """
    U-Net-like architecture conditioned on (z_rest, f) and timestep.

    Encoder (for 28x28 inputs, base_channels=C):
      - init_conv: in_channels -> C                  (28x28)
      - down1: C -> C                                (28x28)
      - pool ->                                     (14x14)
      - down2: C -> 2C                               (14x14)
      - pool ->                                     (7x7)
      - down3: 2C -> 4C                              (7x7)

    Middle:
      - two ResBlocks at 4C, 7x7

    Decoder:
      - up1: 4C (7x7) -> upsample to 14x14, concat skip2 (2C, 14x14),
              ResBlock(6C -> 2C)
      - up2: 2C (14x14) -> upsample to 28x28, concat skip1 (C, 28x28),
              ResBlock(3C -> C)
      - final_conv: C -> in_channels                 (28x28)
    """

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config

        time_dim = config.time_emb_dim
        cond_dim = config.cond_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
        )

        # Project raw condition vector to time embedding space
        self.cond_proj = nn.Linear(cond_dim, time_dim)

        C = config.base_channels

        # Initial conv
        self.init_conv = nn.Conv2d(config.in_channels, C, kernel_size=3, padding=1)

        # Encoder blocks
        self.down1 = ResBlock(C, C, emb_dim=time_dim)        # 28x28
        self.down2 = ResBlock(C, 2 * C, emb_dim=time_dim)    # 14x14
        self.down3 = ResBlock(2 * C, 4 * C, emb_dim=time_dim)  # 7x7

        self.pool = nn.AvgPool2d(2)

        # Middle blocks at 7x7, 4C channels
        self.mid_block1 = ResBlock(4 * C, 4 * C, emb_dim=time_dim)
        self.mid_block2 = ResBlock(4 * C, 4 * C, emb_dim=time_dim)

        # Decoder blocks
        # up1: h (4C, 7x7) -> upsample to 14x14, concat skip2 (2C, 14x14) -> ResBlock(6C -> 2C)
        self.up1 = ResBlock(4 * C + 2 * C, 2 * C, emb_dim=time_dim)
        # up2: h (2C, 14x14) -> upsample to 28x28, concat skip1 (C, 28x28) -> ResBlock(3C -> C)
        self.up2 = ResBlock(2 * C + C, C, emb_dim=time_dim)

        # Final conv to predict noise
        self.final_conv = nn.Conv2d(C, config.in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image x_t, shape (B, in_channels, 28, 28).
            t: Timesteps, shape (B,).
            cond: Condition vector of shape (B, cond_dim), e.g. concat(z_rest, f).

        Returns:
            Predicted noise epsilon_hat of same shape as x.
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.config.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Condition projection and combine
        c_emb = self.cond_proj(cond)
        emb = t_emb + c_emb

        C = self.config.base_channels

        # Encoder
        h0 = self.init_conv(x)          # (B, C, 28, 28)
        h1 = self.down1(h0, emb)        # (B, C, 28, 28)
        p1 = self.pool(h1)              # (B, C, 14, 14)

        h2 = self.down2(p1, emb)        # (B, 2C, 14, 14)
        p2 = self.pool(h2)              # (B, 2C, 7, 7)

        h3 = self.down3(p2, emb)        # (B, 4C, 7, 7)

        # Middle
        h = self.mid_block1(h3, emb)    # (B, 4C, 7, 7)
        h = self.mid_block2(h, emb)     # (B, 4C, 7, 7)

        # Decoder
        # Up1: 7 -> 14, concat skip from h2 (2C, 14x14)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")  # (B, 4C, 14, 14)
        # Ensure spatial sizes match before concat (defensive)
        if h.shape[2:] != h2.shape[2:]:
            H, W = h.shape[2], h.shape[3]
            h2_crop = h2[:, :, :H, :W]
        else:
            h2_crop = h2
        h = torch.cat([h, h2_crop], dim=1)  # (B, 6C, 14, 14)
        h = self.up1(h, emb)               # (B, 2C, 14, 14)

        # Up2: 14 -> 28, concat skip from h1 (C, 28x28)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")  # (B, 2C, 28, 28)
        if h.shape[2:] != h1.shape[2:]:
            H, W = h.shape[2], h.shape[3]
            h1_crop = h1[:, :, :H, :W]
        else:
            h1_crop = h1
        h = torch.cat([h, h1_crop], dim=1)  # (B, 3C, 28, 28)
        h = self.up2(h, emb)               # (B, C, 28, 28)

        out = self.final_conv(h)
        return out