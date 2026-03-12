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
    channel_mults = (1, 2, 4)
    time_emb_dim: int = 128
    cond_emb_dim: int = 128


class ConditionedUNet(nn.Module):
    """
    Simple U-Net-like architecture conditioned on (z_rest, f) and timestep.
    """

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config

        time_dim = config.time_emb_dim
        cond_dim = config.cond_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
        )

        # Will receive external cond embedding of size cond_dim
        self.cond_proj = nn.Linear(cond_dim, time_dim)

        # Encoder
        ch = config.base_channels
        self.init_conv = nn.Conv2d(config.in_channels, ch, kernel_size=3, padding=1)

        downs = []
        in_ch = ch
        channels = [in_ch]
        for mult in config.channel_mults:
            out_ch = config.base_channels * mult
            downs.append(ResBlock(in_ch, out_ch, emb_dim=time_dim))
            channels.append(out_ch)
            in_ch = out_ch
        self.downs = nn.ModuleList(downs)

        # Middle
        self.mid_block1 = ResBlock(in_ch, in_ch, emb_dim=time_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, emb_dim=time_dim)

        # Decoder
        ups = []
        rev_channels = list(reversed(channels))
        in_ch = rev_channels[0]
        for ch_skip in rev_channels[1:]:
            ups.append(
                nn.ModuleDict(
                    {
                        "block": ResBlock(in_ch + ch_skip, ch_skip, emb_dim=time_dim),
                        "upsample": nn.ConvTranspose2d(
                            ch_skip, ch_skip, kernel_size=4, stride=2, padding=1
                        ),
                    }
                )
            )
            in_ch = ch_skip
        self.ups = nn.ModuleList(ups)
        self.final_conv = nn.Conv2d(config.base_channels, config.in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image x_t, shape (B, 1, 28, 28).
            t: Timesteps, shape (B,).
            cond_emb: Condition embedding of shape (B, cond_emb_dim).

        Returns:
            Predicted noise epsilon_hat of same shape as x.
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.config.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Combine with cond embedding
        c_emb = self.cond_proj(cond_emb)
        emb = t_emb + c_emb

        # U-Net forward
        x = self.init_conv(x)
        h_list = [x]
        h = x

        # Down
        for down in self.downs:
            h = down(h, emb)
            h_list.append(h)
            h = nn.AvgPool2d(2)(h)

        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

        # Up
        for up in self.ups:
            skip = h_list.pop()
            h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = up["block"](h, emb)
            h = up["upsample"](h)

        # Final conv to predict noise
        out = self.final_conv(h)
        return out