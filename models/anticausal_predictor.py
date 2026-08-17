"""
Convolutional anticausal predictors: image -> factor regression (and optional digit classification).
Used for Effectiveness-style metrics: h_theta^i(x) approximates y^i from observational training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ConvAnticausalConfig:
    in_channels: int = 1
    image_size: int = 28
    hidden_dims: Tuple[int, ...] = (32, 64, 128)
    head_hidden: int = 256
    regression_dim: int = 2  # number of scalar regression outputs (e.g. t, i)
    num_classes: int = 10
    use_classification: bool = False  # digit label head


class ConvAnticausalPredictor(nn.Module):
    """
    CNN encoder + regression head(s). Optionally a classification head for digit.

    Forward returns dict:
      - "y_reg": (B, regression_dim)
      - "logits": (B, num_classes) if use_classification
    """

    def __init__(self, config: ConvAnticausalConfig) -> None:
        super().__init__()
        self.config = config
        modules: List[nn.Module] = []
        in_ch = config.in_channels
        for h in config.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = h
        self.encoder = nn.Sequential(*modules)
        with torch.no_grad():
            d = torch.zeros(1, config.in_channels, config.image_size, config.image_size)
            flat = self.encoder(d).numel()
        self.fc = nn.Sequential(
            nn.Linear(flat, config.head_hidden),
            nn.ReLU(inplace=True),
        )
        self.reg_head = nn.Linear(config.head_hidden, config.regression_dim)
        self.cls_head: Optional[nn.Linear] = None
        if config.use_classification:
            self.cls_head = nn.Linear(config.head_hidden, config.num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(1)
        h = self.fc(h)
        out: Dict[str, torch.Tensor] = {"y_reg": self.reg_head(h)}
        if self.cls_head is not None:
            out["logits"] = self.cls_head(h)
        return out
