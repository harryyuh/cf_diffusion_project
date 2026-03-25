from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def mse(recon: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(recon, target).item()


def mae(recon: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(recon, target).item()


def bce(recon: torch.Tensor, target: torch.Tensor) -> float:
    return F.binary_cross_entropy(recon, target).item()


def reconstruction_metrics(recon: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Basic reconstruction metrics for images.
    """
    return {
        "mse": mse(recon, target),
        "mae": mae(recon, target),
        "bce": bce(recon, target),
    }

