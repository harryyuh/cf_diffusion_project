from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def load_latent_splits(
    father_path: Path,
    rest_path: Path,
) -> Tuple[List[int], List[int]]:
    """
    Load father_dims and rest_dims indices from JSON files.

    Args:
        father_path: Path to father_dims.json
        rest_path: Path to rest_dims.json

    Returns:
        (father_dims, rest_dims), each a list of indices.
    """
    father_dims = json.loads(father_path.read_text())
    rest_dims = json.loads(rest_path.read_text())
    return father_dims, rest_dims


def split_latents(
    z: torch.Tensor,
    father_dims: List[int],
    rest_dims: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Split latent vectors z into father-related and rest dimensions.

    Args:
        z: Latent tensor of shape (B, D).
        father_dims: Indices of father-related latent dimensions.
        rest_dims: Indices of complementary latent dimensions.

    Returns:
        dict with:
          - 'z_father': (B, len(father_dims))
          - 'z_rest': (B, len(rest_dims))
    """
    device = z.device
    dtype = z.dtype

    if father_dims:
        idx_father = torch.as_tensor(father_dims, dtype=torch.long, device=device)
        z_father = z.index_select(dim=1, index=idx_father)
    else:
        z_father = torch.empty(z.size(0), 0, device=device, dtype=dtype)

    if rest_dims:
        idx_rest = torch.as_tensor(rest_dims, dtype=torch.long, device=device)
        z_rest = z.index_select(dim=1, index=idx_rest)
    else:
        z_rest = torch.empty(z.size(0), 0, device=device, dtype=dtype)

    return {"z_father": z_father, "z_rest": z_rest}

