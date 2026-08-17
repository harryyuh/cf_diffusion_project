"""Pendulum image dataset shared by CA and PAI+LoRA experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PENDULUM_ATTRS = ("pendulum", "light", "shadow_length", "shadow_position")
# Ranges used by the released Causal-Adapter Pendulum data loader.
PENDULUM_MINMAX = np.asarray(
    [[-40.0, 44.0], [60.0, 148.0], [3.0, 14.5], [-0.5, 20.5]], dtype=np.float32
)


def parse_pendulum_filename(path: str | Path) -> np.ndarray:
    parts = Path(path).stem.split("_")
    if len(parts) != 5:
        raise ValueError(f"Expected a_<p>_<l>_<sl>_<sp>.png, got {Path(path).name}")
    return np.asarray([float(x) for x in parts[1:]], dtype=np.float32)


def normalize_pendulum(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(values, dtype=torch.float32)
    lo = torch.from_numpy(PENDULUM_MINMAX[:, 0]).to(x)
    hi = torch.from_numpy(PENDULUM_MINMAX[:, 1]).to(x)
    return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)


def denormalize_pendulum(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(values, dtype=torch.float32)
    lo = torch.from_numpy(PENDULUM_MINMAX[:, 0]).to(x)
    hi = torch.from_numpy(PENDULUM_MINMAX[:, 1]).to(x)
    return x * (hi - lo) + lo


class PendulumDataset(Dataset):
    """Read the official synthetic Pendulum filenames and return [0,1] factors."""

    def __init__(self, root: str | Path, split: str, image_size: int = 256) -> None:
        split_root = Path(root) / split
        self.root = split_root if split_root.is_dir() else Path(root)
        self.paths = sorted(self.root.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No PNG images found under {self.root}")
        raw = np.stack([parse_pendulum_filename(p) for p in self.paths])
        self.raw_factors = torch.from_numpy(raw)
        self.factors = normalize_pendulum(raw)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        out = {
            "image": self.transform(image),
            "path": str(self.paths[index]),
            "factors": self.factors[index],
            "raw_factors": self.raw_factors[index],
            "index": index,
        }
        out.update({name: self.factors[index, i] for i, name in enumerate(PENDULUM_ATTRS)})
        return out

    def factor_matrix(self, names: Sequence[str] = PENDULUM_ATTRS) -> torch.Tensor:
        ids = [PENDULUM_ATTRS.index(name) for name in names]
        return self.factors[:, ids].clone()
