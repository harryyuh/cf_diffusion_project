"""
CelebA: aligned faces + list_attr_celeba / list_eval_partition.

Batches include ``image`` tensor (C,H,W) in [0,1] and one float tensor per attribute
column name (e.g. batch["Male"], batch["Smiling"]) with values in {-1, 1}.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

# Official attribute order in ``list_attr_celeba.txt`` (line 2).
CELEBA_ATTR_ORDER: Tuple[str, ...] = (
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
)


def _read_list_attr_celeba(path: Path) -> pd.DataFrame:
    """Parse list_attr_celeba.txt into a DataFrame indexed by image filename."""
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Unexpected list_attr_celeba format: {path}")
    # Line 0: count; line 1: attribute names only; following lines: filename + attrs.
    attr_cols = lines[1].split()
    if not attr_cols:
        raise ValueError("Bad header in list_attr_celeba")
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) != 1 + len(attr_cols):
            continue
        fname = parts[0]
        vals = [int(x) for x in parts[1:]]
        rows.append((fname, *vals))
    df = pd.DataFrame(rows, columns=["filename", *attr_cols])
    df = df.set_index("filename")
    return df


def _read_partition(path: Path) -> Dict[str, int]:
    """filename -> 0 train, 1 val, 2 test."""
    out: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        out[parts[0]] = int(parts[1])
    return out


class CelebADataset(Dataset):
    """
    Args:
        root: Directory containing ``img_align_celeba``, ``list_attr_celeba.txt``,
              ``list_eval_partition.txt``.
        split: ``train`` | ``val`` | ``test`` (from list_eval_partition).
        factor_cols: Attribute column names to expose in each batch (must exist in list_attr).
        image_size: Resize each side to this (square).
        partition_file / attr_file: override paths relative to ``root``.
    """

    def __init__(
        self,
        root: str,
        split: str,
        factor_cols: Sequence[str],
        image_size: int = 64,
        center_crop_size: int = 0,
        random_horizontal_flip: bool = False,
        partition_file: str = "list_eval_partition.txt",
        attr_file: str = "list_attr_celeba.txt",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split.lower()
        self.factor_cols = list(factor_cols)
        self.image_size = int(image_size)
        self.center_crop_size = int(center_crop_size or 0)
        self.random_horizontal_flip = bool(random_horizontal_flip)

        part_path = self.root / partition_file
        attr_path = self.root / attr_file
        if not part_path.is_file():
            raise FileNotFoundError(part_path)
        if not attr_path.is_file():
            raise FileNotFoundError(attr_path)

        self.img_dir = self.root / "img_align_celeba"
        if not self.img_dir.is_dir():
            raise FileNotFoundError(self.img_dir)

        partition = _read_partition(part_path)
        attrs = _read_list_attr_celeba(attr_path)

        missing_cols = [c for c in self.factor_cols if c not in attrs.columns]
        if missing_cols:
            raise KeyError(f"Unknown attribute columns: {missing_cols}. Example: {list(attrs.columns)[:5]}")

        split_code = {"train": 0, "val": 1, "test": 2}.get(self.split)
        if split_code is None:
            raise ValueError("split must be train, val, or test")

        names = [n for n in attrs.index if partition.get(n) == split_code]
        self._filenames: List[str] = sorted(names)
        self._attrs = attrs.loc[self._filenames, self.factor_cols].astype(np.float32)

        tfm_steps = []
        if self.center_crop_size > 0:
            tfm_steps.append(transforms.CenterCrop(self.center_crop_size))
        if self.random_horizontal_flip:
            tfm_steps.append(transforms.RandomHorizontalFlip())
        tfm_steps.extend(
            [
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self._tfm = transforms.Compose(tfm_steps)

    def __len__(self) -> int:
        return len(self._filenames)

    def factor_matrix(self, cols: Sequence[str]) -> np.ndarray:
        """Ground-truth factors (N, len(cols)) aligned with ``__getitem__`` order."""
        return self._attrs[list(cols)].to_numpy(dtype=np.float64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        name = self._filenames[idx]
        img_path = self.img_dir / name
        img = Image.open(img_path).convert("RGB")
        image = self._tfm(img)

        row = self._attrs.iloc[idx]
        batch: Dict[str, Any] = {"image": image}
        for j, col in enumerate(self.factor_cols):
            v = float(row.iloc[j])
            batch[col] = torch.tensor([v], dtype=torch.float32)
        return batch


def build_ca_official_celeba_complex_sampler(dataset: CelebADataset) -> WeightedRandomSampler:
    """Reproduce the released CA `(No_Beard, Bald)` four-class sampler."""
    factors = dataset.factor_matrix(["No_Beard", "Bald"])
    labels01 = ((factors + 1.0) / 2.0).astype(np.int64)
    combined = labels01[:, 0] * 2 + labels01[:, 1]
    class_counts = np.asarray([25301, 1690, 133756, 2023], dtype=np.float64)
    sample_weights = (1.0 / class_counts)[combined]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


def expand_env_in_cfg(cfg: Dict[str, Any]) -> None:
    if not cfg:
        return
    for k, v in list(cfg.items()):
        if isinstance(v, str) and "$" in v:
            cfg[k] = os.path.expandvars(v)
