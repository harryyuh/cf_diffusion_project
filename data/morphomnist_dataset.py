from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_idx_images(path: Path) -> np.ndarray:
    """Load IDX image file (e.g. MNIST images)."""
    import gzip

    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, num, rows, cols = np.frombuffer(data, dtype=">i4", count=4)
    if magic != 2051:
        raise ValueError(f"Unexpected magic number {magic} in {path}")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(num, rows, cols)
    return images


def _load_idx_labels(path: Path) -> np.ndarray:
    """Load IDX label file (e.g. MNIST labels)."""
    import gzip

    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, num = np.frombuffer(data, dtype=">i4", count=2)
    if magic != 2049:
        raise ValueError(f"Unexpected magic number {magic} in {path}")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels


class MorphoMNISTDataset(Dataset):
    """
    MorphoMNIST dataset reading from IDX image/label files and a CSV of morphometrics.

    Expected files in data_root:
      - train-images-idx3-ubyte.gz
      - train-labels-idx1-ubyte.gz
      - train-morpho.csv
      - t10k-images-idx3-ubyte.gz
      - t10k-labels-idx1-ubyte.gz
      - t10k-morpho.csv

    Each item is a dict with:
      - "image": FloatTensor [1, 28, 28], in [0,1]
      - "label": LongTensor scalar
      - "morpho": dict of morphometric attributes (from CSV row)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "images_csv",
        data_file: Optional[str] = None,
        metadata_file: Optional[str] = None,
        image_key: str = "images",
    ) -> None:
        super().__init__()
        if mode != "images_csv":
            raise ValueError(f"Only mode='images_csv' is supported, got {mode}")

        root_path = Path(root)
        if split == "train":
            img_path = root_path / "train-images-idx3-ubyte.gz"
            label_path = root_path / "train-labels-idx1-ubyte.gz"
            morpho_path = root_path / "train-morpho.csv"
        else:
            img_path = root_path / "t10k-images-idx3-ubyte.gz"
            label_path = root_path / "t10k-labels-idx1-ubyte.gz"
            morpho_path = root_path / "t10k-morpho.csv"

        self.images = _load_idx_images(img_path)
        self.labels = _load_idx_labels(label_path)

        # Load morphometrics CSV as list-of-dicts
        if morpho_path.exists():
            import pandas as pd

            df = pd.read_csv(morpho_path)
            self.morpho_cols = list(df.columns)
            self.morpho_values = df.to_dict(orient="records")
        else:
            self.morpho_cols = []
            self.morpho_values = [{} for _ in range(len(self.images))]

        if not (len(self.images) == len(self.labels) == len(self.morpho_values)):
            raise ValueError("Images, labels, and morphometrics must have the same length.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # [1, 28, 28]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        morpho_row = self.morpho_values[idx]
        morpho_dict: Dict[str, float] = {}
        for k, v in morpho_row.items():
            try:
                morpho_dict[k] = float(v)
            except (TypeError, ValueError):
                continue

        thickness = morpho_dict.get("thickness", 0.0)
        return {
            "image": img,
            "label": label,
            "morpho": morpho_dict,
            "thickness": torch.tensor(thickness, dtype=torch.float32),
        }

