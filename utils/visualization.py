from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_image_grid(
    images: torch.Tensor,
    nrow: int,
    path: Path,
    cmap: Optional[str] = None,
    dpi: Optional[float] = None,
) -> None:
    """
    Save a simple grid of images to disk.

    Args:
        images: Tensor of shape (B, C, H, W) or (B, H, W).
        nrow: Number of images per row (columns in the matplotlib grid).
        path: Output file path.
        cmap: Optional matplotlib colormap (e.g. "gray").
    """
    images = images.detach().cpu()
    if images.dim() == 4 and images.size(1) == 1:
        images = images.squeeze(1)  # (B, H, W)
    elif images.dim() == 4 and images.size(1) == 3:
        # (B, 3, H, W) -> (B, H, W, 3)
        images = images.permute(0, 2, 3, 1)

    b = images.size(0)
    ncol = max(1, int(nrow))
    nrow_grid = (b + ncol - 1) // ncol

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrow_grid, ncol, figsize=(ncol, nrow_grid))
    ax_flat = np.ravel(np.atleast_1d(axes))

    for idx in range(nrow_grid * ncol):
        ax = ax_flat[idx]
        ax.axis("off")
        if idx < b:
            img = images[idx]
            if img.dim() == 2:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)

    plt.tight_layout()
    save_kw: Dict[str, Any] = {}
    if dpi is not None:
        save_kw["dpi"] = float(dpi)
    fig.savefig(path, **save_kw)
    plt.close(fig)

