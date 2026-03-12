from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch


def save_image_grid(
    images: torch.Tensor,
    nrow: int,
    path: Path,
    cmap: Optional[str] = None,
) -> None:
    """
    Save a simple grid of images to disk.

    Args:
        images: Tensor of shape (B, C, H, W) or (B, H, W).
        nrow: Number of images per row.
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
    ncol = nrow
    nrow_grid = (b + ncol - 1) // ncol

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrow_grid, ncol, figsize=(ncol, nrow_grid))
    if nrow_grid == 1 and ncol == 1:
        axes = [[axes]]
    elif nrow_grid == 1:
        axes = [axes]

    idx = 0
    for r in range(nrow_grid):
        for c in range(ncol):
            ax = axes[r][c]
            ax.axis("off")
            if idx < b:
                img = images[idx]
                if img.dim() == 2:
                    ax.imshow(img, cmap=cmap)
                else:
                    ax.imshow(img)
            idx += 1

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

