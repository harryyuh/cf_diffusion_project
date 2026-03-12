from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Path,
    filename: str = "checkpoint.pt",
) -> Path:
    """
    Save a training checkpoint.

    Args:
        state: Dictionary containing model/optimizer state, epoch, config, etc.
        checkpoint_dir: Directory to save the checkpoint.
        filename: File name.

    Returns:
        Path to the saved checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / filename
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint(
    checkpoint_path: Path,
    model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint and optionally restore model and optimizer.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Optional model to load state dict into.
        optimizer: Optional optimizer to load state dict into.
        map_location: Device mapping for torch.load.

    Returns:
        Loaded checkpoint dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint