"""
File I/O utilities for GGCNN2.

Provides helpers for:
    - Loading / saving model checkpoints
    - Reading single depth images for inference
"""

from __future__ import annotations

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save model state dict, unwrapping DataParallel if needed."""
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load a previously saved state dict into *model* and return it."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def load_depth_image(path: str) -> "np.ndarray":  # type: ignore[name-defined]
    """Load a depth image (TIFF or PNG) and return a float32 numpy array."""
    from imageio.v3 import imread  # lazy import
    import numpy as np

    img = imread(path).astype(np.float32)
    return img
