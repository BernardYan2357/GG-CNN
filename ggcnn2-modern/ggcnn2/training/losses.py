"""
Loss functions for GGCNN2 training.

The default combined loss is a sum of MSE losses on four heads:
    - Grasp quality (position)
    - cos(2 * angle)
    - sin(2 * angle)
    - Normalised gripper width
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def combined_mse_loss(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Compute per-head MSE losses and their sum.

    Args:
        predictions: (pos_pred, cos_pred, sin_pred, width_pred) tensors.
        targets:     (pos_true, cos_true, sin_true, width_true) tensors.

    Returns:
        dict with ``loss`` (scalar) and ``losses`` (per-head dict).
    """
    pos_pred, cos_pred, sin_pred, width_pred = predictions
    pos_true, cos_true, sin_true, width_true = targets

    p_loss = F.mse_loss(pos_pred, pos_true)
    cos_loss = F.mse_loss(cos_pred, cos_true)
    sin_loss = F.mse_loss(sin_pred, sin_true)
    width_loss = F.mse_loss(width_pred, width_true)

    return {
        "loss": p_loss + cos_loss + sin_loss + width_loss,
        "losses": {
            "p_loss": p_loss,
            "cos_loss": cos_loss,
            "sin_loss": sin_loss,
            "width_loss": width_loss,
        },
    }
