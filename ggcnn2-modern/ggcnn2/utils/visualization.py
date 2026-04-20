"""
Visualization utilities for GGCNN2.

Provides helpers to:
    - Draw predicted grasp rectangles on images
    - Plot quality / angle / width maps side-by-side
    - Save comparison figures (prediction vs ground-truth)
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ggcnn2.datasets.utils import GraspCollection, GraspRectangle, GraspRectangleCPAW


def draw_grasps(
    ax: "plt.Axes",
    grasps: list[GraspRectangleCPAW],
    color: str = "green",
    label: Optional[str] = None,
) -> None:
    """Draw predicted grasp rectangles on a Matplotlib axis."""
    for i, g in enumerate(grasps):
        rect = g.as_grasp_rectangle
        pts = np.vstack([rect.points, rect.points[0]])  # close the loop
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=1.5)
    if label is not None:
        ax.add_patch(mpatches.Patch(color=color, label=label))


def draw_gt_grasps(
    ax: "plt.Axes",
    collection: GraspCollection,
    color: str = "red",
    label: Optional[str] = None,
    max_draw: int = 5,
) -> None:
    """Draw ground-truth grasps from a GraspCollection."""
    for gr in collection.grasps[:max_draw]:
        pts = np.vstack([gr.points, gr.points[0]])
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=1.0, alpha=0.6)
    if label is not None:
        ax.add_patch(mpatches.Patch(color=color, label=label))


def plot_output(
    rgb: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    q_map: np.ndarray,
    angle_map: np.ndarray,
    width_map: np.ndarray,
    pred_grasps: Optional[list[GraspRectangleCPAW]] = None,
    gt_collection: Optional[GraspCollection] = None,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot a 2x3 grid showing depth, quality, angle, width, and grasp overlays.

    Args:
        rgb:          Optional RGB image (H, W, 3).
        depth:        Optional depth image (H, W).
        q_map:        Quality map (H, W).
        angle_map:    Angle map (H, W).
        width_map:    Width map (H, W).
        pred_grasps:  Predicted GraspRectangleCPAW list.
        gt_collection: Ground-truth GraspCollection.
        save_path:    If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    n_cols = 4 + (rgb is not None) + (depth is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    ax_idx = 0

    if depth is not None:
        axes[ax_idx].imshow(depth, cmap="gray")
        axes[ax_idx].set_title("Depth")
        ax_idx += 1

    if rgb is not None:
        axes[ax_idx].imshow(rgb)
        axes[ax_idx].set_title("RGB")
        ax_idx += 1

    axes[ax_idx].imshow(q_map, cmap="jet", vmin=0, vmax=1)
    axes[ax_idx].set_title("Quality")
    ax_idx += 1

    axes[ax_idx].imshow(angle_map, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
    axes[ax_idx].set_title("Angle")
    ax_idx += 1

    axes[ax_idx].imshow(width_map, cmap="jet")
    axes[ax_idx].set_title("Width")
    ax_idx += 1

    # Overlay grasp predictions and ground truth
    bg = depth if depth is not None else (rgb if rgb is not None else q_map)
    axes[ax_idx].imshow(bg, cmap="gray" if bg.ndim == 2 else None)
    if pred_grasps:
        draw_grasps(axes[ax_idx], pred_grasps, color="lime", label="Predicted")
    if gt_collection:
        draw_gt_grasps(axes[ax_idx], gt_collection, color="red", label="Ground truth")
    if pred_grasps or gt_collection:
        axes[ax_idx].legend(loc="upper right", fontsize=7)
    axes[ax_idx].set_title("Grasps")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
