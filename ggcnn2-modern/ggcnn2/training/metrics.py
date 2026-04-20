"""
Evaluation metrics for GGCNN2.

Provides:
    - post_process: convert raw network output to quality/angle/width maps
    - detect_grasps: extract top-k grasps from processed maps
    - iou: IoU between two GraspRectangle instances
    - max_iou: best IoU between one prediction and all ground-truth grasps
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.feature import peak_local_max
from skimage.filters import gaussian

from ggcnn2.datasets.utils import GraspCollection, GraspRectangle, GraspRectangleCPAW


def post_process(
    pos_pred: torch.Tensor,
    cos_pred: torch.Tensor,
    sin_pred: torch.Tensor,
    width_pred: torch.Tensor,
    gaussian_sigma_q: float = 2.0,
    gaussian_sigma_ang: float = 2.0,
    gaussian_sigma_w: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw model outputs to post-processed numpy maps.

    The function:
        1. Converts to numpy (CPU).
        2. Computes angle from cos/sin.
        3. Rescales width from [0, 1] back to pixels.
        4. Applies Gaussian smoothing.

    Args:
        pos_pred, cos_pred, sin_pred, width_pred: Model output tensors (B=1, 1, H, W).
        gaussian_sigma_*: Sigma for Gaussian blur on each map.

    Returns:
        q_map    : Smoothed quality map (H, W).
        angle_map: Smoothed angle map in radians (H, W).
        width_map: Smoothed width map in pixels (H, W).
    """
    q = pos_pred.cpu().detach().numpy().squeeze()
    ang = (torch.atan2(sin_pred, cos_pred) / 2.0).cpu().detach().numpy().squeeze()
    w = width_pred.cpu().detach().numpy().squeeze() * 150.0

    q_map = gaussian(q, sigma=gaussian_sigma_q, preserve_range=True)
    angle_map = gaussian(ang, sigma=gaussian_sigma_ang, preserve_range=True)
    width_map = gaussian(w, sigma=gaussian_sigma_w, preserve_range=True)

    return q_map, angle_map, width_map


def detect_grasps(
    q_map: np.ndarray,
    angle_map: np.ndarray,
    width_map: np.ndarray,
    num_grasps: int = 1,
    min_distance: int = 20,
    threshold: float = 0.2,
) -> list[GraspRectangleCPAW]:
    """
    Extract top-k grasp candidates from the post-processed maps.

    Args:
        q_map:       Quality map (H, W).
        angle_map:   Angle map in radians (H, W).
        width_map:   Width map in pixels (H, W).
        num_grasps:  Maximum number of grasps to return.
        min_distance: Minimum pixel distance between peaks.
        threshold:   Minimum quality threshold for peaks.

    Returns:
        List of GraspRectangleCPAW objects.
    """
    grasps: list[GraspRectangleCPAW] = []
    peaks = peak_local_max(
        q_map,
        min_distance=min_distance,
        threshold_abs=threshold,
        num_peaks=num_grasps,
    )
    for pt in peaks:
        center = tuple(pt)
        angle = float(angle_map[center])
        width = float(width_map[center])
        length = width / 2.0
        grasps.append(GraspRectangleCPAW(np.array(center), angle, length, width))
    return grasps


def iou(
    pred: GraspRectangle,
    true: GraspRectangle,
    angle_threshold: float = np.pi / 6,
) -> float:
    """
    Compute IoU between two GraspRectangle objects.

    Returns 0.0 immediately if their angles differ by more than *angle_threshold*.
    """
    angle_diff = abs((pred.angle - true.angle + np.pi / 2) % np.pi - np.pi / 2)
    if angle_diff > angle_threshold:
        return 0.0

    rr1, cc1 = pred.polygon_coords()
    rr2, cc2 = true.polygon_coords()

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except (ValueError, AttributeError):
        return 0.0

    canvas = np.zeros((r_max, c_max), dtype=np.uint8)
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1

    union = int((canvas > 0).sum())
    if union == 0:
        return 0.0
    intersection = int((canvas == 2).sum())
    return intersection / union


def max_iou(
    pred_cpaw: GraspRectangleCPAW,
    true_collection: GraspCollection,
) -> float:
    """Return the maximum IoU between a predicted grasp and all ground-truth grasps."""
    pred_rect = pred_cpaw.as_grasp_rectangle
    best = 0.0
    for gt in true_collection.grasps:
        score = iou(pred_rect, gt)
        if score > best:
            best = score
    return best
