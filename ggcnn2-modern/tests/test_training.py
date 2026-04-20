"""Unit tests for training utilities."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ggcnn2.datasets.utils import GraspCollection, GraspRectangle, GraspRectangleCPAW
from ggcnn2.training.losses import combined_mse_loss
from ggcnn2.training.metrics import detect_grasps, iou, max_iou, post_process


class TestLosses:
    def test_combined_mse_loss_zero_when_perfect(self):
        preds = tuple(torch.zeros(1, 1, 10, 10) for _ in range(4))
        targets = tuple(torch.zeros(1, 1, 10, 10) for _ in range(4))
        result = combined_mse_loss(preds, targets)
        assert result["loss"].item() == pytest.approx(0.0)

    def test_combined_mse_loss_positive(self):
        preds = tuple(torch.ones(1, 1, 10, 10) for _ in range(4))
        targets = tuple(torch.zeros(1, 1, 10, 10) for _ in range(4))
        result = combined_mse_loss(preds, targets)
        assert result["loss"].item() > 0.0

    def test_per_head_losses_present(self):
        preds = tuple(torch.rand(1, 1, 10, 10) for _ in range(4))
        targets = tuple(torch.rand(1, 1, 10, 10) for _ in range(4))
        result = combined_mse_loss(preds, targets)
        assert set(result["losses"].keys()) == {"p_loss", "cos_loss", "sin_loss", "width_loss"}


class TestMetrics:
    def _make_fake_outputs(self, h=20, w=20):
        pos = torch.zeros(1, 1, h, w)
        pos[0, 0, h // 2, w // 2] = 1.0
        cos = torch.ones(1, 1, h, w)
        sin = torch.zeros(1, 1, h, w)
        width = torch.full((1, 1, h, w), 0.5)
        return pos, cos, sin, width

    def test_post_process_output_shapes(self):
        pos, cos, sin, width = self._make_fake_outputs(20, 20)
        q, ang, w = post_process(pos, cos, sin, width)
        assert q.shape == (20, 20)
        assert ang.shape == (20, 20)
        assert w.shape == (20, 20)

    def test_detect_grasps_returns_list(self):
        pos, cos, sin, width = self._make_fake_outputs(40, 40)
        q, ang, w = post_process(pos, cos, sin, width)
        grasps = detect_grasps(q, ang, w)
        assert isinstance(grasps, list)

    def test_iou_identical_grasps_is_one(self):
        pts = np.array([[80, 80], [120, 80], [120, 100], [80, 100]], dtype=float)
        gr1 = GraspRectangle(pts.copy())
        gr2 = GraspRectangle(pts.copy())
        score = iou(gr1, gr2)
        assert score > 0.9

    def test_iou_non_overlapping_is_zero(self):
        pts1 = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
        pts2 = np.array([[100, 100], [110, 100], [110, 105], [100, 105]], dtype=float)
        gr1 = GraspRectangle(pts1)
        gr2 = GraspRectangle(pts2)
        score = iou(gr1, gr2)
        assert score == 0.0

    def test_max_iou_finds_best_match(self):
        pts = np.array([[80, 80], [120, 80], [120, 100], [80, 100]], dtype=float)
        pred_cpaw = GraspRectangleCPAW(np.array([90.0, 100.0]), angle=0.0, length=20, width=40)
        gt = GraspCollection([GraspRectangle(pts.copy())])
        score = max_iou(pred_cpaw, gt)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
