"""Unit tests for dataset utilities."""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ggcnn2.datasets.utils import (
    GraspCollection,
    GraspRectangle,
    GraspRectangleCPAW,
)


def make_square_grasp(cx: float = 100, cy: float = 100, half: float = 20) -> GraspRectangle:
    """Create a simple axis-aligned square grasp."""
    pts = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ])
    return GraspRectangle(pts)


class TestGraspRectangle:
    def test_center(self):
        gr = make_square_grasp(100, 100, 20)
        center = gr.center
        assert center[0] == 100
        assert center[1] == 100

    def test_width(self):
        gr = make_square_grasp(100, 100, 20)
        assert math.isclose(gr.width, 40.0, abs_tol=1e-5)

    def test_rotate_preserves_width(self):
        gr = make_square_grasp(100, 100, 20)
        original_width = gr.width
        gr.rotate(np.pi / 4, (100, 100))
        assert math.isclose(gr.width, original_width, rel_tol=1e-4)

    def test_scale(self):
        gr = make_square_grasp(100, 100, 20)
        gr.scale(2.0)
        assert math.isclose(gr.width, 80.0, abs_tol=1e-5)


class TestGraspRectangleCPAW:
    def test_roundtrip(self):
        """CPAW → GraspRectangle should preserve center approximately."""
        cpaw = GraspRectangleCPAW(np.array([100.0, 150.0]), angle=0.3, length=30, width=60)
        rect = cpaw.as_grasp_rectangle
        c = rect.center
        assert abs(int(c[0]) - 100) <= 2
        assert abs(int(c[1]) - 150) <= 2


class TestGraspCollection:
    def test_load_from_jacquard(self, tmp_path):
        # Create a minimal grasps.txt in Jacquard format: x;y;theta;w;h
        grasp_file = tmp_path / "0_grasps.txt"
        grasp_file.write_text("512.0;512.0;0.0;100.0;50.0\n100.0;200.0;45.0;80.0;40.0\n")

        gc = GraspCollection.load_from_jacquard(str(grasp_file), scale=1.0)
        assert len(gc.grasps) == 2

    def test_load_from_jacquard_with_scale(self, tmp_path):
        grasp_file = tmp_path / "0_grasps.txt"
        grasp_file.write_text("512.0;512.0;0.0;100.0;50.0\n")

        gc_full = GraspCollection.load_from_jacquard(str(grasp_file), scale=1.0)
        gc_scaled = GraspCollection.load_from_jacquard(str(grasp_file), scale=0.5)

        center_full = gc_full.center.astype(float)
        center_scaled = gc_scaled.center.astype(float)
        assert np.allclose(center_full * 0.5, center_scaled, atol=1.0)

    def test_generate_maps_shape(self, tmp_path):
        grasp_file = tmp_path / "0_grasps.txt"
        grasp_file.write_text("150.0;150.0;0.0;60.0;30.0\n")

        gc = GraspCollection.load_from_jacquard(str(grasp_file), scale=300 / 1024.0)
        pos, angle, width = gc.generate_maps((300, 300))
        assert pos.shape == (300, 300)
        assert angle.shape == (300, 300)
        assert width.shape == (300, 300)
        assert pos.max() <= 1.0

    def test_rotate_and_zoom(self, tmp_path):
        grasp_file = tmp_path / "0_grasps.txt"
        grasp_file.write_text("512.0;512.0;0.0;100.0;50.0\n")
        gc = GraspCollection.load_from_jacquard(str(grasp_file), scale=1.0)
        gc.rotate(np.pi / 2, (512, 512))
        gc.zoom(0.8, (512, 512))
        # Just ensure it doesn't raise
        assert len(gc.grasps) == 1
