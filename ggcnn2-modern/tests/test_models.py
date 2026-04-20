"""Unit tests for GGCNN2 model."""

from __future__ import annotations

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ggcnn2.models.ggcnn2 import GGCNN2


class TestGGCNN2Architecture:
    """Test GGCNN2 model construction and forward pass."""

    def test_default_construction(self):
        model = GGCNN2()
        assert model is not None

    def test_forward_shape_depth_only(self):
        model = GGCNN2(input_channels=1)
        model.eval()
        x = torch.zeros(1, 1, 300, 300)
        pos, cos, sin, w = model(x)
        assert pos.shape == (1, 1, 300, 300)
        assert cos.shape == (1, 1, 300, 300)
        assert sin.shape == (1, 1, 300, 300)
        assert w.shape == (1, 1, 300, 300)

    def test_forward_shape_rgbd(self):
        model = GGCNN2(input_channels=4)
        model.eval()
        x = torch.zeros(2, 4, 300, 300)
        pos, cos, sin, w = model(x)
        assert pos.shape == (2, 1, 300, 300)

    def test_custom_filter_sizes(self):
        model = GGCNN2(input_channels=1, filter_sizes=[8, 8, 16, 8])
        x = torch.zeros(1, 1, 300, 300)
        pos, cos, sin, w = model(x)
        assert pos.shape == (1, 1, 300, 300)

    def test_compute_loss_returns_expected_keys(self):
        model = GGCNN2(input_channels=1)
        model.eval()
        x = torch.zeros(1, 1, 300, 300)
        y = (
            torch.zeros(1, 1, 300, 300),
            torch.zeros(1, 1, 300, 300),
            torch.zeros(1, 1, 300, 300),
            torch.zeros(1, 1, 300, 300),
        )
        result = model.compute_loss(x, y)
        assert "loss" in result
        assert "losses" in result
        assert "pred" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].shape == ()  # scalar

    def test_loss_is_non_negative(self):
        model = GGCNN2(input_channels=1)
        x = torch.randn(2, 1, 300, 300)
        y = tuple(torch.rand(2, 1, 300, 300) for _ in range(4))
        result = model.compute_loss(x, y)
        assert result["loss"].item() >= 0.0

    def test_xavier_init_applied(self):
        """Check that weights are not all zero after initialisation."""
        model = GGCNN2()
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                assert m.weight.abs().sum().item() > 0.0
                break

    def test_gradient_flows(self):
        model = GGCNN2(input_channels=1)
        x = torch.randn(1, 1, 300, 300, requires_grad=False)
        y = tuple(torch.rand(1, 1, 300, 300) for _ in range(4))
        result = model.compute_loss(x, y)
        result["loss"].backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                return  # at least one param has a gradient
        pytest.fail("No gradients found after backward pass")
