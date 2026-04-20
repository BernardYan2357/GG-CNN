"""
GGCNN2 Model Architecture

Improved Generative Grasping CNN v2 with:
- Deeper encoder-decoder structure
- Dilated convolutions for expanded receptive field
- MaxPooling for multi-scale feature extraction
- Bilinear upsampling for reconstruction
- 4 output heads: position, cos(angle), sin(angle), width
- Xavier weight initialization
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GGCNN2(nn.Module):
    """
    GGCNN2: Improved Generative Grasping CNN.

    Architecture:
        Encoder:
            - Two blocks of Conv(11x11) + Conv(5x5) + MaxPool(2x2)
        Middle:
            - Two dilated convolutions with increasing dilation rates
        Decoder:
            - Bilinear upsampling x2
            - Final 1x1 convolutions for each output head

    Args:
        input_channels: Number of input channels (1 for depth, 4 for depth+RGB).
        filter_sizes: List of 4 integers defining channel sizes at each stage.
        l3_k_size: Kernel size for the dilated convolution layers.
        dilations: List of 2 dilation rates for the dilated convolutions.

    Returns:
        Tuple of (pos, cos, sin, width) tensors, each of shape (B, 1, H, W).
    """

    def __init__(
        self,
        input_channels: int = 1,
        filter_sizes: Optional[list[int]] = None,
        l3_k_size: int = 5,
        dilations: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [16, 16, 32, 16]

        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            # Block 1: large receptive field initial features
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: deeper features
            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions for expanded receptive field
            nn.Conv2d(
                filter_sizes[1], filter_sizes[2],
                kernel_size=l3_k_size,
                dilation=dilations[0],
                stride=1,
                padding=(l3_k_size // 2 * dilations[0]),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                filter_sizes[2], filter_sizes[2],
                kernel_size=l3_k_size,
                dilation=dilations[1],
                stride=1,
                padding=(l3_k_size // 2 * dilations[1]),
                bias=True,
            ),
            nn.ReLU(inplace=True),

            # Bilinear upsampling x2 (replaces TransposedConv for smoother output)
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to all Conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            pos_output: Grasp quality map (B, 1, H, W).
            cos_output: cos(2*angle) map (B, 1, H, W).
            sin_output: sin(2*angle) map (B, 1, H, W).
            width_output: Grasp width map (B, 1, H, W), normalised to [0, 1].
        """
        x = self.features(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(
        self,
        xc: torch.Tensor,
        yc: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict:
        """
        Compute combined MSE loss for all four output heads.

        Args:
            xc: Input depth/RGB tensor on device.
            yc: Tuple of (pos, cos, sin, width) ground-truth tensors on device.

        Returns:
            dict with keys:
                loss      - combined scalar loss
                losses    - per-head loss dict
                pred      - per-head prediction dict
        """
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            "loss": p_loss + cos_loss + sin_loss + width_loss,
            "losses": {
                "p_loss": p_loss,
                "cos_loss": cos_loss,
                "sin_loss": sin_loss,
                "width_loss": width_loss,
            },
            "pred": {
                "pos": pos_pred,
                "cos": cos_pred,
                "sin": sin_pred,
                "width": width_pred,
            },
        }
