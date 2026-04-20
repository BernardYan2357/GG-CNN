"""
JacquardV2 Dataset Loader

Supports:
    - Depth-only, RGB-only, or depth+RGB (4-channel) inputs
    - Random rotation (0°, 90°, 180°, 270°)
    - Random zoom-out in [0.5, 1.0]
    - Train / validation / test split via (start, end) fractions
    - get_raw_grasps() for IoU evaluation
"""

from __future__ import annotations

import glob
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ggcnn2.datasets.utils import (
    DepthImage,
    GraspCollection,
    RGBImage,
)


class JacquardDataset(Dataset):
    """
    PyTorch Dataset for the JacquardV2 grasp detection dataset.

    Expected directory layout::

        <file_dir>/
            <category>/
                <scene_id>/
                    <scene_id>_grasps.txt
                    <scene_id>_RGB.png
                    <scene_id>_perfect_depth.tiff

    Args:
        file_dir: Root directory of the Jacquard dataset.
        include_depth: Whether to include depth channel.
        include_rgb: Whether to include RGB channels.
        start: Fractional start of this split (0.0 – 1.0).
        end: Fractional end of this split (0.0 – 1.0).
        ds_rotate: Rotate the file list before splitting (fraction 0.0 – 1.0).
        random_rotate: Apply random 90° rotation augmentation.
        random_zoom: Apply random zoom-out augmentation.
        output_size: Spatial size to resize images/maps to.
        load_from_npy: If True, load file list from *npy_path*.
        npy_path: Path to .npy file containing a pre-built grasp file list.
    """

    def __init__(
        self,
        file_dir: str,
        include_depth: bool = True,
        include_rgb: bool = False,
        start: float = 0.0,
        end: float = 1.0,
        ds_rotate: float = 0.0,
        random_rotate: bool = False,
        random_zoom: bool = False,
        output_size: int = 300,
        load_from_npy: bool = False,
        npy_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        if not include_depth and not include_rgb:
            raise ValueError("At least one of include_depth or include_rgb must be True.")

        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.output_size = output_size

        # Build or load file list
        if load_from_npy and npy_path is not None:
            grasp_files: list[str] = np.load(npy_path, allow_pickle=True).tolist()
        else:
            grasp_files = sorted(glob.glob(os.path.join(file_dir, "*", "*", "*_grasps.txt")))

        if len(grasp_files) == 0:
            raise FileNotFoundError(
                f"No Jacquard grasp files found under '{file_dir}'. "
                "Check that the path is correct and the dataset is extracted."
            )

        n = len(grasp_files)
        if ds_rotate:
            pivot = int(n * ds_rotate)
            grasp_files = grasp_files[pivot:] + grasp_files[:pivot]

        lo, hi = int(n * start), int(n * end)
        self._grasp_files = grasp_files[lo:hi]
        self._rgb_files = [f.replace("grasps.txt", "RGB.png") for f in self._grasp_files]
        self._depth_files = [
            f.replace("grasps.txt", "perfect_depth.tiff") for f in self._grasp_files
        ]

        if len(self._grasp_files) == 0:
            raise ValueError(
                f"Split [{start}, {end}) produced 0 samples from {n} total files."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        """Convert HxW or CxHxW numpy array to float32 tensor."""
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        return torch.from_numpy(arr.astype(np.float32))

    def _load_depth(self, idx: int, rot: float, zoom: float) -> np.ndarray:
        img = DepthImage.from_tiff(self._depth_files[idx])
        img.rotate(rot)
        img.normalize()
        img.zoom(zoom)
        img.crop_and_resize((self.output_size, self.output_size))
        return img.img

    def _load_rgb(self, idx: int, rot: float, zoom: float) -> np.ndarray:
        img = RGBImage.from_file(self._rgb_files[idx])
        img.rotate(rot)
        img.zoom(zoom)
        img.crop_and_resize((self.output_size, self.output_size))
        img.normalize()
        # Ensure channel-first: (H, W, 3) → (3, H, W)
        if img.img.ndim == 3 and img.img.shape[2] == 3:
            img.img = np.moveaxis(img.img, 2, 0)
        return img.img

    def _load_grasps(
        self, idx: int, rot: float, zoom: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = self.output_size / 1024.0
        grs = GraspCollection.load_from_jacquard(self._grasp_files[idx], scale=scale)
        c = self.output_size // 2
        grs.rotate(rot, (c, c))
        grs.zoom(zoom, (c, c))
        pos_map, angle_map, width_map = grs.generate_maps((self.output_size, self.output_size))
        return pos_map, angle_map, width_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_raw_grasps(
        self, idx: int, rot: float, zoom: float
    ) -> GraspCollection:
        """Return processed GraspCollection for IoU evaluation."""
        scale = self.output_size / 1024.0
        grs = GraspCollection.load_from_jacquard(self._grasp_files[idx], scale=scale)
        c = self.output_size // 2
        grs.rotate(rot, (c, c))
        grs.zoom(zoom, (c, c))
        return grs

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._grasp_files)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        int,
        float,
        float,
    ]:
        """
        Returns:
            x            : Input tensor (C, H, W) — depth (1ch), RGB (3ch), or RGBD (4ch).
            (pos, cos, sin, width): Ground-truth label tensors, each (1, H, W).
            idx          : Dataset index (for get_raw_grasps).
            rot          : Applied rotation in radians.
            zoom_factor  : Applied zoom factor.
        """
        # Augmentation parameters
        rot = random.choice([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]) if self.random_rotate else 0.0
        zoom_factor = float(np.random.uniform(0.5, 1.0)) if self.random_zoom else 1.0

        # Build input tensor
        if self.include_depth and self.include_rgb:
            depth = self._load_depth(idx, rot, zoom_factor)
            rgb = self._load_rgb(idx, rot, zoom_factor)
            x = self._to_tensor(np.concatenate([depth[np.newaxis], rgb], axis=0))
        elif self.include_depth:
            depth = self._load_depth(idx, rot, zoom_factor)
            x = self._to_tensor(depth)
        else:
            rgb = self._load_rgb(idx, rot, zoom_factor)
            x = self._to_tensor(rgb)

        # Build label tensors
        pos_map, angle_map, width_map = self._load_grasps(idx, rot, zoom_factor)

        cos_t = self._to_tensor(np.cos(2.0 * angle_map))
        sin_t = self._to_tensor(np.sin(2.0 * angle_map))
        pos_t = self._to_tensor(pos_map)
        # Clip width to [0, 150] and normalise to [0, 1]
        width_t = self._to_tensor(np.clip(width_map, 0.0, 150.0) / 150.0)

        return x, (pos_t, cos_t, sin_t, width_t), idx, rot, zoom_factor
