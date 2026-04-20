"""
Grasp and image processing utilities for dataset loaders.

Provides:
    - Grasp / GraspRectangle: single grasp representation
    - GraspCollection: multi-grasp container with image generation
    - DepthImage / RGBImage: image wrappers with augmentation helpers
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
from skimage.draw import polygon
from skimage.transform import resize, rotate as skimage_rotate


# ---------------------------------------------------------------------------
# Grasp geometry helpers
# ---------------------------------------------------------------------------

def _str_to_point(token: str) -> list[int]:
    """Parse 'x y' string to [row, col] integer list."""
    x, y = token.split()
    return [int(round(float(y))), int(round(float(x)))]


class GraspRectangle:
    """
    A single grasp described by four corner points.

    Points are stored in (row, col) order as a (4, 2) int array.
    """

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=float)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0).astype(int)

    @property
    def width(self) -> float:
        """Distance between edge 0-1 (gripper opening width)."""
        dp = self.points[1] - self.points[0]
        return float(np.sqrt((dp ** 2).sum()))

    @property
    def length(self) -> float:
        """Distance between edge 1-2 (gripper depth)."""
        dp = self.points[2] - self.points[1]
        return float(np.sqrt((dp ** 2).sum()))

    @property
    def angle(self) -> float:
        """Angle relative to positive x-axis in [-pi/2, pi/2]."""
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return float((np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2)

    # ------------------------------------------------------------------
    # Spatial transforms
    # ------------------------------------------------------------------

    def rotate(self, angle: float, center: tuple[float, float]) -> None:
        """Rotate points by *angle* (radians) around *center* (row, col)."""
        R = np.array([
            [np.cos(-angle),  np.sin(-angle)],
            [-np.sin(-angle), np.cos(-angle)],
        ])
        c = np.array(center).reshape(1, 2)
        self.points = (R @ (self.points - c).T).T + c

    def zoom(self, factor: float, center: tuple[float, float]) -> None:
        """Scale points by 1/factor around *center* (simulates zoom-out)."""
        T = np.array([[1 / factor, 0], [0, 1 / factor]])
        c = np.array(center).reshape(1, 2)
        self.points = (T @ (self.points - c).T).T + c

    def scale(self, factor: float) -> None:
        """Uniform scale from the origin."""
        self.points *= factor

    def offset(self, offset: tuple[float, float]) -> None:
        self.points += np.array(offset).reshape(1, 2)

    # ------------------------------------------------------------------
    # Polygon helpers
    # ------------------------------------------------------------------

    def polygon_coords(self, shape: Optional[tuple[int, int]] = None):
        """Return (rr, cc) arrays of pixels inside this rectangle."""
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape: tuple[int, int]):
        """Return (rr, cc) of the central third of the rectangle."""
        return GraspRectangleCPAW(
            self.center, self.angle, self.length, self.width / 3
        ).as_grasp_rectangle.polygon_coords(shape)


class GraspRectangleCPAW:
    """
    Grasp parameterised by center, angle, length, width.

    Provides :attr:`as_grasp_rectangle` to convert back to corner points.
    """

    def __init__(
        self,
        center: np.ndarray,
        angle: float,
        length: float = 30.0,
        width: float = 60.0,
    ) -> None:
        self.center = np.asarray(center, dtype=float)
        self.angle = float(angle)
        self.length = float(length)
        self.width = float(width)

    @property
    def as_grasp_rectangle(self) -> GraspRectangle:
        """Convert CPAW representation to four-corner GraspRectangle."""
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.width / 2 * yo
        x1 = self.center[1] - self.width / 2 * xo
        y2 = self.center[0] - self.width / 2 * yo
        x2 = self.center[1] + self.width / 2 * xo

        return GraspRectangle(np.array([
            [y1 - self.length / 2 * xo, x1 - self.length / 2 * yo],
            [y2 - self.length / 2 * xo, x2 - self.length / 2 * yo],
            [y2 + self.length / 2 * xo, x2 + self.length / 2 * yo],
            [y1 + self.length / 2 * xo, x1 + self.length / 2 * yo],
        ]))


class GraspCollection:
    """Container for multiple GraspRectangle objects belonging to one scene."""

    def __init__(self, grasps: Optional[list[GraspRectangle]] = None) -> None:
        self.grasps: list[GraspRectangle] = grasps if grasps is not None else []

    def __getattr__(self, attr: str):
        """Delegate unknown attributes to GraspRectangle if callable."""
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: [
                getattr(gr, attr)(*args, **kwargs) for gr in self.grasps
            ]
        raise AttributeError(f"GraspCollection has no attribute '{attr}'")

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def load_from_cornell(cls, filepath: str) -> "GraspCollection":
        """Load grasps from Cornell dataset annotation file."""
        rects: list[GraspRectangle] = []
        with open(filepath) as f:
            while True:
                p0 = f.readline().strip()
                if not p0:
                    break
                if p0[0] == "N":
                    break
                p1, p2, p3 = f.readline().strip(), f.readline().strip(), f.readline().strip()
                rects.append(GraspRectangle(np.array([
                    _str_to_point(p0),
                    _str_to_point(p1),
                    _str_to_point(p2),
                    _str_to_point(p3),
                ])))
        return cls(rects)

    @classmethod
    def load_from_jacquard(cls, filepath: str, scale: float = 1.0) -> "GraspCollection":
        """
        Load grasps from Jacquard dataset annotation file.

        Format per line: ``x;y;theta;w;h``
        """
        rects: list[GraspRectangle] = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                x, y, theta, w, h = (float(v) for v in line.split(";"))
                gr = GraspRectangleCPAW(
                    center=np.array([y, x]),
                    angle=-theta / 180.0 * np.pi,
                    length=h,
                    width=w,
                ).as_grasp_rectangle
                rects.append(gr)
        collection = cls(rects)
        collection.scale(scale)
        return collection

    # ------------------------------------------------------------------
    # Aggregate transforms
    # ------------------------------------------------------------------

    def rotate(self, angle: float, center: tuple[float, float]) -> None:
        for gr in self.grasps:
            gr.rotate(angle, center)

    def zoom(self, factor: float, center: tuple[float, float]) -> None:
        for gr in self.grasps:
            gr.zoom(factor, center)

    def scale(self, factor: float) -> None:
        for gr in self.grasps:
            gr.scale(factor)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def center(self) -> np.ndarray:
        pts = np.vstack([gr.points for gr in self.grasps])
        return pts.mean(axis=0).astype(int)

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------

    def generate_maps(
        self,
        shape: tuple[int, int] = (300, 300),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rasterise all grasps into dense maps.

        Returns:
            pos_map   : float32 (H, W) — grasp quality in [0, 1]
            angle_map : float32 (H, W) — angle in radians
            width_map : float32 (H, W) — gripper width in pixels
        """
        pos_map = np.zeros(shape, dtype=np.float32)
        angle_map = np.zeros(shape, dtype=np.float32)
        width_map = np.zeros(shape, dtype=np.float32)

        for gr in self.grasps:
            rr, cc = gr.compact_polygon_coords(shape)
            pos_map[rr, cc] = 1.0
            angle_map[rr, cc] = gr.angle
            width_map[rr, cc] = gr.width

        return pos_map, angle_map, width_map


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

class RGBImage:
    """Wrapper for an RGB image with augmentation helpers."""

    def __init__(self, img: np.ndarray) -> None:
        self.img = img.copy()

    @classmethod
    def from_file(cls, path: str) -> "RGBImage":
        from imageio.v3 import imread  # lazy import
        return cls(imread(path))

    def normalize(self) -> None:
        """Normalize to [0, 1] and subtract mean."""
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()

    def rotate(self, angle: float, center: Optional[tuple[int, int]] = None) -> None:
        """Rotate by *angle* radians."""
        if center is not None:
            center = (int(center[1]), int(center[0]))
        self.img = skimage_rotate(
            self.img, angle / np.pi * 180,
            center=center, mode="symmetric", preserve_range=True
        ).astype(self.img.dtype)

    def zoom(self, factor: float) -> None:
        """Centre-crop and resize to simulate zoom."""
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc:self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode="symmetric", preserve_range=True).astype(
            self.img.dtype
        )

    def crop_and_resize(self, size: tuple[int, int]) -> None:
        if self.img.shape[:2] != size:
            self.img = resize(self.img, size, preserve_range=True).astype(self.img.dtype)


class DepthImage:
    """Wrapper for a depth image with noise augmentation and normalization."""

    def __init__(self, img: np.ndarray) -> None:
        self.img = img.copy().astype(np.float32)

    @classmethod
    def from_tiff(cls, path: str) -> "DepthImage":
        from imageio.v3 import imread  # lazy import
        return cls(imread(path).astype(np.float32))

    def normalize(self) -> None:
        """
        Normalize by subtracting mean and clipping to [-1, 1].
        Adds small random offsets and noise to improve robustness.
        """
        self.img = np.clip(self.img - self.img.mean(), -1.0, 1.0)
        self.img += random.randint(-200, 200) / 1000.0
        self.img += np.random.normal(size=self.img.shape).astype(np.float32) / 200.0
        self.img += _gradient_2d(
            np.random.randint(0, 20),
            np.random.randint(0, 20),
            self.img.shape[0],
            self.img.shape[1],
            bool(np.random.randint(0, 2)),
        ) / 100.0

    def rotate(self, angle: float, center: Optional[tuple[int, int]] = None) -> None:
        if center is not None:
            center = (int(center[1]), int(center[0]))
        self.img = skimage_rotate(
            self.img, angle / np.pi * 180,
            center=center, mode="symmetric", preserve_range=True
        ).astype(self.img.dtype)

    def zoom(self, factor: float) -> None:
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc:self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode="symmetric", preserve_range=True).astype(
            self.img.dtype
        )

    def crop_and_resize(self, size: tuple[int, int]) -> None:
        if self.img.shape[:2] != size:
            self.img = resize(self.img, size, preserve_range=True).astype(self.img.dtype)


def _gradient_2d(
    start: float, stop: float, width: int, height: int, is_horizontal: bool
) -> np.ndarray:
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width, dtype=np.float32), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height, dtype=np.float32), (width, 1)).T
