"""
Data augmentation helpers for GGCNN2.

These are thin wrappers kept for API completeness; augmentation is primarily
handled inside JacquardDataset.__getitem__ using the image/grasp utilities.
"""

from __future__ import annotations

import random

import numpy as np


def random_rotate_choice() -> float:
    """Return a random rotation in {0, π/2, π, 3π/2}."""
    return random.choice([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])


def random_zoom_factor(lo: float = 0.5, hi: float = 1.0) -> float:
    """Sample a random zoom factor from Uniform[lo, hi]."""
    return float(np.random.uniform(lo, hi))
