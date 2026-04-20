#!/usr/bin/env python3
"""
GGCNN2 Single-Image Prediction Script

Runs inference on a single depth image and visualises the predicted grasp.

Usage::

    python scripts/predict.py \\
        --model trained_models/model_best.pth \\
        --image /path/to/depth.tiff
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ggcnn2.datasets.utils import DepthImage
from ggcnn2.models.ggcnn2 import GGCNN2
from ggcnn2.training.metrics import detect_grasps, post_process
from ggcnn2.utils.io import load_checkpoint
from ggcnn2.utils.visualization import plot_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GGCNN2 inference on a single depth image")
    p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    p.add_argument("--image", required=True, help="Path to depth image (TIFF/PNG)")
    p.add_argument("--output-size", type=int, default=300, help="Image resize target")
    p.add_argument("--out", type=str, default="prediction.png", help="Output visualization path")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    logger = logging.getLogger("predict")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # Load and preprocess image
    img = DepthImage.from_tiff(args.image)
    img.normalize()
    img.crop_and_resize((args.output_size, args.output_size))

    x = torch.from_numpy(img.img[np.newaxis, np.newaxis]).float().to(device)

    # Model
    model = GGCNN2(input_channels=1)
    load_checkpoint(model, args.model, device)
    model = model.to(device).eval()

    with torch.no_grad():
        pos_pred, cos_pred, sin_pred, width_pred = model(x)

    q_map, ang_map, w_map = post_process(pos_pred, cos_pred, sin_pred, width_pred)
    grasp_preds = detect_grasps(q_map, ang_map, w_map)

    logger.info("Detected %d grasp candidate(s)", len(grasp_preds))
    for i, g in enumerate(grasp_preds):
        logger.info(
            "  Grasp %d: center=%s  angle=%.3f rad  width=%.1f px",
            i, tuple(g.center.astype(int).tolist()), g.angle, g.width,
        )

    fig = plot_output(
        rgb=None,
        depth=img.img,
        q_map=q_map,
        angle_map=ang_map,
        width_map=w_map,
        pred_grasps=grasp_preds,
        gt_collection=None,
        save_path=args.out,
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info("Visualization saved to: %s", args.out)


if __name__ == "__main__":
    main()
