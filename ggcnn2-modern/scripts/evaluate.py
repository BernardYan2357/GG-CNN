#!/usr/bin/env python3
"""
GGCNN2 Evaluation Script

Evaluates a trained model on the test split of the Jacquard dataset and
prints accuracy, per-head losses, and optionally saves visualizations.

Usage::

    python scripts/evaluate.py \\
        --model trained_models/241201_1230/model_acc0.8500_epoch47.pth \\
        --dataset-path /data/jacquard \\
        --split-start 0.9 --split-end 1.0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ggcnn2.datasets.jacquard import JacquardDataset
from ggcnn2.models.ggcnn2 import GGCNN2
from ggcnn2.training.metrics import detect_grasps, max_iou, post_process
from ggcnn2.utils.io import load_checkpoint
from ggcnn2.utils.visualization import plot_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GGCNN2 on Jacquard test split")
    p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    p.add_argument("--dataset-path", default="./jacquard", help="Jacquard root directory")
    p.add_argument("--split-start", type=float, default=0.9, help="Dataset split start")
    p.add_argument("--split-end", type=float, default=1.0, help="Dataset split end")
    p.add_argument("--output-size", type=int, default=300, help="Image size")
    p.add_argument("--iou-threshold", type=float, default=0.25, help="IoU success threshold")
    p.add_argument("--num-vis", type=int, default=0,
                   help="Number of samples to visualise (0 = none)")
    p.add_argument("--vis-dir", type=str, default="eval_vis",
                   help="Directory to save visualizations")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logger = logging.getLogger("evaluate")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # Dataset
    dataset = JacquardDataset(
        args.dataset_path,
        include_depth=True,
        include_rgb=False,
        start=args.split_start,
        end=args.split_end,
        random_rotate=False,
        random_zoom=False,
        output_size=args.output_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    model = GGCNN2(input_channels=1)
    load_checkpoint(model, args.model, device)
    model = model.to(device)
    model.eval()

    correct, failed = 0, 0
    total_loss = 0.0

    if args.num_vis > 0:
        os.makedirs(args.vis_dir, exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for i, (x, y, idxs, rots, zooms) in enumerate(loader):
            xc = x.to(device)
            yc = [t.to(device) for t in y]

            loss_dict = model.compute_loss(xc, yc)
            total_loss += loss_dict["loss"].item()

            pred = loss_dict["pred"]
            q_map, ang_map, w_map = post_process(pred["pos"], pred["cos"], pred["sin"], pred["width"])
            grasp_preds = detect_grasps(q_map, ang_map, w_map)

            idx_i = int(idxs[0])
            grasps_true = dataset.get_raw_grasps(idx_i, float(rots[0]), float(zooms[0]))

            success = any(max_iou(gp, grasps_true) > args.iou_threshold for gp in grasp_preds)
            if success:
                correct += 1
            else:
                failed += 1

            # Visualise
            if vis_count < args.num_vis:
                depth_np = x[0, 0].cpu().numpy()
                fig = plot_output(
                    rgb=None,
                    depth=depth_np,
                    q_map=q_map,
                    angle_map=ang_map,
                    width_map=w_map,
                    pred_grasps=grasp_preds,
                    gt_collection=grasps_true,
                    save_path=os.path.join(args.vis_dir, f"sample_{i:04d}.png"),
                )
                import matplotlib.pyplot as plt
                plt.close(fig)
                vis_count += 1

    n = correct + failed
    acc = correct / n if n > 0 else 0.0
    avg_loss = total_loss / max(len(loader), 1)

    logger.info("Results on %d samples:", n)
    logger.info("  Correct:    %d", correct)
    logger.info("  Failed:     %d", failed)
    logger.info("  Accuracy:   %.4f  (%.1f%%)", acc, acc * 100)
    logger.info("  Avg Loss:   %.4f", avg_loss)


if __name__ == "__main__":
    main()
