#!/usr/bin/env python3
"""
GGCNN2 Training Script

Usage::

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml \\
        --dataset-path /data/jacquard \\
        --epochs 50 --lr 0.001 --batch-size 8
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Allow running from the project root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ggcnn2.config import Config
from ggcnn2.datasets.jacquard import JacquardDataset
from ggcnn2.models.ggcnn2 import GGCNN2
from ggcnn2.training.trainer import Trainer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc,assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GGCNN2 on JacquardV2 dataset")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Override dataset path")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--batches-per-epoch", type=int, default=None,
                        help="Override batches per epoch")
    parser.add_argument("--val-batches", type=int, default=None,
                        help="Override validation batches")
    parser.add_argument("--no-depth", action="store_true",
                        help="Exclude depth input")
    parser.add_argument("--use-rgb", action="store_true",
                        help="Include RGB input")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override model save directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use all available GPUs via DataParallel")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    if os.path.isfile(args.config):
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()

    # CLI overrides
    if args.dataset_path:
        cfg.data.dataset_path = args.dataset_path
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.batches_per_epoch is not None:
        cfg.training.batches_per_epoch = args.batches_per_epoch
    if args.val_batches is not None:
        cfg.training.val_batches = args.val_batches
    if args.no_depth:
        cfg.data.include_depth = False
    if args.use_rgb:
        cfg.data.include_rgb = True
    if args.save_dir:
        cfg.logging.save_dir = args.save_dir
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.multi_gpu:
        cfg.training.multi_gpu = True

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level = getattr(logging, cfg.logging.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    set_seed(cfg.training.seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading Jacquard dataset from: %s", cfg.data.dataset_path)
    input_channels = int(cfg.data.include_depth) + 3 * int(cfg.data.include_rgb)
    cfg.model.input_channels = input_channels

    train_dataset = JacquardDataset(
        cfg.data.dataset_path,
        include_depth=cfg.data.include_depth,
        include_rgb=cfg.data.include_rgb,
        start=0.0,
        end=cfg.data.train_split,
        random_rotate=cfg.data.random_rotate,
        random_zoom=cfg.data.random_zoom,
        output_size=cfg.data.output_size,
    )
    val_dataset = JacquardDataset(
        cfg.data.dataset_path,
        include_depth=cfg.data.include_depth,
        include_rgb=cfg.data.include_rgb,
        start=cfg.data.train_split,
        end=1.0,
        random_rotate=False,
        random_zoom=False,
        output_size=cfg.data.output_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and (device.type == "cuda"),
    )

    logger.info("Train samples: %d  Val samples: %d", len(train_dataset), len(val_dataset))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = GGCNN2(
        input_channels=cfg.model.input_channels,
        filter_sizes=cfg.model.filter_sizes,
        l3_k_size=cfg.model.l3_k_size,
        dilations=cfg.model.dilations,
    )

    if cfg.training.multi_gpu and torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", total_params)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = StepLR(optimizer, step_size=cfg.training.lr_step, gamma=cfg.training.lr_gamma)

    # ------------------------------------------------------------------
    # Save folder
    # ------------------------------------------------------------------
    ts = datetime.datetime.now().strftime("%y%m%d_%H%M")
    save_dir = os.path.join(cfg.logging.save_dir, ts)
    os.makedirs(save_dir, exist_ok=True)

    # Persist config
    import json
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    tb_writer = None
    if cfg.logging.tensorboard and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        batches_per_epoch=cfg.training.batches_per_epoch,
        val_batches=cfg.training.val_batches,
        iou_threshold=cfg.training.iou_threshold,
        tb_writer=tb_writer,
    )

    trainer.train(cfg.training.epochs)

    if tb_writer is not None:
        tb_writer.close()

    logger.info("Training complete. Best validation acc: %.4f", trainer.best_acc)


if __name__ == "__main__":
    main()
