"""
Training loop for GGCNN2.

Features:
    - Multi-GPU support via DataParallel
    - Per-epoch train / validate functions
    - TensorBoard logging
    - Progress-bar via tqdm
    - Checkpoint saving (best model by validation accuracy)
    - Learning-rate scheduling
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc,assignment]

from ggcnn2.training.metrics import detect_grasps, max_iou, post_process

logger = logging.getLogger(__name__)


def train_epoch(
    epoch: int,
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    batches_per_epoch: int,
) -> dict[str, Any]:
    """
    Run one training epoch.

    Args:
        epoch:              Current epoch index (for logging).
        model:              The GGCNN2 model (already on *device*).
        device:             Training device.
        loader:             DataLoader for the training split.
        optimizer:          Optimiser instance.
        batches_per_epoch:  Stop after this many batches.

    Returns:
        dict with ``loss`` (float) and ``losses`` (per-head float dict).
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    results: dict[str, Any] = {"loss": 0.0, "losses": {}}
    model.train()
    batch_idx = 0

    iterator = iter(loader)
    if use_tqdm:
        iterator = tqdm(iterator, total=batches_per_epoch, desc=f"Train E{epoch}", leave=False)

    while batch_idx < batches_per_epoch:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        x, y, *_ = batch
        xc = x.to(device)
        yc = [t.to(device) for t in y]

        loss_dict = model.compute_loss(xc, yc)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        results["loss"] += loss.item()
        for name, val in loss_dict["losses"].items():
            results["losses"].setdefault(name, 0.0)
            results["losses"][name] += val.item()

        batch_idx += 1
        if batch_idx % 10 == 0:
            logger.info("Epoch %d  Batch %d/%d  Loss %.4f", epoch, batch_idx, batches_per_epoch, loss.item())

    if batch_idx > 0:
        results["loss"] /= batch_idx
        for k in results["losses"]:
            results["losses"][k] /= batch_idx

    return results


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    batches_per_epoch: int,
    iou_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Run one validation epoch.

    Args:
        model:              The GGCNN2 model.
        device:             Validation device.
        loader:             DataLoader for the validation split (batch_size=1).
        batches_per_epoch:  Stop after this many batches.
        iou_threshold:      IoU threshold to count a grasp as correct.

    Returns:
        dict with ``loss``, ``losses``, ``correct``, ``failed``, ``acc`` fields.
    """
    results: dict[str, Any] = {
        "loss": 0.0,
        "losses": {},
        "correct": 0,
        "failed": 0,
        "acc": 0.0,
    }
    model.eval()
    batch_idx = 0
    length = len(loader)

    for x, y, idxs, rots, zooms in loader:
        if batch_idx >= batches_per_epoch:
            break

        xc = x.to(device)
        yc = [t.to(device) for t in y]

        loss_dict = model.compute_loss(xc, yc)
        loss = loss_dict["loss"]

        results["loss"] += loss.item() / max(length, 1)
        for name, val in loss_dict["losses"].items():
            results["losses"].setdefault(name, 0.0)
            results["losses"][name] += val.item() / max(length, 1)

        pred = loss_dict["pred"]
        q_map, ang_map, w_map = post_process(pred["pos"], pred["cos"], pred["sin"], pred["width"])
        grasp_preds = detect_grasps(q_map, ang_map, w_map)

        # Evaluation per sample in batch (batch_size usually 1 for validation)
        for i in range(x.shape[0]):
            idx_i = int(idxs[i]) if hasattr(idxs[i], "item") else int(idxs[i])
            rot_i = float(rots[i])
            zoom_i = float(zooms[i])
            grasps_true = loader.dataset.get_raw_grasps(idx_i, rot_i, zoom_i)

            correct = any(max_iou(gp, grasps_true) > iou_threshold for gp in grasp_preds)
            if correct:
                results["correct"] += 1
            else:
                results["failed"] += 1

        batch_idx += 1

    total = results["correct"] + results["failed"]
    results["acc"] = results["correct"] / total if total > 0 else 0.0
    logger.info("Validation acc: %.4f  (%d/%d)", results["acc"], results["correct"], total)
    return results


class Trainer:
    """
    High-level training orchestrator for GGCNN2.

    Args:
        model:              The GGCNN2 (or wrapped DataParallel) model.
        device:             torch.device for training.
        train_loader:       DataLoader for training.
        val_loader:         DataLoader for validation.
        optimizer:          Optimiser instance.
        scheduler:          Optional LR scheduler.
        save_dir:           Directory to save checkpoints.
        batches_per_epoch:  Batches used per training epoch.
        val_batches:        Batches used per validation step.
        iou_threshold:      IoU threshold for grasp success.
        tb_writer:          Optional TensorBoard SummaryWriter.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        save_dir: str = "trained_models",
        batches_per_epoch: int = 100,
        val_batches: int = 250,
        iou_threshold: float = 0.25,
        tb_writer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.batches_per_epoch = batches_per_epoch
        self.val_batches = val_batches
        self.iou_threshold = iou_threshold
        self.tb = tb_writer
        self.best_acc = 0.0

        os.makedirs(save_dir, exist_ok=True)

    def train(self, epochs: int) -> None:
        """Run the full training loop for *epochs* epochs."""
        for epoch in range(epochs):
            t0 = time.time()
            train_res = train_epoch(
                epoch, self.model, self.device,
                self.train_loader, self.optimizer, self.batches_per_epoch,
            )
            logger.info(
                "Epoch %d  train_loss=%.4f  elapsed=%.1fs",
                epoch, train_res["loss"], time.time() - t0,
            )

            if self.tb is not None:
                self.tb.add_scalar("loss/train", train_res["loss"], epoch)
                for name, val in train_res["losses"].items():
                    self.tb.add_scalar(f"train_loss/{name}", val, epoch)

            val_res = validate_epoch(
                self.model, self.device,
                self.val_loader, self.val_batches,
                self.iou_threshold,
            )
            if self.tb is not None:
                self.tb.add_scalar("loss/val", val_res["loss"], epoch)
                self.tb.add_scalar("metrics/iou_acc", val_res["acc"], epoch)
                for name, val in val_res["losses"].items():
                    self.tb.add_scalar(f"val_loss/{name}", val, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if val_res["acc"] > self.best_acc:
                self.best_acc = val_res["acc"]
                ckpt_path = os.path.join(
                    self.save_dir,
                    f"model_acc{val_res['acc']:.4f}_epoch{epoch}.pth",
                )
                # Save underlying module if wrapped in DataParallel
                state = (
                    self.model.module.state_dict()
                    if isinstance(self.model, nn.DataParallel)
                    else self.model.state_dict()
                )
                torch.save(state, ckpt_path)
                logger.info("New best model saved: %s", ckpt_path)
