# GGCNN2 Modern

A production-ready implementation of **GGCNN2** (Generative Grasping CNN v2) built for:

- **Python 3.10+**
- **PyTorch 2.0+** (CUDA 12.x or CPU)
- **JacquardV2 dataset**
- **Windows + conda** ✓

---

## Table of Contents

1. [What is GGCNN2?](#what-is-ggcnn2)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Inference on a Single Image](#inference-on-a-single-image)
9. [Configuration Reference](#configuration-reference)
10. [Architecture Details](#architecture-details)
11. [Key Improvements Over the Original](#key-improvements-over-the-original)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)
14. [License](#license)

---

## What is GGCNN2?

GGCNN2 is an improved version of the Generative Grasping CNN (GGCNN), originally proposed in:

> **Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach**  
> Douglas Morrison, Peter Corke, Jürgen Leitner — RSS 2018

Given a **depth image**, GGCNN2 predicts four pixel-aligned maps in a single forward pass:

| Output head | Description |
|---|---|
| **Quality (pos)** | Grasp success probability per pixel |
| **cos(2θ)** | Cosine encoding of gripper angle |
| **sin(2θ)** | Sine encoding of gripper angle |
| **Width** | Normalised gripper opening width |

---

## Project Structure

```
ggcnn2-modern/
├── ggcnn2/
│   ├── __init__.py
│   ├── config.py                  # YAML + CLI configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── ggcnn2.py              # GGCNN2 architecture
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── jacquard.py            # JacquardV2 dataset loader
│   │   └── utils.py               # Grasp + image processing utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training / validation loops + Trainer class
│   │   ├── losses.py              # Combined MSE loss
│   │   └── metrics.py             # post_process, detect_grasps, IoU
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py       # Matplotlib grasp overlays
│       ├── io.py                  # Checkpoint helpers
│       └── augmentation.py        # Augmentation helpers
├── scripts/
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Evaluation on test split
│   ├── predict.py                 # Single-image inference
│   └── download_dataset.py        # Dataset download instructions
├── configs/
│   └── default.yaml               # Default hyperparameters
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_datasets.py
│   └── test_training.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

```bash
# 1. Clone + enter the project directory
cd ggcnn2-modern

# 2. Create and activate a conda environment
conda create -n ggcnn2 python=3.11 -y
conda activate ggcnn2

# 3. Install PyTorch (adjust for your CUDA version – see pytorch.org)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Run the test suite to verify the installation
pytest tests/ -v

# 6. Start training (after setting up the dataset – see below)
python scripts/train.py --config configs/default.yaml \
    --dataset-path /path/to/jacquard
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 |
| CUDA (optional) | ≥ 11.7 recommended |

### Conda (recommended for Windows + Linux)

```bash
conda create -n ggcnn2 python=3.11 -y
conda activate ggcnn2

# PyTorch with CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# OR: CPU-only PyTorch
conda install pytorch torchvision cpuonly -c pytorch -y

# Remaining dependencies
pip install -r requirements.txt
```

### Pip / venv

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate          # Windows PowerShell

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Dataset Setup

The JacquardV2 dataset must be downloaded manually.

```bash
python scripts/download_dataset.py
```

This prints step-by-step instructions. After extraction the layout must be:

```
jacquard/
  <category>/
    <scene_id>/
      <scene_id>_grasps.txt
      <scene_id>_RGB.png
      <scene_id>_perfect_depth.tiff
```

Set the path via:

```bash
# Environment variable (persisted)
export GGCNN2_DATASET_PATH=/path/to/jacquard   # Linux / macOS
set GGCNN2_DATASET_PATH=C:\path\to\jacquard    # Windows cmd

# Or pass at runtime
python scripts/train.py --dataset-path /path/to/jacquard
```

---

## Training

```bash
# Basic training with default config
python scripts/train.py --config configs/default.yaml

# Override common options
python scripts/train.py \
    --config configs/default.yaml \
    --dataset-path /data/jacquard \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --save-dir trained_models/run1

# Use RGB + depth (4-channel input)
python scripts/train.py --use-rgb

# Multi-GPU training
python scripts/train.py --multi-gpu

# CPU-only (for testing)
python scripts/train.py --cpu --epochs 2 --batches-per-epoch 5
```

**TensorBoard** logs are written to `<save_dir>/<timestamp>/tb/`:

```bash
tensorboard --logdir trained_models
```

Checkpoints are saved as `model_acc<val_acc>_epoch<n>.pth` whenever a new best validation accuracy is achieved.

---

## Evaluation

```bash
python scripts/evaluate.py \
    --model trained_models/241201_1230/model_acc0.8500_epoch47.pth \
    --dataset-path /data/jacquard \
    --split-start 0.9 \
    --split-end 1.0

# Save visualizations for first 20 samples
python scripts/evaluate.py \
    --model trained_models/model_best.pth \
    --num-vis 20 \
    --vis-dir eval_vis/
```

---

## Inference on a Single Image

```bash
python scripts/predict.py \
    --model trained_models/model_best.pth \
    --image /path/to/depth.tiff \
    --out prediction.png
```

The script prints detected grasp parameters and saves a visualisation PNG.

---

## Configuration Reference

All options can be set in `configs/default.yaml` and overridden via CLI flags.

### `data`

| Key | Default | Description |
|---|---|---|
| `dataset_path` | `./jacquard` | Root directory (env `GGCNN2_DATASET_PATH`) |
| `output_size` | `300` | Spatial size of input images |
| `include_depth` | `true` | Include depth channel |
| `include_rgb` | `false` | Include RGB channels |
| `train_split` | `0.9` | Fraction of data for training |
| `random_rotate` | `true` | 90° rotation augmentation |
| `random_zoom` | `true` | Zoom-out in [0.5, 1.0] |
| `num_workers` | `4` | DataLoader workers |

### `model`

| Key | Default | Description |
|---|---|---|
| `filter_sizes` | `[16, 16, 32, 16]` | Channel widths at each stage |
| `l3_k_size` | `5` | Kernel size for dilated convolutions |
| `dilations` | `[2, 4]` | Dilation rates |

### `training`

| Key | Default | Description |
|---|---|---|
| `epochs` | `100` | Total training epochs |
| `batch_size` | `8` | Training batch size |
| `batches_per_epoch` | `100` | Steps per epoch |
| `val_batches` | `250` | Steps per validation |
| `lr` | `0.001` | Initial learning rate (Adam) |
| `lr_step` | `20` | StepLR decay interval |
| `lr_gamma` | `0.5` | StepLR decay factor |
| `iou_threshold` | `0.25` | IoU threshold for success |
| `seed` | `42` | Random seed |
| `multi_gpu` | `false` | Enable DataParallel |

---

## Architecture Details

```
Input (1 × 300 × 300)
      │
  ┌───▼────────────────────────────────────────────────────────┐
  │  Block 1: Conv(11×11, 16) → ReLU → Conv(5×5, 16) → ReLU  │
  │           MaxPool(2×2) ──► (16 × 150 × 150)               │
  └───────────────────────────────────────────────────────────┘
      │
  ┌───▼────────────────────────────────────────────────────────┐
  │  Block 2: Conv(5×5, 16) → ReLU → Conv(5×5, 16) → ReLU    │
  │           MaxPool(2×2) ──► (16 × 75 × 75)                 │
  └───────────────────────────────────────────────────────────┘
      │
  ┌───▼────────────────────────────────────────────────────────┐
  │  Dilated conv (5×5, d=2, 32ch) → ReLU                     │
  │  Dilated conv (5×5, d=4, 32ch) → ReLU                     │
  └───────────────────────────────────────────────────────────┘
      │
  ┌───▼────────────────────────────────────────────────────────┐
  │  Bilinear Upsample ×2 + Conv(3×3, 16) → ReLU              │
  │  Bilinear Upsample ×2 + Conv(3×3, 16) → ReLU              │
  │  ──► (16 × 300 × 300)                                      │
  └───────────────────────────────────────────────────────────┘
      │
  ┌───▼──────────────────────────────────────────────────────────────────┐
  │  1×1 Conv → pos (1 ch)  │  cos (1 ch)  │  sin (1 ch)  │  width (1ch)│
  └───────────────────────────────────────────────────────────────────────┘
```

**Loss:** sum of per-head MSE losses on (pos, cos, sin, width).

---

## Key Improvements Over the Original

| Aspect | Original | This implementation |
|---|---|---|
| Python | 3.6 (EOL) | 3.10+ |
| PyTorch | 1.5.1 (2020) | 2.0+ |
| CUDA | 10.2 | 12.x (with CPU fallback) |
| Windows/conda | ✗ | ✓ |
| Type hints | None | Full PEP 526 annotations |
| Config | Hard-coded | YAML + CLI overrides |
| Multi-GPU | None | DataParallel |
| LR scheduling | None | StepLR |
| TensorBoard | TensorBoardX | Native `torch.utils.tensorboard` |
| Tests | None | pytest suite |
| Upsampling | TransposedConv | Bilinear (smoother) |

---

## Troubleshooting

### `FileNotFoundError: No Jacquard grasp files found`

- Check that the `--dataset-path` points to the correct root directory.
- Ensure files follow the naming pattern `*_grasps.txt`.

### CUDA out of memory

- Reduce `batch_size` in the config or via `--batch-size`.
- Reduce `output_size` (e.g. 224 instead of 300).

### `RuntimeError: Expected all tensors to be on the same device`

- Pass `--cpu` for CPU-only mode, or ensure CUDA is available.

### Windows: `num_workers > 0` causes a freeze

Add this to the top of `scripts/train.py`:

```python
import multiprocessing
multiprocessing.freeze_support()
```

Or set `num_workers: 0` in the config.

### ImportError: imageio / tifffile

```bash
pip install imageio[tifffile]
```

---

## API Reference

### `ggcnn2.models.ggcnn2.GGCNN2`

```python
GGCNN2(
    input_channels: int = 1,
    filter_sizes: list[int] | None = None,    # default [16, 16, 32, 16]
    l3_k_size: int = 5,
    dilations: list[int] | None = None,        # default [2, 4]
)
```

**`forward(x)`** → `(pos, cos, sin, width)` – each `(B, 1, H, W)`.

**`compute_loss(xc, yc)`** → `dict` with keys `loss`, `losses`, `pred`.

---

### `ggcnn2.datasets.jacquard.JacquardDataset`

```python
JacquardDataset(
    file_dir: str,
    include_depth: bool = True,
    include_rgb: bool = False,
    start: float = 0.0,
    end: float = 1.0,
    random_rotate: bool = False,
    random_zoom: bool = False,
    output_size: int = 300,
)
```

**`__getitem__(idx)`** → `(x, (pos, cos, sin, width), idx, rot, zoom_factor)`.

**`get_raw_grasps(idx, rot, zoom)`** → `GraspCollection` for IoU evaluation.

---

### `ggcnn2.training.metrics`

```python
post_process(pos, cos, sin, width) -> (q_map, angle_map, width_map)
detect_grasps(q_map, angle_map, width_map, num_grasps=1) -> list[GraspRectangleCPAW]
iou(pred: GraspRectangle, true: GraspRectangle) -> float
max_iou(pred_cpaw, true_collection) -> float
```

---

## License

This project is released under the **MIT License**.

The original GGCNN2 architecture is from:  
Morrison et al., "Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach", RSS 2018.
