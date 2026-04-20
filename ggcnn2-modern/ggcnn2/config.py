"""
Configuration management for GGCNN2.

Loads YAML-based configuration with optional CLI override and
environment variable support for dataset paths.

Usage::

    cfg = Config.from_yaml("configs/default.yaml")
    cfg.merge_cli_args({"training.lr": 1e-4})
    print(cfg.training.lr)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DataConfig:
    dataset_path: str = field(
        default_factory=lambda: os.environ.get("GGCNN2_DATASET_PATH", "./jacquard")
    )
    output_size: int = 300
    include_depth: bool = True
    include_rgb: bool = False
    train_split: float = 0.9
    random_rotate: bool = True
    random_zoom: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    input_channels: int = 1
    filter_sizes: list[int] = field(default_factory=lambda: [16, 16, 32, 16])
    l3_k_size: int = 5
    dilations: list[int] = field(default_factory=lambda: [2, 4])


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    batches_per_epoch: int = 100
    val_batches: int = 250
    lr: float = 1e-3
    lr_step: int = 20
    lr_gamma: float = 0.5
    iou_threshold: float = 0.25
    seed: int = 42
    multi_gpu: bool = False


@dataclass
class LoggingConfig:
    save_dir: str = "trained_models"
    log_level: str = "INFO"
    tensorboard: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        try:
            import yaml  # pyyaml
        except ImportError as exc:
            raise ImportError("pyyaml is required: pip install pyyaml") from exc

        with open(path) as f:
            raw: dict = yaml.safe_load(f) or {}

        cfg = cls()
        _apply_dict(cfg.data, raw.get("data", {}))
        _apply_dict(cfg.model, raw.get("model", {}))
        _apply_dict(cfg.training, raw.get("training", {}))
        _apply_dict(cfg.logging, raw.get("logging", {}))
        return cfg

    def merge_cli_args(self, overrides: dict[str, Any]) -> None:
        """
        Merge flat ``section.key=value`` overrides from CLI parsing.

        Example::

            cfg.merge_cli_args({"training.lr": 5e-4, "data.batch_size": 16})
        """
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            section_name, attr = parts[0], parts[1]
            section = getattr(self, section_name, None)
            if section is not None and hasattr(section, attr):
                setattr(section, attr, value)

    def to_dict(self) -> dict:
        """Serialise config to a plain dict."""
        import dataclasses
        return dataclasses.asdict(self)


def _apply_dict(target: Any, src: dict) -> None:
    for key, value in src.items():
        if hasattr(target, key):
            setattr(target, key, value)
