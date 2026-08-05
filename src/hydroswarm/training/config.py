"""Strict CPU-first training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 31874
    epochs: int = 20
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 10
    scheduler: str = "cosine"
    checkpoint_every_epochs: int = 1
    early_stopping_patience: int = 5
    minimum_delta: float = 0.0
    maximum_runtime_seconds: float = 8 * 60 * 60
    minimum_free_disk_gb: float = 1.0
    num_workers: int = 0
    device: str = "cpu"
    fp32: bool = True
    deterministic: bool = True
    gradnorm_logging: bool = True
    #: core-issues.txt repair item 11: task_gradient_norms runs one extra
    #: torch.autograd.grad call per task loss (7-9 tasks) -- effectively
    #: 7-9 extra backward passes -- every single batch by default. Set
    #: above 1 to only compute it every Nth batch (still gives a usable
    #: trend signal over an epoch at a fraction of the cost); default of 1
    #: preserves the exact prior per-batch behavior for anything that does
    #: not opt in.
    gradnorm_log_every_n_batches: int = 1
    pcgrad_enabled: bool = False
    profile_ordinal_weight: float = 0.0
    task_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive_ints = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "checkpoint_every_epochs": self.checkpoint_every_epochs,
            "gradnorm_log_every_n_batches": self.gradnorm_log_every_n_batches,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers not in (0, 1):
            raise ValueError("CPU training supports zero or one data-loader worker")
        if self.device != "cpu":
            raise ValueError("this governed training foundation is CPU-only")
        if not self.fp32:
            raise ValueError("FP32 must remain enabled for the baseline trainer")
        if self.scheduler not in {"cosine", "linear", "constant"}:
            raise ValueError("scheduler must be cosine, linear, or constant")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "maximum_runtime_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.weight_decay < 0
            or self.minimum_free_disk_gb < 0
            or self.profile_ordinal_weight < 0
        ):
            raise ValueError("weight decay, disk threshold, and ordinal weight cannot be negative")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if any(value < 0 for value in self.task_weights.values()):
            raise ValueError("task weights must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if set(values) == {"training"}:
            values = values["training"]
        if not isinstance(values, dict):
            raise ValueError("training configuration must be a mapping")
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown training settings: {sorted(unknown)}")
        return cls(**values)
