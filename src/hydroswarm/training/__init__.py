"""Governed CPU training, checkpoints, and scientific multitask objectives."""

from .artifacts import RunArtifacts
from .checkpoint import export_model, load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .data import (
    AgentTrajectory,
    CurriculumSchedule,
    CurriculumStage,
    GovernedScenarioDataset,
    ScenarioExample,
    TrajectoryStep,
    collate_scenarios,
    load_scenario_examples_jsonl,
    validate_split_isolation,
)
from .label_audit import audit_corpus, audit_split, cross_split_leakage
from .losses import MultiTaskLoss, compute_multitask_loss, task_gradient_norms
from .registry import ExperimentRegistry, RegistryError, RunHandle
from .sharded_data import ShardedScenarioDataset, write_shards
from .trainer import Trainer, TrainingSummary, set_deterministic_seed

__all__ = [
    "AgentTrajectory",
    "CurriculumSchedule",
    "CurriculumStage",
    "ExperimentRegistry",
    "GovernedScenarioDataset",
    "MultiTaskLoss",
    "RegistryError",
    "RunArtifacts",
    "RunHandle",
    "ScenarioExample",
    "ShardedScenarioDataset",
    "Trainer",
    "TrainingConfig",
    "TrainingSummary",
    "TrajectoryStep",
    "audit_corpus",
    "audit_split",
    "collate_scenarios",
    "compute_multitask_loss",
    "cross_split_leakage",
    "export_model",
    "load_checkpoint",
    "load_scenario_examples_jsonl",
    "save_checkpoint",
    "set_deterministic_seed",
    "task_gradient_norms",
    "validate_split_isolation",
    "write_shards",
]

