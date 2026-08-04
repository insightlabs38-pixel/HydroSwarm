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
    TopologyMetadata,
    TrajectoryStep,
    collate_scenarios,
    load_scenario_examples_jsonl,
    manifest_entry,
    resolve_source_node_id,
    validate_split_isolation,
)
from .label_audit import audit_corpus, audit_split, cross_split_leakage
from .losses import MultiTaskLoss, compute_multitask_loss, task_gradient_norms
from .registry import ExperimentRegistry, RegistryError, RunHandle
from .sampler import GroupBalancedSampler, by_curriculum_stage, by_network, by_source_node, composite_key
from .sharded_data import ShardedScenarioDataset, write_shards
from .split_policy import SplitPolicyViolation, authorize_locked_final_test, load_policy
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
    "TopologyMetadata",
    "Trainer",
    "TrainingConfig",
    "TrainingSummary",
    "TrajectoryStep",
    "GroupBalancedSampler",
    "SplitPolicyViolation",
    "audit_corpus",
    "audit_split",
    "authorize_locked_final_test",
    "by_curriculum_stage",
    "by_network",
    "by_source_node",
    "collate_scenarios",
    "composite_key",
    "compute_multitask_loss",
    "cross_split_leakage",
    "export_model",
    "load_checkpoint",
    "load_policy",
    "load_scenario_examples_jsonl",
    "manifest_entry",
    "resolve_source_node_id",
    "save_checkpoint",
    "set_deterministic_seed",
    "task_gradient_norms",
    "validate_split_isolation",
    "write_shards",
]

