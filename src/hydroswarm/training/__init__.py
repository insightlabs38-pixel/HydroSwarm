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
from .ood_categories import (
    OOD_CATEGORY_BEHAVIOR,
    AbstentionOutcome,
    ExpectedBehavior,
    OODCategory,
    classify_abstention_outcome,
    topology_calibration_is_valid,
)
from .permutation import EquivarianceReport, measure_equivariance, permute_example
from .registry import ExperimentRegistry, RegistryError, RunHandle
from .sampler import GroupBalancedSampler, by_curriculum_stage, by_network, by_source_node, composite_key
from .sharded_data import ShardedScenarioDataset, write_shards
from .split_policy import SplitPolicyViolation, authorize_locked_final_test, load_policy
from .targets_v2 import (
    TARGETS_BY_CATEGORY,
    TARGETS_V2,
    TARGETS_V2_SCHEMA_VERSION,
    EventCause,
    NextStep,
    TargetSchemaError,
    TargetSpec,
    check_schema_version,
    validate_targets_v2,
)
from .trainer import Trainer, TrainingSummary, set_deterministic_seed
from .trajectory_v2 import (
    TRAJECTORY_SCHEMA_VERSION,
    FullTrajectory,
    RemainingBudgets,
    TrajectoryIntegrityError,
    TrajectoryState,
)
from .variable_collate import collate_variable_topology

__all__ = [
    "OOD_CATEGORY_BEHAVIOR",
    "TARGETS_BY_CATEGORY",
    "TARGETS_V2",
    "TARGETS_V2_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "AbstentionOutcome",
    "AgentTrajectory",
    "CurriculumSchedule",
    "CurriculumStage",
    "EventCause",
    "ExpectedBehavior",
    "ExperimentRegistry",
    "FullTrajectory",
    "GovernedScenarioDataset",
    "MultiTaskLoss",
    "NextStep",
    "OODCategory",
    "RegistryError",
    "RemainingBudgets",
    "RunArtifacts",
    "RunHandle",
    "ScenarioExample",
    "ShardedScenarioDataset",
    "TargetSchemaError",
    "TargetSpec",
    "TopologyMetadata",
    "Trainer",
    "TrainingConfig",
    "TrainingSummary",
    "TrajectoryIntegrityError",
    "TrajectoryState",
    "TrajectoryStep",
    "EquivarianceReport",
    "GroupBalancedSampler",
    "SplitPolicyViolation",
    "audit_corpus",
    "audit_split",
    "authorize_locked_final_test",
    "by_curriculum_stage",
    "by_network",
    "by_source_node",
    "check_schema_version",
    "classify_abstention_outcome",
    "collate_scenarios",
    "collate_variable_topology",
    "composite_key",
    "compute_multitask_loss",
    "cross_split_leakage",
    "export_model",
    "load_checkpoint",
    "load_policy",
    "load_scenario_examples_jsonl",
    "manifest_entry",
    "measure_equivariance",
    "permute_example",
    "resolve_source_node_id",
    "save_checkpoint",
    "set_deterministic_seed",
    "task_gradient_norms",
    "topology_calibration_is_valid",
    "validate_split_isolation",
    "validate_targets_v2",
    "write_shards",
]

