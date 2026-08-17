"""Reproducible frozen-scenario and benchmark runners."""

from .benchmark import BenchmarkRunner
from .golden import GoldenScenarioRunner, freeze_golden_inputs
from .live_example import build_live_example_inputs
from .reference_demo import build_reference_incident_artifact, validate_reference_incident_artifact
from .scout_readiness import (
    M10_2_PREFLIGHT_BLOCKED,
    M10_2_READY_FOR_SCIENTIFIC_EVALUATION,
    M9_6_SCOUT_HEAD_AUDIT,
    ScoutHeadTrainingAudit,
    m10_2_readiness,
)
from .scout_state import (
    SCOUT_EVAL_STATE_SCHEMA_VERSION,
    LearnedScoutRecommendation,
    ScoutEvaluationState,
    ScoutStateLeakageError,
    apply_scout_candidate_mask,
    assert_finite_scout_outputs,
    assert_no_target_only_keys,
    build_scout_evaluation_state,
    decode_learned_scout_recommendation,
    select_candidate_node,
)

__all__ = [
    "BenchmarkRunner",
    "GoldenScenarioRunner",
    "build_live_example_inputs",
    "build_reference_incident_artifact",
    "validate_reference_incident_artifact",
    "freeze_golden_inputs",
    "SCOUT_EVAL_STATE_SCHEMA_VERSION",
    "LearnedScoutRecommendation",
    "ScoutEvaluationState",
    "ScoutStateLeakageError",
    "apply_scout_candidate_mask",
    "assert_finite_scout_outputs",
    "assert_no_target_only_keys",
    "build_scout_evaluation_state",
    "decode_learned_scout_recommendation",
    "select_candidate_node",
    "M10_2_PREFLIGHT_BLOCKED",
    "M10_2_READY_FOR_SCIENTIFIC_EVALUATION",
    "M9_6_SCOUT_HEAD_AUDIT",
    "ScoutHeadTrainingAudit",
    "m10_2_readiness",
]
