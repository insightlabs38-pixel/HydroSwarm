"""Canonical hybrid-analysis result and replayable history records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping
from uuid import UUID

from hydroswarm.classical import ClassicalLocalizationResult, EstimatedHydraulicState
from hydroswarm.domain import OODLevel
from hydroswarm.inference.fusion import ControlAction, FusionDiagnostics, TrustFeatures
from hydroswarm.planning import PlanProposal
from hydroswarm.sampling import ActiveSamplingResult

from .ood import OODComponents


class HybridRuntimeMode(StrEnum):
    FULL_HYBRID = "FULL_HYBRID"
    CLASSICAL_SAFE = "CLASSICAL_SAFE"


@dataclass(frozen=True, slots=True)
class SemanticPredictions:
    evidence_sufficiency: float | None = None
    uncertainty: float | None = None
    expected_information_gain: Mapping[str, float] | None = None
    sensor_fault_probability: Mapping[str, float] | None = None
    plan_values: tuple[float, ...] = ()
    plan_validity: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PosteriorSnapshot:
    round_index: int
    observation_count: int
    fused_belief: Mapping[str, float]
    candidate_nodes: tuple[str, ...]
    entropy_bits: float
    evidence_hash: str


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    round_index: int
    observation_count: int
    valid_concentration_count: int
    sensor_nodes: tuple[str, ...]
    evidence_hash: str


@dataclass(frozen=True, slots=True)
class EvidenceChange:
    from_round: int
    to_round: int
    previous_candidates: tuple[str, ...]
    current_candidates: tuple[str, ...]
    removed_candidates: tuple[str, ...]
    added_candidates: tuple[str, ...]
    candidate_contraction: int
    top_probability_change: float


@dataclass(frozen=True, slots=True)
class IncidentAnalysisResult:
    incident_id: UUID
    node_alignment: tuple[str, ...]
    classical_belief: Mapping[str, float]
    neural_belief: Mapping[str, float] | None
    fused_belief: Mapping[str, float]
    classical_localization: ClassicalLocalizationResult
    estimated_hydraulic_state: EstimatedHydraulicState
    trust_features: TrustFeatures
    fusion_diagnostics: FusionDiagnostics | None
    trust_rationale: str
    conformal_candidate_nodes: tuple[str, ...]
    calibrated: bool
    calibration_alpha: float | None
    ood_components: OODComponents
    ood_level: OODLevel
    evidence_sufficient: bool
    planning_allowed: bool
    planning_suppression_reasons: tuple[str, ...]
    control_action: ControlAction
    sample_result: ActiveSamplingResult | None
    plan_proposals: tuple[PlanProposal, ...]
    semantic_predictions: SemanticPredictions
    posterior_history: tuple[PosteriorSnapshot, ...]
    evidence_history: tuple[EvidenceSnapshot, ...]
    comparison_history: tuple[EvidenceChange, ...]
    before_after: EvidenceChange | None
    runtime_mode: HybridRuntimeMode
    neural_failure: str | None
    latencies_ms: Mapping[str, float]
    provenance_hashes: Mapping[str, str]
    evidence_hash: str
