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
    #: core-issues3.txt Phase 15 item 3/8: v4-only advisory outputs, gated
    #: by HybridInferencePipeline's own runtime_enabled_outputs (granular,
    #: not the coarse trained_tasks role check the fields above still use)
    #: -- None whenever the checkpoint has not declared the corresponding
    #: output runtime-enabled, exactly like the fields above default to
    #: None for a checkpoint with no matching head at all. Never
    #: authoritative: the deterministic controller/OOD/calibration
    #: machinery elsewhere in this pipeline does not read these.
    event_presence: bool | None = None
    event_presence_probability: float | None = None
    #: None both when the head is absent/unvalidated AND when the
    #: predicted class itself is a currently-unsupported EventCause member
    #: (AMBIGUOUS/HYDRAULIC_MISMATCH -- core-issues3.txt Phase 6.5/9.3:
    #: "do not runtime-enable unsupported classes") -- the two cases are
    #: deliberately indistinguishable to a caller, since both mean "no
    #: trustworthy value available", not "false"/"zero".
    event_cause: str | None = None
    next_step: str | None = None
    #: core-issues5.txt Section 18.1: the learned 11-category OOD taxonomy
    #: (hydroswarm.training.ood_categories.OODCategory), advisory only --
    #: DO NOT confuse with `IncidentAnalysisResult.ood_level`, the
    #: deterministic 3-level severity, which remains authoritative
    #: regardless of this field (see hydroswarm.inference.authority.
    #: ood_certificate). None whenever the head is absent, unpromoted
    #: (gated by runtime_enabled_outputs, like event_cause above), or the
    #: predicted class is one of the four categories with no real training
    #: examples yet (ood_labels.UNSUPPORTED_OOD_CATEGORIES) -- same
    #: "no trustworthy value" convention as event_cause.
    ood_category: str | None = None


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
