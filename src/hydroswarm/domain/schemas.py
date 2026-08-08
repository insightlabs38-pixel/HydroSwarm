"""Machine-readable contracts at every operator and model boundary.

These schemas intentionally exclude arbitrary prose from safety-critical decisions.
Explanations are derived later from verified fields and provenance events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegative = Annotated[float, Field(ge=0.0)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionType(StrEnum):
    ISOLATE_ZONE = "ISOLATE_ZONE"
    CLOSE_PIPE = "CLOSE_PIPE"
    OPEN_PIPE = "OPEN_PIPE"
    FLUSH_NODE = "FLUSH_NODE"
    MONITOR_NODE = "MONITOR_NODE"
    COLLECT_SAMPLE = "COLLECT_SAMPLE"
    WAIT = "WAIT"
    END_PLAN = "END_PLAN"


class OODLevel(StrEnum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    OUTSIDE_VALIDATED_RANGE = "OUTSIDE_VALIDATED_RANGE"


class PlanDecision(StrEnum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    ABSTAINED = "ABSTAINED"
    PENDING_APPROVAL = "PENDING_APPROVAL"


class AbstentionReason(StrEnum):
    HIGH_DISAGREEMENT = "HIGH_DISAGREEMENT"
    POOR_SENSOR_HEALTH = "POOR_SENSOR_HEALTH"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    INVALID_NETWORK = "INVALID_NETWORK"
    NO_VALID_PLAN = "NO_VALID_PLAN"
    #: core-issues5.txt Section 9: generic fallback only -- reserved for a
    #: verification failure that does not originate from one of
    #: hydroswarm.simulation.wrapper's own distinct SimulationError
    #: subclasses (an unexpected, non-simulator exception). Any real
    #: SimulationError is categorized into one of the specific reasons
    #: below instead; this must never be the ONLY category a governed
    #: simulator failure is capable of producing.
    SIMULATION_FAILURE = "SIMULATION_FAILURE"
    #: hydroswarm.simulation.wrapper.SimulationTimeoutError.
    SIMULATION_TIMEOUT = "SIMULATION_TIMEOUT"
    #: hydroswarm.simulation.wrapper.SimulationIncompleteError.
    SIMULATION_INCOMPLETE = "SIMULATION_INCOMPLETE"
    #: hydroswarm.simulation.wrapper.SimulationUnstableError.
    SIMULATION_UNSTABLE = "SIMULATION_UNSTABLE"
    #: hydroswarm.simulation.wrapper.SimulationBudgetExceeded -- the
    #: incident's exact-simulation budget was exhausted mid-verification
    #: (possibly partway through a multi-hypothesis evaluation).
    SIMULATION_BUDGET_EXCEEDED = "SIMULATION_BUDGET_EXCEEDED"
    #: A real SimulationError occurred specifically while computing
    #: consequence metrics from otherwise-valid simulation results (e.g.
    #: SimulationIncompleteError raised inside calculate_consequences),
    #: distinct from a failure during the hydraulic/chemical simulation
    #: run itself.
    CONSEQUENCE_EVALUATION_FAILURE = "CONSEQUENCE_EVALUATION_FAILURE"
    AMBIGUOUS_EVIDENCE = "AMBIGUOUS_EVIDENCE"


class SensorObservation(FrozenModel):
    sensor_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    observed_at: datetime
    received_at: datetime
    concentration_mg_l: NonNegative | None = None
    pressure_m: float | None = None
    quality: Probability = 1.0
    missing: bool = False
    drift_flag: bool = False
    frozen_flag: bool = False

    @model_validator(mode="after")
    def validate_timing_and_missingness(self) -> SensorObservation:
        if self.received_at < self.observed_at:
            raise ValueError("received_at cannot precede observed_at")
        if self.missing and (self.concentration_mg_l is not None or self.pressure_m is not None):
            raise ValueError("missing observations cannot carry measurements")
        return self


class IncidentCreate(FrozenModel):
    network_id: str = Field(min_length=1)
    detected_at: datetime
    observations: tuple[SensorObservation, ...] = Field(min_length=1)
    contamination_threshold_mg_l: NonNegative = 0.001
    maximum_samples: int = Field(default=5, ge=1, le=5)


class CandidateSet(FrozenModel):
    node_probabilities: dict[str, Probability]
    node_ids: tuple[str, ...]
    coverage_target: Probability = 0.9
    measured_coverage: Probability | None = None
    calibrated: bool = False

    @model_validator(mode="after")
    def validate_probabilities(self) -> CandidateSet:
        total = sum(self.node_probabilities.values())
        if self.node_probabilities and abs(total - 1.0) > 1e-5:
            raise ValueError(f"node probabilities must sum to 1, got {total:.6f}")
        unknown = set(self.node_ids) - set(self.node_probabilities)
        if unknown:
            raise ValueError(f"candidate nodes missing probabilities: {sorted(unknown)}")
        return self


class OperationalAction(FrozenModel):
    action_type: ActionType
    target_id: str | None = None
    start_minute: int = Field(default=0, ge=0)
    duration_minutes: int = Field(default=0, ge=0, le=24 * 60)
    flow_rate_lps: NonNegative | None = None

    @model_validator(mode="after")
    def validate_target(self) -> OperationalAction:
        targetless = {ActionType.WAIT, ActionType.END_PLAN}
        if self.action_type not in targetless and not self.target_id:
            raise ValueError(f"{self.action_type} requires target_id")
        if self.action_type == ActionType.FLUSH_NODE and not self.flow_rate_lps:
            raise ValueError("FLUSH_NODE requires a positive flow_rate_lps")
        return self


class OperationalPlan(FrozenModel):
    plan_id: UUID = Field(default_factory=uuid4)
    incident_id: UUID
    name: str = Field(min_length=1, max_length=80)
    actions: tuple[OperationalAction, ...] = Field(min_length=1, max_length=32)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_version: str = Field(min_length=1)
    requires_operator_approval: Literal[True] = True


class ConsequenceMetrics(FrozenModel):
    population_impacted: NonNegative = 0.0
    contaminant_mass_consumed_mg: NonNegative = 0.0
    volume_above_threshold_l: NonNegative = 0.0
    contaminated_pipe_extent_m: NonNegative = 0.0
    minimum_pressure_m: float
    pressure_violation_minutes: NonNegative = 0.0
    unserved_demand_l: NonNegative = 0.0
    service_availability: Probability = 1.0
    operation_count: int = Field(ge=0)
    containment_time_minutes: NonNegative | None = None
    #: important-issues.txt requirement 12: False whenever the
    #: contamination-exposure fields above (population_impacted,
    #: contaminant_mass_consumed_mg, volume_above_threshold_l,
    #: contaminated_pipe_extent_m, containment_time_minutes) are Pydantic
    #: defaults rather than measured from a real chemical-transport
    #: simulation -- e.g. the legacy hydraulic-only evaluate_plan() path
    #: used when no incident evaluation context is available. Callers must
    #: never present these fields as measured exposure when this is False.
    exposure_evaluated: bool = False


class PlanVerification(FrozenModel):
    plan_id: UUID
    decision: PlanDecision
    simulator: str
    simulator_version: str
    state_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    consequences: ConsequenceMetrics | None = None
    #: important-issues.txt requirement 7: populated only when this plan was
    #: evaluated against a bounded multi-hypothesis source set (runtime,
    #: no ground truth). `consequences` above is the posterior-weighted
    #: expected outcome in that case; this is the worst-case outcome across
    #: the same evaluated hypotheses. None in single-profile (training) or
    #: hydraulic-only (legacy) evaluation.
    worst_case_consequences: ConsequenceMetrics | None = None
    #: important-issues.txt requirements 5/9/10: explicit evaluation
    #: assumptions and provenance -- aggregation policy, evaluated
    #: hypotheses/profile identity, contamination threshold,
    #: population-map identity, consequence-policy version, and the actual
    #: number of exact EPANET simulations this one verification consumed
    #: (which may exceed 1 for a multi-hypothesis evaluation and must not
    #: be silently collapsed to "one simulator call").
    evaluation_provenance: dict[str, Any] | None = None
    rejection_codes: tuple[str, ...] = ()
    abstention_reason: AbstentionReason | None = None
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: core-issues5.txt Section 10 (P0 safety fix): one canonical hash over
    #: every behavior-critical piece of context this verification was
    #: computed against (incident evidence, source hypothesis-set identity/
    #: probabilities, calibration/model/network/signature/normalization
    #: identity, consequence-policy version) -- set by the API layer (which
    #: has access to that context; PlanVerifier itself does not), never by
    #: PlanVerifier. None only for verifications predating this field.
    context_hash: str | None = None
    #: CURRENT until a behavior-critical context component changes (new
    #: sample, re-analysis, recalibration, ...), at which point this plan's
    #: PRIOR verification is marked STALE / SUPERSEDED here -- retained in
    #: audit history, never deleted, but no longer approvable
    #: (approve_plan requires decision == VERIFIED AND status == "CURRENT"
    #: AND context_hash == the incident's current context hash).
    verification_status: Literal["CURRENT", "STALE"] = "CURRENT"

    @model_validator(mode="after")
    def validate_decision_evidence(self) -> PlanVerification:
        if self.decision == PlanDecision.VERIFIED and self.consequences is None:
            raise ValueError("verified plans require consequence metrics")
        if self.decision == PlanDecision.REJECTED and not self.rejection_codes:
            raise ValueError("rejected plans require rejection codes")
        if self.decision == PlanDecision.ABSTAINED and self.abstention_reason is None:
            raise ValueError("abstained plans require an abstention reason")
        return self


class IncidentState(FrozenModel):
    incident_id: UUID = Field(default_factory=uuid4)
    network_id: str
    status: Literal["DETECTED", "ANALYZING", "SAMPLING", "PLANNING", "APPROVAL", "CLOSED"]
    observations: tuple[SensorObservation, ...]
    candidates: CandidateSet | None = None
    disagreement_js: Probability | None = None
    ood_level: OODLevel = OODLevel.NORMAL
    sample_count: int = Field(default=0, ge=0, le=5)
    approval_pending: bool = False
    #: core-issues.txt: cumulative real WNTR/EPANET exact-simulation runs
    #: spent by this incident across its whole lifetime. Persisted here
    #: (not left as a HydraulicSimulator instance attribute, which resets
    #: to zero every time a fresh simulator is constructed for a new
    #: request) so a caller cannot bypass the budget simply by triggering
    #: another verify/plan request against the same incident.
    exact_simulations_used: int = Field(default=0, ge=0)
    #: core-issues5.txt Section 8: "one plan verification" and "one EPANET
    #: simulation" are not equivalent under multi-hypothesis exposure-aware
    #: verification -- exact_simulations_used above is the raw EPANET
    #: execution count; this is the distinct count of PLANS that received a
    #: real exact-verification attempt (VERIFIED, REJECTED, or ABSTAINED),
    #: regardless of how many underlying hypotheses/executions each one
    #: consumed.
    plans_exactly_verified: int = Field(default=0, ge=0)
    #: core-issues5.txt Section 8: cumulative count of verify() calls this
    #: incident satisfied entirely from cache (PlanExposureEvaluation.
    #: cache_hit=True) -- these consume zero new EPANET executions and
    #: must be visible separately from exact_simulations_used, not merged
    #: into it or silently dropped.
    exact_simulation_cache_hits: int = Field(default=0, ge=0)
    #: core-issues5.txt Section 8: settings.exact_plan_simulation_limit -
    #: exact_simulations_used, recomputed and persisted by the API layer on
    #: every verification, so a caller reading IncidentState directly (GET
    #: /api/incidents/{id}) always sees the current budget without needing
    #: to know the configured limit itself or recompute the subtraction.
    remaining_epanet_budget: int = Field(default=0, ge=0)

