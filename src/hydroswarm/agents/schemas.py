"""Typed contracts for deterministic HydroSwarm specialist agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hydroswarm.domain import OODLevel, OperationalPlan, PlanVerification


Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateScore(AgentModel):
    node_id: str = Field(min_length=1)
    probability: Probability


class SentinelOutput(AgentModel):
    top_candidates: tuple[CandidateScore, ...] = Field(min_length=1)
    candidate_region: tuple[str, ...] = Field(min_length=1)
    target_coverage: Probability = 0.90
    start_time_bin: str = "unknown"
    duration_bin: str = "unknown"
    relative_strength: Literal["low", "medium", "high", "unknown"] = "unknown"
    evidence_sufficient: bool
    sensor_fault_nodes: tuple[str, ...] = ()
    uncertainty: Probability
    ood_level: OODLevel = OODLevel.NORMAL

    @model_validator(mode="after")
    def validate_candidates(self) -> SentinelOutput:
        nodes = [item.node_id for item in self.top_candidates]
        if len(set(nodes)) != len(nodes):
            raise ValueError("top candidate node IDs must be unique")
        probability_mass = sum(item.probability for item in self.top_candidates)
        if probability_mass <= 0.0:
            raise ValueError("top candidates must have positive probability mass")
        if probability_mass > 1.0 + 1e-6:
            raise ValueError("top candidate probabilities cannot exceed one")
        return self


class ScoutAction(StrEnum):
    SAMPLE = "SAMPLE"
    STOP = "STOP"


class ScoutOutput(AgentModel):
    action: ScoutAction
    node_id: str | None = None
    expected_information_gain: Annotated[float, Field(ge=0.0)] = 0.0
    expected_candidate_reduction: Probability = 0.0
    estimated_delay_minutes: Annotated[float, Field(ge=0.0)] = 0.0
    estimated_cost: Annotated[float, Field(ge=0.0)] = 0.0
    alternatives: tuple[str, ...] = ()
    reason: str = ""

    @model_validator(mode="after")
    def validate_sample(self) -> ScoutOutput:
        if self.action == ScoutAction.SAMPLE and not self.node_id:
            raise ValueError("SAMPLE requires node_id")
        if self.action == ScoutAction.STOP and self.node_id is not None:
            raise ValueError("STOP cannot include node_id")
        return self


class PlanCandidate(AgentModel):
    plan: OperationalPlan
    estimated_value: Probability = 0.0
    template: str = "deterministic"


class StrategistOutput(AgentModel):
    plans: tuple[PlanCandidate, ...] = Field(min_length=1, max_length=8)
    revision_round: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def unique_actions(self) -> StrategistOutput:
        signatures = [
            tuple(
                (
                    action.action_type.value,
                    action.target_id,
                    action.start_minute,
                    action.duration_minutes,
                    action.flow_rate_lps,
                )
                for action in candidate.plan.actions
            )
            for candidate in self.plans
        ]
        if len(set(signatures)) != len(signatures):
            raise ValueError("strategist plans must have distinct action sequences")
        return self


class FallbackMode(StrEnum):
    FULL_HYBRID = "FULL_HYBRID"
    DEGRADED_HYBRID = "DEGRADED_HYBRID"
    CLASSICAL_SAFE = "CLASSICAL_SAFE"
    SIMULATION_ONLY = "SIMULATION_ONLY"


class FSMState(StrEnum):
    IDLE = "IDLE"
    NETWORK_LOADED = "NETWORK_LOADED"
    NETWORK_VALIDATED = "NETWORK_VALIDATED"
    INCIDENT_LOADED = "INCIDENT_LOADED"
    DATA_QUALITY_CHECK = "DATA_QUALITY_CHECK"
    HYDRAULIC_STATE_ESTIMATION = "HYDRAULIC_STATE_ESTIMATION"
    SOURCE_LOCALIZATION = "SOURCE_LOCALIZATION"
    EVIDENCE_CHECK = "EVIDENCE_CHECK"
    SAMPLE_SELECTION = "SAMPLE_SELECTION"
    SOURCE_UPDATE = "SOURCE_UPDATE"
    PLAN_GENERATION = "PLAN_GENERATION"
    CONSTRAINT_CHECK = "CONSTRAINT_CHECK"
    WNTR_VERIFY = "WNTR_VERIFY"
    REVISE_PLANS = "REVISE_PLANS"
    PLAN_COMPARISON = "PLAN_COMPARISON"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    SIMULATE_RESPONSE = "SIMULATE_RESPONSE"
    COMPLETE = "COMPLETE"


class ToolPermission(StrEnum):
    READ_OBSERVATIONS = "READ_OBSERVATIONS"
    READ_HYDRAULICS = "READ_HYDRAULICS"
    READ_CANDIDATES = "READ_CANDIDATES"
    READ_SAMPLING_HISTORY = "READ_SAMPLING_HISTORY"
    READ_VERIFIER_FEEDBACK = "READ_VERIFIER_FEEDBACK"
    RUN_INFERENCE = "RUN_INFERENCE"
    RECOMMEND_SAMPLE = "RECOMMEND_SAMPLE"
    GENERATE_PLANS = "GENERATE_PLANS"


class SwarmEvent(AgentModel):
    sequence: int = Field(ge=0)
    from_state: FSMState
    to_state: FSMState
    event_type: str
    payload: dict[str, object] = Field(default_factory=dict)
    previous_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class SwarmResult(AgentModel):
    state: FSMState
    fallback_mode: FallbackMode
    awaiting_sample: ScoutOutput | None = None
    selected_plan: OperationalPlan | None = None
    verification: PlanVerification | None = None
    termination_reason: str | None = None
    events: tuple[SwarmEvent, ...]
    run_key: str
    #: core-issues5.txt Section 8: how many distinct plans this run actually
    #: submitted for exact verification (WNTR_VERIFY state), regardless of
    #: decision -- separate from `verification` above, which only ever
    #: carries the single SELECTED plan's outcome. Callers that need the
    #: underlying EPANET execution count (which may exceed this, under
    #: multi-hypothesis exposure-aware verification) read it from the
    #: HydraulicSimulator/IncidentState budget fields instead, not from
    #: this count.
    plans_exactly_verified: int = 0


class ReplayResult(AgentModel):
    state: FSMState
    event_count: int
    final_hash: str
    trajectory: tuple[SwarmEvent, ...]
