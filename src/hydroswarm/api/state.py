"""Local runtime state and typed API-only contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from hydroswarm.domain import (
    CandidateSet,
    ConsequenceMetrics,
    IncidentCreate,
    IncidentState,
    OperationalPlan,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.storage import AuditEvent, AuditLedger, Database, ScenarioStore
from hydroswarm.worker import PersistentJobQueue


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiSettings(ApiModel):
    """Explicit runtime dependencies and safety limits; no environment secrets are exposed."""

    demo_fallback_enabled: bool = True
    maximum_request_bytes: int = Field(default=5 * 1024 * 1024 + 256 * 1024, ge=1)
    worker_threads: Literal[1] = 1
    exact_plan_simulation_limit: int = Field(default=3, ge=1, le=3)


class ServiceStatus(ApiModel):
    status: Literal["ok", "ready", "not_ready"]
    offline: Literal[True] = True
    version: str
    mode: str | None = None
    checks: dict[str, bool] | None = None


class NetworkValidationRequest(ApiModel):
    node_ids: tuple[str, ...] = Field(min_length=1)
    link_count: int = Field(ge=0)


class NetworkRecord(ApiModel):
    network_id: str
    name: str
    version: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    node_count: int
    link_count: int
    valid: bool = True
    validated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    geojson: dict[str, Any] = Field(default_factory=lambda: {"type": "FeatureCollection", "features": []})
    validation_errors: tuple[str, ...] = ()


class SampleRecommendation(ApiModel):
    action: Literal["SAMPLE"] = "SAMPLE"
    node_id: str
    expected_information_gain: Annotated[float, Field(ge=0.0)]
    alternatives: tuple[str, ...] = ()
    runtime_mode: str = "UNKNOWN"
    fallback_reasons: tuple[str, ...] = ()


class PlanGenerationRequest(ApiModel):
    count: int = Field(default=2, ge=1, le=8)


class ApprovalRequest(ApiModel):
    approved: Literal[True]
    operator_id: str = Field(min_length=1, max_length=80)


class ApprovalReceipt(ApiModel):
    incident_id: UUID
    plan_id: UUID
    approved: Literal[True] = True
    operator_id: str
    approved_at: datetime


class ReplayResponse(ApiModel):
    state: IncidentState
    events: tuple[AuditEvent, ...]
    chain_valid: bool


class IncidentExport(ApiModel):
    incident: IncidentState
    plans: tuple[OperationalPlan, ...]
    verifications: tuple[PlanVerification, ...]
    events: tuple[AuditEvent, ...]
    analysis: dict[str, Any] | None = None


class AnalysisResponse(ApiModel):
    incident_id: UUID
    runtime_mode: str
    fallback_reasons: tuple[str, ...]
    node_alignment: tuple[str, ...]
    classical_belief: dict[str, float]
    neural_belief: dict[str, float] | None
    fused_belief: dict[str, float]
    candidate_nodes: tuple[str, ...]
    calibrated: bool
    ood_level: str
    disagreement_js: float | None
    evidence_sufficient: bool
    planning_allowed: bool
    control_action: str
    recommended_sample: str | None
    posterior_history: tuple[dict[str, Any], ...]
    provenance_hashes: dict[str, str]
    latencies_ms: dict[str, float]


class NetworkNodeView(ApiModel):
    node_id: str
    node_type: str
    elevation_m: float
    coordinates: tuple[float, float]
    probability: float = Field(ge=0.0, le=1.0)
    concentration_mg_l: float | None = None
    candidate: bool


class NetworkLinkView(ApiModel):
    link_id: str
    link_type: str
    start_node: str
    end_node: str


class SensorHealthView(ApiModel):
    sensor_id: str
    node_id: str
    health: Literal["HEALTHY", "DRIFT", "MISSING"]
    quality: float = Field(ge=0.0, le=1.0)
    observed_at: datetime
    received_at: datetime
    pressure_m: float | None = None
    concentration_mg_l: float | None = None


class SampleRecommendationView(ApiModel):
    node_id: str
    expected_information_gain: float = Field(ge=0.0)
    alternatives: tuple[str, ...] = ()


class EvidenceHistoryEntryView(ApiModel):
    round_index: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    valid_concentration_count: int = Field(ge=0)
    sensor_nodes: tuple[str, ...]
    evidence_hash: str


class PlanView(ApiModel):
    plan: OperationalPlan
    verification: PlanVerification | None = None


class ExplanationPayload(ApiModel):
    intent: str
    text: str
    facts: dict[str, Any]
    limitations: tuple[str, ...]


class ProvenanceView(ApiModel):
    """Every hash/version an operator needs to trust this exact response.

    ``simulator``/``simulator_version`` are only known once at least one plan
    has undergone authoritative WNTR/EPANET verification for this incident;
    they are None (not fabricated) until then.
    """

    network_hash: str
    feature_schema_hash: str
    model_version: str
    model_checkpoint_hash: str
    calibration_version: str
    calibration_hash: str
    simulator: str | None = None
    simulator_version: str | None = None


class IncidentView(ApiModel):
    """A complete, versioned, live-only snapshot of one incident (overnight-plan.txt
    Task 3.2). Every field here is sourced from live backend state produced by
    the hybrid pipeline for this incident; this schema is never populated from
    demo/fixture content, and ``extra="forbid"`` plus non-Optional typing means
    an assembler bug that fails to supply a required field raises a validation
    error instead of silently shipping an incomplete view.
    """

    schema_version: Literal[1] = 1
    incident_id: UUID
    network_id: str
    runtime_mode: Literal["FULL_HYBRID", "CLASSICAL_SAFE"]
    data_mode: Literal["LIVE"] = "LIVE"
    controller_state: Literal["DETECTED", "ANALYZING", "SAMPLING", "PLANNING", "APPROVAL", "CLOSED"]
    generated_at: datetime
    provenance: ProvenanceView
    candidates: CandidateSet
    disagreement_js: float | None = None
    ood_level: str
    calibration_alpha: float | None = None
    nodes: tuple[NetworkNodeView, ...]
    links: tuple[NetworkLinkView, ...]
    sensor_health: tuple[SensorHealthView, ...]
    sample_recommendation: SampleRecommendationView | None = None
    evidence_history: tuple[EvidenceHistoryEntryView, ...]
    plans: tuple[PlanView, ...]
    selected_plan_id: UUID | None = None
    recommended_plan_id: UUID | None = None
    counterfactual_consequences: dict[str, ConsequenceMetrics]
    explanations: tuple[ExplanationPayload, ...]
    audit_events: tuple[AuditEvent, ...]
    runtime_metrics_ms: dict[str, float]


Verifier = Callable[[OperationalPlan, IncidentState], PlanVerification]


@dataclass(slots=True)
class IncidentRuntime:
    create: IncidentCreate
    state: IncidentState
    plans: dict[UUID, OperationalPlan] = field(default_factory=dict)
    verifications: dict[UUID, PlanVerification] = field(default_factory=dict)
    analysis: Any | None = None
    runtime_mode: str = "UNANALYZED"
    fallback_reasons: tuple[str, ...] = ()
    pipeline: Any | None = None
    swarm: Any | None = None
    progress: dict[str, Any] = field(default_factory=lambda: {"state": "IDLE", "progress": 0.0})


@dataclass(slots=True)
class RuntimeState:
    ledger: AuditLedger
    store: ScenarioStore
    verifier: Verifier | None
    network_directory: Path
    networks: dict[str, NetworkRecord] = field(default_factory=dict)
    incidents: dict[UUID, IncidentRuntime] = field(default_factory=dict)
    pipeline_factory: Any | None = None
    swarm_factory: Any | None = None
    jobs: PersistentJobQueue | None = None

    @classmethod
    def create(
        cls,
        *,
        verifier: Verifier | None,
        ledger_path: str | Path | None = None,
        database_path: str | Path | None = None,
        network_directory: str | Path | None = None,
        pipeline_factory: Any | None = None,
        swarm_factory: Any | None = None,
    ) -> RuntimeState:
        path = database_path or ledger_path
        database = Database(path)
        store = ScenarioStore(database)
        directory = (
            Path(network_directory).expanduser().resolve()
            if network_directory
            else database.path.parent / "networks"
        )
        directory.mkdir(parents=True, exist_ok=True)
        runtime = cls(
            ledger=AuditLedger(database.path),
            store=store,
            verifier=verifier,
            network_directory=directory,
            networks=store.load_networks(),
            incidents=store.load_incidents(),
            pipeline_factory=pipeline_factory,
            swarm_factory=swarm_factory,
        )
        runtime.jobs = PersistentJobQueue(database)
        return runtime

    def persist(self, runtime: IncidentRuntime) -> None:
        self.store.save_incident(runtime)

    @staticmethod
    def state_hash(state: IncidentState) -> str:
        canonical = state.model_dump_json(exclude_none=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append_event(
        self,
        runtime: IncidentRuntime,
        *,
        event_type: str,
        actor: str,
        payload: dict[str, object],
        simulator_version: str = "not-invoked",
    ) -> AuditEvent:
        return self.ledger.append(
            incident_id=runtime.state.incident_id,
            event_type=event_type,
            actor=actor,
            input_state_hash=self.state_hash(runtime.state),
            payload=payload,
            model_version="hydroswarm-api-0.1.0",
            simulator_version=simulator_version,
        )


def utc_now() -> datetime:
    return datetime.now(UTC)


def demo_candidates(observations: tuple[SensorObservation, ...]) -> CandidateSet:
    """Explicit non-authoritative fallback; stable evidence weights are never uniform placeholders."""

    usable_nodes = sorted(
        {
            observation.node_id
            for observation in observations
            if not observation.missing and observation.quality > 0
        }
    )
    if not usable_nodes:
        raise ValueError("incident has no usable sensor observations")
    scores = {
        node_id: sum(
            (item.concentration_mg_l or 0.0) * item.quality + 1e-6 * (index + 1)
            for index, item in enumerate(observations)
            if item.node_id == node_id and not item.missing
        )
        for node_id in usable_nodes
    }
    total = sum(scores.values())
    probabilities = {node_id: score / total for node_id, score in scores.items()}
    return CandidateSet(
        node_probabilities=probabilities,
        node_ids=tuple(usable_nodes),
        calibrated=False,
    )
