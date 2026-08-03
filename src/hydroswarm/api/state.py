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
    IncidentCreate,
    IncidentState,
    OperationalPlan,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.storage import AuditEvent, AuditLedger, Database, ScenarioStore


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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


Verifier = Callable[[OperationalPlan, IncidentState], PlanVerification]


@dataclass(slots=True)
class IncidentRuntime:
    create: IncidentCreate
    state: IncidentState
    plans: dict[UUID, OperationalPlan] = field(default_factory=dict)
    verifications: dict[UUID, PlanVerification] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeState:
    ledger: AuditLedger
    store: ScenarioStore
    verifier: Verifier | None
    network_directory: Path
    networks: dict[str, NetworkRecord] = field(default_factory=dict)
    incidents: dict[UUID, IncidentRuntime] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        verifier: Verifier | None,
        ledger_path: str | Path | None = None,
        database_path: str | Path | None = None,
        network_directory: str | Path | None = None,
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
        return cls(
            ledger=AuditLedger(database.path),
            store=store,
            verifier=verifier,
            network_directory=directory,
            networks=store.load_networks(),
            incidents=store.load_incidents(),
        )

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


def deterministic_candidates(observations: tuple[SensorObservation, ...]) -> CandidateSet:
    """Build a transparent placeholder posterior until model inference is wired."""

    usable_nodes = sorted(
        {
            observation.node_id
            for observation in observations
            if not observation.missing and observation.quality > 0
        }
    )
    if not usable_nodes:
        raise ValueError("incident has no usable sensor observations")
    weight = 1.0 / len(usable_nodes)
    probabilities = {node_id: weight for node_id in usable_nodes}
    return CandidateSet(
        node_probabilities=probabilities,
        node_ids=tuple(usable_nodes),
        calibrated=False,
    )
