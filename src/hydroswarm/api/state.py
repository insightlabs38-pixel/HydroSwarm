"""Local runtime state and typed API-only contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import tempfile
from typing import Annotated, Literal
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
from hydroswarm.storage import AuditEvent, AuditLedger


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceStatus(ApiModel):
    status: Literal["ok", "ready"]
    offline: Literal[True] = True
    version: str


class NetworkValidationRequest(ApiModel):
    node_ids: tuple[str, ...] = Field(min_length=1)
    link_count: int = Field(ge=0)


class NetworkRecord(ApiModel):
    network_id: str
    node_count: int
    link_count: int
    valid: Literal[True] = True
    validated_at: datetime


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
    verifier: Verifier
    networks: dict[str, NetworkRecord] = field(default_factory=dict)
    incidents: dict[UUID, IncidentRuntime] = field(default_factory=dict)

    @classmethod
    def create(
        cls, *, verifier: Verifier, ledger_path: str | Path | None = None
    ) -> RuntimeState:
        if ledger_path is None:
            directory = Path(tempfile.mkdtemp(prefix="hydroswarm-api-"))
            ledger_path = directory / "audit.sqlite3"
        return cls(ledger=AuditLedger(ledger_path), verifier=verifier)

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

