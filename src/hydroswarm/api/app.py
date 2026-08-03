"""Offline FastAPI application for the typed HydroSwarm workflow."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, status

from hydroswarm import __version__
from hydroswarm.domain import (
    ActionType,
    ConsequenceMetrics,
    IncidentCreate,
    IncidentState,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.storage import AuditEvent

from .state import (
    ApprovalReceipt,
    ApprovalRequest,
    IncidentExport,
    IncidentRuntime,
    NetworkRecord,
    NetworkValidationRequest,
    PlanGenerationRequest,
    ReplayResponse,
    RuntimeState,
    SampleRecommendation,
    ServiceStatus,
    Verifier,
    deterministic_candidates,
    utc_now,
)


def _default_verifier(plan: OperationalPlan, state: IncidentState) -> PlanVerification:
    digest = hashlib.sha256(
        f"{state.incident_id}:{plan.plan_id}:{len(state.observations)}".encode()
    ).hexdigest()
    return PlanVerification(
        plan_id=plan.plan_id,
        decision=PlanDecision.VERIFIED,
        simulator="deterministic-local-verifier",
        simulator_version="0.1.0",
        state_hash=digest,
        consequences=ConsequenceMetrics(
            minimum_pressure_m=20.0,
            service_availability=1.0,
            operation_count=len(plan.actions),
        ),
    )


def create_app(
    *,
    verifier: Verifier | None = None,
    ledger_path: str | Path | None = None,
) -> FastAPI:
    """Create a fully local app; deployment must bind it to ``127.0.0.1``."""

    app = FastAPI(
        title="HydroSwarm API",
        version=__version__,
        description="Offline neuro-hydraulic incident decision support",
    )
    runtime_state = RuntimeState.create(
        verifier=verifier or _default_verifier, ledger_path=ledger_path
    )
    app.state.runtime = runtime_state

    def runtime() -> RuntimeState:
        return app.state.runtime

    def incident_or_404(incident_id: UUID) -> IncidentRuntime:
        try:
            return runtime().incidents[incident_id]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="incident not found") from error

    def plan_or_404(incident: IncidentRuntime, plan_id: UUID) -> OperationalPlan:
        try:
            return incident.plans[plan_id]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="plan not found") from error

    @app.get("/api/health", response_model=ServiceStatus)
    def health() -> ServiceStatus:
        return ServiceStatus(status="ok", version=__version__)

    @app.get("/api/version")
    def version() -> dict[str, str | bool]:
        return {"version": __version__, "offline": True}

    @app.get("/api/readiness", response_model=ServiceStatus)
    def readiness() -> ServiceStatus:
        return ServiceStatus(status="ready", version=__version__)

    @app.get("/api/networks", response_model=list[NetworkRecord])
    def list_networks() -> list[NetworkRecord]:
        return [runtime().networks[key] for key in sorted(runtime().networks)]

    @app.post("/api/networks/{network_id}/validate", response_model=NetworkRecord)
    def validate_network(
        network_id: str, request: NetworkValidationRequest
    ) -> NetworkRecord:
        if not network_id.strip():
            raise HTTPException(status_code=422, detail="network_id must not be blank")
        if len(set(request.node_ids)) != len(request.node_ids):
            raise HTTPException(status_code=422, detail="node_ids must be unique")
        record = NetworkRecord(
            network_id=network_id,
            node_count=len(request.node_ids),
            link_count=request.link_count,
            validated_at=utc_now(),
        )
        runtime().networks[network_id] = record
        return record

    @app.post(
        "/api/incidents", response_model=IncidentState, status_code=status.HTTP_201_CREATED
    )
    def create_incident(request: IncidentCreate) -> IncidentState:
        if request.network_id not in runtime().networks:
            raise HTTPException(status_code=409, detail="network must be validated first")
        state = IncidentState(
            network_id=request.network_id,
            status="DETECTED",
            observations=request.observations,
        )
        record = IncidentRuntime(create=request, state=state)
        runtime().incidents[state.incident_id] = record
        runtime().append_event(
            record,
            event_type="INCIDENT_CREATED",
            actor="OPERATOR",
            payload={"network_id": request.network_id},
        )
        return state

    @app.get("/api/incidents/{incident_id}", response_model=IncidentState)
    def get_incident(incident_id: UUID) -> IncidentState:
        return incident_or_404(incident_id).state

    @app.post("/api/incidents/{incident_id}/analyze", response_model=IncidentState)
    def analyze_incident(incident_id: UUID) -> IncidentState:
        record = incident_or_404(incident_id)
        try:
            candidates = deterministic_candidates(record.state.observations)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        runtime().append_event(
            record,
            event_type="ANALYSIS_STARTED",
            actor="HYDRO_SENTINEL",
            payload={"observation_count": len(record.state.observations)},
        )
        record.state = record.state.model_copy(
            update={"status": "SAMPLING", "candidates": candidates}
        )
        runtime().append_event(
            record,
            event_type="SOURCE_LOCALIZED",
            actor="HYDRO_SENTINEL",
            payload={"candidate_nodes": list(candidates.node_ids)},
        )
        return record.state

    @app.post(
        "/api/incidents/{incident_id}/samples/recommend",
        response_model=SampleRecommendation,
    )
    def recommend_sample(incident_id: UUID) -> SampleRecommendation:
        record = incident_or_404(incident_id)
        if record.state.candidates is None:
            raise HTTPException(status_code=409, detail="analyze incident first")
        if record.state.sample_count >= record.create.maximum_samples:
            raise HTTPException(status_code=409, detail="sampling budget exhausted")
        nodes = record.state.candidates.node_ids
        recommendation = SampleRecommendation(
            node_id=nodes[0],
            expected_information_gain=1.0 / len(nodes),
            alternatives=nodes[1:3],
        )
        runtime().append_event(
            record,
            event_type="SAMPLE_RECOMMENDED",
            actor="HYDRO_SCOUT",
            payload=recommendation.model_dump(mode="json"),
        )
        return recommendation

    @app.post("/api/incidents/{incident_id}/samples", response_model=IncidentState)
    def add_sample(incident_id: UUID, observation: SensorObservation) -> IncidentState:
        record = incident_or_404(incident_id)
        if record.state.sample_count >= record.create.maximum_samples:
            raise HTTPException(status_code=409, detail="sampling budget exhausted")
        runtime().append_event(
            record,
            event_type="SAMPLE_RECEIVED",
            actor="OPERATOR",
            payload={"sensor_id": observation.sensor_id, "node_id": observation.node_id},
        )
        record.state = record.state.model_copy(
            update={
                "status": "SAMPLING",
                "observations": (*record.state.observations, observation),
                "sample_count": record.state.sample_count + 1,
            }
        )
        return record.state

    @app.post(
        "/api/incidents/{incident_id}/plans/generate",
        response_model=list[OperationalPlan],
    )
    def generate_plans(
        incident_id: UUID, request: PlanGenerationRequest
    ) -> list[OperationalPlan]:
        record = incident_or_404(incident_id)
        if record.state.candidates is None:
            raise HTTPException(status_code=409, detail="analyze incident first")
        target_nodes = record.state.candidates.node_ids
        plans = [
            OperationalPlan(
                incident_id=incident_id,
                name=f"Monitor candidate {index + 1}",
                actions=(
                    OperationalAction(
                        action_type=ActionType.MONITOR_NODE,
                        target_id=target_nodes[index % len(target_nodes)],
                    ),
                ),
                model_version="hydroswarm-api-0.1.0",
            )
            for index in range(request.count)
        ]
        record.plans.update({plan.plan_id: plan for plan in plans})
        record.state = record.state.model_copy(
            update={"status": "PLANNING", "approval_pending": False}
        )
        runtime().append_event(
            record,
            event_type="PLANS_GENERATED",
            actor="HYDRO_STRATEGIST",
            payload={"plan_ids": [str(plan.plan_id) for plan in plans]},
        )
        return plans

    @app.post(
        "/api/incidents/{incident_id}/plans/{plan_id}/verify",
        response_model=PlanVerification,
    )
    def verify_plan(incident_id: UUID, plan_id: UUID) -> PlanVerification:
        record = incident_or_404(incident_id)
        plan = plan_or_404(record, plan_id)
        verification = runtime().verifier(plan, record.state)
        if verification.plan_id != plan_id:
            raise HTTPException(status_code=500, detail="verifier returned the wrong plan_id")
        record.verifications[plan_id] = verification
        verified = verification.decision == PlanDecision.VERIFIED
        record.state = record.state.model_copy(
            update={
                "status": "APPROVAL" if verified else "PLANNING",
                "approval_pending": verified,
            }
        )
        runtime().append_event(
            record,
            event_type="PLAN_VERIFIED" if verified else "PLAN_REJECTED",
            actor="HYDRO_VERIFIER",
            payload={
                "plan_id": str(plan_id),
                "decision": verification.decision.value,
                "rejection_codes": list(verification.rejection_codes),
            },
            simulator_version=verification.simulator_version,
        )
        return verification

    @app.post(
        "/api/incidents/{incident_id}/plans/{plan_id}/approve",
        response_model=ApprovalReceipt,
    )
    def approve_plan(
        incident_id: UUID, plan_id: UUID, request: ApprovalRequest
    ) -> ApprovalReceipt:
        record = incident_or_404(incident_id)
        plan_or_404(record, plan_id)
        verification = record.verifications.get(plan_id)
        if verification is None or verification.decision != PlanDecision.VERIFIED:
            raise HTTPException(
                status_code=409, detail="only a VERIFIED plan can be approved"
            )
        approved_at = datetime.now(UTC)
        receipt = ApprovalReceipt(
            incident_id=incident_id,
            plan_id=plan_id,
            operator_id=request.operator_id,
            approved_at=approved_at,
        )
        runtime().append_event(
            record,
            event_type="PLAN_APPROVED",
            actor=f"OPERATOR:{request.operator_id}",
            payload={"plan_id": str(plan_id)},
            simulator_version=verification.simulator_version,
        )
        record.state = record.state.model_copy(
            update={"status": "CLOSED", "approval_pending": False}
        )
        return receipt

    @app.get(
        "/api/incidents/{incident_id}/events", response_model=list[AuditEvent]
    )
    def incident_events(incident_id: UUID) -> list[AuditEvent]:
        incident_or_404(incident_id)
        return runtime().ledger.events(incident_id)

    @app.post(
        "/api/incidents/{incident_id}/replay", response_model=ReplayResponse
    )
    def replay_incident(incident_id: UUID) -> ReplayResponse:
        record = incident_or_404(incident_id)
        return ReplayResponse(
            state=record.state,
            events=tuple(runtime().ledger.events(incident_id)),
            chain_valid=runtime().ledger.verify_chain(incident_id),
        )

    @app.get(
        "/api/incidents/{incident_id}/export", response_model=IncidentExport
    )
    def export_incident(incident_id: UUID) -> IncidentExport:
        record = incident_or_404(incident_id)
        return IncidentExport(
            incident=record.state,
            plans=tuple(record.plans.values()),
            verifications=tuple(record.verifications.values()),
            events=tuple(runtime().ledger.events(incident_id)),
        )

    return app


app = create_app()

