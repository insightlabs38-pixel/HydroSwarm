"""Offline FastAPI application for the typed HydroSwarm workflow."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hydroswarm import __version__
from hydroswarm.domain import (
    ActionType,
    IncidentCreate,
    IncidentState,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.storage import AuditEvent
from hydroswarm.networks import MAX_INP_BYTES, NetworkImportError, NetworkImporter
from hydroswarm.simulation import HydraulicSimulator, PlanVerifier
from hydroswarm.simulation.wrapper import wntr

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


def create_app(
    *,
    verifier: Verifier | None = None,
    ledger_path: str | Path | None = None,
    database_path: str | Path | None = None,
    network_directory: str | Path | None = None,
    max_request_bytes: int = MAX_INP_BYTES + 256 * 1024,
) -> FastAPI:
    """Create a fully local app; deployment must bind it to ``127.0.0.1``."""

    app = FastAPI(
        title="HydroSwarm API",
        version=__version__,
        description="Offline neuro-hydraulic incident decision support",
    )
    runtime_state = RuntimeState.create(
        verifier=verifier,
        ledger_path=ledger_path,
        database_path=database_path,
        network_directory=network_directory,
    )
    app.state.runtime = runtime_state
    app.state.network_importer = NetworkImporter(
        runtime_state.store, runtime_state.network_directory
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )

    @app.middleware("http")
    async def constrain_request_size(request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > max_request_bytes
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "invalid content-length"})
            if too_large:
                return JSONResponse(status_code=413, content={"detail": "request body too large"})
        return await call_next(request)

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

    @app.get(
        "/api/health",
        response_model=ServiceStatus,
        response_model_exclude_none=True,
    )
    def health() -> ServiceStatus:
        return ServiceStatus(status="ok", version=__version__)

    @app.get("/api/version")
    def version() -> dict[str, str | bool]:
        return {"version": __version__, "offline": True}

    @app.get("/api/readiness", response_model=ServiceStatus)
    def readiness() -> ServiceStatus | JSONResponse:
        try:
            database_ready = runtime().store.database.ping()
        except Exception:
            database_ready = False
        verifier_ready = runtime().verifier is not None or wntr is not None
        checks = {"database": database_ready, "authoritative_verifier": verifier_ready}
        mode = "injected-verifier" if runtime().verifier is not None else "authoritative-wntr"
        response = ServiceStatus(
            status="ready" if all(checks.values()) else "not_ready",
            version=__version__,
            mode=mode,
            checks=checks,
        )
        if response.status != "ready":
            return JSONResponse(status_code=503, content=response.model_dump(mode="json"))
        return response

    @app.get("/api/networks", response_model=list[NetworkRecord])
    def list_networks() -> list[NetworkRecord]:
        return [runtime().networks[key] for key in sorted(runtime().networks)]

    @app.post(
        "/api/networks/import",
        response_model=NetworkRecord,
        status_code=status.HTTP_201_CREATED,
    )
    async def import_network(file: UploadFile = File(...)) -> NetworkRecord:
        content = await file.read(MAX_INP_BYTES + 1)
        try:
            record = app.state.network_importer.import_bytes(file.filename or "", content)
        except NetworkImportError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        runtime().networks[record.network_id] = record
        return record

    @app.get("/api/networks/{network_id}", response_model=NetworkRecord)
    def get_network(network_id: str) -> NetworkRecord:
        try:
            return runtime().networks[network_id]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="network not found") from error

    @app.post("/api/networks/{network_id}/validate", response_model=NetworkRecord)
    def validate_network(
        network_id: str, request: NetworkValidationRequest
    ) -> NetworkRecord:
        if runtime().verifier is None:
            raise HTTPException(
                status_code=409,
                detail="manual topology validation is disabled; import an authoritative .inp network",
            )
        if not network_id.strip():
            raise HTTPException(status_code=422, detail="network_id must not be blank")
        if len(set(request.node_ids)) != len(request.node_ids):
            raise HTTPException(status_code=422, detail="node_ids must be unique")
        digest = hashlib.sha256(
            f"{network_id}:{','.join(request.node_ids)}:{request.link_count}".encode()
        ).hexdigest()
        record = NetworkRecord(
            network_id=network_id,
            name=network_id,
            version=1,
            sha256=digest,
            node_count=len(request.node_ids),
            link_count=request.link_count,
            validated_at=utc_now(),
            metadata={"node_ids": list(request.node_ids), "link_ids": []},
        )
        runtime().networks[network_id] = record
        runtime().store.save_network(record, inp_path=None)
        return record

    @app.post(
        "/api/incidents", response_model=IncidentState, status_code=status.HTTP_201_CREATED
    )
    def create_incident(request: IncidentCreate) -> IncidentState:
        if request.network_id not in runtime().networks:
            raise HTTPException(status_code=409, detail="network must be validated first")
        network = runtime().networks[request.network_id]
        known_nodes = set(network.metadata.get("node_ids", ()))
        unknown_nodes = sorted({item.node_id for item in request.observations} - known_nodes)
        if unknown_nodes:
            raise HTTPException(status_code=422, detail="observations reference unknown network nodes")
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
        runtime().persist(record)
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
        runtime().persist(record)
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
        known_nodes = set(runtime().networks[record.state.network_id].metadata.get("node_ids", ()))
        if observation.node_id not in known_nodes:
            raise HTTPException(status_code=422, detail="sample references an unknown network node")
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
        runtime().persist(record)
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
        runtime().persist(record)
        return plans

    @app.post(
        "/api/incidents/{incident_id}/plans/{plan_id}/verify",
        response_model=PlanVerification,
    )
    def verify_plan(incident_id: UUID, plan_id: UUID) -> PlanVerification:
        record = incident_or_404(incident_id)
        plan = plan_or_404(record, plan_id)
        if runtime().verifier is not None:
            verification = runtime().verifier(plan, record.state)
        else:
            network_path = runtime().store.network_path(record.state.network_id)
            if not network_path or wntr is None:
                raise HTTPException(
                    status_code=503,
                    detail="authoritative WNTR verification is unavailable",
                )
            verification = PlanVerifier(HydraulicSimulator(network_path)).verify(plan)
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
        runtime().persist(record)
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
        runtime().persist(record)
        runtime().store.save_approval(receipt)
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
