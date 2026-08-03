"""Offline FastAPI application for the typed HydroSwarm workflow."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import networkx as nx

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from hydroswarm import __version__
from hydroswarm.domain import (
    ActionType,
    CandidateSet,
    IncidentCreate,
    IncidentState,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.explanation import (
    EvidenceBundle,
    ExplanationIntent,
    deterministic_operational_summary,
    explain,
)
from hydroswarm.agents import HydroScout, HydroSentinel, HydroStrategist, SwarmController
from hydroswarm.inference import IncidentAnalysisResult
from hydroswarm.preprocessing import SensorSeries
from hydroswarm.storage import AuditEvent
from hydroswarm.networks import MAX_INP_BYTES, NetworkImportError, NetworkImporter
from hydroswarm.simulation import HydraulicSimulator, PlanVerifier
from hydroswarm.simulation.wrapper import wntr
from hydroswarm.runtime import DefaultPipelineFactory

from .state import (
    AnalysisResponse,
    ApiSettings,
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
    demo_candidates,
    utc_now,
)


def create_app(
    *,
    verifier: Verifier | None = None,
    ledger_path: str | Path | None = None,
    database_path: str | Path | None = None,
    network_directory: str | Path | None = None,
    pipeline_factory: object | None = None,
    swarm_factory: object | None = None,
    settings: ApiSettings | None = None,
    max_request_bytes: int = MAX_INP_BYTES + 256 * 1024,
) -> FastAPI:
    """Create a fully local app; deployment must bind it to ``127.0.0.1``."""

    settings = settings or ApiSettings(maximum_request_bytes=max_request_bytes)
    max_request_bytes = settings.maximum_request_bytes
    runtime_state = RuntimeState.create(
        verifier=verifier,
        ledger_path=ledger_path,
        database_path=database_path,
        network_directory=network_directory,
        pipeline_factory=pipeline_factory,
        swarm_factory=swarm_factory,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if runtime_state.jobs:
            runtime_state.jobs.close()

    app = FastAPI(
        title="HydroSwarm API",
        version=__version__,
        description="Offline neuro-hydraulic incident decision support",
        lifespan=lifespan,
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

    def bind_pipeline(record: IncidentRuntime) -> object | None:
        if record.pipeline is not None:
            return record.pipeline
        factory = runtime().pipeline_factory
        if factory is None:
            return None
        if hasattr(factory, "analyze"):
            record.pipeline = factory
        else:
            path = runtime().store.network_path(record.state.network_id)
            record.pipeline = factory(runtime().networks[record.state.network_id], path)
        return record.pipeline

    def sensor_series(record: IncidentRuntime) -> tuple[SensorSeries, ...]:
        grouped: dict[str, list[SensorObservation]] = defaultdict(list)
        for item in record.state.observations:
            grouped[item.node_id].append(item)
        origin = record.create.detected_at
        result: list[SensorSeries] = []
        for node_id, items in sorted(grouped.items()):
            items.sort(key=lambda item: item.observed_at)
            result.append(SensorSeries(
                node_id=node_id,
                timestamps_seconds=tuple((item.observed_at - origin).total_seconds() for item in items),
                concentration_mg_l=tuple(item.concentration_mg_l for item in items),
                pressure_m=tuple(item.pressure_m for item in items),
                health=tuple(item.quality for item in items),
                missing=tuple(item.missing for item in items),
                drift=tuple(item.drift_flag for item in items),
                delayed=tuple(item.received_at > item.observed_at for item in items),
            ))
        return tuple(result)

    def analysis_response(record: IncidentRuntime) -> AnalysisResponse:
        item = record.analysis
        if isinstance(item, IncidentAnalysisResult):
            return AnalysisResponse(
                incident_id=record.state.incident_id,
                runtime_mode=item.runtime_mode.value,
                fallback_reasons=record.fallback_reasons,
                node_alignment=item.node_alignment,
                classical_belief=dict(item.classical_belief),
                neural_belief=dict(item.neural_belief) if item.neural_belief else None,
                fused_belief=dict(item.fused_belief),
                candidate_nodes=item.conformal_candidate_nodes,
                calibrated=item.calibrated,
                ood_level=item.ood_level.value,
                disagreement_js=item.fusion_diagnostics.disagreement_js if item.fusion_diagnostics else None,
                evidence_sufficient=item.evidence_sufficient,
                planning_allowed=item.planning_allowed,
                control_action=item.control_action.value,
                recommended_sample=item.sample_result.recommended_node if item.sample_result else None,
                posterior_history=tuple({
                    "round_index": snap.round_index,
                    "observation_count": snap.observation_count,
                    "candidate_nodes": snap.candidate_nodes,
                    "entropy_bits": snap.entropy_bits,
                    "evidence_hash": snap.evidence_hash,
                } for snap in item.posterior_history),
                provenance_hashes=dict(item.provenance_hashes),
                latencies_ms=dict(item.latencies_ms),
            )
        if isinstance(item, dict):
            return AnalysisResponse.model_validate(item)
        raise HTTPException(status_code=409, detail="incident has not been analyzed")

    def perform_analysis(record: IncidentRuntime) -> AnalysisResponse:
        record.progress = {"state": "RUNNING", "progress": 0.1, "message": "hybrid analysis"}
        pipeline = bind_pipeline(record)
        if pipeline is not None:
            series = sensor_series(record)
            previous = record.analysis if isinstance(record.analysis, IncidentAnalysisResult) else None
            network = pipeline.simulator.network
            if previous is None:
                analysis = pipeline.analyze(
                    record.state.incident_id, network, series,
                    sample_budget_remaining=record.create.maximum_samples - record.state.sample_count,
                )
            else:
                analysis = pipeline.reanalyze_after_sample(
                    previous, network, series,
                    sample_budget_remaining=record.create.maximum_samples - record.state.sample_count,
                )
            record.analysis = analysis
            record.runtime_mode = analysis.runtime_mode.value
            reasons = list(analysis.planning_suppression_reasons)
            if analysis.neural_failure:
                reasons.append(f"NEURAL_FAILURE:{analysis.neural_failure}")
            record.fallback_reasons = tuple(dict.fromkeys(reasons))
            candidates = analysis.conformal_candidate_nodes or tuple(
                node for node, _ in sorted(analysis.fused_belief.items(), key=lambda pair: -pair[1])[:3]
            )
            record.state = record.state.model_copy(update={
                "status": "PLANNING" if analysis.planning_allowed else "SAMPLING",
                "candidates": CandidateSet(
                    node_probabilities=dict(analysis.fused_belief), node_ids=candidates,
                    calibrated=analysis.calibrated,
                ),
                "disagreement_js": analysis.fusion_diagnostics.disagreement_js if analysis.fusion_diagnostics else None,
                "ood_level": analysis.ood_level,
            })
        else:
            if not settings.demo_fallback_enabled:
                raise HTTPException(status_code=503, detail="hybrid pipeline artifacts unavailable")
            candidates = demo_candidates(record.state.observations)
            record.runtime_mode = "DEMO_FALLBACK"
            record.fallback_reasons = ("HYBRID_PIPELINE_NOT_CONFIGURED", "UNVERIFIED_DEMO_ANALYSIS")
            record.state = record.state.model_copy(update={"status": "SAMPLING", "candidates": candidates})
            record.analysis = {
                "incident_id": record.state.incident_id,
                "runtime_mode": record.runtime_mode,
                "fallback_reasons": record.fallback_reasons,
                "node_alignment": tuple(candidates.node_probabilities),
                "classical_belief": candidates.node_probabilities,
                "neural_belief": None,
                "fused_belief": candidates.node_probabilities,
                "candidate_nodes": candidates.node_ids,
                "calibrated": False,
                "ood_level": "CAUTION",
                "disagreement_js": None,
                "evidence_sufficient": False,
                "planning_allowed": False,
                "control_action": "REQUEST_SAMPLE",
                "recommended_sample": candidates.node_ids[0],
                "posterior_history": ({"round_index": record.state.sample_count, "observation_count": len(record.state.observations), "candidate_nodes": candidates.node_ids, "entropy_bits": 0.0, "evidence_hash": runtime().state_hash(record.state)},),
                "provenance_hashes": {"evidence": runtime().state_hash(record.state)},
                "latencies_ms": {},
            }
        record.progress = {"state": "COMPLETE", "progress": 1.0, "message": record.runtime_mode}
        runtime().append_event(
            record, event_type="HYBRID_ANALYSIS_COMPLETED", actor="SWARM_CONTROLLER",
            payload={
                "runtime_mode": record.runtime_mode,
                "fallback_reasons": list(record.fallback_reasons),
                "candidate_nodes": list(record.state.candidates.node_ids),
            },
        )
        runtime().persist(record)
        return analysis_response(record)

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
        pipeline_dependency = runtime().pipeline_factory
        pipeline_ready = pipeline_dependency is not None and (
            hasattr(pipeline_dependency, "analyze") or callable(pipeline_dependency)
        )
        deferred_pipeline = callable(pipeline_dependency) and not hasattr(
            pipeline_dependency, "analyze"
        )
        trained_assets = getattr(pipeline_dependency, "trained_assets_ready", None)
        asset_aware = isinstance(trained_assets, bool)
        checks = {
            "database": database_ready,
            "authoritative_verifier": verifier_ready,
            "hybrid_pipeline": pipeline_ready,
            "signature_artifact": bool(trained_assets) if asset_aware else deferred_pipeline
            or getattr(pipeline_dependency, "signature_artifact", None) is not None,
            "calibration_artifact": bool(trained_assets) if asset_aware else deferred_pipeline
            or getattr(pipeline_dependency, "calibration_artifact", None) is not None,
            "trained_checkpoint": bool(trained_assets) if asset_aware else False,
            "model_or_classical_safe_mode": pipeline_ready,
            "model_worker": runtime().jobs is not None,
        }
        mode = (
            "hybrid-trained-ready" if trained_assets is True
            else "classical-safe-ready" if asset_aware and pipeline_ready
            else "hybrid-ready" if pipeline_ready
            else "injected-verifier" if runtime().verifier is not None
            else "authoritative-wntr"
        )
        response = ServiceStatus(
            status="ready" if database_ready and verifier_ready else "not_ready",
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
            perform_analysis(record)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return record.state

    @app.get("/api/incidents/{incident_id}/analysis", response_model=AnalysisResponse)
    def get_analysis(incident_id: UUID) -> AnalysisResponse:
        return analysis_response(incident_or_404(incident_id))

    @app.post("/api/incidents/{incident_id}/analyze/jobs")
    def queue_analysis(incident_id: UUID):
        record = incident_or_404(incident_id)
        if runtime().jobs is None:
            raise HTTPException(status_code=503, detail="worker queue unavailable")
        def job(progress, cancelled):
            progress(0.15, "loading network and artifacts")
            if cancelled.is_set():
                return {}
            response = perform_analysis(record)
            progress(0.95, "persisting analysis")
            return response.model_dump(mode="json")
        return runtime().jobs.submit(incident_id, "HYBRID_ANALYSIS", job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: UUID):
        item = runtime().jobs.get(job_id) if runtime().jobs else None
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: UUID):
        item = runtime().jobs.cancel(job_id) if runtime().jobs else None
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

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
        analysis = record.analysis
        if isinstance(analysis, IncidentAnalysisResult) and analysis.sample_result:
            sampled = analysis.sample_result
            if sampled.stop or sampled.recommended_node is None:
                raise HTTPException(status_code=409, detail=sampled.stop_reason or "sampling stopped")
            selected = next(item for item in sampled.ranked if item.node_id == sampled.recommended_node)
            recommendation = SampleRecommendation(
                node_id=sampled.recommended_node,
                expected_information_gain=selected.expected_information_gain_bits,
                alternatives=tuple(item.node_id for item in sampled.ranked[1:3]),
                runtime_mode=record.runtime_mode,
                fallback_reasons=record.fallback_reasons,
            )
        else:
            nodes = record.state.candidates.node_ids
            recommendation = SampleRecommendation(
                node_id=nodes[0], expected_information_gain=max(record.state.candidates.node_probabilities.values()),
                alternatives=nodes[1:3], runtime_mode=record.runtime_mode,
                fallback_reasons=record.fallback_reasons,
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
        if record.analysis is not None:
            perform_analysis(record)
        else:
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
        if isinstance(record.analysis, IncidentAnalysisResult):
            if not record.analysis.planning_allowed:
                raise HTTPException(
                    status_code=409,
                    detail={"reason": "PLANNING_SUPPRESSED", "codes": record.fallback_reasons},
                )
            plans = [item.plan for item in record.analysis.plan_proposals[: request.count]]
        else:
            # Explicit demo plans are structured but remain unverified until an authority is injected.
            target_nodes = record.state.candidates.node_ids
            plans = [OperationalPlan(
                incident_id=incident_id, name=f"candidate {index + 1} demo fallback",
                actions=(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id=target_nodes[index % len(target_nodes)]),),
                model_version="DEMO_FALLBACK_UNVERIFIED",
            ) for index in range(request.count)]
        record.plans.update({plan.plan_id: plan for plan in plans})
        record.state = record.state.model_copy(
            update={"status": "PLANNING", "approval_pending": False}
        )
        runtime().append_event(
            record,
            event_type="PLANS_GENERATED",
            actor="HYDRO_STRATEGIST",
            payload={
                "plan_ids": [str(plan.plan_id) for plan in plans],
                "runtime_mode": record.runtime_mode,
                "fallback_reasons": list(record.fallback_reasons),
            },
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
            verification = PlanVerifier(
                HydraulicSimulator(
                    network_path,
                    exact_simulation_budget=settings.exact_plan_simulation_limit,
                )
            ).verify(plan)
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
                "runtime_mode": record.runtime_mode,
                "fallback_reasons": list(record.fallback_reasons),
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
            analysis=(analysis_response(record).model_dump(mode="json") if record.analysis else None),
        )

    def evidence_bundle(record: IncidentRuntime) -> EvidenceBundle:
        analysis = analysis_response(record)
        leading = max(analysis.fused_belief, key=analysis.fused_belief.get) if analysis.fused_belief else None
        verification = next(
            (item for item in record.verifications.values() if item.decision == PlanDecision.VERIFIED),
            None,
        )
        rejected = next(
            (item for item in record.verifications.values() if item.decision == PlanDecision.REJECTED),
            None,
        )
        consequence = verification.consequences if verification else None
        before_after = record.analysis.before_after if isinstance(record.analysis, IncidentAnalysisResult) else None
        return EvidenceBundle(
            source_node=leading,
            source_probability=analysis.fused_belief.get(leading) if leading else None,
            candidate_region=analysis.candidate_nodes,
            candidate_coverage=sum(analysis.fused_belief.get(node, 0.0) for node in analysis.candidate_nodes),
            recommended_sample=analysis.recommended_sample,
            information_gain_bits=(
                record.analysis.sample_result.ranked[0].expected_information_gain_bits
                if isinstance(record.analysis, IncidentAnalysisResult) and record.analysis.sample_result and record.analysis.sample_result.ranked
                else None
            ),
            candidates_before=len(before_after.previous_candidates) if before_after else None,
            candidates_after=len(before_after.current_candidates) if before_after else None,
            selected_plan=str(verification.plan_id) if verification else None,
            rejected_plan=str(rejected.plan_id) if rejected else None,
            rejection_codes=rejected.rejection_codes if rejected else (),
            exposure_reduction_mg=None,
            pressure_violation_minutes=consequence.pressure_violation_minutes if consequence else None,
            service_availability=consequence.service_availability if consequence else None,
            disagreement_js=analysis.disagreement_js,
            ood_level=analysis.ood_level,
            approval_pending=record.state.approval_pending,
            abstention_reason=", ".join(record.fallback_reasons) or None,
            supporting_sensors=tuple(item.sensor_id for item in record.state.observations if not item.missing),
            removed_candidates={node: "new evidence reduced calibrated region" for node in before_after.removed_candidates} if before_after else None,
        )

    @app.post("/api/incidents/{incident_id}/workflow")
    def run_swarm_workflow(incident_id: UUID):
        record = incident_or_404(incident_id)
        if record.analysis is None:
            perform_analysis(record)
        if record.swarm is not None:
            return record.swarm.run()
        analysis = analysis_response(record)
        candidate_region = record.state.candidates.node_ids

        sentinel = HydroSentinel(inference=lambda _state: {
            "top_candidates": [
                {"node_id": node, "probability": probability}
                for node, probability in sorted(analysis.fused_belief.items(), key=lambda pair: -pair[1])
                if probability > 0
            ],
            "candidate_region": candidate_region,
            "evidence_sufficient": analysis.planning_allowed or record.runtime_mode == "DEMO_FALLBACK",
            "uncertainty": 1.0 - max(analysis.fused_belief.values()),
            "ood_level": analysis.ood_level,
        })
        scout = HydroScout(inference=lambda _state: {
            "action": "SAMPLE" if analysis.recommended_sample else "STOP",
            "node_id": analysis.recommended_sample,
            "expected_information_gain": 0.01 if analysis.recommended_sample else 0.0,
            "reason": "hybrid pipeline recommendation",
        })
        proposals = (
            [item.plan for item in record.analysis.plan_proposals]
            if isinstance(record.analysis, IncidentAnalysisResult)
            else list(record.plans.values())
        )
        if not proposals and record.runtime_mode == "DEMO_FALLBACK":
            target = candidate_region[0]
            proposals = [OperationalPlan(
                incident_id=incident_id,
                name="candidate 1 demo fallback",
                actions=(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id=target),),
                model_version="DEMO_FALLBACK_UNVERIFIED",
            )]
        strategist = HydroStrategist(inference=lambda state: {
            "plans": [{"plan": plan, "estimated_value": 0.5} for plan in proposals],
            "revision_round": int(state.get("planning_round", 0)),
        })

        if runtime().verifier is not None:
            class InjectedVerifier:
                def prescreen(self, plan):
                    return ()

                def verify(self, plan):
                    return runtime().verifier(plan, record.state)
            authority = InjectedVerifier()
            graph = nx.MultiDiGraph()
            graph.add_nodes_from(runtime().networks[record.state.network_id].metadata.get("node_ids", ()))
            network = graph
        else:
            path = runtime().store.network_path(record.state.network_id)
            if not path or wntr is None:
                raise HTTPException(status_code=503, detail="authoritative WNTR workflow unavailable")
            authority = PlanVerifier(
                HydraulicSimulator(
                    path,
                    exact_simulation_budget=settings.exact_plan_simulation_limit,
                )
            )
            network = authority.simulator.network
        record.swarm = (
            runtime().swarm_factory(sentinel, scout, strategist, authority)
            if runtime().swarm_factory
            else SwarmController(
                sentinel=sentinel, scout=scout, strategist=strategist, verifier=authority
            )
        )
        record.swarm.start(network, record.state)
        result = record.swarm.run()
        if result.selected_plan:
            record.plans[result.selected_plan.plan_id] = result.selected_plan
        if result.verification:
            record.verifications[result.verification.plan_id] = result.verification
        runtime().append_event(
            record, event_type="SWARM_WORKFLOW_UPDATED", actor="SWARM_CONTROLLER",
            payload={"fsm_state": result.state.value, "runtime_mode": analysis.runtime_mode,
                     "fallback_reasons": list(analysis.fallback_reasons)},
        )
        runtime().persist(record)
        return result

    @app.get("/api/incidents/{incident_id}/summary")
    def incident_summary(incident_id: UUID) -> dict[str, object]:
        record = incident_or_404(incident_id)
        return {
            "summary": deterministic_operational_summary(evidence_bundle(record)),
            "runtime_mode": record.runtime_mode,
            "fallback_reasons": record.fallback_reasons,
        }

    @app.get("/api/incidents/{incident_id}/explanations/{intent}")
    def incident_explanation(incident_id: UUID, intent: ExplanationIntent):
        return explain(intent, evidence_bundle(incident_or_404(incident_id)))

    @app.websocket("/ws/incidents/{incident_id}")
    async def incident_websocket(websocket: WebSocket, incident_id: UUID) -> None:
        try:
            record = runtime().incidents[incident_id]
        except KeyError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        try:
            await websocket.send_json({
                "incident_id": str(incident_id),
                "status": record.state.status,
                "progress": record.progress,
                "runtime_mode": record.runtime_mode,
                "fallback_reasons": record.fallback_reasons,
            })
            while True:
                await websocket.receive_text()
                await websocket.send_json({"status": record.state.status, "progress": record.progress})
        except WebSocketDisconnect:
            return

    frontend_candidates = (
        Path.cwd() / "frontend" / "dist",
        Path(__file__).resolve().parents[3] / "frontend" / "dist",
    )
    frontend_dist = next(
        (path for path in frontend_candidates if (path / "index.html").is_file()), None
    )
    if frontend_dist is not None:
        # Registered last so typed API/WebSocket routes always take precedence.
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="operator-console")

    return app


app = create_app(pipeline_factory=DefaultPipelineFactory())
