"""Offline FastAPI application for the typed HydroSwarm workflow."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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
from hydroswarm.calibration.conformal import CALIBRATION_SCHEMA_VERSION
from hydroswarm.inference import MODEL_VERSION, IncidentAnalysisResult
from hydroswarm.preprocessing import SensorSeries
from hydroswarm.storage import AuditEvent
from hydroswarm.networks import MAX_INP_BYTES, NetworkImportError, NetworkImporter
from hydroswarm.simulation import (
    MAXIMUM_EVALUATION_HYPOTHESES,
    HydraulicSimulator,
    PlanEvaluationContext,
    PlanVerifier,
    WeightedSourceHypothesis,
)
from hydroswarm.simulation.wrapper import IncidentSourceProfile, wntr
from hydroswarm.runtime import DefaultPipelineFactory

from .state import (
    AnalysisResponse,
    ApiSettings,
    ApprovalReceipt,
    ApprovalRequest,
    EvidenceHistoryEntryView,
    ExplanationPayload,
    IncidentExport,
    IncidentRuntime,
    IncidentView,
    NetworkLinkView,
    NetworkNodeView,
    NetworkRecord,
    NetworkValidationRequest,
    PlanGenerationRequest,
    PlanView,
    ProvenanceView,
    ReplayResponse,
    RuntimeState,
    SampleRecommendation,
    SampleRecommendationView,
    SensorHealthView,
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
        # core-issues.txt: distinguish "genuinely never analyzed" from
        # "this incident's in-memory analysis did not survive a restart"
        # (record.state.candidates, unlike record.analysis, is persisted
        # and would still be populated in the latter case) -- an explicit,
        # different error rather than one message covering both.
        if record.state.candidates is not None:
            raise HTTPException(
                status_code=409,
                detail="incident analysis did not survive a restart; POST /analyze to reanalyze",
            )
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
            # core-issues5.txt Section 8: seed the incident's own exact-
            # simulation budget from the currently configured limit -- every
            # later verification only ever spends down from here, never
            # resets to a fresh full budget.
            remaining_epanet_budget=settings.exact_plan_simulation_limit,
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
        # core-issues.txt: record.analysis (the live IncidentAnalysisResult)
        # is never persisted to disk -- only record.state.candidates is. A
        # process restart leaves record.analysis=None while
        # state.candidates still shows this incident WAS analyzed, which
        # would otherwise silently fall through to the stale-candidates
        # branch below as if that were still current. Explicitly redo the
        # analysis first rather than serve a recommendation derived from
        # evidence that predates however long ago the restart happened.
        if record.analysis is None:
            perform_analysis(record)
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
        # core-issues.txt: record.state.candidates (persisted) is checked
        # here too, not just the in-memory-only record.analysis -- a
        # restored incident that was analyzed before a restart must still
        # reanalyze on a new sample, not silently skip straight to persist
        # as if it had never been analyzed at all.
        if record.analysis is not None or record.state.candidates is not None:
            perform_analysis(record)
        else:
            runtime().persist(record)
        # core-issues5.txt Section 10 (P0 safety fix): a new sample changes
        # the incident's evidence -- any plan verified under the PRIOR
        # evidence state must stop being approvable, not remain valid
        # forever. Runs after perform_analysis/persist above so the
        # comparison uses the fully updated posterior/analysis context.
        _invalidate_stale_verifications(record)
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

    def _runtime_evaluation_context(record: IncidentRuntime) -> PlanEvaluationContext | None:
        """important-issues.txt requirement 7: at runtime there is no true
        source profile. Build a bounded (<= MAXIMUM_EVALUATION_HYPOTHESES)
        credible-hypothesis evaluation context from the incident's own
        CALIBRATED localization evidence -- never an inferred point
        estimate presented as ground truth, and never built at all when
        calibration is not yet valid (requirement 12: abstain / mark
        unavailable rather than fabricate a zero/None exposure value --
        `PlanVerifier.verify` falls back to the hydraulic-only path, whose
        `ConsequenceMetrics.exposure_evaluated=False`, whenever this
        returns None).

        Each hypothesis uses `IncidentSourceProfile`'s own governed default
        timing/strength (start_minute=0, duration_minutes=60,
        strength_mg_min=10.0 -- the same baseline profile shape
        evaluation/golden.py's fixture uses) at the candidate node: this
        codebase does not yet surface a validated per-node start-time/
        duration/relative-strength prediction to the API layer
        (core-issues4.txt's output-governance work has not promoted those
        Sentinel heads into `runtime_enabled_outputs`), so only node
        identity and its calibrated probability -- real, calibrated
        evidence -- vary between hypotheses. Record and probability are
        never presented as certain: `evaluation_provenance` on the
        resulting `PlanVerification` carries every evaluated hypothesis and
        the aggregation mode used.
        """

        candidates = record.state.candidates
        if candidates is None or not candidates.calibrated or not candidates.node_probabilities:
            return None
        ranked = sorted(candidates.node_probabilities.items(), key=lambda item: (-item[1], item[0]))
        hypotheses = tuple(
            WeightedSourceHypothesis(profile=IncidentSourceProfile(source_node_id=node), probability=probability)
            for node, probability in ranked[:MAXIMUM_EVALUATION_HYPOTHESES]
            if probability > 0.0
        )
        if not hypotheses:
            return None
        return PlanEvaluationContext(
            contamination_threshold_mg_l=record.create.contamination_threshold_mg_l,
            hypotheses=hypotheses,
        )

    def _verification_context_hash(record: IncidentRuntime) -> str:
        """core-issues5.txt Section 10 (P0 safety fix): one canonical hash
        over every behavior-critical component a plan verification depends
        on. `record.analysis.provenance_hashes` (when a real analysis has
        run) already bundles model/checkpoint, network, topology,
        signature-artifact, normalization, fusion-config, and calibration
        identity plus the incident's own evidence hash (see
        HybridInferencePipeline.analyze's own `provenance` dict) -- reused
        directly here rather than re-derived, so this can never silently
        drift out of sync with what analysis actually used. Source
        hypothesis-set identity/probabilities come from
        `record.state.candidates` directly (changes whenever a new sample
        changes the posterior); consequence-policy identity is
        approximated by the incident's own configured contamination
        threshold (the one consequence-policy parameter this API layer
        currently exposes per-incident).

        Any change to this hash between when a verification was recorded
        and when approval is attempted means that verification's context is
        no longer current -- see verify_plan/approve_plan.
        """

        payload: dict[str, Any] = {
            "network_id": record.state.network_id,
            "observations": [item.model_dump(mode="json") for item in record.state.observations],
            "candidates": (
                record.state.candidates.model_dump(mode="json")
                if record.state.candidates is not None
                else None
            ),
            "contamination_threshold_mg_l": record.create.contamination_threshold_mg_l,
        }
        if record.analysis is not None and hasattr(record.analysis, "provenance_hashes"):
            payload["analysis_provenance"] = dict(record.analysis.provenance_hashes)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _invalidate_stale_verifications(record: IncidentRuntime) -> None:
        """core-issues5.txt Section 10: after evidence/analysis changes,
        mark every existing CURRENT verification whose recorded
        context_hash no longer matches the incident's current context as
        STALE -- retained in `record.verifications` (audit history), never
        deleted, but no longer approvable. A verification with no
        context_hash at all (predates this field) is treated as stale too:
        there is nothing to prove it still matches."""

        current_hash = _verification_context_hash(record)
        for plan_id, verification in list(record.verifications.items()):
            if verification.verification_status != "CURRENT":
                continue
            if verification.context_hash == current_hash:
                continue
            record.verifications[plan_id] = verification.model_copy(
                update={"verification_status": "STALE"}
            )
            runtime().append_event(
                record,
                event_type="PLAN_VERIFICATION_STALE",
                actor="SYSTEM",
                payload={"plan_id": str(plan_id), "reason": "EVIDENCE_CHANGED"},
                simulator_version=verification.simulator_version,
            )

    @app.post(
        "/api/incidents/{incident_id}/plans/{plan_id}/verify",
        response_model=PlanVerification,
    )
    def verify_plan(incident_id: UUID, plan_id: UUID) -> PlanVerification:
        record = incident_or_404(incident_id)
        plan = plan_or_404(record, plan_id)
        exact_runs_consumed = 0
        if runtime().verifier is not None:
            verification = runtime().verifier(plan, record.state)
        else:
            network_path = runtime().store.network_path(record.state.network_id)
            if not network_path or wntr is None:
                raise HTTPException(
                    status_code=503,
                    detail="authoritative WNTR verification is unavailable",
                )
            # core-issues.txt: the exact-simulation budget belongs to the
            # incident, not to this one HydraulicSimulator instance -- a
            # fresh instance constructed on every /verify call must only
            # ever be given what this incident has not already spent,
            # never a full budget reset.
            remaining_budget = max(
                0, settings.exact_plan_simulation_limit - record.state.exact_simulations_used
            )
            if remaining_budget <= 0:
                raise HTTPException(
                    status_code=409,
                    detail="exact simulation budget exhausted for this incident",
                )
            simulator = HydraulicSimulator(network_path, exact_simulation_budget=remaining_budget)
            try:
                # important-issues.txt requirement 12: return actual
                # contamination consequences when sufficient calibrated
                # incident evidence exists (_runtime_evaluation_context
                # returns None, falling back to the hydraulic-only legacy
                # path, otherwise).
                evaluation_context = _runtime_evaluation_context(record)
                verification = PlanVerifier(simulator).verify(plan, evaluation_context)
            finally:
                exact_runs_consumed = simulator.exact_runs
        if verification.plan_id != plan_id:
            raise HTTPException(status_code=500, detail="verifier returned the wrong plan_id")
        # core-issues5.txt Section 10: stamp the context this verification
        # was computed against -- CURRENT now (evidence has not moved since
        # `record` was read above), but note that approve_plan below always
        # recomputes and compares the hash fresh rather than trusting this
        # flag alone, so a change that lands between this write and a later
        # approval attempt is still caught even if some other evidence-
        # mutating path failed to eagerly mark it STALE.
        verification = verification.model_copy(
            update={"context_hash": _verification_context_hash(record), "verification_status": "CURRENT"}
        )
        record.verifications[plan_id] = verification
        verified = verification.decision == PlanDecision.VERIFIED
        # core-issues5.txt Section 8: plans_exactly_verified counts THIS
        # plan verification attempt (one per /verify call, regardless of
        # decision) -- distinct from exact_simulations_used, which counts
        # the underlying EPANET executions it consumed (0 for a cache hit
        # or an injected/non-simulator verifier, possibly >1 for a
        # multi-hypothesis exposure-aware evaluation).
        cache_hit = bool(
            verification.evaluation_provenance and verification.evaluation_provenance.get("cache_hit")
        )
        new_exact_simulations_used = record.state.exact_simulations_used + exact_runs_consumed
        record.state = record.state.model_copy(
            update={
                "status": "APPROVAL" if verified else "PLANNING",
                "approval_pending": verified,
                "exact_simulations_used": new_exact_simulations_used,
                "plans_exactly_verified": record.state.plans_exactly_verified + 1,
                "exact_simulation_cache_hits": (
                    record.state.exact_simulation_cache_hits + (1 if cache_hit else 0)
                ),
                "remaining_epanet_budget": max(
                    0, settings.exact_plan_simulation_limit - new_exact_simulations_used
                ),
            }
        )
        runtime().append_event(
            record,
            # core-issues5.txt Section 9: distinguish a real WNTR REJECTED
            # decision from an ABSTAINED (simulator/governed failure)
            # decision in the audit trail -- both previously collapsed into
            # the same "PLAN_REJECTED" event type.
            event_type=(
                "PLAN_VERIFIED"
                if verified
                else "PLAN_VERIFICATION_FAILED"
                if verification.decision == PlanDecision.ABSTAINED
                else "PLAN_REJECTED"
            ),
            actor="HYDRO_VERIFIER",
            payload={
                "plan_id": str(plan_id),
                "decision": verification.decision.value,
                "rejection_codes": list(verification.rejection_codes),
                "abstention_reason": (
                    verification.abstention_reason.value if verification.abstention_reason else None
                ),
                "exact_simulation_count": exact_runs_consumed,
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
        # core-issues5.txt Section 10 (P0 safety fix): approval requires
        # decision == VERIFIED (checked above) AND status == CURRENT AND
        # context_hash == the incident's CURRENT context hash, recomputed
        # fresh here rather than trusting verification_status alone -- a
        # defensive re-check that catches any evidence-changing path that
        # did not (or could not) eagerly call _invalidate_stale_verifications.
        current_hash = _verification_context_hash(record)
        if verification.verification_status != "CURRENT" or verification.context_hash != current_hash:
            raise HTTPException(
                status_code=409,
                detail="verification is stale: incident evidence has changed since this plan was "
                "verified; re-verify before approval",
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
        # important-issues.txt requirement 13: populate exposure_reduction_mg
        # from the exact verified plan versus the exact NO_ACTION comparator
        # -- structural NO_ACTION identity (a plan whose only action is
        # END_PLAN), matching planning/response.py's own NO_ACTION template,
        # not a name-string convention. Only populated when BOTH the
        # selected plan's and NO_ACTION's consequences are
        # exposure_evaluated=True (requirement 12: never present a
        # hydraulic-only Pydantic default as a measured exposure reduction).
        no_action_verification = next(
            (
                item
                for plan_id, item in record.verifications.items()
                if plan_id in record.plans
                and all(action.action_type == ActionType.END_PLAN for action in record.plans[plan_id].actions)
            ),
            None,
        )
        no_action_consequence = no_action_verification.consequences if no_action_verification else None
        exposure_reduction_mg = None
        if (
            consequence is not None
            and consequence.exposure_evaluated
            and no_action_consequence is not None
            and no_action_consequence.exposure_evaluated
        ):
            exposure_reduction_mg = (
                no_action_consequence.contaminant_mass_consumed_mg - consequence.contaminant_mass_consumed_mg
            )
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
            exposure_reduction_mg=exposure_reduction_mg,
            pressure_violation_minutes=consequence.pressure_violation_minutes if consequence else None,
            service_availability=consequence.service_availability if consequence else None,
            disagreement_js=analysis.disagreement_js,
            ood_level=analysis.ood_level,
            approval_pending=record.state.approval_pending,
            abstention_reason=", ".join(record.fallback_reasons) or None,
            supporting_sensors=tuple(item.sensor_id for item in record.state.observations if not item.missing),
            removed_candidates={node: "new evidence reduced calibrated region" for node in before_after.removed_candidates} if before_after else None,
        )

    def _validate_incident_view(view: IncidentView) -> None:
        """core-issues.txt: explicit runtime validation for the /view
        response, beyond the type/range checks Pydantic already enforces
        per-field. These are cross-field invariants an assembler bug could
        violate silently -- a dangling plan/node reference would otherwise
        ship to the operator console as if it were a valid selection."""

        node_ids = {node.node_id for node in view.nodes}
        plan_ids = {item.plan.plan_id for item in view.plans}
        missing_candidate_nodes = set(view.candidates.node_ids) - node_ids
        if missing_candidate_nodes:
            raise HTTPException(
                status_code=500,
                detail=f"incident view candidates reference unknown nodes: {sorted(missing_candidate_nodes)}",
            )
        if view.selected_plan_id is not None and view.selected_plan_id not in plan_ids:
            raise HTTPException(
                status_code=500,
                detail="incident view selected_plan_id does not reference a plan in this view",
            )
        if view.recommended_plan_id is not None and view.recommended_plan_id not in plan_ids:
            raise HTTPException(
                status_code=500,
                detail="incident view recommended_plan_id does not reference a plan in this view",
            )
        if view.sample_recommendation is not None and view.sample_recommendation.node_id not in node_ids:
            raise HTTPException(
                status_code=500,
                detail="incident view sample_recommendation references an unknown node",
            )

    @app.get("/api/incidents/{incident_id}/view", response_model=IncidentView)
    def get_incident_view(incident_id: UUID) -> IncidentView:
        record = incident_or_404(incident_id)
        analysis = record.analysis
        if not isinstance(analysis, IncidentAnalysisResult):
            # DEMO_FALLBACK / not-yet-analyzed states have no verified
            # provenance (model/calibration hashes, sensor-grounded node
            # topology) to report; a complete IncidentView must never be
            # backed by unverified demo content (overnight-plan.txt Task 3.2).
            raise HTTPException(
                status_code=409,
                detail=(
                    "incident view requires a completed hybrid analysis "
                    "(FULL_HYBRID or CLASSICAL_SAFE); analyze the incident with "
                    "the hybrid pipeline configured before requesting /view"
                ),
            )
        network = runtime().networks.get(record.state.network_id)
        if network is None or "nodes" not in network.metadata or "links" not in network.metadata:
            raise HTTPException(
                status_code=503,
                detail=(
                    "network topology metadata is incomplete for a live incident "
                    "view; import the network via /api/networks/import rather than "
                    "manual /validate"
                ),
            )

        candidate_nodes = set(analysis.conformal_candidate_nodes)
        latest_observation: dict[str, SensorObservation] = {}
        for item in record.state.observations:
            current = latest_observation.get(item.node_id)
            if current is None or item.observed_at >= current.observed_at:
                latest_observation[item.node_id] = item

        nodes = tuple(
            NetworkNodeView(
                node_id=entry["node_id"],
                node_type=entry["node_type"],
                elevation_m=entry["elevation_m"],
                coordinates=tuple(entry["coordinates"][:2]) if len(entry["coordinates"]) >= 2 else (0.0, 0.0),
                probability=analysis.fused_belief.get(entry["node_id"], 0.0),
                concentration_mg_l=(
                    latest_observation[entry["node_id"]].concentration_mg_l
                    if entry["node_id"] in latest_observation
                    else None
                ),
                candidate=entry["node_id"] in candidate_nodes,
            )
            for entry in network.metadata["nodes"]
        )
        links = tuple(
            NetworkLinkView(
                link_id=entry["link_id"],
                link_type=entry["link_type"],
                start_node=entry["start_node"],
                end_node=entry["end_node"],
            )
            for entry in network.metadata["links"]
        )
        sensor_health = tuple(
            SensorHealthView(
                sensor_id=observation.sensor_id,
                node_id=observation.node_id,
                health=(
                    "MISSING" if observation.missing
                    else "DRIFT" if (observation.drift_flag or observation.frozen_flag)
                    else "HEALTHY"
                ),
                quality=observation.quality,
                observed_at=observation.observed_at,
                received_at=observation.received_at,
                pressure_m=observation.pressure_m,
                concentration_mg_l=observation.concentration_mg_l,
            )
            for observation in latest_observation.values()
        )

        sample_recommendation: SampleRecommendationView | None = None
        sampled = analysis.sample_result
        if sampled is not None and not sampled.stop and sampled.recommended_node is not None:
            selected = next(
                (item for item in sampled.ranked if item.node_id == sampled.recommended_node), None
            )
            if selected is not None:
                sample_recommendation = SampleRecommendationView(
                    node_id=sampled.recommended_node,
                    expected_information_gain=selected.expected_information_gain_bits,
                    alternatives=tuple(item.node_id for item in sampled.ranked[1:3]),
                )

        evidence_history = tuple(
            EvidenceHistoryEntryView(
                round_index=snapshot.round_index,
                observation_count=snapshot.observation_count,
                valid_concentration_count=snapshot.valid_concentration_count,
                sensor_nodes=snapshot.sensor_nodes,
                evidence_hash=snapshot.evidence_hash,
            )
            for snapshot in analysis.evidence_history
        )

        plans = tuple(
            PlanView(plan=plan, verification=record.verifications.get(plan.plan_id))
            for plan in record.plans.values()
        )
        audit_events = tuple(runtime().ledger.events(incident_id))
        # "Selected" means an operator actually approved it -- the audit
        # ledger's PLAN_APPROVED payload is the sole authoritative record of
        # which plan_id that was; a VERIFIED decision alone only means a
        # candidate is eligible for approval, not that it was chosen.
        selected_plan_id = next(
            (
                UUID(str(event.payload["plan_id"]))
                for event in reversed(audit_events)
                if event.event_type == "PLAN_APPROVED"
            ),
            None,
        )
        recommended_plan_id = (
            analysis.plan_proposals[0].plan.plan_id if analysis.plan_proposals else None
        )
        counterfactual_consequences = {
            str(plan_id): item.consequences
            for plan_id, item in record.verifications.items()
            if item.consequences is not None
        }

        bundle = evidence_bundle(record)
        explanations = []
        for intent in ExplanationIntent:
            grounded = explain(intent, bundle)
            explanations.append(ExplanationPayload(
                intent=intent.value,
                text=grounded.text,
                facts=dict(grounded.facts),
                limitations=grounded.limitations,
            ))
        explanations = tuple(explanations)

        latest_verification = next(reversed(record.verifications.values()), None)
        # core-issues.txt: fail closed rather than returning empty
        # provenance hashes. HybridInferencePipeline.analyze() always
        # populates network/feature_schema/model in provenance_hashes when
        # it runs at all (see hydroswarm.inference.pipeline); a missing key
        # here means something is fundamentally wrong with this analysis
        # result, not that "" is a legitimate hash an operator should be
        # asked to trust. calibration is a genuine exception: "none" is
        # the pipeline's own explicit, meaningful sentinel for "no
        # calibration artifact was configured", not a silent omission.
        required_provenance = {
            "network": analysis.provenance_hashes.get("network"),
            "feature_schema": analysis.provenance_hashes.get("feature_schema"),
            "model": analysis.provenance_hashes.get("model"),
        }
        missing_provenance = [key for key, value in required_provenance.items() if not value]
        if missing_provenance:
            raise HTTPException(
                status_code=500,
                detail=f"incident view is missing required provenance hashes: {missing_provenance}",
            )
        provenance = ProvenanceView(
            network_hash=required_provenance["network"],
            feature_schema_hash=required_provenance["feature_schema"],
            model_version=MODEL_VERSION,
            model_checkpoint_hash=required_provenance["model"],
            calibration_version=CALIBRATION_SCHEMA_VERSION,
            calibration_hash=analysis.provenance_hashes.get("calibration", "none"),
            simulator=latest_verification.simulator if latest_verification else None,
            simulator_version=latest_verification.simulator_version if latest_verification else None,
        )

        view = IncidentView(
            incident_id=record.state.incident_id,
            network_id=record.state.network_id,
            runtime_mode=analysis.runtime_mode.value,
            controller_state=record.state.status,
            generated_at=utc_now(),
            provenance=provenance,
            candidates=record.state.candidates,
            disagreement_js=analysis.fusion_diagnostics.disagreement_js if analysis.fusion_diagnostics else None,
            ood_level=analysis.ood_level.value,
            calibration_alpha=analysis.calibration_alpha,
            nodes=nodes,
            links=links,
            sensor_health=sensor_health,
            sample_recommendation=sample_recommendation,
            evidence_history=evidence_history,
            plans=plans,
            selected_plan_id=selected_plan_id,
            recommended_plan_id=recommended_plan_id,
            counterfactual_consequences=counterfactual_consequences,
            explanations=explanations,
            audit_events=audit_events,
            runtime_metrics_ms=dict(analysis.latencies_ms),
        )
        _validate_incident_view(view)
        return view

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

        real_simulator: HydraulicSimulator | None = None
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
            # core-issues.txt: same incident-level budget accounting as
            # /verify -- a fresh HydraulicSimulator here must only ever be
            # given what this incident has not already spent.
            remaining_budget = max(
                0, settings.exact_plan_simulation_limit - record.state.exact_simulations_used
            )
            if remaining_budget <= 0:
                raise HTTPException(
                    status_code=409,
                    detail="exact simulation budget exhausted for this incident",
                )
            real_simulator = HydraulicSimulator(path, exact_simulation_budget=remaining_budget)
            authority = PlanVerifier(real_simulator)
            network = authority.simulator.network
        record.swarm = (
            runtime().swarm_factory(sentinel, scout, strategist, authority)
            if runtime().swarm_factory
            else SwarmController(
                sentinel=sentinel, scout=scout, strategist=strategist, verifier=authority
            )
        )
        record.swarm.start(network, record.state)
        try:
            result = record.swarm.run()
        finally:
            # core-issues5.txt Section 8/9: persisted in `finally` so a
            # swarm run that raises or times out mid-way still records
            # whatever exact EPANET executions/plan-verification attempts
            # it actually consumed -- never silently dropped. Reads
            # record.swarm.result() (the swarm's own live internal state,
            # updated incrementally as it steps) rather than the `result`
            # local above, which is never assigned if `.run()` itself
            # raised.
            if real_simulator is not None:
                new_exact_simulations_used = (
                    record.state.exact_simulations_used + real_simulator.exact_runs
                )
                record.state = record.state.model_copy(
                    update={
                        "exact_simulations_used": new_exact_simulations_used,
                        "plans_exactly_verified": (
                            record.state.plans_exactly_verified
                            + record.swarm.result().plans_exactly_verified
                        ),
                        "remaining_epanet_budget": max(
                            0, settings.exact_plan_simulation_limit - new_exact_simulations_used
                        ),
                    }
                )
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
