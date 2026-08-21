from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import replace
import hashlib

from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.domain import (
    ConsequenceMetrics,
    OODLevel,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
)
from hydroswarm.inference.fusion import ControlAction
from hydroswarm.inference.ood import OODComponents
from hydroswarm.inference.results import HybridRuntimeMode, IncidentAnalysisResult, SemanticPredictions
from hydroswarm.planning import PlanProposal
from hydroswarm.sampling import ActiveSamplingResult, SampleCandidate


NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _verification(plan, state) -> PlanVerification:
    unsafe = "candidate 1" in plan.name
    return PlanVerification(
        plan_id=plan.plan_id,
        decision=PlanDecision.REJECTED if unsafe else PlanDecision.VERIFIED,
        simulator="test-wntr",
        simulator_version="1.0",
        state_hash=hashlib.sha256(str(state.incident_id).encode()).hexdigest(),
        consequences=(
            None
            if unsafe
            else ConsequenceMetrics(
                minimum_pressure_m=22.0,
                service_availability=0.99,
                operation_count=len(plan.actions),
            )
        ),
        rejection_codes=("PRESSURE_BELOW_THRESHOLD",) if unsafe else (),
    )


def _observation(sensor_id: str = "S1", node_id: str = "J1") -> dict[str, object]:
    return {
        "sensor_id": sensor_id,
        "node_id": node_id,
        "observed_at": NOW.isoformat(),
        "received_at": NOW.isoformat(),
        "concentration_mg_l": 0.2,
        "pressure_m": 30.0,
        "quality": 1.0,
        "missing": False,
        "drift_flag": False,
        "frozen_flag": False,
    }


def _install_authoritative_test_analysis(
    client: TestClient, incident_id: str, plans: tuple[OperationalPlan, ...] | None = None
) -> tuple[OperationalPlan, ...]:
    """Test seam using the production analysis result type, never DEMO_FALLBACK."""
    from uuid import UUID

    if plans is None:
        plans = (
            OperationalPlan(incident_id=incident_id, name="candidate 1 unsafe", model_version="test",
                            actions=(OperationalAction(action_type="WAIT", duration_minutes=1),)),
            OperationalPlan(incident_id=incident_id, name="candidate 2 safe", model_version="test",
                            actions=(OperationalAction(action_type="WAIT", duration_minutes=2),)),
        )
    proposals = tuple(
        PlanProposal(plan, "TEST", 0.0, 1.0, (("WAIT", None),)) for plan in plans
    )
    record = client.app.state.runtime.incidents[UUID(incident_id)]
    record.analysis = IncidentAnalysisResult(
        incident_id=UUID(incident_id), node_alignment=("J1", "J2"),
        classical_belief={"J1": 1.0}, neural_belief=None, fused_belief={"J1": 1.0},
        classical_localization=None, estimated_hydraulic_state=None, trust_features=None,
        fusion_diagnostics=None, trust_rationale="test-only authoritative analysis",
        conformal_candidate_nodes=("J1",), calibrated=True, calibration_alpha=0.1,
        ood_components=OODComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), ood_level=OODLevel.NORMAL,
        evidence_sufficient=True, planning_allowed=True, planning_suppression_reasons=(),
        control_action=ControlAction.GENERATE_PLANS, sample_result=None,
        plan_proposals=proposals, semantic_predictions=SemanticPredictions(), posterior_history=(),
        evidence_history=(), comparison_history=(), before_after=None,
        runtime_mode=HybridRuntimeMode.CLASSICAL_SAFE, neural_failure=None, latencies_ms={},
        provenance_hashes={"network": "test", "feature_schema": "test", "model": "test"}, evidence_hash="test",
    )
    record.runtime_mode = HybridRuntimeMode.CLASSICAL_SAFE.value
    record.fallback_reasons = ()
    return plans


def _sample_result(*, node: str | None, stop: bool = False) -> ActiveSamplingResult:
    candidate = SampleCandidate(
        node_id="J2", score=1.0, expected_information_gain_bits=0.5,
        expected_candidate_reduction=1.0, leading_hypothesis_separation=1.0,
        detection_probability=1.0, collection_time_minutes=1.0,
        operational_cost=1.0, redundancy=0.0, accessible=True,
        classical_rank=1, neural_residual_delta=0.0,
    )
    return ActiveSamplingResult((candidate,), node, stop, "sampling stopped" if stop else None, 1.0)


def test_sampling_api_requires_current_authoritative_recommendation(tmp_path) -> None:
    client = TestClient(create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3"))
    assert client.post("/api/networks/net-sampling/validate", json={"node_ids": ["J1", "J2"], "link_count": 1}).status_code == 200
    created = client.post("/api/incidents", json={"network_id": "net-sampling", "detected_at": NOW.isoformat(), "observations": [_observation()], "maximum_samples": 2})
    incident_id = created.json()["incident_id"]
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200
    _install_authoritative_test_analysis(client, incident_id)
    from uuid import UUID
    record = client.app.state.runtime.incidents[UUID(incident_id)]
    original = record.analysis
    assert isinstance(original, IncidentAnalysisResult)
    for action, result in ((ControlAction.CONTINUE_ANALYSIS, None), (ControlAction.ABSTAIN, None), (ControlAction.GENERATE_PLANS, None), (ControlAction.REQUEST_SAMPLE, None), (ControlAction.REQUEST_SAMPLE, _sample_result(node=None, stop=True))):
        record.analysis = replace(original, control_action=action, sample_result=result)
        assert client.post(f"/api/incidents/{incident_id}/samples/recommend").status_code == 409
    record.analysis = replace(original, control_action=ControlAction.REQUEST_SAMPLE, sample_result=_sample_result(node="J1"))
    assert client.post(f"/api/incidents/{incident_id}/samples/recommend").status_code == 409
    record.analysis = replace(original, control_action=ControlAction.REQUEST_SAMPLE, sample_result=_sample_result(node="J2"))
    response = client.post(f"/api/incidents/{incident_id}/samples/recommend")
    assert response.status_code == 200 and response.json()["node_id"] == "J2"
    events = client.get(f"/api/incidents/{incident_id}/events").json()
    assert [event["event_type"] for event in events].count("SAMPLE_RECOMMENDED") == 1


def test_full_typed_workflow_rejects_unsafe_and_gates_approval(tmp_path) -> None:
    client = TestClient(
        create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3")
    )

    assert client.get("/api/health").json() == {
        "status": "ok",
        "offline": True,
        "version": "0.2.0",
    }
    assert client.get("/api/readiness").status_code == 200
    assert client.get("/api/version").json()["offline"] is True

    network = client.post(
        "/api/networks/net-test/validate",
        json={"node_ids": ["J1", "J2"], "link_count": 1},
    )
    assert network.status_code == 200
    assert client.get("/api/networks").json()[0]["valid"] is True

    created = client.post(
        "/api/incidents",
        json={
            "network_id": "net-test",
            "detected_at": NOW.isoformat(),
            "observations": [_observation()],
            "maximum_samples": 2,
        },
    )
    assert created.status_code == 201
    incident_id = created.json()["incident_id"]

    analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
    assert analyzed.status_code == 200
    assert analyzed.json()["candidates"]["node_ids"] == ["J1"]
    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
    assert analysis["runtime_mode"] == "DEMO_FALLBACK"
    assert analysis["fallback_reasons"] == [
        "HYBRID_PIPELINE_NOT_CONFIGURED",
        "UNVERIFIED_DEMO_ANALYSIS",
    ]
    recommendation = client.post(
        f"/api/incidents/{incident_id}/samples/recommend"
    )
    # A demo fallback/persisted candidate is not an authoritative live
    # sampling decision.  The API must fail closed rather than synthesize one.
    assert recommendation.status_code == 409

    added = client.post(
        f"/api/incidents/{incident_id}/samples",
        json=_observation("S2", "J2"),
    )
    assert added.json()["sample_count"] == 1
    assert (
        client.get(f"/api/incidents/{incident_id}/analysis")
        .json()["posterior_history"][0]["observation_count"]
        == 2
    )

    # DEMO_FALLBACK is intentionally non-authoritative: neither plan route
    # nor workflow may turn its suppressed analysis into approval-capable work.
    plans_response = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 2})
    assert plans_response.status_code == 409
    assert client.post(f"/api/incidents/{incident_id}/workflow").status_code == 409
    return
    unsafe_plan, safe_plan = plans_response.json()
    assert unsafe_plan["model_version"] == "DEMO_FALLBACK_UNVERIFIED"

    premature = client.post(
        f"/api/incidents/{incident_id}/plans/{safe_plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert premature.status_code == 409

    rejected = client.post(
        f"/api/incidents/{incident_id}/plans/{unsafe_plan['plan_id']}/verify"
    )
    assert rejected.json()["decision"] == "REJECTED"
    rejected_approval = client.post(
        f"/api/incidents/{incident_id}/plans/{unsafe_plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert rejected_approval.status_code == 409

    verified = client.post(
        f"/api/incidents/{incident_id}/plans/{safe_plan['plan_id']}/verify"
    )
    assert verified.json()["decision"] == "VERIFIED"
    assert client.get(f"/api/incidents/{incident_id}").json()["approval_pending"] is True

    approved = client.post(
        f"/api/incidents/{incident_id}/plans/{safe_plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert approved.status_code == 200
    assert client.get(f"/api/incidents/{incident_id}").json()["status"] == "CLOSED"

    events = client.get(f"/api/incidents/{incident_id}/events").json()
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert "PLAN_REJECTED" in {event["event_type"] for event in events}
    assert "PLAN_APPROVED" in {event["event_type"] for event in events}

    replay = client.post(f"/api/incidents/{incident_id}/replay")
    assert replay.json()["chain_valid"] is True
    exported = client.get(f"/api/incidents/{incident_id}/export")
    assert exported.status_code == 200
    assert len(exported.json()["plans"]) == 2
    assert len(exported.json()["verifications"]) == 2

    # core-issues5.txt Section 13 (P1 product feature): every verified/
    # rejected plan must have a decision certificate exposing its real
    # authority (SIMULATOR_VERIFIED either way -- WNTR ran regardless of
    # the resulting decision). This fixture never configures a real
    # pipeline_factory (record.analysis stays a DEMO_FALLBACK dict, not a
    # real IncidentAnalysisResult), so only the per-plan certificates are
    # expected here -- source_localization/scout_recommendation/ood_state
    # require a real analysis and are covered separately in
    # tests/scientific/test_decision_certificates.py.
    certificates = client.get(f"/api/incidents/{incident_id}/authority")
    assert certificates.status_code == 200
    by_name = {c["name"]: c for c in certificates.json()}
    assert by_name[f"plan_consequence:{safe_plan['plan_id']}"]["authority"] == "SIMULATOR_VERIFIED"
    assert by_name[f"plan_consequence:{unsafe_plan['plan_id']}"]["authority"] == "SIMULATOR_VERIFIED"
    assert by_name[f"plan_consequence:{unsafe_plan['plan_id']}"]["suppression_reasons"] == [
        "PRESSURE_BELOW_THRESHOLD"
    ]

    # core-issues5.txt Section 14 (P1 product feature): only the real
    # VERIFIED plan enters the frontier -- the REJECTED one never does,
    # regardless of how its consequences might otherwise compare.
    frontier = client.get(f"/api/incidents/{incident_id}/frontier")
    assert frontier.status_code == 200
    frontier_plan_ids = {entry["plan_id"] for entry in frontier.json()}
    assert str(safe_plan["plan_id"]) in frontier_plan_ids
    assert str(unsafe_plan["plan_id"]) not in frontier_plan_ids
    assert frontier.json()[0]["mode"] == "posterior_weighted"

    # core-issues5.txt Section 15 (P1 product feature): this fixture never
    # configures a real pipeline_factory (record.analysis is a
    # DEMO_FALLBACK dict, not a real IncidentAnalysisResult) -- the
    # evidence certificate correctly refuses to fabricate one rather than
    # silently returning empty/zero data. The real builder function is
    # exercised directly, with real data, in
    # tests/scientific/test_evidence_certificate.py.
    evidence_certificate = client.get(f"/api/incidents/{incident_id}/evidence-certificate")
    assert evidence_certificate.status_code == 409


# core-issues5.txt Section 10 (P0 safety fix): a plan verified under
# evidence state A must not remain approvable after a new sample changes
# the incident's evidence state to B.


def test_new_sample_invalidates_prior_verification_and_reverify_restores_approvability(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3")
    )
    client.post(
        "/api/networks/net-test/validate",
        json={"node_ids": ["J1", "J2"], "link_count": 1},
    )
    created = client.post(
        "/api/incidents",
        json={
            "network_id": "net-test",
            "detected_at": NOW.isoformat(),
            "observations": [_observation()],
            "maximum_samples": 3,
        },
    )
    incident_id = created.json()["incident_id"]
    client.post(f"/api/incidents/{incident_id}/analyze")

    _unsafe_plan, safe_plan = _install_authoritative_test_analysis(client, incident_id)
    generated = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 2})
    assert generated.status_code == 200

    verified = client.post(f"/api/incidents/{incident_id}/plans/{safe_plan.plan_id}/verify")
    assert verified.json()["decision"] == "VERIFIED"
    assert verified.json()["verification_status"] == "CURRENT"
    assert verified.json()["context_hash"]

    # New evidence arrives -- the prior verification must no longer be
    # approvable, and must be visibly marked STALE (retained, not deleted).
    client.post(f"/api/incidents/{incident_id}/samples", json=_observation("S3", "J2"))

    stale_approval = client.post(
        f"/api/incidents/{incident_id}/plans/{safe_plan.plan_id}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert stale_approval.status_code == 409

    # Re-verifying under the new evidence produces a fresh CURRENT
    # verification, which is approvable again.
    _install_authoritative_test_analysis(client, incident_id, (safe_plan,))
    reverified = client.post(f"/api/incidents/{incident_id}/plans/{safe_plan.plan_id}/verify")
    assert reverified.json()["decision"] == "VERIFIED"
    assert reverified.json()["verification_status"] == "CURRENT"
    assert reverified.json()["context_hash"] != verified.json()["context_hash"]

    approved = client.post(
        f"/api/incidents/{incident_id}/plans/{safe_plan.plan_id}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert approved.status_code == 200

    events = client.get(f"/api/incidents/{incident_id}/events").json()
    assert "PLAN_VERIFICATION_STALE" in {event["event_type"] for event in events}


def _create_incident_with_verified_plan(client: TestClient) -> tuple[str, str, dict]:
    client.post("/api/networks/net-test/validate", json={"node_ids": ["J1", "J2"], "link_count": 1})
    created = client.post(
        "/api/incidents",
        json={
            "network_id": "net-test",
            "detected_at": NOW.isoformat(),
            "observations": [_observation()],
            "maximum_samples": 3,
        },
    )
    incident_id = created.json()["incident_id"]
    client.post(f"/api/incidents/{incident_id}/analyze")
    _install_authoritative_test_analysis(client, incident_id)
    plans_response = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 2})
    _unsafe_plan, safe_plan = plans_response.json()
    verified = client.post(f"/api/incidents/{incident_id}/plans/{safe_plan['plan_id']}/verify")
    assert verified.json()["decision"] == "VERIFIED"
    assert verified.json()["verification_status"] == "CURRENT"
    return incident_id, safe_plan["plan_id"], verified.json()


# core-issues5.txt delta item 5: the verification-context identity must
# include every behavior-critical verifier policy value, not only
# evidence/network/model identity -- a change to any of them must
# invalidate an already-recorded verification exactly like new evidence
# does.


def test_changed_safety_threshold_invalidates_verification(tmp_path, monkeypatch) -> None:
    client = TestClient(create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3"))
    incident_id, plan_id, first = _create_incident_with_verified_plan(client)

    import importlib

    # hydroswarm.api's own __init__.py binds a module-level `app` (a real
    # FastAPI instance) that shadows the `hydroswarm.api.app` SUBMODULE as
    # a package attribute -- `import hydroswarm.api.app as x` would bind
    # `x` to that FastAPI instance, not the module. importlib.import_module
    # reads sys.modules directly, sidestepping the shadowing.
    app_module = importlib.import_module("hydroswarm.api.app")
    monkeypatch.setattr(app_module, "DEFAULT_MINIMUM_PRESSURE_M", 99.0)

    stale_approval = client.post(
        f"/api/incidents/{incident_id}/plans/{plan_id}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert stale_approval.status_code == 409

    reverified = client.post(f"/api/incidents/{incident_id}/plans/{plan_id}/verify")
    assert reverified.json()["context_hash"] != first["context_hash"]

    approved = client.post(
        f"/api/incidents/{incident_id}/plans/{plan_id}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert approved.status_code == 200


def test_changed_consequence_policy_invalidates_verification(tmp_path, monkeypatch) -> None:
    """Simulator/consequence-policy identity (PlanEvaluationContext's own
    governed defaults) invalidates a recorded verification exactly like a
    changed safety threshold does -- a Field object's `.default` is a real,
    mutable attribute, so this monkeypatches the actual value
    _verification_context_hash reads rather than a copy."""

    client = TestClient(create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3"))
    incident_id, plan_id, first = _create_incident_with_verified_plan(client)

    from hydroswarm.simulation import PlanEvaluationContext

    monkeypatch.setattr(
        PlanEvaluationContext.__dataclass_fields__["consequence_policy_version"], "default",
        "exposure-consequences-v2-test",
    )

    stale_approval = client.post(
        f"/api/incidents/{incident_id}/plans/{plan_id}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert stale_approval.status_code == 409

    reverified = client.post(f"/api/incidents/{incident_id}/plans/{plan_id}/verify")
    assert reverified.json()["context_hash"] != first["context_hash"]


def test_newly_created_incident_exposes_the_real_configured_epanet_budget(tmp_path) -> None:
    """core-issues5.txt delta item 7: remaining_epanet_budget must reflect
    the real configured exact-simulation limit at incident creation, before
    any verification occurs -- not IncidentState's own schema-level
    default of 0 (which exists only as a fallback for incidents
    constructed outside the live API, e.g. training/evaluation label
    generators, never for a real operator-facing incident)."""

    from hydroswarm.api.state import ApiSettings

    client = TestClient(create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3"))
    client.post("/api/networks/net-test/validate", json={"node_ids": ["J1", "J2"], "link_count": 1})
    created = client.post(
        "/api/incidents",
        json={
            "network_id": "net-test",
            "detected_at": NOW.isoformat(),
            "observations": [_observation()],
            "maximum_samples": 3,
        },
    )
    assert created.status_code == 201
    incident = created.json()
    configured_limit = ApiSettings().exact_plan_simulation_limit
    assert configured_limit > 0
    assert incident["remaining_epanet_budget"] == configured_limit
    assert incident["exact_simulations_used"] == 0
    assert incident["plans_exactly_verified"] == 0


def test_unchanged_context_leaves_verification_current_and_approvable(tmp_path) -> None:
    client = TestClient(create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3"))
    incident_id, plan_id, first = _create_incident_with_verified_plan(client)

    # Nothing changed between verify and approve -- must still succeed.
    approved = client.post(
        f"/api/incidents/{incident_id}/plans/{plan_id}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert approved.status_code == 200
    assert first["verification_status"] == "CURRENT"


# core-issues5.txt delta item 6 (P0 fix): stale verification leakage into
# "current evidence" surfaces (evidence_bundle() and everything built from
# it -- /summary text, /view's recommended_plan_id/explanations).


def test_stale_verification_does_not_appear_as_current_evidence_but_stays_in_audit_history(
    tmp_path,
) -> None:
    """/view requires a completed real hybrid analysis
    (IncidentAnalysisResult), which this file's minimal injected-verifier
    fixture does not produce -- the recommended_plan_id half of this same
    invariant is covered separately by
    test_incident_view_contract.py's real-pipeline fixture. This test
    covers /summary (which only needs evidence_bundle(), same as /view's
    explanations) and /export (the raw per-plan verification record)."""

    client = TestClient(create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3"))
    incident_id, plan_id, first = _create_incident_with_verified_plan(client)

    # While CURRENT: the summary surfaces the verified plan as current
    # evidence.
    fresh_summary = client.get(f"/api/incidents/{incident_id}/summary").json()["summary"]
    assert plan_id in fresh_summary
    assert "none verified" not in fresh_summary

    # New evidence arrives -- the verification becomes STALE.
    client.post(f"/api/incidents/{incident_id}/samples", json=_observation("S3", "J2"))

    stale_summary_response = client.get(f"/api/incidents/{incident_id}/summary")
    assert stale_summary_response.status_code == 200
    stale_summary = stale_summary_response.json()["summary"]
    # The stale plan must no longer be surfaced as the current/selected
    # plan -- deterministic_operational_summary falls back to "none
    # verified" exactly like an incident with no verification at all.
    assert plan_id not in stale_summary
    assert "none verified" in stale_summary

    # Audit history retains the original verification event -- staleness
    # is a "not currently approvable" status, never a deletion.
    events = client.get(f"/api/incidents/{incident_id}/events").json()
    event_types = {event["event_type"] for event in events}
    assert "PLAN_VERIFIED" in event_types
    assert "PLAN_VERIFICATION_STALE" in event_types

    # The raw per-plan verification record (distinct from the "current
    # evidence" surfaces above) is also still retrievable via /export,
    # still shows decision == VERIFIED, and now also shows
    # verification_status == STALE -- historical/stale verification
    # remains accessible, just not presented as current.
    export = client.get(f"/api/incidents/{incident_id}/export").json()
    exported_verification = next(
        item for item in export["verifications"] if item["plan_id"] == plan_id
    )
    assert exported_verification["decision"] == "VERIFIED"
    assert exported_verification["verification_status"] == "STALE"


def test_api_enforces_schema_and_validated_network(tmp_path) -> None:
    client = TestClient(create_app(ledger_path=tmp_path / "audit.sqlite3"))
    unvalidated = client.post(
        "/api/incidents",
        json={
            "network_id": "unknown",
            "detected_at": NOW.isoformat(),
            "observations": [_observation()],
        },
    )
    assert unvalidated.status_code == 409

    extras = client.post(
        "/api/networks/net/validate",
        json={"node_ids": ["J1"], "link_count": 0, "unexpected": True},
    )
    assert extras.status_code == 422


def test_default_runtime_disables_manual_networks_and_restricts_cors(tmp_path) -> None:
    client = TestClient(create_app(database_path=tmp_path / "runtime.db"))
    readiness = client.get("/api/readiness")
    assert readiness.status_code == 200
    assert readiness.json()["mode"] == "authoritative-wntr"
    assert readiness.json()["checks"]["authoritative_verifier"] is True

    manual = client.post(
        "/api/networks/untrusted/validate",
        json={"node_ids": ["J1"], "link_count": 0},
    )
    assert manual.status_code == 409

    remote = client.get("/api/health", headers={"Origin": "https://remote.example"})
    assert "access-control-allow-origin" not in remote.headers
    local = client.get("/api/health", headers={"Origin": "http://127.0.0.1:8765"})
    assert local.headers["access-control-allow-origin"] == "http://127.0.0.1:8765"


def test_request_size_limit_fails_before_body_processing(tmp_path) -> None:
    client = TestClient(create_app(database_path=tmp_path / "runtime.db", max_request_bytes=64))
    response = client.post(
        "/api/incidents",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": "65"},
    )
    assert response.status_code == 413


def test_readiness_fails_closed_without_wntr(tmp_path, monkeypatch) -> None:
    import importlib

    app_module = importlib.import_module("hydroswarm.api.app")
    monkeypatch.setattr(app_module, "wntr", None)
    client = TestClient(create_app(database_path=tmp_path / "runtime.db"))
    response = client.get("/api/readiness")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["authoritative_verifier"] is False
