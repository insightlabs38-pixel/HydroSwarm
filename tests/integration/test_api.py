from __future__ import annotations

from datetime import UTC, datetime
import hashlib

from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.domain import (
    ConsequenceMetrics,
    PlanDecision,
    PlanVerification,
)


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


def test_full_typed_workflow_rejects_unsafe_and_gates_approval(tmp_path) -> None:
    client = TestClient(
        create_app(verifier=_verification, ledger_path=tmp_path / "audit.sqlite3")
    )

    assert client.get("/api/health").json() == {
        "status": "ok",
        "offline": True,
        "version": "0.1.0",
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
    assert recommendation.json()["node_id"] == "J1"
    assert recommendation.json()["runtime_mode"] == "DEMO_FALLBACK"

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

    plans_response = client.post(
        f"/api/incidents/{incident_id}/plans/generate", json={"count": 2}
    )
    assert plans_response.status_code == 200
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

    plans_response = client.post(
        f"/api/incidents/{incident_id}/plans/generate", json={"count": 2}
    )
    _unsafe_plan, safe_plan = plans_response.json()

    verified = client.post(f"/api/incidents/{incident_id}/plans/{safe_plan['plan_id']}/verify")
    assert verified.json()["decision"] == "VERIFIED"
    assert verified.json()["verification_status"] == "CURRENT"
    assert verified.json()["context_hash"]

    # New evidence arrives -- the prior verification must no longer be
    # approvable, and must be visibly marked STALE (retained, not deleted).
    client.post(f"/api/incidents/{incident_id}/samples", json=_observation("S3", "J2"))

    stale_approval = client.post(
        f"/api/incidents/{incident_id}/plans/{safe_plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert stale_approval.status_code == 409

    # Re-verifying under the new evidence produces a fresh CURRENT
    # verification, which is approvable again.
    reverified = client.post(f"/api/incidents/{incident_id}/plans/{safe_plan['plan_id']}/verify")
    assert reverified.json()["decision"] == "VERIFIED"
    assert reverified.json()["verification_status"] == "CURRENT"
    assert reverified.json()["context_hash"] != verified.json()["context_hash"]

    approved = client.post(
        f"/api/incidents/{incident_id}/plans/{safe_plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert approved.status_code == 200

    events = client.get(f"/api/incidents/{incident_id}/events").json()
    assert "PLAN_VERIFICATION_STALE" in {event["event_type"] for event in events}


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
