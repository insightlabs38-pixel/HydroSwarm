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
    recommendation = client.post(
        f"/api/incidents/{incident_id}/samples/recommend"
    )
    assert recommendation.json()["node_id"] == "J1"

    added = client.post(
        f"/api/incidents/{incident_id}/samples",
        json=_observation("S2", "J2"),
    )
    assert added.json()["sample_count"] == 1

    plans_response = client.post(
        f"/api/incidents/{incident_id}/plans/generate", json={"count": 2}
    )
    assert plans_response.status_code == 200
    unsafe_plan, safe_plan = plans_response.json()

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

