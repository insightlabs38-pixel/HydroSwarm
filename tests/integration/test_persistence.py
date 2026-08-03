from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.domain import ConsequenceMetrics, PlanDecision, PlanVerification


NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _verifier(plan, state) -> PlanVerification:
    return PlanVerification(
        plan_id=plan.plan_id,
        decision=PlanDecision.VERIFIED,
        simulator="test-authoritative-verifier",
        simulator_version="1.0",
        state_hash=hashlib.sha256(str(state.incident_id).encode()).hexdigest(),
        consequences=ConsequenceMetrics(
            minimum_pressure_m=21.0,
            service_availability=0.98,
            operation_count=len(plan.actions),
        ),
    )


def _incident_payload() -> dict:
    return {
        "network_id": "persisted-network",
        "detected_at": NOW.isoformat(),
        "observations": [
            {
                "sensor_id": "S1",
                "node_id": "J1",
                "observed_at": NOW.isoformat(),
                "received_at": NOW.isoformat(),
                "concentration_mg_l": 0.2,
                "pressure_m": 25.0,
            }
        ],
    }


def test_runtime_reconstructs_complete_scenario_from_sqlite(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    first = TestClient(create_app(verifier=_verifier, database_path=database))
    assert first.post(
        "/api/networks/persisted-network/validate",
        json={"node_ids": ["J1", "J2"], "link_count": 1},
    ).status_code == 200
    created = first.post("/api/incidents", json=_incident_payload())
    incident_id = created.json()["incident_id"]
    first.post(f"/api/incidents/{incident_id}/analyze")
    plan = first.post(
        f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}
    ).json()[0]
    assert first.post(
        f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify"
    ).json()["decision"] == "VERIFIED"
    assert first.post(
        f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "operator-persist"},
    ).status_code == 200

    second_app = create_app(verifier=_verifier, database_path=database)
    second = TestClient(second_app)
    restored = second.get(f"/api/incidents/{incident_id}")
    assert restored.status_code == 200
    assert restored.json()["status"] == "CLOSED"
    exported = second.get(f"/api/incidents/{incident_id}/export").json()
    assert len(exported["plans"]) == 1
    assert len(exported["verifications"]) == 1
    assert second.post(f"/api/incidents/{incident_id}/replay").json()["chain_valid"] is True

    counts = second_app.state.runtime.store.table_counts()
    assert counts == {
        "networks": 1,
        "incidents": 1,
        "observations": 1,
        "posteriors": 1,
        "plans": 1,
        "verifications": 1,
        "approvals": 1,
    }

