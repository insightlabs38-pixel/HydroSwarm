from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import time

from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.domain import ConsequenceMetrics, PlanDecision, PlanVerification


NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _observation(sensor: str, node: str, concentration: float) -> dict[str, object]:
    return {
        "sensor_id": sensor,
        "node_id": node,
        "observed_at": NOW.isoformat(),
        "received_at": NOW.isoformat(),
        "concentration_mg_l": concentration,
        "pressure_m": 30.0,
        "quality": 1.0,
        "missing": False,
        "drift_flag": False,
        "frozen_flag": False,
    }


def _authority(plan, state) -> PlanVerification:
    return PlanVerification(
        plan_id=plan.plan_id,
        decision=PlanDecision.VERIFIED,
        simulator="test-authority",
        simulator_version="1",
        state_hash=hashlib.sha256(str(state.incident_id).encode()).hexdigest(),
        consequences=ConsequenceMetrics(
            minimum_pressure_m=25.0,
            service_availability=1.0,
            operation_count=len(plan.actions),
        ),
    )


def test_default_api_exposes_fallback_reanalysis_worker_and_live_status(tmp_path) -> None:
    app = create_app(verifier=_authority, database_path=tmp_path / "runtime.sqlite3")
    with TestClient(app) as client:
        network = client.post(
            "/api/networks/demo/validate",
            json={"node_ids": ["J1", "J2"], "link_count": 1},
        )
        assert network.status_code == 200

        created = client.post(
            "/api/incidents",
            json={
                "network_id": "demo",
                "detected_at": NOW.isoformat(),
                "observations": [_observation("S1", "J1", 0.25)],
                "maximum_samples": 2,
            },
        )
        incident_id = created.json()["incident_id"]

        assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200
        first = client.get(f"/api/incidents/{incident_id}/analysis").json()
        assert first["runtime_mode"] == "DEMO_FALLBACK"
        assert "HYBRID_PIPELINE_NOT_CONFIGURED" in first["fallback_reasons"]
        assert first["planning_allowed"] is False

        added = client.post(
            f"/api/incidents/{incident_id}/samples",
            json=_observation("S2", "J2", 0.5),
        )
        assert added.status_code == 200
        second = client.get(f"/api/incidents/{incident_id}/analysis").json()
        assert second["posterior_history"][0]["observation_count"] == 2
        assert second["fused_belief"] != first["fused_belief"]

        queued = client.post(f"/api/incidents/{incident_id}/analyze/jobs").json()
        job_id = queued["job_id"]
        for _ in range(200):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["state"] in {"COMPLETE", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.01)
        assert job["state"] == "COMPLETE"
        assert job["progress"] == 1.0
        assert job["result"]["runtime_mode"] == "DEMO_FALLBACK"

        summary = client.get(f"/api/incidents/{incident_id}/summary")
        assert summary.status_code == 200
        assert summary.json()["runtime_mode"] == "DEMO_FALLBACK"
        explanation = client.get(
            f"/api/incidents/{incident_id}/explanations/WHY_SOURCE"
        )
        assert explanation.status_code == 200

        workflow = client.post(f"/api/incidents/{incident_id}/workflow")
        assert workflow.status_code == 409
        assert workflow.json()["detail"]["reason"] == "PLANNING_SUPPRESSED"

        with client.websocket_connect(f"/ws/incidents/{incident_id}") as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["incident_id"] == incident_id
            assert snapshot["runtime_mode"] == "DEMO_FALLBACK"

        exported = client.get(f"/api/incidents/{incident_id}/export").json()
        assert exported["analysis"]["runtime_mode"] == "DEMO_FALLBACK"
        event_types = {
            event["event_type"]
            for event in client.get(f"/api/incidents/{incident_id}/events").json()
        }
        assert "SAMPLE_RECEIVED" in event_types
        assert "HYBRID_ANALYSIS_COMPLETED" in event_types
