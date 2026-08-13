from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.domain import ConsequenceMetrics, OODLevel, OperationalAction, OperationalPlan, PlanDecision, PlanVerification
from hydroswarm.inference.fusion import ControlAction
from hydroswarm.inference.ood import OODComponents
from hydroswarm.inference.results import HybridRuntimeMode, IncidentAnalysisResult, SemanticPredictions
from hydroswarm.planning import PlanProposal


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


def _install_authoritative_test_analysis(client: TestClient, incident_id: str) -> OperationalPlan:
    """Exercise persistence through the same strict analysis type as production."""
    from uuid import UUID

    plan = OperationalPlan(
        incident_id=incident_id, name="persistence-authorized-plan", model_version="test",
        actions=(OperationalAction(action_type="WAIT", duration_minutes=1),),
    )
    proposal = PlanProposal(plan, "TEST", 0.0, 1.0, (("WAIT", None),))
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
        plan_proposals=(proposal,), semantic_predictions=SemanticPredictions(), posterior_history=(),
        evidence_history=(), comparison_history=(), before_after=None,
        runtime_mode=HybridRuntimeMode.CLASSICAL_SAFE, neural_failure=None, latencies_ms={},
        provenance_hashes={"network": "test", "feature_schema": "test", "model": "test"}, evidence_hash="test",
    )
    record.runtime_mode = HybridRuntimeMode.CLASSICAL_SAFE.value
    record.fallback_reasons = ()
    return plan


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
    _install_authoritative_test_analysis(first, incident_id)
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


def test_a_new_sample_after_restart_still_triggers_reanalysis(tmp_path) -> None:
    # core-issues.txt: record.analysis (the live IncidentAnalysisResult) is
    # never persisted -- only record.state.candidates is. A restored
    # incident's in-memory record.analysis is None even though it really
    # was analyzed before the restart; adding a new sample must still
    # trigger reanalysis rather than silently skip it just because the
    # in-memory analysis object did not survive.
    database = tmp_path / "runtime.db"
    first = TestClient(create_app(verifier=_verifier, database_path=database))
    assert first.post(
        "/api/networks/persisted-network/validate",
        json={"node_ids": ["J1", "J2"], "link_count": 1},
    ).status_code == 200
    created = first.post("/api/incidents", json=_incident_payload())
    incident_id = created.json()["incident_id"]
    assert first.post(f"/api/incidents/{incident_id}/analyze").status_code == 200
    events_before_restart = first.get(f"/api/incidents/{incident_id}/events").json()
    analyzed_count_before = sum(
        1 for event in events_before_restart if event["event_type"] == "HYBRID_ANALYSIS_COMPLETED"
    )
    assert analyzed_count_before == 1

    # Simulate a process restart: a fresh app/runtime backed by the same
    # database. The restored IncidentRuntime.analysis is None; only
    # state.candidates survived.
    second = TestClient(create_app(verifier=_verifier, database_path=database))
    added = second.post(
        f"/api/incidents/{incident_id}/samples",
        json={
            "sensor_id": "S2",
            "node_id": "J2",
            "observed_at": NOW.isoformat(),
            "received_at": NOW.isoformat(),
            "concentration_mg_l": 0.3,
            "pressure_m": 24.0,
        },
    )
    assert added.status_code == 200
    events_after = second.get(f"/api/incidents/{incident_id}/events").json()
    analyzed_count_after = sum(
        1 for event in events_after if event["event_type"] == "HYBRID_ANALYSIS_COMPLETED"
    )
    assert analyzed_count_after == analyzed_count_before + 1
