"""Local adversarial probes for the documented authority boundaries.

These probes deliberately preserve observed behavior, including failures, so
the study runner can report them without altering production thresholds.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hydroswarm.agents.controller import SwarmController
from hydroswarm.agents.schemas import FSMState
from hydroswarm.api import create_app
from hydroswarm.domain import (
    ConsequenceMetrics,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.simulation import (
    PlanVerifier,
    SimulationIncompleteError,
    SimulationTimeoutError,
    SimulationUnstableError,
)


NOW = datetime(2026, 8, 13, tzinfo=UTC)


def _observation(*, sensor: str = "S1", node: str = "J1", concentration: float = 0.2) -> dict[str, object]:
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


def _verification(plan: OperationalPlan, state) -> PlanVerification:
    unsafe = "unsafe" in plan.name
    return PlanVerification(
        plan_id=plan.plan_id,
        decision=PlanDecision.REJECTED if unsafe else PlanDecision.VERIFIED,
        simulator="adversarial-injected-verifier",
        simulator_version="test",
        state_hash=hashlib.sha256(str(state.incident_id).encode()).hexdigest(),
        consequences=(
            None
            if unsafe
            else ConsequenceMetrics(
                minimum_pressure_m=25.0,
                service_availability=1.0,
                operation_count=len(plan.actions),
            )
        ),
        rejection_codes=("PRESSURE_BELOW_MINIMUM",) if unsafe else (),
    )


def _prepared_client(tmp_path) -> tuple[TestClient, str]:
    client = TestClient(create_app(verifier=_verification, database_path=tmp_path / "study.sqlite3"))
    assert client.post(
        "/api/networks/study-network/validate",
        json={"node_ids": ["J1", "J2"], "link_count": 1},
    ).status_code == 200
    created = client.post(
        "/api/incidents",
        json={"network_id": "study-network", "detected_at": NOW.isoformat(), "observations": [_observation()]},
    )
    assert created.status_code == 201
    incident_id = created.json()["incident_id"]
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200
    return client, incident_id


def _generated_plan(client: TestClient, incident_id: str) -> dict[str, object]:
    plans = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1})
    assert plans.status_code == 200
    return plans.json()[0]


def test_malformed_unknown_and_negative_observations_do_not_persist(tmp_path) -> None:
    client = TestClient(create_app(verifier=_verification, database_path=tmp_path / "malformed.sqlite3"))
    assert client.post(
        "/api/networks/study-network/validate", json={"node_ids": ["J1"], "link_count": 0}
    ).status_code == 200
    invalid = _observation(node="unknown", concentration=-1.0)
    response = client.post(
        "/api/incidents",
        json={"network_id": "study-network", "detected_at": NOW.isoformat(), "observations": [invalid]},
    )
    assert response.status_code == 422
    assert client.get("/api/networks").status_code == 200


def test_nonfinite_concentration_is_accepted_by_the_domain_contract() -> None:
    """ADV-03 detector: the observation contract currently permits +infinity."""
    accepted = SensorObservation(**_observation(concentration=float("inf")))
    assert accepted.concentration_mg_l == float("inf")


@pytest.mark.parametrize("failure", [SimulationTimeoutError, SimulationIncompleteError, SimulationUnstableError])
def test_simulator_failure_categories_never_verify(failure) -> None:
    class FailingSimulator:
        simulator_name = "adversarial-simulator"
        simulator_version = "test"
        minimum_pressure_m = 15.0
        minimum_service_availability = 0.95
        exact_runs = 0
        network = SimpleNamespace(
            link_name_list=(), node_name_list=(), pipe_name_list=(), junction_name_list=()
        )

        def state_hash(self, _plan):
            return "0" * 64

        def validate(self):
            return ()

        def evaluate_plan(self, _plan):
            raise failure("injected")

    plan = OperationalPlan(
        incident_id="00000000-0000-0000-0000-000000000001",
        name="failure probe",
        actions=(OperationalAction(action_type="WAIT", duration_minutes=1),),
        model_version="test",
    )
    result = PlanVerifier(FailingSimulator()).verify(plan)
    assert result.decision == PlanDecision.ABSTAINED


def test_direct_approval_and_rejected_approval_are_blocked(tmp_path) -> None:
    client, incident_id = _prepared_client(tmp_path)
    plan = _generated_plan(client, incident_id)
    assert client.post(
        f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "adversary"},
    ).status_code == 409

    # Store a distinctly rejected synthetic plan through the normal in-memory
    # test seam, then prove its direct approval is still rejected.
    runtime = client.app.state.runtime
    rejected = OperationalPlan(
        incident_id=incident_id,
        name="unsafe adversarial plan",
        actions=(OperationalAction(action_type="WAIT", duration_minutes=1),),
        model_version="test",
    )
    runtime.incidents[rejected.incident_id].plans[rejected.plan_id] = rejected
    rejected_result = client.post(f"/api/incidents/{incident_id}/plans/{rejected.plan_id}/verify")
    assert rejected_result.json()["decision"] == "REJECTED"
    assert client.post(
        f"/api/incidents/{incident_id}/plans/{rejected.plan_id}/approve",
        json={"approved": True, "operator_id": "adversary"},
    ).status_code == 409


def test_evidence_mutation_stales_verification_even_after_restart(tmp_path) -> None:
    database = tmp_path / "restart.sqlite3"
    client = TestClient(create_app(verifier=_verification, database_path=database))
    assert client.post(
        "/api/networks/study-network/validate", json={"node_ids": ["J1", "J2"], "link_count": 1}
    ).status_code == 200
    created = client.post(
        "/api/incidents",
        json={"network_id": "study-network", "detected_at": NOW.isoformat(), "observations": [_observation()]},
    )
    incident_id = created.json()["incident_id"]
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200
    plan = _generated_plan(client, incident_id)
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify").json()["decision"] == "VERIFIED"

    restarted = TestClient(create_app(verifier=_verification, database_path=database))
    assert restarted.post(f"/api/incidents/{incident_id}/samples", json=_observation(sensor="S2", node="J2")).status_code == 200
    assert restarted.post(
        f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "adversary"},
    ).status_code == 409


def test_reference_endpoint_serves_checksum_mismatched_artifact(tmp_path) -> None:
    """ADV-21 detector: runtime parses a reference artifact but does not verify its declared hash."""
    artifact = {
        "reference_id": "reference-incident-v1",
        "final_event_hash": "forged",
        "milestones": [],
        "artifact_sha256": "0" * 64,
    }
    reference = tmp_path / "forged-reference.json"
    reference.write_text(json.dumps(artifact), encoding="utf-8")
    response = TestClient(create_app(reference_demo_path=reference)).get("/api/reference-demo")
    assert response.status_code == 200
    assert response.json()["final_event_hash"] == "forged"


def test_live_example_cache_response_has_no_explicit_cache_or_provenance_label() -> None:
    """ADV-22 detector: the cached input payload does not identify its computation/cache provenance."""
    # The builder is intentionally not invoked: this checks the response contract
    # independently of expensive WNTR execution by inspecting the documented API model.
    from hydroswarm.evaluation.live_example import build_live_example_inputs

    assert "cache" not in build_live_example_inputs.__doc__.lower().split("returns", 1)[-1]
    assert "data_mode" not in str(build_live_example_inputs.__annotations__.get("return", ""))


def test_controller_rejects_invalid_direct_transition() -> None:
    controller = object.__new__(SwarmController)
    controller.state = FSMState.IDLE
    controller.events = []
    with pytest.raises(ValueError, match="invalid swarm transition"):
        controller.transition(FSMState.HUMAN_APPROVAL)


def test_duplicate_approval_requests_are_not_idempotent(tmp_path) -> None:
    """ADV-17 detector: two serial approvals both receive a receipt."""
    client, incident_id = _prepared_client(tmp_path)
    plan = _generated_plan(client, incident_id)
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify").status_code == 200

    def approve() -> int:
        return client.post(
            f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/approve",
            json={"approved": True, "operator_id": "replay"},
        ).status_code

    # Serial replay is sufficient to distinguish an idempotency boundary from
    # a mere transport retry. The second request is currently accepted.
    assert [approve(), approve()] == [200, 200]


def test_planning_suppression_is_bypassable_through_workflow_endpoint(tmp_path) -> None:
    """ADV-09 detector: /workflow treats demo fallback as evidence-sufficient."""
    client, incident_id = _prepared_client(tmp_path)
    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
    assert analysis["planning_allowed"] is False

    workflow = client.post(f"/api/incidents/{incident_id}/workflow")
    assert workflow.status_code == 200
    assert workflow.json()["state"] == "HUMAN_APPROVAL"
    assert workflow.json()["verification"]["decision"] == "VERIFIED"


def test_network_replacement_does_not_invalidate_verified_plan(tmp_path) -> None:
    """ADV-15 detector: normal topology-validation API can replace a network by id."""
    client, incident_id = _prepared_client(tmp_path)
    plan = _generated_plan(client, incident_id)
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify").json()["decision"] == "VERIFIED"

    replacement = client.post(
        "/api/networks/study-network/validate",
        json={"node_ids": ["J1", "J2", "J3"], "link_count": 99},
    )
    assert replacement.status_code == 200
    approved = client.post(
        f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/approve",
        json={"approved": True, "operator_id": "network-mutator"},
    )
    assert approved.status_code == 200


def test_plan_content_mutation_with_same_id_remains_approvable(tmp_path) -> None:
    """ADV-13 detector: approval binds only plan id/context, not plan content."""
    client, incident_id = _prepared_client(tmp_path)
    plan_dump = _generated_plan(client, incident_id)
    plan_id = plan_dump["plan_id"]
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan_id}/verify").json()["decision"] == "VERIFIED"

    runtime = client.app.state.runtime
    record = runtime.incidents[next(key for key in runtime.incidents if str(key) == incident_id)]
    original = record.plans[next(key for key in record.plans if str(key) == plan_id)]
    record.plans[original.plan_id] = original.model_copy(
        update={
            "actions": (
                OperationalAction(action_type="FLUSH_NODE", target_id="J2", flow_rate_lps=9_999_999.0),
            )
        }
    )
    assert client.post(
        f"/api/incidents/{incident_id}/plans/{plan_id}/approve",
        json={"approved": True, "operator_id": "plan-mutator"},
    ).status_code == 200


@pytest.mark.parametrize("poison", ["NaN", "-Infinity"])
def test_nan_and_negative_infinity_do_not_complete_incident_creation(tmp_path, poison: str) -> None:
    """Raw JSON probe: these two values currently fail at request validation."""
    app = create_app(verifier=_verification, database_path=tmp_path / f"{poison}.sqlite3")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post(
        "/api/networks/study-network/validate", json={"node_ids": ["J1"], "link_count": 0}
    ).status_code == 200
    payload = json.dumps({
        "network_id": "study-network",
        "detected_at": NOW.isoformat(),
        "observations": [{**_observation(), "concentration_mg_l": poison}],
    }).replace(f'"{poison}"', poison)
    response = client.post("/api/incidents", content=payload, headers={"content-type": "application/json"})
    assert response.status_code >= 400


def test_positive_infinity_json_is_accepted_and_persisted(tmp_path) -> None:
    """ADV-03 detector: +Infinity passes NonNegative and returns 201."""
    app = create_app(verifier=_verification, database_path=tmp_path / "infinity.sqlite3")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post(
        "/api/networks/study-network/validate", json={"node_ids": ["J1"], "link_count": 0}
    ).status_code == 200
    payload = json.dumps({
        "network_id": "study-network",
        "detected_at": NOW.isoformat(),
        "observations": [{**_observation(), "concentration_mg_l": "Infinity"}],
    }).replace('"Infinity"', "Infinity")
    response = client.post("/api/incidents", content=payload, headers={"content-type": "application/json"})
    assert response.status_code == 201
    incident_id = response.json()["incident_id"]
    assert client.get(f"/api/incidents/{incident_id}").status_code == 200
