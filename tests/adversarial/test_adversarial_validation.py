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
from hypothesis import given, strategies as st

from hydroswarm.agents.controller import SwarmController
from hydroswarm.agents.schemas import FSMState
from hydroswarm.api import create_app
from hydroswarm.api.state import ApiSettings
from hydroswarm.domain import (
    ConsequenceMetrics,
    OODLevel,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.inference.fusion import ControlAction
from hydroswarm.inference.ood import OODComponents
from hydroswarm.inference.results import HybridRuntimeMode, IncidentAnalysisResult, SemanticPredictions
from hydroswarm.planning import PlanProposal
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


def _install_authoritative_test_analysis(client: TestClient, incident_id: str) -> OperationalPlan:
    """Test-only real analysis type; no production fallback can satisfy this gate."""
    from uuid import UUID

    plan = OperationalPlan(
        incident_id=incident_id,
        name="authorized invariant plan",
        actions=(OperationalAction(action_type="WAIT", duration_minutes=1),),
        model_version="test",
    )
    proposal = PlanProposal(
        plan=plan,
        template="TEST",
        predicted_value=0.0,
        predicted_validity=1.0,
        diversity_key=(("WAIT", None),),
    )
    runtime = client.app.state.runtime
    record = runtime.incidents[UUID(incident_id)]
    record.analysis = IncidentAnalysisResult(
        incident_id=UUID(incident_id), node_alignment=("J1", "J2"),
        classical_belief={"J1": 1.0}, neural_belief=None, fused_belief={"J1": 1.0},
        classical_localization=None, estimated_hydraulic_state=None, trust_features=None,
        fusion_diagnostics=None, trust_rationale="test-only authoritative analysis",
        conformal_candidate_nodes=("J1",), calibrated=True, calibration_alpha=0.1,
        ood_components=OODComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ood_level=OODLevel.NORMAL, evidence_sufficient=True, planning_allowed=True,
        planning_suppression_reasons=(), control_action=ControlAction.GENERATE_PLANS,
        sample_result=None, plan_proposals=(proposal,), semantic_predictions=SemanticPredictions(),
        posterior_history=(), evidence_history=(), comparison_history=(), before_after=None,
        runtime_mode=HybridRuntimeMode.CLASSICAL_SAFE, neural_failure=None, latencies_ms={},
        provenance_hashes={"network": "test", "feature_schema": "test", "model": "test"},
        evidence_hash="test",
    )
    record.runtime_mode = HybridRuntimeMode.CLASSICAL_SAFE.value
    record.fallback_reasons = ()
    return plan


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


def test_nonfinite_concentration_is_rejected_by_the_domain_contract() -> None:
    """Regression for ADV-03: +infinity is never valid evidence."""
    with pytest.raises(ValueError):
        SensorObservation(**_observation(concentration=float("inf")))


@given(field=st.sampled_from(("concentration_mg_l", "pressure_m", "quality")),
       value=st.sampled_from((float("nan"), float("inf"), float("-inf"))))
def test_all_externally_reachable_observation_numbers_reject_nonfinite(field: str, value: float) -> None:
    """Property regression: every numerical sensor field rejects all nonfinite encodings."""
    payload = _observation()
    payload[field] = value
    with pytest.raises(ValueError):
        SensorObservation(**payload)


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
    # Suppressed DEMO_FALLBACK cannot generate an approval-capable plan.
    assert client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}).status_code == 409
    from uuid import uuid4
    plan_id = uuid4()
    assert client.post(
        f"/api/incidents/{incident_id}/plans/{plan_id}/approve",
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
    assert rejected_result.status_code == 409
    assert client.post(
        f"/api/incidents/{incident_id}/plans/{rejected.plan_id}/approve",
        json={"approved": True, "operator_id": "adversary"},
    ).status_code == 409


def test_evidence_mutation_after_restart_remains_nonapprovable(tmp_path) -> None:
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
    restarted = TestClient(create_app(verifier=_verification, database_path=database))
    assert restarted.post(f"/api/incidents/{incident_id}/samples", json=_observation(sensor="S2", node="J2")).status_code == 200
    assert restarted.post(f"/api/incidents/{incident_id}/workflow").status_code == 409


def test_reference_endpoint_rejects_checksum_mismatched_artifact(tmp_path) -> None:
    """Regression for ADV-21: forged reference JSON fails closed."""
    artifact = {
        "reference_id": "reference-incident-v1",
        "final_event_hash": "forged",
        "milestones": [],
        "artifact_sha256": "0" * 64,
    }
    reference = tmp_path / "forged-reference.json"
    reference.write_text(json.dumps(artifact), encoding="utf-8")
    response = TestClient(create_app(reference_demo_path=reference)).get("/api/reference-demo")
    assert response.status_code == 503


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


def test_suppressed_workflow_never_creates_approval_receipts(tmp_path) -> None:
    """Regression for ADV-17/09: no approval lifecycle starts from suppression."""
    client, incident_id = _prepared_client(tmp_path)
    assert client.post(f"/api/incidents/{incident_id}/workflow").status_code == 409
    assert client.app.state.runtime.store.table_counts()["approvals"] == 0


def test_planning_suppression_is_not_bypassable_through_workflow_endpoint(tmp_path) -> None:
    """Regression for ADV-09: both planning paths apply one authority gate."""
    client, incident_id = _prepared_client(tmp_path)
    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
    assert analysis["planning_allowed"] is False

    workflow = client.post(f"/api/incidents/{incident_id}/workflow")
    assert workflow.status_code == 409


def test_network_id_cannot_be_replaced(tmp_path) -> None:
    """Regression for ADV-15: a network id is immutable."""
    client, incident_id = _prepared_client(tmp_path)
    replacement = client.post(
        "/api/networks/study-network/validate",
        json={"node_ids": ["J1", "J2", "J3"], "link_count": 99},
    )
    assert replacement.status_code == 409


def test_suppressed_plan_content_never_reaches_approval(tmp_path) -> None:
    """Regression for ADV-13: no plan can be verified from suppressed analysis."""
    client, incident_id = _prepared_client(tmp_path)
    assert client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}).status_code == 409


def test_approval_requires_current_exact_plan_network_and_context_bindings(tmp_path) -> None:
    """ADV-13/15/17 regression: one verification cannot authorize a changed object."""
    client, incident_id = _prepared_client(tmp_path)
    plan = _install_authoritative_test_analysis(client, incident_id)
    generated = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1})
    assert generated.status_code == 200
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan.plan_id}/verify").status_code == 200

    runtime = client.app.state.runtime
    record = runtime.incidents[plan.incident_id]
    record.plans[plan.plan_id] = plan.model_copy(update={"name": "mutated after verification"})
    assert client.post(
        f"/api/incidents/{incident_id}/plans/{plan.plan_id}/approve",
        json={"approved": True, "operator_id": "adversary"},
    ).status_code == 409

    # Re-verification restores only the newly bound plan; a changed network
    # identity then independently makes approval fail closed.
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan.plan_id}/verify").status_code == 200
    network = runtime.networks["study-network"]
    runtime.networks["study-network"] = network.model_copy(update={"sha256": "f" * 64})
    assert client.post(
        f"/api/incidents/{incident_id}/plans/{plan.plan_id}/approve",
        json={"approved": True, "operator_id": "adversary"},
    ).status_code == 409


def test_approval_is_one_way_and_duplicate_receipts_are_not_persisted(tmp_path) -> None:
    """ADV-17 regression: only one current verified approval may close an incident."""
    client, incident_id = _prepared_client(tmp_path)
    plan = _install_authoritative_test_analysis(client, incident_id)
    assert client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}).status_code == 200
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan.plan_id}/verify").status_code == 200
    approval = {"approved": True, "operator_id": "operator-1"}
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan.plan_id}/approve", json=approval).status_code == 200
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan.plan_id}/approve", json=approval).status_code == 409
    assert client.app.state.runtime.store.table_counts()["approvals"] == 1
    events = client.get(f"/api/incidents/{incident_id}/events").json()
    assert [event["event_type"] for event in events].count("PLAN_APPROVED") == 1


def test_sample_and_approval_race_cannot_bypass_terminal_or_stale_guards(tmp_path) -> None:
    """Barrier-driven concurrency regression: a sample and approval cannot both commit."""
    from threading import Barrier

    client, incident_id = _prepared_client(tmp_path)
    plan = _install_authoritative_test_analysis(client, incident_id)
    assert client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}).status_code == 200
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan.plan_id}/verify").status_code == 200
    barrier = Barrier(2)

    def sample() -> int:
        barrier.wait()
        return client.post(f"/api/incidents/{incident_id}/samples", json=_observation(sensor="S2", node="J2")).status_code

    def approve() -> int:
        barrier.wait()
        return client.post(
            f"/api/incidents/{incident_id}/plans/{plan.plan_id}/approve",
            json={"approved": True, "operator_id": "race-operator"},
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        sample_status, approval_status = tuple(pool.map(lambda fn: fn(), (sample, approve)))
    assert sorted((sample_status, approval_status)) == [200, 409]
    assert client.app.state.runtime.store.table_counts()["approvals"] <= 1


def test_verify_and_approval_race_serializes_a_valid_lifecycle(tmp_path) -> None:
    """Barrier-driven concurrency regression: approval is never prior to verification."""
    from threading import Barrier

    client, incident_id = _prepared_client(tmp_path)
    plan = _install_authoritative_test_analysis(client, incident_id)
    assert client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}).status_code == 200
    barrier = Barrier(2)

    def verify() -> int:
        barrier.wait()
        return client.post(f"/api/incidents/{incident_id}/plans/{plan.plan_id}/verify").status_code

    def approve() -> int:
        barrier.wait()
        return client.post(
            f"/api/incidents/{incident_id}/plans/{plan.plan_id}/approve",
            json={"approved": True, "operator_id": "race-operator"},
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        verify_status, approval_status = tuple(pool.map(lambda fn: fn(), (verify, approve)))
    assert verify_status == 200
    assert approval_status in {200, 409}
    events = [event["event_type"] for event in client.get(f"/api/incidents/{incident_id}/events").json()]
    if approval_status == 200:
        assert events.index("PLAN_VERIFIED") < events.index("PLAN_APPROVED")


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


def test_positive_infinity_json_is_rejected_before_persistence(tmp_path) -> None:
    """Regression for ADV-03: raw nonfinite JSON gets a serializable 422."""
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
    assert response.status_code == 422
    assert client.app.state.runtime.store.table_counts()["incidents"] == 0


def test_duplicate_and_stale_samples_are_non_evidence(tmp_path) -> None:
    """Regression for ADV-04: retransmit is idempotent and predating data fails."""
    client, incident_id = _prepared_client(tmp_path)
    duplicate = _observation(sensor="S1", node="J1")
    stale = {**_observation(sensor="old", node="J2"), "observed_at": "2000-01-01T00:00:00+00:00"}
    assert client.post(f"/api/incidents/{incident_id}/samples", json=duplicate).status_code == 200
    assert client.post(f"/api/incidents/{incident_id}/samples", json=stale).status_code == 422
    state = client.get(f"/api/incidents/{incident_id}").json()
    assert state["sample_count"] == 0
    assert len(state["observations"]) == 1


def test_oversized_and_invalid_content_length_requests_fail_before_incident_creation(tmp_path) -> None:
    """ADV-23 positive control: middleware rejects basic resource-abuse shapes."""
    client = TestClient(
        create_app(
            verifier=_verification,
            database_path=tmp_path / "limits.sqlite3",
            settings=ApiSettings(maximum_request_bytes=128),
        )
    )
    assert client.post("/api/incidents", content=b"x" * 129).status_code == 413
    assert client.post("/api/incidents", content=b"{}", headers={"content-length": "not-an-int"}).status_code == 400
