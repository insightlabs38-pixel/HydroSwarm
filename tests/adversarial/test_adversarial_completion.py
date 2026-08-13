"""Completion probes for ADV-05 and ADV-25.

These are local-only, deterministic tests.  They exercise the real pipeline
and API boundaries without changing scientific or operational policy.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
import torch
from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st

from hydroswarm.api import create_app
from hydroswarm.calibration import CalibrationArtifact, CalibrationReport
from hydroswarm.classical import SignatureArtifact, SignatureCacheKey, SignatureLibrary, SourceHypothesis
from hydroswarm.domain import (
    ConsequenceMetrics,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.evaluation import validate_reference_incident_artifact
from hydroswarm.inference import HybridInferencePipeline
from hydroswarm.preprocessing import DEFAULT_FEATURE_SCHEMA, SensorSeries
from hydroswarm.simulation import HydraulicSimulator, build_wntr_network


NOW = datetime(2026, 8, 13, tzinfo=UTC)


class _PriorFollowingModel:
    """Deterministic model used only to make the real pipeline reproducible."""

    def eval(self):
        return self

    def __call__(self, batch):
        prior = batch["classical_prior"].clamp_min(1e-7)
        nodes = prior.shape[1]
        return {
            "source_node_logits": prior.log(),
            "evidence_sufficiency": torch.tensor([[0.9]]),
            "uncertainty": torch.tensor([[0.1]]),
            "expected_information_gain": torch.full((1, nodes), 0.05),
            "sensor_fault_logits": torch.full((1, nodes), -4.0),
            "ood_logits": torch.tensor([[4.0, 0.0, -4.0]]),
            "plan_value": torch.zeros(1, 8),
            "plan_validity_logits": torch.tensor([[[0.0, 2.0]] * 8]),
        }


def _signature_artifact() -> SignatureArtifact:
    hypotheses = tuple(SourceHypothesis(node, 0, 60, 1.0, "nominal") for node in ("J1", "J2", "J3"))
    signatures = {
        "J1": [[0.0, 0.0, 0.0], [1.00, 0.15, 0.05]],
        "J2": [[0.0, 0.0, 0.0], [0.78, 1.00, 0.10]],
        "J3": [[0.0, 0.0, 0.0], [0.65, 0.20, 1.00]],
    }
    library = SignatureLibrary()
    for hypothesis in hypotheses:
        library.add(hypothesis.identifier, signatures[hypothesis.source_node])
    return SignatureArtifact(
        key=SignatureCacheKey("a" * 64, "b" * 64, "adv-completion", "c" * 64, "d" * 64),
        library=library,
        hypotheses=hypotheses,
        sensor_nodes=("J1", "J2", "J3"),
        sample_times_seconds=(0, 3600),
        cache_hit=True,
        artifact_hash="e" * 64,
    )


def _pipeline() -> tuple[HybridInferencePipeline, object]:
    network = build_wntr_network()
    network.options.time.duration = 3600
    model = _PriorFollowingModel()
    model_hash = HybridInferencePipeline._fingerprint_model(model)
    calibration = CalibrationArtifact(
        schema_version="hydroswarm-calibration-v1",
        alpha=0.1,
        model_hash=model_hash,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        dataset_manifest_hash="f" * 64,
        global_scores=(0.8, 0.8, 0.8, 0.8),
        mondrian_scores={}, network_scores={}, report=CalibrationReport(1.0, 1.0, 0.0, {}, {}, 4),
    )
    return (
        HybridInferencePipeline(
            simulator=HydraulicSimulator(network), signature_artifact=_signature_artifact(), model=model,
            model_hash=model_hash, calibration_artifact=calibration, maximum_planning_candidates=1,
        ),
        network,
    )


def _series(node: str, concentration: float, *, health: float = 1.0, drift: bool = False) -> SensorSeries:
    return SensorSeries(
        node_id=node, timestamps_seconds=(0.0, 3600.0), concentration_mg_l=(0.0, concentration),
        pressure_m=(25.0, 24.0), health=(health, health), missing=(False, False),
        drift=(drift, drift), delayed=(False, False),
    )


def _verification(plan: OperationalPlan, state) -> PlanVerification:
    return PlanVerification(
        plan_id=plan.plan_id, decision=PlanDecision.VERIFIED, simulator="completion-injected-verifier",
        simulator_version="test", state_hash=hashlib.sha256(str(state.incident_id).encode()).hexdigest(),
        consequences=ConsequenceMetrics(minimum_pressure_m=25.0, service_availability=1.0, operation_count=len(plan.actions)),
    )


def _observation(
    *, sensor: str = "S1", node: str = "J1", concentration: object = 0.78,
    observed_at: datetime = NOW, received_at: datetime | None = None,
    quality: object = 1.0, missing: object = False, drift: object = False, frozen: object = False,
) -> dict[str, object]:
    return {
        "sensor_id": sensor, "node_id": node, "observed_at": observed_at.isoformat(),
        "received_at": (received_at or observed_at).isoformat(), "concentration_mg_l": concentration,
        "pressure_m": 24.0, "quality": quality, "missing": missing,
        "drift_flag": drift, "frozen_flag": frozen,
    }


def _api_client(tmp_path: Path) -> tuple[TestClient, object]:
    pipeline, network = _pipeline()
    client = TestClient(create_app(pipeline_factory=pipeline, verifier=_verification, database_path=tmp_path / "completion.sqlite3"))
    response = client.post(
        "/api/networks/completion-network/validate",
        json={"node_ids": list(network.node_name_list), "link_count": len(network.link_name_list)},
    )
    assert response.status_code == 200
    return client, network


def _create_and_analyze(client: TestClient, observation: dict[str, object]) -> tuple[str, dict[str, object]]:
    created = client.post(
        "/api/incidents",
        json={"network_id": "completion-network", "detected_at": NOW.isoformat(), "observations": [observation]},
    )
    assert created.status_code == 201, created.text
    incident_id = created.json()["incident_id"]
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200
    return incident_id, client.get(f"/api/incidents/{incident_id}/analysis").json()


def _assert_analysis_shape(analysis: dict[str, object]) -> None:
    probabilities = analysis["fused_belief"]
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert set(analysis["candidate_nodes"]).issubset(probabilities)


def _canonical_plan_hash(plan: OperationalPlan) -> str:
    return hashlib.sha256(json.dumps(
        plan.model_dump(mode="json", exclude_none=False), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


pytestmark = pytest.mark.real_simulation


def test_adv05_contradictory_pair_is_suppressed_by_real_disagreement() -> None:
    """A05-1: incompatible healthy trajectories must not silently retain planning authority."""
    pipeline, network = _pipeline()
    before = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])
    after = pipeline.analyze(uuid4(), network, [_series("J1", 0.78), _series("J3", 1.0)])
    assert before.planning_allowed is True
    assert after.planning_allowed is False
    assert after.fusion_diagnostics is not None
    assert after.fusion_diagnostics.disagreement_js >= pipeline.disagreement_threshold
    assert after.control_action.value != "GENERATE_PLANS"
    assert after.ood_level.value == "NORMAL"  # This case is governed by disagreement, not OOD.
    assert after.evidence_sufficient is True
    _assert_analysis_shape({"fused_belief": dict(after.fused_belief), "candidate_nodes": after.conformal_candidate_nodes})


def test_adv05_outlier_and_bias_trajectories_remain_numerically_well_formed() -> None:
    """A05-3/A05-6: extreme/bias inputs remain finite; this records rather than invents robust-outlier policy."""
    pipeline, network = _pipeline()
    coherent = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])
    outlier = pipeline.analyze(uuid4(), network, [_series("J1", 1_000_000.0)])
    biased = pipeline.analyze(uuid4(), network, [_series("J1", 0.78), _series("J1", 0.78)])
    for result in (coherent, outlier, biased):
        _assert_analysis_shape({"fused_belief": dict(result.fused_belief), "candidate_nodes": result.conformal_candidate_nodes})
    # Observed behavior: this prototype has no declared robust-outlier gate;
    # record the conservative fact without treating it as a threshold failure.
    assert outlier.ood_level.value == "NORMAL"
    assert outlier.planning_allowed is True
    assert biased.planning_allowed == coherent.planning_allowed


def test_adv05_quality_degrades_authority_while_drift_is_recorded_as_an_observation() -> None:
    """A05-5: low quality must not increase authority; record the current drift-only behavior honestly."""
    pipeline, network = _pipeline()
    healthy = pipeline.analyze(uuid4(), network, [_series("J1", 0.78, health=1.0)])
    low_quality = pipeline.analyze(uuid4(), network, [_series("J1", 0.78, health=0.01)])
    drift = pipeline.analyze(uuid4(), network, [_series("J1", 0.78, health=1.0, drift=True)])
    assert healthy.planning_allowed is True
    assert low_quality.planning_allowed is False
    assert low_quality.trust_features.healthy_sensor_fraction < healthy.trust_features.healthy_sensor_fraction
    # `drift` is passed as a model feature but is not a deterministic
    # authority suppressor in this fixture. This is recorded in COMPLETION,
    # not silently treated as evidence of robust sensor-fault handling.
    assert drift.planning_allowed == healthy.planning_allowed


def test_adv05_near_tie_remains_broad_and_nonplanning() -> None:
    """A05-1: a close two-source posterior must not be presented as a single strong source."""
    pipeline, network = _pipeline()
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.4), _series("J2", 0.65)])
    ranked = sorted(result.fused_belief.values(), reverse=True)
    assert ranked[0] - ranked[1] < 0.5
    assert len(result.conformal_candidate_nodes) > pipeline.maximum_planning_candidates
    assert result.planning_allowed is False
    assert result.control_action.value != "GENERATE_PLANS"
    _assert_analysis_shape({"fused_belief": dict(result.fused_belief), "candidate_nodes": result.conformal_candidate_nodes})


def test_adv05_conflict_after_verified_plan_stales_authority(tmp_path: Path) -> None:
    """A05-2: contradictory new evidence must reanalyze and make a VERIFIED plan nonapprovable."""
    client, _network = _api_client(tmp_path)
    incident_id, before = _create_and_analyze(client, _observation())
    assert before["planning_allowed"] is True
    plans = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1})
    assert plans.status_code == 200
    plan_id = plans.json()[0]["plan_id"]
    assert client.post(f"/api/incidents/{incident_id}/plans/{plan_id}/verify").json()["verification_status"] == "CURRENT"
    changed = client.post(
        f"/api/incidents/{incident_id}/samples",
        json=_observation(sensor="S2", node="J3", concentration=1.0, observed_at=NOW + timedelta(minutes=1)),
    )
    assert changed.status_code == 200
    after = client.get(f"/api/incidents/{incident_id}/analysis").json()
    assert after["planning_allowed"] is False
    assert after["disagreement_js"] > before["disagreement_js"]
    verification = client.get(f"/api/incidents/{incident_id}/export").json()["verifications"][0]
    assert verification["verification_status"] == "STALE"
    assert client.post(
        f"/api/incidents/{incident_id}/plans/{plan_id}/approve",
        json={"approved": True, "operator_id": "completion"},
    ).status_code == 409


@pytest.mark.xfail(strict=True, reason="ADV-27 HIGH: duplicate initial evidence is accepted before sample identity controls run")
def test_adv27_duplicate_initial_observation_is_rejected_atomically(tmp_path: Path) -> None:
    """A25-B: exact retransmission in incident creation must not enter durable evidence twice."""
    client, _network = _api_client(tmp_path)
    payload = _observation()
    response = client.post(
        "/api/incidents",
        json={"network_id": "completion-network", "detected_at": NOW.isoformat(), "observations": [payload, payload]},
    )
    assert response.status_code == 422
    assert client.app.state.runtime.store.table_counts()["incidents"] == 0


@pytest.mark.xfail(strict=True, reason="ADV-28 HIGH: frozen_flag is not carried into SensorSeries or authority gating")
def test_adv28_frozen_sensor_cannot_retain_planning_authority(tmp_path: Path) -> None:
    """A05-5: frozen evidence must reduce authority rather than plan identically to healthy evidence."""
    healthy_client, _network = _api_client(tmp_path / "healthy")
    frozen_client, _network = _api_client(tmp_path / "frozen")
    _healthy_id, healthy = _create_and_analyze(healthy_client, _observation(frozen=False))
    _frozen_id, frozen = _create_and_analyze(frozen_client, _observation(frozen=True))
    assert healthy["planning_allowed"] is True
    assert frozen["planning_allowed"] is False


@pytest.mark.xfail(strict=True, reason="ADV-29 MEDIUM: validation-error ctx contains a non-JSON-serializable ValueError")
def test_adv29_received_before_observed_is_controlled_422_without_mutation(tmp_path: Path) -> None:
    """A25-A/F: a cross-field timestamp error must remain a controlled 422, never a 500."""
    client = TestClient(
        create_app(verifier=_verification, database_path=tmp_path / "timestamp.sqlite3"),
        raise_server_exceptions=False,
    )
    assert client.post("/api/networks/fuzz/validate", json={"node_ids": ["J1", "J2"], "link_count": 1}).status_code == 200
    incident_id = client.post(
        "/api/incidents", json={"network_id": "fuzz", "detected_at": NOW.isoformat(), "observations": [_observation()]},
    ).json()["incident_id"]
    before = client.get(f"/api/incidents/{incident_id}").json()
    response = client.post(
        f"/api/incidents/{incident_id}/samples",
        json=_observation(received_at=NOW - timedelta(seconds=1)),
    )
    assert response.status_code == 422
    assert client.get(f"/api/incidents/{incident_id}").json() == before


_FINITE = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12)
_NUMERIC = st.one_of(_FINITE, st.sampled_from((float("nan"), float("inf"), float("-inf"))))


@settings(max_examples=180, deadline=None)
@given(concentration=_NUMERIC, pressure=_NUMERIC, quality=_NUMERIC, missing=st.booleans())
def test_adv25_sensor_observation_contract_is_finite(concentration: float, pressure: float, quality: float, missing: bool) -> None:
    """A25-A: accepted domain observations always contain finite bounded numerical state."""
    payload = _observation(concentration=concentration, quality=quality, missing=missing)
    payload["pressure_m"] = pressure
    try:
        observation = SensorObservation(**payload)
    except ValueError:
        return
    assert not observation.missing or (observation.concentration_mg_l is None and observation.pressure_m is None)
    for value in (observation.concentration_mg_l, observation.pressure_m, observation.quality):
        assert value is None or math.isfinite(value)
    assert observation.concentration_mg_l is None or observation.concentration_mg_l >= 0.0
    assert 0.0 <= observation.quality <= 1.0


_WEIRD_ID = st.one_of(st.just(""), st.text(min_size=1, max_size=80), st.just("水\N{SNOWMAN}"))
_OBS_FIELD = st.one_of(st.none(), st.integers(), st.text(max_size=40), st.lists(st.integers(), max_size=2), _NUMERIC)


@settings(max_examples=50, deadline=None)
@given(
    sensor_id=_WEIRD_ID, node_id=_WEIRD_ID, concentration=_OBS_FIELD,
    quality=_OBS_FIELD, unexpected=st.booleans(),
)
def test_adv25_malformed_sample_requests_are_atomic_and_controlled(
    sensor_id: object, node_id: object, concentration: object, quality: object,
    unexpected: bool,
) -> None:
    """A25-A/F: malformed sample requests return controlled 4xx with no durable mutation."""
    with TemporaryDirectory() as directory:
        client = TestClient(create_app(verifier=_verification, database_path=Path(directory) / "fuzz.sqlite3"))
        assert client.post("/api/networks/fuzz/validate", json={"node_ids": ["J1", "J2"], "link_count": 1}).status_code == 200
        created = client.post(
            "/api/incidents", json={"network_id": "fuzz", "detected_at": NOW.isoformat(), "observations": [_observation()]},
        )
        incident_id = created.json()["incident_id"]
        payload = _observation(
            sensor=sensor_id if isinstance(sensor_id, str) else "S1",
            node=node_id if isinstance(node_id, str) else "J1",
            concentration=concentration,
            quality=quality,
            received_at=NOW,
        )
        if not isinstance(sensor_id, str):
            payload["sensor_id"] = sensor_id
        if not isinstance(node_id, str):
            payload["node_id"] = node_id
        if unexpected:
            payload["unexpected"] = {"nested": True}
        before = client.get(f"/api/incidents/{incident_id}").json()
        # Raw encoding intentionally carries NaN/Infinity to the server;
        # TestClient's `json=` helper correctly refuses them before an API
        # boundary can be exercised.
        response = client.post(
            f"/api/incidents/{incident_id}/samples",
            content=json.dumps(payload, ensure_ascii=False, allow_nan=True),
            headers={"content-type": "application/json"},
        )
        after = client.get(f"/api/incidents/{incident_id}").json()
        if response.status_code >= 400:
            assert response.status_code < 500
            assert after["observations"] == before["observations"]
            assert after["sample_count"] == before["sample_count"]
            assert str(Path(directory)) not in response.text
        else:
            accepted = SensorObservation.model_validate(payload)
            assert math.isfinite(accepted.quality)
            assert after["sample_count"] == before["sample_count"] + 1


@settings(max_examples=45, deadline=None)
@given(
    network_id=st.one_of(st.just("fuzz"), st.just("missing"), st.text(max_size=12)),
    node_id=st.sampled_from(("J1", "J2", "unknown")),
    maximum_samples=st.integers(min_value=-2, max_value=4),
    duplicate=st.booleans(),
)
def test_adv25_incident_creation_is_atomic_for_invalid_combinations(
    network_id: str, node_id: str, maximum_samples: int, duplicate: bool,
) -> None:
    """A25-B: rejected incident creation never persists a partial incident or unknown-node observation."""
    with TemporaryDirectory() as directory:
        client = TestClient(create_app(verifier=_verification, database_path=Path(directory) / "create.sqlite3"))
        assert client.post("/api/networks/fuzz/validate", json={"node_ids": ["J1", "J2"], "link_count": 1}).status_code == 200
        observation = _observation(node=node_id)
        payload = {
            "network_id": network_id, "detected_at": NOW.isoformat(), "maximum_samples": maximum_samples,
            "observations": [observation, observation] if duplicate else [observation],
        }
        response = client.post("/api/incidents", json=payload)
        counts = client.app.state.runtime.store.table_counts()
        if response.status_code >= 400:
            assert response.status_code < 500
            assert counts["incidents"] == 0 and counts["observations"] == 0
        else:
            state = response.json()
            assert state["network_id"] == "fuzz"
            assert all(item["node_id"] in {"J1", "J2"} for item in state["observations"])


@settings(max_examples=75, deadline=None)
@given(
    name=st.text(min_size=1, max_size=30),
    duration=st.integers(min_value=0, max_value=120),
    mutation=st.text(min_size=1, max_size=30),
)
def test_adv25_plan_serialization_and_content_mutation_change_hash(name: str, duration: int, mutation: str) -> None:
    """A25-C/E: canonical plan content is stable through serialization and differs after a relevant mutation."""
    plan = OperationalPlan(
        incident_id=uuid4(), name=name, model_version="fuzz",
        actions=(OperationalAction(action_type="WAIT", duration_minutes=duration),),
    )
    restored = OperationalPlan.model_validate_json(plan.model_dump_json(exclude_none=False))
    changed = plan.model_copy(update={"name": mutation})
    assert _canonical_plan_hash(plan) == _canonical_plan_hash(restored)
    if mutation != name:
        assert _canonical_plan_hash(plan) != _canonical_plan_hash(changed)


@settings(max_examples=40, deadline=None)
@given(operations=st.lists(st.sampled_from(("analyze", "recommend", "sample", "generate", "verify", "approve", "workflow")), min_size=1, max_size=8))
def test_adv25_suppressed_state_machine_sequences_never_create_approval(operations: list[str]) -> None:
    """A25-D: no short sequence can advance planning-suppressed evidence into a receipt or CLOSED state."""
    with TemporaryDirectory() as directory:
        client = TestClient(create_app(verifier=_verification, database_path=Path(directory) / "state.sqlite3"))
        assert client.post("/api/networks/fuzz/validate", json={"node_ids": ["J1", "J2"], "link_count": 1}).status_code == 200
        created = client.post(
            "/api/incidents", json={"network_id": "fuzz", "detected_at": NOW.isoformat(), "observations": [_observation()]},
        )
        incident_id = created.json()["incident_id"]
        fake_plan = str(uuid4())
        for operation in operations:
            if operation == "analyze":
                response = client.post(f"/api/incidents/{incident_id}/analyze")
            elif operation == "recommend":
                response = client.post(f"/api/incidents/{incident_id}/samples/recommend")
            elif operation == "sample":
                response = client.post(f"/api/incidents/{incident_id}/samples", json=_observation(sensor="S2", node="J2", observed_at=NOW + timedelta(minutes=1)))
            elif operation == "generate":
                response = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1})
            elif operation == "verify":
                response = client.post(f"/api/incidents/{incident_id}/plans/{fake_plan}/verify")
            elif operation == "approve":
                response = client.post(f"/api/incidents/{incident_id}/plans/{fake_plan}/approve", json={"approved": True, "operator_id": "fuzz"})
            else:
                response = client.post(f"/api/incidents/{incident_id}/workflow")
            assert response.status_code < 500
        state = client.get(f"/api/incidents/{incident_id}").json()
        assert state["status"] != "CLOSED"
        assert client.app.state.runtime.store.table_counts()["approvals"] == 0


def test_adv25_reference_mutation_fails_integrity_validation() -> None:
    """A25-E: any semantic reference artifact mutation must fail its checksum validation."""
    path = Path("artifacts/reference-demo/reference-incident-v1.json")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    tampered = deepcopy(artifact)
    tampered["event_count"] += 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_reference_incident_artifact(tampered)


def test_adv25_live_cache_provenance_identities_are_stable() -> None:
    """A25-E: a cache HIT may change only cache_status, never frozen-input identities."""
    client = TestClient(create_app())
    first = client.get("/api/live-example-inputs")
    second = client.get("/api/live-example-inputs")
    assert first.status_code == second.status_code == 200
    first_payload, second_payload = first.json(), second.json()
    assert first_payload["cache_status"] == "MISS"
    assert second_payload["cache_status"] == "HIT"
    for key in ("execution_mode", "input_source", "input_sha256", "network_sha256", "scenario_sha256"):
        assert first_payload[key] == second_payload[key]
