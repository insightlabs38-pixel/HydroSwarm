"""Contract tests for GET /api/incidents/{incident_id}/view (overnight-plan.txt Task 3.2).

Mirrors the real HybridInferencePipeline fixture from
tests/e2e/test_iterative_pipeline.py -- the same model/artifact wiring, driven
through the real HTTP API this time -- so the view endpoint is exercised
against a genuine IncidentAnalysisResult rather than the DEMO_FALLBACK path.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
import hashlib
from uuid import UUID

import torch
import wntr
from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.calibration import CalibrationArtifact, CalibrationReport
from hydroswarm.calibration.conformal import CALIBRATION_SCHEMA_VERSION
from hydroswarm.classical import (
    SignatureArtifact,
    SignatureCacheKey,
    SignatureLibrary,
    SourceHypothesis,
)
from hydroswarm.domain import CandidateSet, ConsequenceMetrics, PlanDecision, PlanVerification
from hydroswarm.inference import MODEL_VERSION, HybridInferencePipeline
from hydroswarm.preprocessing import DEFAULT_FEATURE_SCHEMA
from hydroswarm.simulation import HydraulicSimulator, build_wntr_network


NOW = datetime(2026, 8, 3, tzinfo=UTC)


class PriorFollowingModel:
    """Deterministic stand-in neural head: matches test_iterative_pipeline.py."""

    def eval(self):
        return self

    def __call__(self, batch):
        prior = batch["classical_prior"].clamp_min(1e-7)
        nodes = prior.shape[1]
        return {
            "source_node_logits": prior.log(),
            "evidence_sufficiency": torch.tensor([[0.9]]),
            "expected_information_gain": torch.full((1, nodes), 0.05),
            "ood_logits": torch.tensor([[4.0, 0.0, -4.0]]),
        }


def _build_pipeline():
    network = build_wntr_network()
    network.options.time.duration = 3600
    model = PriorFollowingModel()
    hypotheses = tuple(
        SourceHypothesis(node, 0, 60, 1.0, "nominal") for node in ("J1", "J2", "J3")
    )
    signatures = {
        "J1": [[0.0, 0.0, 0.0], [1.00, 0.15, 0.05]],
        "J2": [[0.0, 0.0, 0.0], [0.78, 1.00, 0.10]],
        "J3": [[0.0, 0.0, 0.0], [0.65, 0.20, 1.00]],
    }
    library = SignatureLibrary()
    for hypothesis in hypotheses:
        library.add(hypothesis.identifier, signatures[hypothesis.source_node])
    artifact = SignatureArtifact(
        key=SignatureCacheKey("a" * 64, "b" * 64, "test", "c" * 64, "d" * 64),
        library=library,
        hypotheses=hypotheses,
        sensor_nodes=("J1", "J2", "J3"),
        sample_times_seconds=(0, 3600),
        cache_hit=True,
        artifact_hash="e" * 64,
    )
    calibration = CalibrationArtifact(
        schema_version=CALIBRATION_SCHEMA_VERSION, alpha=0.1,
        model_hash=HybridInferencePipeline._fingerprint_model(model),
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint, dataset_manifest_hash="f" * 64,
        global_scores=(0.8, 0.8, 0.8, 0.8), mondrian_scores={}, network_scores={},
        report=CalibrationReport(1.0, 1.0, 0.0, {}, {}, 4),
    )
    pipeline = HybridInferencePipeline(
        simulator=HydraulicSimulator(network), signature_artifact=artifact, model=model,
        calibration_artifact=calibration,
        maximum_planning_candidates=1,
    )
    return pipeline, network


def _observation(sensor_id: str, node_id: str, offset_seconds: float, concentration: float) -> dict[str, object]:
    at = (NOW + timedelta(seconds=offset_seconds)).isoformat()
    return {
        "sensor_id": sensor_id,
        "node_id": node_id,
        "observed_at": at,
        "received_at": at,
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
        simulator_version="1.2.3",
        state_hash=hashlib.sha256(str(state.incident_id).encode()).hexdigest(),
        consequences=ConsequenceMetrics(
            minimum_pressure_m=25.0,
            service_availability=0.97,
            operation_count=len(plan.actions),
        ),
    )


def _import_network(client: TestClient, network) -> dict[str, object]:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".inp", delete=False) as handle:
        wntr.network.write_inpfile(network, handle.name)
        handle.seek(0)
        with open(handle.name, "rb") as fh:
            response = client.post(
                "/api/networks/import",
                files={"file": ("hydroswarm-demo.inp", fh, "text/plain")},
            )
    assert response.status_code == 201, response.text
    return response.json()


def _drive_to_planning_allowed(client: TestClient, incident_id: str) -> None:
    """Two analysis rounds -- mirrors test_iterative_pipeline.py's before/after
    -- pushes planning_allowed True via a real candidate-set contraction."""

    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200
    for offset, concentration in ((0.0, 0.0), (3600.0, 1.0)):
        added = client.post(
            f"/api/incidents/{incident_id}/samples",
            json=_observation("S2", "J2", offset, concentration),
        )
        assert added.status_code == 200
    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
    assert analysis["planning_allowed"] is True
    assert analysis["runtime_mode"] == "FULL_HYBRID"


def test_incident_view_requires_completed_analysis(tmp_path) -> None:
    pipeline, network = _build_pipeline()
    client = TestClient(create_app(pipeline_factory=pipeline, database_path=tmp_path / "runtime.sqlite3"))
    imported = _import_network(client, network)

    created = client.post(
        "/api/incidents",
        json={
            "network_id": imported["network_id"],
            "detected_at": NOW.isoformat(),
            "observations": [_observation("S1", "J1", 0.0, 0.0), _observation("S1", "J1", 3600.0, 0.78)],
        },
    )
    incident_id = created.json()["incident_id"]

    response = client.get(f"/api/incidents/{incident_id}/view")
    assert response.status_code == 409


def test_incident_view_requires_full_topology_metadata(tmp_path) -> None:
    """A network validated through the legacy manual /validate endpoint has no
    node/link topology to render; the live view must fail closed, not guess
    coordinates or omit fields silently."""

    pipeline, _network = _build_pipeline()
    client = TestClient(
        create_app(pipeline_factory=pipeline, verifier=_authority, database_path=tmp_path / "runtime.sqlite3")
    )
    manual = client.post(
        "/api/networks/manual-demo/validate",
        json={"node_ids": ["J1", "J2", "J3", "J4", "R1", "T1"], "link_count": 7},
    )
    assert manual.status_code == 200

    created = client.post(
        "/api/incidents",
        json={
            "network_id": "manual-demo",
            "detected_at": NOW.isoformat(),
            "observations": [_observation("S1", "J1", 0.0, 0.0), _observation("S1", "J1", 3600.0, 0.78)],
        },
    )
    incident_id = created.json()["incident_id"]
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200
    assert client.get(f"/api/incidents/{incident_id}/analysis").json()["runtime_mode"] == "FULL_HYBRID"

    response = client.get(f"/api/incidents/{incident_id}/view")
    assert response.status_code == 503


def test_incident_view_fails_closed_on_a_missing_provenance_hash(tmp_path) -> None:
    # core-issues.txt: "fail closed rather than returning empty provenance
    # hashes." A missing/empty required provenance hash must refuse the
    # view (500), never silently ship ProvenanceView(network_hash="", ...)
    # as if that were a trustworthy value.
    pipeline, network = _build_pipeline()
    app = create_app(pipeline_factory=pipeline, database_path=tmp_path / "runtime.sqlite3")
    client = TestClient(app)
    imported = _import_network(client, network)

    created = client.post(
        "/api/incidents",
        json={
            "network_id": imported["network_id"],
            "detected_at": NOW.isoformat(),
            "observations": [_observation("S1", "J1", 0.0, 0.0), _observation("S1", "J1", 3600.0, 0.78)],
        },
    )
    incident_id = created.json()["incident_id"]
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200

    record = app.state.runtime.incidents[UUID(incident_id)]
    corrupted = dict(record.analysis.provenance_hashes)
    del corrupted["model"]
    record.analysis = dataclasses.replace(record.analysis, provenance_hashes=corrupted)

    response = client.get(f"/api/incidents/{incident_id}/view")
    assert response.status_code == 500
    assert "provenance" in response.json()["detail"]


def test_incident_view_fails_closed_on_a_dangling_candidate_node_reference(tmp_path) -> None:
    # core-issues.txt: explicit runtime validation for the /view response.
    # A candidate set referencing a node that does not exist in this
    # incident's own topology is an assembler-consistency bug that per-field
    # type checks alone would not catch -- it must not ship silently.
    pipeline, network = _build_pipeline()
    app = create_app(pipeline_factory=pipeline, database_path=tmp_path / "runtime.sqlite3")
    client = TestClient(app)
    imported = _import_network(client, network)

    created = client.post(
        "/api/incidents",
        json={
            "network_id": imported["network_id"],
            "detected_at": NOW.isoformat(),
            "observations": [_observation("S1", "J1", 0.0, 0.0), _observation("S1", "J1", 3600.0, 0.78)],
        },
    )
    incident_id = created.json()["incident_id"]
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200

    record = app.state.runtime.incidents[UUID(incident_id)]
    record.state = record.state.model_copy(
        update={
            "candidates": CandidateSet(
                node_probabilities={"NOT-A-REAL-NODE": 1.0},
                node_ids=("NOT-A-REAL-NODE",),
            )
        }
    )

    response = client.get(f"/api/incidents/{incident_id}/view")
    assert response.status_code == 500
    assert "unknown nodes" in response.json()["detail"]


def test_incident_view_returns_complete_verified_contract(tmp_path) -> None:
    pipeline, network = _build_pipeline()
    client = TestClient(
        create_app(
            pipeline_factory=pipeline,
            verifier=_authority,
            database_path=tmp_path / "runtime.sqlite3",
        )
    )
    imported = _import_network(client, network)

    created = client.post(
        "/api/incidents",
        json={
            "network_id": imported["network_id"],
            "detected_at": NOW.isoformat(),
            "observations": [_observation("S1", "J1", 0.0, 0.0), _observation("S1", "J1", 3600.0, 0.78)],
            "maximum_samples": 3,
        },
    )
    incident_id = created.json()["incident_id"]
    _drive_to_planning_allowed(client, incident_id)

    generated = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 2})
    assert generated.status_code == 200
    plans = generated.json()
    assert len(plans) == 2

    verifications = []
    for plan in plans:
        verified = client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify")
        assert verified.status_code == 200
        verifications.append(verified.json())
        assert verified.json()["decision"] == "VERIFIED"

    approved_plan_id = plans[0]["plan_id"]
    approval = client.post(
        f"/api/incidents/{incident_id}/plans/{approved_plan_id}/approve",
        json={"approved": True, "operator_id": "operator-1"},
    )
    assert approval.status_code == 200

    response = client.get(f"/api/incidents/{incident_id}/view")
    assert response.status_code == 200
    view = response.json()

    # Identity and modes.
    assert view["schema_version"] == 1
    assert UUID(view["incident_id"]) is not None
    assert view["network_id"] == imported["network_id"]
    assert view["runtime_mode"] == "FULL_HYBRID"
    assert view["data_mode"] == "LIVE"
    assert view["controller_state"] == "CLOSED"

    # Provenance: model/checkpoint/calibration/network hashes, all real and non-placeholder.
    provenance = view["provenance"]
    assert provenance["model_version"] == MODEL_VERSION
    assert provenance["model_checkpoint_hash"] == HybridInferencePipeline._fingerprint_model(pipeline.model)
    assert provenance["calibration_version"] == CALIBRATION_SCHEMA_VERSION
    assert provenance["calibration_hash"] == pipeline.calibration_artifact.artifact_hash
    assert provenance["feature_schema_hash"] == DEFAULT_FEATURE_SCHEMA.fingerprint
    assert len(provenance["network_hash"]) == 64
    assert provenance["simulator"] == "test-authority"
    assert provenance["simulator_version"] == "1.2.3"

    # Candidate set / calibration validity.
    assert view["candidates"]["calibrated"] is True
    assert 0.0 <= view["calibration_alpha"] <= 1.0

    # Map topology: every node from the imported .inp file, with coordinates.
    node_ids = {node["node_id"] for node in view["nodes"]}
    assert node_ids == {"R1", "J1", "J2", "J3", "J4", "T1"}
    for node in view["nodes"]:
        assert len(node["coordinates"]) == 2
    link_ids = {link["link_id"] for link in view["links"]}
    assert len(link_ids) == 7

    # Sensor health reflects the accumulated live observations.
    sensor_nodes = {entry["node_id"] for entry in view["sensor_health"]}
    assert sensor_nodes == {"J1", "J2"}
    assert all(entry["health"] == "HEALTHY" for entry in view["sensor_health"])

    # Evidence, plans, counterfactuals, explanations, audit, and runtime metrics.
    # Three rounds: the initial /analyze, then one real reanalysis per
    # /samples call (each call appends exactly one observation and triggers
    # its own full reanalysis -- unlike the direct-pipeline test this mirrors,
    # which batches both new J2 readings into a single reanalyze_after_sample
    # call and so only sees one extra round).
    assert len(view["evidence_history"]) == 3
    returned_plan_ids = {plan["plan"]["plan_id"] for plan in view["plans"]}
    assert returned_plan_ids == {plan["plan_id"] for plan in plans}
    assert all(plan["verification"] is not None for plan in view["plans"])
    assert view["selected_plan_id"] == approved_plan_id
    assert view["recommended_plan_id"] in returned_plan_ids
    assert set(view["counterfactual_consequences"]) == returned_plan_ids
    for consequences in view["counterfactual_consequences"].values():
        assert consequences["service_availability"] == 0.97
    assert {payload["intent"] for payload in view["explanations"]} == {
        "WHY_SOURCE", "WHY_SAMPLE", "WHAT_CHANGED", "WHY_PLAN_REJECTED",
        "COMPARE_PLANS", "UNCERTAINTY_REMAINS", "WHICH_SENSOR_MATTERED",
    }
    event_types = {event["event_type"] for event in view["audit_events"]}
    assert {"INCIDENT_CREATED", "PLANS_GENERATED", "PLAN_VERIFIED", "PLAN_APPROVED"} <= event_types
    assert "hydraulic_simulation" in view["runtime_metrics_ms"]

    # extra="forbid" on every nested model means an unknown/undeclared field
    # anywhere in this response would already have failed FastAPI's
    # response_model validation before we got here -- this call succeeding
    # with status 200 is itself part of the contract.


def test_incident_view_rejects_unknown_fields_client_side_too() -> None:
    """Guards the contract itself: IncidentView must stay extra='forbid' so a
    server-side typo (an extra or misspelled field) fails loudly instead of
    silently shipping an incomplete/incorrect view."""

    from hydroswarm.api.state import IncidentView

    assert IncidentView.model_config.get("extra") == "forbid"
