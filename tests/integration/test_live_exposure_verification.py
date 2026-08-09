"""important-issues.txt requirement 18: "live /verify returns real exposure
values for a deterministic fixture" -- exercised through the actual
non-injected (real WNTR/EPANET) `/api/incidents/{id}/plans/{id}/verify`
endpoint, not the test-only injectable `verifier=` callable other API tests
use (that seam bypasses PlanVerifier/HydraulicSimulator entirely, so it
cannot exercise the fix)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
import wntr
from fastapi.testclient import TestClient

from hydroswarm.api.app import create_app
from hydroswarm.domain import CandidateSet
from hydroswarm.simulation import build_wntr_network

pytest.importorskip("wntr")

NOW = datetime(2026, 8, 3, tzinfo=UTC)


pytestmark = pytest.mark.real_simulation


def _import_network(client: TestClient, tmp_path) -> str:
    model = build_wntr_network()
    model.options.time.duration = 2 * 3_600
    network_path = tmp_path / "net.inp"
    wntr.network.write_inpfile(model, str(network_path))
    with open(network_path, "rb") as handle:
        response = client.post(
            "/api/networks/import", files={"file": ("net.inp", handle, "text/plain")}
        )
    assert response.status_code == 201, response.text
    return response.json()["network_id"]


def test_live_verify_returns_real_exposure_when_candidates_are_calibrated(tmp_path) -> None:
    app = create_app(
        network_directory=tmp_path / "networks",
        ledger_path=tmp_path / "ledger.db",
        database_path=tmp_path / "db.sqlite",
    )
    client = TestClient(app)
    network_id = _import_network(client, tmp_path)

    created = client.post(
        "/api/incidents",
        json={
            "network_id": network_id,
            "detected_at": NOW.isoformat(),
            "observations": [
                {
                    "sensor_id": "S-J2",
                    "node_id": "J2",
                    "observed_at": NOW.isoformat(),
                    "received_at": NOW.isoformat(),
                    "concentration_mg_l": 0.05,
                    "pressure_m": 20.0,
                }
            ],
            "contamination_threshold_mg_l": 0.001,
        },
    )
    assert created.status_code == 201
    incident_id = UUID(created.json()["incident_id"])
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200

    #: Force a CALIBRATED candidate set (this test's own deterministic
    #: fixture, standing in for a real trained+calibrated pipeline, which is
    #: out of scope here) so `_runtime_evaluation_context` builds a real
    #: bounded hypothesis set instead of returning None.
    record = app.state.runtime.incidents[incident_id]
    record.state = record.state.model_copy(
        update={
            "candidates": CandidateSet(
                node_probabilities={"J1": 0.7, "J3": 0.3},
                node_ids=("J1", "J3"),
                calibrated=True,
            )
        }
    )

    plans_response = client.post(
        f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}
    )
    assert plans_response.status_code == 200
    plan = plans_response.json()[0]

    #: A single verify() call over 2 bounded hypotheses consumes 1 (cached
    #: baseline) + 2 (one EPANET run per hypothesis) = 3 real simulations --
    #: exactly ApiSettings.exact_plan_simulation_limit's default budget for
    #: this whole incident. This is itself a real, honest consequence of
    #: important-issues.txt requirement 10 (no longer silently 1 simulator
    #: call per verification) worth noting for operational budget tuning,
    #: not something this test works around.
    response = client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify")
    assert response.status_code == 200
    result = response.json()

    assert result["consequences"] is not None
    assert result["consequences"]["exposure_evaluated"] is True
    assert result["worst_case_consequences"] is not None
    provenance = result["evaluation_provenance"]
    assert provenance is not None
    assert provenance["hypotheses"]["mode"] == "hypotheses"
    assert {h["profile"]["source_node_id"] for h in provenance["hypotheses"]["hypotheses"]} == {"J1", "J3"}
    #: important-issues.txt requirement 10: two hypotheses -> two real
    #: EPANET runs, never silently reported as one simulator call.
    assert provenance["exact_simulation_count"] == 2
    # A genuinely contaminated fixture must show real, nonzero exposure.
    assert result["consequences"]["contaminant_mass_consumed_mg"] > 0


def test_live_verify_falls_back_to_hydraulic_only_without_calibrated_candidates(tmp_path) -> None:
    """important-issues.txt requirement 12: before calibration is valid,
    /verify must abstain from claiming a measured exposure value rather than
    silently reporting a Pydantic-default zero as though it were real."""

    app = create_app(
        network_directory=tmp_path / "networks",
        ledger_path=tmp_path / "ledger.db",
        database_path=tmp_path / "db.sqlite",
    )
    client = TestClient(app)
    network_id = _import_network(client, tmp_path)

    created = client.post(
        "/api/incidents",
        json={
            "network_id": network_id,
            "detected_at": NOW.isoformat(),
            "observations": [
                {
                    "sensor_id": "S-J2",
                    "node_id": "J2",
                    "observed_at": NOW.isoformat(),
                    "received_at": NOW.isoformat(),
                    "concentration_mg_l": 0.05,
                    "pressure_m": 20.0,
                }
            ],
        },
    )
    incident_id = UUID(created.json()["incident_id"])
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200

    plans_response = client.post(
        f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}
    )
    plan = plans_response.json()[0]
    response = client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify")

    assert response.status_code == 200
    body = response.json()
    assert body["consequences"]["exposure_evaluated"] is False
    assert body["worst_case_consequences"] is None
    assert body["evaluation_provenance"] is None


# core-issues5.txt Section 8 (P0 safety fix): "one plan verification" and
# "one EPANET simulation" are not equivalent -- the incident's own budget
# counters must reflect this distinction and cannot be reset by retrying.


def test_incident_budget_counters_update_after_multi_hypothesis_verify(tmp_path) -> None:
    app = create_app(
        network_directory=tmp_path / "networks",
        ledger_path=tmp_path / "ledger.db",
        database_path=tmp_path / "db.sqlite",
    )
    client = TestClient(app)
    network_id = _import_network(client, tmp_path)

    created = client.post(
        "/api/incidents",
        json={
            "network_id": network_id,
            "detected_at": NOW.isoformat(),
            "observations": [
                {
                    "sensor_id": "S-J2",
                    "node_id": "J2",
                    "observed_at": NOW.isoformat(),
                    "received_at": NOW.isoformat(),
                    "concentration_mg_l": 0.05,
                    "pressure_m": 20.0,
                }
            ],
            "contamination_threshold_mg_l": 0.001,
        },
    )
    incident_id = UUID(created.json()["incident_id"])
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200

    # Freshly created incident: full budget, no verifications yet.
    initial = client.get(f"/api/incidents/{incident_id}").json()
    assert initial["remaining_epanet_budget"] == 3  # ApiSettings.exact_plan_simulation_limit default
    assert initial["plans_exactly_verified"] == 0
    assert initial["exact_simulation_cache_hits"] == 0

    record = app.state.runtime.incidents[incident_id]
    record.state = record.state.model_copy(
        update={
            "candidates": CandidateSet(
                node_probabilities={"J1": 0.7, "J3": 0.3},
                node_ids=("J1", "J3"),
                calibrated=True,
            )
        }
    )
    plan = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}).json()[0]

    response = client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify")
    assert response.status_code == 200
    provenance = response.json()["evaluation_provenance"]
    assert provenance["hypotheses_per_plan"] == 2

    # 1 cached baseline + 2 per-hypothesis EPANET runs = 3 -- exactly the
    # default budget, all consumed by this ONE plan verification.
    after_first = client.get(f"/api/incidents/{incident_id}").json()
    assert after_first["exact_simulations_used"] == 3
    assert after_first["remaining_epanet_budget"] == 0
    assert after_first["plans_exactly_verified"] == 1

    # Retrying (even a different plan) must not reclaim any budget --
    # a fresh HydraulicSimulator is constructed per request, but the
    # incident-level counter is what actually gates it.
    second_plan = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1}).json()[0]
    retry = client.post(f"/api/incidents/{incident_id}/plans/{second_plan['plan_id']}/verify")
    assert retry.status_code == 409

    still_exhausted = client.get(f"/api/incidents/{incident_id}").json()
    assert still_exhausted["remaining_epanet_budget"] == 0
    assert still_exhausted["exact_simulations_used"] == 3
    # The failed/blocked retry must not be counted as a real verification
    # attempt (it never reached the verifier at all).
    assert still_exhausted["plans_exactly_verified"] == 1
