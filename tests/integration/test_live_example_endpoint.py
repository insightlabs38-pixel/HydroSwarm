"""SUB-12.1 P1 #4: GET /api/live-example-inputs serves real, WNTR-simulated
reference inputs for the 'Run Live Example' judge path, through the same
env-var-first path resolver pattern as the frozen V4 bundle (SUB-1) and
the reference-demo artifact (SUB-4/5)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hydroswarm.api import create_app

#: Drives a real bounded WNTR simulation on first call -- see
#: pyproject.toml's real_simulation marker docstring.
pytestmark = pytest.mark.real_simulation


def test_live_example_inputs_serves_real_reference_data() -> None:
    client = TestClient(create_app())

    response = client.get("/api/live-example-inputs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["network_filename"] == "live_example_network.inp"
    assert payload["candidate_signatures_mg_l"]
    assert payload["initial_observation"]["node_id"]


def test_live_example_inputs_honors_explicit_frozen_scenario_dir_override(tmp_path) -> None:
    scenario_dir = tmp_path / "custom-frozen"
    scenario_dir.mkdir()
    missing_client = TestClient(create_app(frozen_scenario_dir=scenario_dir))

    response = missing_client.get("/api/live-example-inputs")

    assert response.status_code == 404
    assert "frozen scenario fixtures not found" in response.json()["detail"]


def test_live_example_inputs_is_cached_across_requests_within_one_app_instance() -> None:
    client = TestClient(create_app())

    first = client.get("/api/live-example-inputs").json()
    second = client.get("/api/live-example-inputs").json()

    assert first["cache_status"] == "MISS"
    assert second["cache_status"] == "HIT"
    for key in ("execution_mode", "input_source", "computed_at", "input_sha256", "network_sha256", "scenario_sha256"):
        assert first[key] == second[key]
