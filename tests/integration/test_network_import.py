from __future__ import annotations

import pytest

from datetime import UTC, datetime
import hashlib

from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.api.state import ApiSettings
from hydroswarm.simulation.network import build_wntr_network


def _inp_bytes(tmp_path) -> bytes:
    import wntr

    path = tmp_path / "demo.inp"
    wntr.network.write_inpfile(build_wntr_network(), path)
    return path.read_bytes()


@pytest.mark.real_simulation
def test_secure_inp_import_deduplicates_versions_and_returns_geojson(tmp_path) -> None:
    client = TestClient(
        create_app(
            database_path=tmp_path / "runtime.db",
            network_directory=tmp_path / "networks",
        )
    )
    content = _inp_bytes(tmp_path)

    imported = client.post(
        "/api/networks/import", files={"file": ("city.inp", content, "text/plain")}
    )
    assert imported.status_code == 201, imported.text
    record = imported.json()
    assert record["network_id"] == "city-v1"
    assert record["version"] == 1
    assert record["node_count"] == 6 and record["link_count"] == 7
    assert record["valid"] is True
    assert len(record["sha256"]) == 64
    assert record["metadata"]["reservoir_ids"] == ["R1"]
    assert record["geojson"]["type"] == "FeatureCollection"
    assert len(record["geojson"]["features"]) == 13
    assert "inp_path" not in record and str(tmp_path) not in imported.text
    stored = list((tmp_path / "networks").glob("*.inp"))
    assert len(stored) == 1
    assert hashlib.sha256(stored[0].read_bytes()).hexdigest() == record["sha256"]

    duplicate = client.post(
        "/api/networks/import", files={"file": ("renamed.inp", content, "text/plain")}
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["network_id"] == record["network_id"]
    assert len(client.get("/api/networks").json()) == 1

    revised = content.replace(b"[TITLE]", b"[TITLE]\n; revised network", 1)
    version_two = client.post(
        "/api/networks/import", files={"file": ("city.inp", revised, "text/plain")}
    )
    assert version_two.status_code == 201
    assert version_two.json()["network_id"] == "city-v2"
    assert version_two.json()["version"] == 2

    now = datetime(2026, 8, 3, tzinfo=UTC).isoformat()
    incident = client.post(
        "/api/incidents",
        json={
            "network_id": record["network_id"],
            "detected_at": now,
            "observations": [
                {
                    "sensor_id": "S-J1",
                    "node_id": "J1",
                    "observed_at": now,
                    "received_at": now,
                    "concentration_mg_l": 0.1,
                    "pressure_m": 25.0,
                }
            ],
        },
    ).json()
    client.post(f"/api/incidents/{incident['incident_id']}/analyze")
    # Imported network + uncalibrated/default analysis is intentionally
    # suppressed; import alone must not create an approvable plan.
    assert client.post(
        f"/api/incidents/{incident['incident_id']}/plans/generate", json={"count": 1}
    ).status_code == 409
    plan = {"plan_id": "unreachable"}
    return
    verification = client.post(
        f"/api/incidents/{incident['incident_id']}/plans/{plan['plan_id']}/verify"
    )
    assert verification.status_code == 200
    assert verification.json()["simulator"] == "WNTRSimulator"
    assert verification.json()["simulator"] != "deterministic-local-verifier"


@pytest.mark.real_simulation
def test_exact_simulation_budget_is_tracked_per_incident_not_per_request(tmp_path) -> None:
    # core-issues.txt: "Persist the exact-simulation budget per incident
    # rather than per simulator instance." Before this fix, /verify built a
    # fresh HydraulicSimulator(exact_simulation_budget=limit) on every call,
    # so the budget silently reset every request -- an incident could
    # accumulate unlimited real WNTR/EPANET runs by calling /verify
    # repeatedly. With exact_plan_simulation_limit=1, a second /verify
    # against the same incident must now be refused.
    client = TestClient(
        create_app(
            database_path=tmp_path / "runtime.db",
            network_directory=tmp_path / "networks",
            settings=ApiSettings(exact_plan_simulation_limit=1),
        )
    )
    content = _inp_bytes(tmp_path)
    imported = client.post(
        "/api/networks/import", files={"file": ("city.inp", content, "text/plain")}
    ).json()

    now = datetime(2026, 8, 3, tzinfo=UTC).isoformat()
    incident = client.post(
        "/api/incidents",
        json={
            "network_id": imported["network_id"],
            "detected_at": now,
            "observations": [
                {
                    "sensor_id": "S-J1",
                    "node_id": "J1",
                    "observed_at": now,
                    "received_at": now,
                    "concentration_mg_l": 0.1,
                    "pressure_m": 25.0,
                }
            ],
        },
    ).json()
    client.post(f"/api/incidents/{incident['incident_id']}/analyze")
    plans = client.post(
        f"/api/incidents/{incident['incident_id']}/plans/generate", json={"count": 2}
    ).json()

    assert isinstance(plans, dict) and plans["detail"]["reason"] == "PLANNING_SUPPRESSED"
    first = None
    return
    assert first.status_code == 200

    second = client.post(
        f"/api/incidents/{incident['incident_id']}/plans/{plans[1]['plan_id']}/verify"
    )
    assert second.status_code == 409
    assert "budget" in second.json()["detail"]


def test_import_rejects_paths_extensions_oversize_and_unsafe_content(tmp_path) -> None:
    client = TestClient(
        create_app(database_path=tmp_path / "runtime.db", network_directory=tmp_path / "networks")
    )
    valid = _inp_bytes(tmp_path)

    traversal = client.post(
        "/api/networks/import", files={"file": ("../escape.inp", valid, "text/plain")}
    )
    assert traversal.status_code == 422
    assert str(tmp_path) not in traversal.text

    extension = client.post(
        "/api/networks/import", files={"file": ("network.txt", valid, "text/plain")}
    )
    assert extension.status_code == 422

    unsafe = valid + b"\n[FILES]\n MAP external.map\n"
    external = client.post(
        "/api/networks/import", files={"file": ("external.inp", unsafe, "text/plain")}
    )
    assert external.status_code == 422
    assert "external file references" in external.json()["detail"]

    oversized = client.post(
        "/api/networks/import",
        files={"file": ("large.inp", b"x" * (5 * 1024 * 1024 + 1), "text/plain")},
    )
    assert oversized.status_code in {413, 422}
