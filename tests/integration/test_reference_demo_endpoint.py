"""SUB-5: GET /api/reference-demo serves the SUB-4 governed artifact through
the same env-var-first path resolver used for the frozen V4 bundle (SUB-1),
and fails closed (404) rather than silently returning nothing when the
artifact is missing."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from hydroswarm.api import create_app


def test_reference_demo_serves_the_real_committed_artifact() -> None:
    client = TestClient(create_app())

    response = client.get("/api/reference-demo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["reference_id"] == "reference-incident-v1"
    assert len(payload["milestones"]) == 11
    assert payload["final_event_hash"]


def test_reference_demo_rejects_unverified_explicit_path_override(tmp_path) -> None:
    custom_path = tmp_path / "custom-reference.json"
    custom_path.write_text(json.dumps({"reference_id": "custom-test"}), encoding="utf-8")
    client = TestClient(create_app(reference_demo_path=custom_path))

    response = client.get("/api/reference-demo")

    assert response.status_code == 503


def test_reference_demo_fails_closed_when_artifact_missing(tmp_path) -> None:
    missing_path = tmp_path / "does-not-exist.json"
    client = TestClient(create_app(reference_demo_path=missing_path))

    response = client.get("/api/reference-demo")

    assert response.status_code == 404
    assert "reference-demo artifact not found" in response.json()["detail"]


def test_reference_demo_fails_closed_on_invalid_json(tmp_path) -> None:
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not valid json", encoding="utf-8")
    client = TestClient(create_app(reference_demo_path=corrupt_path))

    response = client.get("/api/reference-demo")

    assert response.status_code == 500
    assert "not valid JSON" in response.json()["detail"]
