"""SUB-12.1 #23: verify the Docker PR-gate workflow's structure -- real
amd64+arm64 builds, release-like hardening, no publishing, and every
required runtime check wired in. A real `docker build`/`docker run`
could not be executed in this sandbox (see
reports/submission-readiness/sub3-docker-sandbox-limitation.md); these
tests verify everything that does not require a working container
runtime, plus the underlying verify script's HTTP-driving logic, which
was independently validated end to end against a real locally-launched
`hydroswarm start` server (not a container, but the identical
application/API surface the script talks to)."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from uuid import uuid4

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "docker-verify.yml"
VERIFY_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "docker_ci_verify.py"


def _load_verify_script_module():
    spec = importlib.util.spec_from_file_location("docker_ci_verify_under_test", VERIFY_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as handle:
        return yaml.safe_load(handle)


def test_workflow_file_exists_and_parses() -> None:
    assert WORKFLOW_PATH.is_file()
    workflow = _load_workflow()
    assert "docker-verify" in workflow["jobs"]


def test_triggers_are_pull_request_and_dispatch_only() -> None:
    workflow = _load_workflow()
    # PyYAML's safe_load parses the bare `on:` key as the boolean True
    # (YAML 1.1's implicit boolean resolution), not the string "on".
    triggers = set(workflow[True])
    assert triggers == {"pull_request", "workflow_dispatch"}


def test_builds_both_amd64_and_arm64() -> None:
    workflow = _load_workflow()
    platforms = [
        entry["platform"]
        for entry in workflow["jobs"]["docker-verify"]["strategy"]["matrix"]["include"]
    ]
    assert set(platforms) == {"linux/amd64", "linux/arm64"}


def test_never_pushes_or_logs_in_to_a_registry() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["docker-verify"]["steps"]
    assert not any(step.get("uses", "").startswith("docker/login-action") for step in steps)
    assert "env" not in workflow["jobs"]["docker-verify"] or "ghcr.io" not in str(
        workflow["jobs"]["docker-verify"].get("env", {})
    )
    build_steps = [step for step in steps if step.get("name", "").startswith("Build image")]
    assert build_steps and "docker build" in build_steps[0]["run"]


def test_pr_docker_verification_uses_native_runners_without_qemu_or_buildx() -> None:
    workflow = _load_workflow()
    assert workflow["jobs"]["docker-verify"]["runs-on"] == "${{ matrix.runner }}"
    runners = {
        entry["platform"]: entry["runner"]
        for entry in workflow["jobs"]["docker-verify"]["strategy"]["matrix"]["include"]
    }
    assert runners == {"linux/amd64": "ubuntu-24.04", "linux/arm64": "ubuntu-24.04-arm"}
    uses = "\n".join(step.get("uses", "") for step in workflow["jobs"]["docker-verify"]["steps"])
    assert "setup-qemu" not in uses
    assert "setup-buildx" not in uses


def test_runs_with_release_like_hardening() -> None:
    text = WORKFLOW_PATH.read_text()
    for flag in ("--read-only", "--tmpfs /tmp", "--cap-drop ALL", "--security-opt no-new-privileges"):
        assert flag in text
    assert "-p 127.0.0.1:8765:8765" in text


def test_wires_every_required_runtime_check() -> None:
    text = WORKFLOW_PATH.read_text()
    assert "docker_ci_verify.py health" in text
    assert "self-test --strict" in text
    assert "real EPANET quality smoke" in text
    assert "docker_ci_verify.py live-workflow" in text
    assert "docker restart" in text
    assert "docker_ci_verify.py verify-persistence" in text
    assert "--dns 0.0.0.0" in text


def test_cleans_up_containers_and_volumes_even_on_failure() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["docker-verify"]["steps"]
    cleanup_steps = [step for step in steps if step.get("name") == "Clean up"]
    assert cleanup_steps, "expected an always()-guarded cleanup step"
    assert cleanup_steps[0]["if"] == "always()"


def test_verify_script_covers_the_full_real_production_lifecycle() -> None:
    text = VERIFY_SCRIPT_PATH.read_text()
    for endpoint in (
        "/api/health",
        "/api/readiness",
        "/api/reference-demo",
        "/api/live-example-inputs",
        "/api/networks/import",
        "/api/incidents",
        "/samples/recommend",
        "/plans/generate",
        "/verify",
        "/approve",
    ):
        assert endpoint in text


def test_live_workflow_approves_the_first_current_verified_plan(monkeypatch) -> None:
    """A verified plan creates the sole current approval boundary for the CI run."""
    module = _load_verify_script_module()
    incident_id = str(uuid4())
    first_plan_id, second_plan_id = str(uuid4()), str(uuid4())
    requested_paths: list[str] = []

    def request(_base_url, _method, path, **_kwargs):
        requested_paths.append(path)
        if path == "/api/live-example-inputs":
            return 200, {
                "network_filename": "network.inp",
                "network_inp_text": "[TITLE]",
                "candidate_signatures_mg_l": {"J1": 1.0},
                "initial_observation": {
                    "sensor_id": "S-J1",
                    "node_id": "J1",
                    "concentration_mg_l": 1.0,
                    "pressure_m": 25.0,
                },
                "contamination_threshold_mg_l": 0.1,
            }
        if path == "/api/networks/import":
            return 201, {"network_id": "network-id"}
        if path == "/api/incidents":
            return 201, {"incident_id": incident_id}
        if path.endswith("/samples/recommend"):
            return 200, {"node_id": "J1"}
        if path.endswith("/plans/generate"):
            return 200, [{"plan_id": first_plan_id}, {"plan_id": second_plan_id}]
        if path.endswith(f"/plans/{first_plan_id}/verify"):
            return 200, {"decision": "VERIFIED"}
        if path.endswith(f"/plans/{first_plan_id}/approve"):
            return 200, {"receipt_id": str(uuid4())}
        if path.endswith("/view"):
            return 200, {"runtime_mode": "LIVE", "status": "CLOSED"}
        if path.endswith("/analyze") or path.endswith("/samples"):
            return 200, {}
        raise AssertionError(f"unexpected request: {path}")

    monkeypatch.setattr(module, "_wait_for_health", lambda _base_url: None)
    monkeypatch.setattr(module, "_request", request)
    monkeypatch.setattr(module, "_utc_now", lambda: "2026-01-01T00:00:00+00:00")

    assert module.cmd_live_workflow(argparse.Namespace(base_url="http://test", state_file=None)) == 0
    assert f"/api/incidents/{incident_id}/plans/{first_plan_id}/verify" in requested_paths
    assert f"/api/incidents/{incident_id}/plans/{second_plan_id}/verify" not in requested_paths
    assert f"/api/incidents/{incident_id}/plans/{first_plan_id}/approve" in requested_paths


def test_persistence_check_does_not_reanalyze_a_closed_incident(monkeypatch, tmp_path) -> None:
    """CLOSED is terminal, so restart validation reads durable state and audit history."""
    module = _load_verify_script_module()
    incident_id = str(uuid4())
    state_file = tmp_path / "docker-ci-state.json"
    state_file.write_text(
        f'{{"incident_id":"{incident_id}","runtime_mode":"LIVE","status":"CLOSED"}}'
    )
    requested_paths: list[str] = []

    def request(_base_url, _method, path, **_kwargs):
        requested_paths.append(path)
        if path == f"/api/incidents/{incident_id}":
            return 200, {"status": "CLOSED"}
        if path == f"/api/incidents/{incident_id}/replay":
            return 200, {"chain_valid": True, "events": [{"event_type": "PLAN_APPROVED"}]}
        raise AssertionError(f"unexpected request: {path}")

    monkeypatch.setattr(module, "_wait_for_health", lambda _base_url: None)
    monkeypatch.setattr(module, "_request", request)

    assert module.cmd_verify_persistence(
        argparse.Namespace(base_url="http://test", state_file=str(state_file))
    ) == 0
    assert not any(path.endswith("/analyze") for path in requested_paths)
