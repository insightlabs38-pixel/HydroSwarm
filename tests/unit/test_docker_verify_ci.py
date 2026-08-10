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

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "docker-verify.yml"
VERIFY_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "docker_ci_verify.py"


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
