"""SUB-12.1 #22: verify the cross-platform native CI workflow's structure
-- the matrix covers every required platform, each job runs the real
documented setup script (not a mocked/fixture path), and each job proves
a real running server via native_ci_smoke.py. A real GitHub Actions run
could not be executed from this sandbox; these tests verify everything
that does not require actually dispatching the workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "native-cross-platform.yml"

REQUIRED_PLATFORMS = {
    "linux-x86_64": "ubuntu-latest",
    "linux-arm64": "ubuntu-24.04-arm",
    "windows-x86_64": "windows-latest",
    "macos-arm64": "macos-14",
    "macos-x86_64": "macos-13",
}


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as handle:
        return yaml.safe_load(handle)


def test_workflow_file_exists_and_parses() -> None:
    assert WORKFLOW_PATH.is_file()
    workflow = _load_workflow()
    assert "native-verify" in workflow["jobs"]


def test_matrix_covers_every_required_platform() -> None:
    workflow = _load_workflow()
    matrix = workflow["jobs"]["native-verify"]["strategy"]["matrix"]["include"]
    by_name = {entry["name"]: entry["os"] for entry in matrix}
    assert by_name == REQUIRED_PLATFORMS


def test_matrix_does_not_fail_fast_so_one_platform_cannot_hide_another() -> None:
    workflow = _load_workflow()
    assert workflow["jobs"]["native-verify"]["strategy"]["fail-fast"] is False


def test_each_platform_runs_its_own_real_documented_setup_script() -> None:
    text = WORKFLOW_PATH.read_text()
    for script in (
        "setup_hydroswarm_linux.sh",
        "setup_hydroswarm_macos.sh",
        "setup_hydroswarm_windows.ps1",
    ):
        assert script in text


def test_each_platform_runs_the_real_http_smoke_script() -> None:
    text = WORKFLOW_PATH.read_text()
    assert "native_ci_smoke.py" in text


def test_smoke_script_hits_health_and_reference_demo_and_shuts_down_cleanly() -> None:
    smoke = (PROJECT_ROOT / "scripts" / "native_ci_smoke.py").read_text()
    assert "/api/health" in smoke
    assert "/api/reference-demo" in smoke
    # Kills the whole process group/tree, not just the immediate child --
    # `hydroswarm.cli start` blocks on a uvicorn *grandchild* subprocess,
    # so a plain process.terminate() would leave uvicorn (and the bound
    # port) running. See _stop()'s own docstring for the full story.
    assert "killpg" in smoke or "taskkill" in smoke
