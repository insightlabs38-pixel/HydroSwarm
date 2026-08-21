"""SUB-12.1 #22: verify the cross-platform native CI workflow's structure
-- the matrix covers every required platform, each job runs the real
documented setup script (not a mocked/fixture path), and each job proves
a real running server via native_ci_smoke.py. A real GitHub Actions run
could not be executed from this sandbox; these tests verify everything
that does not require actually dispatching the workflow.

macOS Intel/x86_64 is deliberately absent from REQUIRED_PLATFORMS: a real
CI run on a genuine macos-15-intel runner confirmed `pip install` fails
outright for the frozen `torch>=2.5` requirement -- upstream PyTorch has
published no macOS x86_64 wheel beyond 2.2.x, so that platform cannot
install the current runtime at all (not a HydroSwarm portability defect).
Do not re-add it here without first confirming upstream PyTorch has
resumed macOS x86_64 wheel distribution for the pinned torch requirement.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "native-cross-platform.yml"

# platform.machine() spelling differs across OSes -- see the workflow's own
# architecture-check step for why this can't just be "same string everywhere".
REQUIRED_PLATFORMS = {
    "linux-x86_64": {"os": "ubuntu-latest", "expected_arch": "x86_64"},
    "linux-arm64": {"os": "ubuntu-24.04-arm", "expected_arch": "aarch64"},
    "windows-x86_64": {"os": "windows-latest", "expected_arch": "AMD64"},
    "macos-arm64": {"os": "macos-15", "expected_arch": "arm64"},
}


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as handle:
        return yaml.safe_load(handle)


def test_workflow_file_exists_and_parses() -> None:
    assert WORKFLOW_PATH.is_file()
    workflow = _load_workflow()
    assert "native-verify" in workflow["jobs"]


def test_matrix_covers_every_required_platform_with_correct_runner_and_arch() -> None:
    workflow = _load_workflow()
    matrix = workflow["jobs"]["native-verify"]["strategy"]["matrix"]["include"]
    by_name = {
        entry["name"]: {"os": entry["os"], "expected_arch": entry["expected_arch"]}
        for entry in matrix
    }
    assert by_name == REQUIRED_PLATFORMS


def test_macos_intel_is_not_claimed_as_a_supported_native_platform() -> None:
    """Locks the platform-support claim down: macos-x86_64 must not
    silently reappear in the matrix (see module docstring for why)."""
    workflow = _load_workflow()
    matrix = workflow["jobs"]["native-verify"]["strategy"]["matrix"]["include"]
    names = {entry["name"] for entry in matrix}
    assert "macos-x86_64" not in names
    oses = {entry["os"] for entry in matrix}
    assert "macos-15-intel" not in oses
    assert not any(o.startswith("macos-13") or o == "macos-12" for o in oses)


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
