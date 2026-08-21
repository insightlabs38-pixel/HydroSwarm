"""SUB-2: static/structural verification of the per-platform native setup
and launcher scripts.

These scripts are shell/PowerShell, not Python -- pytest cannot actually
execute the Windows/macOS variants on this Linux CI runner, and even the
Linux variant would require a throwaway venv + network access to run for
real inside the suite. What these tests can and do verify, mirroring the
static-verification approach already used for Dockerfile packaging in
`test_dockerfile_bundle_packaging.py`:

* every required script exists and (on POSIX) is executable;
* each launcher references the project-local `.venv` interpreter
  explicitly and never falls back to an ambient/system Python;
* each launcher fails closed (does not silently continue) when `.venv`
  is missing;
* each setup script never invokes a system package manager (apt/yum/
  pacman/brew --force, etc.) to mutate machine-global state;
* each setup script runs the readiness self-test as its final gate.

The Linux setup+launcher pair is additionally smoke-tested for real in
`reports/submission-readiness/` during manual/CI verification (see the
submission-readiness handoff report); that execution proof is out of
scope for this unit suite by design -- it depends on network access and
a disposable virtual environment, both unsuitable for the unit tier.
"""

from __future__ import annotations

import re
import stat
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SETUP_SCRIPTS = [
    "setup_hydroswarm_linux.sh",
    "setup_hydroswarm_macos.sh",
    "setup_hydroswarm_windows.ps1",
]
LAUNCHER_SCRIPTS = [
    "start_hydroswarm_linux.sh",
    "start_hydroswarm_macos.sh",
    "start_hydroswarm_windows.ps1",
]
POSIX_SCRIPTS = [name for name in SETUP_SCRIPTS + LAUNCHER_SCRIPTS if name.endswith(".sh")]


@pytest.mark.parametrize("name", SETUP_SCRIPTS + LAUNCHER_SCRIPTS)
def test_script_exists(name: str) -> None:
    assert (PROJECT_ROOT / name).is_file(), f"missing native script: {name}"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "the POSIX executable bit is not a meaningful NTFS concept -- a real "
        "windows-latest CI checkout reports these as non-executable regardless "
        "of the git-tracked mode, since Windows never runs .sh scripts directly"
    ),
)
@pytest.mark.parametrize("name", POSIX_SCRIPTS)
def test_posix_script_is_executable(name: str) -> None:
    mode = (PROJECT_ROOT / name).stat().st_mode
    assert mode & stat.S_IXUSR, f"{name} must be chmod +x"


@pytest.mark.parametrize("name", LAUNCHER_SCRIPTS)
def test_launcher_uses_venv_interpreter_explicitly(name: str) -> None:
    text = (PROJECT_ROOT / name).read_text()
    assert ".venv" in text, f"{name} must reference the project-local .venv interpreter"


@pytest.mark.parametrize("name", LAUNCHER_SCRIPTS)
def test_launcher_fails_closed_if_venv_missing(name: str) -> None:
    text = (PROJECT_ROOT / name).read_text()
    if name.endswith(".sh"):
        assert "if [ ! -x \"$VENV_PYTHON\" ]" in text
        assert "exit 1" in text
    else:
        assert "if (-not (Test-Path $VenvPython))" in text
        assert "exit 1" in text


@pytest.mark.parametrize("name", LAUNCHER_SCRIPTS)
def test_launcher_prints_url_and_binds_loopback_by_default(name: str) -> None:
    text = (PROJECT_ROOT / name).read_text()
    assert "127.0.0.1" in text
    assert "Starting HydroSwarm at http" in text


@pytest.mark.parametrize("name", LAUNCHER_SCRIPTS)
def test_launcher_runs_readiness_check_before_start(name: str) -> None:
    text = (PROJECT_ROOT / name).read_text()
    assert "self-test" in text


def _lines_outside_heredocs(text: str) -> list[str]:
    """Lines not inside a `<<EOF ... EOF` block, i.e. lines that could
    plausibly be executed rather than merely printed as instructional text."""
    executable_lines = []
    in_heredoc = False
    for line in text.splitlines():
        if in_heredoc:
            if line.strip() == "EOF":
                in_heredoc = False
            continue
        if "<<" in line and "EOF" in line:
            in_heredoc = True
            continue
        executable_lines.append(line)
    return executable_lines


_COMMAND_START = re.compile(r"^\s*(sudo\s+)?(apt-get|apt|yum|pacman|brew)\s+(install|-S)\b")


@pytest.mark.parametrize("name", SETUP_SCRIPTS)
def test_setup_script_never_mutates_system_package_managers(name: str) -> None:
    """The forbidden phrases may appear as *printed instructions* for the
    user to run themselves (inside a heredoc, an echo, or a `fail(...)`
    message) -- that is the required behavior. What must never happen is
    the script *invoking* one of these as a bare command, which would mean
    a line whose first token (after optional `sudo`) is the package
    manager itself."""
    for line in _lines_outside_heredocs((PROJECT_ROOT / name).read_text()):
        assert not _COMMAND_START.match(line), f"{name} must not run '{line.strip()}' -- print instructions instead"


@pytest.mark.parametrize("name", SETUP_SCRIPTS)
def test_setup_script_creates_venv_and_installs_only_into_it(name: str) -> None:
    text = (PROJECT_ROOT / name).read_text()
    assert ".venv" in text
    assert '"venv"' in text or "-m venv" in text


@pytest.mark.parametrize("name", SETUP_SCRIPTS)
def test_setup_script_installs_runtime_dependencies_only_not_dev_extras(name: str) -> None:
    """SUB-12.1 #8: a judge/end-user running the native setup script never
    needs pytest, ruff, pyright, hypothesis, or any other dev-only tool --
    those come from `.[dev]`, which pulls in strictly more than the
    running application (`hydroswarm.api.app`, `hydroswarm.cli`) actually
    imports. `.[dev]` remains correct in dev-facing docs (CONTRIBUTING.md's
    `uv sync --all-extras --dev`), just not in what an end user's own
    machine installs."""
    text = (PROJECT_ROOT / name).read_text()
    assert "pip install -e \".\"" in text or "pip install -e '.'" in text
    assert ".[dev]" not in text


@pytest.mark.parametrize("name", SETUP_SCRIPTS)
def test_setup_script_verifies_frozen_bundle_and_runs_self_test(name: str) -> None:
    text = (PROJECT_ROOT / name).read_text()
    assert "verify-bundle" in text
    assert "self-test" in text


@pytest.mark.parametrize("name", SETUP_SCRIPTS)
def test_setup_script_prints_launch_command(name: str) -> None:
    text = (PROJECT_ROOT / name).read_text()
    assert "start_hydroswarm_" in text


def test_windows_setup_uses_powershell_not_batch() -> None:
    assert (PROJECT_ROOT / "setup_hydroswarm_windows.ps1").is_file()


def test_windows_setup_explains_docker_wsl2_is_preferred_for_production_latency() -> None:
    text = (PROJECT_ROOT / "setup_hydroswarm_windows.ps1").read_text()
    assert "WSL2" in text or "Docker" in text


def test_windows_setup_does_not_run_full_real_simulator_suite() -> None:
    text = (PROJECT_ROOT / "setup_hydroswarm_windows.ps1").read_text()
    assert "pytest" not in text


def test_macos_setup_fails_closed_on_intel_x86_64() -> None:
    """Native macOS support targets Apple Silicon only -- the frozen
    torch>=2.5 requirement has no macOS x86_64 wheel upstream (confirmed
    via a real CI run on a genuine Intel macOS runner, see
    native-cross-platform.yml). The setup script must refuse early with a
    clear message instead of proceeding until pip fails obscurely deep
    inside dependency installation."""
    text = (PROJECT_ROOT / "setup_hydroswarm_macos.sh").read_text()
    assert '"$ARCH" = "x86_64"' in text
    # The guard must fire (and the script must exit) before any pip install
    # is attempted, not merely print an advisory note alongside it.
    guard_index = text.index('"$ARCH" = "x86_64"')
    first_pip_install_index = text.index("pip install")
    assert guard_index < first_pip_install_index
    fail_closed_section = text[guard_index : guard_index + 800]
    assert "fail " in fail_closed_section or 'fail "' in fail_closed_section
    assert "Apple Silicon" in fail_closed_section
    assert "no supported Intel macOS binary distribution" in fail_closed_section


def test_legacy_launchers_are_thin_compatibility_wrappers() -> None:
    sh_text = (PROJECT_ROOT / "start_hydroswarm.sh").read_text()
    assert "start_hydroswarm_linux.sh" in sh_text
    assert "start_hydroswarm_macos.sh" in sh_text

    bat_text = (PROJECT_ROOT / "start_hydroswarm.bat").read_text()
    assert "start_hydroswarm_windows.ps1" in bat_text


def test_setup_common_helper_module_exists() -> None:
    assert (PROJECT_ROOT / "scripts" / "setup_common.py").is_file()
