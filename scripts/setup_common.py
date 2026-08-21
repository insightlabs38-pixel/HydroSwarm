"""Shared preflight/setup helpers for the per-platform setup scripts.

`setup_hydroswarm_linux.sh`, `setup_hydroswarm_macos.sh`, and
`setup_hydroswarm_windows.ps1` are the public entry points a judge or
operator actually runs. They differ only in shell syntax and OS-specific
messaging (apt/brew hints, Rosetta warnings, WSL2 guidance). Every check
that has real logic behind it -- Python version gating, venv creation,
frozen-bundle verification, frontend build detection, readiness gating --
lives here once, so the three scripts cannot silently drift out of sync
with each other the way the pre-SUB-1 bundle-path resolution did.

Each platform script calls this module as `python3 scripts/setup_common.py
<subcommand> [args...]` using whatever ambient/system Python it locates
*before* a venv exists (bootstrap steps), and via the freshly created
`.venv` interpreter for steps that need the project installed
(bundle/frontend/self-test verification).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

MIN_PYTHON = (3, 12)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return PROJECT_ROOT


def cmd_check_python(_args: argparse.Namespace) -> int:
    """Verify the interpreter invoking this command is 64-bit CPython >= 3.12."""
    version_ok = sys.version_info[:2] >= MIN_PYTHON
    is_64bit = sys.maxsize > 2**32
    report = {
        "ok": version_ok and is_64bit,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "is_64bit": is_64bit,
        "machine": platform.machine(),
    }
    print(json.dumps(report))
    if not report["ok"]:
        if not version_ok:
            print(
                f"error: Python {'.'.join(map(str, MIN_PYTHON))}+ required, "
                f"found {platform.python_version()} at {sys.executable}",
                file=sys.stderr,
            )
        if not is_64bit:
            print(f"error: a 64-bit Python interpreter is required, found {sys.executable}", file=sys.stderr)
        return 1
    return 0


def cmd_verify_bundle(_args: argparse.Namespace) -> int:
    """Verify the frozen V5 release bundle exists and hashes match, using the
    same resolver/factory the running application actually serves from
    (`hydroswarm.api.app`, `hydroswarm.cli.run_self_test`) so setup-time
    verification cannot drift from what the application actually checks at
    startup. This never falls back to the historical V4 bundle: it succeeds
    only when the V5 bundle itself is present and hash-verified, regardless
    of whether the V4 bundle happens to also exist in the checkout."""
    sys.path.insert(0, str(_project_root() / "src"))
    from hydroswarm.runtime import V5PipelineFactory, resolve_v5_bundle_dir

    bundle_dir = resolve_v5_bundle_dir(_project_root())
    factory = V5PipelineFactory(bundle_dir)
    report = {
        "ok": factory.trained_assets_ready,
        "bundle_dir": str(bundle_dir),
        "fallback_reason": factory.fallback_reason,
        "model_sha256": factory.model_hash,
    }
    print(json.dumps(report))
    if not report["ok"]:
        print(f"error: frozen HydroCore-v5 bundle failed verification: {factory.fallback_reason}", file=sys.stderr)
        return 1
    return 0


def cmd_frontend_status(_args: argparse.Namespace) -> int:
    """Report whether a built frontend already exists, so setup scripts can
    skip the Node/npm build step when a prebuilt `frontend/dist` is present."""
    index_html = _project_root() / "frontend" / "dist" / "index.html"
    report = {"built": index_html.is_file()}
    print(json.dumps(report))
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    """Run the real application self-test using the current interpreter
    (expected to be the freshly created `.venv` interpreter) and print its
    human-readable readiness summary. `--strict` (SUB-12.1 #21) also
    requires a FITTED calibration, the reference-demo artifact, and a
    built frontend -- not just that the interpreter/model/WNTR run at all."""
    sys.path.insert(0, str(_project_root() / "src"))
    from hydroswarm.cli import render_self_test_report, run_self_test

    try:
        report = run_self_test(strict=args.strict)
    except Exception as exc:  # noqa: BLE001 -- setup must report, not crash, on any failure
        print(f"HydroSwarm readiness\n\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(render_self_test_report(report))
    return 0 if report.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-python", help="Verify interpreter is 64-bit Python >= 3.12.")
    subparsers.add_parser("verify-bundle", help="Verify the frozen V5 release bundle.")
    subparsers.add_parser("frontend-status", help="Report whether frontend/dist is already built.")
    self_test_parser = subparsers.add_parser(
        "self-test", help="Run and render the application readiness self-test."
    )
    self_test_parser.add_argument(
        "--strict",
        action="store_true",
        help="Also require a FITTED calibration, the reference-demo artifact, and a built frontend.",
    )

    args = parser.parse_args(argv)
    handlers = {
        "check-python": cmd_check_python,
        "verify-bundle": cmd_verify_bundle,
        "frontend-status": cmd_frontend_status,
        "self-test": cmd_self_test,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
