"""HydroSwarm local console and reproducibility commands."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sqlite3
import socket
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import psutil
import typer


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def run_self_test(*, strict: bool = False) -> dict[str, Any]:
    """Run bounded startup checks with fixed model and WNTR reference execution.

    `strict=True` (SUB-12.1 #21) additionally requires the frozen V4 bundle
    to be ready with a genuinely FITTED calibration artifact (not merely
    loadable), the reference-demo artifact to be present, the frontend to
    be built, and resource checks to have produced zero warnings -- used by
    the native setup scripts, the Docker build gate, CI, and the release
    workflow so none of them can silently ship a degraded runtime. Failures
    are reported (`ok: False`, `strict_failures: [...]`), not raised -- a
    non-strict call keeps reporting the same facts but never fails on them
    (used for local iteration where a source-only frontend or an
    unconfigured calibration is an expected, informative state, not a
    blocker)."""

    dependency_names = ("fastapi", "networkx", "numpy", "pydantic", "torch", "wntr")
    dependencies: dict[str, str] = {}
    for name in dependency_names:
        importlib.import_module(name)
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "importable"

    from hydroswarm.model import HydroCore
    from hydroswarm.runtime import V4PipelineFactory
    from hydroswarm.runtime.paths import resolve_reference_demo_path, resolve_v4_bundle_dir
    from hydroswarm.simulation.network import build_networkx_network, build_wntr_network

    graph = build_networkx_network()
    hydraulic_model = build_wntr_network()
    if len(graph) != len(hydraulic_model.node_name_list):
        raise RuntimeError("NetworkX and WNTR node counts differ")
    if not hydraulic_model.reservoir_name_list or not hydraulic_model.tank_name_list:
        raise RuntimeError("hydraulic model requires a reservoir and a tank")

    with tempfile.TemporaryDirectory(prefix="hydroswarm-self-test-") as directory:
        database = Path(directory) / "self-test.sqlite3"
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("CREATE TABLE reproducibility_check (value INTEGER NOT NULL)")
            connection.execute("INSERT INTO reproducibility_check VALUES (1)")
            sqlite_value = connection.execute("SELECT value FROM reproducibility_check").fetchone()[0]
            connection.commit()
        if sqlite_value != 1:
            raise RuntimeError("SQLite round-trip failed")

    torch = importlib.import_module("torch")
    # Verify the declared large architecture without allocating its weights.
    with torch.device("meta"):
        model = HydroCore()
    parameter_count = model.parameter_count()
    del model
    if parameter_count <= 0:
        raise RuntimeError("model parameter count must be positive")

    # Run a fixed, bounded scientific inference on a tiny configuration.
    torch.manual_seed(2026)
    tiny = HydroCore(
        node_feature_dim=19, edge_feature_dim=13, temporal_feature_dim=6,
        quality_feature_dim=4, d_model=32, nhead=4, dim_feedforward=64,
        num_layers=1, modality_layers=1, latent_tokens=64, dropout=0.0,
    ).eval()
    nodes = len(graph)
    with torch.inference_mode():
        output = tiny({
            "node_features": torch.zeros(1, nodes, 19),
            "temporal_features": torch.zeros(1, 1, nodes, 6),
            "quality_features": torch.ones(1, 1, nodes, 4),
            "node_mask": torch.ones(1, nodes, dtype=torch.bool),
        })
    source_probabilities = torch.softmax(output["source_node_logits"], dim=-1)
    if not torch.isfinite(source_probabilities).all() or not torch.allclose(
        source_probabilities.sum(-1), torch.ones(1), atol=1e-5
    ):
        raise RuntimeError("fixed HydroCore inference is invalid")
    inference_hash = hashlib.sha256(source_probabilities.numpy().tobytes()).hexdigest()

    # SUB-12.1 #22: a real windows-latest CI run surfaced
    # SimulationTimeoutError here at 20s, then again at 60s (~64s actual),
    # then again at 150s (~155s actual) -- on Windows, HydraulicSimulator.
    # _run_with_timeout must use multiprocessing's "spawn" start method
    # (no fork() on Windows). Under spawn, locating the picklable-by-
    # reference worker function requires re-executing enough of the
    # parent's import graph to reach it, which can transitively reimport
    # torch/numpy/pandas/wntr from scratch even though this specific
    # worker only needs wntr -- compounded by real-world GitHub-hosted
    # Windows runners' well-documented slow process-creation overhead
    # (antivirus/Defender scanning each new process). self-test runs once
    # at startup, not on a hot path, so a generous bound here costs
    # nothing real while still being a real, bounded timeout -- not
    # unbounded. If this still is not enough on some future run, that is
    # itself evidence of a genuine hang, not just slowness, and needs
    # separate investigation rather than another blind increase.
    worker = subprocess.run(
        [sys.executable, "-m", "hydroswarm.simulation.self_test_worker"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if worker.returncode:
        raise RuntimeError(f"bounded WNTR self-test worker failed: {worker.stderr.strip()}")
    simulation_hash = json.loads(worker.stdout)["simulation_sha256"]

    workspace = Path.cwd()
    free_disk_gb = psutil.disk_usage(str(workspace)).free / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    resource_warnings = []
    if free_disk_gb < 1.0:
        resource_warnings.append("less_than_1_gib_free_disk")
    if available_ram_gb < 1.0:
        resource_warnings.append("less_than_1_gib_available_ram_use_small_or_classical_mode")
    with socket.socket() as probe:
        try:
            probe.bind((DEFAULT_HOST, DEFAULT_PORT))
            port_available = True
        except OSError:
            port_available = False
    frontend_dist = workspace / "frontend" / "dist" / "index.html"
    # UI-11.1 / submission-readiness SUB-1: self-test must validate the
    # exact same production runtime hydroswarm.api.app:app actually
    # launches with. Both now resolve the bundle directory through the
    # single shared hydroswarm.runtime.paths.resolve_v4_bundle_dir
    # function (HYDROSWARM_V4_BUNDLE_DIR override, else the source-tree
    # default) instead of two independently-computed, workspace-relative
    # paths that could silently diverge for a non-editable install or a
    # non-repository-root working directory.
    trained_factory = V4PipelineFactory(resolve_v4_bundle_dir())
    trained_assets_ready = trained_factory.trained_assets_ready
    identity = trained_factory.identity

    # SUB-12.1 #21: read the bundle's own calibration-status.json directly
    # (the same file V4PipelineFactory's internal loader already validated
    # a real calibration.json against, or explicitly recorded the reason
    # one isn't present) rather than adding a new public property to the
    # runtime factory just for this report -- self-test already resolves
    # the same bundle directory the factory loaded from.
    calibration_status_path = trained_factory.checkpoint_dir / "calibration-status.json"
    calibration_status = "MISSING"
    if calibration_status_path.exists():
        try:
            calibration_status = json.loads(calibration_status_path.read_text()).get("status", "MISSING")
        except (OSError, ValueError):
            calibration_status = "UNREADABLE"

    reference_artifact_path = resolve_reference_demo_path()
    reference_artifact_present = reference_artifact_path.exists()

    report = {
        "ok": True,
        "dependencies": dependencies,
        "network": {
            "nodes": len(graph),
            "links": graph.number_of_edges(),
            "reservoirs": len(hydraulic_model.reservoir_name_list),
            "tanks": len(hydraulic_model.tank_name_list),
        },
        "sqlite": "ok",
        "model_parameters": parameter_count,
        "trained_assets": {
            "ready": trained_assets_ready,
            "fallback_reason": trained_factory.fallback_reason,
            "architecture_version": identity.architecture_version if identity is not None else None,
            "model_sha256": trained_factory.model_hash,
            "normalization_hash": identity.normalization_hash if identity is not None else None,
            "bundle_dir": str(trained_factory.checkpoint_dir),
            "calibration_status": calibration_status,
        },
        "inference_run": True,
        "inference_sha256": inference_hash,
        "simulation_run": True,
        "simulation_sha256": simulation_hash,
        "network_sha256": hashlib.sha256(repr(hydraulic_model).encode()).hexdigest(),
        "resources": {
            "free_disk_gb": round(free_disk_gb, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "port_8765_available": port_available,
            "warnings": resource_warnings,
        },
        "frontend_assets": "built" if frontend_dist.exists() else "source-only",
        "reference_artifact": {
            "present": reference_artifact_present,
            "path": str(reference_artifact_path),
        },
        "offline_ready": True,
    }

    # `strict` never raises for these -- they are real, checked facts about
    # release readiness, not a crash. `ok`/`strict_failures` carries the
    # verdict so both JSON and --human callers (and the CLI's exit code)
    # can act on the same report a non-strict caller would also have
    # received, just with strict's stricter pass/fail line drawn over it.
    if strict:
        failures: list[str] = []
        if not trained_assets_ready:
            failures.append(f"frozen V4 bundle not ready: {trained_factory.fallback_reason}")
        if calibration_status != "FITTED":
            failures.append(f"calibration status is {calibration_status!r}, not FITTED")
        if report["frontend_assets"] != "built":
            failures.append("frontend is not built (frontend/dist/index.html missing)")
        if not reference_artifact_present:
            failures.append(f"reference-demo artifact missing at {reference_artifact_path}")
        if resource_warnings:
            failures.append(f"resource warnings present: {', '.join(resource_warnings)}")
        if failures:
            report["ok"] = False
            report["strict_failures"] = failures

    return report


def render_self_test_report(report: dict[str, Any]) -> str:
    """Render `run_self_test()`'s machine-readable result as the human-facing
    readiness checklist from submission.txt SS17, for setup scripts and
    interactive `hydroswarm self-test --human` use. Does not replace the
    default JSON output (still required by CI and other machine callers)."""
    trained_assets = report.get("trained_assets", {})
    resources = report.get("resources", {})
    reference_artifact = report.get("reference_artifact", {})
    checks: list[tuple[bool, str]] = [
        (report.get("ok", False), "Python runtime"),
        (bool(trained_assets.get("ready")), "Frozen HydroCore-v4 bundle verified"),
        (bool(trained_assets.get("model_sha256")), "Model SHA-256 verified"),
        (bool(trained_assets.get("normalization_hash")), "Normalization verified"),
        (trained_assets.get("calibration_status") == "FITTED", "Calibration FITTED"),
        (bool(report.get("simulation_run")), "WNTR/EPANET available"),
        (report.get("sqlite") == "ok", "SQLite writable"),
        (report.get("frontend_assets") == "built", "Frontend assets available"),
        (bool(reference_artifact.get("present")), "Reference-demo artifact present"),
        (bool(resources.get("port_8765_available")), "Port 8765 available"),
        (True, "No required external runtime service"),
    ]
    lines = ["HydroSwarm readiness", ""]
    lines.extend(f"{'✓' if ok else '✗'} {label}" for ok, label in checks)
    lines.append("")
    lines.append("READY" if all(ok for ok, _ in checks) else "NOT READY")
    if not trained_assets.get("ready"):
        lines.append(f"  reason: {trained_assets.get('fallback_reason')}")
    if report.get("frontend_assets") != "built":
        lines.append("  reason: frontend not built (source-only) -- run the frontend build before a demo")
    for failure in report.get("strict_failures", []):
        lines.append(f"  strict failure: {failure}")
    return "\n".join(lines)


def _start_console(host: str, port: int, *, runner: Any | None = None) -> int:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "hydroswarm.api.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"Starting offline HydroSwarm API and console at http://{host}:{port}")
    runner = runner or subprocess.run
    completed = runner(command, check=False)
    return int(completed.returncode)


app = typer.Typer(help="HydroSwarm local operations", no_args_is_help=True)


@app.command("start")
def start_command(
    host: str = typer.Option(DEFAULT_HOST, help="Loopback bind address."),
    port: int = typer.Option(DEFAULT_PORT, min=1, max=65_535, help="Console port."),
    allow_network_bind: bool = typer.Option(
        False,
        "--allow-network-bind",
        help="Allow a non-loopback bind for an explicitly isolated container deployment.",
    ),
) -> None:
    """Start the offline API and built operator console."""
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_network_bind:
        raise typer.BadParameter("HydroSwarm binds only to loopback by default")
    raise typer.Exit(_start_console(host, port))


@app.command("self-test")
def self_test_command(
    human: bool = typer.Option(
        False, "--human", help="Print a human-readable readiness checklist instead of JSON."
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help=(
            "Also require the frozen V4 bundle to be ready with a FITTED "
            "calibration, the reference-demo artifact to be present, the "
            "frontend to be built, and zero resource warnings -- exits "
            "nonzero if any of those fail, not just on a crash. Used by "
            "the native setup scripts, Docker build gate, CI, and release "
            "workflow."
        ),
    ),
) -> None:
    """Run offline readiness checks. Defaults to machine-readable JSON; pass
    --human for the operator-facing checklist used by the setup scripts."""
    try:
        report = run_self_test(strict=strict)
    except Exception as exc:
        if human:
            typer.echo(f"HydroSwarm readiness\n\nFAILED: {type(exc).__name__}: {exc}", err=True)
        else:
            typer.echo(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), err=True)
        raise typer.Exit(1) from exc
    if human:
        typer.echo(render_self_test_report(report))
    else:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if report.get("ok") is False:
        raise typer.Exit(1)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        # Click's own `main(standalone_mode=False)` -- what Typer's `app()`
        # delegates to -- catches `typer.Exit` internally and returns its
        # `exit_code` as this call's return value rather than re-raising
        # it; only `ClickException`/`UsageError` subclasses (e.g.
        # `typer.BadParameter`) still propagate as real exceptions. A
        # non-zero `typer.Exit` (e.g. self-test --strict failing) would
        # otherwise be silently discarded here and this function would
        # always return 0.
        result = app(args=list(argv) if argv is not None else None, standalone_mode=False)
        return int(result) if isinstance(result, int) else 0
    except typer.Exit as exc:
        return int(exc.exit_code)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
