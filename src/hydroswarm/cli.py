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


def run_self_test() -> dict[str, Any]:
    """Run bounded startup checks with fixed model and WNTR reference execution."""

    dependency_names = ("fastapi", "networkx", "numpy", "pydantic", "torch", "wntr")
    dependencies: dict[str, str] = {}
    for name in dependency_names:
        importlib.import_module(name)
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "importable"

    from hydroswarm.model import HydroCore
    from hydroswarm.runtime import DefaultPipelineFactory
    from hydroswarm.simulation.network import build_networkx_network, build_wntr_network
    from hydroswarm.simulation.wrapper import HydraulicSimulator

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

    simulator = HydraulicSimulator(hydraulic_model, timeout_seconds=20.0)
    state = simulator.calculate_state(3600)
    if not state.pressure_m or not all(map(lambda value: value == value, state.pressure_m.values())):
        raise RuntimeError("fixed WNTR reference simulation is invalid")
    simulation_hash = simulator.state_hash()

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
    trained_factory = DefaultPipelineFactory(workspace)
    trained_assets_ready = trained_factory.trained_assets_ready

    return {
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
        },
        "inference_run": True,
        "inference_sha256": inference_hash,
        "simulation_run": True,
        "simulation_sha256": simulation_hash,
        "network_sha256": simulator.state_hash(),
        "resources": {
            "free_disk_gb": round(free_disk_gb, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "port_8765_available": port_available,
            "warnings": resource_warnings,
        },
        "frontend_assets": "built" if frontend_dist.exists() else "source-only",
        "offline_ready": True,
    }


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
def self_test_command() -> None:
    """Run machine-readable offline readiness checks."""
    try:
        typer.echo(json.dumps(run_self_test(), indent=2, sort_keys=True))
    except Exception as exc:
        typer.echo(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), err=True)
        raise typer.Exit(1) from exc


def main(argv: Sequence[str] | None = None) -> int:
    try:
        app(args=list(argv) if argv is not None else None, standalone_mode=False)
        return 0
    except typer.Exit as exc:
        return int(exc.exit_code)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
