"""HydroSwarm local console and reproducibility commands."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def run_self_test() -> dict[str, Any]:
    """Run bounded startup checks without model inference or external services."""

    dependency_names = ("fastapi", "networkx", "numpy", "pydantic", "torch", "wntr")
    dependencies: dict[str, str] = {}
    for name in dependency_names:
        importlib.import_module(name)
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "importable"

    from hydroswarm.model import HydroCore
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

    # Parameter enumeration validates construction and the model-size invariant;
    # no forward pass, weights, network calls, or long inference are involved.
    torch = importlib.import_module("torch")
    # Meta tensors preserve every declared parameter shape while avoiding weight
    # allocation and initialization, keeping this check quick on operator laptops.
    with torch.device("meta"):
        model = HydroCore()
    parameter_count = model.parameter_count()
    del model
    if parameter_count <= 0:
        raise RuntimeError("model parameter count must be positive")

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
        "inference_run": False,
    }


def _start_console(host: str, port: int, *, runner: Any | None = None) -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve().parent / "console" / "app.py"),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"Starting offline HydroSwarm console at http://{host}:{port}")
    runner = runner or subprocess.run
    completed = runner(command, check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hydroswarm", description="HydroSwarm local operations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="start the offline Streamlit console")
    start.add_argument("--host", default=DEFAULT_HOST, help="bind address (default: 127.0.0.1)")
    start.add_argument("--port", default=DEFAULT_PORT, type=int, help="console port (default: 8765)")
    subparsers.add_parser("self-test", help="run fast local reproducibility checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "start":
        if not 1 <= args.port <= 65_535:
            print("error: port must be between 1 and 65535", file=sys.stderr)
            return 2
        return _start_console(args.host, args.port)
    try:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # A command boundary should report every failed check.
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
