"""Measure deterministic CPU HydroCore inference, memory, disk size, and scale behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
from time import perf_counter

import psutil
from safetensors.torch import save_file
import torch

from hydroswarm.model import HydroCore


def _batch(nodes: int) -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.zeros(1, nodes, 19),
        "temporal_features": torch.zeros(1, 1, nodes, 6),
        "quality_features": torch.ones(1, 1, nodes, 4),
        "node_mask": torch.ones(1, nodes, dtype=torch.bool),
    }


def _measure(model: HydroCore, nodes: int, iterations: int) -> dict[str, float | int | str]:
    batch = _batch(nodes)
    with torch.inference_mode():
        for _ in range(2):
            model(batch)
        first = model(batch)["source_node_logits"].detach().clone()
        timings: list[float] = []
        rss_before = psutil.Process(os.getpid()).memory_info().rss
        for _ in range(iterations):
            started = perf_counter()
            model(batch)["source_node_logits"]
            timings.append((perf_counter() - started) * 1000)
        rss_after = psutil.Process(os.getpid()).memory_info().rss
        repeated = model(batch)["source_node_logits"]
    absolute_error = float(torch.max(torch.abs(first - repeated)))
    return {
        "nodes": nodes,
        "iterations": iterations,
        "median_latency_ms": statistics.median(timings),
        "p95_latency_ms": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
        "minimum_latency_ms": min(timings),
        "process_rss_before_mb": rss_before / (1024 * 1024),
        "process_rss_after_mb": rss_after / (1024 * 1024),
        "rss_delta_mb": (rss_after - rss_before) / (1024 * 1024),
        "repeat_max_absolute_error": absolute_error,
        "output_sha256": hashlib.sha256(repeated.numpy().tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--stress-nodes", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/results/performance.json"))
    args = parser.parse_args()
    if args.nodes < 2 or args.stress_nodes < args.nodes or args.iterations < 2:
        parser.error("require nodes >= 2, stress-nodes >= nodes, and iterations >= 2")

    torch.manual_seed(2026)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = HydroCore.from_variant(args.variant).eval()
    with tempfile.TemporaryDirectory(prefix="hydroswarm-performance-") as directory:
        artifact = Path(directory) / f"hydrocore-{args.variant}.safetensors"
        state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        save_file(state, artifact)
        artifact_bytes = artifact.stat().st_size
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()

    report = {
        "schema_version": "hydroswarm-performance-v1",
        "measured": True,
        "precision": "FP32",
        "device": "CPU",
        "variant": args.variant,
        "parameters": model.parameter_count(),
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "model_artifact_bytes": artifact_bytes,
        "model_artifact_sha256": artifact_sha256,
        "reference": _measure(model, args.nodes, args.iterations),
        "large_graph_stress": _measure(model, args.stress_nodes, max(2, args.iterations // 2)),
        "optimization_status": {
            "eager_fp32": "measured",
            "onnx": "not_run_optional_dependency",
            "openvino_fp32": "not_run_optional_dependency",
            "openvino_int8": "not_run_requires_governed_quantization_calibration",
        },
        "limitations": [
            "Latency is measured on the current host and is not portable across CPUs.",
            "Process RSS includes the Python runtime and native libraries already loaded.",
            "The stress case measures canonical model scaling, not a 1000-node WNTR solve.",
            "Random seeded weights establish runtime and numerical repeatability, not accuracy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
