"""Optionally export HydroCore through ONNX to OpenVINO and benchmark FP32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from hydroswarm.model import HydroCore


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: HydroCore) -> None:
        super().__init__()
        self.model = model

    def forward(self, node_features: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        return self.model({"node_features": node_features, "node_mask": node_mask})["source_node_logits"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--output", type=Path, default=Path("models/openvino"))
    parser.add_argument("--nodes", type=int, default=32)
    args = parser.parse_args()
    try:
        import openvino as ov
    except ImportError as exc:
        parser.error("OpenVINO is optional; install hydroswarm[openvino] before exporting")
        raise AssertionError from exc
    model = ExportWrapper(HydroCore.from_variant(args.variant).eval())
    node_features = torch.zeros(1, args.nodes, 19)
    node_mask = torch.ones(1, args.nodes, dtype=torch.bool)
    args.output.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(model, example_input=(node_features, node_mask))
    target = args.output / f"hydrocore-{args.variant}-fp32.xml"
    ov.save_model(ov_model, target, compress_to_fp16=False)
    compiled = ov.Core().compile_model(target, "CPU")
    for _ in range(5):
        compiled([node_features.numpy(), node_mask.numpy()])
    timings = []
    for _ in range(25):
        started = perf_counter()
        compiled([node_features.numpy(), node_mask.numpy()])
        timings.append((perf_counter() - started) * 1000)
    report = {
        "variant": args.variant,
        "precision": "FP32",
        "nodes": args.nodes,
        "median_latency_ms": float(np.median(timings)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "xml_bytes": target.stat().st_size,
        "bin_bytes": target.with_suffix(".bin").stat().st_size,
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
