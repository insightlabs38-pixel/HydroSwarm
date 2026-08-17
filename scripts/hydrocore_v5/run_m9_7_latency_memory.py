"""Milestone 9.7 Section 10: engineering-only latency/memory preflight for
frozen HydroCore-S vs. the newly frozen HydroCore-M architecture.

Measures forward latency, peak process RSS delta, and serialized checkpoint
size on synthetic/train-side-shaped CPU inputs (device="cpu", matching
configs/training-v5-causal.yaml). No development_holdout data, no accuracy
number, and no locked split is touched. Both S and M were already frozen
deterministically (parameter-count-only, Section 3) BEFORE this script runs
-- these measurements are never used to retune either architecture.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_7_latency_memory.py

Writes:
    reports/evaluation/hydrocore-v5/m9-7/m9-7-latency-memory.json
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT  # noqa: E402

OUT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-7" / "m9-7-latency-memory.json"

SHARED_MODEL_CONFIG = dict(
    prior_mode="feature_only",
    event_control_heads=True,
    scout_control_heads=True,
    strategist_mode="candidate_conditioned",
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True,
    ood_category_head=True,
)
AGE_FIX_ONLY_MODEL_KWARGS = dict(
    temporal_feature_dim=6, quality_feature_dim=4, elapsed_time_normalization="window_relative"
)

NODES = 25  # golden-reference-scale node count, matching M9.6's evaluation topology order of magnitude.
STEPS = 25  # FULL_HISTORY_DEPTH, matching the frozen training/evaluation depth grid's own maximum.
ITERATIONS = 20
WARMUP = 5


def _batch(nodes: int, steps: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2026)
    return {
        "node_features": torch.randn(1, nodes, 19, generator=generator),
        "temporal_features": torch.randn(1, steps, nodes, 6, generator=generator),
        "quality_features": torch.randn(1, steps, nodes, 4, generator=generator),
        "node_mask": torch.ones(1, nodes, dtype=torch.bool),
    }


def _measure(model: HydroCore, *, nodes: int, steps: int, iterations: int, warmup: int) -> dict[str, object]:
    batch = _batch(nodes, steps)
    process = psutil.Process(os.getpid())
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        rss_before = process.memory_info().rss
        timings: list[float] = []
        for _ in range(iterations):
            started = perf_counter()
            output = model(batch)["source_node_logits"]
            timings.append((perf_counter() - started) * 1000.0)
        rss_after = process.memory_info().rss
    return {
        "nodes": nodes,
        "steps": steps,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "median_latency_ms": statistics.median(timings),
        "mean_latency_ms": statistics.fmean(timings),
        "p95_latency_ms": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
        "minimum_latency_ms": min(timings),
        "maximum_latency_ms": max(timings),
        "process_rss_before_mb": rss_before / (1024 * 1024),
        "process_rss_after_mb": rss_after / (1024 * 1024),
        "rss_delta_mb": (rss_after - rss_before) / (1024 * 1024),
        "output_shape": list(output.shape),
    }


def _checkpoint_size(model: HydroCore, name: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"hydroswarm-m9-7-{name}-") as directory:
        artifact = Path(directory) / f"hydrocore-{name}.safetensors"
        state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        save_file(state, artifact)
        return {
            "checkpoint_bytes": artifact.stat().st_size,
            "checkpoint_mb": artifact.stat().st_size / (1024 * 1024),
            "checkpoint_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }


def main() -> int:
    torch.manual_seed(2026)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    s_model = HydroCore.from_variant(
        "small", use_adapters=False, **AGE_FIX_ONLY_MODEL_KWARGS, **SHARED_MODEL_CONFIG
    ).eval()
    m_model = HydroCore.from_variant(
        "small_v5_capacity_m", use_adapters=False, **AGE_FIX_ONLY_MODEL_KWARGS, **SHARED_MODEL_CONFIG
    ).eval()

    s_params = sum(p.numel() for p in s_model.parameters())
    m_params = sum(p.numel() for p in m_model.parameters())

    s_measurement = _measure(s_model, nodes=NODES, steps=STEPS, iterations=ITERATIONS, warmup=WARMUP)
    m_measurement = _measure(m_model, nodes=NODES, steps=STEPS, iterations=ITERATIONS, warmup=WARMUP)
    s_checkpoint = _checkpoint_size(s_model, "small")
    m_checkpoint = _checkpoint_size(m_model, "small_v5_capacity_m")

    report = {
        "schema_version": 1,
        "purpose": (
            "Milestone 9.7 Section 10: engineering-only latency/memory preflight, measured AFTER "
            "HydroCore-M's architecture was already frozen deterministically (Section 3) -- never "
            "used to tune M's width."
        ),
        "milestone": "M9.7",
        "environment": {
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "hydrocore_s": {"variant": "small", "parameter_count": s_params, **s_measurement, **s_checkpoint},
        "hydrocore_m": {"variant": "small_v5_capacity_m", "parameter_count": m_params, **m_measurement, **m_checkpoint},
        "relative": {
            "parameter_ratio_M_over_S": m_params / s_params,
            "median_latency_ratio_M_over_S": m_measurement["median_latency_ms"] / s_measurement["median_latency_ms"],
            "checkpoint_size_ratio_M_over_S": m_checkpoint["checkpoint_mb"] / s_checkpoint["checkpoint_mb"],
        },
        "no_predictive_evaluation_performed": True,
        "no_post_hoc_cost_threshold_imposed": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
