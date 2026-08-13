"""Capability diagnostic Section 18: feature normalization/distribution
drift audit.

Compares key feature-channel distributions across TRAIN / VALIDATION /
CALIBRATION split tensors (`data/learning-v2/cycle-b2-joint-v4/
tensors-normalized/{split}`, loaded via `hydroswarm.training.
ShardedScenarioDataset` -- the exact tensors HydroCore actually trains and
calibrates against) and, separately, whatever coarse proxies are directly
available from the already-committed LIVE nominal population
(`reports/evaluation/live-robustness/post-remediation-results.json`,
filtered to `perturbation_type=="nominal"`, n=12).

Channels covered:
  - node_features columns: 4 (pressure_value), 6 (concentration, log1p),
    8 (health), 10 (missing) -- indices confirmed this session by reading
    src/hydroswarm/preprocessing/builder.py's node_rows.append(...) call
    (the canonical column order every corpus shard and this diagnostic's
    own scripts share).
  - temporal_features channels: 0 (concentration log1p), 1 (pressure/100),
    2 (age/86400), 3 (missing), 4 (drift), 5 (delayed).
  - quality_features channels: 0 (health), 1 (missing), 2 (drift),
    3 (age/86400).

IMPORTANT HONESTY NOTE (per task instructions): LIVE's raw JSON records
carry only summary scalars (`observation_count`, `sensor_count`, ...), NOT
raw per-node-per-timestep feature tensors -- a full apples-to-apples
raw-channel comparison against LIVE requires re-running LIVE-shaped
evidence through the real feature builder, which
`scripts/capability_diagnostic/temporal_evidence_ablation.py` (Section 8/9,
`temporal-ablation.json`) and `scripts/capability_diagnostic/
train_serve_parity_full.py` (Section 6, `train-serve-parity.json`) already
did directly (their `evidence_contract`/`temporal_ablation_latest_k`
sections ARE the real tensor-level LIVE-vs-training comparison). This
script does NOT re-derive that; it reports what IS honestly, directly
computable from the raw LIVE JSON alone (observation_count/sensor_count as
a coarse "how many of this incident's instrumented sensors ever reported"
proxy, compared against the train-split's per-example fraction-of-cells-
missing in the temporal missing channel) and clearly labels it as a coarse,
non-apples-to-apples proxy, citing the other two scripts for the real
tensor-level answer.

Sampling: fixed seed 20260813, up to 300 examples per split (train has
9000; validation/calibration have 1000 each) for speed, per protocol's
"sample a few hundred for speed" instruction. Shard checksum verification
is skipped for speed (already verified by other scripts/gates this
session/repo); this script only reads, never writes, corpus shards.

No locked-test access: train/validation/calibration are all non-locked
governed splits; LIVE data is reused, already-committed, non-locked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.training import ShardedScenarioDataset  # noqa: E402

JOINT_CORPUS_ROOT = ROOT / "data" / "learning-v2" / "cycle-b2-joint-v4" / "tensors-normalized"
LIVE_RESULTS_PATH = ROOT / "reports" / "evaluation" / "live-robustness" / "post-remediation-results.json"
SEED = 20260813
MAX_SAMPLES_PER_SPLIT = 300

NODE_FEATURE_CHANNELS = {
    "pressure_value": 4,
    "concentration_log1p": 6,
    "health": 8,
    "missing": 10,
}
TEMPORAL_FEATURE_CHANNELS = {
    "concentration_log1p": 0,
    "pressure_scaled": 1,
    "age_scaled": 2,
    "missing": 3,
    "drift": 4,
    "delayed": 5,
}
QUALITY_FEATURE_CHANNELS = {
    "health": 0,
    "missing": 1,
    "drift": 2,
    "age_scaled": 3,
}

PERCENTILES = (5, 25, 50, 75, 95)


def _describe(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    n_total = int(values.size)
    n_nan = int(np.isnan(values).sum())
    if finite.size == 0:
        return {"n_total": n_total, "n_nan": n_nan, "n_finite": 0}
    percentiles = {f"p{p}": float(np.percentile(finite, p)) for p in PERCENTILES}
    return {
        "n_total": n_total,
        "n_nan": n_nan,
        "n_finite": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        **percentiles,
    }


def _load_split_samples(split: str, rng: np.random.Generator) -> dict[str, Any]:
    shard_dir = JOINT_CORPUS_ROOT / split
    dataset = ShardedScenarioDataset(shard_dir, expected_split=split)
    total = len(dataset)
    n_sample = min(MAX_SAMPLES_PER_SPLIT, total)
    indices = sorted(rng.choice(total, size=n_sample, replace=False).tolist())

    node_channel_values: dict[str, list[float]] = {name: [] for name in NODE_FEATURE_CHANNELS}
    temporal_channel_values: dict[str, list[float]] = {name: [] for name in TEMPORAL_FEATURE_CHANNELS}
    quality_channel_values: dict[str, list[float]] = {name: [] for name in QUALITY_FEATURE_CHANNELS}
    per_example_missing_fraction: list[float] = []

    for index in indices:
        example = dataset[index]
        node_features = example.inputs["node_features"].numpy()
        temporal_features = example.inputs["temporal_features"].numpy()
        quality_features = example.inputs["quality_features"].numpy()

        for name, col in NODE_FEATURE_CHANNELS.items():
            node_channel_values[name].extend(node_features[:, col].tolist())
        for name, col in TEMPORAL_FEATURE_CHANNELS.items():
            temporal_channel_values[name].extend(temporal_features[:, :, col].reshape(-1).tolist())
        for name, col in QUALITY_FEATURE_CHANNELS.items():
            quality_channel_values[name].extend(quality_features[:, :, col].reshape(-1).tolist())

        # Per-example fraction of (time, node) cells marked missing in the
        # temporal channel -- used as the TRAIN-side half of the coarse
        # LIVE-comparison proxy below.
        missing_cells = temporal_features[:, :, TEMPORAL_FEATURE_CHANNELS["missing"]]
        per_example_missing_fraction.append(float(np.mean(missing_cells)))

    return {
        "split": split,
        "total_examples_in_split": total,
        "n_sampled": n_sample,
        "node_features": {name: _describe(np.asarray(values)) for name, values in node_channel_values.items()},
        "temporal_features": {name: _describe(np.asarray(values)) for name, values in temporal_channel_values.items()},
        "quality_features": {name: _describe(np.asarray(values)) for name, values in quality_channel_values.items()},
        "per_example_temporal_missing_fraction": _describe(np.asarray(per_example_missing_fraction)),
    }


def _load_live_nominal() -> dict[str, Any]:
    records = json.loads(LIVE_RESULTS_PATH.read_text(encoding="utf-8"))
    nominal = [r for r in records if r.get("perturbation_type") == "nominal"]
    observation_counts = np.asarray([r["observation_count"] for r in nominal if r.get("observation_count") is not None], dtype=float)
    sensor_counts = np.asarray([r["sensor_count"] for r in nominal if r.get("sensor_count") is not None], dtype=float)
    node_counts = np.asarray([r["node_count"] for r in nominal if r.get("node_count") is not None], dtype=float)
    observed_fraction = np.asarray(
        [r["observation_count"] / r["sensor_count"] for r in nominal if r.get("sensor_count")],
        dtype=float,
    )
    implied_missing_fraction_coarse = 1.0 - observed_fraction
    return {
        "n_records": len(nominal),
        "perturbation_levels_present": sorted({r.get("perturbation_level") for r in nominal}),
        "observation_count": _describe(observation_counts),
        "sensor_count": _describe(sensor_counts),
        "node_count": _describe(node_counts),
        "observed_fraction_of_sensors_reporting": _describe(observed_fraction),
        "implied_missing_fraction_coarse_proxy": _describe(implied_missing_fraction_coarse),
        "raw_values": {
            "observation_count": observation_counts.tolist(),
            "sensor_count": sensor_counts.tolist(),
            "observed_fraction": observed_fraction.tolist(),
        },
    }


def _pct_outside_range(values: list[float], low: float, high: float) -> float:
    if not values:
        return 0.0
    outside = sum(1 for v in values if v < low or v > high)
    return outside / len(values)


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed for this diagnostic"

    split_seed_offsets = {"train": 0, "validation": 1, "calibration": 2}
    splits = {}
    for split in ("train", "validation", "calibration"):
        splits[split] = _load_split_samples(split, np.random.default_rng(SEED + split_seed_offsets[split]))

    live_nominal = _load_live_nominal()

    # Coarse LIVE-vs-train comparison: LIVE's implied_missing_fraction_coarse_proxy
    # (1 - observation_count/sensor_count, a SINGLE-SNAPSHOT per-sensor
    # presence/absence measure) vs train's per_example_temporal_missing_fraction
    # (a per-(time,node)-cell measure over the full 25-timestep trajectory).
    # NOT apples-to-apples (different denominators/granularity) -- reported
    # as a coarse proxy only, explicitly labeled, per this script's honesty note.
    train_missing_range = (
        splits["train"]["per_example_temporal_missing_fraction"].get("min"),
        splits["train"]["per_example_temporal_missing_fraction"].get("max"),
    )
    live_pct_outside_train_missing_range = None
    if all(v is not None for v in train_missing_range):
        live_implied_missing = [1.0 - v for v in live_nominal["raw_values"]["observed_fraction"]]
        live_pct_outside_train_missing_range = _pct_outside_range(
            live_implied_missing, train_missing_range[0], train_missing_range[1],
        )

    # Flag top-drifted channels: compare train vs validation and train vs
    # calibration mean/std per channel (these ARE fully apples-to-apples --
    # same tensor construction, same schema, different governed splits).
    drift_flags: list[dict[str, Any]] = []
    for group_name, channel_map in (
        ("node_features", NODE_FEATURE_CHANNELS),
        ("temporal_features", TEMPORAL_FEATURE_CHANNELS),
        ("quality_features", QUALITY_FEATURE_CHANNELS),
    ):
        for channel_name in channel_map:
            train_stats = splits["train"][group_name][channel_name]
            for other_split in ("validation", "calibration"):
                other_stats = splits[other_split][group_name][channel_name]
                if "mean" not in train_stats or "mean" not in other_stats:
                    continue
                mean_diff = abs(train_stats["mean"] - other_stats["mean"])
                std_pool = max(train_stats["std"], 1e-9)
                drift_flags.append({
                    "group": group_name,
                    "channel": channel_name,
                    "compared_split": other_split,
                    "train_mean": train_stats["mean"],
                    "other_mean": other_stats["mean"],
                    "abs_mean_diff": mean_diff,
                    "abs_mean_diff_in_train_std_units": mean_diff / std_pool,
                    "train_std": train_stats["std"],
                    "other_std": other_stats["std"],
                })
    drift_flags.sort(key=lambda item: item["abs_mean_diff_in_train_std_units"], reverse=True)
    top_drifted = drift_flags[:8]

    report = {
        "schema_version": 1,
        "section": "18_feature_normalization_distribution_drift_audit",
        "locked_test_opened_before": locked_before,
        "seed": SEED,
        "max_samples_per_split": MAX_SAMPLES_PER_SPLIT,
        "channel_definitions": {
            "node_features": NODE_FEATURE_CHANNELS,
            "temporal_features": TEMPORAL_FEATURE_CHANNELS,
            "quality_features": QUALITY_FEATURE_CHANNELS,
        },
        "splits": splits,
        "live_nominal": live_nominal,
        "coarse_live_vs_train_missing_fraction_comparison": {
            "caveat": "NOT apples-to-apples: LIVE proxy is a single-snapshot per-SENSOR presence/absence fraction "
            "(1 - observation_count/sensor_count); train stat is a per-(time,node)-CELL missing fraction over the "
            "full 25-timestep trajectory. Reported only because it is the one thing directly computable from the "
            "raw committed LIVE JSON without re-running LIVE evidence through the real feature builder -- for the "
            "REAL tensor-level apples-to-apples comparison, see reports/evaluation/capability-diagnostic/"
            "temporal-ablation.json (evidence_contract section: LIVE sends exactly 1 observation/sensor vs "
            "training's 25) and train-serve-parity.json (Section 6's full tensor-diff results).",
            "train_temporal_missing_fraction_range_observed": train_missing_range,
            "live_pct_outside_train_missing_fraction_range": live_pct_outside_train_missing_range,
        },
        "top_drifted_channels_train_vs_validation_or_calibration": top_drifted,
        "cross_reference": {
            "for_real_live_tensor_level_comparison": [
                "reports/evaluation/capability-diagnostic/temporal-ablation.json",
                "reports/evaluation/capability-diagnostic/train-serve-parity.json",
            ],
            "for_error_clustering_confidence_sections_37_38": "reports/evaluation/capability-diagnostic/feature-distribution.json (pre-existing, NOT overwritten by this script)",
        },
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "feature-distribution-drift.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"top_drifted_channels": top_drifted[:5]}, indent=2, default=str))
    print(json.dumps(report["coarse_live_vs_train_missing_fraction_comparison"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
