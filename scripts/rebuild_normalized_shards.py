"""Rebuild a sharded corpus's tensor shards with governed normalization applied
(overnight-plan.txt Task 0.6 / core-issues.txt Phase 3 item 14).

Corpus generation (scripts/generate_cycle_b_corpus.py) writes raw, unnormalized
node_features/edge_features -- normalization statistics can only be fit once
the training split exists (scripts/fit_normalization.py), which makes fitting
and applying normalization a necessarily separate, later pass over the same
corpus. This script is that pass: for every split directory under
<corpus-dir>/tensors/ (train/validation/calibration/development_holdout, plus
any ood-<category> directories), it verifies shard checksums, applies the
supplied node/edge NormalizationStats to node_features/edge_features via the
exact same NormalizationStats.transform() call HydraulicFeatureBuilder.build()
applies at live-inference time (see hydroswarm/preprocessing/builder.py), and
writes the result to a sibling <corpus-dir>/tensors-normalized/ tree. Every
other tensor (masks, temporal/quality features, targets, topology metadata)
is untouched -- verified by asserting each split's manifest_hash (which
hashes only scenario/topology governance fields, never feature values) is
identical between the raw and normalized copies.

Deliberately NOT an in-place rewrite of <corpus-dir>/tensors/: the raw,
unnormalized shards must survive the rebuild so a later governance check
(the "normalization train-only ownership" corpus gate -- see
scripts/run_corpus_gates.py) can recompute normalization statistics directly
from tensors/train and confirm they match what was actually fit, rather than
merely trusting a log line. Training and evaluation scripts should be pointed
at tensors-normalized/, not tensors/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats
from hydroswarm.training import ShardedScenarioDataset
from hydroswarm.training.data import ScenarioExample
from hydroswarm.training.sharded_data import write_shards


def _discover_split_dirs(tensors_root: Path) -> dict[str, str]:
    """Map each split directory name under tensors_root to its expected
    ScenarioExample.split value. OOD holdout directories (ood-<category>)
    are generated with split=DEVELOPMENT_HOLDOUT (see
    generate_cycle_b_corpus.py's _generate_*_ood_holdout_* helpers) even
    though they live outside the plain development_holdout directory."""

    mapping: dict[str, str] = {}
    for candidate in sorted(tensors_root.iterdir()):
        if not candidate.is_dir() or not (candidate / "manifest.json").exists():
            continue
        if candidate.name.startswith("ood-"):
            mapping[candidate.name] = "development_holdout"
        else:
            mapping[candidate.name] = candidate.name
    return mapping


def _apply_normalization(
    example: ScenarioExample,
    *,
    node_stats: NormalizationStats,
    edge_stats: NormalizationStats,
) -> ScenarioExample:
    new_inputs = dict(example.inputs)
    if "node_features" in new_inputs:
        if node_stats.schema_version != DEFAULT_FEATURE_SCHEMA.version:
            raise ValueError("node normalization schema version is incompatible")
        new_inputs["node_features"] = torch.as_tensor(
            node_stats.transform(new_inputs["node_features"].numpy()), dtype=torch.float32
        )
    if "edge_features" in new_inputs:
        if edge_stats.schema_version != DEFAULT_FEATURE_SCHEMA.version:
            raise ValueError("edge normalization schema version is incompatible")
        new_inputs["edge_features"] = torch.as_tensor(
            edge_stats.transform(new_inputs["edge_features"].numpy()), dtype=torch.float32
        )
    return dataclasses.replace(example, inputs=new_inputs)


def rebuild_split(
    source_dir: Path,
    destination_dir: Path,
    *,
    expected_split: str,
    node_stats: NormalizationStats,
    edge_stats: NormalizationStats,
) -> dict[str, Any]:
    dataset = ShardedScenarioDataset(source_dir, expected_split=expected_split)
    dataset.verify_shard_checksums()
    before_hash = dataset.manifest_hash
    shard_size = int(dataset.manifest["shard_size"])
    example_count = len(dataset)

    if destination_dir.exists() and any(destination_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty destination: {destination_dir}")

    examples = [
        _apply_normalization(dataset[position], node_stats=node_stats, edge_stats=edge_stats)
        for position in range(example_count)
    ]
    manifest = write_shards(examples, destination_dir, shard_size=shard_size)

    rebuilt = ShardedScenarioDataset(destination_dir, expected_split=expected_split)
    rebuilt.verify_shard_checksums()
    after_hash = rebuilt.manifest_hash
    if after_hash != before_hash:
        raise ValueError(
            f"{destination_dir}: manifest_hash differs from raw source {source_dir} "
            f"({before_hash} -> {after_hash}); rebuild must only change feature "
            "values, never scenario/topology governance metadata"
        )
    if len(rebuilt) != example_count:
        raise ValueError(f"{destination_dir}: example count changed after rebuild ({example_count} -> {len(rebuilt)})")

    return {
        "example_count": example_count,
        "shard_count": len(manifest["shards"]),
        "manifest_hash": after_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True, help="e.g. data/learning-v2/cycle-b2")
    parser.add_argument("--node-normalization", type=Path, required=True)
    parser.add_argument("--edge-normalization", type=Path, required=True)
    parser.add_argument(
        "--output-tensors-dirname",
        default="tensors-normalized",
        help="written under --corpus-dir, sibling to the raw tensors/ tree (default: tensors-normalized)",
    )
    parser.add_argument("--report-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tensors_root = args.corpus_dir / "tensors"
    if not tensors_root.is_dir():
        raise SystemExit(f"no tensors directory under {args.corpus_dir}")
    output_root = args.corpus_dir / args.output_tensors_dirname
    if output_root == tensors_root:
        raise SystemExit("--output-tensors-dirname must not be 'tensors' -- the raw corpus must survive the rebuild")

    node_stats = NormalizationStats.load(args.node_normalization)
    edge_stats = NormalizationStats.load(args.edge_normalization)

    split_dirs = _discover_split_dirs(tensors_root)
    if "train" not in split_dirs:
        raise SystemExit(f"no train split found under {tensors_root}; refusing to rebuild an incomplete corpus")

    results: dict[str, Any] = {}
    for name, expected_split in split_dirs.items():
        results[name] = rebuild_split(
            tensors_root / name,
            output_root / name,
            expected_split=expected_split,
            node_stats=node_stats,
            edge_stats=edge_stats,
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_dir": str(args.corpus_dir),
        "raw_tensors_dir": str(tensors_root),
        "normalized_tensors_dir": str(output_root),
        "node_normalization_path": str(args.node_normalization),
        "node_normalization_sha256": node_stats.fingerprint,
        "edge_normalization_path": str(args.edge_normalization),
        "edge_normalization_sha256": edge_stats.fingerprint,
        "splits_rebuilt": results,
    }
    report_output = args.report_output or (args.corpus_dir / "normalization-rebuild-report.json")
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
