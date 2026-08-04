"""Fit training-only feature normalization (overnight-plan.txt Task 0.6).

Governed by construction, not just convention: --split only accepts "train"
(argparse choices), so this tool cannot be pointed at validation,
calibration, development, OOD, or locked-test data even by a typo. If a
manifest file's own records disagree with the split it was invoked for, that
is treated as a hard error rather than silently normalizing on the wrong
data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats
from hydroswarm.training import load_scenario_examples_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train",),
        default="train",
        help="always 'train' -- kept explicit so the intent is visible in run commands",
    )
    parser.add_argument("--node-output", type=Path, required=True)
    parser.add_argument("--edge-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    examples = load_scenario_examples_jsonl(args.train_manifest)
    wrong_split = sorted({example.split for example in examples} - {"train"})
    if wrong_split:
        raise SystemExit(
            f"refusing to fit normalization: manifest contains non-train splits {wrong_split}; "
            "normalization must never be fit on validation/calibration/development/OOD/test data"
        )

    node_arrays = [example.inputs["node_features"].numpy() for example in examples if "node_features" in example.inputs]
    edge_arrays = [example.inputs["edge_features"].numpy() for example in examples if "edge_features" in example.inputs]
    if not node_arrays or not edge_arrays:
        raise SystemExit("manifest examples must include node_features and edge_features to fit normalization")

    node_stats = NormalizationStats.fit(
        np.concatenate(node_arrays, axis=0), DEFAULT_FEATURE_SCHEMA.node_features
    )
    edge_stats = NormalizationStats.fit(
        np.concatenate(edge_arrays, axis=0), DEFAULT_FEATURE_SCHEMA.edge_features
    )
    node_hash = node_stats.save(args.node_output)
    edge_hash = edge_stats.save(args.edge_output)
    print(
        json.dumps(
            {
                "train_examples": len(examples),
                "node_normalization_sha256": node_hash,
                "edge_normalization_sha256": edge_hash,
                "node_output": str(args.node_output),
                "edge_output": str(args.edge_output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
