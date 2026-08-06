"""Merge a trajectory corpus's (scripts/generate_trajectory_corpus.py)
Sentinel-level targets (OOD class, control, auxiliary) into an existing
tensor-shard corpus, producing an immediately trainable enriched shard set
(core-issues2.txt Phase 7/8 bridge).

Why this exists: build_incident_trajectory's Scout/Strategist output is
inherently sequential (multiple sample steps, multiple plan candidates)
and does not fit ScenarioExample's flat per-example .targets dict --
that's why generate_trajectory_corpus.py writes it to a separate JSONL
file rather than tensor shards. But the *other* new targets Phase 4-6
added (ood_class, next_step, evidence_sufficiency, sensor_reconstruction,
future_concentration, travel_time) ARE flat per-example values, already
stored inside that same JSONL's "targets" field (build_incident_trajectory
merges them into example.targets before generate_trajectory_corpus.py
serializes it). This script re-attaches exactly those flat targets onto
the original ScenarioExample's tensors (inputs untouched) and re-writes
sharded storage via write_shards -- no re-simulation, no re-training-
data-generation, just a join by scenario_id.

Never mutates the source shard directory; always writes to --output.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from hydroswarm.training.data import ScenarioExample
from hydroswarm.training.sharded_data import ShardedScenarioDataset, write_shards


def _load_trajectory_targets(jsonl_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    targets_by_scenario: dict[str, dict[str, torch.Tensor]] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tensors: dict[str, torch.Tensor] = {}
        for key, value in record["targets"].items():
            if key.endswith("_mask"):
                tensors[key] = torch.tensor(value, dtype=torch.bool)
            elif isinstance(value, bool):
                tensors[key] = torch.tensor(value)
            elif isinstance(value, int):
                tensors[key] = torch.tensor(value)
            elif isinstance(value, list):
                tensors[key] = torch.tensor(value, dtype=torch.float32)
            else:
                tensors[key] = torch.tensor(value, dtype=torch.float32)
        targets_by_scenario[record["scenario_id"]] = tensors
    return targets_by_scenario


def merge(
    tensor_shard_dir: Path, trajectory_jsonl: Path, output_dir: Path, *, expected_split: str
) -> dict[str, Any]:
    dataset = ShardedScenarioDataset(tensor_shard_dir, expected_split=expected_split)
    new_targets = _load_trajectory_targets(trajectory_jsonl)

    merged_examples: list[ScenarioExample] = []
    matched = 0
    for position in range(len(dataset)):
        example = dataset[position]
        extra = new_targets.get(example.scenario_id)
        if extra is None:
            merged_examples.append(example)
            continue
        matched += 1
        merged_examples.append(dataclasses.replace(example, targets={**example.targets, **extra}))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_shards(merged_examples, output_dir)
    return {
        "examples_total": len(merged_examples),
        "examples_enriched": matched,
        "examples_unchanged": len(merged_examples) - matched,
        "manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tensor-shard-dir", type=Path, required=True)
    parser.add_argument("--trajectory-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = merge(args.tensor_shard_dir, args.trajectory_jsonl, args.output, expected_split=args.split)
    print(
        f"merged {result['examples_enriched']}/{result['examples_total']} examples "
        f"({result['examples_unchanged']} unchanged, no trajectory record) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
