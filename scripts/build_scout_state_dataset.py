"""core-issues3.txt Phase 10 item 2 / core-issues4.txt Section I: build a
sharded, directly-trainable Scout-state dataset from a trajectory corpus
(scripts/generate_trajectory_corpus.py).

Scoping decision, deliberate and documented (not an oversight)
---------------------------------------------------------------
`hydroswarm.training.scout_trajectory.build_scout_trajectory` (core-issues3.txt
Phase 5) genuinely produces a MULTI-step trajectory when given a real
`reconstruction` (each step after the first reveals a new measurement and
updates the posterior). But `HydroCore` has NO input mechanism to condition
its Scout outputs on "which candidates are already sampled" -- no
`already_sampled_mask` (or equivalent) exists anywhere in `HydroBatch`
(confirmed by inspection of `model/core.py`). Training against every step
of a multi-step trajectory would therefore hand the model CONTRADICTORY
targets for the *same* input representation (only `already_sampled` differs
between steps, and the model cannot see it) -- systematic label noise, not
richer supervision.

This is the same class of gap `future_concentration_target`'s Phase 7.4
disable already established a precedent for: the DATA-GENERATION capability
(Phase 5's incremental revelation) has outpaced what the CURRENT
architecture can actually condition on, and the honest fix is to scope the
dataset to what the architecture can genuinely learn from today, not to
silently train on data the model cannot properly interpret. A real
closed-loop Scout architecture (conditioning on already_sampled/revealed
evidence) is a materially larger change than a dataset-builder script,
tracked as follow-up work, not attempted here.

This script therefore uses ONLY each trajectory's step 0 (`scout.steps[0]`)
-- the initial Scout decision, before any active sampling, where
`already_sampled` is always empty and the state is unambiguous given the
base scenario's own passive sensor evidence alone. This is still genuinely
useful supervision (a real "which candidate should we sample first" signal
derived from real information-gain computation), just not the full
multi-step trajectory.

No new collator is needed: Scout's governed targets (`sample_node`,
`information_gain`, `candidate_reduction`, `should_continue_sampling`, plus
`*_mask` companions) are already per-node arrays in the exact same shape
convention `sensor_fault` uses, so the existing, already-tested
`collate_variable_topology` handles them without modification -- this
script only needs to MERGE them onto the base `ScenarioExample` tensors
(exactly `scripts/merge_trajectory_targets.py`'s established pattern,
applied to a nested `scout.steps[0].targets` path instead of the top-level
flat `targets`).
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


def _load_scout_step0_targets(jsonl_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    targets_by_scenario: dict[str, dict[str, torch.Tensor]] = {}
    with jsonl_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            steps = record["scout"]["steps"]
            if not steps:
                # A scenario where no candidate exists at all (Phase 2 item
                # 4's "mask examples where no useful candidate exists") --
                # correctly has zero Scout steps; skip rather than fabricate.
                continue
            step0 = steps[0]
            already_sampled = step0["diagnostics"].get("already_sampled")
            if already_sampled:
                raise ValueError(
                    f"scenario {record['scenario_id']}: expected step 0 to have an empty "
                    f"already_sampled set, got {already_sampled!r} -- trajectory step ordering "
                    "assumption violated"
                )
            tensors: dict[str, torch.Tensor] = {}
            for key, value in step0["targets"].items():
                if key.endswith("_mask") or isinstance(value, bool):
                    tensors[key] = torch.tensor(value, dtype=torch.bool)
                elif isinstance(value, int):
                    tensors[key] = torch.tensor(value)
                elif isinstance(value, list):
                    tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    tensors[key] = torch.tensor(value, dtype=torch.float32)
            targets_by_scenario[record["scenario_id"]] = tensors
    return targets_by_scenario


def build(
    tensor_shard_dir: Path, trajectory_jsonl: Path, output_dir: Path, *, expected_split: str
) -> dict[str, Any]:
    dataset = ShardedScenarioDataset(tensor_shard_dir, expected_split=expected_split)
    dataset.verify_shard_checksums()
    scout_targets = _load_scout_step0_targets(trajectory_jsonl)

    merged_examples: list[ScenarioExample] = []
    matched = 0
    no_candidate_count = 0
    for position in range(len(dataset)):
        example = dataset[position]
        extra = scout_targets.get(example.scenario_id)
        if extra is None:
            # Phase 2 item 4: "mask examples where no useful candidate
            # exists" -- masked, never dropped, so this dataset stays in
            # 1:1 correspondence with the base corpus (consistent with
            # every other masked target in this codebase; a dropped row
            # would silently break split/network-balance accounting and
            # cross-corpus scenario_id joins downstream).
            no_candidate_count += 1
            node_count = example.inputs["node_features"].shape[0]
            extra = {
                "sample_node": torch.tensor(-1),
                "sample_node_mask": torch.tensor(False),
                "information_gain": torch.zeros(node_count, dtype=torch.float32),
                "information_gain_mask": torch.zeros(node_count, dtype=torch.bool),
                "candidate_reduction": torch.zeros(node_count, dtype=torch.float32),
                "candidate_reduction_mask": torch.zeros(node_count, dtype=torch.bool),
                "should_continue_sampling": torch.tensor(False),
            }
        else:
            matched += 1
        merged_examples.append(dataclasses.replace(example, targets={**example.targets, **extra}))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_shards(merged_examples, output_dir)
    return {
        "examples_total": len(dataset),
        "examples_with_scout_step0": matched,
        "examples_with_no_candidate_masked": no_candidate_count,
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
    result = build(args.tensor_shard_dir, args.trajectory_jsonl, args.output, expected_split=args.split)
    print(
        f"{result['examples_with_scout_step0']}/{result['examples_total']} examples have a usable Scout "
        f"step 0 ({result['examples_with_no_candidate_masked']} masked, no candidate) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
