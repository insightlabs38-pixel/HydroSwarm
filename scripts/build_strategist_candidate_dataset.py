"""core-issues3.txt Phase 10 item 3 / core-issues4.txt Section I: build a
sharded, directly-trainable candidate-conditioned Strategist dataset from a
trajectory corpus (scripts/generate_trajectory_corpus.py).

`hydroswarm.training.strategist_trajectory.build_strategist_trajectory`
produces exactly one step per scenario ("Build a single-step trajectory,"
its own docstring) -- unlike Scout, there is no multi-step conditioning
ambiguity to scope around here (see build_scout_state_dataset.py's
docstring for that unrelated issue). Every scenario's `strategist.steps[0]`
carries the full bounded candidate set (canonically 9 templates, core-
issues3.txt Phase 3.1: exactly WNTR-verified, never heuristic-prescreened)
as `labels` (semantic identity: action_template name, primary_target_id/
type) aligned 1:1 with `targets` (governed per-candidate target values:
plan_validity, plan_value, the five consequence proxies, plus an already-
resolved `target_pointer` integer index).

This script builds BOTH halves the candidate-conditioned architecture
needs from that pair:

- INPUT side (`HydroBatch`'s `plan_template_ids`/`plan_target_type`/
  `plan_target_node_index`/`plan_target_link_index`/`plan_features`/
  `plan_mask`): built from `labels`' semantic identity plus `targets`'
  already-resolved `target_pointer` (split into the node- or link-index
  field `primary_target_type` selects; the other is masked, matching
  every other DUAL_INDEX_TARGETS convention in this codebase).
- TARGET side (`plan_validity`/`plan_value`/`exposure_proxy`/
  `pressure_risk_proxy`/`service_loss_proxy`/`containment_time_proxy`/
  `plan_regret_proxy`): copied directly from `targets` -- already exact-
  WNTR-derived, nothing recomputed here.

`plan_features` scoping decision, deliberate and documented (not an
oversight)
---------------------------------------------------------------------
CandidatePlanEncoder's own docstring describes `plan_features` as
"deterministic proposal features (e.g. the classical prescreener's own
predicted_value/predicted_validity, action count, start/duration/
magnitude, source-region overlap, critical-demand relevance, downstream
relationship)". None of that is present in the persisted trajectory JSONL
today -- computing it for real requires the original network/incident
context at GENERATION time (inside strategist_trajectory.py), not
something derivable from the already-serialized labels alone. Inventing
plausible-looking numbers here would be exactly the kind of fabricated
label this project's own restrictions forbid ("All labels must be derived
from exact simulation, deterministic policy, ... Do not invent manually
invented labels" -- core-issues2.txt Phase 1 item 4, applies equally to
input features masquerading as derived facts).

This script therefore builds `plan_features` from ONLY genuinely available,
leakage-free, purely structural facts already in `labels[i]`/`targets[i]`
-- properties of the action_template/target type themselves, never
anything verification-derived. See `PLAN_FEATURE_NAMES` and
`_plan_features_for_label` below for the real 6 dims (matching
`HydroCore.PLAN_FEATURE_DIM`'s default width so this works with an
unmodified `HydroCore.from_variant(...)` call). A richer, generation-time-
computed feature set (the classical prescreener's own predicted_value/
predicted_validity, action count, source-region overlap, critical-demand
relevance, downstream relationship) is real follow-up work, not attempted
here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from hydroswarm.model.candidate_plan_encoder import TARGET_TYPE_INDEX
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_INDEX
from hydroswarm.planning.candidate_tensorizer import PLAN_FEATURE_NAMES
from hydroswarm.training.data import ScenarioExample
from hydroswarm.training.sharded_data import ShardedScenarioDataset, write_shards

#: HydroBatch's candidate-plan INPUT fields (model/core.py Section E) --
#: everything else this script writes onto an example is a governed target.
_PLAN_INPUT_KEYS = frozenset(
    {"plan_template_ids", "plan_target_type", "plan_target_node_index", "plan_target_link_index", "plan_features", "plan_mask"}
)

#: core-issues5.txt Section 6: PLAN_FEATURE_NAMES itself now lives in
#: hydroswarm.planning.candidate_tensorizer (the live-runtime candidate
#: tensorizer's own module) so the 6-dimensional plan_features schema has
#: exactly one definition shared by both this training-corpus builder and
#: live PASS-2 candidate scoring -- see that module's docstring for why the
#: 6th dimension's live-vs-training meaning differs slightly.

_PLAN_TARGET_VALUE_KEYS = (
    "plan_validity",
    "plan_value",
    "exposure_proxy",
    "pressure_risk_proxy",
    "service_loss_proxy",
    "containment_time_proxy",
    "plan_regret_proxy",
)


def _plan_features_for_label(label: dict[str, Any], target: dict[str, Any]) -> list[float]:
    target_type = label["primary_target_type"]
    return [
        float(target_type == "NONE"),
        float(target_type == "NODE"),
        float(target_type == "LINK"),
        float(label["primary_target_id"] is not None),
        float(bool(label.get("is_no_response_comparator", False))),
        float(bool(target.get("plan_validity_mask", True))),
    ]


def _candidate_plan_tensors(labels: list[dict[str, Any]], targets: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    plan_count = len(labels)
    template_ids = torch.zeros(plan_count, dtype=torch.long)
    target_type = torch.zeros(plan_count, dtype=torch.long)
    node_index = torch.full((plan_count,), -1, dtype=torch.long)
    link_index = torch.full((plan_count,), -1, dtype=torch.long)
    features = torch.zeros(plan_count, len(PLAN_FEATURE_NAMES), dtype=torch.float32)
    plan_mask = torch.ones(plan_count, dtype=torch.bool)

    value_arrays: dict[str, list[float]] = {key: [] for key in _PLAN_TARGET_VALUE_KEYS}
    mask_arrays: dict[str, list[bool]] = {f"{key}_mask": [] for key in _PLAN_TARGET_VALUE_KEYS}

    for position, (label, target) in enumerate(zip(labels, targets, strict=True)):
        template_ids[position] = ACTION_TEMPLATE_INDEX[label["action_template"]]
        primary_type = label["primary_target_type"]
        target_type[position] = TARGET_TYPE_INDEX[primary_type]
        pointer = int(target["target_pointer"])
        pointer_valid = bool(target.get("target_pointer_mask", False))
        # If pointer_valid is False (strategist_trajectory.py's own
        # _resolve_target_pointer returns (-1, False) rather than invent
        # one), both index fields correctly stay at their -1 sentinel --
        # the model's own is_node_target/is_link_target gating already
        # requires target_type == that pointer's class AND a real,
        # non-sentinel index, so an unresolved pointer simply never
        # contributes a gathered embedding regardless of target_type.
        if pointer_valid and primary_type == "NODE":
            node_index[position] = pointer
        elif pointer_valid and primary_type == "LINK":
            link_index[position] = pointer
        features[position] = torch.tensor(_plan_features_for_label(label, target))
        for key in _PLAN_TARGET_VALUE_KEYS:
            value_arrays[key].append(float(target[key]))
            mask_arrays[f"{key}_mask"].append(bool(target.get(f"{key}_mask", True)))

    tensors: dict[str, torch.Tensor] = {
        "plan_template_ids": template_ids,
        "plan_target_type": target_type,
        "plan_target_node_index": node_index,
        "plan_target_link_index": link_index,
        "plan_features": features,
        "plan_mask": plan_mask,
    }
    for key in _PLAN_TARGET_VALUE_KEYS:
        dtype = torch.bool if key == "plan_validity" else torch.float32
        tensors[key] = torch.tensor(value_arrays[key], dtype=dtype)
        tensors[f"{key}_mask"] = torch.tensor(mask_arrays[f"{key}_mask"], dtype=torch.bool)
    return tensors


def _masked_placeholder(plan_count: int) -> dict[str, torch.Tensor]:
    """No usable Strategist step (defensive -- build_strategist_trajectory's
    own docstring says this should not happen, but masked-not-dropped is
    this codebase's universal convention for "no real data here," matching
    build_scout_state_dataset.py's identical fallback)."""

    tensors: dict[str, torch.Tensor] = {
        "plan_template_ids": torch.zeros(plan_count, dtype=torch.long),
        "plan_target_type": torch.zeros(plan_count, dtype=torch.long),
        "plan_target_node_index": torch.full((plan_count,), -1, dtype=torch.long),
        "plan_target_link_index": torch.full((plan_count,), -1, dtype=torch.long),
        "plan_features": torch.zeros(plan_count, len(PLAN_FEATURE_NAMES), dtype=torch.float32),
        "plan_mask": torch.zeros(plan_count, dtype=torch.bool),
    }
    for key in _PLAN_TARGET_VALUE_KEYS:
        dtype = torch.bool if key == "plan_validity" else torch.float32
        tensors[key] = torch.zeros(plan_count, dtype=dtype)
        tensors[f"{key}_mask"] = torch.zeros(plan_count, dtype=torch.bool)
    return tensors


def _load_strategist_step0(jsonl_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    tensors_by_scenario: dict[str, dict[str, torch.Tensor]] = {}
    with jsonl_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            steps = record["strategist"]["steps"]
            if not steps:
                continue
            step0 = steps[0]
            tensors_by_scenario[record["scenario_id"]] = _candidate_plan_tensors(step0["labels"], step0["targets"])
    return tensors_by_scenario


def build(
    tensor_shard_dir: Path, trajectory_jsonl: Path, output_dir: Path, *, expected_split: str
) -> dict[str, Any]:
    dataset = ShardedScenarioDataset(tensor_shard_dir, expected_split=expected_split)
    dataset.verify_shard_checksums()
    strategist_candidates = _load_strategist_step0(trajectory_jsonl)

    # Per-scenario candidate count is NOT required to be uniform across a
    # split -- collate_variable_topology's own plan-padding (Phase 10.3)
    # pads a genuinely variable P across a batch, same as it pads a
    # genuinely variable node count. The masked "no usable step" placeholder
    # below uses the canonical full template count (every real scenario's
    # own upper bound) purely as its own size, not an assumption that every
    # OTHER scenario in the split shares it.
    canonical_plan_count = len(ACTION_TEMPLATE_INDEX)

    merged_examples: list[ScenarioExample] = []
    matched = 0
    no_step_count = 0
    for position in range(len(dataset)):
        example = dataset[position]
        candidate = strategist_candidates.get(example.scenario_id)
        if candidate is None:
            no_step_count += 1
            candidate_inputs = _masked_placeholder(canonical_plan_count)
        else:
            matched += 1
            candidate_inputs = candidate

        new_inputs = {key: value for key, value in candidate_inputs.items() if key in _PLAN_INPUT_KEYS}
        new_targets = {key: value for key, value in candidate_inputs.items() if key not in _PLAN_INPUT_KEYS}
        merged_examples.append(
            dataclasses.replace(
                example,
                inputs={**example.inputs, **new_inputs},
                targets={**example.targets, **new_targets},
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_shards(merged_examples, output_dir)
    return {
        "examples_total": len(dataset),
        "examples_with_strategist_step0": matched,
        "examples_with_no_step_masked": no_step_count,
        "canonical_plan_count": canonical_plan_count,
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
        f"{result['examples_with_strategist_step0']}/{result['examples_total']} examples have a usable "
        f"Strategist step 0 ({result['examples_with_no_step_masked']} masked, no step; "
        f"{result['canonical_plan_count']} candidates/example) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
