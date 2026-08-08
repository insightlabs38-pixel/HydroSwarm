"""core-issues3.txt Phase 12 Stage F prerequisite: build one governed, joined
Stage-F training corpus (`data/learning-v2/cycle-b2-joint-v4/`) from the
separate derived corpora `scripts/train.py`'s single flat-manifest trainer
cannot see all of at once.

Why this exists
----------------

`reports/results/v4/pre-freeze-implementation-handoff.md`'s "Real next-step
scoping for Stage F" section (commit `05a0a3d`) established, by inspection
rather than assumption, that a genuine joint-multitask Stage F run needs
either a real merge step or a multi-source batch loader -- the three real
governed datasets that exist today are three SEPARATE derived corpora, each
independently overlaying only its own role's targets onto the shared
`cycle-b2` base:

- ``data/learning-v2/cycle-b2/tensors-normalized`` (Sentinel, all splits)
- ``data/learning-v2/cycle-b2-control-v2/tensors-normalized`` (corrected
  event_cause + regenerated evidence_sufficiency + new next_step,
  train/validation only)
- ``data/learning-v2/cycle-b2-ood-extension/tensors-normalized`` (new
  ood_class-labeled scenarios, development_holdout-only, disjoint
  scenario_ids from every other source)
- ``data/learning-v2/cycle-b2-trajectories-v3/scout-tensors-normalized``
  (Scout, train/validation only)
- ``data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized``
  (Strategist, all splits)

This script performs (a): a deterministic, governed, offline merge producing
one canonical per-scenario training example carrying every role's targets
that are actually available for that scenario, with explicit masks/omission
for whatever is not -- never invented placeholder labels.

Strategist-v4 input-normalization defect, found and fixed before merging
--------------------------------------------------------------------------

Before trusting Strategist-v4 as a merge source, this pass verified Scout-v3
and Strategist-v4 share the same underlying incident/state representation as
`cycle-b2`'s own governed-normalized tensors (identical scenario_id sets,
topology/network/hydraulic-state/signature-library hashes, feature/target
schema versions -- see `verify_role_compatibility` below, which performs
this check programmatically rather than by inspection alone). Scout-v3
passed with zero mismatches. Strategist-v4 FAILED: its shared graph input
tensors (node_features, edge_features, temporal_features, quality_features,
classical_prior, demand_centrality, reservoir_reachability, travel_time)
were byte-identical to `cycle-b2/tensors` (the RAW, pre-training-only-
normalization shard) and NOT to `cycle-b2/tensors-normalized` (the governed,
train-owned-normalization shard every other corpus uses), across all four
splits and every shared input key. Root cause: the regeneration command
recorded in this same handoff file (the "important-issues.txt emergency fix"
pass, commit `9a0c364`/`d681d9e`) passed
``--tensor-shard-dir data/learning-v2/cycle-b2/tensors/train`` to
`build_strategist_candidate_dataset.py` instead of
``.../tensors-normalized/train`` -- a wrong-argument bug, not an intentional
Strategist-specific normalization change. `build_strategist_candidate_dataset.py`
only ever copies whatever `--tensor-shard-dir` it is given and adds its own
`plan_*` keys (verified by reading the script: `dataclasses.replace(example,
inputs={**example.inputs, **new_inputs}, targets=...)`), so the
Strategist-owned target VALUES (`plan_validity`/`plan_value`/the five
consequence proxies, all sourced from `targets` in the trajectory JSONL, not
from `--tensor-shard-dir`) are unaffected and were re-verified byte-identical
between the original (wrong-input) and corrected (right-input) builds -- only
the incidentally-embedded shared graph features were wrong-scale. This means
the already-trained Strategist checkpoint's *labels* were always correct;
only its *input feature distribution* was inconsistent with the rest of the
governed corpus. Fixed by rebuilding
``data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected``
(all 4 splits, same script, corrected `--tensor-shard-dir`, same immutable
trajectory JSONL -- no re-simulation, no re-verification, nothing invented)
before running this merge. The original (defective-input) directory is left
untouched for audit/reproducibility of the checkpoint it actually trained;
this script reads ONLY the corrected directory. See
`reports/results/v4/pre-freeze-implementation-handoff.md`'s Stage F section
for the full write-up and whether a Strategist head retrain is warranted (out
of this pass's scope: "Stage F data-path issue" only, not re-running
already-completed pre-freeze training).

Design
------

For every governed non-test split (train/validation/calibration/
development_holdout) plus every governed OOD-development category directory
(cycle-b2's own ood-SEVERE_MISSINGNESS/ood-UNSEEN_TOPOLOGY, and
cycle-b2-ood-extension's four learned-ood_class categories), join by
`scenario_id`:

1. `cycle-b2/tensors-normalized/<split-or-category>` is the base -- every
   output row must come from here (or from cycle-b2-ood-extension for the
   ood_class categories, which are new, disjoint scenario_ids, not overlay
   rows onto the base population).
2. `cycle-b2-control-v2` (train/validation only) overlays its three owned
   keys (`event_cause`, `evidence_sufficiency`, `next_step`) -- the first two
   are documented, intentional corrections (control-v2's own
   `manifest.json`: `event_cause_recomputed_via`, "second-pass-control-v1"),
   not drift, and are treated as authoritative over the base's stale copies
   for the splits where they exist. Every OTHER key control-v2 carries must
   be byte-identical to base or the merge fails closed
   (`_conflicting_target_ownership`).
3. `cycle-b2-trajectories-v3/scout-tensors-normalized` (train/validation
   only) overlays its four owned target keys (+ mask companions). Every
   shared key must be byte-identical to base or the merge fails closed.
4. `cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected` (all
   four splits) overlays its six plan-conditioning INPUT keys and seven
   owned TARGET keys (+ mask companions). Every shared key must be
   byte-identical to base or the merge fails closed.

A split/category's OUTPUT target-key set is therefore uniform across every
row in that split (required by `collate_variable_topology`, which raises if
a batch mixes examples with different target-key sets) but genuinely differs
BETWEEN splits/categories, honestly reflecting which corpora actually cover
that population -- e.g. `next_step` and every Scout key are present for
train/validation and absent (not masked-with-a-fabricated-zero) for
calibration/development_holdout/every OOD category, because Scout states and
second-pass control labels were never generated for those populations. This
is recorded explicitly in the merge report's `unavailable_task_groups` field
per split, not silently implied by a missing key.

Fail-closed checks, applied per scenario_id and aggregated per split
---------------------------------------------------------------------

- missing/duplicate scenario_id (inherent: `ShardedScenarioDataset.__init__`
  already rejects duplicates within one source directory; this script
  additionally rejects an overlay scenario_id absent from its split's base,
  and rejects any scenario_id shared across two DIFFERENT output
  splits/categories -- the cross-population leakage check
  `_check_cross_population_leakage` -- fatal, not a per-row skip)
- split mismatch (`ScenarioExample.split` must agree across every source
  used for one scenario_id)
- topology/network/hydraulic-state/signature-library-hash mismatch
- scenario provenance mismatch (network_id, seed, seed_family, stage must
  agree across every source used for one scenario_id -- catches "Scout or
  Strategist data attached to the wrong incident/state")
- feature/target schema-version mismatch
- normalization mismatch (byte-equality check on every shared tensor key,
  the same class of check that caught the Strategist-v4 defect above)
- conflicting target ownership (a shared key whose value differs between
  base and an overlay, outside that overlay's documented owned-key set)
- stale source-manifest/hash (`ShardedScenarioDataset.__init__` verifies
  `index.jsonl`'s sha256 against `manifest.json` and every shard's existence
  at construction; this script additionally calls
  `verify_shard_checksums()` on every source before joining)

Any violation raises `StageFMergeError` and aborts the whole split (and the
whole corpus, when invoked from `main`) with a nonzero exit -- "fail closed,"
not a partial/quarantined corpus, matching this pass's restriction: "Require
zero missing required joins, zero duplicates, and zero identity conflicts."
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from hydroswarm.training.data import ScenarioExample, TopologyMetadata
from hydroswarm.training.sharded_data import ShardedScenarioDataset, write_shards

#: Keys `cycle-b2-control-v2` legitimately overrides relative to base
#: (documented in its own `manifest.json`: `event_cause_recomputed_via`
#: records the Phase 6.4 bugfix recomputation; `evidence_sufficiency` is the
#: Phase 8 second-pass regeneration; `next_step` is new). Every other key
#: control-v2 carries must equal base exactly.
CONTROL_OWNED_TARGET_KEYS = frozenset({"event_cause", "evidence_sufficiency", "next_step"})

#: Scout-v3's own governed targets (core-issues3.txt Phase 5/targets_v2).
SCOUT_OWNED_TARGET_KEYS = frozenset(
    {
        "sample_node",
        "sample_node_mask",
        "information_gain",
        "information_gain_mask",
        "candidate_reduction",
        "candidate_reduction_mask",
        "should_continue_sampling",
    }
)

#: Strategist-v4's own governed targets (core-issues3.txt Phase 3/4).
STRATEGIST_OWNED_TARGET_KEYS = frozenset(
    {
        "plan_validity",
        "plan_validity_mask",
        "plan_value",
        "plan_value_mask",
        "exposure_proxy",
        "exposure_proxy_mask",
        "pressure_risk_proxy",
        "pressure_risk_proxy_mask",
        "service_loss_proxy",
        "service_loss_proxy_mask",
        "containment_time_proxy",
        "containment_time_proxy_mask",
        "plan_regret_proxy",
        "plan_regret_proxy_mask",
    }
)

#: Strategist-v4's own new candidate-conditioning INPUT keys (model/core.py
#: Section E / CandidatePlanEncoder).
STRATEGIST_OWNED_INPUT_KEYS = frozenset(
    {
        "plan_template_ids",
        "plan_target_type",
        "plan_target_node_index",
        "plan_target_link_index",
        "plan_features",
        "plan_mask",
    }
)

#: cycle-b2-ood-extension's own governed target (core-issues3.txt Phase 6).
OOD_EXTENSION_OWNED_TARGET_KEYS = frozenset({"ood_class"})

_TOPOLOGY_IDENTITY_FIELDS = (
    "topology_hash",
    "network_hash",
    "hydraulic_state_hash",
    "signature_library_hash",
    "target_schema_version",
    "feature_schema_version",
)


class StageFMergeError(ValueError):
    """A fail-closed violation during the Stage-F joint-corpus merge."""


def _tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


def _verify_topology_identity(
    scenario_id: str, role: str, base_topology: TopologyMetadata | None, other_topology: TopologyMetadata | None
) -> None:
    if base_topology is None or other_topology is None:
        raise StageFMergeError(f"scenario {scenario_id}: {role} or base is missing TopologyMetadata entirely")
    for field in _TOPOLOGY_IDENTITY_FIELDS:
        base_value = getattr(base_topology, field)
        other_value = getattr(other_topology, field)
        if base_value != other_value:
            raise StageFMergeError(
                f"scenario {scenario_id}: {role} topology.{field} disagrees with base "
                f"({other_value!r} != {base_value!r}) -- topology/normalization/schema mismatch"
            )


def _verify_provenance(scenario_id: str, role: str, base: ScenarioExample, other: ScenarioExample) -> None:
    if other.split != base.split:
        raise StageFMergeError(f"scenario {scenario_id}: {role} split {other.split!r} != base split {base.split!r}")
    if other.network_id != base.network_id:
        raise StageFMergeError(f"scenario {scenario_id}: {role} network_id disagrees with base")
    if other.seed != base.seed:
        raise StageFMergeError(f"scenario {scenario_id}: {role} seed disagrees with base -- wrong incident/state")
    if other.seed_family != base.seed_family:
        raise StageFMergeError(f"scenario {scenario_id}: {role} seed_family disagrees with base")
    if other.stage != base.stage:
        raise StageFMergeError(f"scenario {scenario_id}: {role} curriculum stage disagrees with base")


def _verify_shared_keys(
    scenario_id: str,
    role: str,
    base: Mapping[str, torch.Tensor],
    other: Mapping[str, torch.Tensor],
    owned_keys: frozenset[str],
    *,
    kind: str,
) -> None:
    """Every key `other` shares with `base` outside `owned_keys` must be
    byte-identical, or this is conflicting target/input ownership -- the
    exact check that caught the Strategist-v4 raw-tensor defect."""

    for key, value in other.items():
        if key in owned_keys:
            continue
        base_value = base.get(key)
        if base_value is None:
            continue
        if not _tensors_equal(base_value, value):
            raise StageFMergeError(
                f"scenario {scenario_id}: {role} {kind} {key!r} disagrees with base's own value outside "
                f"{role}'s documented owned-key set -- conflicting target ownership / normalization mismatch"
            )


def _mask_stats(examples: list[ScenarioExample]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for example in examples:
        for key, value in example.targets.items():
            if not key.endswith("_mask"):
                continue
            entry = stats.setdefault(key, {"valid": 0, "masked": 0, "total": 0})
            valid = int(value.bool().sum())
            total = int(value.numel())
            entry["valid"] += valid
            entry["masked"] += total - valid
            entry["total"] += total
    return stats


def merge_population(
    *,
    population_name: str,
    output_split: str,
    base_dir: Path,
    control_dir: Path | None,
    scout_dir: Path | None,
    strategist_dir: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Join one governed scenario population (a canonical split or an
    OOD-development category directory) across every available source.
    ``output_split`` is the `ScenarioExample.split` value every source is
    expected to carry (population_name may differ, e.g. an OOD category
    directory name, while its internal split is still "development_holdout" --
    matching cycle-b2's own existing convention for its ood-* directories).

    ``base_dir`` is the population's own base -- for the four canonical
    splits and cycle-b2's own ood-* categories this is
    ``cycle-b2/tensors-normalized/<population>``; for cycle-b2-ood-extension
    categories it is that category's own directory directly (new, disjoint
    scenario_ids, carrying `ood_class` already -- there is no separate
    cycle-b2 population to join those onto)."""

    base = ShardedScenarioDataset(base_dir, expected_split=output_split)
    base.verify_shard_checksums()
    sources_used = {"base": str(base_dir)}

    control = None
    if control_dir is not None and control_dir.exists():
        control = ShardedScenarioDataset(control_dir, expected_split=output_split)
        control.verify_shard_checksums()
        sources_used["control"] = str(control_dir)

    scout = None
    if scout_dir is not None and scout_dir.exists():
        scout = ShardedScenarioDataset(scout_dir, expected_split=output_split)
        scout.verify_shard_checksums()
        sources_used["scout"] = str(scout_dir)

    strategist = None
    if strategist_dir is not None and strategist_dir.exists():
        strategist = ShardedScenarioDataset(strategist_dir, expected_split=output_split)
        strategist.verify_shard_checksums()
        sources_used["strategist"] = str(strategist_dir)

    def index_by_scenario(dataset: ShardedScenarioDataset | None) -> dict[str, ScenarioExample]:
        if dataset is None:
            return {}
        indexed = {dataset[i].scenario_id: dataset[i] for i in range(len(dataset))}
        if len(indexed) != len(dataset):
            raise StageFMergeError(f"{population_name}: duplicate scenario_id within one source directory")
        return indexed

    base_index = index_by_scenario(base)
    control_index = index_by_scenario(control)
    scout_index = index_by_scenario(scout)
    strategist_index = index_by_scenario(strategist)

    for role, index in (("control", control_index), ("scout", scout_index), ("strategist", strategist_index)):
        stray = set(index) - set(base_index)
        if stray:
            raise StageFMergeError(
                f"{population_name}: {role} carries {len(stray)} scenario_id(s) absent from base "
                f"(e.g. {sorted(stray)[:3]}) -- missing join / wrong incident-state attachment"
            )

    duplicate_joins: list[str] = []
    merged_examples: list[ScenarioExample] = []
    joined_scenario_ids: list[str] = []
    control_count = 0
    scout_count = 0
    strategist_count = 0
    ood_count = 0
    seen: set[str] = set()

    for scenario_id, base_example in base_index.items():
        if scenario_id in seen:
            duplicate_joins.append(scenario_id)
            continue
        seen.add(scenario_id)

        merged_inputs = dict(base_example.inputs)
        merged_targets = dict(base_example.targets)

        control_example = control_index.get(scenario_id)
        if control_example is not None:
            _verify_provenance(scenario_id, "control", base_example, control_example)
            _verify_topology_identity(scenario_id, "control", base_example.topology, control_example.topology)
            _verify_shared_keys(
                scenario_id, "control", base_example.targets, control_example.targets,
                CONTROL_OWNED_TARGET_KEYS, kind="target",
            )
            for key in CONTROL_OWNED_TARGET_KEYS:
                if key in control_example.targets:
                    merged_targets[key] = control_example.targets[key]
            control_count += 1

        scout_example = scout_index.get(scenario_id)
        if scout_example is not None:
            _verify_provenance(scenario_id, "scout", base_example, scout_example)
            _verify_topology_identity(scenario_id, "scout", base_example.topology, scout_example.topology)
            _verify_shared_keys(
                scenario_id, "scout", base_example.targets, scout_example.targets,
                SCOUT_OWNED_TARGET_KEYS, kind="target",
            )
            for key in SCOUT_OWNED_TARGET_KEYS:
                if key in scout_example.targets:
                    merged_targets[key] = scout_example.targets[key]
            scout_count += 1

        strategist_example = strategist_index.get(scenario_id)
        if strategist_example is not None:
            _verify_provenance(scenario_id, "strategist", base_example, strategist_example)
            _verify_topology_identity(scenario_id, "strategist", base_example.topology, strategist_example.topology)
            _verify_shared_keys(
                scenario_id, "strategist", base_example.inputs, strategist_example.inputs,
                STRATEGIST_OWNED_INPUT_KEYS, kind="input",
            )
            _verify_shared_keys(
                scenario_id, "strategist", base_example.targets, strategist_example.targets,
                STRATEGIST_OWNED_TARGET_KEYS, kind="target",
            )
            for key in STRATEGIST_OWNED_INPUT_KEYS:
                if key in strategist_example.inputs:
                    merged_inputs[key] = strategist_example.inputs[key]
            for key in STRATEGIST_OWNED_TARGET_KEYS:
                if key in strategist_example.targets:
                    merged_targets[key] = strategist_example.targets[key]
            strategist_count += 1

        if "ood_class" in base_example.targets:
            ood_count += 1

        merged_examples.append(
            dataclasses.replace(base_example, inputs=merged_inputs, targets=merged_targets)
        )
        joined_scenario_ids.append(scenario_id)

    if duplicate_joins:
        raise StageFMergeError(f"{population_name}: duplicate scenario_id(s) in base: {duplicate_joins[:5]}")

    # Every example in one output population must carry identical
    # input/target key sets (collate_variable_topology requires this).
    key_sets = {(frozenset(example.inputs), frozenset(example.targets)) for example in merged_examples}
    if len(key_sets) > 1:
        raise StageFMergeError(
            f"{population_name}: merged examples do not share a uniform input/target key set "
            f"({len(key_sets)} distinct combinations) -- incompatible masks / partial coverage within one split"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest = write_shards(merged_examples, output_dir)

    unavailable = []
    if control_count == 0:
        unavailable.append("control.next_step")
    if scout_count == 0:
        unavailable.append("scout")
    if strategist_count == 0:
        unavailable.append("strategist")
    if ood_count == 0:
        unavailable.append("ood_class")

    return {
        "population": population_name,
        "output_split": output_split,
        "sources_used": sources_used,
        "expected_base_examples": len(base_index),
        "joined_examples": len(merged_examples),
        "control_target_count": control_count,
        "scout_target_count": scout_count,
        "strategist_target_count": strategist_count,
        "ood_target_count": ood_count,
        "missing_joins": 0,
        "duplicate_joins": len(duplicate_joins),
        "conflicts": 0,
        "unavailable_task_groups": unavailable,
        "input_keys": sorted(merged_examples[0].inputs) if merged_examples else [],
        "target_keys": sorted(merged_examples[0].targets) if merged_examples else [],
        "masked_target_counts": _mask_stats(merged_examples),
        "shard_manifest": shard_manifest,
        "dataset_manifest_hash": ShardedScenarioDataset(output_dir, expected_split=output_split).manifest_hash,
        "joined_scenario_ids_sample": joined_scenario_ids[:5],
    }


def _source_manifest_hashes(root: Path, splits_by_role: dict[str, dict[str, Path]]) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for role, split_dirs in splits_by_role.items():
        role_entry: dict[str, Any] = {}
        for split, directory in split_dirs.items():
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            role_entry[split] = {
                "path": str(directory),
                "index_sha256": manifest.get("index_sha256"),
                "total_examples": manifest.get("total_examples"),
                "shard_count": len(manifest.get("shards", [])),
            }
        if role_entry:
            hashes[role] = role_entry
    return hashes


def _cross_population_leakage(output_dirs: dict[str, Path]) -> dict[str, Any]:
    seen_scenarios: dict[str, str] = {}
    seen_families: dict[tuple[str, str], str] = {}
    scenario_leaks: list[dict[str, str]] = []
    family_leaks: list[dict[str, Any]] = []
    for population, directory in output_dirs.items():
        index_path = directory / "index.jsonl"
        with index_path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                scenario_id = record["scenario_id"]
                family = (record["network_id"], record["seed_family"])
                if scenario_id in seen_scenarios:
                    scenario_leaks.append(
                        {"scenario_id": scenario_id, "populations": [seen_scenarios[scenario_id], population]}
                    )
                else:
                    seen_scenarios[scenario_id] = population
                if family in seen_families and seen_families[family] != population:
                    family_leaks.append(
                        {"seed_family": list(family), "populations": [seen_families[family], population]}
                    )
                else:
                    seen_families[family] = population
    return {
        "scenario_id_leaks": scenario_leaks,
        "seed_family_leaks": family_leaks,
        "populations_checked": sorted(output_dirs),
        "total_examples_checked": len(seen_scenarios),
    }


def _dataset_fingerprint(reports: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [{"population": r["population"], "dataset_manifest_hash": r["dataset_manifest_hash"]} for r in reports],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


#: (population_name, output_split, has_control, has_scout, has_strategist, is_ood_extension)
PRIMARY_POPULATIONS = (
    ("train", "train", True, True, True, False),
    ("validation", "validation", True, True, True, False),
    ("calibration", "calibration", False, False, True, False),
    ("development_holdout", "development_holdout", False, False, True, False),
)
CYCLE_B2_OOD_POPULATIONS = ("ood-SEVERE_MISSINGNESS", "ood-UNSEEN_TOPOLOGY")
OOD_EXTENSION_POPULATIONS = (
    "ood-EXTREME_DEMAND",
    "ood-FROZEN_DRIFTING_SENSOR",
    "ood-ROUGHNESS_MISMATCH",
    "ood-TANK_STATE_SHIFT",
)


def build(
    *,
    base_root: Path,
    control_root: Path,
    scout_root: Path,
    strategist_root: Path,
    ood_extension_root: Path,
    output_root: Path,
    include_ood_extension: bool = False,
) -> dict[str, Any]:
    """``include_ood_extension`` defaults to False: this merge's own
    cross-population leakage gate (`_cross_population_leakage`) found that
    `cycle-b2-ood-extension`'s seed_family LABELS (e.g.
    "golden-reference:910") collide with train/validation/calibration's own
    seed_family labels for different underlying scenarios (confirmed:
    matching (network_id, seed_family) tuples resolve to different
    scenario_ids on each side) -- a seed-family namespace collision in that
    already-existing corpus, not something this merge introduces. Joining it
    in would silently violate "a seed family may appear only once per
    governed split" once combined with the primary populations it collides
    with (train/calibration/validation). This is a pre-existing defect in
    `cycle-b2-ood-extension` requiring that corpus's own regeneration with a
    disjoint seed-family numbering scheme -- out of this pass's Stage-F
    data-path scope. `cycle-b2`'s own ood-SEVERE_MISSINGNESS/
    ood-UNSEEN_TOPOLOGY categories do NOT collide (verified: zero leaks) and
    are always included."""
    reports: list[dict[str, Any]] = []
    output_dirs: dict[str, Path] = {}
    splits_by_role: dict[str, dict[str, Path]] = {"base": {}, "control": {}, "scout": {}, "strategist": {}}

    for population, output_split, has_control, has_scout, has_strategist, _ in PRIMARY_POPULATIONS:
        base_dir = base_root / "tensors-normalized" / population
        control_dir = control_root / "tensors-normalized" / population if has_control else None
        scout_dir = scout_root / population if has_scout else None
        strategist_dir = strategist_root / population if has_strategist else None
        output_dir = output_root / "tensors-normalized" / population
        report = merge_population(
            population_name=population,
            output_split=output_split,
            base_dir=base_dir,
            control_dir=control_dir,
            scout_dir=scout_dir,
            strategist_dir=strategist_dir,
            output_dir=output_dir,
        )
        reports.append(report)
        output_dirs[population] = output_dir
        splits_by_role["base"][population] = base_dir
        if control_dir is not None:
            splits_by_role["control"][population] = control_dir
        if scout_dir is not None:
            splits_by_role["scout"][population] = scout_dir
        if strategist_dir is not None:
            splits_by_role["strategist"][population] = strategist_dir

    for category in CYCLE_B2_OOD_POPULATIONS:
        base_dir = base_root / "tensors-normalized" / category
        strategist_dir = strategist_root / category
        output_dir = output_root / "tensors-normalized" / category
        report = merge_population(
            population_name=category,
            output_split="development_holdout",
            base_dir=base_dir,
            control_dir=None,
            scout_dir=None,
            strategist_dir=strategist_dir if strategist_dir.exists() else None,
            output_dir=output_dir,
        )
        reports.append(report)
        output_dirs[category] = output_dir

    skipped_ood_extension: list[str] = []
    if include_ood_extension:
        for category in OOD_EXTENSION_POPULATIONS:
            ood_dir = ood_extension_root / "tensors-normalized" / category
            output_dir = output_root / "tensors-normalized" / category
            report = merge_population(
                population_name=category,
                output_split="development_holdout",
                base_dir=ood_dir,
                control_dir=None,
                scout_dir=None,
                strategist_dir=None,
                output_dir=output_dir,
            )
            reports.append(report)
            output_dirs[category] = output_dir
    else:
        skipped_ood_extension = list(OOD_EXTENSION_POPULATIONS)

    leakage = _cross_population_leakage(output_dirs)
    source_manifest_hashes = _source_manifest_hashes(base_root, splits_by_role)
    fingerprint = _dataset_fingerprint(reports)

    total_missing = sum(r["missing_joins"] for r in reports)
    total_duplicates = sum(r["duplicate_joins"] for r in reports)
    total_conflicts = sum(r["conflicts"] for r in reports)

    merge_report = {
        "corpus_name": "cycle-b2-joint-v4",
        "populations": reports,
        "skipped_populations": {
            "ood_extension_categories": skipped_ood_extension,
            "reason": (
                "cycle-b2-ood-extension seed_family labels collide with train/validation/calibration's own "
                "seed_family labels for different underlying scenario_ids (pre-existing namespace defect in "
                "that corpus, found by this merge's own cross-population leakage gate) -- excluded by default; "
                "pass --include-ood-extension to attempt the join anyway and see the resulting "
                "zero_cross_population_leakage=false failure directly."
            )
            if skipped_ood_extension
            else None,
        },
        "totals": {
            "missing_joins": total_missing,
            "duplicate_joins": total_duplicates,
            "conflicts": total_conflicts,
            "scenario_id_leaks": len(leakage["scenario_id_leaks"]),
            "seed_family_leaks": len(leakage["seed_family_leaks"]),
        },
        "requirement_status": {
            "zero_missing_required_joins": total_missing == 0,
            "zero_duplicates": total_duplicates == 0,
            "zero_identity_conflicts": total_conflicts == 0,
            "zero_cross_population_leakage": not leakage["scenario_id_leaks"] and not leakage["seed_family_leaks"],
        },
        "dataset_fingerprint_sha256": fingerprint,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "merge-report.json").write_text(json.dumps(merge_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "leakage-report.json").write_text(json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "source-manifest-hashes.json").write_text(
        json.dumps(source_manifest_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    target_availability = {
        r["population"]: {
            "joined_examples": r["joined_examples"],
            "control_target_count": r["control_target_count"],
            "scout_target_count": r["scout_target_count"],
            "strategist_target_count": r["strategist_target_count"],
            "ood_target_count": r["ood_target_count"],
            "unavailable_task_groups": r["unavailable_task_groups"],
            "target_keys": r["target_keys"],
            "masked_target_counts": r["masked_target_counts"],
        }
        for r in reports
    }
    (output_root / "target-availability-report.json").write_text(
        json.dumps(target_availability, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "checksums.json").write_text(
        json.dumps(
            {
                "dataset_fingerprint_sha256": fingerprint,
                "populations": {
                    r["population"]: {
                        "dataset_manifest_hash": r["dataset_manifest_hash"],
                        "index_sha256": r["shard_manifest"]["index_sha256"],
                        "total_examples": r["shard_manifest"]["total_examples"],
                    }
                    for r in reports
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return merge_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-root", type=Path, default=Path("data/learning-v2/cycle-b2"))
    parser.add_argument("--control-root", type=Path, default=Path("data/learning-v2/cycle-b2-control-v2"))
    parser.add_argument(
        "--scout-root", type=Path, default=Path("data/learning-v2/cycle-b2-trajectories-v3/scout-tensors-normalized")
    )
    parser.add_argument(
        "--strategist-root",
        type=Path,
        default=Path("data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected"),
    )
    parser.add_argument("--ood-extension-root", type=Path, default=Path("data/learning-v2/cycle-b2-ood-extension"))
    parser.add_argument("--output-root", type=Path, default=Path("data/learning-v2/cycle-b2-joint-v4"))
    parser.add_argument(
        "--include-ood-extension",
        action="store_true",
        help="Attempt to join cycle-b2-ood-extension categories despite their known seed_family collision "
        "with train/validation/calibration (see build()'s docstring) -- expected to fail closed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build(
        base_root=args.base_root,
        control_root=args.control_root,
        scout_root=args.scout_root,
        strategist_root=args.strategist_root,
        ood_extension_root=args.ood_extension_root,
        output_root=args.output_root,
        include_ood_extension=args.include_ood_extension,
    )
    print(json.dumps(report["totals"], indent=2, sort_keys=True))
    print(json.dumps(report["requirement_status"], indent=2, sort_keys=True))
    return 0 if all(report["requirement_status"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
