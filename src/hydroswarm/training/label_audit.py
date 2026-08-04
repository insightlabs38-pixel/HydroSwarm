"""Target/label quality audit and sanity baselines (overnight-plan.txt Task 0.5).

A target is not ready for training if this audit or its baseline behavior is
nonsensical. This module computes structural label-quality checks and a
handful of baselines that are directly computable from a corpus's existing
tensors (classical_prior, source_candidate_mask, source_node) without
invoking the simulator, so it can run cheaply as a pre-training gate.

Baselines it cannot compute from stored tensors alone -- nearest-positive-
sensor, classical expected-information-gain sampling, deterministic plan
template ranking -- require live signature/simulation state that is only
produced once the Scout/Strategist label generators (Task 2.3/2.4) exist;
this module reports them as "not_available" with a reason rather than
fabricating a value.

Locked-test-data discipline: structural checks (counts, schema, finite
values, cross-split leakage) may look at every split including the locked
test split, because leakage detection inherently requires seeing test-split
identifiers. Baseline *accuracies* are computed only for the caller-supplied
``decision_splits`` (train/validation/calibration by default) and are always
``None`` for any split outside that set, so nothing here can be used to tune
against locked-test performance.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .data import ScenarioExample

_DEFAULT_DECISION_SPLITS = ("train", "validation", "calibration")


def _finite_violations(examples: Sequence[ScenarioExample]) -> list[dict[str, str]]:
    violations = []
    for example in examples:
        for bucket_name, bucket in (("inputs", example.inputs), ("targets", example.targets)):
            for key, tensor in bucket.items():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    violations.append(
                        {"scenario_id": example.scenario_id, "bucket": bucket_name, "tensor": key}
                    )
    return violations


def _duplicate_scenario_ids(examples: Sequence[ScenarioExample]) -> list[str]:
    counts = Counter(example.scenario_id for example in examples)
    return sorted(scenario_id for scenario_id, count in counts.items() if count > 1)


def _near_duplicate_scenarios(examples: Sequence[ScenarioExample]) -> list[list[str]]:
    """Flag scenarios whose node_features + all target values hash identically.

    A cheap, conservative proxy for near-duplicate detection: two distinct
    scenario_ids with byte-identical governing state are very likely the same
    underlying incident generated twice (e.g. a seed-family bug), not two
    genuinely different incidents that coincidentally match.
    """

    groups: dict[str, list[str]] = {}
    for example in examples:
        digest = hashlib.sha256()
        if "node_features" in example.inputs:
            digest.update(example.inputs["node_features"].numpy().tobytes())
        for key in sorted(example.targets):
            digest.update(key.encode())
            digest.update(example.targets[key].numpy().tobytes())
        groups.setdefault(digest.hexdigest(), []).append(example.scenario_id)
    return sorted(
        (sorted(ids) for ids in groups.values() if len(ids) > 1),
        key=lambda ids: ids[0],
    )


def _impossible_labels(examples: Sequence[ScenarioExample]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for example in examples:
        source_node = example.targets.get("source_node")
        candidate_mask = example.inputs.get("source_candidate_mask")
        if source_node is not None and candidate_mask is not None:
            index = int(source_node)
            if index < 0 or index >= candidate_mask.numel():
                violations.append(
                    {
                        "scenario_id": example.scenario_id,
                        "reason": "source_node index out of range",
                        "index": index,
                        "candidate_count": int(candidate_mask.numel()),
                    }
                )
            elif candidate_mask[index] <= 0:
                violations.append(
                    {
                        "scenario_id": example.scenario_id,
                        "reason": "source_node points at a masked-out (infeasible) candidate",
                        "index": index,
                    }
                )
        sensor_fault = example.targets.get("sensor_fault")
        if sensor_fault is not None:
            values = sensor_fault.float()
            if bool(((values < 0) | (values > 1)).any()):
                violations.append(
                    {
                        "scenario_id": example.scenario_id,
                        "reason": "sensor_fault target contains values outside [0, 1]",
                    }
                )
        for key in ("duration", "relative_strength", "start_time"):
            value = example.targets.get(key)
            if value is not None and int(value) < 0:
                violations.append(
                    {
                        "scenario_id": example.scenario_id,
                        "reason": f"{key} target is negative",
                        "value": int(value),
                    }
                )
    return violations


def _class_histogram(examples: Sequence[ScenarioExample], target_key: str) -> dict[str, int] | None:
    present = [example for example in examples if target_key in example.targets]
    if not present:
        return None
    counts: Counter[int] = Counter()
    for example in present:
        counts[int(example.targets[target_key])] += 1
    return {str(value): count for value, count in sorted(counts.items())}


def _sensor_fault_prevalence(examples: Sequence[ScenarioExample]) -> dict[str, Any] | None:
    """Overall and per-network-node fault-positive rates.

    ``sensor_fault`` is node-indexed with a per-example node count, so a
    genuinely multi-topology corpus (different networks, different node
    counts within the same split) cannot be ``torch.stack``ed across
    examples the way a single-topology corpus like learning-v1 could --
    node index 2 also isn't the same physical node across two different
    topologies, so a single "per_node_positive_rate" list would be
    meaningless if it mixed them. ``overall_positive_rate`` is computed by
    concatenation (shape-agnostic); per-node rates are reported separately
    per network_id, where node identity is actually consistent.
    """

    present = [example for example in examples if "sensor_fault" in example.targets]
    if not present:
        return None
    all_values = torch.cat([example.targets["sensor_fault"].float().reshape(-1) for example in present])
    per_network: dict[str, list[float]] = {}
    grouped: dict[str, list[ScenarioExample]] = {}
    for example in present:
        grouped.setdefault(example.network_id, []).append(example)
    for network_id, group in grouped.items():
        shapes = {tuple(example.targets["sensor_fault"].shape) for example in group}
        if len(shapes) != 1:
            continue  # inconsistent node count even within one network_id; skip rather than guess
        stacked = torch.stack([example.targets["sensor_fault"].float() for example in group])
        per_network[network_id] = [float(value) for value in stacked.mean(dim=0)]
    return {
        "examples_with_target": len(present),
        "overall_positive_rate": float(all_values.mean()),
        "per_node_positive_rate_by_network": dict(sorted(per_network.items())),
    }


def _network_balance(examples: Sequence[ScenarioExample]) -> dict[str, int]:
    counts = Counter(example.network_id for example in examples)
    return dict(sorted(counts.items()))


def _source_balance_by_network(examples: Sequence[ScenarioExample]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[int]] = {}
    for example in examples:
        if "source_node" not in example.targets:
            continue
        grouped.setdefault(example.network_id, Counter())[int(example.targets["source_node"])] += 1
    return {
        network_id: {str(value): count for value, count in sorted(counter.items())}
        for network_id, counter in sorted(grouped.items())
    }


def _missingness(examples: Sequence[ScenarioExample], expected_target_keys: Sequence[str]) -> dict[str, int]:
    missing: Counter[str] = Counter()
    for example in examples:
        for key in expected_target_keys:
            if key not in example.targets:
                missing[key] += 1
    return {key: missing[key] for key in expected_target_keys}


def _baseline_accuracies(examples: Sequence[ScenarioExample]) -> dict[str, Any]:
    with_source = [example for example in examples if "source_node" in example.targets]
    if not with_source:
        return {"note": "no source_node targets present in this split"}

    random_terms = []
    for example in with_source:
        candidate_mask = example.inputs.get("source_candidate_mask")
        candidate_count = int(candidate_mask.sum()) if candidate_mask is not None else None
        if candidate_count:
            random_terms.append(1.0 / candidate_count)
    random_source_accuracy = sum(random_terms) / len(random_terms) if random_terms else None

    source_counts = Counter(int(example.targets["source_node"]) for example in with_source)
    majority_class, majority_count = source_counts.most_common(1)[0]
    majority_source_accuracy = majority_count / len(with_source)

    with_prior = [example for example in with_source if "classical_prior" in example.inputs]
    classical_prior_accuracy = None
    if with_prior:
        correct = sum(
            1
            for example in with_prior
            if int(torch.argmax(example.inputs["classical_prior"])) == int(example.targets["source_node"])
        )
        classical_prior_accuracy = correct / len(with_prior)

    return {
        "random_source": random_source_accuracy,
        "majority_source": {
            "accuracy": majority_source_accuracy,
            "majority_class": majority_class,
        },
        "classical_signature_prior": classical_prior_accuracy,
        "nearest_positive_sensor": {
            "value": None,
            "reason": "requires per-sensor concentration traces from live/cached signature "
            "artifacts, not present in the frozen canonical-tensor rows; add once "
            "Task 2.3 Scout label generation exposes sensor-level signatures.",
        },
        "classical_eig_sample": {
            "value": None,
            "reason": "requires Scout candidate enumeration (Task 2.3); not computable from "
            "stored Sentinel-only tensors.",
        },
        "deterministic_plan_template_ranking": {
            "value": None,
            "reason": "requires Strategist plan labels and WNTR verification (Task 2.4); not "
            "computable from stored Sentinel-only tensors.",
        },
    }


def audit_split(
    name: str,
    examples: Sequence[ScenarioExample],
    *,
    compute_baselines: bool,
    expected_target_keys: Sequence[str] = (
        "source_node",
        "duration",
        "relative_strength",
        "start_time",
        "sensor_fault",
    ),
) -> dict[str, Any]:
    if not examples:
        return {"split": name, "count": 0, "note": "no examples provided"}
    return {
        "split": name,
        "count": len(examples),
        "network_balance": _network_balance(examples),
        "source_balance_by_network": _source_balance_by_network(examples),
        "target_missingness": _missingness(examples, expected_target_keys),
        "target_class_histograms": {
            key: histogram
            for key in expected_target_keys
            if key != "sensor_fault"
            for histogram in (_class_histogram(examples, key),)
            if histogram is not None
        },
        "sensor_fault_prevalence": _sensor_fault_prevalence(examples),
        "duplicate_scenario_ids": _duplicate_scenario_ids(examples),
        "near_duplicate_groups": _near_duplicate_scenarios(examples),
        "impossible_labels": _impossible_labels(examples),
        "finite_value_violations": _finite_violations(examples),
        "sanity_baselines": (
            _baseline_accuracies(examples)
            if compute_baselines
            else {
                "note": (
                    f"split {name!r} is not in decision_splits; baseline accuracies are "
                    "intentionally omitted so this audit can never be used to tune against "
                    "locked-test performance."
                )
            }
        ),
    }


def cross_split_leakage(splits: Mapping[str, Sequence[ScenarioExample]]) -> dict[str, Any]:
    """Scenario-ID and seed-family leakage across every provided split.

    Unlike hydroswarm.training.data.validate_split_isolation (which raises
    on the first violation and is meant to be a hard training-time gate),
    this collects every violation so the audit report is complete.
    """

    scenario_owner: dict[str, str] = {}
    family_owner: dict[tuple[str, str], str] = {}
    scenario_leaks: list[dict[str, str]] = []
    family_leaks: list[dict[str, Any]] = []
    for split_name, examples in splits.items():
        for example in examples:
            if example.scenario_id in scenario_owner and scenario_owner[example.scenario_id] != split_name:
                scenario_leaks.append(
                    {
                        "scenario_id": example.scenario_id,
                        "splits": sorted({scenario_owner[example.scenario_id], split_name}),
                    }
                )
            else:
                scenario_owner.setdefault(example.scenario_id, split_name)

            family = (example.network_id, example.seed_family)
            if family in family_owner and family_owner[family] != split_name:
                family_leaks.append(
                    {
                        "network_id": family[0],
                        "seed_family": family[1],
                        "splits": sorted({family_owner[family], split_name}),
                    }
                )
            else:
                family_owner.setdefault(family, split_name)
    return {
        "scenario_id_leaks": scenario_leaks,
        "seed_family_leaks": family_leaks,
        "topology_leak_note": (
            "topology_hash does not exist yet (added in Task 1.1); until then, leakage across "
            "genuinely different topologies cannot be distinguished from expected network_id "
            "reuse across splits within a single topology's hydraulic-regime variants."
        ),
    }


def audit_corpus(
    splits: Mapping[str, Sequence[ScenarioExample]],
    *,
    decision_splits: Sequence[str] = _DEFAULT_DECISION_SPLITS,
) -> dict[str, Any]:
    return {
        "decision_splits": list(decision_splits),
        "splits": {
            name: audit_split(name, examples, compute_baselines=name in decision_splits)
            for name, examples in splits.items()
        },
        "cross_split_leakage": cross_split_leakage(splits),
    }
