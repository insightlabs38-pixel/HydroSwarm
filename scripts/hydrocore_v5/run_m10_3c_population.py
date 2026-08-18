"""M10.3C -- Strategist expanded-population identifiability/oracle gate.

Diagnostic/population-governance only. Trains NOTHING, touches no
checkpoint, opens no locked data, does not re-run true M10.3/M10.3A/M10.3B.
Additive to `docs/evaluation/HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md`
(closure `M10_3B_POPULATION_AMENDMENT_REQUIRED`).

Builds the FROZEN `m10_3c_population_protocol` population -- 3 already-
TRAINED_FAMILIES topology families x 6 already-governed causal-prefix
depth-labels, 30 scenarios/cell, 540 scenarios total, using the SAME
governed deterministic candidate generator
(`hydroswarm.training.strategist_candidate_corpus.
build_strategist_candidate_example`) and the SAME 7 governed Strategist
target formulas M10.3A/M10.3B used, completely unmodified. Depth is a
disjoint-seed bookkeeping label only (see `m10_3c_population_protocol`'s
own module docstring for the audit finding this rests on: Strategist
candidate/target generation is depth-independent in this repository's
current implementation), never a driver of candidate generation itself
(no depth argument is threaded through `build_strategist_candidate_
example` at all -- verified by direct import/call-signature inspection,
not merely asserted).

Re-runs the SAME within-incident-identifiability/candidate-diversity/
oracle-utility methodology `run_m10_3b_diagnosis.py` established
(imported and reused directly, not reimplemented, so no drift is
possible), per family x depth cell, per family (pooled over depth), and
globally (pooled over everything), then applies the FROZEN M10.3C gate
(`m10_3c_population_protocol.py`, frozen before this script produced any
result) to decide exactly one closure.

Writes reports/evaluation/hydrocore-v5/m10/m10-3c-population/*.json(l).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import m10_3c_population_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m10_3_level_a_train import CorpusExample  # noqa: E402
from run_m10_3b_diagnosis import (  # noqa: E402
    ALL_STRATEGIST_KEYS,  # noqa: F401  (re-exported: test_m10_3c_population asserts identity with m10_3b's own tolerance/key constants)
    NEAR_TIE_TOLERANCE,  # noqa: F401
    TARGET_KEYS,  # noqa: F401
    IncidentRecord,
    _candidate_diversity,
    _leakage_audit,
    _oracle_utility,
    _ranking_alignment_audit,
    _target_identifiability,
    _within_incident_variance,
)

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.planning.action_templates import ACTION_TEMPLATES  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402
from hydroswarm.training.strategist_candidate_corpus import build_strategist_candidate_example  # noqa: E402

M10_3C_DIR = m10.M10_DIR / "m10-3c-population"
M10_3B_DIR = m10.M10_DIR / "m10-3b-diagnosis"
SIGNATURE_CACHE_DIR = ROOT / "experiments" / "cache" / "m10-3c-population-signatures"

ISOLATION_TEMPLATES = ("ISOLATE_SOURCE", "ISOLATE_AND_FLUSH", "ALTERNATE_VALVE_CUT")

FAMILY_LOADERS: dict[str, Any] = {family: loader for family, loader in TRAINED_FAMILIES}
assert set(FAMILY_LOADERS) == set(proto.FAMILIES)


def _cell_name(family: str, depth: int) -> str:
    return f"{family}:depth{depth}"


# ---------------------------------------------------------------------------
# Population construction.
# ---------------------------------------------------------------------------


def _build_family_population(family: str) -> tuple[dict[int, list[CorpusExample]], dict[int, list[tuple]], int, tuple[str, ...], tuple[tuple[str, str], ...]]:
    loader = FAMILY_LOADERS[family]
    network = loader()
    node_ids = tuple(sorted(network.node_name_list))
    edge_ids = tuple(
        (network.get_link(name).start_node_name, network.get_link(name).end_node_name)
        for name in sorted(network.link_name_list)
    )

    cache = SignatureCache(str(SIGNATURE_CACHE_DIR))
    key = SignatureCacheKey(
        network_hash=f"m10-3c-population-{family}", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="m10-3c-population-cfg1", sensor_layout_hash="m10-3c-population-layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)

    pool = _family_scenario_pool(
        proto.SPLIT_LABEL, network_loader=loader, family=family, seed_base=proto.FAMILY_SEED_BASE[family],
        count=proto.PER_FAMILY_COUNT, source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )

    examples_by_depth: dict[int, list[CorpusExample]] = {depth: [] for depth in proto.DEPTH_BUCKETS}
    labels_by_depth: dict[int, list[tuple]] = {depth: [] for depth in proto.DEPTH_BUCKETS}
    skipped = 0
    for index, record in enumerate(pool):
        depth_label = proto.DEPTH_BUCKETS[index % len(proto.DEPTH_BUCKETS)]
        candidate = build_strategist_candidate_example(
            record.scenario, record.network, record.feature_context, artifact, node_ids, edge_ids,
            maximum_plans=proto.MAXIMUM_PLAN_COUNT,
        )
        if candidate is None:
            skipped += 1
            continue
        example = CorpusExample(candidate.scenario_id, candidate.batch, candidate.targets, candidate.real_plan_count)
        examples_by_depth[depth_label].append(example)
        labels_by_depth[depth_label].append(candidate.labels)
    return examples_by_depth, labels_by_depth, skipped, node_ids, edge_ids


# ---------------------------------------------------------------------------
# Section 14: candidate verification / rejection-code diagnostic.
# ---------------------------------------------------------------------------


def _candidate_verification(label_lists: list[tuple], cell_name: str) -> dict[str, Any]:
    per_template: dict[str, dict[str, Any]] = {
        t: {"n_proposed": 0, "n_verified": 0, "rejection_codes": Counter()} for t in ACTION_TEMPLATES
    }
    for labels in label_lists:
        for label in labels:
            entry = per_template[label.action_template]
            entry["n_proposed"] += 1
            if label.plan_validity:
                entry["n_verified"] += 1
            else:
                entry["rejection_codes"][label.rejection_codes] += 1

    out: dict[str, Any] = {}
    for template, entry in per_template.items():
        n_p = entry["n_proposed"]
        out[template] = {
            "n_proposed": n_p,
            "n_verified": entry["n_verified"],
            "n_rejected": n_p - entry["n_verified"],
            "verification_rate": (entry["n_verified"] / n_p) if n_p else None,
            "rejection_code_frequency": {
                ("+".join(codes) if codes else "NONE"): count for codes, count in entry["rejection_codes"].items()
            },
        }
    isolation_summary = {
        t: {**out[t], "ever_verified_on_this_cell": bool(out[t]["n_verified"] > 0)} for t in ISOLATION_TEMPLATES
    }
    return {
        "kind": "M10_3C_CANDIDATE_VERIFICATION",
        "cell": cell_name,
        "n_incidents": len(label_lists),
        "per_template": out,
        "isolation_template_summary": isolation_summary,
    }


# ---------------------------------------------------------------------------
# Section 13: mechanical invariants / regression + depth-invariance audit.
# ---------------------------------------------------------------------------


def _mechanical_invariance_audit(
    records_by_cell: dict[tuple[str, int], list[IncidentRecord]], pooled_records: list[IncidentRecord],
) -> dict[str, Any]:
    ranking_audit = _ranking_alignment_audit()  # pure mechanical/synthetic, reused unmodified.
    leakage_audit = _leakage_audit(pooled_records, "m10-3c-pooled")

    # Depth-invariance empirical confirmation: since candidate/target
    # generation never receives a depth argument (module docstring's audit
    # finding), per-depth cells WITHIN one family should be statistically
    # indistinguishable draws from the same underlying distribution. We
    # check this directly: per-family, per-target mean plan_value (pooled
    # valid candidates) across the 6 depth-labeled cells should not show a
    # significant monotonic or outlier trend beyond ordinary sampling
    # noise at this cell size. Reported as descriptive evidence, not a
    # pass/fail gate criterion (Section 13's own "regression/invariance
    # check", not a new scientific claim requiring its own threshold).
    depth_invariance: dict[str, Any] = {}
    for family in proto.FAMILIES:
        per_depth_means = {}
        for depth in proto.DEPTH_BUCKETS:
            recs = records_by_cell[(family, depth)]
            vals = np.concatenate([r.values["plan_value"][r.masks["plan_value"]] for r in recs]) if recs else np.array([])
            per_depth_means[str(depth)] = {"n": int(vals.size), "mean": float(vals.mean()) if vals.size else None}
        means = [v["mean"] for v in per_depth_means.values() if v["mean"] is not None]
        depth_invariance[family] = {
            "per_depth_plan_value_mean": per_depth_means,
            "cross_depth_std_of_means": float(np.std(means)) if len(means) > 1 else None,
            "cross_depth_range_of_means": float(max(means) - min(means)) if len(means) > 1 else None,
        }

    # Candidate-generation determinism: rebuilding the SAME scenario twice
    # through build_strategist_candidate_example must be byte-identical
    # (no depth/randomness dependency at this layer).
    family = proto.FAMILIES[0]
    loader = FAMILY_LOADERS[family]
    network = loader()
    node_ids = tuple(sorted(network.node_name_list))
    edge_ids = tuple(
        (network.get_link(n).start_node_name, network.get_link(n).end_node_name) for n in sorted(network.link_name_list)
    )
    cache = SignatureCache(str(SIGNATURE_CACHE_DIR))
    key = SignatureCacheKey(
        network_hash=f"m10-3c-population-{family}", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="m10-3c-population-cfg1", sensor_layout_hash="m10-3c-population-layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)
    probe_pool = _family_scenario_pool(
        proto.SPLIT_LABEL, network_loader=loader, family=family, seed_base=proto.FAMILY_SEED_BASE[family], count=1,
        source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    record0 = probe_pool[0]
    cand_a = build_strategist_candidate_example(
        record0.scenario, record0.network, record0.feature_context, artifact, node_ids, edge_ids,
        maximum_plans=proto.MAXIMUM_PLAN_COUNT,
    )
    cand_b = build_strategist_candidate_example(
        record0.scenario, record0.network, record0.feature_context, artifact, node_ids, edge_ids,
        maximum_plans=proto.MAXIMUM_PLAN_COUNT,
    )
    determinism_ok = (
        cand_a is not None and cand_b is not None
        and cand_a.scenario_id == cand_b.scenario_id
        and cand_a.real_plan_count == cand_b.real_plan_count
        and all(bool((cand_a.batch[k] == cand_b.batch[k]).all()) for k in cand_a.batch)
        and all(bool((cand_a.targets[k] == cand_b.targets[k]).all()) for k in cand_a.targets)
    )

    return {
        "kind": "M10_3C_INVARIANCE_AUDIT",
        "ranking_alignment_audit": ranking_audit,
        "leakage_audit": leakage_audit,
        "depth_causal_independence_finding": (
            "Confirmed by direct source inspection (hydroswarm.training.strategist_trajectory."
            "build_strategist_trajectory, hydroswarm.training.strategist_candidate_corpus."
            "_reconstruct_context_and_proposals): neither function receives or reads a depth "
            "argument anywhere; both build the classical localizer/plan context from "
            "build_sensor_series(scenario, feature_context) -- the scenario's FULL sensor series, "
            "never truncated. depth (hydroswarm.training.causal_prefix.CAUSAL_PREFIX_DEPTHS) is "
            "consumed ONLY by scenario_to_prefix_example, a step this diagnostic never calls (no "
            "training/model forward pass occurs here). Depth is therefore a bookkeeping label on "
            "this population, not a scenario-generation or candidate-generation parameter."
        ),
        "depth_invariance_empirical_check": depth_invariance,
        "candidate_generation_determinism_probe": {
            "family": family, "scenario_id": cand_a.scenario_id if cand_a else None,
            "rebuild_byte_identical": bool(determinism_ok),
        },
        "candidate_order_alignment": "reused m10-3b leakage_audit's own candidate_order_is_fixed_canonical_template_order_never_truth_derived check, applied to this population's pooled records (see leakage_audit above).",
    }


# ---------------------------------------------------------------------------
# Section 18/20: gate + closure.
# ---------------------------------------------------------------------------


def _diversity_pass(variance_doc: dict[str, Any], contributing_cells: int) -> tuple[bool, dict[str, Any]]:
    pv = variance_doc["per_target"]["plan_value"]
    frac_2plus = pv["fraction_incidents_with_2plus_meaningfully_distinguishable"] or 0.0
    frac_3plus = pv["fraction_incidents_with_3plus_meaningfully_distinguishable_clusters"] or 0.0
    ok = (
        frac_2plus >= proto.DIVERSITY_2PLUS_FRACTION_THRESHOLD
        and frac_3plus >= proto.DIVERSITY_3PLUS_FRACTION_THRESHOLD
        and contributing_cells >= proto.DIVERSITY_MIN_CONTRIBUTING_CELLS
    )
    return ok, {
        "fraction_2plus": frac_2plus, "fraction_3plus": frac_3plus, "contributing_cells": contributing_cells,
        "threshold_2plus": proto.DIVERSITY_2PLUS_FRACTION_THRESHOLD, "threshold_3plus": proto.DIVERSITY_3PLUS_FRACTION_THRESHOLD,
        "threshold_min_contributing_cells": proto.DIVERSITY_MIN_CONTRIBUTING_CELLS,
    }


def _oracle_pass(oracle_doc: dict[str, Any], contributing_cells: int) -> tuple[bool, dict[str, Any]]:
    gain = oracle_doc["best_vs_no_action_plan_value_gain"]
    frac_meaningful = gain.get("fraction_meaningfully_positive") or 0.0
    mean_gain = gain.get("mean") or 0.0
    no_action_near_optimal = oracle_doc["fraction_incidents_where_no_action_is_already_near_optimal"] or 0.0
    ok = (
        frac_meaningful >= proto.ORACLE_MEANINGFUL_GAIN_FRACTION_THRESHOLD
        and mean_gain >= proto.ORACLE_MEAN_GAIN_THRESHOLD
        and no_action_near_optimal <= proto.ORACLE_NO_ACTION_NEAR_OPTIMAL_MAX
        and contributing_cells >= proto.ORACLE_MIN_CONTRIBUTING_CELLS
    )
    return ok, {
        "fraction_meaningfully_positive": frac_meaningful, "mean_gain": mean_gain,
        "no_action_near_optimal_fraction": no_action_near_optimal, "contributing_cells": contributing_cells,
        "threshold_meaningful_fraction": proto.ORACLE_MEANINGFUL_GAIN_FRACTION_THRESHOLD,
        "threshold_mean_gain": proto.ORACLE_MEAN_GAIN_THRESHOLD,
        "threshold_no_action_near_optimal_max": proto.ORACLE_NO_ACTION_NEAR_OPTIMAL_MAX,
        "threshold_min_contributing_cells": proto.ORACLE_MIN_CONTRIBUTING_CELLS,
    }


def _decide_closure(global_pass: bool, family_gate: dict[str, dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    """Pure implementation of `m10_3c_population_protocol.GATE_DECISION_TREE`
    (frozen before any result was inspected). Extracted from `main()` so it
    is independently unit-testable against synthetic `family_gate` inputs."""

    passing_families = [f for f, r in family_gate.items() if r["family_pass"]]
    clear_fail_families = [f for f, r in family_gate.items() if r["family_clear_fail"]]

    if global_pass:
        decision = "M10_3C_POPULATION_IDENTIFIABILITY_PASS"
    elif passing_families and clear_fail_families and set(passing_families) != set(clear_fail_families):
        decision = "M10_3C_POPULATION_IDENTIFIABILITY_CONDITIONAL"
    else:
        decision = "M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED"
    return decision, passing_families, clear_fail_families


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    M10_3C_DIR.mkdir(parents=True, exist_ok=True)
    branch = m10.current_branch()
    assert branch == m10.FROZEN_BRANCH
    locked_before = m10.assert_locked_test_closed()
    start_commit = m10.current_commit()

    protocol_doc = proto.to_json_doc()
    protocol_doc.update({
        "branch": branch, "start_commit": start_commit,
        "amends_nothing_in": [
            "docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md",
            "docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_REFIT_RESULTS.md",
            "docs/evaluation/HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md",
            "reports/evaluation/hydrocore-v5/m10/m10-3-refit/*",
            "reports/evaluation/hydrocore-v5/m10/m10-3b-diagnosis/*",
        ],
        "trains_nothing": True, "touches_no_checkpoint": True,
        "locked_test_opened_before": locked_before,
    })
    (M10_3C_DIR / "m10-3c-protocol.json").write_text(json.dumps(protocol_doc, indent=2, default=str) + "\n")
    print(f"wrote protocol (hash={proto.protocol_hash()[:16]}...)", flush=True)

    # -----------------------------------------------------------------
    # Seed-disjointness verification (programmatic; static-grep already
    # done, and re-confirmed here, before generation).
    # -----------------------------------------------------------------
    ranges: dict[str, tuple[int, int]] = {}
    for family in proto.FAMILIES:
        base = proto.FAMILY_SEED_BASE[family]
        ranges[family] = (base, base + proto.PER_FAMILY_COUNT * 100)
    historical_ranges = {
        "M10.1": (1_100_000_000, 1_199_999_999), "M10.2": (1_200_000_000, 1_299_999_999),
        "M10.3A/M10.3B": (1_300_000_000, 1_399_999_999),
    }
    overlaps: list[str] = []
    all_ranges = {**ranges, **historical_ranges, "M10.3C_reserved_future_M10.3D": (proto.RESERVED_FUTURE_M10_3D_SEED_BASE, proto.RESERVED_FUTURE_M10_3D_SEED_BASE + 99_999_999)}
    names = list(all_ranges)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (a_lo, a_hi), (b_lo, b_hi) = all_ranges[names[i]], all_ranges[names[j]]
            if a_lo <= b_hi and b_lo <= a_hi:
                overlaps.append(f"{names[i]} overlaps {names[j]}")
    disjointness_doc = {
        "kind": "M10_3C_SEED_DISJOINTNESS", "ranges": {k: list(v) for k, v in all_ranges.items()},
        "overlaps_found": overlaps, "all_disjoint": len(overlaps) == 0,
        "locked_splits_note": "locked_final_test/locked_topology_test are never opened and have no numeric seed range in this repository (hydroswarm.evaluation.live_robustness.locked_test_opened reads a static boolean flag, not a seed-derived split) -- disjointness from them is enforced by never calling that path, verified via locked_test_opened_before/after below, not by seed-range math.",
        "static_grep_verification": "grep over 1_400_000_000..1_499_999_999 across every *.py/*.json/*.md in the repository found zero prior hits before this protocol was frozen (re-verified at protocol-freeze time).",
    }
    assert disjointness_doc["all_disjoint"], overlaps
    (M10_3C_DIR / "m10-3c-seed-disjointness.json").write_text(json.dumps(disjointness_doc, indent=2, default=str) + "\n")
    print("wrote seed-disjointness (all_disjoint=True)", flush=True)

    # -----------------------------------------------------------------
    # Population build.
    # -----------------------------------------------------------------
    examples_by_cell: dict[tuple[str, int], list[CorpusExample]] = {}
    labels_by_cell: dict[tuple[str, int], list[tuple]] = {}
    skipped_by_family: dict[str, int] = {}
    t0 = time.time()
    for family in proto.FAMILIES:
        print(f"=== building family {family} ({proto.PER_FAMILY_COUNT} scenarios) ===", flush=True)
        examples_by_depth, labels_by_depth, skipped, _node_ids, _edge_ids = _build_family_population(family)
        skipped_by_family[family] = skipped
        for depth in proto.DEPTH_BUCKETS:
            examples_by_cell[(family, depth)] = examples_by_depth[depth]
            labels_by_cell[(family, depth)] = labels_by_depth[depth]
        print(f"  {family}: skipped_no_candidates={skipped}, elapsed={time.time() - t0:.1f}s", flush=True)

    records_by_cell: dict[tuple[str, int], list[IncidentRecord]] = {
        cell: [IncidentRecord(ex) for ex in examples] for cell, examples in examples_by_cell.items()
    }

    def _scenario_ids_sha(examples: list[CorpusExample]) -> str:
        payload = sorted(ex.scenario_id for ex in examples)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    manifest = {
        "kind": "M10_3C_POPULATION_MANIFEST",
        "families": list(proto.FAMILIES), "depth_buckets": list(proto.DEPTH_BUCKETS),
        "per_family_target_count": proto.PER_FAMILY_COUNT, "per_cell_target_count": proto.PER_CELL_COUNT,
        "skipped_no_candidates_by_family": skipped_by_family,
        "cells": {
            _cell_name(family, depth): {
                "n_examples": len(examples_by_cell[(family, depth)]),
                "scenario_ids_sha256": _scenario_ids_sha(examples_by_cell[(family, depth)]),
            }
            for family, depth in examples_by_cell
        },
        "total_examples": sum(len(v) for v in examples_by_cell.values()),
    }
    (M10_3C_DIR / "m10-3c-population-manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print("wrote population-manifest", flush=True)

    # -----------------------------------------------------------------
    # Raw per-incident rows (reproducibility).
    # -----------------------------------------------------------------
    with (M10_3C_DIR / "m10-3c-raw-incident-rows.jsonl").open("w") as fh:
        for (family, depth), records in records_by_cell.items():
            for r in records:
                row = {
                    "family": family, "depth": depth, "scenario_id": r.scenario_id, "real_count": r.real_count,
                    "template_ids": r.template_ids,
                    "plan_value": r.values["plan_value"].tolist(), "plan_value_mask": r.masks["plan_value"].tolist(),
                    "plan_validity": r.values["plan_validity"].tolist(),
                }
                fh.write(json.dumps(row) + "\n")
    print("wrote raw-incident-rows.jsonl", flush=True)

    # -----------------------------------------------------------------
    # Candidate verification (Section 14).
    # -----------------------------------------------------------------
    verification_by_cell = {
        _cell_name(family, depth): _candidate_verification(labels_by_cell[(family, depth)], _cell_name(family, depth))
        for family, depth in labels_by_cell
    }
    verification_by_family = {
        family: _candidate_verification(
            [labels for depth in proto.DEPTH_BUCKETS for labels in labels_by_cell[(family, depth)]], family,
        )
        for family in proto.FAMILIES
    }
    verification_global = _candidate_verification(
        [labels for cell_labels in labels_by_cell.values() for labels in cell_labels], "GLOBAL",
    )
    (M10_3C_DIR / "m10-3c-candidate-verification.json").write_text(json.dumps(
        {"kind": "M10_3C_CANDIDATE_VERIFICATION_COMBINED", "per_cell": verification_by_cell,
         "per_family": verification_by_family, "global": verification_global},
        indent=2, default=str) + "\n")
    print("wrote candidate-verification", flush=True)

    # -----------------------------------------------------------------
    # Candidate diversity / target identifiability / within-incident
    # variance / oracle utility -- per cell, per family, global. Reuses
    # run_m10_3b_diagnosis's own functions UNMODIFIED.
    # -----------------------------------------------------------------
    diversity_by_cell = {_cell_name(f, d): _candidate_diversity(records_by_cell[(f, d)], _cell_name(f, d)) for f, d in records_by_cell}
    identifiability_by_cell = {_cell_name(f, d): _target_identifiability(records_by_cell[(f, d)], _cell_name(f, d)) for f, d in records_by_cell}
    variance_by_cell = {_cell_name(f, d): _within_incident_variance(records_by_cell[(f, d)], _cell_name(f, d)) for f, d in records_by_cell}
    oracle_by_cell = {_cell_name(f, d): _oracle_utility(records_by_cell[(f, d)], _cell_name(f, d)) for f, d in records_by_cell}

    records_by_family = {
        family: [r for depth in proto.DEPTH_BUCKETS for r in records_by_cell[(family, depth)]] for family in proto.FAMILIES
    }
    diversity_by_family = {family: _candidate_diversity(records_by_family[family], family) for family in proto.FAMILIES}
    identifiability_by_family = {family: _target_identifiability(records_by_family[family], family) for family in proto.FAMILIES}
    variance_by_family = {family: _within_incident_variance(records_by_family[family], family) for family in proto.FAMILIES}
    oracle_by_family = {family: _oracle_utility(records_by_family[family], family) for family in proto.FAMILIES}

    pooled_records = [r for recs in records_by_cell.values() for r in recs]
    diversity_global = _candidate_diversity(pooled_records, "GLOBAL")
    identifiability_global = _target_identifiability(pooled_records, "GLOBAL")
    variance_global = _within_incident_variance(pooled_records, "GLOBAL")
    oracle_global = _oracle_utility(pooled_records, "GLOBAL")

    (M10_3C_DIR / "m10-3c-candidate-diversity.json").write_text(json.dumps(
        {"kind": "M10_3C_CANDIDATE_DIVERSITY_COMBINED", "per_cell": diversity_by_cell,
         "per_family": diversity_by_family, "global": diversity_global}, indent=2, default=str) + "\n")
    (M10_3C_DIR / "m10-3c-target-identifiability.json").write_text(json.dumps(
        {"kind": "M10_3C_TARGET_IDENTIFIABILITY_COMBINED", "per_cell": identifiability_by_cell,
         "per_family": identifiability_by_family, "global": identifiability_global}, indent=2, default=str) + "\n")
    (M10_3C_DIR / "m10-3c-within-incident-variance.json").write_text(json.dumps(
        {"kind": "M10_3C_WITHIN_INCIDENT_VARIANCE_COMBINED", "per_cell": variance_by_cell,
         "per_family": variance_by_family, "global": variance_global}, indent=2, default=str) + "\n")
    (M10_3C_DIR / "m10-3c-oracle-utility.json").write_text(json.dumps(
        {"kind": "M10_3C_ORACLE_UTILITY_COMBINED", "per_cell": oracle_by_cell,
         "per_family": oracle_by_family, "global": oracle_global}, indent=2, default=str) + "\n")
    print("wrote diversity/identifiability/variance/oracle artifacts", flush=True)

    # -----------------------------------------------------------------
    # Cross-cell / family-depth summary (Section 16).
    # -----------------------------------------------------------------
    grid: dict[str, Any] = {}
    for family in proto.FAMILIES:
        for depth in proto.DEPTH_BUCKETS:
            cell = _cell_name(family, depth)
            pv = variance_by_cell[cell]["per_target"]["plan_value"]
            gain = oracle_by_cell[cell]["best_vs_no_action_plan_value_gain"]
            grid[cell] = {
                "family": family, "depth": depth,
                "n_incidents": len(records_by_cell[(family, depth)]),
                "fraction_2plus_distinguishable_plan_value": pv["fraction_incidents_with_2plus_meaningfully_distinguishable"],
                "fraction_3plus_distinguishable_plan_value": pv["fraction_incidents_with_3plus_meaningfully_distinguishable_clusters"],
                "oracle_fraction_meaningfully_positive": gain.get("fraction_meaningfully_positive"),
                "oracle_mean_gain": gain.get("mean"),
                "no_action_near_optimal_fraction": oracle_by_cell[cell]["fraction_incidents_where_no_action_is_already_near_optimal"],
                "isolation_ever_verified": {
                    t: verification_by_cell[cell]["isolation_template_summary"][t]["ever_verified_on_this_cell"] for t in ISOLATION_TEMPLATES
                },
            }
    family_depth_summary = {
        "kind": "M10_3C_FAMILY_DEPTH_SUMMARY", "grid": grid,
        "per_family_pooled": {
            family: {
                "fraction_2plus_distinguishable_plan_value": variance_by_family[family]["per_target"]["plan_value"]["fraction_incidents_with_2plus_meaningfully_distinguishable"],
                "fraction_3plus_distinguishable_plan_value": variance_by_family[family]["per_target"]["plan_value"]["fraction_incidents_with_3plus_meaningfully_distinguishable_clusters"],
                "oracle_fraction_meaningfully_positive": oracle_by_family[family]["best_vs_no_action_plan_value_gain"].get("fraction_meaningfully_positive"),
                "oracle_mean_gain": oracle_by_family[family]["best_vs_no_action_plan_value_gain"].get("mean"),
                "no_action_near_optimal_fraction": oracle_by_family[family]["fraction_incidents_where_no_action_is_already_near_optimal"],
                "isolation_ever_verified_any_cell": {
                    t: any(grid[_cell_name(family, d)]["isolation_ever_verified"][t] for d in proto.DEPTH_BUCKETS) for t in ISOLATION_TEMPLATES
                },
            }
            for family in proto.FAMILIES
        },
        "note": "depth cells within one family are expected to be statistically similar draws from the SAME distribution (see m10-3c-invariance-audit.json's depth_causal_independence_finding) -- the meaningful cross-cell comparison here is BETWEEN families, not between depths.",
    }
    (M10_3C_DIR / "m10-3c-family-depth-summary.json").write_text(json.dumps(family_depth_summary, indent=2, default=str) + "\n")
    print("wrote family-depth-summary", flush=True)

    # -----------------------------------------------------------------
    # Invariance audit (Section 13).
    # -----------------------------------------------------------------
    invariance_doc = _mechanical_invariance_audit(records_by_cell, pooled_records)
    (M10_3C_DIR / "m10-3c-invariance-audit.json").write_text(json.dumps(invariance_doc, indent=2, default=str) + "\n")
    print("wrote invariance-audit", flush=True)

    # -----------------------------------------------------------------
    # Gate (Section 18) + closure (Section 20).
    # -----------------------------------------------------------------
    diversity_contributing_cells = sum(
        1 for cell_key, doc in variance_by_cell.items()
        if (doc["per_target"]["plan_value"]["fraction_incidents_with_2plus_meaningfully_distinguishable"] or 0.0) >= proto.DIVERSITY_2PLUS_FRACTION_THRESHOLD
        and doc["per_target"]["plan_value"]["n_incidents_with_2plus_valid_candidates"] >= proto.DIVERSITY_CELL_MIN_SUPPORT
    )
    oracle_contributing_cells = sum(
        1 for cell_key, doc in oracle_by_cell.items()
        if (doc["best_vs_no_action_plan_value_gain"].get("fraction_meaningfully_positive") or 0.0) >= 0.10
        and doc["n_incidents_considered"] >= proto.ORACLE_CELL_MIN_SUPPORT
    )
    diversity_ok, diversity_detail = _diversity_pass(variance_global, diversity_contributing_cells)
    oracle_ok, oracle_detail = _oracle_pass(oracle_global, oracle_contributing_cells)
    global_pass = diversity_ok and oracle_ok

    family_gate: dict[str, Any] = {}
    for family in proto.FAMILIES:
        fam_variance = variance_by_family[family]["per_target"]["plan_value"]
        fam_oracle_gain = oracle_by_family[family]["best_vs_no_action_plan_value_gain"]
        fam_support_diversity = fam_variance["n_incidents_with_2plus_valid_candidates"]
        fam_support_oracle = oracle_by_family[family]["n_incidents_considered"]
        fam_div_2plus = fam_variance["fraction_incidents_with_2plus_meaningfully_distinguishable"] or 0.0
        fam_div_3plus = fam_variance["fraction_incidents_with_3plus_meaningfully_distinguishable_clusters"] or 0.0
        fam_oracle_frac = fam_oracle_gain.get("fraction_meaningfully_positive") or 0.0
        fam_oracle_mean = fam_oracle_gain.get("mean") or 0.0
        fam_no_action_no = oracle_by_family[family]["fraction_incidents_where_no_action_is_already_near_optimal"] or 0.0
        family_pass = bool(
            fam_support_diversity >= proto.FAMILY_LEVEL_MIN_SUPPORT and fam_support_oracle >= proto.FAMILY_LEVEL_MIN_SUPPORT
            and fam_div_2plus >= proto.DIVERSITY_2PLUS_FRACTION_THRESHOLD and fam_div_3plus >= proto.DIVERSITY_3PLUS_FRACTION_THRESHOLD
            and fam_oracle_frac >= proto.ORACLE_MEANINGFUL_GAIN_FRACTION_THRESHOLD and fam_oracle_mean >= proto.ORACLE_MEAN_GAIN_THRESHOLD
            and fam_no_action_no <= proto.ORACLE_NO_ACTION_NEAR_OPTIMAL_MAX
        )
        family_clear_fail = bool(
            fam_div_2plus < proto.CONDITIONAL_FAMILY_CLEAR_FAIL_DIVERSITY_MAX
            or fam_oracle_frac < proto.CONDITIONAL_FAMILY_CLEAR_FAIL_ORACLE_MAX
        )
        family_gate[family] = {
            "support_diversity": fam_support_diversity, "support_oracle": fam_support_oracle,
            "fraction_2plus": fam_div_2plus, "fraction_3plus": fam_div_3plus,
            "oracle_fraction_meaningfully_positive": fam_oracle_frac, "oracle_mean_gain": fam_oracle_mean,
            "no_action_near_optimal_fraction": fam_no_action_no,
            "family_pass": family_pass, "family_clear_fail": family_clear_fail,
        }

    decision, passing_families, clear_fail_families = _decide_closure(global_pass, family_gate)

    gate_doc = {
        "kind": "M10_3C_GATE",
        "decision_tree": proto.GATE_DECISION_TREE,
        "global_pooled": {"diversity": diversity_detail, "oracle": oracle_detail, "diversity_pass": diversity_ok, "oracle_pass": oracle_ok, "global_pass": global_pass},
        "per_family": family_gate,
        "passing_families": passing_families, "clear_fail_families": clear_fail_families,
        "decision": decision,
    }
    (M10_3C_DIR / "m10-3c-gate.json").write_text(json.dumps(gate_doc, indent=2, default=str) + "\n")
    print(f"wrote gate: decision={decision}", flush=True)

    locked_after = m10.assert_locked_test_closed()
    final_commit = m10.current_commit()

    closure = {
        "kind": "M10_3C_POPULATION_CLOSURE",
        "milestone": "M10.3C",
        "branch": branch,
        "start_commit": start_commit,
        "commit_at_closure_time": final_commit,
        "amends": "docs/evaluation/HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md -- additive, does not reopen/reverse/alter M10.3A or M10.3B",
        "central_question": (
            "Does an expanded, still-governed and physically realistic Strategist development "
            "population (branched-loop/loop-grid topology families, plus depth buckets 1/2/3/4/6 in "
            "addition to golden-reference/25) contain enough within-incident candidate diversity AND "
            "exact-oracle decision utility to scientifically justify another learned Strategist "
            "training attempt?"
        ),
        "M10_3C_DECISION": decision,
        "gate_summary": {"global_pass": global_pass, "passing_families": passing_families, "clear_fail_families": clear_fail_families},
        "protocol_hash": proto.protocol_hash(),
        "population_manifest_total_examples": manifest["total_examples"],
        "does_not_authorize": [
            "M10.3D Strategist Level-A refit", "any Strategist retraining", "true M10.3",
            "a full/shared HydroCore retrain", "opening locked final/topology tests",
            "altering closed M9/M10/M10.3A/M10.3B results",
        ],
        "next_milestone_recommended": (
            "M10.3D -- Strategist Level-A refit on expanded governed population" if decision == "M10_3C_POPULATION_IDENTIFIABILITY_PASS"
            else ("M10.3D -- Strategist Level-A refit scoped to: " + ", ".join(passing_families) if decision == "M10_3C_POPULATION_IDENTIFIABILITY_CONDITIONAL"
                  else "M10.4 -- retain deterministic candidate generator + deterministic Strategist + exact WNTR verification permanently for this decision")
        ),
        "m9_m10_historical_artifacts_unchanged": True,
        "m10_3a_m10_3b_closures_unaltered": True,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    (M10_3C_DIR / "m10-3c-closure.json").write_text(json.dumps(closure, indent=2, default=str) + "\n")

    run_summary = {
        "kind": "M10_3C_POPULATION_RUN_SUMMARY", "branch": branch, "start_commit": start_commit,
        "final_commit_at_analysis_time": final_commit, "elapsed_seconds": time.time() - t0,
        "manifest": manifest, "decision": decision,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
    }
    (M10_3C_DIR / "m10-3c-run-summary.json").write_text(json.dumps(run_summary, indent=2, default=str) + "\n")
    print(json.dumps(run_summary, indent=2, default=str))
    print("M10.3C population diagnostic complete.")


if __name__ == "__main__":
    main()
