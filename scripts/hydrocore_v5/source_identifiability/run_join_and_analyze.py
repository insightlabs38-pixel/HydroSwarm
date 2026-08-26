#!/usr/bin/env python3
"""Phase 5: join per-incident identifiability metrics to the frozen
M11.6 locked evaluation's own recorded HydroCore-v5 outcomes
(`m11-6-raw-incidents.jsonl`, read-only) and compute every "Required
comparison" the analysis protocol calls for. Writes one combined table and
one results JSON; never modifies the locked evaluation artifacts it reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_identifiability import common, stats_utils  # noqa: E402

CONFIRMATORY_PATH = common.OUTPUT_ROOT / "confirmatory" / "confirmatory-identifiability.jsonl"
OUTPUT_DIR = common.OUTPUT_ROOT / "joined"
JOINED_TABLE_PATH = OUTPUT_DIR / "joined-incidents.jsonl"
COMPARISONS_PATH = OUTPUT_DIR / "required-comparisons.json"

PRIMARY_IDENT_PATH = ("identifiability", "normalized", "correlation", "identifiability_score")


def _get(record: dict, path: tuple[str, ...]):
    value = record
    for key in path:
        value = value[key]
    return value


def _load_outcomes() -> dict[tuple[str, int], dict]:
    outcomes = {}
    for line in common.M11_6_RAW_INCIDENTS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        outcomes[(row["split"], row["seed"])] = row
    return outcomes


def build_joined_table() -> list[dict]:
    identifiability_records = [
        json.loads(line) for line in CONFIRMATORY_PATH.read_text().splitlines() if line.strip()
    ]
    outcomes = _load_outcomes()
    joined = []
    for record in identifiability_records:
        key = (record["split"], record["seed"])
        outcome = outcomes.get(key)
        if outcome is None:
            raise KeyError(f"no HydroCore-v5 outcome found for {key}")
        true_struct = record["true_source_structural"]
        joined.append(
            {
                "split": record["split"],
                "seed": record["seed"],
                "source_node": record["source_node"],
                "network_family": record["network_family"],
                "topology_id": record["topology_id"],
                "condition_kind": record["condition_kind"],
                "known_topology": record["known_topology"],
                "n_candidates": record["n_candidates"],
                "identifiability_score": _get(record, PRIMARY_IDENT_PATH),
                "identifiability_score_raw_rmse": _get(
                    record, ("identifiability", "raw", "rmse", "identifiability_score")
                ),
                "ambiguity_fraction": _get(
                    record, ("identifiability", "normalized", "correlation", "ambiguity_fraction_percentile")
                ),
                "nearest_competitor_distance": _get(
                    record, ("identifiability", "normalized", "correlation", "nearest_competitor_distance")
                ),
                "oracle_top1": record["oracle_observed"]["top1"],
                "oracle_top3": record["oracle_observed"]["top3"],
                "oracle_rank": record["oracle_observed"]["true_source_rank"],
                "oracle_mrr": record["oracle_observed"]["mrr"],
                "oracle_residual_margin": record["oracle_observed"]["residual_margin"],
                "stress_treatment": record["stress_treatment"],
                "betweenness_centrality": true_struct["betweenness_centrality"],
                "closeness_centrality": true_struct["closeness_centrality"],
                "degree": true_struct["degree"],
                "is_leaf": true_struct["is_leaf"],
                "sensor_distance_hops": true_struct["sensor_distance_hops"],
                "reservoir_distance_hops": true_struct["reservoir_distance_hops"],
                "hydrocore_top1_correct": bool(outcome["top1_correct"]),
                "hydrocore_top3_correct": bool(outcome["top3_correct"]),
                "hydrocore_reciprocal_rank": outcome["reciprocal_rank"],
                "hydrocore_posterior_entropy": outcome["posterior_entropy"],
                "hydrocore_candidate_set_size": outcome["candidate_set_size"],
                "hydrocore_calibrated": outcome["calibrated"],
                "hydrocore_conformal_truth_coverage": outcome["conformal_truth_coverage"],
                "hydrocore_outcome": outcome["outcome"],
                "hydrocore_control_action": outcome["control_action"],
                # M11.6's `outcome` field takes exactly three values:
                # VERIFIED (a plan was verified/actionable), SUPPRESSED
                # (decision withheld pending more sampling/evidence), and
                # ABSTAINED (explicit abstention). Both non-VERIFIED
                # outcomes mean the system did not commit to a confident,
                # actionable answer -- only VERIFIED is "fully committed."
                "hydrocore_abstained": outcome["outcome"] in ("SUPPRESSED", "ABSTAINED"),
            }
        )
    return joined


def required_comparisons(table: list[dict]) -> dict:
    n = len(table)
    ident = [r["identifiability_score"] for r in table]
    terciles = stats_utils.tercile_labels(ident)
    for record, label in zip(table, terciles):
        record["identifiability_tercile"] = label

    # 1. HydroCore accuracy vs identifiability tercile/quartile.
    by_tercile = {}
    for label in ("T1", "T2", "T3"):
        subset = [r for r in table if r["identifiability_tercile"] == label]
        by_tercile[label] = {
            "n": len(subset),
            "mean_identifiability_score": float(np.mean([r["identifiability_score"] for r in subset])),
            "hydrocore_top1_rate": float(np.mean([r["hydrocore_top1_correct"] for r in subset])),
            "hydrocore_top3_rate": float(np.mean([r["hydrocore_top3_correct"] for r in subset])),
            "hydrocore_mean_reciprocal_rank": float(np.mean([r["hydrocore_reciprocal_rank"] for r in subset])),
            "oracle_top1_rate": float(np.mean([r["oracle_top1"] for r in subset])),
        }
    comparison_1 = {
        "by_tercile": by_tercile,
        "t3_minus_t1_top1_bootstrap": stats_utils.unpaired_bootstrap_diff(
            [r["hydrocore_top1_correct"] for r in table if r["identifiability_tercile"] == "T3"],
            [r["hydrocore_top1_correct"] for r in table if r["identifiability_tercile"] == "T1"],
        ),
    }

    # 2. low vs high centrality (betweenness, median split), before AND
    #    after conditioning on identifiability tercile.
    centrality_labels = stats_utils.median_split([r["betweenness_centrality"] for r in table])
    for record, label in zip(table, centrality_labels):
        record["centrality_group"] = label
    unconditioned = stats_utils.unpaired_bootstrap_diff(
        [r["hydrocore_top1_correct"] for r in table if r["centrality_group"] == "HIGH"],
        [r["hydrocore_top1_correct"] for r in table if r["centrality_group"] == "LOW"],
    )
    conditioned_by_tercile = {}
    for label in ("T1", "T2", "T3"):
        stratum = [r for r in table if r["identifiability_tercile"] == label]
        high = [r["hydrocore_top1_correct"] for r in stratum if r["centrality_group"] == "HIGH"]
        low = [r["hydrocore_top1_correct"] for r in stratum if r["centrality_group"] == "LOW"]
        conditioned_by_tercile[label] = stats_utils.unpaired_bootstrap_diff(high, low)
    comparison_2 = {
        "unconditioned_high_minus_low_top1": unconditioned,
        "conditioned_on_identifiability_tercile": conditioned_by_tercile,
        "correlation_betweenness_vs_identifiability_score": _spearman(
            [r["betweenness_centrality"] for r in table], ident
        ),
    }

    # 3. short vs long source-to-sensor distance, before/after conditioning.
    # `sensor_distance_hops` is heavily tied at 0 (the true source IS a
    # sensor in most small networks here, e.g. every golden-reference
    # incident), so a median split degenerates to an empty group; split on
    # "is the source directly instrumented" (hops == 0, "LOW") vs. "must be
    # inferred indirectly" (hops > 0, "HIGH") instead -- itself a more
    # physically meaningful distinction than an arbitrary median.
    distance_values = [r["sensor_distance_hops"] for r in table]
    distance_labels = ["LOW" if d == 0 else "HIGH" for d in distance_values]
    for record, label in zip(table, distance_labels):
        record["sensor_distance_group"] = label
    unconditioned_distance = stats_utils.unpaired_bootstrap_diff(
        [r["hydrocore_top1_correct"] for r in table if r["sensor_distance_group"] == "LOW"],
        [r["hydrocore_top1_correct"] for r in table if r["sensor_distance_group"] == "HIGH"],
    )
    conditioned_distance = {}
    for label in ("T1", "T2", "T3"):
        stratum = [r for r in table if r["identifiability_tercile"] == label]
        short = [r["hydrocore_top1_correct"] for r in stratum if r["sensor_distance_group"] == "LOW"]
        long = [r["hydrocore_top1_correct"] for r in stratum if r["sensor_distance_group"] == "HIGH"]
        conditioned_distance[label] = stats_utils.unpaired_bootstrap_diff(short, long)
    comparison_3 = {
        "unconditioned_short_minus_long_top1": unconditioned_distance,
        "conditioned_on_identifiability_tercile": conditioned_distance,
        "correlation_sensor_distance_vs_identifiability_score": _spearman(distance_values, ident),
    }

    # 4. known vs unseen topology at similar identifiability levels.
    comparison_4 = {}
    for label in ("T1", "T2", "T3"):
        stratum = [r for r in table if r["identifiability_tercile"] == label]
        known = [r["hydrocore_top1_correct"] for r in stratum if r["known_topology"]]
        unseen = [r["hydrocore_top1_correct"] for r in stratum if not r["known_topology"]]
        comparison_4[label] = stats_utils.unpaired_bootstrap_diff(known, unseen)
    comparison_4["overall_known_minus_unseen"] = stats_utils.unpaired_bootstrap_diff(
        [r["hydrocore_top1_correct"] for r in table if r["known_topology"]],
        [r["hydrocore_top1_correct"] for r in table if not r["known_topology"]],
    )
    comparison_4["overall_identifiability_known_minus_unseen"] = stats_utils.unpaired_bootstrap_diff(
        [r["identifiability_score"] for r in table if r["known_topology"]],
        [r["identifiability_score"] for r in table if not r["known_topology"]],
    )

    # 5. oracle vs HydroCore on the SAME examples -- paired bootstrap.
    comparison_5 = {
        "top1": common.m91.paired_bootstrap(
            [r["oracle_top1"] for r in table], [float(r["hydrocore_top1_correct"]) for r in table]
        ),
        "top3": common.m91.paired_bootstrap(
            [r["oracle_top3"] for r in table], [float(r["hydrocore_top3_correct"]) for r in table]
        ),
        "reciprocal_rank": common.m91.paired_bootstrap(
            [r["oracle_mrr"] for r in table], [r["hydrocore_reciprocal_rank"] for r in table]
        ),
    }

    # 6/7. joint failure/success table between oracle and HydroCore.
    both_fail = sum(1 for r in table if not r["hydrocore_top1_correct"] and not bool(r["oracle_top1"]))
    hydrocore_fails_oracle_succeeds = sum(
        1 for r in table if not r["hydrocore_top1_correct"] and bool(r["oracle_top1"])
    )
    hydrocore_fails = sum(1 for r in table if not r["hydrocore_top1_correct"])
    hydrocore_succeeds_oracle_fails = sum(
        1 for r in table if r["hydrocore_top1_correct"] and not bool(r["oracle_top1"])
    )
    comparison_6_7 = {
        "n_incidents": n,
        "n_hydrocore_top1_failures": hydrocore_fails,
        "n_hydrocore_fails_oracle_succeeds": hydrocore_fails_oracle_succeeds,
        "fraction_of_hydrocore_failures_where_oracle_succeeds": (
            hydrocore_fails_oracle_succeeds / hydrocore_fails if hydrocore_fails else None
        ),
        "n_both_fail": both_fail,
        "fraction_both_fail_of_all": both_fail / n,
        "fraction_both_fail_of_hydrocore_failures": (both_fail / hydrocore_fails if hydrocore_fails else None),
        "n_hydrocore_succeeds_oracle_fails": hydrocore_succeeds_oracle_fails,
    }

    # 8. clean vs stressed identifiability degradation (unpaired, grouped
    #    by condition_kind, restricted to `locked_final_test` since that is
    #    the only split with non-NOMINAL conditions -- comparing against
    #    ITS OWN NOMINAL rows only, not `locked_topology_test`'s, so the
    #    baseline isn't confounded by known-vs-unseen topology differences
    #    (comparison 4 already shows those differ).
    comparison_8 = {}
    final_test_rows = [r for r in table if r["split"] == "locked_final_test"]
    nominal_scores = [r["identifiability_score"] for r in final_test_rows if r["condition_kind"] == "NOMINAL"]
    for condition in sorted({r["condition_kind"] for r in final_test_rows}):
        if condition == "NOMINAL":
            continue
        stressed_scores = [r["identifiability_score"] for r in final_test_rows if r["condition_kind"] == condition]
        stress_treatment = next(r["stress_treatment"] for r in table if r["condition_kind"] == condition)
        comparison_8[condition] = {
            "stress_treatment": stress_treatment,
            **stats_utils.unpaired_bootstrap_diff(stressed_scores, nominal_scores),
        }

    return {
        "n_incidents": n,
        "comparison_1_hydrocore_vs_identifiability_tercile": comparison_1,
        "comparison_2_centrality_conditioned_on_identifiability": comparison_2,
        "comparison_3_sensor_distance_conditioned_on_identifiability": comparison_3,
        "comparison_4_known_vs_unseen_at_similar_identifiability": comparison_4,
        "comparison_5_oracle_vs_hydrocore_paired": comparison_5,
        "comparison_6_7_failure_overlap": comparison_6_7,
        "comparison_8_clean_vs_stressed_identifiability": comparison_8,
    }


def _spearman(x: list[float], y: list[float]) -> float:
    x_arr, y_arr = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x_rank = _rankdata(x_arr)
    y_rank = _rankdata(y_arr)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # average ties
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            average_rank = float(np.mean(np.arange(i, j + 1)))
            for k in range(i, j + 1):
                ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_joined_table()
    comparisons = required_comparisons(table)
    with JOINED_TABLE_PATH.open("w") as handle:
        for record in table:
            handle.write(json.dumps(record) + "\n")
    COMPARISONS_PATH.write_text(json.dumps(comparisons, indent=2))
    print(f"joined {len(table)} incidents -> {JOINED_TABLE_PATH}")
    print(f"required comparisons -> {COMPARISONS_PATH}")


if __name__ == "__main__":
    main()
