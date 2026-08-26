"""Phase 3 (subgroup stratification) + Phase 5 (error taxonomy) over the
M11.6 diagnostic table (branch exp/failure-mode-diagnostics).

Reads reports/evaluation/failure-mode-diagnostics/m11-6-diagnostic-table.jsonl
(built by build_m11_6_diagnostic_table.py from frozen locked evidence; this
script itself never touches data/locked/ or any m9-*/m10-*/m11-* report).

All statistics here are DESCRIPTIVE, matching m11-6-metrics.json's own
`"topology_shift_predictive": "DESCRIPTIVE_NON_GATING"` framing for the
novel-topology split: with n=125 total (n=20 novel-topology), no subgroup
comparison in this script is claimed as a confirmatory statistical finding.
Every subgroup table reports n and flags MINIMUM_GROUP_SIZE violations
rather than suppressing them silently.

Usage: python3 scripts/hydrocore_v5_experimental/failure_mode_diagnostics/analyze_m11_6_failure_modes.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
TABLE_PATH = ROOT / "reports" / "evaluation" / "failure-mode-diagnostics" / "m11-6-diagnostic-table.jsonl"
OUTPUT_DIR = ROOT / "reports" / "evaluation" / "failure-mode-diagnostics"

MINIMUM_GROUP_SIZE = 10  # matches SplitConformalCalibrator's own convention (conformal.py)

STRESS_CONDITIONS = {
    "LOW_COVERAGE_ACTIVE_SAMPLING",
    "SENSOR_DROPOUT",
    "SENSOR_HEALTH_DEGRADED",
    "MEASUREMENT_NOISE",
    "SEVERITY_SHIFT",
    "AMBIGUITY_DISAGREEMENT",
}


def load_table() -> list[dict[str, Any]]:
    with TABLE_PATH.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "small_sample": n < MINIMUM_GROUP_SIZE,
        "top1": _mean([row["top1_correct"] for row in rows]),
        "top3": _mean([row["top3_correct"] for row in rows]),
        "mrr": _mean([row["reciprocal_rank"] for row in rows]),
        "calibrated_rate": _mean([row["calibrated"] for row in rows]),
        "actionable_rate": _mean([row["actionable"] for row in rows]),
        "mean_candidate_set_size": _mean([row["candidate_set_size"] for row in rows]),
        "mean_posterior_entropy": _mean([row["posterior_entropy"] for row in rows]),
        "conformal_truth_coverage_rate": _mean([row["conformal_truth_coverage"] for row in rows]),
    }


def stratify(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return {str(group_key): group_metrics(group_rows) for group_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0]))}


def tercile_bins(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted(row[field] for row in rows if row.get(field) is not None)
    if len(values) < MINIMUM_GROUP_SIZE:
        return {"note": f"insufficient non-null values for {field} to bin (n={len(values)})"}
    third = len(values) // 3
    low_cut, high_cut = values[third], values[2 * third]

    def bucket(row: dict[str, Any]) -> str:
        value = row.get(field)
        if value is None:
            return "missing"
        if value <= low_cut:
            return f"low(<={low_cut:.4f})"
        if value <= high_cut:
            return f"mid(<={high_cut:.4f})"
        return f"high(>{high_cut:.4f})"

    return stratify(rows, bucket)


def build_subgroup_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["seen_topology"]]
    novel = [row for row in rows if not row["seen_topology"]]
    known_nominal = [row for row in known if row["condition_kind"] == "NOMINAL"]

    return {
        "overall": group_metrics(rows),
        "by_seen_topology": {"known": group_metrics(known), "novel_unseen": group_metrics(novel)},
        "known_vs_novel_NOMINAL_only": {
            "note": "novel-topology evidence is 100% NOMINAL; this is the only apples-to-apples condition_kind comparison across seen_topology",
            "known_nominal": group_metrics(known_nominal),
            "novel_nominal": group_metrics(novel),
        },
        "by_condition_kind": stratify(rows, lambda row: row["condition_kind"]),
        "by_condition_kind_known_only": stratify(known, lambda row: row["condition_kind"]),
        "by_network_family_or_topology_id": stratify(rows, lambda row: row["topology_id"]),
        "by_node_count": stratify(rows, lambda row: row["node_count"]),
        "by_source_degree": stratify(rows, lambda row: row["source_degree"]),
        "by_source_is_boundary_node": stratify(rows, lambda row: row["source_is_boundary_node"]),
        "by_source_hops_to_reservoir": stratify(rows, lambda row: row["source_hops_to_reservoir"]),
        "by_graph_diameter": stratify(rows, lambda row: row["graph_diameter"]),
        "by_source_betweenness_centrality_tercile": tercile_bins(rows, "source_betweenness_centrality"),
        "by_source_closeness_centrality_tercile": tercile_bins(rows, "source_closeness_centrality"),
        "by_source_normalized_graph_position_tercile": tercile_bins(rows, "source_normalized_graph_position"),
        "stress_vs_nominal_known_only": {
            "nominal": group_metrics([row for row in known if row["condition_kind"] == "NOMINAL"]),
            "any_stress_condition": group_metrics([row for row in known if row["condition_kind"] in STRESS_CONDITIONS]),
        },
    }


# ---------------------------------------------------------------------------
# Phase 5: error taxonomy (overlap allowed; only categories with supporting
# evidence in the available fields are populated; each is reported with the
# fraction of TOP-1 FAILURES it plausibly covers, not the fraction of all
# examples).
# ---------------------------------------------------------------------------


def build_error_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if not row["top1_correct"]]
    n_failures = len(failures)
    if n_failures == 0:
        return {"n_top1_failures": 0, "note": "no top-1 failures in this population"}

    def fraction(predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        matched = [row for row in failures if predicate(row)]
        return {"n": len(matched), "fraction_of_top1_failures": len(matched) / n_failures}

    categories = {
        "ranking_failure_true_source_in_top3": {
            **fraction(lambda row: row["top3_correct"]),
            "definition": "top1 wrong but top3 correct -- model's belief set contains the truth, ordering does not",
        },
        "source_absent_from_top3": {
            **fraction(lambda row: not row["top3_correct"]),
            "definition": "truth outside the top-3 candidates entirely -- a representation/evidence gap, not just a ranking slip",
        },
        "topology_transfer_failure": {
            **fraction(lambda row: not row["seen_topology"]),
            "definition": "failure occurred on a topology absent from training (locked_topology_test)",
        },
        "stress_induced_known_topology": {
            **fraction(lambda row: row["seen_topology"] and row["condition_kind"] in STRESS_CONDITIONS),
            "definition": "failure on a KNOWN topology under a non-NOMINAL stress condition_kind",
        },
        "ambiguity_or_low_coverage": {
            **fraction(lambda row: row["condition_kind"] in ("AMBIGUITY_DISAGREEMENT", "LOW_COVERAGE_ACTIVE_SAMPLING")),
            "definition": "condition_kind is explicitly an ambiguous-evidence / low-sensor-coverage stress category -- plausibly intrinsically hard given available evidence, not necessarily a model defect",
        },
        "calibration_or_ood_fail_closed": {
            **fraction(lambda row: not row["calibrated"]),
            "definition": "calibration was correctly withheld (calibrated=False) for this example -- the failure to be ACTIONABLE is governance behaving as designed, distinct from the localization failure itself",
        },
        "high_confidence_wrong_top1": {
            **fraction(lambda row: (row.get("candidate_set_size") or 99) <= 2),
            "definition": "small candidate set (<=2) despite wrong top1 -- a confidently-wrong prediction, not a diffuse/uncertain one",
        },
        "network_identity_or_canonicalization_issue": {
            "n": 0,
            "fraction_of_top1_failures": 0.0,
            "definition": (
                "PR #12's live-serving network-identity/.inp-round-trip hashing "
                "defect is NOT evidenced in this population: M11.6 uses one fixed, "
                "non-round-tripped .inp per known family and per novel topology "
                "(single network_sha256 per family, verified in "
                "build_m11_6_diagnostic_table.py), so that specific mechanism does "
                "not apply here. Listed at n=0 rather than omitted, so this "
                "explicitly-checked-and-ruled-out category is visible, not silently dropped."
            ),
        },
        "insufficient_evidence_source_ambiguous_by_construction": {
            **fraction(lambda row: row["condition_kind"] == "AMBIGUITY_DISAGREEMENT"),
            "definition": "condition_kind explicitly constructs source ambiguity (AMBIGUITY_DISAGREEMENT) -- ceiling on achievable top1 is not 1.0 by design",
        },
    }
    return {"n_top1_failures": n_failures, "n_total": len(rows), "categories": categories}


def main() -> None:
    rows = load_table()
    subgroup_report = build_subgroup_report(rows)
    taxonomy = build_error_taxonomy(rows)
    taxonomy_known = build_error_taxonomy([row for row in rows if row["seen_topology"]])
    taxonomy_novel = build_error_taxonomy([row for row in rows if not row["seen_topology"]])

    (OUTPUT_DIR / "m11-6-subgroup-metrics.json").write_text(
        json.dumps(subgroup_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "m11-6-error-taxonomy.json").write_text(
        json.dumps(
            {"overall": taxonomy, "known_topology_only": taxonomy_known, "novel_topology_only": taxonomy_novel},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"n={len(rows)}; top1 failures overall={taxonomy['n_top1_failures']}, known={taxonomy_known['n_top1_failures']}, novel={taxonomy_novel['n_top1_failures']}")
    print("Wrote m11-6-subgroup-metrics.json and m11-6-error-taxonomy.json")


if __name__ == "__main__":
    main()
