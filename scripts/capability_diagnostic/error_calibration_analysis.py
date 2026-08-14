"""Capability diagnostic Sections 37-38: error clustering and confidence/
calibration analysis, mined entirely from the already-executed 264-run LIVE
post-remediation dataset.

Outputs:
  reports/evaluation/capability-diagnostic/feature-distribution.json
  (keys: error_clustering, confidence_analysis)

Filename choice: reused per the protocol's declared artifact list
(`reports/evaluation/capability-diagnostic/feature-distribution.json` is
listed among the predeclared Section 3 data-source/output paths for this
kind of per-incident feature mining); no better-fitting existing name was
found in docs/evaluation/CAPABILITY_DIAGNOSTIC_PROTOCOL.md or protocol.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

DATA_PATH = ROOT / "reports/evaluation/live-robustness/post-remediation-results.json"
OUT_PATH = ROOT / "reports/evaluation/capability-diagnostic/feature-distribution.json"

N_BINS = 10


def top1_and_top2(belief: dict[str, float]) -> tuple[str, float, str | None, float]:
    ordered = sorted(belief.items(), key=lambda kv: (-kv[1], kv[0]))
    top1_node, top1_p = ordered[0]
    if len(ordered) > 1:
        top2_node, top2_p = ordered[1]
    else:
        top2_node, top2_p = None, 0.0
    return top1_node, top1_p, top2_node, top2_p


def decile_bin(value: float, lo: float, hi: float, n_bins: int = N_BINS) -> int:
    if hi <= lo:
        return 0
    frac = (value - lo) / (hi - lo)
    idx = int(frac * n_bins)
    return min(max(idx, 0), n_bins - 1)


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for this diagnostic"
    locked_before = locked_test_opened(ROOT)

    all_records: list[dict[str, Any]] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    total_records = len(all_records)

    # Records with a known top1_correct outcome (excludes ABSTAINED/errored).
    scored = [r for r in all_records if r.get("top1_correct") is not None]
    n_scored = len(scored)
    overall_failure_rate = sum(1 for r in scored if r["top1_correct"] is False) / n_scored if n_scored else None

    # ---------------- Section 37: error clustering ----------------
    def slice_failure_rates(key_fn, label: str) -> dict[str, Any]:
        buckets: dict[Any, list[bool]] = defaultdict(list)
        for r in scored:
            k = key_fn(r)
            if k is None:
                continue
            buckets[k].append(r["top1_correct"] is False)
        out = {}
        for k, vals in buckets.items():
            n = len(vals)
            fail_rate = sum(vals) / n
            out[str(k)] = {
                "n": n,
                "failure_rate": fail_rate,
                "failure_rate_minus_overall_base_rate": (
                    fail_rate - overall_failure_rate if overall_failure_rate is not None else None
                ),
            }
        return {"dimension": label, "slices": out}

    error_clustering_by_dimension = {
        "perturbation_type": slice_failure_rates(lambda r: r["perturbation_type"], "perturbation_type"),
        "perturbation_level": slice_failure_rates(lambda r: r["perturbation_level"], "perturbation_level"),
        "topology_class": slice_failure_rates(lambda r: r["topology_class"], "topology_class"),
        "network_id": slice_failure_rates(lambda r: r["network_id"], "network_id"),
        "sensor_count": slice_failure_rates(lambda r: r["sensor_count"], "sensor_count"),
        "node_count": slice_failure_rates(lambda r: r["node_count"], "node_count"),
        "observation_count": slice_failure_rates(lambda r: r["observation_count"], "observation_count"),
        "ood_level": slice_failure_rates(lambda r: r["ood_level"], "ood_level"),
        "calibrated": slice_failure_rates(lambda r: r["calibrated"], "calibrated"),
    }

    # disagreement_js and posterior_entropy are continuous -> bin into
    # quartiles for the clustering view (deciles reserved for the dedicated
    # confidence-analysis binning below).
    def quartile_slices(key_fn, label: str) -> dict[str, Any]:
        vals_present = [(r, key_fn(r)) for r in scored if key_fn(r) is not None]
        if not vals_present:
            return {"dimension": label, "slices": {}, "note": "no non-null values present"}
        values = sorted(v for _, v in vals_present)
        n = len(values)
        cuts = [values[int(n * q)] for q in (0.25, 0.5, 0.75)]

        def q_of(v: float) -> str:
            if v <= cuts[0]:
                return "Q1_lowest"
            if v <= cuts[1]:
                return "Q2"
            if v <= cuts[2]:
                return "Q3"
            return "Q4_highest"

        buckets: dict[str, list[bool]] = defaultdict(list)
        for r, v in vals_present:
            buckets[q_of(v)].append(r["top1_correct"] is False)
        out = {}
        for k, fails in buckets.items():
            n_k = len(fails)
            fail_rate = sum(fails) / n_k
            out[k] = {
                "n": n_k,
                "failure_rate": fail_rate,
                "failure_rate_minus_overall_base_rate": fail_rate - overall_failure_rate,
            }
        return {"dimension": label, "quartile_cuts": cuts, "slices": out}

    error_clustering_by_dimension["disagreement_js_quartiles"] = quartile_slices(
        lambda r: r.get("disagreement_js"), "disagreement_js"
    )
    error_clustering_by_dimension["posterior_entropy_quartiles"] = quartile_slices(
        lambda r: r.get("posterior_entropy"), "posterior_entropy"
    )

    # Rank slices by how much they exceed base rate (most disproportionate
    # failure concentration first), only for slices with n>=5 to avoid noise.
    disproportionate: list[dict[str, Any]] = []
    for dim, block in error_clustering_by_dimension.items():
        for slice_key, stats in block["slices"].items():
            if stats["n"] >= 5 and stats["failure_rate_minus_overall_base_rate"] is not None:
                disproportionate.append(
                    {
                        "dimension": dim,
                        "slice": slice_key,
                        "n": stats["n"],
                        "failure_rate": stats["failure_rate"],
                        "excess_over_base_rate": stats["failure_rate_minus_overall_base_rate"],
                    }
                )
    disproportionate.sort(key=lambda x: x["excess_over_base_rate"], reverse=True)

    error_clustering = {
        "n_scored_records": n_scored,
        "n_excluded_no_top1_correct_known": total_records - n_scored,
        "overall_failure_rate_base_rate": overall_failure_rate,
        "by_dimension": error_clustering_by_dimension,
        "most_disproportionately_failing_slices_n_ge_5": disproportionate[:15],
    }

    # ---------------- Section 38: confidence / calibration analysis ----------------
    conf_records = [r for r in scored if r.get("true_source_probability") is not None]
    conf_values = [r["true_source_probability"] for r in conf_records]
    conf_lo, conf_hi = (min(conf_values), max(conf_values)) if conf_values else (0.0, 1.0)

    conf_bins: list[list[dict[str, Any]]] = [[] for _ in range(N_BINS)]
    for r in conf_records:
        b = decile_bin(r["true_source_probability"], conf_lo, conf_hi)
        conf_bins[b].append(r)

    def bin_stats(bin_records: list[dict[str, Any]], value_key) -> dict[str, Any] | None:
        if not bin_records:
            return None
        n = len(bin_records)
        acc = sum(1 for r in bin_records if r["top1_correct"] is True) / n
        vals = [value_key(r) for r in bin_records]
        return {
            "n": n,
            "mean_value": sum(vals) / n,
            "value_range": [min(vals), max(vals)],
            "empirical_accuracy": acc,
        }

    confidence_calibration_curve = []
    for i, b in enumerate(conf_bins):
        stats = bin_stats(b, lambda r: r["true_source_probability"])
        confidence_calibration_curve.append({"bin_index": i, **(stats or {"n": 0})})

    # ECE: weighted mean |confidence - accuracy| over bins with n>0.
    ece_terms = []
    for b in confidence_calibration_curve:
        if b.get("n", 0) > 0:
            ece_terms.append((b["n"], abs(b["mean_value"] - b["empirical_accuracy"])))
    n_total_ece = sum(n for n, _ in ece_terms)
    ece = sum(n * gap for n, gap in ece_terms) / n_total_ece if n_total_ece else None

    # accuracy vs posterior_entropy (binned by decile over observed range)
    entropy_records = [r for r in scored if r.get("posterior_entropy") is not None]
    entropy_values = [r["posterior_entropy"] for r in entropy_records]
    ent_lo, ent_hi = (min(entropy_values), max(entropy_values)) if entropy_values else (0.0, 1.0)
    entropy_bins: list[list[dict[str, Any]]] = [[] for _ in range(N_BINS)]
    for r in entropy_records:
        b = decile_bin(r["posterior_entropy"], ent_lo, ent_hi)
        entropy_bins[b].append(r)
    accuracy_vs_entropy = []
    for i, b in enumerate(entropy_bins):
        stats = bin_stats(b, lambda r: r["posterior_entropy"])
        accuracy_vs_entropy.append({"bin_index": i, **(stats or {"n": 0})})

    # accuracy vs margin between top-1 and top-2 fused_belief probability
    margin_records = []
    for r in scored:
        fb = r.get("fused_belief")
        if not fb:
            continue
        _, p1, _, p2 = top1_and_top2(fb)
        margin_records.append({**r, "_margin": p1 - p2})
    margin_values = [r["_margin"] for r in margin_records]
    mar_lo, mar_hi = (min(margin_values), max(margin_values)) if margin_values else (0.0, 1.0)
    margin_bins: list[list[dict[str, Any]]] = [[] for _ in range(N_BINS)]
    for r in margin_records:
        b = decile_bin(r["_margin"], mar_lo, mar_hi)
        margin_bins[b].append(r)
    accuracy_vs_margin = []
    for i, b in enumerate(margin_bins):
        stats = bin_stats(b, lambda r: r["_margin"])
        accuracy_vs_margin.append({"bin_index": i, **(stats or {"n": 0})})

    # "wrong and confident" vs "appropriately uncertain" verdict:
    # look at the highest-confidence bin(s) (top ~20% by count) and see if
    # their empirical accuracy is high (well-calibrated / appropriately
    # confident) or low (wrong and confident / miscalibrated); and check
    # whether failures concentrate in low-confidence / high-entropy bins.
    nonempty_conf_bins = [b for b in confidence_calibration_curve if b.get("n", 0) > 0]
    high_conf_bins = sorted(nonempty_conf_bins, key=lambda b: b["mean_value"], reverse=True)[: max(1, len(nonempty_conf_bins) // 5)]
    low_conf_bins = sorted(nonempty_conf_bins, key=lambda b: b["mean_value"])[: max(1, len(nonempty_conf_bins) // 5)]
    high_conf_n = sum(b["n"] for b in high_conf_bins)
    high_conf_acc = (
        sum(b["n"] * b["empirical_accuracy"] for b in high_conf_bins) / high_conf_n if high_conf_n else None
    )
    low_conf_n = sum(b["n"] for b in low_conf_bins)
    low_conf_acc = sum(b["n"] * b["empirical_accuracy"] for b in low_conf_bins) / low_conf_n if low_conf_n else None

    if high_conf_acc is not None and low_conf_acc is not None:
        if high_conf_acc >= 0.7 and high_conf_acc > low_conf_acc:
            verdict = "APPROPRIATELY_UNCERTAIN"
        elif high_conf_acc < 0.5:
            verdict = "WRONG_AND_CONFIDENT"
        else:
            verdict = "MIXED"
    else:
        verdict = "INCONCLUSIVE_INSUFFICIENT_DATA"

    confidence_analysis = {
        "n_scored_records": n_scored,
        "n_with_true_source_probability": len(conf_records),
        "confidence_range_observed": [conf_lo, conf_hi],
        "confidence_vs_accuracy_decile_bins": confidence_calibration_curve,
        "expected_calibration_error_ece": ece,
        "posterior_entropy_range_observed": [ent_lo, ent_hi],
        "accuracy_vs_posterior_entropy_decile_bins": accuracy_vs_entropy,
        "top1_top2_margin_range_observed": [mar_lo, mar_hi],
        "accuracy_vs_top1_top2_margin_decile_bins": accuracy_vs_margin,
        "high_confidence_quintile": {
            "n": high_conf_n,
            "mean_confidence": (
                sum(b["n"] * b["mean_value"] for b in high_conf_bins) / high_conf_n if high_conf_n else None
            ),
            "empirical_accuracy": high_conf_acc,
        },
        "low_confidence_quintile": {
            "n": low_conf_n,
            "mean_confidence": (
                sum(b["n"] * b["mean_value"] for b in low_conf_bins) / low_conf_n if low_conf_n else None
            ),
            "empirical_accuracy": low_conf_acc,
        },
        "verdict": verdict,
        "verdict_method": (
            "Compares empirical accuracy in the highest-confidence quintile of bins against the "
            "lowest-confidence quintile. APPROPRIATELY_UNCERTAIN: high-confidence accuracy is "
            ">=0.7 and exceeds low-confidence accuracy (errors concentrate where the model is "
            "already less sure). WRONG_AND_CONFIDENT: high-confidence accuracy is <0.5 (the "
            "model is often wrong even when it reports high true_source_probability -- a "
            "miscalibration red flag). MIXED: neither clean pattern holds."
        ),
    }

    report = {
        "schema_version": 1,
        "sections": "37_error_clustering, 38_confidence_calibration_analysis",
        "locked_test_opened": locked_before,
        "source_data": str(DATA_PATH.relative_to(ROOT)),
        "total_records_in_dataset": total_records,
        "error_clustering": error_clustering,
        "confidence_analysis": confidence_analysis,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    locked_after = locked_test_opened(ROOT)
    print(
        json.dumps(
            {
                "n_scored": n_scored,
                "overall_failure_rate": overall_failure_rate,
                "top_disproportionate_slices": disproportionate[:5],
                "ece": ece,
                "confidence_verdict": verdict,
                "high_conf_acc": high_conf_acc,
                "low_conf_acc": low_conf_acc,
                "locked_test_opened_after": locked_after,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
