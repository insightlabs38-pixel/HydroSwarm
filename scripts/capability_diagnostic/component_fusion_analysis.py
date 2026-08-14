"""Capability diagnostic Sections 14-15: neural/classical/hybrid decomposition
and fusion diagnostic, mined entirely from the already-executed 264-run LIVE
post-remediation dataset (`reports/evaluation/live-robustness/post-remediation-results.json`).

This script performs NO new inference and calls NO production pipeline code.
It only reads already-recorded `neural_belief` / `classical_belief` /
`fused_belief` per-node probability dicts from the real dataset and computes
argmax comparisons and a pure-numpy offline counterfactual fusion sweep over
those already-recorded distributions. `fuse_source_probabilities` (the real
runtime fusion function in src/hydroswarm/inference/fusion.py) is never
imported or called here -- production fusion policy is out of scope to
change or exercise, per the protocol's production-behavior freeze.

Outputs:
  reports/evaluation/capability-diagnostic/component-decomposition.json
  reports/evaluation/capability-diagnostic/fusion-analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

DATA_PATH = ROOT / "reports/evaluation/live-robustness/post-remediation-results.json"
COMPONENT_OUT = ROOT / "reports/evaluation/capability-diagnostic/component-decomposition.json"
FUSION_OUT = ROOT / "reports/evaluation/capability-diagnostic/fusion-analysis.json"

SWEEP_POINTS = [
    ("neural_1.00_classical_0.00", 1.00),
    ("neural_0.75_classical_0.25", 0.75),
    ("neural_0.50_classical_0.50", 0.50),
    ("neural_0.25_classical_0.75", 0.25),
    ("neural_0.00_classical_1.00", 0.00),
]


def argmax_node(belief: dict[str, float] | None) -> str | None:
    if not belief:
        return None
    best_node = None
    best_p = float("-inf")
    # Deterministic tie-break: iterate in sorted node-id order, strict >
    # keeps the first (alphabetically smallest) node id on ties, matching
    # a stable, reproducible convention (not the production tie-break,
    # which is out of scope here -- this is diagnostic-only argmax).
    for node in sorted(belief.keys()):
        p = belief[node]
        if p > best_p:
            best_p = p
            best_node = node
    return best_node


def blended_belief(neural: dict[str, float], classical: dict[str, float], w_neural: float) -> dict[str, float]:
    nodes = sorted(set(neural) | set(classical))
    out = {}
    for node in nodes:
        n = neural.get(node, 0.0)
        c = classical.get(node, 0.0)
        out[node] = w_neural * n + (1.0 - w_neural) * c
    return out


def usable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records where neural_belief, classical_belief, fused_belief, and
    source_node are all present (i.e. analysis actually completed;
    excludes the ABSTAINED/error records where the run failed before
    belief computation)."""
    out = []
    for r in records:
        if r.get("neural_belief") and r.get("classical_belief") and r.get("fused_belief") and r.get("source_node"):
            out.append(r)
    return out


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for this diagnostic"
    locked_before = locked_test_opened(ROOT)

    all_records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    total_records = len(all_records)
    records = usable_records(all_records)
    excluded = total_records - len(records)

    # ---------- Section 14: neural/classical/hybrid decomposition ----------
    def classify(rec: dict[str, Any]) -> str:
        source = rec["source_node"]
        n_ok = argmax_node(rec["neural_belief"]) == source
        c_ok = argmax_node(rec["classical_belief"]) == source
        if n_ok and c_ok:
            return "neural_correct_classical_correct"
        if n_ok and not c_ok:
            return "neural_correct_classical_wrong"
        if not n_ok and c_ok:
            return "neural_wrong_classical_correct"
        return "both_wrong"

    def decomposition_block(recs: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(recs)
        counts: dict[str, int] = {
            "neural_correct_classical_correct": 0,
            "neural_correct_classical_wrong": 0,
            "neural_wrong_classical_correct": 0,
            "both_wrong": 0,
        }
        for r in recs:
            counts[classify(r)] += 1
        rates = {k: (v / n if n else None) for k, v in counts.items()}
        neural_top1 = sum(1 for r in recs if argmax_node(r["neural_belief"]) == r["source_node"]) / n if n else None
        classical_top1 = (
            sum(1 for r in recs if argmax_node(r["classical_belief"]) == r["source_node"]) / n if n else None
        )
        fused_top1 = sum(1 for r in recs if argmax_node(r["fused_belief"]) == r["source_node"]) / n if n else None
        return {
            "n": n,
            "counts": counts,
            "rates": rates,
            "neural_only_top1": neural_top1,
            "classical_only_top1": classical_top1,
            "fused_top1": fused_top1,
        }

    overall = decomposition_block(records)

    by_perturbation: dict[str, Any] = {}
    perturbation_types = sorted({r["perturbation_type"] for r in records})
    for pt in perturbation_types:
        subset = [r for r in records if r["perturbation_type"] == pt]
        by_perturbation[pt] = decomposition_block(subset)

    component_report = {
        "schema_version": 1,
        "section": "14_component_decomposition",
        "locked_test_opened": locked_before,
        "source_data": str(DATA_PATH.relative_to(ROOT)),
        "total_records_in_dataset": total_records,
        "records_excluded_missing_beliefs": excluded,
        "records_excluded_reason": (
            "ABSTAINED/errored runs (e.g. error_class=ANALYZE_409) where "
            "neural_belief/classical_belief/fused_belief were never computed; "
            "excluded from argmax comparisons since there is no belief to argmax."
        ),
        "records_used": len(records),
        "overall": overall,
        "by_perturbation_type": by_perturbation,
        "tie_break_note": (
            "argmax ties broken by lexicographically smallest node id (diagnostic-only "
            "convention, distinct from and not a claim about production tie-break behavior)."
        ),
    }
    COMPONENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMPONENT_OUT.write_text(json.dumps(component_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---------- Section 15: fusion diagnostic ----------
    fusion_harms: list[dict[str, Any]] = []
    fusion_outcome_counts = {
        "fusion_improves_over_both": 0,  # fused correct, neither component alone was correct
        "fusion_matches_neural_only": 0,  # fused correct, neural correct, classical wrong
        "fusion_matches_classical_only": 0,  # fused correct, classical correct, neural wrong
        "fusion_matches_both_correct": 0,  # fused correct, both components correct
        "fusion_harms_neural_was_right": 0,  # neural correct, classical wrong, fused wrong
        "fusion_harms_classical_was_right": 0,  # classical correct, neural wrong, fused wrong
        "fusion_harms_both_were_right": 0,  # both correct, fused wrong
        "fusion_wrong_both_wrong": 0,  # both wrong, fused wrong (fusion didn't help but nothing to harm)
    }
    for r in records:
        source = r["source_node"]
        n_ok = argmax_node(r["neural_belief"]) == source
        c_ok = argmax_node(r["classical_belief"]) == source
        f_ok = argmax_node(r["fused_belief"]) == source
        if f_ok:
            if n_ok and c_ok:
                fusion_outcome_counts["fusion_matches_both_correct"] += 1
            elif n_ok and not c_ok:
                fusion_outcome_counts["fusion_matches_neural_only"] += 1
            elif c_ok and not n_ok:
                fusion_outcome_counts["fusion_matches_classical_only"] += 1
            else:
                fusion_outcome_counts["fusion_improves_over_both"] += 1
        else:
            if n_ok and c_ok:
                fusion_outcome_counts["fusion_harms_both_were_right"] += 1
                fusion_harms.append(
                    {
                        "run_id": r["run_id"],
                        "source_node": source,
                        "neural_argmax": argmax_node(r["neural_belief"]),
                        "classical_argmax": argmax_node(r["classical_belief"]),
                        "fused_argmax": argmax_node(r["fused_belief"]),
                        "fusion_trust": r.get("fusion_trust"),
                        "which_was_right": "both",
                        "perturbation_type": r["perturbation_type"],
                        "perturbation_level": r["perturbation_level"],
                    }
                )
            elif n_ok and not c_ok:
                fusion_outcome_counts["fusion_harms_neural_was_right"] += 1
                fusion_harms.append(
                    {
                        "run_id": r["run_id"],
                        "source_node": source,
                        "neural_argmax": argmax_node(r["neural_belief"]),
                        "classical_argmax": argmax_node(r["classical_belief"]),
                        "fused_argmax": argmax_node(r["fused_belief"]),
                        "fusion_trust": r.get("fusion_trust"),
                        "which_was_right": "neural",
                        "perturbation_type": r["perturbation_type"],
                        "perturbation_level": r["perturbation_level"],
                    }
                )
            elif c_ok and not n_ok:
                fusion_outcome_counts["fusion_harms_classical_was_right"] += 1
                fusion_harms.append(
                    {
                        "run_id": r["run_id"],
                        "source_node": source,
                        "neural_argmax": argmax_node(r["neural_belief"]),
                        "classical_argmax": argmax_node(r["classical_belief"]),
                        "fused_argmax": argmax_node(r["fused_belief"]),
                        "fusion_trust": r.get("fusion_trust"),
                        "which_was_right": "classical",
                        "perturbation_type": r["perturbation_type"],
                        "perturbation_level": r["perturbation_level"],
                    }
                )
            else:
                fusion_outcome_counts["fusion_wrong_both_wrong"] += 1

    n_records = len(records)
    fusion_harms_total = (
        fusion_outcome_counts["fusion_harms_neural_was_right"]
        + fusion_outcome_counts["fusion_harms_classical_was_right"]
        + fusion_outcome_counts["fusion_harms_both_were_right"]
    )
    at_least_one_component_correct = sum(
        1
        for r in records
        if argmax_node(r["neural_belief"]) == r["source_node"] or argmax_node(r["classical_belief"]) == r["source_node"]
    )

    # "If neural is good but hybrid is bad, do NOT retrain the neural model
    # first" pattern check: does neural-correct/fused-wrong happen more
    # than classical-correct/fused-wrong?
    neural_was_right_fusion_wrong = (
        fusion_outcome_counts["fusion_harms_neural_was_right"] + fusion_outcome_counts["fusion_harms_both_were_right"]
    )
    classical_was_right_fusion_wrong = (
        fusion_outcome_counts["fusion_harms_classical_was_right"]
        + fusion_outcome_counts["fusion_harms_both_were_right"]
    )
    pattern_neural_good_hybrid_bad_holds = neural_was_right_fusion_wrong > 0 and (
        neural_was_right_fusion_wrong >= classical_was_right_fusion_wrong
    )

    # fusion_trust distribution, and whether it correlates with which
    # branch fusion effectively followed (argmax(fused) == argmax(neural)
    # vs argmax(fused) == argmax(classical) vs neither/both).
    def branch_followed(r: dict[str, Any]) -> str:
        f = argmax_node(r["fused_belief"])
        n = argmax_node(r["neural_belief"])
        c = argmax_node(r["classical_belief"])
        matches_n = f == n
        matches_c = f == c
        if matches_n and matches_c:
            return "matches_both"
        if matches_n:
            return "matches_neural"
        if matches_c:
            return "matches_classical"
        return "matches_neither"

    trust_values = [r.get("fusion_trust") for r in records if r.get("fusion_trust") is not None]
    trust_by_branch: dict[str, list[float]] = {
        "matches_both": [],
        "matches_neural": [],
        "matches_classical": [],
        "matches_neither": [],
    }
    for r in records:
        if r.get("fusion_trust") is not None:
            trust_by_branch[branch_followed(r)].append(r["fusion_trust"])

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "mean": None, "min": None, "max": None}
        return {
            "n": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    trust_distribution = {
        "overall": stats(trust_values),
        "by_branch_followed": {k: stats(v) for k, v in trust_by_branch.items()},
        "branch_followed_counts": {k: len(v) for k, v in trust_by_branch.items()},
    }

    # ---------- Offline counterfactual fusion sweep (pure numpy blend) ----------
    def top1_for_weight(recs: list[dict[str, Any]], w_neural: float) -> float | None:
        if not recs:
            return None
        correct = 0
        for r in recs:
            blend = blended_belief(r["neural_belief"], r["classical_belief"], w_neural)
            if argmax_node(blend) == r["source_node"]:
                correct += 1
        return correct / len(recs)

    def sweep_block(recs: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(recs)
        if n == 0:
            return {"n": 0}
        neural_top1 = sum(1 for r in recs if argmax_node(r["neural_belief"]) == r["source_node"]) / n
        classical_top1 = sum(1 for r in recs if argmax_node(r["classical_belief"]) == r["source_node"]) / n
        current_dynamic_top1 = sum(1 for r in recs if argmax_node(r["fused_belief"]) == r["source_node"]) / n
        sweep = {label: top1_for_weight(recs, w) for label, w in SWEEP_POINTS}
        sweep["current_dynamic_trust_fused_belief_as_recorded"] = current_dynamic_top1
        best_component = max(neural_top1, classical_top1)
        regret = current_dynamic_top1 - best_component
        return {
            "n": n,
            "neural_only_top1": neural_top1,
            "classical_only_top1": classical_top1,
            "sweep_top1_by_neural_weight": sweep,
            "current_dynamic_trust_top1": current_dynamic_top1,
            "best_single_component_top1": best_component,
            "regret_current_dynamic_vs_best_component": regret,
        }

    sweep_overall = sweep_block(records)
    sweep_by_perturbation = {pt: sweep_block([r for r in records if r["perturbation_type"] == pt]) for pt in perturbation_types}

    fusion_report = {
        "schema_version": 1,
        "section": "15_fusion_diagnostic",
        "locked_test_opened": locked_before,
        "source_data": str(DATA_PATH.relative_to(ROOT)),
        "note_production_fusion_untouched": (
            "src/hydroswarm/inference/fusion.py::fuse_source_probabilities was never imported "
            "or called by this script. The offline sweep below is a pure-numpy-free (plain "
            "python dict) weighted blend computed directly over the already-recorded "
            "neural_belief/classical_belief distributions in the dataset -- diagnostic-only, "
            "not a proposal to change production fusion policy."
        ),
        "records_used": n_records,
        "fusion_outcome_counts": fusion_outcome_counts,
        "fusion_harms_total_count": fusion_harms_total,
        "fusion_harms_rate_of_used_records": (fusion_harms_total / n_records if n_records else None),
        "fusion_harms_cases": fusion_harms,
        "at_least_one_component_correct_count": at_least_one_component_correct,
        "pattern_check": {
            "description": (
                "Protocol explicitly warns: 'If neural is good but hybrid is bad, do NOT "
                "retrain the neural model first.' This checks whether neural-correct/"
                "fusion-wrong cases occur at least as often as classical-correct/fusion-wrong "
                "cases in this dataset."
            ),
            "neural_was_right_but_fusion_wrong_count": neural_was_right_fusion_wrong,
            "classical_was_right_but_fusion_wrong_count": classical_was_right_fusion_wrong,
            "pattern_neural_good_hybrid_bad_holds": pattern_neural_good_hybrid_bad_holds,
        },
        "fusion_trust_distribution": trust_distribution,
        "offline_counterfactual_fusion_sweep": {
            "overall": sweep_overall,
            "by_perturbation_type": sweep_by_perturbation,
            "sweep_points": {label: w for label, w in SWEEP_POINTS},
        },
    }
    FUSION_OUT.write_text(json.dumps(fusion_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    locked_after = locked_test_opened(ROOT)
    print(
        json.dumps(
            {
                "records_used": n_records,
                "records_excluded": excluded,
                "overall_neural_top1": overall["neural_only_top1"],
                "overall_classical_top1": overall["classical_only_top1"],
                "overall_fused_top1": overall["fused_top1"],
                "fusion_harms_total": fusion_harms_total,
                "pattern_neural_good_hybrid_bad_holds": pattern_neural_good_hybrid_bad_holds,
                "locked_test_opened_after": locked_after,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
