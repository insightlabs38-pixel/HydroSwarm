"""Phase 4: paired CONTROL vs EXPERIMENTAL_TOPOLOGY_RELATIVE per-example
analysis on the unseen-topology (coastal-branch) population (branch
exp/failure-mode-diagnostics).

Consumes the per-example rows produced by
rerun_topology_pilot_with_logging.py (a fresh re-run of exp/topology-
generalization's exact protocol, since the original pilot never persisted
per-example predictions or checkpoints -- see that script's docstring).
Does not retrain anything itself; purely descriptive/statistical.

Usage: python3 scripts/hydrocore_v5_experimental/failure_mode_diagnostics/analyze_paired_pilot.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ROWS_DIR = ROOT / "reports" / "evaluation" / "failure-mode-diagnostics" / "pilot-rerun"
OUTPUT_PATH = ROOT / "reports" / "evaluation" / "failure-mode-diagnostics" / "paired-pilot-analysis.json"

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_INTERVAL = 0.90
POPULATION = "ood-UNSEEN_TOPOLOGY"


def load_rows(arm: str) -> dict[str, dict[str, Any]]:
    path = ROWS_DIR / f"{arm.lower()}-{POPULATION}-rows.jsonl"
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {row["scenario_id"]: row for row in rows}


def paired_bootstrap(control: list[float], experimental: list[float]) -> dict[str, Any]:
    control_arr = np.asarray(control)
    experimental_arr = np.asarray(experimental)
    observed = float(np.mean(experimental_arr) - np.mean(control_arr))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(control_arr)
    deltas = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, n, size=n)
        deltas[index] = experimental_arr[sample].mean() - control_arr[sample].mean()
    alpha = (1 - BOOTSTRAP_INTERVAL) / 2
    lower, upper = np.quantile(deltas, [alpha, 1 - alpha])
    return {
        "observed": observed,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n": n,
        "resamples": BOOTSTRAP_RESAMPLES,
        "excludes_zero": bool(lower > 0 or upper < 0),
    }


def transition_table(control: dict[str, dict], experimental: dict[str, dict], field: str) -> dict[str, int]:
    ids = sorted(control)
    counts = {"control_correct_experimental_correct": 0, "control_correct_experimental_wrong": 0,
              "control_wrong_experimental_correct": 0, "control_wrong_experimental_wrong": 0}
    for scenario_id in ids:
        control_ok = bool(control[scenario_id][field])
        experimental_ok = bool(experimental[scenario_id][field])
        key = f"control_{'correct' if control_ok else 'wrong'}_experimental_{'correct' if experimental_ok else 'wrong'}"
        counts[key] += 1
    return counts


def classify_regime(top1_transitions: dict[str, int], margin_deltas: list[float], probability_identical_fraction: float) -> dict[str, Any]:
    n = sum(top1_transitions.values())
    net_gain = top1_transitions["control_wrong_experimental_correct"] - top1_transitions["control_correct_experimental_wrong"]
    churn = top1_transitions["control_wrong_experimental_correct"] + top1_transitions["control_correct_experimental_wrong"]
    mean_abs_margin_delta = statistics.fmean(abs(value) for value in margin_deltas) if margin_deltas else 0.0

    if probability_identical_fraction > 0.95:
        verdict = "A_LARGELY_IGNORED"
        rationale = f"{probability_identical_fraction:.1%} of examples have bit-identical top1 AND top3 status between arms; representation change has near-zero behavioral effect on this population"
    elif churn == 0:
        verdict = "A_LARGELY_IGNORED"
        rationale = "zero top1 transitions in either direction despite non-identical probabilities -- reweights probabilities without changing any decision"
    elif net_gain == 0 and churn > 0:
        verdict = "B_INFLUENTIAL_BUT_NOISY"
        rationale = f"{churn} examples flip top1 status but gains and losses exactly cancel ({churn // 2} each way or asymmetric-but-net-zero); representation is doing something, just not something net-useful here"
    elif net_gain < 0:
        verdict = "D_SYSTEMATICALLY_HARMFUL"
        rationale = f"net top1 change is negative ({net_gain}); more examples flip correct->wrong than wrong->correct"
    elif net_gain > 0:
        verdict = "C_BENEFICIAL_IN_THIS_REGIME"
        rationale = f"net top1 change is positive ({net_gain}) on this population"
    else:
        verdict = "E_INCONCLUSIVE"
        rationale = "no single pattern dominates"
    return {
        "verdict": verdict,
        "rationale": rationale,
        "n": n,
        "net_top1_gain": net_gain,
        "top1_churn": churn,
        "mean_abs_margin_delta": mean_abs_margin_delta,
    }


def main() -> None:
    control = load_rows("CONTROL")
    experimental = load_rows("EXPERIMENTAL_TOPOLOGY_RELATIVE")
    shared_ids = sorted(set(control) & set(experimental))
    if set(control) != set(experimental):
        raise ValueError(
            f"CONTROL/EXPERIMENTAL row sets differ: {len(control)} vs {len(experimental)} "
            f"(shared={len(shared_ids)}) -- paired analysis requires the identical example set"
        )
    n = len(shared_ids)

    top1_transitions = transition_table(control, experimental, "top1")
    top3_transitions = transition_table(control, experimental, "top3")

    rank_deltas = [experimental[sid]["true_source_rank"] - control[sid]["true_source_rank"] for sid in shared_ids
                   if control[sid]["true_source_rank"] is not None and experimental[sid]["true_source_rank"] is not None]
    margin_deltas = [experimental[sid]["margin_top1_top2"] - control[sid]["margin_top1_top2"] for sid in shared_ids
                     if control[sid].get("margin_top1_top2") is not None and experimental[sid].get("margin_top1_top2") is not None]
    entropy_deltas = [experimental[sid]["posterior_entropy_bits"] - control[sid]["posterior_entropy_bits"] for sid in shared_ids]

    identical_top1_and_top3 = sum(
        1 for sid in shared_ids
        if control[sid]["top1"] == experimental[sid]["top1"] and control[sid]["top3"] == experimental[sid]["top3"]
    )
    probability_identical_fraction = identical_top1_and_top3 / n

    control_top1 = [control[sid]["top1"] for sid in shared_ids]
    experimental_top1 = [experimental[sid]["top1"] for sid in shared_ids]
    control_top3 = [control[sid]["top3"] for sid in shared_ids]
    experimental_top3 = [experimental[sid]["top3"] for sid in shared_ids]
    control_rr = [control[sid]["reciprocal_rank"] for sid in shared_ids]
    experimental_rr = [experimental[sid]["reciprocal_rank"] for sid in shared_ids]

    # subgroup: does the augmentation help/hurt differently by source degree
    # or graph position bucket? (exploratory -- single unseen topology, so
    # "per-topology" collapses to this one family; reported per source-degree
    # instead, the next-most-granular structural covariate available.)
    by_source_degree: dict[str, Any] = {}
    for degree in sorted({control[sid].get("source_degree") for sid in shared_ids} - {None}):
        subset = [sid for sid in shared_ids if control[sid].get("source_degree") == degree]
        if len(subset) < 5:
            continue
        by_source_degree[str(degree)] = {
            "n": len(subset),
            "control_top1": statistics.fmean(control[sid]["top1"] for sid in subset),
            "experimental_top1": statistics.fmean(experimental[sid]["top1"] for sid in subset),
        }

    result = {
        "population": POPULATION,
        "n": n,
        "top1_transition_table": top1_transitions,
        "top3_transition_table": top3_transitions,
        "bootstrap": {
            "top1_delta": paired_bootstrap(control_top1, experimental_top1),
            "top3_delta": paired_bootstrap(control_top3, experimental_top3),
            "mrr_delta": paired_bootstrap(control_rr, experimental_rr),
        },
        "true_source_rank_delta": {
            "n": len(rank_deltas),
            "mean": statistics.fmean(rank_deltas) if rank_deltas else None,
            "median": statistics.median(rank_deltas) if rank_deltas else None,
            "n_improved": sum(1 for value in rank_deltas if value < 0),
            "n_worsened": sum(1 for value in rank_deltas if value > 0),
            "n_unchanged": sum(1 for value in rank_deltas if value == 0),
        },
        "margin_top1_top2_delta": {
            "n": len(margin_deltas),
            "mean": statistics.fmean(margin_deltas) if margin_deltas else None,
        },
        "posterior_entropy_delta": {
            "n": len(entropy_deltas),
            "mean": statistics.fmean(entropy_deltas) if entropy_deltas else None,
        },
        "fraction_examples_with_identical_top1_and_top3_status": probability_identical_fraction,
        "by_source_degree": by_source_degree,
        "regime_classification": classify_regime(top1_transitions, margin_deltas, probability_identical_fraction),
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["top1_transition_table"], indent=2))
    print(json.dumps(result["regime_classification"], indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
