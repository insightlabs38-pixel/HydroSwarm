"""physics-informed-localizer-scale-validation (EXPERIMENTAL, NON-RELEASE):
the task's explicit "Top-3 tension analysis" -- does `C1_C2` change the
completed validation study's `C2`-specific finding (strong positive Top-1,
positive MRR, small negative Top-3 point estimate whose CI crossed zero)?

`analyze_results.py`'s `pooled_paired_bootstrap`/`required_pairwise_
comparisons` already give C1_C2/C2 vs A_CONTROL on Top-1/Top-3/MRR, and
C1_C2 vs C2 on Top-1 -- this script adds exactly the pieces those don't
cover, reusing the same loaded rows / `paired_bootstrap` convention
(2000 resamples, seed 20260826, 90% CI) rather than a new statistic:

  - C1_C2 vs C2 head-to-head on Top-3 and MRR (pooled across all 3 fresh
    seeds, matched by (seed, scenario_id));
  - paired Top-3 transition counts between C2 and C1_C2 directly (not just
    each vs A_CONTROL);
  - count of examples where C1_C2 gets Top-1 right but the true source
    still falls outside its own Top-3 (a real, if narrow, residual
    tension even when Top-1 improves);
  - count of examples where C2's true-source rank is WORSE than
    A_CONTROL's (C2 "harmed" that example's ranking) but C1_C2's rank on
    the same example is better than C2's (C1 "recovers" what C2 harmed);
  - true-source-rank distribution (mean/median/stdev) per arm, pooled.

Writes reports/evaluation/physics-informed-localizer-scale-validation/
pooled/top3-tension-analysis.json.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent


def _load_scale_analyze_results():
    """Loads this experiment's own `analyze_results.py` (the thin wrapper
    that retargets `RESULTS_ROOT`/`ARM_NAMES`/`REQUIRED_PAIRWISE_
    COMPARISONS` onto the completed validation branch's own
    `analyze_results` module) under a private module name. A plain
    `import analyze_results` here would self-collide: that wrapper module
    is ALSO named `analyze_results.py` and itself does `import
    analyze_results as base` to reach the completed branch's module --
    importing it under the same generic name from a second file makes
    Python register the not-yet-initialized wrapper in `sys.modules`
    before that inner import line runs, so `base` would silently become
    the wrapper itself rather than the real completed-branch module."""

    spec = importlib.util.spec_from_file_location(
        "physics_informed_localizer_scale_validation_analyze_results", THIS_DIR / "analyze_results.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scale_analysis = _load_scale_analyze_results()
base = scale_analysis.base
RESULTS_ROOT = scale_analysis.RESULTS_ROOT
SEEDS: tuple[int, ...] = scale_analysis.FRESH_SEEDS
POPULATION = "ood-UNSEEN_TOPOLOGY"


def _matched_rows(seed: int, arm_a: str, arm_b: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows_a = {row["scenario_id"]: row for row in base.load_rows(seed, arm_a, POPULATION) if row.get("has_source")}
    rows_b = {row["scenario_id"]: row for row in base.load_rows(seed, arm_b, POPULATION) if row.get("has_source")}
    shared = sorted(set(rows_a) & set(rows_b))
    return [(rows_a[sid], rows_b[sid]) for sid in shared]


def c1_c2_vs_c2_top3_mrr() -> dict[str, Any]:
    c2_top3: list[float] = []
    c1c2_top3: list[float] = []
    c2_mrr: list[float] = []
    c1c2_mrr: list[float] = []
    per_seed_n: dict[str, int] = {}
    for seed in SEEDS:
        pairs = _matched_rows(seed, "C2", "C1_C2")
        per_seed_n[str(seed)] = len(pairs)
        for c2_row, c1c2_row in pairs:
            c2_top3.append(float(c2_row["top3"]))
            c1c2_top3.append(float(c1c2_row["top3"]))
            c2_mrr.append(float(c2_row["reciprocal_rank"]))
            c1c2_mrr.append(float(c1c2_row["reciprocal_rank"]))
    return {
        "population": POPULATION,
        "per_seed_n": per_seed_n,
        "total_n": len(c2_top3),
        "top3_delta_c1_c2_minus_c2": base.paired_bootstrap(c2_top3, c1c2_top3),
        "mrr_delta_c1_c2_minus_c2": base.paired_bootstrap(c2_mrr, c1c2_mrr),
    }


def c1_c2_vs_c2_top3_transitions() -> dict[str, Any]:
    table = {"both_correct": 0, "c2_only": 0, "c1_c2_only": 0, "both_wrong": 0}
    per_seed_n: dict[str, int] = {}
    for seed in SEEDS:
        pairs = _matched_rows(seed, "C2", "C1_C2")
        per_seed_n[str(seed)] = len(pairs)
        for c2_row, c1c2_row in pairs:
            if c2_row["top3"] and c1c2_row["top3"]:
                table["both_correct"] += 1
            elif c2_row["top3"] and not c1c2_row["top3"]:
                table["c2_only"] += 1
            elif not c2_row["top3"] and c1c2_row["top3"]:
                table["c1_c2_only"] += 1
            else:
                table["both_wrong"] += 1
    return {"population": POPULATION, "per_seed_n": per_seed_n, "top3_transition_table": table, "net": table["c1_c2_only"] - table["c2_only"]}


def top1_improved_but_outside_top3() -> dict[str, Any]:
    """Examples where C1_C2 gets Top-1 right but the true source still
    falls outside C1_C2's own Top-3 -- impossible in principle (a correct
    Top-1 is trivially inside Top-3 whenever k>=1), included here as an
    explicit sanity check the task's own wording implies should be
    checked; also reports, more usefully, examples where C1_C2 improves
    Top-1 OVER A_CONTROL specifically while its own Top-3 is wrong."""

    count_top1_correct_top3_wrong = 0
    count_top1_gain_but_top3_wrong = 0
    per_seed: dict[str, dict[str, int]] = {}
    for seed in SEEDS:
        control_rows = {row["scenario_id"]: row for row in base.load_rows(seed, "A_CONTROL", POPULATION) if row.get("has_source")}
        c1c2_rows = {row["scenario_id"]: row for row in base.load_rows(seed, "C1_C2", POPULATION) if row.get("has_source")}
        shared = sorted(set(control_rows) & set(c1c2_rows))
        seed_top1_correct_top3_wrong = 0
        seed_top1_gain_but_top3_wrong = 0
        for sid in shared:
            c1c2_row = c1c2_rows[sid]
            control_row = control_rows[sid]
            if c1c2_row["top1"] and not c1c2_row["top3"]:
                seed_top1_correct_top3_wrong += 1
            if c1c2_row["top1"] and not control_row["top1"] and not c1c2_row["top3"]:
                seed_top1_gain_but_top3_wrong += 1
        per_seed[str(seed)] = {
            "top1_correct_top3_wrong": seed_top1_correct_top3_wrong,
            "top1_gain_over_control_but_top3_wrong": seed_top1_gain_but_top3_wrong,
        }
        count_top1_correct_top3_wrong += seed_top1_correct_top3_wrong
        count_top1_gain_but_top3_wrong += seed_top1_gain_but_top3_wrong
    return {
        "population": POPULATION,
        "per_seed": per_seed,
        "total_top1_correct_top3_wrong": count_top1_correct_top3_wrong,
        "total_top1_gain_over_control_but_top3_wrong": count_top1_gain_but_top3_wrong,
        "note": (
            "top1_correct_top3_wrong should always be 0 by construction (a correct Top-1 "
            "prediction is trivially within Top-3 whenever there are >=1 candidates); reported "
            "as an explicit sanity check, not a real tension. The second count is the ecologically "
            "relevant one but is also expected near-zero for the same reason -- Top-1 correctness "
            "structurally implies Top-3 correctness in this harness's `localization_top_k` metric."
        ),
    }


def c1_recovers_ranking_harmed_by_c2() -> dict[str, Any]:
    """For each example present in all three arms: 'C2 harmed' = C2's
    true-source rank is worse (numerically higher) than A_CONTROL's on
    that same example; 'C1 recovers' = C1_C2's rank on that example is
    better (numerically lower) than C2's. Reports both the recovery count
    and how many of those are fully recovered (C1_C2's rank <= A_CONTROL's
    original rank, not just improved over C2)."""

    per_seed: dict[str, dict[str, int]] = {}
    total_harmed = 0
    total_recovered = 0
    total_fully_recovered = 0
    for seed in SEEDS:
        control_rows = {row["scenario_id"]: row for row in base.load_rows(seed, "A_CONTROL", POPULATION) if row.get("has_source")}
        c2_rows = {row["scenario_id"]: row for row in base.load_rows(seed, "C2", POPULATION) if row.get("has_source")}
        c1c2_rows = {row["scenario_id"]: row for row in base.load_rows(seed, "C1_C2", POPULATION) if row.get("has_source")}
        shared = sorted(set(control_rows) & set(c2_rows) & set(c1c2_rows))
        seed_harmed = 0
        seed_recovered = 0
        seed_fully_recovered = 0
        for sid in shared:
            control_rank = control_rows[sid].get("true_source_rank")
            c2_rank = c2_rows[sid].get("true_source_rank")
            c1c2_rank = c1c2_rows[sid].get("true_source_rank")
            if control_rank is None or c2_rank is None or c1c2_rank is None:
                continue
            if c2_rank > control_rank:
                seed_harmed += 1
                if c1c2_rank < c2_rank:
                    seed_recovered += 1
                    if c1c2_rank <= control_rank:
                        seed_fully_recovered += 1
        per_seed[str(seed)] = {"harmed_by_c2": seed_harmed, "recovered_by_c1": seed_recovered, "fully_recovered": seed_fully_recovered}
        total_harmed += seed_harmed
        total_recovered += seed_recovered
        total_fully_recovered += seed_fully_recovered
    return {
        "population": POPULATION,
        "per_seed": per_seed,
        "total_examples_c2_harmed_vs_control": total_harmed,
        "total_recovered_by_c1_partial_or_full": total_recovered,
        "total_fully_recovered_to_at_least_control_rank": total_fully_recovered,
        "recovery_fraction": (total_recovered / total_harmed) if total_harmed else None,
    }


def true_source_rank_distribution() -> dict[str, Any]:
    result: dict[str, Any] = {"population": POPULATION, "by_arm": {}}
    for arm in ("A_CONTROL", "C2", "C1_C2"):
        ranks: list[int] = []
        for seed in SEEDS:
            for row in base.load_rows(seed, arm, POPULATION):
                if row.get("has_source") and row.get("true_source_rank") is not None:
                    ranks.append(int(row["true_source_rank"]))
        result["by_arm"][arm] = {
            "n": len(ranks),
            "mean": statistics.fmean(ranks) if ranks else None,
            "median": statistics.median(ranks) if ranks else None,
            "stdev": statistics.stdev(ranks) if len(ranks) > 1 else (0.0 if ranks else None),
            "rank_1_fraction": (sum(1 for r in ranks if r == 1) / len(ranks)) if ranks else None,
            "rank_le_3_fraction": (sum(1 for r in ranks if r <= 3) / len(ranks)) if ranks else None,
        }
    return result


def main() -> None:
    pooled_dir = RESULTS_ROOT / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "description": (
            "Task-required Top-3 tension analysis: does C1_C2 resolve, worsen, or leave "
            "unchanged the completed validation study's C2-specific finding (strong positive "
            "Top-1/MRR, small negative Top-3 point estimate whose 90% CI crossed zero)?"
        ),
        "c1_c2_vs_c2_top3_and_mrr": c1_c2_vs_c2_top3_mrr(),
        "c1_c2_vs_c2_top3_transitions": c1_c2_vs_c2_top3_transitions(),
        "top1_improved_but_outside_top3": top1_improved_but_outside_top3(),
        "c1_recovers_ranking_harmed_by_c2": c1_recovers_ranking_harmed_by_c2(),
        "true_source_rank_distribution": true_source_rank_distribution(),
    }
    (pooled_dir / "top3-tension-analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote pooled/top3-tension-analysis.json under {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
