"""physics-informed-localizer-scale-validation (EXPERIMENTAL, NON-RELEASE):
paired statistical analysis of `run_experiment.py`'s fresh-seed evaluation
outputs.

Thin wrapper around `physics_informed_localizer_validation.analyze_results`
(imported, not reimplemented): reuses its `paired_bootstrap` convention
(2000 resamples, deterministic bootstrap seed 20260826, 90% percentile
interval -- "HydroSwarm's established convention", unchanged), its per-seed
metric-table/subgroup/paired-transition logic, and its cross-seed pooling
logic, all unmodified. Only the module-level `RESULTS_ROOT` and `ARM_NAMES`
are retargeted to this branch's own 3-arm / 3-fresh-seed results directory,
plus one addition: `required_pairwise_comparisons` is called with this
branch's own required comparison set (`C1_C2 vs A_CONTROL`, `C2 vs
A_CONTROL`, `C1_C2 vs C2` -- the task's three key comparisons) instead of
the completed branch's 8-way C-family grid.

Produces (all under
reports/evaluation/physics-informed-localizer-scale-validation/):
  - seed-<seed>/metric-table.{json,md}, centrality-subgroups.json,
    distance-subgroups.json, subgroup-paired-bootstrap.json,
    paired-transitions.json (per fresh seed)
  - pooled/cross-seed-summary.json, pooled/pooled-paired-bootstrap.json,
    pooled/pooled-subgroup-bootstrap.json,
    pooled/required-pairwise-comparisons.json (C1_C2 vs A_CONTROL, C2 vs
    A_CONTROL, C1_C2 vs C2), pooled/parameter-counts.json
  - pooled/cross-study-six-seed-summary.json: a separate, explicitly-
    labeled DESCRIPTIVE meta-summary combining the completed branch's 3
    committed seeds with this branch's 3 fresh seeds, for A_CONTROL and C2
    ONLY (the only two arms with six-seed data available -- C1_C2 has no
    prior-study seeds and is never given six-seed treatment).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_validation"))
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental"))

import analyze_results as base  # noqa: E402  (physics_informed_localizer_validation's own module)

EXPERIMENT_NAME = "physics-informed-localizer-scale-validation"
RESULTS_ROOT = ROOT / "reports" / "evaluation" / EXPERIMENT_NAME
ARM_NAMES = ("A_CONTROL", "C2", "C1_C2")

# Retarget the imported module's own globals (every function in it closes
# over these via the module's __dict__ at call time) so all reads/writes
# land under THIS branch's own results directory, never the completed
# branch's `physics-informed-localizer-validation` one.
base.RESULTS_ROOT = RESULTS_ROOT
base.ARM_NAMES = ARM_NAMES

#: This branch's own three key comparisons (task Section "Key comparisons"),
#: replacing the completed branch's 8-way C-family grid. Convention
#: unchanged: (baseline, comparison), "observed" sign is
#: comparison-minus-baseline.
REQUIRED_PAIRWISE_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("A_CONTROL", "C1_C2"),
    ("A_CONTROL", "C2"),
    ("C2", "C1_C2"),
)
base.REQUIRED_PAIRWISE_COMPARISONS = REQUIRED_PAIRWISE_COMPARISONS

#: The completed validation branch's own 3 pre-declared seeds, whose
#: per-seed evaluation.json files are read (never written) to build the
#: descriptive six-seed A_CONTROL/C2 meta-summary. Immutable prior-study
#: data.
PRIOR_STUDY_RESULTS_ROOT = ROOT / "reports" / "evaluation" / "physics-informed-localizer-validation"
PRIOR_STUDY_SEEDS: tuple[int, ...] = (20260814, 20260901, 20260915)
FRESH_SEEDS: tuple[int, ...] = (20260929, 20261013, 20261027)


def _load_prior_evaluation(seed: int, arm: str) -> dict[str, Any] | None:
    path = PRIOR_STUDY_RESULTS_ROOT / f"seed-{seed}" / f"{arm.lower()}-evaluation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cross_study_six_seed_summary() -> dict[str, Any]:
    """DESCRIPTIVE ONLY (not a new confirmatory pooled bootstrap): for
    A_CONTROL and C2, combine the prior study's 3 committed seeds with this
    branch's 3 fresh seeds into a single six-seed table (per-seed Top-1/
    Top-3/MRR on ood-UNSEEN_TOPOLOGY plus mean/median/stdev/sign-count),
    explicitly labeled by which seeds come from which study. C1_C2 has no
    prior-study data and is intentionally excluded here -- reported
    separately, fresh-3-seed-only, elsewhere in this branch's outputs."""

    result: dict[str, Any] = {"population": "ood-UNSEEN_TOPOLOGY", "arms": {}}
    for arm in ("A_CONTROL", "C2"):
        per_seed: dict[str, Any] = {}
        for seed in PRIOR_STUDY_SEEDS:
            evaluation = _load_prior_evaluation(seed, arm)
            pop = evaluation["populations"]["ood-UNSEEN_TOPOLOGY"] if evaluation else None
            per_seed[str(seed)] = {
                "study": "physics-informed-localizer-validation (prior, committed, immutable)",
                "top1": pop["top1"] if pop else None,
                "top3": pop["top3"] if pop else None,
                "mrr": pop["mrr"] if pop else None,
            }
        for seed in FRESH_SEEDS:
            evaluation = base.load_evaluation(seed, arm)
            pop = evaluation["populations"]["ood-UNSEEN_TOPOLOGY"] if evaluation else None
            per_seed[str(seed)] = {
                "study": "physics-informed-localizer-scale-validation (this branch, fresh)",
                "top1": pop["top1"] if pop else None,
                "top3": pop["top3"] if pop else None,
                "mrr": pop["mrr"] if pop else None,
            }
        top1_values = [entry["top1"] for entry in per_seed.values() if entry["top1"] is not None]
        control_top1_by_seed = {}
        for seed in list(PRIOR_STUDY_SEEDS) + list(FRESH_SEEDS):
            ctrl_eval = _load_prior_evaluation(seed, "A_CONTROL") if seed in PRIOR_STUDY_SEEDS else base.load_evaluation(seed, "A_CONTROL")
            ctrl_pop = ctrl_eval["populations"]["ood-UNSEEN_TOPOLOGY"] if ctrl_eval else None
            control_top1_by_seed[seed] = ctrl_pop["top1"] if ctrl_pop else None
        deltas = []
        for seed in list(PRIOR_STUDY_SEEDS) + list(FRESH_SEEDS):
            top1 = per_seed[str(seed)]["top1"]
            ctrl = control_top1_by_seed[seed]
            if top1 is not None and ctrl is not None:
                deltas.append(top1 - ctrl)
        import statistics as _statistics

        result["arms"][arm] = {
            "per_seed": per_seed,
            "n_seeds": len(top1_values),
            "top1_mean": _statistics.fmean(top1_values) if top1_values else None,
            "top1_median": _statistics.median(top1_values) if top1_values else None,
            "top1_stdev": _statistics.stdev(top1_values) if len(top1_values) > 1 else (0.0 if top1_values else None),
            "top1_delta_vs_control_mean": (_statistics.fmean(deltas) if deltas else None) if arm != "A_CONTROL" else 0.0,
            "n_seeds_positive_delta_vs_control": (sum(1 for d in deltas if d > 0) if arm != "A_CONTROL" else None),
            "n_seeds_negative_delta_vs_control": (sum(1 for d in deltas if d < 0) if arm != "A_CONTROL" else None),
        }
    result["note"] = (
        "DESCRIPTIVE cross-study pooling only, not a new confirmatory statistical test: "
        "the 3 prior-study seeds and 3 fresh seeds were run under the completed "
        "physics-informed-localizer-validation branch and this branch respectively, at "
        "different times, and are combined here only to describe six-seed consistency for "
        "A_CONTROL/C2. C1_C2 has no prior-study seeds and is deliberately not given a "
        "six-seed summary; its confirmatory evidence is fresh-3-seed-only "
        "(see pooled-paired-bootstrap.json / cross-seed-summary.json in this branch's own "
        "results directory)."
    )
    return result


def main() -> None:
    base.main()

    pooled_dir = RESULTS_ROOT / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    six_seed = cross_study_six_seed_summary()
    (pooled_dir / "cross-study-six-seed-summary.json").write_text(
        json.dumps(six_seed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote pooled/cross-study-six-seed-summary.json (descriptive, A_CONTROL/C2 only) under {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
