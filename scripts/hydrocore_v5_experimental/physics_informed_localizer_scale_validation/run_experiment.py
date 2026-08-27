"""physics-informed-localizer-scale-validation (EXPERIMENTAL, NON-RELEASE):
focused confirmation follow-up to the completed `exp/physics-informed-
localizer-validation` branch (final report:
`reports/evaluation/physics-informed-localizer-validation/FINAL_REPORT.md`).

That branch recommended, as its next step (Section 17): drop `C3`
(no measurable Top-1 value, significant Top-3 regression when isolated),
test `C1_C2` (not previously run), and pre-register an effect-size --not
just direction-- replication bar. This script does exactly that and
nothing else: it is NOT another architecture search.

This is a thin wrapper, not a fork: it imports
`physics_informed_localizer_validation.run_experiment` unmodified and
reuses its `ARMS` registry (which already defines `C1_C2` with identical
`model_kwargs` to `C2`/`C_FULL`/`C1`/`C3`, per that branch's own Phase 5
requirement), its `_mask_physics_columns` ablation mechanism, its dataset/
training/evaluation functions, and its harness structure (stratified
family sampling, OODDetector/SplitConformalCalibrator reuse, proxy
actionable/abstention metrics, per-row logging) byte-for-byte. Only the
module-level configuration is retargeted here, before any run: the arm
subset (`A_CONTROL`, `C2`, `C1_C2` only -- `C_FULL`/`C3`/
`B_CANDIDATE_CONDITIONED`/`A_CAPACITY_MATCHED` already have their answers
from the completed branch and are not re-run), a disjoint set of 3 fresh
pre-declared seeds, and a new, separate results/run root so the completed
branch's own seed directories, manifests, and reports are never touched or
overwritten.

Usage:
  python3 scripts/hydrocore_v5_experimental/physics_informed_localizer_scale_validation/run_experiment.py \\
      --seed 20260929 --arms A_CONTROL,C2,C1_C2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_validation"))
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental"))

import run_experiment as base  # noqa: E402  (physics_informed_localizer_validation's own module)

#: Phase 0 pre-registration (see the plan doc): three fresh, independent
#: confirmation seeds, disjoint from the completed branch's own
#: (20260814, 20260901, 20260915). Fixed before any training on this
#: branch; never selected or replaced based on results.
SEEDS: tuple[int, ...] = (20260929, 20261013, 20261027)

#: Exactly the three priority arms the task calls for. `C_FULL`, `C3`,
#: `B_CANDIDATE_CONDITIONED`, and `A_CAPACITY_MATCHED` are answered
#: questions from the completed branch and are deliberately not re-run.
PRIORITY_ORDER: tuple[str, ...] = ("A_CONTROL", "C2", "C1_C2")

EXPERIMENT_NAME = "physics-informed-localizer-scale-validation"
RUN_ROOT = ROOT / "experiments" / EXPERIMENT_NAME / "runs"
RESULTS_ROOT = ROOT / "reports" / "evaluation" / EXPERIMENT_NAME

# Retarget the imported module's own globals so every function inside it
# (train_arm/evaluate_arm/run_seed, all of which close over these names at
# call time via the module's __dict__) writes/reads from THIS branch's own
# seeds and result/run roots, never the completed branch's own
# `physics-informed-localizer-validation` directories.
base.SEEDS = SEEDS
base.RUN_ROOT = RUN_ROOT
base.RESULTS_ROOT = RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="single seed to run; default: all of SEEDS")
    parser.add_argument("--arms", type=str, default=None, help="comma-separated arm names; default: PRIORITY_ORDER")
    args = parser.parse_args()
    arm_names = args.arms.split(",") if args.arms else list(PRIORITY_ORDER)
    for name in arm_names:
        if name not in base.ARMS:
            raise SystemExit(f"unknown arm {name!r}; choices: {sorted(base.ARMS)}")
        if name not in PRIORITY_ORDER:
            raise SystemExit(
                f"arm {name!r} is out of scope for physics-informed-localizer-scale-validation "
                f"(only {PRIORITY_ORDER} are run on this branch; that question is already answered "
                "by the completed physics-informed-localizer-validation branch)"
            )

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for seed in seeds:
        if seed not in SEEDS:
            raise SystemExit(f"seed {seed} is not in the pre-declared fresh SEEDS list {SEEDS}")
        base.run_seed(seed, arm_names)


if __name__ == "__main__":
    main()
