"""Shared, governed helpers for Milestone 9.6: the final exact-compute-
parity confirmation of the selected HydroCore-S training recipe.

M9.6 is a CONFIRMATORY TRAINING milestone, not an architecture search. It
removes the last methodological limitation in the historical M9.0a
INTERLEAVED evidence -- one seed (`20260814`) stopped at 1200 optimizer
steps (early-stopping) while the other two INTERLEAVED seeds and every
CURRENT seed used 1350 -- AND a second, independent limitation the M9.6
governing prompt also calls out (Section 14): even when a run trains the
full 20 epochs / 1350 steps without early stopping, the shared
`Trainer.fit()`/`train_arm_b2()` code always re-loads the BEST-VALIDATION
checkpoint (not the final-step one) into the model right before exporting
`model-export.safetensors` -- so "matches_arm_a_total_optimizer_steps=True"
in a historical record does NOT imply the *exported* checkpoint reflects
1350 steps of training (e.g. `m8-7-runs/AGE_FIX_ONLY-seed20260815.json`:
`stopped_early=False`, `epochs_completed=20`, but the exported checkpoint's
`global_steps=1275`, the BEST epoch, not the LAST one).

M9.6 fixes BOTH by:
  1. training with `early_stopping_patience=0` (an existing, already-
     validated `TrainingConfig` value -- 0 is falsy, so
     `Trainer.fit()`'s/`train_arm_b2()`'s own
     `if config.early_stopping_patience and stale_epochs >= ...` early-exit
     is a structural no-op; nothing about the training LOOP is touched),
     guaranteeing every arm/seed always completes all 20 epochs = 1350
     optimizer steps; and
  2. after `fit()`/`train_arm_b2()` returns, loading the LAST periodic
     checkpoint (`summary["final_checkpoint"]`, saved every epoch since
     `checkpoint_every_epochs=1` -- with early stopping disabled this is
     guaranteed to be epoch 20's checkpoint) into a FRESH model instance
     and exporting THAT as the canonical M9.6 checkpoint, while still
     preserving the best-validation export path/hash for the record
     (Section 14: "preserve both ... but PREDECLARE which one is
     promotion-authoritative" -- FINAL STEP 1350 is authoritative here).

Neither fix touches `hydroswarm/training/trainer.py`,
`run_m8_7_arm.py`, `run_m9_0_arm_b.py`, or `run_m9_0a_arm_b2.py` in place;
M9.6's own training wrapper scripts import their helpers unmodified and
apply only the `early_stopping_patience=0` config override plus the
final-step re-export, exactly mirroring how M9.0a itself reused M9.0's
helpers without editing them.

M9.6 also reuses `_build_family_pools()` (ARM_B, imported unmodified from
`run_m9_0_arm_b`) and `build_scenario_pool()` (ARM_A, imported unmodified
from `hydroswarm.training.causal_prefix`, exactly as `run_m8_7_arm.py`
uses it) for train/validation data -- these draw from
`run_m7_topology.SEED_BASES`'s `"train"`/`"validation"` roles
(940,000,000-970,000,000) and the causal-prefix module's own
`SPLIT_SEED_RANGES` (~900,000,000-903,999,999) respectively, both already
disjoint from every calibration/development namespace in this repo
(M9.4/M9.5/M9.5R start at 990,000,000+; M9.6's own calibration/development
below starts at 998,000,000+) -- so "fresh M9.6 train/validation seed
ranges...physically disjoint [from calibration/development]" (governing
Section 11) is satisfied BY CONSTRUCTION without inventing a new curriculum
or a new train/validation pool (Section 6: "Do NOT invent a new
curriculum"). This is a documented scope decision, not an oversight.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import m9_4_common as m4  # noqa: E402
import m9_5_common as m5  # noqa: E402
import m9_5r_common as m5r  # noqa: E402

# ---------------------------------------------------------------------------
# Re-exported, UNMODIFIED governance/utility primitives.
# ---------------------------------------------------------------------------

current_commit = m4.current_commit
current_branch = m4.current_branch
assert_locked_test_closed = m4.assert_locked_test_closed
checkpoint_sha256 = m4.checkpoint_sha256
wilson_interval_90 = m4.wilson_interval_90
relative_close = m4.relative_close
full_junction_list = m4.full_junction_list

ROOT_PATH = m4.ROOT
ALL_FAMILY_LOADERS = m4.ALL_FAMILY_LOADERS
FROZEN_BRANCH = m4.FROZEN_BRANCH

SEEDS: tuple[int, ...] = m4.SEEDS
DEPTHS: tuple[int, ...] = m4.DEPTHS
EARLY_DEPTHS = m4.EARLY_DEPTHS
MID_DEPTHS = m4.MID_DEPTHS
MATURE_DEPTHS = m4.MATURE_DEPTHS
assert DEPTHS == (1, 2, 3, 4, 6, 12, 25)

TRAINED_FAMILIES: tuple[str, ...] = m4.TRAINED_FAMILIES  # golden-reference, branched-loop, loop-grid
UNSEEN_DEVELOPMENT_FAMILIES: tuple[str, ...] = m4.UNSEEN_DEVELOPMENT_FAMILIES  # coastal-branch, tree-branch, dense-loop
ALL_FAMILIES: tuple[str, ...] = m4.ALL_FAMILIES
ARM_A_KNOWN_FAMILIES = m4.ARM_A_KNOWN_FAMILIES  # ("golden-reference",)
ARM_B_KNOWN_FAMILIES = m4.ARM_B2_KNOWN_FAMILIES  # TRAINED_FAMILIES

ALPHA = m4.ALPHA
assert ALPHA == 0.1
MINIMUM_GROUP_SIZE = m4.MINIMUM_GROUP_SIZE
assert MINIMUM_GROUP_SIZE == 10
OPERATIONAL_COVERAGE_FLOOR = m4.OPERATIONAL_COVERAGE_FLOOR  # 0.85
NOMINAL_COVERAGE_TARGET = m4.NOMINAL_COVERAGE_TARGET  # 0.90
LOOP_GRID_HARD_SOURCE_PAIRS = m4.LOOP_GRID_HARD_SOURCE_PAIRS

#: Section 15/30: reuse the M9.5R candidate-set-size pathology rule unchanged.
CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD = m5r.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD
assert CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD == 0.8

# ---------------------------------------------------------------------------
# Provenance identities of prior, formally-closed milestones.
# ---------------------------------------------------------------------------

M9_4_CLOSURE_PATH = m5.M9_4_CLOSURE_PATH
M9_4_MANIFEST_PATH = m5.M9_4_MANIFEST_PATH
M9_5_CLOSURE_PATH = m5.M9_5_CLOSURE_PATH
M9_5R_CLOSURE_PATH = m5r.M9_5R_CLOSURE_PATH
M9_5R_MANIFEST_PATH = m5r.M9_5R_MANIFEST_PATH
M9_5R_CODE_COMMIT = "b6e41404eda87a22abeffad12968e89a8b1a8496"
M9_5R_ARTIFACT_COMMIT = "9948f718434851c2dd57b7a0d064d983ffa21b9c"
M9_5R_METADATA_FIX_COMMIT = "4fd80a4f1e39e997e241bf3bc4a135745ce1b1e5"

# ---------------------------------------------------------------------------
# Section 9: exact compute parity -- the central M9.6 requirement.
# ---------------------------------------------------------------------------

TOTAL_OPTIMIZER_STEPS = 1350
#: Trainer.__init__'s own formula for ARM_A (600 golden-only train scenarios
#: / batch_size=2 / grad_accum=4, ceil, x20 epochs) -- ARM_B's scheduler is
#: constructed to match this SAME value exactly (never a per-family
#: recomputation), reusing run_m9_0a_arm_b2.py's own frozen constant.
SCHEDULER_TOTAL_STEPS = 1500
ARM_A_OPTIMIZER_STEPS_PER_EPOCH: tuple[int, ...] = (15, 30, 45, 60) + (75,) * 16
assert sum(ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == TOTAL_OPTIMIZER_STEPS
assert len(ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == 20

ARMS: tuple[str, ...] = ("ARM_A_M9_6", "ARM_B_M9_6")

#: Section 14: canonical checkpoint selection policy, frozen before training.
CANONICAL_CHECKPOINT_POLICY = "FINAL_STEP_1350"
assert CANONICAL_CHECKPOINT_POLICY == "FINAL_STEP_1350"

# ---------------------------------------------------------------------------
# Section 26/17: fresh M9.6 calibration/development seed namespace. Floor
# clear of M9.4 (990.0-991.2M), M9.5 (995.0-996.2M), and M9.5R
# (997.0-997.6M); stride/step give ample headroom for 20 repeats/source over
# the largest family (8 sources).
# ---------------------------------------------------------------------------

M9_6_SEED_BASE_FLOOR = 998_000_000
M9_6_SEED_BASE_STEP = 100_000
M9_6_SOURCE_STRIDE = 10_000
M9_6_ROLES: tuple[str, ...] = ("calibration_m9_6", "development_m9_6")

CALIBRATION_REPEATS_PER_SOURCE = 20
DEVELOPMENT_REPEATS_PER_SOURCE = 20

M9_6_SEED_BASES: dict[tuple[str, str], int] = {}
_next = M9_6_SEED_BASE_FLOOR
for _family in ALL_FAMILIES:  # fixed declared order: golden-reference, branched-loop, loop-grid, coastal-branch, tree-branch, dense-loop
    for _role in M9_6_ROLES:
        M9_6_SEED_BASES[(_family, _role)] = _next
        _next += M9_6_SEED_BASE_STEP

_m9_5r_max_seed_base_ceiling = max(m5r.M9_5R_SEED_BASES.values()) + m5r.M9_5R_SEED_BASE_STEP
assert _m9_5r_max_seed_base_ceiling <= M9_6_SEED_BASE_FLOOR, "M9.6 seed bases must not collide with M9.5R's range"
assert max(m4.m7.SEED_BASES.values()) < M9_6_SEED_BASE_FLOOR, "M9.6 seed bases must not collide with M7's range (also covers M9.6's own reused train/validation pools)"
assert (7 * M9_6_SOURCE_STRIDE + max(CALIBRATION_REPEATS_PER_SOURCE, DEVELOPMENT_REPEATS_PER_SOURCE) - 1) < M9_6_SEED_BASE_STEP


def m9_6_seed_base(family: str, role: str) -> int:
    return M9_6_SEED_BASES[(family, role)]


#: Section 21: predeclared paired-bootstrap constants.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_INTERVAL = 0.90

#: Section 22: predeclared, frozen predictive-generalization gate thresholds
#: (identical semantics/values to M9.4's own gate).
GENERALIZATION_MATURE_DELTA_MUST_BE_POSITIVE = True
GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED = 2
GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP = 5.0
#: Section 23: known-family preservation thresholds.
GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP = 5.0
GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP = 3.0
GUARDRAIL_MAX_MRR_REGRESSION = 0.03

REPORT_DIR = m4.REPORT_DIR
M9_6_DIR = REPORT_DIR / "m9-6"
M9_6_TRAINING_RUNS_DIR = M9_6_DIR / "m9-6-training-runs"
RUN_ROOT_M9_6 = ROOT / "experiments" / "runs" / "hydrocore-v5-causal-m9-6"

M9_6_PROTOCOL_PATH = M9_6_DIR / "m9-6-protocol.json"
M9_6_MANIFEST_PATH = M9_6_DIR / "m9-6-manifest.json"
M9_6_TRAINING_PARITY_PATH = M9_6_DIR / "m9-6-training-parity.json"
M9_6_SOURCE_POLICY_PATH = M9_6_DIR / "m9-6-source-policy.json"
M9_6_DEVELOPMENT_REPRESENTATIVENESS_PATH = M9_6_DIR / "m9-6-development-representativeness.json"
M9_6_CANONICAL_PREDICTIONS_PATH = M9_6_DIR / "m9-6-canonical-predictions.jsonl"
M9_6_DEPTH_METRICS_PATH = M9_6_DIR / "m9-6-depth-metrics.json"
M9_6_FAMILY_METRICS_PATH = M9_6_DIR / "m9-6-family-metrics.json"
M9_6_SOURCE_CONDITIONAL_PATH = M9_6_DIR / "m9-6-source-conditional.json"
M9_6_PAIRED_BOOTSTRAP_PATH = M9_6_DIR / "m9-6-paired-bootstrap.json"
M9_6_KNOWN_FAMILY_GUARDRAILS_PATH = M9_6_DIR / "m9-6-known-family-guardrails.json"
M9_6_CALIBRATION_REPRESENTATIVENESS_PATH = M9_6_DIR / "m9-6-calibration-representativeness.json"
M9_6_CANONICAL_CALIBRATION_PATH = M9_6_DIR / "m9-6-canonical-calibration.jsonl"
M9_6_CALIBRATION_RESULTS_PATH = M9_6_DIR / "m9-6-calibration-results.json"
M9_6_CANDIDATE_SET_ANALYSIS_PATH = M9_6_DIR / "m9-6-candidate-set-analysis.json"
M9_6_LOOP_GRID_J1_PATH = M9_6_DIR / "m9-6-loop-grid-j1.json"
M9_6_GUARDRAILS_PATH = M9_6_DIR / "m9-6-guardrails.json"
M9_6_SUMMARY_PATH = M9_6_DIR / "m9-6-summary.md"
M9_6_CLOSURE_PATH = M9_6_DIR / "m9-6-closure.json"

#: Section 33: frozen decision-category codes/names.
DECISION_NAMES: dict[str, str] = {
    "A": "EXACT_COMPUTE_INTERLEAVED_CONFIRMATION_PASS",
    "B": "PREDICTIVE_GAIN_NOT_CONFIRMED",
    "C": "PREDICTIVE_GAIN_CONFIRMED_CALIBRATION_FAILS",
    "D": "KNOWN_FAMILY_REGRESSION_BLOCKER",
    "E": "COMPUTE_PARITY_OR_TRAINING_PROTOCOL_BLOCKER",
    "F": "CALIBRATION_OR_REPRESENTATIVENESS_BLOCKER",
    "G": "INCONCLUSIVE",
}


def depth_bucket_of(depth: int) -> str:
    if depth in EARLY_DEPTHS:
        return "EARLY"
    if depth in MID_DEPTHS:
        return "MID"
    return "MATURE"


environment_info = m5r.environment_info
