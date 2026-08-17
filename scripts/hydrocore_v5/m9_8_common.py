"""Shared, governed helpers for Milestone 9.8: the HydroCore-S vs
HydroCore-M capacity comparison preregistered in M9.7
(`reports/evaluation/hydrocore-v5/m9-7/m9-7-m9-8-preregistration.json`)
and corrected by M9.7A
(`reports/evaluation/hydrocore-v5/m9-7a/m9-7a-amendment.json`).

M9.8 is a CAPACITY-COMPARISON milestone, not an architecture search: exactly
two arms, ARM_S (frozen HydroCore-S, reusing M9.6's own canonical
FINAL_STEP_1350 checkpoints per M9.7A) and ARM_M (freshly trained
HydroCore-M, identical recipe except capacity, also checkpointed at its own
canonical FINAL_STEP_1350 export). Both arms use the SAME
EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING recipe M9.6 froze for
ARM_B_M9_6 -- unlike M9.6 itself (which compared ARM_A single-family vs
ARM_B interleaved), M9.8 holds the TRAINING RECIPE fixed and varies ONLY
capacity, so both M9.8 arms are "ARM_B-style": both know
TRAINED_FAMILIES = (golden-reference, branched-loop, loop-grid).

Reuses UNMODIFIED re-exports from `m9_6_common`/`m9_4_common` for every
governance primitive, depth grid, family list, and threshold that M9.8 does
NOT redefine -- only the M9.8-specific seed namespace, bootstrap seed,
practical-effect threshold, and decision-category names are new here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import m9_4_common as m4  # noqa: E402
import m9_6_common as m6  # noqa: E402

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
environment_info = m6.environment_info
depth_bucket_of = m6.depth_bucket_of

ROOT_PATH = m4.ROOT
ALL_FAMILY_LOADERS = m4.ALL_FAMILY_LOADERS
FROZEN_BRANCH = m4.FROZEN_BRANCH

SEEDS: tuple[int, ...] = m4.SEEDS
assert SEEDS == (20260814, 31874, 20260815)
DEPTHS: tuple[int, ...] = m4.DEPTHS
EARLY_DEPTHS = m4.EARLY_DEPTHS
MID_DEPTHS = m4.MID_DEPTHS
MATURE_DEPTHS = m4.MATURE_DEPTHS
assert DEPTHS == (1, 2, 3, 4, 6, 12, 25)

TRAINED_FAMILIES: tuple[str, ...] = m4.TRAINED_FAMILIES  # golden-reference, branched-loop, loop-grid
UNSEEN_DEVELOPMENT_FAMILIES: tuple[str, ...] = m4.UNSEEN_DEVELOPMENT_FAMILIES  # coastal-branch, tree-branch, dense-loop
ALL_FAMILIES: tuple[str, ...] = m4.ALL_FAMILIES

#: Both M9.8 arms are trained on the SAME 3 families (unlike M9.6's ARM_A,
#: which was single-family golden-reference-only) -- there is only one
#: "known families" set in M9.8, shared by ARM_S and ARM_M alike.
KNOWN_FAMILIES: tuple[str, ...] = TRAINED_FAMILIES

ALPHA = m4.ALPHA
assert ALPHA == 0.1
MINIMUM_GROUP_SIZE = m4.MINIMUM_GROUP_SIZE
assert MINIMUM_GROUP_SIZE == 10
OPERATIONAL_COVERAGE_FLOOR = m4.OPERATIONAL_COVERAGE_FLOOR  # 0.85
NOMINAL_COVERAGE_TARGET = m4.NOMINAL_COVERAGE_TARGET  # 0.90
LOOP_GRID_HARD_SOURCE_PAIRS = m4.LOOP_GRID_HARD_SOURCE_PAIRS

CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD = m6.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD
assert CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD == 0.8

# ---------------------------------------------------------------------------
# Provenance identities of prior, formally-closed milestones this one
# inherits from (M9.6, M9.7, M9.7A) -- never altered here.
# ---------------------------------------------------------------------------

M9_6_CLOSURE_PATH = m6.M9_6_CLOSURE_PATH
M9_6_TRAINING_RUNS_DIR = m6.M9_6_TRAINING_RUNS_DIR
M9_7_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-7"
M9_7_CLOSURE_PATH = M9_7_DIR / "m9-7-closure.json"
M9_7_PREREGISTRATION_PATH = M9_7_DIR / "m9-7-m9-8-preregistration.json"
M9_7_TRAINING_PARITY_PLAN_PATH = M9_7_DIR / "m9-7-training-parity-plan.json"
M9_7_SELECTED_M_ARCHITECTURE_PATH = M9_7_DIR / "m9-7-selected-m-architecture.json"
M9_7_SEMANTIC_PARITY_AUDIT_PATH = M9_7_DIR / "m9-7-semantic-parity-audit.json"
M9_7A_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-7a"
M9_7A_AMENDMENT_PATH = M9_7A_DIR / "m9-7a-amendment.json"
M9_7A_CLOSURE_PATH = M9_7A_DIR / "m9-7a-closure.json"

# ---------------------------------------------------------------------------
# Frozen architecture identities (M9.7, unchanged here -- reasserted, not
# redefined).
# ---------------------------------------------------------------------------

S_VARIANT = "small"
S_PARAMETER_COUNT = 4_182_612
M_VARIANT = "small_v5_capacity_m"
M_PARAMETER_COUNT = 13_919_572

# ---------------------------------------------------------------------------
# Section 9: exact compute parity -- inherited unchanged from M9.6/M9.7.
# ---------------------------------------------------------------------------

TOTAL_OPTIMIZER_STEPS = 1350
SCHEDULER_TOTAL_STEPS = 1500
ARM_A_OPTIMIZER_STEPS_PER_EPOCH: tuple[int, ...] = m6.ARM_A_OPTIMIZER_STEPS_PER_EPOCH
assert sum(ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == TOTAL_OPTIMIZER_STEPS
assert len(ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == 20

ARMS: tuple[str, ...] = ("ARM_S_M9_8", "ARM_M_M9_8")

#: M9.7A Section 6/10: canonical checkpoint selection policy, authoritative
#: for BOTH arms -- best-validation checkpoints may be saved/reported
#: diagnostically but never determine the M9.8 promotion result.
CANONICAL_CHECKPOINT_POLICY = "FINAL_STEP_1350"
assert CANONICAL_CHECKPOINT_POLICY == m6.CANONICAL_CHECKPOINT_POLICY

# ---------------------------------------------------------------------------
# Section 12/13: FROZEN M9.8 development/calibration seed namespace, taken
# verbatim (not computed via a formula) from the governing M9.8 prompt --
# these exact integers are the frozen pre-execution decision, recorded here
# so every M9.8 script imports the SAME single source of truth.
# ---------------------------------------------------------------------------

M9_8_SOURCE_STRIDE = 10_000
CALIBRATION_REPEATS_PER_SOURCE = 20
DEVELOPMENT_REPEATS_PER_SOURCE = 20

M9_8_DEVELOPMENT_SEED_BASES: dict[str, int] = {
    "golden-reference": 1_000_100_000,
    "branched-loop": 1_000_300_000,
    "loop-grid": 1_000_500_000,
    "coastal-branch": 1_000_700_000,
    "tree-branch": 1_000_900_000,
    "dense-loop": 1_001_100_000,
}
M9_8_CALIBRATION_SEED_BASES: dict[str, int] = {
    "golden-reference": 1_000_000_000,
    "branched-loop": 1_000_200_000,
    "loop-grid": 1_000_400_000,
}
assert set(M9_8_DEVELOPMENT_SEED_BASES) == set(ALL_FAMILIES)
assert set(M9_8_CALIBRATION_SEED_BASES) == set(TRAINED_FAMILIES)

#: Expected incident counts (prompt Sections 12/13), verified against
#: full_junction_list at manifest-freeze time -- 4/7/8/6/5/6 sources x 20 =
#: 720 development, (4+7+8) x 20 = 380 calibration.
EXPECTED_DEVELOPMENT_INCIDENTS_PER_FAMILY: dict[str, int] = {
    "golden-reference": 80, "branched-loop": 140, "loop-grid": 160,
    "coastal-branch": 120, "tree-branch": 100, "dense-loop": 120,
}
EXPECTED_TOTAL_DEVELOPMENT_INCIDENTS = 720
EXPECTED_CALIBRATION_INCIDENTS_PER_FAMILY: dict[str, int] = {
    "golden-reference": 80, "branched-loop": 140, "loop-grid": 160,
}
EXPECTED_TOTAL_CALIBRATION_INCIDENTS = 380


def m9_8_development_seed_base(family: str) -> int:
    return M9_8_DEVELOPMENT_SEED_BASES[family]


def m9_8_calibration_seed_base(family: str) -> int:
    return M9_8_CALIBRATION_SEED_BASES[family]


# ---------------------------------------------------------------------------
# Section 18: predeclared paired-bootstrap constants. bootstrap_seed is
# M9.8's OWN frozen value (m9-7-m9-8-preregistration.json's
# statistical_procedure.bootstrap_seed), distinct from M9.6's 20260818.
# ---------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260819
BOOTSTRAP_INTERVAL = 0.90

# ---------------------------------------------------------------------------
# Section 19/26: predeclared, frozen guardrail thresholds -- taken verbatim
# from m9-7-m9-8-preregistration.json's practical_effect_threshold.rules,
# NOT reused blindly from M9.6 (M9.6's own unseen-family regression bound
# was 5.0pp; M9.8's frozen bound, per its own preregistration guardrail B,
# is 3.0pp -- a genuinely different, independently frozen number).
# ---------------------------------------------------------------------------

PRIMARY_EFFECT_MINIMUM_ABSOLUTE_DELTA = 0.02
GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED = 2
GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP = 3.0
GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP = 5.0
GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP = 3.0
GUARDRAIL_MAX_MRR_REGRESSION = 0.03

REPORT_DIR = m4.REPORT_DIR
M9_8_DIR = REPORT_DIR / "m9-8"
M9_8_TRAINING_RUNS_DIR = M9_8_DIR / "m9-8-training-runs"
RUN_ROOT_M9_8 = ROOT / "experiments" / "runs" / "hydrocore-v5-causal-m9-8"

M9_8_EXECUTION_MANIFEST_PATH = M9_8_DIR / "m9-8-execution-manifest.json"
M9_8_SOURCE_POLICY_PATH = M9_8_DIR / "m9-8-source-policy.json"
M9_8_TRAINING_PARITY_PATH = M9_8_DIR / "m9-8-training-parity.json"
M9_8_DEVELOPMENT_REPRESENTATIVENESS_PATH = M9_8_DIR / "m9-8-development-representativeness.json"
M9_8_CALIBRATION_REPRESENTATIVENESS_PATH = M9_8_DIR / "m9-8-calibration-representativeness.json"
M9_8_CANONICAL_PREDICTIONS_PATH = M9_8_DIR / "m9-8-canonical-predictions.jsonl"
M9_8_CANONICAL_CALIBRATION_PATH = M9_8_DIR / "m9-8-canonical-calibration.jsonl"
M9_8_DEPTH_METRICS_PATH = M9_8_DIR / "m9-8-depth-metrics.json"
M9_8_FAMILY_METRICS_PATH = M9_8_DIR / "m9-8-family-metrics.json"
M9_8_SEED_METRICS_PATH = M9_8_DIR / "m9-8-seed-metrics.json"
M9_8_SOURCE_CONDITIONAL_PATH = M9_8_DIR / "m9-8-source-conditional.json"
M9_8_PAIRED_BOOTSTRAP_PATH = M9_8_DIR / "m9-8-paired-bootstrap.json"
M9_8_KNOWN_FAMILY_GUARDRAILS_PATH = M9_8_DIR / "m9-8-known-family-guardrails.json"
M9_8_CALIBRATION_RESULTS_PATH = M9_8_DIR / "m9-8-calibration-results.json"
M9_8_CANDIDATE_SET_ANALYSIS_PATH = M9_8_DIR / "m9-8-candidate-set-analysis.json"
M9_8_ENGINEERING_COST_PATH = M9_8_DIR / "m9-8-engineering-cost.json"
M9_8_GUARDRAILS_PATH = M9_8_DIR / "m9-8-guardrails.json"
M9_8_SUMMARY_PATH = M9_8_DIR / "m9-8-summary.md"
M9_8_CLOSURE_PATH = M9_8_DIR / "m9-8-closure.json"

#: Section 27: frozen decision-category codes/names.
DECISION_NAMES: dict[str, str] = {
    "A": "HYDROCORE_M_MEANINGFUL_CAPACITY_GAIN_VALIDATED",
    "B": "HYDROCORE_M_NO_MEANINGFUL_CAPACITY_GAIN",
    "C": "HYDROCORE_M_PREDICTIVE_GAIN_BUT_GUARDRAIL_FAILURE",
    "D": "HYDROCORE_M_INCONCLUSIVE",
    "E": "ENGINEERING_OR_COMPARABILITY_BLOCKER",
}

HYDROCORE_L_AUTHORIZED = False
