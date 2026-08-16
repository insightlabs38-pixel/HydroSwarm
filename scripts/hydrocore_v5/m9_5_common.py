"""Shared, governed helpers for Milestone 9.5: source-representative
calibration-support confirmation study for the frozen CURRENT (ARM_A) and
STEP_MATCHED_INTERLEAVED_MULTI_FAMILY (ARM_B2) HydroCore-S predictors.

M9.5 is a CALIBRATION-SUPPORT / FROZEN-CHECKPOINT study: no predictor is
trained, tuned, fine-tuned, or promoted; architecture, alpha=0.1, and the
B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH SplitConformalCalibrator construction are
unchanged; `locked_final_test`/`locked_topology_test` are never opened. The
ONLY intervention is increasing the number of independent, source-
representative calibration incidents.

This milestone follows up M9.4's finding
(`reports/evaluation/hydrocore-v5/m9-4/m9-4-closure.json`,
`M9_4_DECISION="B"`, `INTERLEAVED_PREDICTIVE_GAIN_CONFIRMED_CALIBRATION_FAILS`)
that the M9.4 calibration pools (4 independent physical incidents/source --
16/28/32 incidents for golden-reference/branched-loop/loop-grid) were too
small: even ARM_A/CURRENT, which historically calibrated successfully,
dropped to ~0.60-0.68 marginal coverage on golden-reference. M9.5 does NOT
edit `m9_4_common.py`/`run_m9_4_source_representative.py`/
`run_m9_4_decide.py` in place (M9.4 evidence must remain reproducible
byte-for-byte); instead it adds a new, parallel, larger-support calibration
population here and in `run_m9_5_source_representative.py`/
`run_m9_5_decide.py`, reusing M9.4's full-source-enumeration policy and the
UNCHANGED `SplitConformalCalibrator`.
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

# ---------------------------------------------------------------------------
# Re-exported, UNMODIFIED governance/utility primitives from M9.4 (read-only
# reuse -- never redefine calibration/alpha/threshold semantics here).
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
RUNS_M8_7 = m4.RUNS_M8_7
RUNS_M9_0A = m4.RUNS_M9_0A

FROZEN_BRANCH = m4.FROZEN_BRANCH
M9_4_CODE_COMMIT = "f2e7857f00be6e33420439f44b6ededa0e6c396f"
M9_4_METADATA_FIX_COMMIT = "3c575ba87bc3f495d33da68c2c5add1b2d257de2"
M9_4_CLOSURE_PATH = m4.M9_4_CLOSURE_PATH
M9_4_MANIFEST_PATH = m4.M9_4_MANIFEST_PATH

SEEDS: tuple[int, ...] = m4.SEEDS
DEPTHS: tuple[int, ...] = m4.DEPTHS
EARLY_DEPTHS = m4.EARLY_DEPTHS
MID_DEPTHS = m4.MID_DEPTHS
MATURE_DEPTHS = m4.MATURE_DEPTHS

#: Section 9: calibration validity applies ONLY to trained families -- do
#: NOT build nested support levels for unseen families (Section 10:
#: "Do NOT use unseen topology families to fit calibration").
TRAINED_FAMILIES: tuple[str, ...] = m4.TRAINED_FAMILIES  # golden-reference, branched-loop, loop-grid
ARM_A_KNOWN_FAMILIES = m4.ARM_A_KNOWN_FAMILIES  # ("golden-reference",)
ARM_B2_KNOWN_FAMILIES = m4.ARM_B2_KNOWN_FAMILIES  # TRAINED_FAMILIES

ALPHA = m4.ALPHA
assert ALPHA == 0.1
MINIMUM_GROUP_SIZE = m4.MINIMUM_GROUP_SIZE
assert MINIMUM_GROUP_SIZE == 10
OPERATIONAL_COVERAGE_FLOOR = m4.OPERATIONAL_COVERAGE_FLOOR  # 0.85
NOMINAL_COVERAGE_TARGET = m4.NOMINAL_COVERAGE_TARGET  # 0.90

ARM_A_TOTAL_OPTIMIZER_STEPS = m4.ARM_A_TOTAL_OPTIMIZER_STEPS
ARM_B2_TOTAL_OPTIMIZER_STEPS_BY_SEED = m4.ARM_B2_TOTAL_OPTIMIZER_STEPS_BY_SEED
LOOP_GRID_HARD_SOURCE_PAIRS = m4.LOOP_GRID_HARD_SOURCE_PAIRS

# ---------------------------------------------------------------------------
# Section 5/6: predeclared, nested calibration-support levels. 20 is the
# ONLY promotion-relevant level; 4/8/12 are diagnostic-only. Frozen BEFORE
# any M9.5 result is viewed.
# ---------------------------------------------------------------------------

SUPPORT_LEVELS: tuple[int, ...] = (4, 8, 12, 20)
PRIMARY_SUPPORT = 20
assert PRIMARY_SUPPORT == max(SUPPORT_LEVELS)
assert list(SUPPORT_LEVELS) == sorted(SUPPORT_LEVELS)

#: Section 8: development support -- 20/source preferred if practical
#: (attempted first; see run_m9_5_source_representative.py's runtime-
#: feasibility note in the module docstring for the go/no-go check actually
#: applied before committing to this value).
DEVELOPMENT_REPEATS_PER_SOURCE = 20

# ---------------------------------------------------------------------------
# Section 14: quantile-stability bootstrap (distinct seed from M9.4's
# paired-bootstrap seed 20260816 and M9.0a's 20260815 -- independently
# predeclared per the governing M9.5 prompt).
# ---------------------------------------------------------------------------

QUANTILE_BOOTSTRAP_RESAMPLES = 2000
QUANTILE_BOOTSTRAP_SEED = 20260817

# ---------------------------------------------------------------------------
# Section 4/7/8: M9.5-only seed-base scheme. TWO roles per TRAINED family
# only -- `calibration_m9_5` (nested support levels via repeat<N filtering)
# and `development_m9_5` -- each with its own seed base, stride 10_000/source
# (headroom for repeat in range(20), vs M9.4's stride 1_000 for repeat in
# range(4)). Floor 995_000_000 is clear of M7's ~940M-970M range AND M9.4's
# 990_000_000-991_200_000 range (asserted below).
# ---------------------------------------------------------------------------

M9_5_SEED_BASE_FLOOR = 995_000_000
M9_5_SEED_BASE_STEP = 200_000
M9_5_SOURCE_STRIDE = 10_000
M9_5_ROLES: tuple[str, ...] = ("calibration_m9_5", "development_m9_5")

M9_5_SEED_BASES: dict[tuple[str, str], int] = {}
_next = M9_5_SEED_BASE_FLOOR
for _family in TRAINED_FAMILIES:  # fixed declared order: golden-reference, branched-loop, loop-grid
    for _role in M9_5_ROLES:
        M9_5_SEED_BASES[(_family, _role)] = _next
        _next += M9_5_SEED_BASE_STEP

_m9_4_max_seed_base_ceiling = max(m4.M9_4_SEED_BASES.values()) + m4.M9_4_SEED_BASE_STEP
assert _m9_4_max_seed_base_ceiling <= M9_5_SEED_BASE_FLOOR, "M9.5 seed bases must not collide with M9.4's range"
assert max(m4.m7.SEED_BASES.values()) < M9_5_SEED_BASE_FLOOR, "M9.5 seed bases must not collide with M7's range"
# Headroom check: max offset within one (family, role) block must stay below
# the block step, i.e. source_index * stride + repeat < M9_5_SEED_BASE_STEP.
assert (7 * M9_5_SOURCE_STRIDE + max(SUPPORT_LEVELS) - 1) < M9_5_SEED_BASE_STEP


def m9_5_seed_base(family: str, role: str) -> int:
    return M9_5_SEED_BASES[(family, role)]


REPORT_DIR = m4.REPORT_DIR
M9_5_DIR = REPORT_DIR / "m9-5"
M9_5_FIGURES_DIR = M9_5_DIR / "figures"
M9_5_MANIFEST_PATH = M9_5_DIR / "m9-5-manifest.json"
M9_5_SOURCE_POLICY_PATH = M9_5_DIR / "m9-5-source-policy.json"
M9_5_REPRESENTATIVENESS_AUDIT_PATH = M9_5_DIR / "m9-5-representativeness-audit.json"
M9_5_CANONICAL_CALIBRATION_PATH = M9_5_DIR / "m9-5-canonical-calibration.jsonl"
M9_5_SUPPORT_CURVE_PATH = M9_5_DIR / "m9-5-support-curve.json"
M9_5_QUANTILE_STABILITY_PATH = M9_5_DIR / "m9-5-quantile-stability.json"
M9_5_CALIBRATION_RESULTS_PATH = M9_5_DIR / "m9-5-calibration-results.json"
M9_5_SOURCE_CONDITIONAL_PATH = M9_5_DIR / "m9-5-source-conditional.json"
M9_5_LOOP_GRID_J1_PATH = M9_5_DIR / "m9-5-loop-grid-j1.json"
M9_5_CANDIDATE_SET_ANALYSIS_PATH = M9_5_DIR / "m9-5-candidate-set-analysis.json"
M9_5_CONTROL_ARM_ANALYSIS_PATH = M9_5_DIR / "m9-5-control-arm-analysis.json"
M9_5_GUARDRAILS_PATH = M9_5_DIR / "m9-5-guardrails.json"
M9_5_SUMMARY_PATH = M9_5_DIR / "m9-5-summary.md"
M9_5_CLOSURE_PATH = M9_5_DIR / "m9-5-closure.json"


def depth_bucket_of(depth: int) -> str:
    if depth in EARLY_DEPTHS:
        return "EARLY"
    if depth in MID_DEPTHS:
        return "MID"
    return "MATURE"


def environment_info() -> dict[str, Any]:
    import platform

    import torch

    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }
