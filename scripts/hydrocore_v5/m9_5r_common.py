"""Shared, governed helpers for Milestone 9.5R: independent, one-shot
confirmation of HydroCore-S calibration at the already-predeclared adequate
support level (20 independent physical calibration incidents/source), using
completely fresh, disjoint, source-representative calibration and
development populations for the frozen CURRENT (ARM_A) and STEP_MATCHED_
INTERLEAVED_MULTI_FAMILY (ARM_B2) HydroCore-S predictors.

M9.5R is NOT a reopening or reinterpretation of M9.5
(`reports/evaluation/hydrocore-v5/m9-5/m9-5-closure.json`,
`M9_5_DECISION="E"`, `REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER`), which
remains formally closed and untouched. M9.5R exists because M9.5's formal
decision was blocked by a Section-12 sub-check that incorrectly required a
FRESH small-n calibration draw to qualitatively reproduce M9.4's own
unusually poor small-n coverage pattern -- a stricter reading than M9.5's
governing prompt asked for (see M9.5's own
`section_12_gate_analysis_ADDED_POST_HOC_FOR_TRANSPARENCY` block). M9.5R
replaces that mis-specified reproduction expectation with a corrected sanity
gate that checks ONLY implementation/provenance invariants (Section 11 A-L
below) -- never any particular numerical/coverage outcome -- and asks a
narrower, cleanly falsifiable question against completely fresh data: does
the exact, unmodified conformal calibration method reproduce valid
calibration at exactly 20 independent physical calibration incidents/source?

M9.5R is a CALIBRATION-CONFIRMATION / FROZEN-CHECKPOINT study: no predictor
is trained, tuned, fine-tuned, or promoted; architecture, alpha=0.1, and the
B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH `SplitConformalCalibrator` construction
are unchanged; `locked_final_test`/`locked_topology_test` are never opened.
M9.5R runs EXACTLY ONE calibration-support condition (20 repeats/source) --
it does NOT repeat M9.5's 4/8/12/20 support curve and does NOT sweep support
sizes.

M9.5R does NOT edit `m9_4_common.py`/`m9_5_common.py`/
`run_m9_4_source_representative.py`/`run_m9_5_source_representative.py`/
`run_m9_4_decide.py`/`run_m9_5_decide.py` in place (M9.4/M9.5 evidence must
remain reproducible byte-for-byte and formally closed as-is); instead it
adds a new, parallel, single-support calibration/development population
here and in `run_m9_5r_source_representative.py`/`run_m9_5r_decide.py`,
reusing M9.4's full-source-enumeration policy and the UNCHANGED
`SplitConformalCalibrator` directly (not routed through M9.5's module, to
keep M9.5R's own provenance chain independent of M9.5's specific code path
-- the calibration METHOD is identical; the M9.5R CODE PATH is new).
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

SEEDS: tuple[int, ...] = m4.SEEDS
DEPTHS: tuple[int, ...] = m4.DEPTHS
EARLY_DEPTHS = m4.EARLY_DEPTHS
MID_DEPTHS = m4.MID_DEPTHS
MATURE_DEPTHS = m4.MATURE_DEPTHS

#: Section 9/16: calibration validity applies ONLY to trained families.
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
# Provenance identities of the two prior, formally-closed milestones this
# run confirms/follows up on. Read-only references -- M9.5R never writes
# into M9.4's or M9.5's directories.
# ---------------------------------------------------------------------------

M9_4_CODE_COMMIT = m5.M9_4_CODE_COMMIT
M9_4_METADATA_FIX_COMMIT = m5.M9_4_METADATA_FIX_COMMIT
M9_4_CLOSURE_PATH = m5.M9_4_CLOSURE_PATH
M9_4_MANIFEST_PATH = m5.M9_4_MANIFEST_PATH

M9_5_CODE_COMMIT = "ff79856620c74593e47c069277cb41936e2fc6eb"
M9_5_METADATA_FIX_COMMIT = "b7f4fdaf8474e444a8f7c6d49776fda4a1b6ef4b"
M9_5_CLOSURE_PATH = m5.M9_5_CLOSURE_PATH
M9_5_MANIFEST_PATH = m5.M9_5_MANIFEST_PATH

# ---------------------------------------------------------------------------
# Section 7: EXACTLY ONE primary calibration-support condition. There is no
# support-level tuple/sweep in M9.5R by construction (see governing prompt
# Section 7/24: "no 4/8/12 support sweep exists").
# ---------------------------------------------------------------------------

PRIMARY_SUPPORT = 20
CALIBRATION_REPEATS_PER_SOURCE = PRIMARY_SUPPORT
DEVELOPMENT_REPEATS_PER_SOURCE = 20

# ---------------------------------------------------------------------------
# Section 4/8: M9.5R-only seed-base scheme. TWO roles per TRAINED family --
# `calibration_m9_5r` and `development_m9_5r` -- each with its own seed
# base, stride 10_000/source (headroom for repeat in range(20)). Floor
# 997_000_000 is clear of M9.4's 990_000_000-991_200_000 range AND M9.5's
# 995_000_000-996_200_000 range (asserted below).
# ---------------------------------------------------------------------------

M9_5R_SEED_BASE_FLOOR = 997_000_000
M9_5R_SEED_BASE_STEP = 100_000
M9_5R_SOURCE_STRIDE = 10_000
M9_5R_ROLES: tuple[str, ...] = ("calibration_m9_5r", "development_m9_5r")

M9_5R_SEED_BASES: dict[tuple[str, str], int] = {}
_next = M9_5R_SEED_BASE_FLOOR
for _family in TRAINED_FAMILIES:  # fixed declared order: golden-reference, branched-loop, loop-grid
    for _role in M9_5R_ROLES:
        M9_5R_SEED_BASES[(_family, _role)] = _next
        _next += M9_5R_SEED_BASE_STEP

_m9_5_max_seed_base_ceiling = max(m5.M9_5_SEED_BASES.values()) + m5.M9_5_SEED_BASE_STEP
assert _m9_5_max_seed_base_ceiling <= M9_5R_SEED_BASE_FLOOR, "M9.5R seed bases must not collide with M9.5's range"
_m9_4_max_seed_base_ceiling = max(m4.M9_4_SEED_BASES.values()) + m4.M9_4_SEED_BASE_STEP
assert _m9_4_max_seed_base_ceiling <= m5.M9_5_SEED_BASE_FLOOR, "sanity: M9.5 must not collide with M9.4 either"
assert max(m4.m7.SEED_BASES.values()) < M9_5R_SEED_BASE_FLOOR, "M9.5R seed bases must not collide with M7's range"
# Headroom check: max offset within one (family, role) block must stay below
# the block step, i.e. source_index * stride + repeat < M9_5R_SEED_BASE_STEP.
assert (7 * M9_5R_SOURCE_STRIDE + max(CALIBRATION_REPEATS_PER_SOURCE, DEVELOPMENT_REPEATS_PER_SOURCE) - 1) < M9_5R_SEED_BASE_STEP


def m9_5r_seed_base(family: str, role: str) -> int:
    return M9_5R_SEED_BASES[(family, role)]


REPORT_DIR = m4.REPORT_DIR
M9_5R_DIR = REPORT_DIR / "m9-5r"
M9_5R_PROTOCOL_PATH = M9_5R_DIR / "m9-5r-protocol.json"
M9_5R_MANIFEST_PATH = M9_5R_DIR / "m9-5r-manifest.json"
M9_5R_SOURCE_POLICY_PATH = M9_5R_DIR / "m9-5r-source-policy.json"
M9_5R_REPRESENTATIVENESS_AUDIT_PATH = M9_5R_DIR / "m9-5r-representativeness-audit.json"
M9_5R_SANITY_GATE_PATH = M9_5R_DIR / "m9-5r-sanity-gate.json"
M9_5R_CANONICAL_CALIBRATION_PATH = M9_5R_DIR / "m9-5r-canonical-calibration.jsonl"
M9_5R_CALIBRATION_RESULTS_PATH = M9_5R_DIR / "m9-5r-calibration-results.json"
M9_5R_CONTROL_ARM_PATH = M9_5R_DIR / "m9-5r-control-arm.json"
M9_5R_SOURCE_CONDITIONAL_PATH = M9_5R_DIR / "m9-5r-source-conditional.json"
M9_5R_LOOP_GRID_J1_PATH = M9_5R_DIR / "m9-5r-loop-grid-j1.json"
M9_5R_CANDIDATE_SET_ANALYSIS_PATH = M9_5R_DIR / "m9-5r-candidate-set-analysis.json"
M9_5R_PREDICTIVE_SANITY_PATH = M9_5R_DIR / "m9-5r-predictive-sanity.json"
M9_5R_GUARDRAILS_PATH = M9_5R_DIR / "m9-5r-guardrails.json"
M9_5R_SUMMARY_PATH = M9_5R_DIR / "m9-5r-summary.md"
M9_5R_CLOSURE_PATH = M9_5R_DIR / "m9-5r-closure.json"

#: Section 15: reuse the SAME candidate-set-size pathology rule M9.5 used
#: (full_set_rate > 0.8 at the primary support level) -- frozen here, before
#: results, never invented/changed after seeing data.
CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD = 0.8

#: Section 19: frozen decision-category codes/names (do not alter after
#: seeing results).
DECISION_NAMES: dict[str, str] = {
    "A": "INDEPENDENT_CALIBRATION_CONFIRMATION_PASS",
    "B": "INTERLEAVED_PARTIAL_CALIBRATION_FAILURE",
    "C": "CONTROL_CALIBRATION_FAILURE",
    "D": "REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER",
    "E": "CANDIDATE_SET_GUARD_FAILURE",
    "F": "INCONCLUSIVE",
}


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
