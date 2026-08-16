"""Shared, governed helpers for the Milestone 9.3 calibration root-cause
diagnostic study.

M9.3 is DIAGNOSTIC / ANALYSIS-ONLY: no predictor is trained, tuned, or
promoted; alpha stays 0.1; `locked_final_test`/`locked_topology_test` are
never opened. Constants here are either reused VERBATIM from the frozen
M9.0a/M9.0b machinery (`run_m9_0a_evaluate.py`, `run_m9_0b_evaluate.py`,
`m9_0b_calibration_schemes.py`, `run_m7_topology.py`) or are new M9.3-only
diagnostic constants declared before any result is seen (bootstrap seed,
learning-curve fractions, case-study counts).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

import m9_0b_calibration_schemes as schemes  # noqa: E402
import run_m7_topology as m7  # noqa: E402

FROZEN_BRANCH = "exp/hydrocore-v5-causal"
M9_0A_PROTOCOL_FROZEN_COMMIT = "d8439830f70e9922f4d7d7e94e2378d94a232efe"
M9_0A_RESULTS_COMMIT = "c61dcc19845c8d319178b5cc607c08be0ed10abc"
M9_0B_PROTOCOL_FROZEN_COMMIT = "3b353167d598d76efc6fbde303387388fbc3ccbf"
M9_0B_RESULTS_COMMIT = "c7f7bddba9513e748185cd53fde6c003e7213c79"
M9_2_CLOSURE_COMMIT = "ef3383a939e01b820b66d6be25d08f829ade572d"

ALPHA = schemes.ALPHA
assert ALPHA == 0.1
MINIMUM_GROUP_SIZE = schemes.MINIMUM_GROUP_SIZE
SCHEME_NAMES = schemes.SCHEME_NAMES

SEEDS = (20260814, 31874, 20260815)
EARLY_DEPTHS = (1, 2, 3)
MID_DEPTHS = (4, 6)
MATURE_DEPTHS = (12, 25)
DEPTHS = (1, 2, 3, 4, 6, 12, 25)

KNOWN_FAMILIES = tuple(name for name, _ in m7.TRAINED_FAMILIES)  # golden-reference, branched-loop, loop-grid
UNSEEN_FAMILIES = tuple(name for name, _ in m7.UNSEEN_FAMILIES)  # coastal-branch, tree-branch, dense-loop
ARM_A_KNOWN_FAMILIES = ("golden-reference",)
ARM_B2_KNOWN_FAMILIES = KNOWN_FAMILIES

OPERATIONAL_COVERAGE_FLOOR = 0.85
NOMINAL_COVERAGE_TARGET = 1.0 - ALPHA  # 0.90

#: Section 8/10: predeclared descriptive-uncertainty method (Wilson), and
#: the diagnostic bootstrap seed -- SAME constant M9.2 used for its own
#: diagnostic-only bootstrap, distinct from M9.0a's promotion bootstrap seed.
M9_3_BOOTSTRAP_SEED = 20260816
M9_3_BOOTSTRAP_RESAMPLES = 2000
DESCRIPTIVE_CI_METHOD = "WILSON_90"

#: Section 11: predeclared calibration-support learning-curve fractions.
LEARNING_CURVE_FRACTIONS = (0.25, 0.50, 0.75, 1.0)
LEARNING_CURVE_RESAMPLES_PER_FRACTION = 200
LEARNING_CURVE_SEED = 20260816

#: Section 19: predeclared support tiers for sample-size estimation (diagnostic only).
SUPPORT_TIERS = (25, 50, 100, 200, 500)

#: Section 16: deterministic case-study export count.
CASE_STUDY_TOP_N = 10

REPORT_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5"
M9_0A_RESULTS_PATH = REPORT_DIR / "m9-0a-results.json"
M9_0A_TOPOLOGY_PATH = REPORT_DIR / "m9-0a-topology-generalization.json"
M9_0A_CALIBRATION_PATH = REPORT_DIR / "m9-0a-calibration.json"
M9_0B_RESULTS_PATH = REPORT_DIR / "m9-0b-results.json"
M9_0B_CAL_BY_SEED_PATH = REPORT_DIR / "m9-0b-calibration-by-seed.json"
M9_0B_GROUP_SUPPORT_PATH = REPORT_DIR / "m9-0b-group-support.json"
M9_0B_UNSEEN_TRANSFER_PATH = REPORT_DIR / "m9-0b-unseen-transfer.json"
RUNS_M8_7 = REPORT_DIR / "m8-7-runs"
RUNS_M9_0A = REPORT_DIR / "m9-0a-runs"

M9_3_DIR = REPORT_DIR / "m9-3"
M9_3_FIGURES_DIR = M9_3_DIR / "figures"
M9_3_MANIFEST_PATH = M9_3_DIR / "m9-3-manifest.json"
M9_3_CANONICAL_PATH = M9_3_DIR / "m9-3-canonical-calibration-diagnostics.jsonl"
M9_3_REPRODUCTION_PATH = M9_3_DIR / "m9-3-reproduction.json"
M9_3_SUPPORT_ANALYSIS_PATH = M9_3_DIR / "m9-3-support-analysis.json"
M9_3_SCORE_SHIFT_PATH = M9_3_DIR / "m9-3-score-shift.json"
M9_3_QUANTILE_STABILITY_PATH = M9_3_DIR / "m9-3-quantile-stability.json"
M9_3_LEARNING_CURVES_PATH = M9_3_DIR / "m9-3-support-learning-curves.json"
M9_3_FAMILY_HETEROGENEITY_PATH = M9_3_DIR / "m9-3-family-heterogeneity.json"
M9_3_DEPTH_ANALYSIS_PATH = M9_3_DIR / "m9-3-depth-analysis.json"
M9_3_CONFIDENCE_ANALYSIS_PATH = M9_3_DIR / "m9-3-confidence-analysis.json"
M9_3_SOURCE_CONDITIONAL_PATH = M9_3_DIR / "m9-3-source-conditional.json"
M9_3_MISCOVERAGE_CASES_PATH = M9_3_DIR / "m9-3-miscoverage-cases.json"
M9_3_COUNTERFACTUAL_PATH = M9_3_DIR / "m9-3-counterfactual-diagnostics.json"
M9_3_EXCHANGEABILITY_PATH = M9_3_DIR / "m9-3-exchangeability-audit.json"
M9_3_IMPLEMENTATION_AUDIT_PATH = M9_3_DIR / "m9-3-implementation-audit.json"
M9_3_ROOT_CAUSE_PATH = M9_3_DIR / "m9-3-root-cause.json"
M9_3_SUMMARY_PATH = M9_3_DIR / "m9-3-summary.md"
M9_3_CLOSURE_PATH = M9_3_DIR / "m9-3-closure.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def current_commit() -> str:
    return _git("rev-parse", "HEAD")


def current_branch() -> str:
    return _git("branch", "--show-current")


def assert_locked_test_closed() -> bool:
    opened = locked_test_opened(ROOT)
    if opened:
        raise AssertionError("locked_final_test/locked_topology_test must remain unopened for M9.3")
    return opened


def wilson_interval_90(successes: int, n: int) -> tuple[float, float]:
    """Wilson 90% score interval -- the predeclared descriptive-uncertainty
    method for M9.3 (Section 8), distinct from m9_0b_calibration_schemes's
    95%-only helper. z is the two-sided 90% normal quantile."""

    import math

    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.6448536269514722
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def finite_sample_resolution(n: int) -> float:
    """Section 7: how much one miscovered example moves empirical coverage
    for a group of size n -- 1/(n+1) is the standard split-conformal exact
    finite-sample resolution (matches `_quantile`'s own rank formula,
    ceil((n+1)*(1-alpha)))."""

    return 1.0 / (n + 1) if n > 0 else float("nan")
