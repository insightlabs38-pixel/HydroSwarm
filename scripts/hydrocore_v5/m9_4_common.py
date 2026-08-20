"""Shared, governed helpers for Milestone 9.4: source-representative,
exchangeability-corrected re-evaluation of the frozen CURRENT (ARM_A) and
STEP_MATCHED_INTERLEAVED_MULTI_FAMILY (ARM_B2) HydroCore-S predictors.

M9.4 is a FROZEN-CHECKPOINT RE-EVALUATION milestone: no predictor is
trained, tuned, fine-tuned, or promoted; architecture, alpha=0.1, and the
B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH calibration construction are unchanged;
`locked_final_test`/`locked_topology_test` are never opened.

This milestone follows up M9.3's root-cause finding
(`reports/evaluation/hydrocore-v5/m9-3/m9-3-closure.json`,
`M9_3_RECOMMENDATION="D"`, `DATASET_EXCHANGEABILITY_OR_GENERATOR_FIX_REQUIRED`):
`run_m7_topology._generate_eval_scenarios` truncates every topology
family's candidate source-junction set to `EVAL_MAX_SOURCES=4` (alphabetical
truncation), so families with >4 junctions (branched-loop 7, loop-grid 8,
coastal-branch 6, tree-branch 5, dense-loop 6) were never evaluated over
their true source population. M9.4 does NOT edit `run_m7_topology.py` in
place (historical M7/M9.0/M9.0a/M9.0b/M9.3 evidence must remain
reproducible byte-for-byte); instead it adds a new, parallel, full-source
scenario-generation policy here and in `run_m9_4_source_representative.py`.
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
M9_3_CLOSURE_COMMIT = "009be2ea822417458596bfe643308a69992aa060"

ALPHA = schemes.ALPHA
assert ALPHA == 0.1
MINIMUM_GROUP_SIZE = schemes.MINIMUM_GROUP_SIZE
assert MINIMUM_GROUP_SIZE == 10

SEEDS: tuple[int, ...] = (20260814, 31874, 20260815)
EARLY_DEPTHS = (1, 2, 3)
MID_DEPTHS = (4, 6)
MATURE_DEPTHS = (12, 25)
DEPTHS = (1, 2, 3, 4, 6, 12, 25)

TRAINED_FAMILIES = tuple(name for name, _ in m7.TRAINED_FAMILIES)  # golden-reference, branched-loop, loop-grid
UNSEEN_DEVELOPMENT_FAMILIES = tuple(name for name, _ in m7.UNSEEN_FAMILIES)  # coastal-branch, tree-branch, dense-loop
ALL_FAMILIES = TRAINED_FAMILIES + UNSEEN_DEVELOPMENT_FAMILIES

ARM_A_KNOWN_FAMILIES = ("golden-reference",)
ARM_B2_KNOWN_FAMILIES = TRAINED_FAMILIES

OPERATIONAL_COVERAGE_FLOOR = 0.85
NOMINAL_COVERAGE_TARGET = 1.0 - ALPHA  # 0.90

#: Section 13: predeclared paired-bootstrap constants (M9.4's OWN bootstrap
#: seed, distinct from M9.0a's promotion-bootstrap seed 20260815 and M9.3's
#: diagnostic-bootstrap seed 20260816 happens to coincide in VALUE with
#: M9.3's -- both were independently predeclared as "2026-08-16"-derived
#: constants; this is intentional per the governing M9.4 prompt, not reused
#: from M9.3's code).
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_INTERVAL = 0.90

#: Section 14/15: predeclared gates, never altered after seeing results.
GENERALIZATION_MATURE_DELTA_MUST_BE_POSITIVE = True
GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED = 2
GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP = 5.0
GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP = 5.0
GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP = 3.0
GUARDRAIL_MAX_MRR_REGRESSION = 0.03

#: Section 4: preferred default repeats per candidate source node, applied
#: to the COMPLETE (untruncated) junction list for every family.
REPEATS_PER_SOURCE = 4

#: Section 8: legacy-reproduction float tolerance. Frozen-checkpoint
#: inference under identical seeds/scenarios/model weights on CPU fp32 is
#: bit-for-bit deterministic (no training, no dropout, no RNG at
#: inference time beyond what the deterministic scenario generator itself
#: consumes), so a tight relative tolerance is appropriate.
LEGACY_REPRODUCTION_RELATIVE_TOLERANCE = 1e-6

#: Section 20: loop-grid confusion pairs called out explicitly by M9.3.
LOOP_GRID_HARD_SOURCE_PAIRS = (("J1", "J7"), ("J1", "J8"), ("J7", "J1"), ("J8", "J1"))

#: Section 2 legacy target figures being bridged against (M9.0a-summary.md,
#: mean over 3 seeds unless noted) -- used only by the legacy-reproduction
#: check, never as a substitute for actually re-running inference.
LEGACY_POOLED_UNSEEN_MATURE_NEURAL_TOP1_GAIN_PP = 6.60
LEGACY_PER_SEED_POOLED_MATURE_DIFFS = (0.0417, 0.0729, 0.0833)
LEGACY_ARM_A_MARGINAL_COVERAGE = (0.9554, 0.9554, 0.9464)
LEGACY_ARM_B2_MARGINAL_COVERAGE = (0.8482, 0.8452, 0.8214)

#: Section 22: known M9.0a optimizer-step-parity limitation, preserved (not
#: fixed here -- M9.4 does not retrain).
ARM_A_TOTAL_OPTIMIZER_STEPS = 1350
ARM_B2_TOTAL_OPTIMIZER_STEPS_BY_SEED = {20260814: 1200, 31874: 1350, 20260815: 1350}

REPORT_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5"
RUNS_M8_7 = REPORT_DIR / "m8-7-runs"
RUNS_M9_0A = REPORT_DIR / "m9-0a-runs"

M9_4_DIR = REPORT_DIR / "m9-4"
M9_4_FIGURES_DIR = M9_4_DIR / "figures"
M9_4_MANIFEST_PATH = M9_4_DIR / "m9-4-manifest.json"
M9_4_SOURCE_POLICY_PATH = M9_4_DIR / "m9-4-source-policy.json"
M9_4_REPRESENTATIVENESS_AUDIT_PATH = M9_4_DIR / "m9-4-representativeness-audit.json"
M9_4_LEGACY_REPRODUCTION_PATH = M9_4_DIR / "m9-4-legacy-reproduction.json"
M9_4_PREDICTIONS_PATH = M9_4_DIR / "m9-4-predictions.jsonl"
M9_4_DEPTH_METRICS_PATH = M9_4_DIR / "m9-4-depth-metrics.json"
M9_4_FAMILY_METRICS_PATH = M9_4_DIR / "m9-4-family-metrics.json"
M9_4_SOURCE_CONDITIONAL_PATH = M9_4_DIR / "m9-4-source-conditional.json"
M9_4_LEGACY_VS_FULL_SOURCE_PATH = M9_4_DIR / "m9-4-legacy-vs-full-source.json"
M9_4_PAIRED_BOOTSTRAP_PATH = M9_4_DIR / "m9-4-paired-bootstrap.json"
M9_4_CALIBRATION_PATH = M9_4_DIR / "m9-4-calibration.json"
M9_4_CONFUSION_MATRICES_PATH = M9_4_DIR / "m9-4-confusion-matrices.json"
M9_4_GUARDRAILS_PATH = M9_4_DIR / "m9-4-guardrails.json"
M9_4_SUMMARY_PATH = M9_4_DIR / "m9-4-summary.md"
M9_4_CLOSURE_PATH = M9_4_DIR / "m9-4-closure.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def current_commit() -> str:
    return _git("rev-parse", "HEAD")


def current_branch() -> str:
    return _git("branch", "--show-current")


def assert_locked_test_closed() -> bool:
    opened = locked_test_opened(ROOT)
    if opened:
        raise AssertionError("locked_final_test/locked_topology_test must remain unopened for M9.4")
    return opened


# ---------------------------------------------------------------------------
# Section 4/5: M9.4-only seed-base scheme. Two roles per family --
# `calibration_m9_4` and `development_m9_4` -- each with its own seed base,
# so no M9.4 pool ever draws the same (seed, network) pair as another M9.4
# pool OR as anything M7/M9.0/M9.0a/M9.0b already used
# (`m7.SEED_BASES` occupies roughly 940_000_000-970_000_000; this table
# starts at 990_000_000, entirely clear of that range).
# ---------------------------------------------------------------------------

M9_4_SEED_BASE_FLOOR = 990_000_000
M9_4_SEED_BASE_STEP = 100_000
M9_4_ROLES: tuple[str, ...] = ("calibration_m9_4", "development_m9_4")

M9_4_SEED_BASES: dict[tuple[str, str], int] = {}
_next = M9_4_SEED_BASE_FLOOR
for _family in ALL_FAMILIES:  # fixed declared order: golden-reference, branched-loop, loop-grid, coastal-branch, tree-branch, dense-loop
    for _role in M9_4_ROLES:
        M9_4_SEED_BASES[(_family, _role)] = _next
        _next += M9_4_SEED_BASE_STEP

assert min(m7.SEED_BASES.values()) >= 940_000_000
assert max(m7.SEED_BASES.values()) < M9_4_SEED_BASE_FLOOR, "M9.4 seed bases must not collide with M7's range"


def m9_4_seed_base(family: str, role: str) -> int:
    return M9_4_SEED_BASES[(family, role)]


def wilson_interval_90(successes: int, n: int) -> tuple[float, float]:
    """Wilson 90% score interval -- same predeclared descriptive-uncertainty
    method M9.3 used (Section 8/19 representativeness reporting)."""

    import math

    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.6448536269514722
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def relative_close(a: float, b: float, *, rel_tol: float = LEGACY_REPRODUCTION_RELATIVE_TOLERANCE) -> bool:
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= rel_tol


def checkpoint_sha256(path: str) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_junction_list(family: str, loader: Any) -> tuple[str, ...]:
    """Complete, deterministically sorted candidate source-junction set for
    `family` -- NO `EVAL_MAX_SOURCES`-style truncation (Section 4 rule 3)."""

    network_probe = loader()
    return tuple(sorted(network_probe.junction_name_list))


ALL_FAMILY_LOADERS: dict[str, Any] = dict(m7.TRAINED_FAMILIES) | dict(m7.UNSEEN_FAMILIES)
