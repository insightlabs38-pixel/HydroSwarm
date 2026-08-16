"""Shared, governed helpers for the Milestone 9.2 diagnostic study.

Milestone 9.2 is DIAGNOSTIC / ANALYSIS-ONLY: it trains nothing, tunes
nothing, and never opens `locked_final_test`/`locked_topology_test`. Every
constant below is either copied verbatim from the frozen M9.1 runner
(`scripts/hydrocore_v5/m9_1_common.py`) -- reused, not reimplemented, so the
diagnostic population/depths/seeds can never silently drift from what M9.1
actually evaluated -- or is a new M9.2-only constant explicitly pinned by the
milestone brief (bootstrap seed, depth-quartile scheme, case-study counts).

This module implements no scientific policy of its own for anything M9.1
already pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import m9_1_common as m91  # noqa: E402

# ---------------------------------------------------------------------------
# Reused verbatim from M9.1 (never redefined independently here).
# ---------------------------------------------------------------------------

PROTOCOL_FROZEN_AT_COMMIT = m91.PROTOCOL_FROZEN_AT_COMMIT  # 0f05be1d...
M9_1_CLOSURE_COMMIT = "3b6ee2a3529faffdcc2288cda294a4ef4d6f0765"
FROZEN_BRANCH = m91.FROZEN_BRANCH
NOVEL_ARMS = m91.NOVEL_ARMS  # (GRAPH_ODE, GRAPH_CDE, GRAPH_SDE)
ALL_ARMS = m91.ALL_ARMS  # (CURRENT, GRAPH_ODE, GRAPH_CDE, GRAPH_SDE)
SCREENING_SEEDS = m91.SCREENING_SEEDS  # (20260814, 31874)
CONFIRMATION_SEED = m91.CONFIRMATION_SEED  # 20260815 -- NEVER used in M9.2 pairing
EARLY_DEPTHS = m91.EARLY_DEPTHS
MID_DEPTHS = m91.MID_DEPTHS
MATURE_DEPTHS = m91.MATURE_DEPTHS
DEPTH_BUCKET_OF = m91.DEPTH_BUCKET_OF
CAUSAL_PREFIX_DEPTHS = m91.CAUSAL_PREFIX_DEPTHS  # (1, 2, 3, 4, 6, 12, 25)
NETWORK_FAMILY = m91.NETWORK_FAMILY
ALPHA = m91.ALPHA
EPS = m91.EPS

#: Section 1 governance: CURRENT's third M8.7-reused checkpoint seed is a
#: documented M9.1 baseline artifact only -- it has no matching novel-arm
#: seed and MUST NEVER enter a cross-arm paired M9.2 comparison.
EXCLUDED_UNPAIRED_SEED = 20260815
assert EXCLUDED_UNPAIRED_SEED == CONFIRMATION_SEED

# ---------------------------------------------------------------------------
# M9.2-only pinned constants (declared BEFORE any analysis is run).
# ---------------------------------------------------------------------------

#: Section 5 bootstrap: deterministic, DIAGNOSTIC-only, distinct from M9.1's
#: own promotion-test bootstrap seed (20260815) so the two are never
#: conflatable in an artifact diff.
M9_2_BOOTSTRAP_SEED = 20260816
M9_2_BOOTSTRAP_RESAMPLES = 2000
M9_2_BOOTSTRAP_INTERVAL = 0.90

#: Section 7: "large" rank regression/improvement thresholds, pinned in the
#: milestone brief itself ("worsens by >= 3 and >= 5"), not tuned here.
RANK_DELTA_LARGE_THRESHOLDS = (3, 5)

#: Section 12: case-study export counts, pinned in the milestone brief.
CASE_STUDY_TOP_N = 10

REPORT_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5"
M9_1_RESULTS_PATH = REPORT_DIR / "m9-1-results.json"
M9_1_CALIBRATION_PATH = REPORT_DIR / "m9-1-calibration.json"
M9_1_GUARDRAILS_PATH = REPORT_DIR / "m9-1-guardrails.json"
M9_1_CLOSURE_PATH = REPORT_DIR / "m9-1-closure.json"

M9_2_DIR = REPORT_DIR / "m9-2"
M9_2_FIGURES_DIR = M9_2_DIR / "figures"
M9_2_MANIFEST_PATH = M9_2_DIR / "m9-2-manifest.json"
M9_2_CANONICAL_PATH = M9_2_DIR / "m9-2-canonical-diagnostics.jsonl"
M9_2_DEPTH_METRICS_PATH = M9_2_DIR / "m9-2-depth-metrics.json"
M9_2_DISAGREEMENTS_PATH = M9_2_DIR / "m9-2-disagreements.json"
M9_2_RANK_ANALYSIS_PATH = M9_2_DIR / "m9-2-rank-analysis.json"
M9_2_TOPOLOGY_ANALYSIS_PATH = M9_2_DIR / "m9-2-topology-analysis.json"
M9_2_MISSINGNESS_ANALYSIS_PATH = M9_2_DIR / "m9-2-missingness-analysis.json"
M9_2_CALIBRATION_DIAGNOSTICS_PATH = M9_2_DIR / "m9-2-calibration-diagnostics.json"
M9_2_CASE_STUDIES_PATH = M9_2_DIR / "m9-2-case-studies.json"
M9_2_SUMMARY_PATH = M9_2_DIR / "m9-2-summary.md"
M9_2_CLOSURE_PATH = M9_2_DIR / "m9-2-closure.json"

# ---------------------------------------------------------------------------
# Governance re-verification (independent of M9.1's own assertions -- M9.2
# must not simply trust M9.1's artifacts without re-checking lock state
# itself, per Section 1.A).
# ---------------------------------------------------------------------------


def assert_code_under_test_commit() -> str:
    return m91.assert_code_under_test_commit()


def assert_locked_test_closed() -> bool:
    return m91.assert_locked_test_closed()


def current_commit() -> str:
    return m91.current_commit()


# ---------------------------------------------------------------------------
# Bootstrap: reuses M9.1's exact resampling loop (same formula, same
# incident-level unit, same percentile convention) with the M9.2-only
# bootstrap seed. Never reimplemented independently.
# ---------------------------------------------------------------------------


def paired_bootstrap_m9_2(candidate_per_incident, control_per_incident) -> dict[str, Any]:
    return m91.paired_bootstrap(
        candidate_per_incident,
        control_per_incident,
        resamples=M9_2_BOOTSTRAP_RESAMPLES,
        seed=M9_2_BOOTSTRAP_SEED,
        interval=M9_2_BOOTSTRAP_INTERVAL,
    )
