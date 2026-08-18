"""M10.3C -- Strategist expanded-population identifiability/oracle-gate
amendment: frozen protocol constants, hashed BEFORE any candidate-
verification/diversity/identifiability/oracle result is inspected.

Diagnostic/population-governance only. Trains NOTHING, touches no
checkpoint, opens no locked data, does not re-run true M10.3/M10.3A/B.
Additive to `docs/evaluation/HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md`
(M10.3B closure: `M10_3B_POPULATION_AMENDMENT_REQUIRED`) -- this module
freezes the amended population M10.3B's own closure recommended, before
any of its own results are inspected.

## Central question

Does an expanded, still-governed and physically realistic Strategist
development population (the two other already-TRAINED_FAMILIES topology
families, plus the five other already-governed causal-prefix depth
buckets) contain enough within-incident candidate diversity AND
exact-oracle decision utility to scientifically justify another learned
Strategist training attempt (M10.3D)?

## Audit finding this protocol is built on (Section 7 of the M10.3C task
spec: trace the actual candidate-generation/depth machinery before
assuming numeric semantics)

Traced directly from source (`hydroswarm.training.strategist_trajectory.
build_strategist_trajectory`, `hydroswarm.training.strategist_candidate_
corpus._reconstruct_context_and_proposals`/`build_strategist_candidate_
example`): Strategist candidate generation and target-label computation
(`generate_response_plans`, `generate_strategist_labels`, exact WNTR
verification, `plan_value_policy.evaluate_plan_value`) are built from
`build_sensor_series(scenario, feature_context)` -- the SCENARIO'S FULL,
UNTRUNCATED sensor evidence -- and never receive a `depth` argument
anywhere in that call chain. `depth` (`hydroswarm.training.causal_prefix.
CAUSAL_PREFIX_DEPTHS = (1, 2, 3, 4, 6, 12, 25)`, `truncate_causal_prefix`)
is exclusively a HydroCore causal-prefix MODEL-INPUT truncation concept,
consumed only by `scenario_to_prefix_example` (used to build the causal-
prefix example a TRAINING run would feed the frozen backbone) -- a step
`_build_corpus` (`run_m10_3_level_a_train.py`) calls SEPARATELY from, and
in addition to, `build_strategist_candidate_example` (which needs no depth
argument at all). Consequently: **depth has no causal effect whatsoever on
Strategist candidate proposals, WNTR verification outcomes, or any of the 7
governed Strategist targets in this repository's current implementation.**
It affects only what evidence-truncated input a future LEARNED Strategist
would be trained/conditioned on, never the exact-WNTR-verified population
this diagnostic measures. This finding is reported prominently in
`m10-3c-invariance-audit.json` and the accompanying document -- it changes
nothing about the topology-family axis (which DOES affect candidate
verification and diversity, since candidate generation is network-topology-
dependent via `PlanVerifier`/`HydraulicSimulator`), but means the "depth"
axis in this population functions as a disjoint-seed bookkeeping
partition (preserving the task's required family x depth reporting grid
and preparing depth-labeled cells for a future M10.3D causal-prefix
input construction) rather than a source of physical candidate diversity.
Because of this, per-cell (family, depth) results are expected -- and are
verified here, not merely assumed -- to be statistically consistent within
one family across every depth label; the real diversity axis this
population tests is topology family alone.

This does NOT weaken or bypass the task's required grid: every one of the
3 families x 6 depths = 18 cells is generated from its own disjoint seed
block and reported individually (Section "family x depth cells" below),
exactly matching the task's Section 14/15/16 requirements.
"""

from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------------------
# Section 1: topology families (already governed, already TRAINED_FAMILIES
# -- `scripts/hydrocore_v5/run_m7_topology.py::TRAINED_FAMILIES`). No new
# family, no unseen/locked family. Order matches TRAINED_FAMILIES exactly.
# ---------------------------------------------------------------------------

FAMILIES: tuple[str, ...] = ("golden-reference", "branched-loop", "loop-grid")

# ---------------------------------------------------------------------------
# Section 2: causal-prefix depth buckets (already governed --
# `hydroswarm.training.causal_prefix.CAUSAL_PREFIX_DEPTHS = (1, 2, 3, 4, 6,
# 12, 25)`). This task's own spec (Section 8) requests exactly {1, 2, 3, 4,
# 6, 25} -- a strict, already-governed SUBSET of CAUSAL_PREFIX_DEPTHS
# (omitting only 12, the second MATURE-bucket depth already partially
# represented by 25) -- never an invented depth value. See the module
# docstring's audit finding: depth is a bookkeeping/future-M10.3D-input
# label for this diagnostic, not a driver of candidate/target outcomes.
# ---------------------------------------------------------------------------

DEPTH_BUCKETS: tuple[int, ...] = (1, 2, 3, 4, 6, 25)
assert DEPTH_BUCKETS == tuple(sorted(set(DEPTH_BUCKETS)))

MAXIMUM_PLAN_COUNT = 9  # ACTION_TEMPLATE_COUNT -- identical to m10_3_refit_protocol.MAXIMUM_PLAN_COUNT.

# ---------------------------------------------------------------------------
# Section 3: seed namespace (development-only, disjoint from every
# historical range). Base 1_400_000_000 -- continues the per-milestone
# seed-block convention (M10.1=1_100_000_000, M10.2=1_200_000_000,
# M10.3A/B=1_300_000_000). Verified disjoint by grep over
# 1_400_000_000..1_499_999_999 across every *.py/*.json/*.md in the
# repository (zero prior hits) before this module was frozen.
#
# Family offsets spaced 10_000_000 apart (matching run_m7_topology.py's own
# >=1_000_000-family-block convention, widened here for headroom): each
# family's per-scenario seed is `family_base + index * 100`
# (`_family_scenario_pool`'s own convention), so a 10_000_000 gap leaves
# comfortable room even for a much larger PER_FAMILY_COUNT than this
# protocol actually uses.
# ---------------------------------------------------------------------------

SEED_NAMESPACE_BASE = 1_400_000_000
FAMILY_SEED_OFFSET: dict[str, int] = {
    "golden-reference": 0,
    "branched-loop": 10_000_000,
    "loop-grid": 20_000_000,
}
FAMILY_SEED_BASE: dict[str, int] = {family: SEED_NAMESPACE_BASE + offset for family, offset in FAMILY_SEED_OFFSET.items()}

#: Reserved (NOT generated by this diagnostic -- disjoint namespace only,
#: preserved for a future M10.3D refit's own fresh train/validation split
#: IF M10.3C passes; materializing it now would be premature scope this
#: task does not authorize). Spaced far enough past the largest possible
#: FAMILY_SEED_BASE + PER_FAMILY_COUNT * 100 (20_000_000 + 180*100 =
#: 20_018_000, well under 50_000_000) to guarantee no overlap even if a
#: future amendment enlarges PER_FAMILY_COUNT moderately.
RESERVED_FUTURE_M10_3D_SEED_BASE = 1_450_000_000

SOURCE_ROUND_ROBIN = True  # matches m10_3_refit_protocol.SOURCE_ROUND_ROBIN (M10.3A/B's own convention).
SPLIT_LABEL = "development_holdout"  # hydroswarm.data.scenarios.DatasetSplit.DEVELOPMENT_HOLDOUT -- never TEST/locked.

# ---------------------------------------------------------------------------
# Section 4: population size (frozen BEFORE any result is inspected).
#
# PER_FAMILY_COUNT = 180 scenarios/family, round-robin-labeled across the 6
# DEPTH_BUCKETS (index % 6) -> exactly 30 scenarios/cell, 3 x 6 = 18 cells,
# 540 scenarios total. Rationale:
#   - 30/cell exceeds this program's own established GATE_MIN_SUPPORT=20
#     (m10_3_refit_protocol.py) with margin, on every one of 18 cells
#     equally -- no cell is starved.
#   - Balanced by construction (round-robin depth-label assignment,
#     identical PER_FAMILY_COUNT across all 3 families) -- golden-
#     reference/depth-25 is NOT over-weighted merely because it is the only
#     cell with prior (M10.3A/B) data; it receives the SAME 30-scenario
#     allocation as every other cell.
#   - 540 total is the same order of magnitude as M10.3A/B's own single-
#     family population (250 train + 300 validation = 550), so this
#     amendment is a genuine breadth expansion (3x more families, 6x more
#     depth labels) at comparable total scenario-generation cost, not an
#     order-of-magnitude scale-up this diagnostic-only task does not need.
#   - Diagnostic-only, single split (`SPLIT_LABEL` above): M10.3C performs
#     no training, so no separate train/validation split is required here
#     (Section 9's own instruction); `RESERVED_FUTURE_M10_3D_SEED_BASE`
#     preserves independent, disjoint headroom for that later need.
# ---------------------------------------------------------------------------

PER_FAMILY_COUNT = 180
assert PER_FAMILY_COUNT % len(DEPTH_BUCKETS) == 0
PER_CELL_COUNT = PER_FAMILY_COUNT // len(DEPTH_BUCKETS)
TOTAL_SCENARIO_COUNT = PER_FAMILY_COUNT * len(FAMILIES)

# ---------------------------------------------------------------------------
# Section 5: near-tie tolerances -- REUSED VERBATIM from
# `scripts/hydrocore_v5/run_m10_3b_diagnosis.py::NEAR_TIE_TOLERANCE`
# (imported directly there, not redefined here, so no drift is possible).
# No protocol-level mathematical reason requires a different tolerance:
# the underlying target formulas/scales (`plan_value_policy.py`) are
# byte-identical for every family/depth -- these targets are computed on
# a fixed, train-owned scale (`plan_value_policy.py`'s own constants),
# never relative to a per-topology-family unit, so the SAME tolerances
# remain the physically correct "smallest distinguishable difference"
# regardless of which of the 3 already-governed families is evaluated.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Section 6: M10.3C population-sufficiency gate (frozen HERE, BEFORE any
# candidate-verification/diversity/identifiability/oracle result in this
# task is inspected -- Section 18 of the task spec). Anchored explicitly
# against M10.3B's own negative baseline (the single-family/single-depth
# pilot): fraction_incidents_2plus_distinguishable(plan_value)=17.5%,
# fraction_3plus=0.0%, oracle best-vs-NO_ACTION meaningfully-positive
# fraction=9.6%, mean gain=0.022, NO_ACTION-already-near-optimal=90.4%
# (`reports/evaluation/hydrocore-v5/m10/m10-3b-diagnosis/m10-3b-within-
# incident-variance.json`, `m10-3b-oracle-utility.json`, validation split).
#
# PASS requires BOTH population-diversity AND oracle-utility criteria,
# each evaluated POOLED across all 18 cells, at thresholds set well above
# (roughly double or more) the M10.3B baseline so a marginal improvement
# cannot pass by chance, AND requires the signal not be concentrated in
# fewer than DIVERSITY_MIN_CONTRIBUTING_CELLS/ORACLE_MIN_CONTRIBUTING_CELLS
# of the 18 cells (Section 18's "not driven by one tiny cell only" /
# "gains should not be concentrated solely in rare pathological cases").
# ---------------------------------------------------------------------------

#: Population-diversity criteria (target: plan_value, the primary ranking
#: target -- Section 19 of the task spec explicitly does not require every
#: target to be non-degenerate).
DIVERSITY_2PLUS_FRACTION_THRESHOLD = 0.35  # >= ~2x M10.3B's 17.5% baseline.
DIVERSITY_3PLUS_FRACTION_THRESHOLD = 0.10  # materially above M10.3B's 0.0%.
DIVERSITY_MIN_CONTRIBUTING_CELLS = 3  # of 18; each independently >= threshold with its own n>=20 support.
DIVERSITY_CELL_MIN_SUPPORT = 20  # matches GATE_MIN_SUPPORT.

#: Oracle-utility criteria (best exact-WNTR-verified candidate vs NO_ACTION,
#: plan_value scale).
ORACLE_MEANINGFUL_GAIN_FRACTION_THRESHOLD = 0.20  # >= ~2x M10.3B's 9.6% baseline.
ORACLE_MEAN_GAIN_THRESHOLD = 0.05  # >= ~2.3x M10.3B's 0.022 mean gain.
ORACLE_NO_ACTION_NEAR_OPTIMAL_MAX = 0.75  # materially below M10.3B's 90.4%.
ORACLE_MIN_CONTRIBUTING_CELLS = 3  # of 18; each independently contributes >=10% meaningfully-positive-gain incidents.
ORACLE_CELL_MIN_SUPPORT = 20

#: CONDITIONAL (Section 20-C) family-level thresholds: a family independently
#: clears the FULL pass bar pooled over its own 6 depth cells (n well above
#: support minimums), while >=1 other family clearly fails (materially below
#: half the pass thresholds) -- a real, preregistered, operationally
#: meaningful domain distinction (topology family/redundancy regime), not a
#: post-hoc favorable-cell selection.
CONDITIONAL_FAMILY_CLEAR_FAIL_DIVERSITY_MAX = 0.20  # family-level fraction_2plus below this counts as "clearly fails".
CONDITIONAL_FAMILY_CLEAR_FAIL_ORACLE_MAX = 0.10  # family-level oracle-meaningful-gain-fraction below this counts as "clearly fails".
#: Minimum per-family support (incidents with >=2 valid candidates, resp.
#: incidents considered for oracle) required before a family-level PASS/
#: CONDITIONAL/clear-fail judgment is trusted at all -- comfortably above
#: DIVERSITY_CELL_MIN_SUPPORT/ORACLE_CELL_MIN_SUPPORT since a family pools
#: 6 cells.
FAMILY_LEVEL_MIN_SUPPORT = 100

#: Complete decision tree (frozen here, before any result is inspected --
#: Section 20 of the task spec: exactly one principal closure).
#:  1. If the population cannot be validly built (>50% of scenarios in any
#:     one family fail to produce a usable candidate example, or a hard
#:     data/runtime error prevents completing the diagnostic) ->
#:     M10_3C_POPULATION_EVALUATION_BLOCKED. State exact cause.
#:  2. Else compute GLOBAL_PASS = diversity_pass(pooled across all 18
#:     cells) AND oracle_pass(pooled across all 18 cells), each per the
#:     thresholds above.
#:  3. If GLOBAL_PASS -> M10_3C_POPULATION_IDENTIFIABILITY_PASS.
#:  4. Else, compute FAMILY_PASS independently for each of the 3 families
#:     (pooled over that family's own 6 depth cells, requiring
#:     FAMILY_LEVEL_MIN_SUPPORT), using the SAME full threshold set as
#:     GLOBAL_PASS. Also compute FAMILY_CLEAR_FAIL (family-level
#:     diversity_2plus < CONDITIONAL_FAMILY_CLEAR_FAIL_DIVERSITY_MAX OR
#:     family-level oracle-meaningful-gain-fraction <
#:     CONDITIONAL_FAMILY_CLEAR_FAIL_ORACLE_MAX). If >=1 family is
#:     FAMILY_PASS and >=1 (different) family is FAMILY_CLEAR_FAIL ->
#:     M10_3C_POPULATION_IDENTIFIABILITY_CONDITIONAL, scoped to exactly the
#:     FAMILY_PASS family/families (a real, preregistered, operationally
#:     meaningful topology-family/redundancy-regime distinction, decided by
#:     this frozen rule alone -- never a post-hoc favorable-cell choice).
#:  5. Else if >=1 family is FAMILY_PASS but no family is FAMILY_CLEAR_FAIL
#:     (results cluster together without a clean separating regime) -> do
#:     NOT declare CONDITIONAL (Section 20 forbids inventing a favorable
#:     split without a real preregistered domain distinction backing it);
#:     the conservative, decided-in-advance default in this ambiguous case
#:     is M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED.
#:  6. Else (no family is FAMILY_PASS) -> M10_3C_LEARNED_STRATEGIST_NOT_
#:     JUSTIFIED.
GATE_DECISION_TREE = (
    "1) unbuildable population -> BLOCKED. "
    "2) pooled diversity_pass AND pooled oracle_pass -> PASS. "
    "3) else >=1 FAMILY_PASS and >=1 different FAMILY_CLEAR_FAIL -> CONDITIONAL scoped to the passing family/families. "
    "4) else >=1 FAMILY_PASS but no FAMILY_CLEAR_FAIL (no clean regime split) -> NOT_JUSTIFIED (conservative default, no cherry-picking). "
    "5) else (no family passes) -> NOT_JUSTIFIED."
)


def payload() -> dict[str, object]:
    return {
        "families": list(FAMILIES),
        "depth_buckets": list(DEPTH_BUCKETS),
        "maximum_plan_count": MAXIMUM_PLAN_COUNT,
        "seed_namespace_base": SEED_NAMESPACE_BASE,
        "family_seed_offset": FAMILY_SEED_OFFSET,
        "family_seed_base": FAMILY_SEED_BASE,
        "reserved_future_m10_3d_seed_base": RESERVED_FUTURE_M10_3D_SEED_BASE,
        "source_round_robin": SOURCE_ROUND_ROBIN,
        "split_label": SPLIT_LABEL,
        "per_family_count": PER_FAMILY_COUNT,
        "per_cell_count": PER_CELL_COUNT,
        "total_scenario_count": TOTAL_SCENARIO_COUNT,
        "gate": {
            "diversity_2plus_fraction_threshold": DIVERSITY_2PLUS_FRACTION_THRESHOLD,
            "diversity_3plus_fraction_threshold": DIVERSITY_3PLUS_FRACTION_THRESHOLD,
            "diversity_min_contributing_cells": DIVERSITY_MIN_CONTRIBUTING_CELLS,
            "diversity_cell_min_support": DIVERSITY_CELL_MIN_SUPPORT,
            "oracle_meaningful_gain_fraction_threshold": ORACLE_MEANINGFUL_GAIN_FRACTION_THRESHOLD,
            "oracle_mean_gain_threshold": ORACLE_MEAN_GAIN_THRESHOLD,
            "oracle_no_action_near_optimal_max": ORACLE_NO_ACTION_NEAR_OPTIMAL_MAX,
            "oracle_min_contributing_cells": ORACLE_MIN_CONTRIBUTING_CELLS,
            "oracle_cell_min_support": ORACLE_CELL_MIN_SUPPORT,
            "conditional_family_clear_fail_diversity_max": CONDITIONAL_FAMILY_CLEAR_FAIL_DIVERSITY_MAX,
            "conditional_family_clear_fail_oracle_max": CONDITIONAL_FAMILY_CLEAR_FAIL_ORACLE_MAX,
            "family_level_min_support": FAMILY_LEVEL_MIN_SUPPORT,
            "decision_tree": GATE_DECISION_TREE,
        },
        "m10_3b_baseline": {
            "fraction_2plus_distinguishable_plan_value": 0.175,
            "fraction_3plus_distinguishable_plan_value": 0.0,
            "oracle_meaningful_gain_fraction": 0.096,
            "oracle_mean_gain": 0.022,
            "no_action_near_optimal_fraction": 0.904,
        },
    }


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, default=str).encode()).hexdigest()


def to_json_doc() -> dict[str, object]:
    return {"kind": "M10_3C_POPULATION_PROTOCOL", "milestone": "M10.3C", **payload(), "protocol_hash": protocol_hash()}
