"""TRUE Milestone 10.2: learned-vs-deterministic Scout scientific comparison
-- frozen protocol constants, hashed BEFORE any decision-utility metric is
computed. Distinct from `m10_2_refit_protocol.py` (which froze the Level-A
Scout REFIT's own training/gate protocol, already executed and accepted:
`M10_2_SCOUT_REFIT_A_ACCEPTED`). This module governs the SEPARATE, later
milestone that decides whether the accepted Level-A refit's learned Scout
should be scientifically promoted over the deterministic Scout baseline.

Frozen document: `docs/evaluation/HYDROCORE_V5_M10_2_TRUE_EVALUATION_PROTOCOL.md`.
This module is imported by both the execution script
(`run_m10_2_true_evaluation.py`) and the decision script
(`run_m10_2_true_decide.py`) so both read the exact same frozen values --
no value here may be changed after any decision-utility result is inspected.
"""

from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------------------
# Section A: checkpoints (frozen, approved by the authorizing task prompt).
# Parent M9.6 teacher hashes are the SAME canonical hashes M9.6/M9.7A/M10
# have used throughout -- never altered here.
# ---------------------------------------------------------------------------

LEVEL_A_REFIT_CHECKPOINT_SHA256: dict[int, str] = {
    20260814: "7d8638f6a84afb570e02a6dc1be4c5e9438191226de15d7ce4d40d273c94004a",
    31874: "cecc125cd599d6d1469a9aa34724b13dc6e35497e9aeac445204efcc4fa0c819",
    20260815: "7e09704b817ad0167211f286406656c6a70d6fc24d1a3b0a9f97f240434c59a1",
}

PARENT_M9_6_TEACHER_SHA256: dict[int, str] = {
    20260814: "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
    31874: "527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332",
    20260815: "b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c",
}

# ---------------------------------------------------------------------------
# Section B: population (development-only, disjoint from every locked split,
# from the M10.2 refit's own TRAIN (1_200_000_000, count=250) and VALIDATION
# (1_200_100_000, count=300 post-amendment) seed ranges, and from every other
# seed base in the repository -- verified disjoint by grep over
# 1_200_200_000..1_200_299_999 before this document was frozen (zero prior
# hits)). Scoped to the SAME single family (golden-reference) and the SAME
# depth/budget/noise Level A was actually trained and accepted under
# (`m10_2_refit_protocol.FAMILY/DEPTH/MAXIMUM_SAMPLES/NOISE_SCALE_MG_L`) --
# evaluating on a family or budget the accepted checkpoint never saw would
# confound "does learned Scout generalize beyond its trained scope" with
# this milestone's own question ("does the ACCEPTED Level-A checkpoint beat
# deterministic Scout"); widening scope is explicitly out of this task's
# authorization (Level A's own protocol Section 4: "a future, separately
# authorized amendment may widen family coverage").
# ---------------------------------------------------------------------------

FAMILY = "golden-reference"
DEPTH = 25  # MATURE bucket -- matches Level A's own frozen depth, unchanged.
MAXIMUM_SAMPLES = 3  # matches Level A's own MAXIMUM_SAMPLES pilot bound.
NOISE_SCALE_MG_L = 0.5  # matches Level A's own frozen noise scale.

EVAL_SEED_BASE = 1_200_200_000
EVAL_COUNT = 100  # task's own "100 paired incidents preferred" bar, met exactly.
SOURCE_ROUND_ROBIN = True  # matches Level A's own TRAIN/VALIDATION convention.

# ---------------------------------------------------------------------------
# Section C: "actionable / source-resolved" definition (frozen; reuses the
# EXISTING governed source-posterior/calibration semantics -- never an ad
# hoc threshold invented for this milestone). Matches Milestone 5's own
# precedent (`scripts/hydrocore_v5/run_m5_sampling.py`) naming discipline:
# called `candidate_gate_pass`, not "actionable", since it is only the
# candidate-count planning gate, not full product actionability.
#
# CANDIDATE_GATE_K reuses `hydroswarm.simulation.wrapper.
# MAXIMUM_EVALUATION_HYPOTHESES` (=3) -- the SAME governed production
# constant M5 itself reused for the identical purpose. It is a coincidence
# (not circular reasoning) that MAXIMUM_SAMPLES above also equals 3: one is
# the sample BUDGET (how many Scout samples either policy may take), the
# other is the candidate-SET-SIZE gate threshold (how small the calibrated
# candidate set must shrink to count as resolved) -- both happen to reuse
# already-frozen values of 3 from two independent prior milestones (Level
# A's own MAXIMUM_SAMPLES; M5/production's own MAXIMUM_EVALUATION_HYPOTHESES),
# neither invented here.
#
# Calibration: the frozen M9.6 B_DEPTH_AWARE `SplitConformalCalibrator`,
# alpha=0.1, fit from `reports/evaluation/hydrocore-v5/m9-6/
# m9-6-canonical-calibration.jsonl`'s ARM_B_M9_6 support examples AS-IS (no
# refit, no new support data) -- the EXACT SAME frozen-support-refit pattern
# `scripts/hydrocore_v5/run_m10_1_decide.py::_fit_frozen_calibrator` already
# used for M10.1 (deterministic given the same examples/alpha/grouping, so
# this reproduces the identical calibrator every time, not a new fit against
# new data). `network_id=f"{FAMILY}:{depth_bucket_of(DEPTH)}"` =
# "golden-reference:MATURE", independently confirmed present in that support
# set with 480 examples (>= minimum_group_size=10), so `.selection()`'s
# NETWORK_SPECIFIC branch is always taken regardless of the `condition`
# argument's value -- `condition=None` is passed throughout, deliberately,
# since it is provably irrelevant to which quantile is selected here.
# ---------------------------------------------------------------------------

CANDIDATE_GATE_K = 3
CALIBRATION_ALPHA = 0.1
CALIBRATION_SUPPORT_PATH = "reports/evaluation/hydrocore-v5/m9-6/m9-6-canonical-calibration.jsonl"
CALIBRATION_SUPPORT_ARM = "ARM_B_M9_6"
CALIBRATION_MINIMUM_GROUP_SIZE = 10

# ---------------------------------------------------------------------------
# Section D: pairing / predictor-consistency disclosure (frozen; see the
# protocol document's own Audit section for the full derivation). Both arms
# use the SAME Level-A refit checkpoint (Section A) for every forward pass
# in every round -- never the original M9.6 teacher. `role_features`/
# `residual_features` (round/budget/accessibility) are populated identically
# for BOTH arms at every round, including round 0, using the exact
# `hydroswarm.training.scout_training_state.build_scout_training_state_batch`
# schema Level A was trained under (`SCOUT_TRAINING_STATE_SCHEMA_VERSION =
# "scout-training-state-v1"`, unchanged). This means round-0 source-node
# numbers in THIS evaluation are NOT bit-identical to M9's own historical,
# frozen Sentinel characterization (which never populated these two
# channels) -- this is a new, self-consistent, WITHIN-this-evaluation
# comparison between two Scout policies sharing one predictor, never a
# restatement of M9's own frozen Sentinel result, and M9's own artifacts are
# untouched by this milestone.
# ---------------------------------------------------------------------------

STATE_SCHEMA_VERSION = "scout-training-state-v1"  # reused unmodified from Level A.

# ---------------------------------------------------------------------------
# Section E: statistics (frozen; reused unmodified from the M9/M10/M10.1
# cross-milestone convention).
# ---------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.90
BOOTSTRAP_SEED = 20260819

# ---------------------------------------------------------------------------
# Section F: promotion rule (frozen BEFORE any decision-utility result is
# inspected). See the protocol document for full prose; encoded here as the
# exact numeric thresholds `run_m10_2_true_decide.py` applies mechanically.
# ---------------------------------------------------------------------------

#: Primary metric: fraction of incidents with candidate_gate_pass achieved
#: at or before round MAXIMUM_SAMPLES ("actionable_within_3_samples").
PRIMARY_METRIC = "actionable_within_budget"
#: Promotion requires a POSITIVE point estimate (learned - deterministic)
#: in all three seeds, AND a 90% paired-bootstrap CI whose lower bound
#: exceeds zero in at least two of the three seeds (task requirement:
#: "reasonably consistent across the three refit seeds rather than being
#: driven by one seed" -- a majority-seed, CI-supported bar, not a
#: unanimous-CI bar, since a single underpowered seed's CI crossing zero
#: while its point estimate still improves is not evidence AGAINST a real
#: effect).
PROMOTION_MIN_SEEDS_POSITIVE_POINT_ESTIMATE = 3
PROMOTION_MIN_SEEDS_CI_EXCLUDES_ZERO = 2
#: No-material-regression bars (frozen; "not CI-confidently worse" -- the
#: paired difference's 90% CI lower bound must not itself exceed zero in
#: the REGRESSING direction for the metric in question).
NO_REGRESSION_METRICS = (
    "never_actionable_fraction",  # learned must not be CI-confidently worse (higher).
    "source_top1_final_round",  # learned must not be CI-confidently worse (lower).
)


def payload() -> dict[str, object]:
    return {
        "level_a_refit_checkpoint_sha256": LEVEL_A_REFIT_CHECKPOINT_SHA256,
        "parent_m9_6_teacher_sha256": PARENT_M9_6_TEACHER_SHA256,
        "family": FAMILY,
        "depth": DEPTH,
        "maximum_samples": MAXIMUM_SAMPLES,
        "noise_scale_mg_l": NOISE_SCALE_MG_L,
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_count": EVAL_COUNT,
        "source_round_robin": SOURCE_ROUND_ROBIN,
        "candidate_gate_k": CANDIDATE_GATE_K,
        "calibration_alpha": CALIBRATION_ALPHA,
        "calibration_support_path": CALIBRATION_SUPPORT_PATH,
        "calibration_support_arm": CALIBRATION_SUPPORT_ARM,
        "calibration_minimum_group_size": CALIBRATION_MINIMUM_GROUP_SIZE,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_ci": BOOTSTRAP_CI,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_metric": PRIMARY_METRIC,
        "promotion_min_seeds_positive_point_estimate": PROMOTION_MIN_SEEDS_POSITIVE_POINT_ESTIMATE,
        "promotion_min_seeds_ci_excludes_zero": PROMOTION_MIN_SEEDS_CI_EXCLUDES_ZERO,
        "no_regression_metrics": list(NO_REGRESSION_METRICS),
    }


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, default=str).encode()).hexdigest()


def to_json_doc() -> dict[str, object]:
    return {"kind": "M10_2_TRUE_EVALUATION_PROTOCOL", "milestone": "M10.2-true", **payload(), "protocol_hash": protocol_hash()}
