# HydroCore-v5 Milestone 8.7 protocol (frozen before any arm is evaluated)

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md`. Introduced because
Milestone 8.6 (`reports/evaluation/hydrocore-v5/m8-6-summary.md`) demonstrated:

- `ABSOLUTE_TIME_ORIGIN_LEAKAGE` in `HydraulicFeatureBuilder.build`'s
  never-observed-node `measurement_age` fallback, and
- `TEMPORAL_FEATURE_USAGE_WEAK_OR_PARTIAL` (the explicit timestamp pathway is
  measurably inert; the derived age-feature pathway is measurably used).

This document freezes the M8.7 protocol BEFORE any arm is trained or
evaluated. It is not altered after seeing arm performance.

## 1. Splits

Reuses the existing, unmodified `hydroswarm.training.causal_prefix.
build_scenario_pool` train/validation/calibration/development_holdout splits
(the same governance every M1/M3/M4/M5/M6/M7/M8 script already uses) --
physical incidents are disjoint across splits by construction (frozen at
corpus-generation time, before any causal-prefix view is derived). No locked
data (`locked_final_test`/`locked_topology_test`) is used at any point. No
correction to this governance is required for the temporal-feature change:
Arms B/C consume the SAME physical scenario pool and split assignment as Arm
A, differing only in feature-builder/architecture keyword arguments applied
uniformly within each split.

## 2. Seeds

Two seeds for screening, per the frozen v5 seed policy (`HYDROCORE_V5_
EXPERIMENT_PROTOCOL.md` Section 2) and M1's own precedent
(`scripts/hydrocore_v5/run_m1_matrix.sh`): **20260814, 31874**, run for ALL
THREE arms. A third seed, **20260815** (chosen now, before any result is
seen, following the existing date-seed convention), is run ONLY for whichever
arm/representation the screening pass provisionally selects, before that
arm's conclusion is treated as promotion-relevant -- mirroring
`run_m1_matrix.sh`'s own documented practice ("a third seed is added later
only for whichever arm the promotion rule provisionally selects").

## 3. Parameter budget

Arms A and B: identical architecture, ~4,182,612 parameters (the current
HydroCore "small" variant, `SHARED_MODEL_CONFIG` from `run_m1_arm.py`,
unmodified). Arm C: same variant with `temporal_feature_dim=7` (+1),
`quality_feature_dim=5` (+1), `elapsed_time_normalization="fixed_scale"` --
measured parameter delta **+384 parameters (+0.0092%)**, far inside the
predeclared <=2% preferred / <=5% maximum budget.

## 4. Optimizer / schedule / epochs / batch size / checkpoint rule

Identical for all three arms: `configs/training-v5-causal.yaml`, loaded via
`TrainingConfig.from_yaml(..., require_complete_task_weights=True)`, UNCHANGED
from Milestone 1 -- AdamW-style optimizer per `hydroswarm.training.trainer`,
`learning_rate=0.0003`, cosine schedule, `warmup_steps=10`, `epochs=20`,
`batch_size=2`, `gradient_accumulation_steps=4`, `gradient_clip_norm=1.0`,
`early_stopping_patience=5`, checkpoint selection = lowest validation loss at
full-history (depth=25) evaluation, `deterministic=True`. Only `seed` and
`gradnorm_log_every_n_batches` (a compute-cost-only knob, unchanged from M1)
are overridden in memory; the committed YAML is not edited.

## 5. Task weights

Identical `task_weights` block in `configs/training-v5-causal.yaml` for all
three arms -- this milestone changes feature REPRESENTATION, not multitask
weighting (that remains Milestone 2's frozen decision, `PCGRAD_JUSTIFIED`,
`pcgrad_enabled=false`, unchanged here).

## 6. Causal-prefix policy

All three arms use Arm-A's full-history-control depth policy
(`hydroswarm.training.causal_prefix.ARM_POLICIES["A"]`, the corrected
full-history control), not a re-run of Milestones 1B/C's uniform/early-
weighted prefix policies (out of scope for M8.7 -- M8.7 is a representation
study at Milestone-1's OWN winning depth policy, not a re-run of the
depth-policy ablation itself). Held-out evaluation still exercises multiple
depths (Section 9 below) exactly as `evaluate_m1_depths.py` already does.

## 7. Temporal transformation definitions (frozen before evaluation)

- **CURRENT_CONTROL (Arm A)**: `HydraulicFeatureBuilder.build(...,
  unobserved_age_sentinel="incident_elapsed")` (the default -- unchanged
  behavior), `HydroCore(temporal_feature_dim=6, quality_feature_dim=4,
  elapsed_time_normalization="window_relative")` (the default).
- **AGE_FIX_ONLY (Arm B)**: `HydraulicFeatureBuilder.build(...,
  unobserved_age_sentinel="fixed")` -- never-observed nodes' `measurement_age`
  becomes the fixed constant `NEVER_OBSERVED_MEASUREMENT_AGE_SENTINEL_SECONDS
  = 86400.0` (reads as exactly 1.0 after the column's own 86,400s
  normalization divisor -- "as stale as this scale considers a reading to
  be", not an arbitrary new magnitude) instead of the incident's raw elapsed
  `now`. Architecture identical to Arm A.
- **AGE_FIX_PLUS_RELATIVE_TIME (Arm C)**: Arm B's age fix, PLUS
  `HydraulicFeatureBuilder.build(..., include_relative_gap_feature=True)`
  (a new per-timestep `time_since_previous_report` channel, seconds since
  THAT SENSOR's own previous report, 0.0 for a series' first point, divided
  by the same 86,400s convention) and `HydroCore(temporal_feature_dim=7,
  quality_feature_dim=5, elapsed_time_normalization="fixed_scale")` (the
  explicit `timestamps` pathway's sinusoidal phase now divides by a FIXED
  86,400s constant instead of the window's own span, so it preserves actual
  elapsed magnitude instead of normalizing it away).
- Optional Arm D (`AGE_FIX_REMOVE_INERT_TIMESTAMP_PATH`): Arm B with
  `batch["timestamps"]` neutralized at inference/training time (the same
  neutralization M8.6 Section 8 arm B already implements) -- run only if
  trivial and only after A/B/C are complete (Section 3 of the milestone
  instructions).

## 8. Numerical tolerances (predeclared, not relaxed after results)

- Structural invariance (node/edge/sensor-order, relabeling): max abs
  posterior diff <= 1e-4 (identical to M8.6 Section 3-6).
- Timestamp-origin invariance (Arms B/C only): max abs posterior diff <=
  1e-4 (STRICTER than M8.6's Section-7 tolerance of 1e-2, since Arms B/C are
  specifically expected to have REMOVED the origin-dependence -- if fixed,
  discrepancy should be at the same float32 noise floor as every other
  structural-invariance class, not merely "smaller than before").
- Guardrails (Section 16 of the milestone instructions, restated verbatim
  here as part of the frozen protocol): EARLY top1 must not regress >5
  percentage points vs Arm A; MATURE top1 must not regress >3 percentage
  points; MRR must not materially regress; conformal marginal coverage must
  not materially fall below nominal (1-alpha=0.90); candidate sets must
  remain operationally useful (mean size well under the network's node
  count); origin invariance must be restored for B/C; no NaN/Inf in any
  arm's outputs; fail-closed OOD/disagreement behavior unchanged.

## 9. Standard evaluation depths

Matches Milestone 1's own depth grid: report-count depths 1, 2, 3, 4, 6, 12,
mature/full (25), aggregated into EARLY (1-3), MID (4-6), MATURE (12, 25).

## 10. Statistical comparison procedure

Paired per-incident comparison (same development_holdout incidents, same
depth, across arms) wherever both arms' rows exist for that incident/depth.
Where a promotion-relevant A-vs-B or A-vs-C metric difference is reported at
the 3-seed stage, a paired bootstrap over incidents (2,000 resamples,
seed=20260815, matching this document's own third-seed choice for
reproducibility) is used to report a 90% interval on the difference. Raw
per-incident, per-seed results are preserved in
`reports/evaluation/hydrocore-v5/m8-7-results.json` and
`m8-7-temporal-diagnostics.json`, never collapsed to a mean before being
recorded.

## 11. Promotion criteria

Exactly the four outcomes (A-D) and the `CORRECTNESS_FIX_REQUIRED` /
`RELATIVE_TIME_CAPABILITY_GAIN_VALIDATED` decisions specified in the M8.7
milestone instructions, Sections 17-18 -- reproduced in
`reports/evaluation/hydrocore-v5/m8-7-summary.md`'s own Decisions section,
not restated here to avoid two documents drifting apart.
