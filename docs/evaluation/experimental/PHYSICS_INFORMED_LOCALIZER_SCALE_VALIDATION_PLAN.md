# physics-informed-localizer-scale-validation: plan (pre-registration)

Branch: `exp/physics-informed-localizer-scale-validation`. Based on the
completed `exp/physics-informed-localizer-validation` at
`0534fe143fa5c1068de5880fe24d5958ad87406a` (that branch's final report:
`reports/evaluation/physics-informed-localizer-validation/FINAL_REPORT.md`).
**Experimental, non-release.** No change to `models/hydrocore-v5-release`,
`data/locked/`, HydroSwarm v0.2.1, HydroCore-v5's release artifacts, M11.6
locked evidence, any hackathon claim, or any governance module. This is a
**focused confirmation experiment**, not an architecture search: no GNN, no
attention-stack expansion, no new physics features, no unrelated
hyperparameter tuning, no model-size scaling, no gate relaxation. The
completed validation branch's reports, seed directories, manifests,
results, and plan are treated as immutable and are not modified.

## 0. What the completed validation study found (recap, treated as fact)

- `C_FULL` (all 3 physics-compatibility features) reproduced the original
  pilot's unseen-topology Top-1 gain in direction across all 3 pre-declared
  seeds (20260814/20260901/20260915): pooled +3.10pp over `A_CONTROL`,
  90% CI [+1.31, +5.00]pp.
- A parameter-matched generic-capacity control (`A_CAPACITY_MATCHED`, same
  ~+4.6% parameter delta, no candidate/graph/physics information) did
  **not** reproduce the gain (pooled -1.07pp, CI includes zero, and
  `C_FULL` beats it head-to-head).
- Candidate conditioning without any physics feature
  (`B_CANDIDATE_CONDITIONED`) was **significantly worse** than
  `A_CONTROL`, pooled.
- Physics-feature ablation identified `C2`
  (`hop_magnitude_compatibility`) as the dominant driver: +3.45pp vs
  `A_CONTROL`, statistically indistinguishable from `C_FULL`, positive in
  all 3 seeds.
- `C1` (`nearest_sensor_log_concentration`) contributed a smaller but real,
  significant increment (+1.67pp vs `A_CONTROL`).
- `C3` (`hop_arrival_time_compatibility`) contributed no measurable Top-1
  value and significantly regressed Top-3 when isolated.
- `C_FULL`/`C2` significantly improved the low-centrality and long-distance
  hard subgroups once pooled across 3 seeds.
- No known-topology, calibration, OOD, or governance regression was
  observed for any arm.
- `C2` alone showed a small **negative** unseen-topology Top-3 point
  estimate whose CI included zero -- flagged, not resolved.
- Effect *direction* was stable (3/3 seeds positive for `C_FULL`/`C2`) but
  effect *magnitude* varied materially by seed (+6.4pp / +1.1pp / +1.8pp).
- The report's own recommendation (Section 17): drop `C3`, test `C1+C2` on
  fresh seeds, and pre-register an effect-size (not just direction) bar.

**This branch answers exactly that recommendation and nothing more.**

## 1. Primary hypothesis and endpoint (pre-registered)

**H_primary**: the simplified two-feature `C1_C2` candidate-conditioned
localizer (nearest-sensor concentration + hop-magnitude compatibility,
`C3` zeroed) preserves the replicated unseen-topology Top-1/MRR benefit
found for `C_FULL`/`C2`, while avoiding or reducing `C2`'s possible Top-3
ranking-shape tension.

**Primary endpoint (fixed before any fresh-seed training began):
`ood-UNSEEN_TOPOLOGY` Top-1 accuracy.** All primary conclusions are based
on this endpoint. Secondary endpoints: unseen-topology Top-3; unseen-
topology MRR; true-source rank; known-topology (`validation`,
`development_holdout`) performance; low-centrality subgroup; long
source-to-sensor-distance subgroup; calibration/OOD proxy behavior;
abstention behavior; hard safety counters (reported as 0 for the same
documented harness-scope reason as the completed validation study --
Section 6 below).

## 2. Pre-declared fresh seeds (do not select or replace based on results)

`SEEDS = (20260929, 20261013, 20261027)` -- three new, independent
confirmation seeds, disjoint from the completed study's own
`(20260814, 20260901, 20260915)`. Fixed before any arm was trained on this
branch. These seeds are never reused to overwrite or extend any prior
committed run artifact.

## 3. Arms (exactly three; priority order)

All arms share HydroCore-v5's `small` variant, `event_control_heads=True`,
identical training config (`configs/training-v5-causal.yaml`, 6 epochs,
`fp32=True`, `deterministic=True`, CPU), identical corpus
(`data/learning-v2/cycle-b2/tensors-normalized`), and identical per-seed
train/validation/development-holdout/calibration/OOD splits across every
arm at that seed (same stratified-family-sampling / capped-index logic as
the completed validation study, same 600-example training size, same
validation/development/OOD evaluation sizes).

| arm | `localizer_mode` | active physics columns | purpose |
|---|---|---|---|
| `A_CONTROL` | `default` | -- | existing frozen-equivalent control (unmodified `source_node_head`) |
| `C2` | `candidate_conditioned` | `hop_magnitude_compatibility` only | independent fresh-seed replication of the previously identified driver |
| `C1_C2` | `candidate_conditioned` | `nearest_sensor_log_concentration` + `hop_magnitude_compatibility` | primary comparison: does adding C1 to C2 preserve/improve the benefit and reduce the Top-3 tension? |

`C2` and `C1_C2` use **identical model architecture and parameter count**
(`localizer_physics_feature_dim=3` always, same
`CandidateConditionedLocalizer`); only the active physics-feature input
columns differ, exactly as in the completed validation study's own
`_mask_physics_columns` ablation mechanism (reused unmodified). `C3`
(`hop_arrival_time_compatibility`) is zeroed/disabled in both `C2` and
`C1_C2`.

`C_FULL`, `C3`, `B_CANDIDATE_CONDITIONED`, and `A_CAPACITY_MATCHED` are
**not** re-run on this branch -- those questions are already answered by
the completed validation study, and re-running them would not serve this
branch's narrower confirmation question. They are only revisited if a
required sanity check reveals an implementation defect (not to chase a
better result).

## 4. Compute budget

3 arms x 3 seeds = 9 full training/evaluation runs, ~10-15 minutes each on
CPU based on the completed validation study's own recorded
`elapsed_seconds` (757-919s per run) -- target ~2 hours of unattended
wall time. No additional seeds, arms, epochs, or pairwise experiments are
run merely because compute remains; any leftover time is spent on
validation, statistical analysis, tests, and report quality.

## 5. Effect-size replication bar (pre-registered, fixed before training)

For `C1_C2` vs `A_CONTROL`, pooled fresh-seed `ood-UNSEEN_TOPOLOGY` Top-1:

- **Strong replication**: point estimate > 0, positive direction in at
  least 2/3 fresh seeds, AND pooled 90% CI lower bound >= +1.0
  percentage point.
- **Weak / inconclusive replication**: point estimate > 0, but lower CI
  bound < +1.0pp or strong seed instability remains.
- **Failed replication**: pooled point estimate <= 0, or a majority of
  fresh seeds regress.

This bar is intentionally stronger than "CI excludes zero" and is
informed by (not identical to) the completed study's own +1.31pp pooled
lower bound. It is not altered after seeing results.

## 6. Safety / governance (unchanged)

Identical to the completed validation study: this pilot-scale
localization-only harness never exercises the sampling/planning/execution
control loop that produces non-zero `hard_safety_counters` at the M11.6
evaluation tier, so every arm reports all eight counters as 0 -- because
those code paths are never invoked here, not because they were
independently re-verified at that tier. No governance module
(`hydroswarm.inference.ood`, `hydroswarm.calibration.conformal`, any
actionability gate, conformal calibration logic, simulator verification,
or approval boundary) is modified by this branch. This localization-only
experimental harness does not exercise the full sampling/planning/
execution control loop, so zero hard counters must not be represented as
an independent re-validation of the full M11.6 safety tier.

## 7. Statistical convention (unchanged, "HydroSwarm's established convention")

Paired bootstrap: 2000 resamples, deterministic bootstrap seed `20260826`
(same as the completed validation study), 90% percentile interval. Per-seed
paired comparisons match by `scenario_id` within a seed; pooled cross-seed
bootstrap concatenates every seed's own paired values matched by
`(seed, scenario_id)`, never mixing examples across differently-sampled
splits under the same id. Both individual per-seed results AND the pooled
fresh-three-seed summary are always reported together -- never only the
pooled figure. A separate, explicitly-labeled **descriptive** six-seed
cross-study meta-summary is additionally produced for `A_CONTROL` and `C2`
only (combining the prior committed seeds with the fresh ones); `C1_C2` has
no prior-study seeds and is never given six-seed treatment.

## 8. Git / Git LFS scope (pre-registered, bandwidth-scarce)

Only these existing Cycle-B2 normalized splits are required and fetched:
`train/`, `validation/`, `calibration/`, `development_holdout/`,
`ood-UNSEEN_TOPOLOGY/` under
`data/learning-v2/cycle-b2/tensors-normalized/`. No other LFS area
(`ood-SEVERE_MISSINGNESS`, `cycle-b2-control-v2`, `cycle-b2-ood-extension`,
trajectory corpora, Scout/Strategist corpora, `joint-v4`, model
candidate/control checkpoints, migration archives) is fetched unless the
experiment unexpectedly proves it absolutely requires one -- in which case
the experiment is modified to avoid the dependency rather than broadening
the fetch. No already-downloaded LFS objects are pruned.

## 9. Decision rule (fixed before results)

- **REJECT** if `C1_C2`'s pooled fresh-seed point estimate vs `A_CONTROL`
  is <= 0, or a majority of fresh seeds regress, or a real known-topology/
  calibration regression is found.
- **CONTINUE_RESEARCH** if `C1_C2` shows weak/inconclusive replication
  (positive but below the +1.0pp lower-CI bar, or seed-unstable), or if it
  is not clearly better than `C2` alone, or if the Top-3 tension is neither
  resolved nor clearly worsened.
- **CANDIDATE_FOR_FULL_SCALE_VALIDATION** only if the selected physics
  representation (`C2` or `C1_C2`, whichever wins) meets the Section 5
  effect-size bar (or provides equivalently compelling fresh-seed
  evidence), remains directionally stable across the 3 fresh seeds,
  survives the paired comparison against `A_CONTROL`, shows no meaningful
  known-topology/calibration regression, and resolves or at least does not
  worsen the Top-3 concern.

This branch is **not merged**. All work is committed and pushed only to
`exp/physics-informed-localizer-scale-validation`.
