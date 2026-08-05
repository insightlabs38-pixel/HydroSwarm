# Bundle F finalist selection recommendation

Source: `reports/results/v3/stage3-finalist-training.json` (Stage 3 finalist training, 6 runs:
{E2, E0, E1} x seeds {20260810, 20260811}, all completed successfully, zero failures, ~12.0h
total wall time). This document is a **recommendation for human review**, not an autonomous
final selection -- per overnight-plan.txt's Stage 6, a true final selection also requires Stage 4
controls (HydroMono-S, no-adapter HydroCore-S, current baseline, classical-only), Stage 5
(updated HydroCore-M, gated on this selection), and full-trajectory evaluation, none of which
have run yet. The agent operating this run is explicitly not authorized to promote a new default
checkpoint automatically or open the locked final test without a precommitted, satisfied
selection record -- this document is input to that record, not the record itself.

## Per-seed results

All three finalists hit the 2-hour per-run runtime ceiling (`stop_reason: runtime_budget`) at
11-13 epochs rather than early-stopping (patience 3 was never reached) -- validation loss was
still improving, slowly, when each run was cut off. This means these numbers likely understate
each configuration's ceiling; a longer per-run budget (Stage 6, "final candidate" treatment)
would likely improve all three further, plausibly by comparable amounts.

| Finalist (2-seed avg) | val top1 | val ECE | dev_holdout top1 | dev_holdout ECE | OOD unseen-topology top1 | OOD unseen-topology coverage | OOD severe-missingness top1 | OOD severe-missingness coverage |
|---|---|---|---|---|---|---|---|---|
| E0 (baseline, prior_mode=feature_and_logit) | 0.7170 | 0.0166 | 0.7129 | 0.0156 | 0.5125 | 0.7911 | 0.6524 | 0.8955 |
| E1 (prior_mode=feature_only) | 0.7205 | 0.0190 | 0.7068 | 0.0209 | **0.5304** | **0.8482** | 0.6507 | 0.8887 |
| E2 (prior_mode=logit_only) | **0.7261** | 0.0182 | **0.7154** | 0.0254 | 0.4946 | 0.8232 | **0.6610** | 0.8716 |

All three conformal calibration artifacts hit their target coverage (0.9073 measured against a
0.90 nominal target from alpha=0.1) almost identically -- the conformal procedure itself is
working correctly regardless of which architecture it wraps; this is not a differentiator
between the three.

## The trade-off

**E2** (logit_only prior injection) has the best in-distribution accuracy: the highest
validation top-1 (+0.6pp over E1, +0.9pp over E0) and the highest development-holdout top-1.

**E1** (feature_only prior injection) has the best out-of-distribution generalization to a
genuinely unseen topology: +3.6pp top-1 and +2.5pp measured calibrated coverage over E2 on
`UNSEEN_TOPOLOGY`, the single hardest and most safety-relevant distribution shift Cycle B
exercises (a topology neither trained on nor calibrated against). E1's in-distribution accuracy
is very close behind E2's (0.6pp on validation, 0.9pp on development_holdout) -- not a
meaningful sacrifice for the OOD gain.

**E0** (the current baseline / default `feature_and_logit` prior injection -- the architecture
behind the currently promoted checkpoint) is not best on any single metric, but has the best
in-distribution calibration honesty (lowest val/dev ECE) and is competitive everywhere else.

## Recommendation: E1 (prior_mode=feature_only)

This run recommends **E1** as the strongest S candidate to carry into Stage 4 controls and
Stage 5, prioritizing out-of-distribution generalization over a marginal in-distribution
accuracy edge. This follows directly from the product's own stated safety mandate (this run's
key restrictions explicitly forbid weakening OOD, calibration, abstention, or human-approval
boundaries) and from HydroSwarm's product description in overnight-plan.txt itself: the system
must be "explicit about uncertainty and operating range" and must have "OOD and invalid
calibration ... suppress unsafe recommendations." A configuration that localizes an unseen
topology's contamination source correctly 3.6 points more often, with better-calibrated
coverage under that same shift, is directly serving that mandate -- more so than winning
validation top-1 by 0.6 points on topologies the model has already trained on.

This is a judgment call, not a mechanical read of a single predeclared score (Stage 3, unlike
Stage 2, does not predeclare a single finalist-selection formula in the plan -- Stage 6 instead
lists the dimensions to weigh: ECE, coverage/set size, OOD behavior, abstention/unsafe-non-
abstention). It is recorded here explicitly, with the losing metric (E2's in-distribution edge)
stated plainly, so a human reviewer can override it if they weigh the trade-off differently.

## What this recommendation does not yet cover

- **Stage 4 controls** (HydroMono-S, no-adapter HydroCore-S, current-architecture baseline,
  classical-only baseline) have not been trained under the same budget for comparison. It
  remains possible a control matches or beats E1 -- Stage 4 exists specifically to check this
  before treating any screened architecture as a foregone conclusion.
- **Full-trajectory evaluation** (Task 7.3) has not run for any finalist; Cycle B does not
  generate Scout/Strategist trajectory labels (documented in
  `data/learning-v2/cycle-b/dataset-report.json`'s limitations).
- **Abstention / unsafe-non-abstention recording**, one of Stage 6's explicit requirements, has
  not been computed for any finalist -- this would require running the full live inference
  pipeline (OOD detector, fusion, control-action logic) per scenario, not just the offline
  batch prediction path this script uses.
- **Runtime budget was the binding constraint for all six runs**, not architecture capacity or
  convergence -- these numbers should be read as "best achieved in 2 hours," not "best each
  architecture can do." A Stage 6 "final candidate" longer run (the plan explicitly allows a
  longer budget for the actual finalist) could change the ranking.
