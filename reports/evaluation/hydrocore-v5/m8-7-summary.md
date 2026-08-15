# Milestone 8.7 summary: corrected temporal representation and matched-size retraining

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` Section 8.5. Full protocol frozen in `docs/evaluation/HYDROCORE_V5_M8_7_PROTOCOL.md` before any arm was trained.

## Methodological note: AGE_FIX_ONLY's training data coincided numerically with CURRENT_CONTROL's

All three arms train exclusively at full-history depth (25, matching Arm A's own established depth policy -- frozen protocol Section 6). At full history, `now` (the incident's last-observed timestamp, used by the ORIGINAL/buggy never-observed-node age fallback) always equals the window duration, 86,400s -- the EXACT same value chosen for AGE_FIX_ONLY's fixed sentinel. Verified directly (not assumed): a diff of `node_features[..., 9]` between `unobserved_age_sentinel="incident_elapsed"` and `"fixed"` at depth=25 on a real training scenario was exactly 0.0 for every node. AGE_FIX_ONLY's TRAINING data was therefore numerically identical to CURRENT_CONTROL's for this corpus (confirming AGE_FIX_ONLY-seed20260814's best_validation_loss matched CURRENT_CONTROL-seed20260814's to full float64 precision) -- the age-fix's effect is invisible in aggregate full-history training loss BY CONSTRUCTION, not because the fix failed to apply. Its effect shows up correctly at EVALUATION time instead (shorter causal-prefix depths and the timestamp-origin-translation tests below), where `now` genuinely varies and the fixed sentinel does not. This is expected and does not undermine AGE_FIX_ONLY's origin-invariance result -- that property is a function of the (always-fixed-at-inference) feature computation, independent of what the checkpoint's weights happened to be trained on.

## Standard localization (development_holdout, screening seeds 20260814/31874, averaged)

| arm | EARLY top1 | MID top1 | MATURE top1 | MRR | all finite | OOD finite |
|---|---|---|---|---|---|---|
| AGE_FIX_ONLY | 0.4528 | 0.8958 | 0.9896 | 0.8295 | True | True |
| AGE_FIX_PLUS_RELATIVE_TIME | 0.4514 | 0.8979 | 0.9958 | 0.8295 | True | True |
| CURRENT_CONTROL | 0.4556 | 0.8979 | 0.9896 | 0.8303 | True | True |

## Timestamp-origin invariance (Section 10)

| arm | golden-reference restored | dev-grid-25 restored |
|---|---|---|
| AGE_FIX_ONLY | True | True |
| AGE_FIX_PLUS_RELATIVE_TIME | True | True |
| CURRENT_CONTROL | False | False |

## Temporal sensitivity (Sections 7/8, representative seed 31874)

| arm | post-onset N=2 identical-fraction | N=2 mean L1 | post-onset N=3 identical-fraction | N=3 mean L1 | matched-physical-time max spread (pp) |
|---|---|---|---|---|---|
| AGE_FIX_ONLY | 1.000 | 0.000414 | 0.938 | 0.091086 | 0.00 |
| AGE_FIX_PLUS_RELATIVE_TIME | 1.000 | 0.000275 | 0.938 | 0.086612 | 0.00 |
| CURRENT_CONTROL | 1.000 | 0.000402 | 0.938 | 0.091344 | 0.00 |

The discrete predicted-node identity fraction is IDENTICAL across all three arms (the golden-reference network's posteriors are already highly confident/saturated, e.g. ~0.9999 on the true source, for most post-onset cases, leaving little room for any representation change to flip a discrete top-1 decision across cadences). The underlying RAW posteriors do differ measurably between arms (confirmed directly: AGE_FIX_PLUS_RELATIVE_TIME's per-row probability vectors are numerically distinct from CURRENT_CONTROL's at the same incident/cadence, e.g. 0.99989 vs 0.99997 on one representative case) -- but AGE_FIX_PLUS_RELATIVE_TIME's mean L1 cross-cadence distance at N=2 (0.000275) is actually LOWER than CURRENT_CONTROL's (0.000402), i.e. trending toward LESS cross-cadence sensitivity, not more. This is a genuine, honestly-reported negative result for the capability question, not a wiring defect: the relative-gap feature is confirmed active (verified via raw feature-tensor and posterior differences) but does not materially change the model's discrete or aggregate cross-cadence behavior on this evaluation population.

## Irregular telemetry robustness (Section 11, representative seed 31874)

| arm | clean | jitter | unequal-intervals | delayed | gaps |
|---|---|---|---|---|---|
| AGE_FIX_ONLY | 0.750 | 0.542 | 0.500 | 0.750 | 0.708 |
| AGE_FIX_PLUS_RELATIVE_TIME | 0.750 | 0.542 | 0.500 | 0.750 | 0.750 |
| CURRENT_CONTROL | 0.750 | 0.542 | 0.500 | 0.750 | 0.708 |

## Calibration (B_DEPTH_AWARE, alpha=0.1, representative seed 31874)

| arm | marginal coverage | mean candidate-set size | singleton rate |
|---|---|---|---|
| AGE_FIX_ONLY | 0.899 | 1.865 | 0.545 |
| AGE_FIX_PLUS_RELATIVE_TIME | 0.904 | 1.843 | 0.583 |
| CURRENT_CONTROL | 0.899 | 1.863 | 0.546 |

## Guardrails (Section 16, predeclared, not relaxed after results)

| arm | EARLY regression (pp) | MATURE regression (pp) | MRR regression | coverage ok | candidate-set ok | guardrails passed |
|---|---|---|---|---|---|---|
| AGE_FIX_ONLY | 0.28 | 0.00 | 0.0008 | True | True | True |
| AGE_FIX_PLUS_RELATIVE_TIME | 0.42 | -0.62 | 0.0008 | True | True | True |

## Decisions

CORRECTNESS_FIX_REQUIRED: **YES**
RELATIVE_TIME_CAPABILITY_GAIN_VALIDATED: **NO**

AGE_FIX_ONLY restores timestamp-origin invariance: True
AGE_FIX_PLUS_RELATIVE_TIME restores timestamp-origin invariance: True

**PRIMARY DECISION: PROMOTE_AGE_FIX_ONLY**

Selected representation for future M9 small model: AGE_FIX_ONLY
M9 scientifically unblocked: YES

locked tests opened: before=False, after=False. No model promoted to production. No safety/authority semantics changed.
