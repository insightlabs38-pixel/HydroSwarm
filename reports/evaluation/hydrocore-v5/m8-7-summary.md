# Milestone 8.7 summary: corrected temporal representation and matched-size retraining

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` Section 8.5. Full protocol frozen in `docs/evaluation/HYDROCORE_V5_M8_7_PROTOCOL.md` before any arm was trained.

This document has two parts. Part 1 is the ORIGINAL TWO-SEED SCREENING RESULT (seeds 20260814, 31874; all three arms: CURRENT_CONTROL, AGE_FIX_ONLY, AGE_FIX_PLUS_RELATIVE_TIME), reproduced verbatim below -- unmodified, not overwritten, not erased. Part 2 is the THREE-SEED PROMOTION-CONFIRMATION RESULT (M8.7 closure run: preregistered third seed 20260815, AGE_FIX_ONLY only, per the frozen protocol Section 2). The two parts are kept separate deliberately so neither result is misread as superseding or silently absorbing the other.

## Methodological note: AGE_FIX_ONLY's training data coincided numerically with CURRENT_CONTROL's

All three arms train exclusively at full-history depth (25, matching Arm A's own established depth policy -- frozen protocol Section 6). At full history, `now` (the incident's last-observed timestamp, used by the ORIGINAL/buggy never-observed-node age fallback) always equals the window duration, 86,400s -- the EXACT same value chosen for AGE_FIX_ONLY's fixed sentinel. Verified directly (not assumed): a diff of `node_features[..., 9]` between `unobserved_age_sentinel="incident_elapsed"` and `"fixed"` at depth=25 on a real training scenario was exactly 0.0 for every node. AGE_FIX_ONLY's TRAINING data was therefore numerically identical to CURRENT_CONTROL's for this corpus (confirming AGE_FIX_ONLY-seed20260814's best_validation_loss matched CURRENT_CONTROL-seed20260814's to full float64 precision) -- the age-fix's effect is invisible in aggregate full-history training loss BY CONSTRUCTION, not because the fix failed to apply. Its effect shows up correctly at EVALUATION time instead (shorter causal-prefix depths and the timestamp-origin-translation tests below), where `now` genuinely varies and the fixed sentinel does not. This is expected and does not undermine AGE_FIX_ONLY's origin-invariance result -- that property is a function of the (always-fixed-at-inference) feature computation, independent of what the checkpoint's weights happened to be trained on.

---

# PART 1 -- TWO-SEED SCREENING RESULT (frozen, seeds 20260814/31874, all three arms)

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

## Screening decisions

CORRECTNESS_FIX_REQUIRED: **YES**
RELATIVE_TIME_CAPABILITY_GAIN_VALIDATED: **NO**

AGE_FIX_ONLY restores timestamp-origin invariance: True
AGE_FIX_PLUS_RELATIVE_TIME restores timestamp-origin invariance: True

**PRIMARY SCREENING DECISION: PROMOTE_AGE_FIX_ONLY** (provisional -- promotion-relevant only after the preregistered third seed, per frozen protocol Section 2)

Selected representation for future M9 small model: AGE_FIX_ONLY (provisional pending third seed)

locked tests opened: before=False, after=False. No model promoted to production. No safety/authority semantics changed.

---

# PART 2 -- THREE-SEED PROMOTION-CONFIRMATION RESULT (M8.7 closure: third seed 20260815, AGE_FIX_ONLY only)

Per the frozen protocol (Section 2), the preregistered third seed, **20260815**, was trained and evaluated ONLY for the provisionally selected arm, **AGE_FIX_ONLY**, using the exact same configuration (train/validation/calibration/development_holdout splits, causal-prefix policy, architecture, optimizer, schedule, task weights, checkpoint-selection criterion) as AGE_FIX_ONLY's seeds 20260814/31874. `CURRENT_CONTROL` and `AGE_FIX_PLUS_RELATIVE_TIME` were NOT retrained or re-decided; their 2-seed results above are unchanged and still authoritative for those two arms. Full machine-readable detail: `reports/evaluation/hydrocore-v5/m8-7-closure.json`, and the new per-seed run record `reports/evaluation/hydrocore-v5/m8-7-runs/AGE_FIX_ONLY-seed20260815.json`.

## Training, seed 20260815

- checkpoint SHA256: `728484a261bf8e7d41138ba44501df153856a1a1448b84eeb1df090091f3b387`
- parameter count: 4,182,612 (identical to seeds 20260814/31874 -- same architecture)
- train/validation manifest hashes: identical to seeds 20260814/31874 (same splits, verified byte-for-byte)
- epochs completed: 20 / 20 (stop_reason=`maximum_epochs`, not early-stopped)
- best epoch: 18; best validation loss: 1.2790485370159148
- numerically stable: no NaN/Inf anywhere; all outputs finite; OOD logits finite

## AGE_FIX_ONLY, per-seed standard localization (development_holdout)

| seed | EARLY top1 | MID top1 | MATURE top1 | overall MRR | all finite | OOD finite |
|---|---|---|---|---|---|---|
| 20260814 | 0.4500 | 0.9042 | 0.9917 | 0.8300 | True | True |
| 31874 | 0.4556 | 0.8875 | 0.9875 | 0.8291 | True | True |
| 20260815 | 0.4500 | 0.9000 | 0.9917 | 0.8290 | True | True |
| **3-seed mean** | **0.4519** | **0.8972** | **0.9903** | **0.8293** | True | True |
| 3-seed population stdev | 0.0026 | -- | 0.0020 | 0.0004 | -- | -- |

Low cross-seed variance (stdev EARLY/MATURE/MRR all <0.3pp / <0.05) -- **no seed-specific instability detected**. This is a valid, non-cherry-picked third seed: it was trained once, evaluated once, and reported as obtained.

## Timestamp-origin invariance, seed 20260815 (Section 7 of the closure instructions)

| network | offsets tested | sparse evidence | no-event evidence | max abs posterior diff | result |
|---|---|---|---|---|---|
| golden-reference | +1h, +24h, +7d | pass | pass | 0.0 | **PASS** |
| dev-grid-25 | +1h, +24h, +7d | pass | pass | 0.0 | **PASS** |

Tolerance (frozen, not relaxed): max abs posterior diff <= 1e-4. Both networks measured exactly 0.0 -- at the float32 noise floor, not merely "smaller than before."

## Calibration, seed 20260815 (B_DEPTH_AWARE, alpha=0.1, governed calibration split)

| metric | value |
|---|---|
| marginal coverage | 0.9179 |
| mean candidate-set size | 1.887 |
| singleton rate | 0.565 |
| EARLY coverage | 0.825 |
| MID coverage | 0.975 |
| MATURE coverage | 1.000 |

Frozen calibration guardrail (marginal coverage >= 0.85): **PASSED**. As with the 2-seed screening result, EARLY conditional coverage (0.825) sits below the nominal 0.90 target even though marginal coverage passes -- this is preserved here explicitly as a **documented limitation**, not silently resolved by the third seed.

## Three-seed AGE_FIX_ONLY vs 2-seed CURRENT_CONTROL guardrails (Section 16, frozen, not relaxed)

| comparison | EARLY regression (pp) | MATURE regression (pp) | MRR regression | coverage ok | candidate-set ok | guardrails passed |
|---|---|---|---|---|---|---|
| AGE_FIX_ONLY (3-seed mean) vs CURRENT_CONTROL (2-seed mean, unchanged) | 0.37 | -0.07 (improved) | 0.0010 | True | True | **True** |

## Paired incident bootstrap (Section 10/12 of the frozen protocol)

AGE_FIX_ONLY (seed 20260815) vs CURRENT_CONTROL (representative seed 31874), paired per development_holdout incident, mean top1 averaged over the standard depth grid, 2,000 resamples, bootstrap seed 20260815, 90% interval:

- observed mean difference: **+0.0012** (AGE_FIX_ONLY minus CURRENT_CONTROL)
- 90% CI: **[-0.0095, +0.0107]** -- includes zero
- n=120 incidents

The CI including zero confirms AGE_FIX_ONLY is statistically indistinguishable from CURRENT_CONTROL on standard localization capability -- consistent with AGE_FIX_ONLY being promoted as a **correctness fix**, not a capability improvement.

## Structural regression / stages_through forwarding

- `tests/scientific/test_temporal_representation_m8_7.py` (9 tests, includes `test_stages_through_forwards_feature_kwargs`, `test_node_order_permutation_equivariance_holds_under_arm_c_options`, `test_relative_gap_feature_is_causal_under_truncation`, origin-invariance and sentinel-semantics tests): **9/9 PASSED**, confirmed BEFORE training seed 20260815 (regression from M8.7's own training-time bug fix remains fixed; no code changes were needed or made).
- `tests/unit/test_permutation.py` (node-order/edge-order/sensor-order/relabeling equivariance): **13/13 PASSED**.

## Safety / authority / OOD

- All AGE_FIX_ONLY-seed20260815 outputs finite; OOD logits finite.
- No safety threshold, authority semantics, WNTR-authority, planning-authority, or active-sampling-authority code was modified during this closure run.
- No calibration alpha, K, or fallback-hierarchy change.
- Locked final/topology evaluation: unopened throughout (`locked_test_opened_before`/`after` = False in every script run this closure).

## Closure promotion gate (Section 13 of the closure instructions -- all 10 criteria)

| # | criterion | result |
|---|---|---|
| 1 | seed 20260815 trains successfully, numerically stable | PASS |
| 2 | timestamp-origin invariance passes both networks | PASS |
| 3 | EARLY top1 does not materially violate guardrail (<=5pp) | PASS (0.37pp) |
| 4 | MATURE top1 does not materially violate guardrail (<=3pp) | PASS (-0.07pp, improved) |
| 5 | MRR remains within guardrail | PASS (0.0010) |
| 6 | marginal conformal coverage satisfies requirement | PASS (0.899 repr.-seed / 0.918 seed-20260815) |
| 7 | candidate-set size operationally acceptable | PASS (~1.87-1.89 vs 6-node golden network) |
| 8 | no OOD/safety/authority regression | PASS |
| 9 | no seed-specific instability invalidates representation | PASS |
| 10 | locked data remains unopened | PASS |

**All 10 criteria pass.**

## FINAL M8.7 DECISION

    PROMOTE_AGE_FIX_ONLY

CORRECTNESS_FIX_REQUIRED: **YES**
RELATIVE_TIME_CAPABILITY_GAIN_VALIDATED: **NO** (unchanged from screening -- not reopened this closure)

Selected representation for future M9 experiments: **AGE_FIX_ONLY**
- `unobserved_age_sentinel = "fixed"`
- relative-gap representation: NOT PROMOTED
- relative-time capability gain: NOT VALIDATED
- temporal limitation from M6/M6B: STILL A KNOWN LIMITATION
- passive calibration: B_DEPTH_AWARE, alpha=0.1
- active sampling: advisory (unchanged)
- topology-diverse M7 model: NOT PROMOTED (unchanged)
- cadence-diverse M6B model: NOT PROMOTED (unchanged)

M8_7_FORMALLY_COMPLETE: **YES**
M9_0_SCIENTIFICALLY_UNBLOCKED: **YES**

locked tests opened: before=False, after=False. No model promoted to production. No safety/authority semantics changed. M9.0 not started.
