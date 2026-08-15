# Milestone 9.0b summary: multi-topology calibration grouping study

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_0B_PROTOCOL.md`. Tests whether a different Mondrian grouping/fallback construction over the unmodified `SplitConformalCalibrator`, at fixed alpha=0.1 and the frozen (unretrained) M9.0a `ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY` predictor checkpoints, can restore safety-valid known-family coverage.

## Scheme results (safety + actionability, per seed)

### CURRENT_FAMILY_DEPTH

| seed | marginal | EARLY | MID | MATURE | mean norm. set size | safety valid |
|---|---|---|---|---|---|---|
| 20260814 | 0.8482 | 0.8125 | 0.8542 | 0.8958 | 0.4270 | False |
| 31874 | 0.8452 | 0.8056 | 0.8542 | 0.8958 | 0.4461 | False |
| 20260815 | 0.8214 | 0.7847 | 0.8229 | 0.8750 | 0.4213 | False |

All 3 seeds safety valid: **False** | Candidate-set guardrail pass: **True** | Operationally valid: **False** | mean normalized set size (across seeds): 0.4315

### POOLED_DEPTH_AWARE

| seed | marginal | EARLY | MID | MATURE | mean norm. set size | safety valid |
|---|---|---|---|---|---|---|
| 20260814 | 0.8631 | 0.8611 | 0.8542 | 0.8750 | 0.4432 | False |
| 31874 | 0.8482 | 0.8333 | 0.8438 | 0.8750 | 0.4501 | False |
| 20260815 | 0.8125 | 0.7847 | 0.8125 | 0.8542 | 0.4223 | False |

All 3 seeds safety valid: **False** | Candidate-set guardrail pass: **True** | Operationally valid: **False** | mean normalized set size (across seeds): 0.4385

### BROAD_FALLBACK_CONTROL

| seed | marginal | EARLY | MID | MATURE | mean norm. set size | safety valid |
|---|---|---|---|---|---|---|
| 20260814 | 0.8452 | 0.8264 | 0.8438 | 0.8750 | 0.4296 | False |
| 31874 | 0.8065 | 0.7500 | 0.8229 | 0.8750 | 0.4309 | False |
| 20260815 | 0.7708 | 0.6736 | 0.8125 | 0.8750 | 0.3965 | False |

All 3 seeds safety valid: **False** | Candidate-set guardrail pass: **True** | Operationally valid: **False** | mean normalized set size (across seeds): 0.4190

### HIERARCHICAL_CONSERVATIVE

| seed | marginal | EARLY | MID | MATURE | mean norm. set size | safety valid |
|---|---|---|---|---|---|---|
| 20260814 | 0.8750 | 0.8750 | 0.8542 | 0.8958 | 0.4583 | False |
| 31874 | 0.8929 | 0.9167 | 0.8542 | 0.8958 | 0.4883 | False |
| 20260815 | 0.8631 | 0.8819 | 0.8229 | 0.8750 | 0.4671 | False |

All 3 seeds safety valid: **False** | Candidate-set guardrail pass: **True** | Operationally valid: **False** | mean normalized set size (across seeds): 0.4712

## Trained-family check (per scheme, min marginal coverage across seeds)

| scheme | golden-reference | branched-loop | loop-grid |
|---|---|---|---|
| CURRENT_FAMILY_DEPTH | 0.9464 | 0.7946 | 0.7143 |
| POOLED_DEPTH_AWARE | 0.8214 | 0.8929 | 0.7232 |
| BROAD_FALLBACK_CONTROL | 0.7768 | 0.8125 | 0.7232 |
| HIERARCHICAL_CONSERVATIVE | 0.9464 | 0.9107 | 0.7232 |

## Diagnosis of the M9.0a anomaly (Section 13)

Mean (family:depth quantile - pooled-depth quantile), across all cells/seeds: -0.0462
family:depth quantiles systematically SMALLER than pooled-depth (Hypothesis A support): **True**
Mean BROAD_FALLBACK_CONTROL quantile: 0.9293
Mean POOLED_DEPTH_AWARE quantile: 0.9280
Broad-fallback quantiles more conservative than pooled-depth (Hypothesis C support): **True**

## Unseen-topology calibration transfer (diagnostic only, not used to select a scheme)

| scheme | family | mean marginal coverage (3 seeds) |
|---|---|---|
| CURRENT_FAMILY_DEPTH | coastal-branch | 0.8006 |
| CURRENT_FAMILY_DEPTH | tree-branch | 0.9464 |
| CURRENT_FAMILY_DEPTH | dense-loop | 0.8571 |
| POOLED_DEPTH_AWARE | coastal-branch | 0.7887 |
| POOLED_DEPTH_AWARE | tree-branch | 0.9673 |
| POOLED_DEPTH_AWARE | dense-loop | 0.8899 |
| BROAD_FALLBACK_CONTROL | coastal-branch | 0.8006 |
| BROAD_FALLBACK_CONTROL | tree-branch | 0.9464 |
| BROAD_FALLBACK_CONTROL | dense-loop | 0.8571 |
| HIERARCHICAL_CONSERVATIVE | coastal-branch | 0.7887 |
| HIERARCHICAL_CONSERVATIVE | tree-branch | 0.9673 |
| HIERARCHICAL_CONSERVATIVE | dense-loop | 0.8899 |

## FINAL M9.0b DECISION

    INTERLEAVED_PREDICTOR_CALIBRATION_NOT_RESOLVED

Selected scheme: **NONE**

APS_RAPS_CALIBRATION_FOLLOWUP_WARRANTED: **False**

## M9.1 RECIPE

- representation: AGE_FIX_ONLY
- topology training: SINGLE_FAMILY_CURRENT_TRAINING
- calibration: B_DEPTH_AWARE (existing)
- alpha: 0.1
- interleaved predictor operationally promoted: NO
- M9_1_SCIENTIFICALLY_UNBLOCKED: YES

locked tests opened: before=False, after=False. No predictor retrained. No production calibration code changed. No M9.1/M9 capacity work begun.
