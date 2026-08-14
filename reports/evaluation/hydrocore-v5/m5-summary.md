# Milestone 5 summary: decision-aware active sampling

Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED
Calibration: B_DEPTH_AWARE (Milestone 3 frozen scheme, refit identically here).
K = 3, sample budget = 3, 108 paired incidents (25%=1, 50%=2, 75%=3 coverage x depth (2, 3, 6), 12/cell).

## Arm summaries

| arm | n | actionable<=1 | actionable<=2 | actionable<=3 | median samples | never actionable | final top-1 | final top-3 |
|---|---|---|---|---|---|---|---|---|
| RANDOM_VALID_UNSAMPLED | 108 | 0.917 | 0.963 | 0.963 | 0.0 | 0.037 | 0.880 | 0.972 |
| CURRENT_EIG | 108 | 0.963 | 0.972 | 0.972 | 0 | 0.028 | 0.852 | 0.954 |
| CANDIDATE_CONTRACTION | 108 | 0.954 | 0.972 | 0.972 | 0 | 0.028 | 0.852 | 0.954 |
| DECISION_GAIN | 108 | 0.963 | 0.972 | 0.972 | 0 | 0.028 | 0.852 | 0.954 |

## Paired bootstrap comparison vs RANDOM_VALID_UNSAMPLED (95% CI, 5000 resamples)

| arm | actionable<=3 delta (pp) | 95% CI | meets >=10pp+CI bar | samples-to-resolution delta | 95% CI | clearly lower |
|---|---|---|---|---|---|---|
| CURRENT_EIG | 0.9 | [0.0, 2.8] | False | 0.0 | [0.0, 0.0] | False |
| CANDIDATE_CONTRACTION | 0.9 | [0.0, 2.8] | False | 0.0 | [0.0, 0.0] | False |
| DECISION_GAIN | 0.9 | [0.0, 2.8] | False | 0.0 | [0.0, 0.0] | False |

## Limitations

Ceiling-effect caveat: 80.6% of incidents were ALREADY actionable (candidate-set size in [1, 3]) before any sample was taken, on the golden-reference network's 4-junction action space -- K=3 already covers most of that space, so there is limited room for any sampling policy (including CURRENT_EIG, the production baseline) to demonstrate a large effect on this specific network. The comparison above is real and not invalidated by this, but a null/near-null result here should not be read as ruling out a larger effect on a bigger network with more source candidates; it was not tested (out of scope for this milestone -- topology diversity is Milestone 7).

**Exit decision: ACTIVE_SAMPLING_REMAINS_ADVISORY**

Per the predeclared promotion rule (experiments.txt 5.5 / this script's module docstring): a policy is promoted only if it clears >=10pp actionable<=3 improvement with a 95% CI excluding zero, or a clearly (CI-excluding-zero) lower paired samples-to-resolution distribution -- never a bare point-estimate difference. If no arm clears this bar, active sampling remains advisory, matching the M0/CAP-REM-02 baseline finding this milestone was designed to test.
