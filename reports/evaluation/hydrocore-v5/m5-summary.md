# Milestone 5 summary: decision-aware active sampling

Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED
Calibration: B_DEPTH_AWARE (Milestone 3 frozen scheme, refit identically here).
K = 3, sample budget = 3, 108 paired incidents (25%=1, 50%=2, 75%=3 coverage x depth (2, 3, 6), 12/cell).

## Arm summaries (candidate_gate_pass = candidate-count planning gate ONLY, NOT full product actionability -- see Limitations)

| arm | n | gate<=1 | gate<=2 | gate<=3 | median samples | never gate-pass | final top-1 | final top-3 |
|---|---|---|---|---|---|---|---|---|
| RANDOM_VALID_UNSAMPLED | 108 | 0.917 | 0.963 | 0.963 | 0.0 | 0.037 | 0.880 | 0.972 |
| CURRENT_EIG | 108 | 0.963 | 0.972 | 0.972 | 0 | 0.028 | 0.852 | 0.954 |
| CANDIDATE_CONTRACTION | 108 | 0.954 | 0.972 | 0.972 | 0 | 0.028 | 0.852 | 0.954 |
| DECISION_GAIN | 108 | 0.963 | 0.972 | 0.972 | 0 | 0.028 | 0.852 | 0.954 |

## Conformal coverage after adaptive sampling (target ~90%, alpha=0.1; material-undercoverage bar = 5.0pp, matching Milestone 3)

Overall: n=535, empirical coverage=0.955, 95% CI=[0.9375997070989007, 0.972680666732875], materially below target=False

### By arm

| arm | n | empirical coverage | 95% CI | materially below target |
|---|---|---|---|---|
| RANDOM_VALID_UNSAMPLED | 138 | 0.978 | [0.953929589743228, 1.0] | False |
| CURRENT_EIG | 132 | 0.947 | [0.9087401456413979, 0.9851992482979961] | False |
| CANDIDATE_CONTRACTION | 133 | 0.947 | [0.9094183229876673, 0.9853185191175957] | False |
| DECISION_GAIN | 132 | 0.947 | [0.9087401456413979, 0.9851992482979961] | False |

### By round (0 = initial, before any sample)

| round | n | empirical coverage | 95% CI | materially below target |
|---|---|---|---|---|
| 0 | 432 | 0.981 | [0.9687681792091974, 0.9941947837537656] | False |
| 1 | 84 | 0.845 | [0.7678920713751493, 0.9225841191010412] | True |
| 2 | 18 | 0.833 | [0.6611648829463883, 1.0] | True |
| 3 | 1 | 1.000 | [1.0, 1.0] | False |

### By initial evidence depth

| depth | n | empirical coverage | 95% CI | materially below target |
|---|---|---|---|---|
| 2 | 188 | 1.000 | [1.0, 1.0] | False |
| 3 | 203 | 0.901 | [0.8604807872694731, 0.9424748777551575] | False |
| 6 | 144 | 0.972 | [0.9453807491322332, 0.9990636953122112] | False |

### By initial sensor coverage

| coverage | n | empirical coverage | 95% CI | materially below target |
|---|---|---|---|---|
| 25% | 218 | 0.890 | [0.8483576073891135, 0.9314589063723544] | False |
| 50% | 165 | 1.000 | [1.0, 1.0] | False |
| 75% | 152 | 1.000 | [1.0, 1.0] | False |

**COVERAGE DEGRADATION DETECTED after active sampling: round(s) [1, 2] have empirical coverage materially below the 90% target (by >5.0pp) even though aggregate by-arm coverage looks close to target (the aggregate number masks this, exactly the subgroup-failure pattern Milestone 3 found and corrected for). This suggests the frozen B_DEPTH_AWARE calibration -- fit on fixed evidence depths from the original PASSIVE corpus -- may not fully transfer to the different evidence distribution created by actively adding a new sensor node mid-incident. Per instruction, calibration fitting is NOT re-tuned in this correction; this is reported as a genuine, unresolved limitation of applying Milestone 3's calibration to Milestone 5's active-sampling setting, and any apparent per-arm improvement in candidate_gate_pass during these rounds should be read with this degradation in mind.**

## Planning-compatible actionability (experimental)

planning_compatible (experiments.txt Milestone 5 correction requirement 3) is NOT_EVALUATED for every state in this experiment: 2 of its 6 required components (disagreement_below_threshold, ood_normal) require production machinery (full classical/neural fusion diagnostics; the real deterministic OODDetector) this experiment does not build, and are explicitly marked NOT_EVALUATED rather than approximated from the model's own advisory-only OOD heads or omitted from the composite. calibration_applicable, candidate_count_in_k (== candidate_gate_pass), model_evidence_sufficient (raw evidence_sufficiency head >= 0.55, production's own threshold/convention), and sensor_guards_pass (ALL_SENSORS_FROZEN) ARE faithfully computed per state -- see per-state fields in `rows[*].states[*]`.

## Initially non-gate-pass subset (the population that actually exercises the sampler; planning_compatible is NOT_EVALUATED end-to-end, so this uses candidate_gate_pass -- see above)

| arm | N requiring sampling | resolved<=1 | resolved<=2 | resolved<=3 | never resolved | median samples | mean samples |
|---|---|---|---|---|---|---|---|
| RANDOM_VALID_UNSAMPLED | 21 | 0.571 | 0.810 | 0.810 | 0.190 | 1 | 1.2941176470588236 |
| CURRENT_EIG | 21 | 0.810 | 0.857 | 0.857 | 0.143 | 1.0 | 1.0555555555555556 |
| CANDIDATE_CONTRACTION | 21 | 0.762 | 0.857 | 0.857 | 0.143 | 1.0 | 1.1111111111111112 |
| DECISION_GAIN | 21 | 0.810 | 0.857 | 0.857 | 0.143 | 1.0 | 1.0555555555555556 |

## Paired bootstrap comparison vs RANDOM_VALID_UNSAMPLED (95% CI, 5000 resamples)

| arm | gate-pass<=3 delta (pp) | 95% CI | meets >=10pp+CI bar | coverage materially below target | samples-to-resolution delta | 95% CI | clearly lower |
|---|---|---|---|---|---|---|---|
| CURRENT_EIG | 0.9 | [0.0, 2.8] | False | False | 0.0 | [0.0, 0.0] | False |
| CANDIDATE_CONTRACTION | 0.9 | [0.0, 2.8] | False | False | 0.0 | [0.0, 0.0] | False |
| DECISION_GAIN | 0.9 | [0.0, 2.8] | False | False | 0.0 | [0.0, 0.0] | False |

## Limitations

Ceiling-effect caveat: 80.6% of incidents ALREADY passed the candidate gate (candidate-set size in [1, 3]) before any sample was taken, on the golden-reference network's 4-junction action space -- K=3 already covers most of that space, so there is limited room for any sampling policy (including CURRENT_EIG, the production baseline) to demonstrate a large effect on this specific network. The comparison above is real and not invalidated by this, but a null/near-null result here should not be read as ruling out a larger effect on a bigger network with more source candidates; it was not tested (out of scope for this milestone -- topology diversity is Milestone 7).

**Exit decision: ACTIVE_SAMPLING_REMAINS_ADVISORY**

Per the predeclared promotion rule (experiments.txt 5.5 / this script's module docstring, Milestone-5 correction): a policy is promoted only if it clears >=10pp candidate_gate_pass<=3 improvement with a 95% CI excluding zero, or a clearly (CI-excluding-zero) lower paired samples-to-resolution distribution -- never a bare point-estimate difference -- AND its own empirical candidate-set coverage is not materially below the alpha=0.1 target; an arm that only looks better because it under-covers the true source is never promoted. If no arm clears this bar, active sampling remains advisory, matching the M0/CAP-REM-02 baseline finding this milestone was designed to test.
