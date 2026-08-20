# Milestone 10.1 summary: OOD / fusion validation

n_rows=1944 (seeds=[20260814, 31874, 20260815], families=['golden-reference', 'branched-loop', 'loop-grid', 'coastal-branch', 'tree-branch', 'dense-loop'], conditions=['IN_DISTRIBUTION', 'SENSOR_DROPOUT', 'SEVERITY_SHIFT'], depths reduced grid).

## Predictive (overall)
classical: {'n': 1944, 'top1': 0.4382716049382716, 'top3': 0.6867283950617284, 'mrr': 0.6066541740152861, 'nll': 1.6753162317386723, 'brier': 0.7306356966353226}
neural:    {'n': 1944, 'top1': 0.46399176954732513, 'top3': 0.7196502057613169, 'mrr': 0.6304581863609651, 'nll': 1.4563475891718605, 'brier': 0.6301046050781547}
fused:     {'n': 1944, 'top1': 0.4660493827160494, 'top3': 0.7103909465020576, 'mrr': 0.6296492259455233, 'nll': 1.7742641905990295, 'brier': 0.7097273824913922}

## Fusion effect (paired bootstrap, 2000 resamples, 90% CI)
fused vs neural-alone top1 delta: {'delta': 0.00205761316872428, 'ci90': [-0.00257201646090535, 0.006687242798353909], 'n': 1944}
fused vs classical-alone top1 delta: {'delta': 0.027777777777777776, 'ci90': [0.014917695473251029, 0.040123456790123455], 'n': 1944}

## OOD detection
classical (OODDetector.combined) AUROC: 0.5801866818523629
neural (ood_category head) AUROC: 0.5078465514597981
TPR/FPR at caution threshold (0.45): 0.0/0.0

## Calibration (frozen M9.6 fit reused, not refit)
{
  "classical": {
    "coverage": 0.7731481481481481,
    "mean_set_size": 4.054012345679013,
    "n": 1944,
    "alpha": 0.1,
    "coverage_floor": 0.85,
    "interpretable_against_frozen_calibrator": false
  },
  "neural": {
    "coverage": 0.9099794238683128,
    "mean_set_size": 3.8924897119341564,
    "n": 1944,
    "alpha": 0.1,
    "coverage_floor": 0.85,
    "interpretable_against_frozen_calibrator": true
  },
  "fused": {
    "coverage": 0.7438271604938271,
    "mean_set_size": 2.7391975308641974,
    "n": 1944,
    "alpha": 0.1,
    "coverage_floor": 0.85,
    "interpretable_against_frozen_calibrator": false
  },
  "disclosure": "The frozen M9.6 calibrator was fit exclusively on neural probabilities (run_m9_6_evaluate.py:328). Only calibration_report['neural'] is a valid coverage measurement against that fit; 'classical' and 'fused' entries apply the SAME neural-fit nonconformity thresholds to a different probability distribution and are not meaningful coverage claims -- reported only for descriptive transparency, never used in any guardrail.",
  "invalid_calibration_fail_closed_verified": true
}

## System behavior
{
  "abstention_rate": 0.012345679012345678,
  "n_rows": 1944,
  "learned_vs_classical_disagreement_mean_js": 0.12684168061037845,
  "fraction_fusion_changes_top1_vs_neural_alone": 0.015432098765432098
}

## Guardrails
{
  "no_in_distribution_regression": true,
  "measurable_ood_detection_improvement": false,
  "no_unsafe_confidence_increase": true,
  "no_invalid_calibration_acceptance_increase": true,
  "deterministic_fail_safe_available": true,
  "all_outputs_finite": true,
  "no_authority_boundary_regression": true
}

## M10_1_DECISION = LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED

locked_test_opened before/after: False/False.
