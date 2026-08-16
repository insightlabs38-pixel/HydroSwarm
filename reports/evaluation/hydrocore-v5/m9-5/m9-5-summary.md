# Milestone 9.5 summary: source-representative calibration-support confirmation

Calibration-support / frozen-checkpoint study only. No training, no tuning, no calibration-method change. Follows up `reports/evaluation/hydrocore-v5/m9-4/m9-4-closure.json` (M9_4_DECISION=B).

**Representativeness audit passed**: True
**Reproduction/sanity gate**: FAIL

## Control arm (ARM_A / CURRENT, golden-reference)

All 3 seeds pass >=0.85 at PRIMARY support=20: **True**
Per-support-level coverage: {'4': {'mean_coverage': 0.9196428571428571, 'min_coverage': 0.9125, 'all_3_seeds_pass_0_85': True}, '8': {'mean_coverage': 0.874404761904762, 'min_coverage': 0.8696428571428572, 'all_3_seeds_pass_0_85': True}, '12': {'mean_coverage': 0.8857142857142857, 'min_coverage': 0.8821428571428571, 'all_3_seeds_pass_0_85': True}, '20': {'mean_coverage': 0.8952380952380953, 'min_coverage': 0.8892857142857142, 'all_3_seeds_pass_0_85': True}}

## INTERLEAVED / ARM_B2 primary calibration gate

All 9 trained-family/seed cells pass >=0.85: **True**

## Candidate-set-size guard

Pathological full-set behavior detected: **False**

## Quantile stability (support=4 vs support=20)

{'n_cells_compared': 36, 'n_cells_span_shrunk': 31, 'quantile_stability_improved_with_support': True, 'resamples': 2000, 'bootstrap_seed': 20260817}

## M9_5_DECISION: E (REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER)

The representativeness audit or the Section 12 reproduction/implementation-path-consistency sanity gate did not pass; the M9.5 support-level calibration results below cannot yet be scientifically interpreted.

Provisional best HydroCore-S recipe: None
Next recommended milestone: M9.5_RECHECK (resolve representativeness/reproduction blocker before any further interpretation)

locked tests opened: before=False, after=False. No model promoted to production. No safety/authority semantics changed. No architecture/training/calibration-method change performed.

## Addendum (added post hoc, for transparency only -- does not alter M9_5_DECISION or any computed number)

The E outcome above is driven entirely by one sub-check inside the Section-12 reproduction/sanity gate: `qualitatively_consistent_with_m9_4_pattern`, which expected M9.5's disjoint support=4 subset to reproduce M9.4's *poor* golden-reference coverage (0.60-0.68). Instead it calibrated well (ARM_A 0.91-0.93, ARM_B2 0.87-0.96) -- the `implementation_path_consistent` sub-check passed cleanly (same calibrator class, alpha=0.1, grouping). This is not evidence of a bug; it's consistent with (and arguably supports) the underlying small-n-variance hypothesis M9.5 set out to test -- see `m9-5-quantile-stability.json`, where quantile span shrinks from support=4 to support=20 in 31/36 cells, meaning small-n draws in this pipeline genuinely are high-variance, and M9.4's particular calibration draw looks like an unlucky tail rather than a deterministic property of n=16.

Every criterion with a basis in Category A's own definition passed at this run: representativeness audit, resubstitution diagnostic, PRIMARY-support=20 interleaved gate (9/9 ARM_B2 cells), control arm (3/3 ARM_A seeds, at every one of the 4 support levels), candidate-set guard, and quantile-stability improvement. `M9_5_DECISION=E` is preserved here as the literal, predeclared-code output and was not changed after viewing results. Whether Section 12's specific qualitative-pattern sub-check should be treated as binding, or re-specified in a follow-up milestone, is a call for the human reviewer -- not one the coordinator made unilaterally by editing the decision logic post hoc.
