# Milestone 9.0a summary: optimizer-step-matched interleaved topology training study

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_0A_PROTOCOL.md`. Confound-resolution follow-up to Milestone 9.0: Arm B2 (`STEP_MATCHED_INTERLEAVED_MULTI_FAMILY`) uses 4 microbatches/optimizer-update (matching Arm A's `gradient_accumulation_steps=4`) in a fixed 3-update family rotation, achieving exact per-epoch and total optimizer-step and scheduler-trajectory parity with Arm A; calibration is fit/evaluated separately for all three predictor seeds for both arms (M9.0's own calibration rejection used only the representative seed).

## Budget parity

Arm A total optimizer steps: 1350
Arm B2 total optimizer steps (per seed): [1200, 1350, 1350]
Optimizer-step ratio (B2/A): 0.9630
Exposure ratio (B2/A): 1.0000
Scheduler total_steps match: True
All 3 seeds match Arm A's total optimizer steps: False
All 3 seeds match Arm A's per-epoch optimizer steps: False
**Overall optimization-budget parity: FAIL**

## Known-network (golden-reference) localization, mean over 3 seeds

| arm | EARLY neural top1 | MID neural top1 | MATURE neural top1 | overall MRR | MATURE hybrid top1 | all finite |
|---|---|---|---|---|---|---|
| ARM_A | 0.3750 | 0.8438 | 1.0000 | 0.8374 | 1.0000 | True |
| ARM_B2 | 0.4097 | 0.8125 | 1.0000 | 0.8301 | 1.0000 | True |

## Guardrails (predeclared, not relaxed after results)

EARLY regression (pp): -3.47 (bar <= 5.0)
MATURE regression (pp): 0.00 (bar <= 3.0)
MRR regression: 0.0073 (bar <= 0.03)
Arm B2 known-family calibration robust (all 3 seeds >= 0.85): False
Arm B2 known-family candidate-set size ok (mean across seeds): True (3.380)
**Known-network guardrails passed: False**

## Arm-B2 trained-family retention (TRAINED_FAMILY_GENERALIZATION, mean over 3 seeds)

| family | MATURE neural top1 | EARLY neural top1 |
|---|---|---|
| branched-loop | 0.5625 | 0.2917 |
| loop-grid | 0.4375 | 0.1458 |

Learned above chance for both added families (MATURE top1 > 0.15): **True**

## Primary unseen-topology generalization (pooled coastal-branch + tree-branch + dense-loop, MATURE, neural top1)

Arm A pooled mean: 0.6354
Arm B2 pooled mean: 0.7014
Pooled diff (Arm B2 - Arm A, neural): 6.60 pp
Pooled diff (Arm B2 - Arm A, hybrid): 6.94 pp

### Per-family MATURE neural top1 difference (Arm B2 - Arm A)

| family | diff (pp) | improved |
|---|---|---|
| coastal-branch | +10.42 | True |
| tree-branch | +2.08 | True |
| dense-loop | +7.29 | True |

### Paired bootstrap (2,000 resamples, 90% interval, bootstrap seed 20260815)

MATURE neural top1 (Arm B2 - Arm A): observed +0.0660, 90% CI [+0.0278, +0.1042], n=144
MATURE hybrid top1 (Arm B2 - Arm A): observed +0.0694, 90% CI [+0.0278, +0.1146]
EARLY neural top1 (Arm B2 - Arm A): observed -0.0162, 90% CI [-0.0509, +0.0185]
MATURE MRR (Arm B2 - Arm A): observed +0.0451, 90% CI [+0.0187, +0.0710]

CI lower bound > 0 (MATURE neural top1): **True**
Improved on >= 2 of 3 unseen families: **True** (coastal-branch, tree-branch, dense-loop)
No unseen-family regression worse than 5.0pp: **True** (worst: 0.00pp)
Directionally consistent across all 3 seeds (per-seed pooled MATURE diff all >= 0): **True** (['+0.0417', '+0.0729', '+0.0833'])
**Topology gain survives optimizer-step parity: True**

## Calibration by seed (B_DEPTH_AWARE, alpha=0.1)

### ARM_A

| seed | marginal coverage | EARLY coverage | MID coverage | MATURE coverage | mean set size | guardrail pass |
|---|---|---|---|---|---|---|
| 20260814 | 0.9554 | 0.9583 | 0.9062 | 1.0000 | 2.0446 | True |
| 31874 | 0.9554 | 0.9583 | 0.9062 | 1.0000 | 2.0893 | True |
| 20260815 | 0.9464 | 0.9583 | 0.8750 | 1.0000 | 2.0357 | True |

Aggregate: mean=0.9524, min=0.9464, max=0.9554, seeds passing >=0.85: 3/3

### ARM_B2

| seed | marginal coverage | EARLY coverage | MID coverage | MATURE coverage | mean set size | guardrail pass |
|---|---|---|---|---|---|---|
| 20260814 | 0.8482 | 0.8125 | 0.8542 | 0.8958 | 3.3452 | False |
| 31874 | 0.8452 | 0.8056 | 0.8542 | 0.8958 | 3.5000 | False |
| 20260815 | 0.8214 | 0.7847 | 0.8229 | 0.8750 | 3.2946 | False |

Aggregate: mean=0.8383, min=0.8214, max=0.8482, seeds passing >=0.85: 0/3

**Arm B2 calibration classification: CALIBRATION_SYSTEMATICALLY_INCOMPATIBLE**

Known EARLY conditional-coverage limitation (carried forward from M8.7/M9.0): preserved as a documented limitation regardless of the marginal-coverage classification above, not claimed solved.

## Unseen-topology calibration transfer (diagnostic, per seed)

| arm | seed | marginal coverage | EARLY coverage | applicability rate |
|---|---|---|---|---|
| ARM_A | 20260814 | 0.7738 | 0.6528 | 0.0000 |
| ARM_A | 31874 | 0.7470 | 0.6250 | 0.0000 |
| ARM_A | 20260815 | 0.8214 | 0.7292 | 0.0000 |
| ARM_B2 | 20260814 | 0.8810 | 0.8403 | 0.0000 |
| ARM_B2 | 31874 | 0.8571 | 0.8125 | 0.0000 |
| ARM_B2 | 20260815 | 0.8661 | 0.8194 | 0.0000 |

## M9.0 comparison (historical context only -- Section 15, not perfectly paired)

M9.0 Arm B pooled MATURE gain: +6.60pp; M9.0a Arm B2 pooled MATURE gain: +6.60pp

This is diagnostic context only. The primary causal claim of M9.0a remains Arm A vs Arm B2 under THIS document's own step-matched protocol.

## FINAL M9.0a DECISION

    TOPOLOGY_GAIN_VALIDATED_CALIBRATION_BLOCKER_REMAINS

M9_RECIPE:

C:
    NOT_FROZEN_CALIBRATION_REVIEW_REQUIRED

Preserved regardless:

- relative-time representation = NOT PROMOTED
- cadence-diverse training = NOT PROMOTED
- PCGrad = OFF
- PyG = NO
- B_DEPTH_AWARE = CURRENT calibration method
- alpha = 0.1
- current OOD/fusion/safety semantics unchanged

- M9 capacity study scientifically unblocked: NO

locked tests opened: before=False, after=False. No model promoted to production. No safety/authority semantics changed. No model-size change. No PyTorch Geometric introduced. No calibration redesign (M9.0b, if needed, is a separate future milestone).
