# Milestone 9.0 summary: interleaved topology-diversity training study

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_0_PROTOCOL.md`. Retests Milestone 7's topology-diversity question with true cross-family gradient interleaving instead of M7's sequential 3-phase curriculum. Arm A (`SINGLE_FAMILY_CONTROL`) reuses the existing Milestone-8.7 `AGE_FIX_ONLY` checkpoints verbatim (verified comparable -- protocol Section 2). Arm B (`INTERLEAVED_MULTI_FAMILY`) trained fresh, 3 seeds, on the SAME golden-reference/branched-loop/loop-grid corpus Milestone 7's EXPANDED arm used, interleaved one family-pure microbatch per family per optimizer step.

## Known-network (golden-reference) localization, mean over 3 seeds

| arm | EARLY neural top1 | MID neural top1 | MATURE neural top1 | overall MRR | MATURE hybrid top1 | all finite |
|---|---|---|---|---|---|---|
| ARM_A | 0.3750 | 0.8438 | 1.0000 | 0.8374 | 1.0000 | True |
| ARM_B | 0.4097 | 0.8125 | 1.0000 | 0.8264 | 1.0000 | True |

## Guardrails (Section 18, predeclared, not relaxed after results)

EARLY regression (pp): -3.47 (bar <= 5.0)
MATURE regression (pp): 0.00 (bar <= 3.0)
MRR regression: 0.0110 (bar <= 0.03)
Arm B known-family marginal coverage ok: False
Arm B known-family candidate-set size ok: True
**Known-network guardrails passed: False**

## Arm-B trained-family retention (TRAINED_FAMILY_GENERALIZATION, mean over 3 seeds)

| family | MATURE neural top1 | EARLY neural top1 |
|---|---|---|
| branched-loop | 0.5625 | 0.3125 |
| loop-grid | 0.4375 | 0.1458 |

Learned above chance for both added families (MATURE top1 > 0.15): **True**

## Primary unseen-topology generalization (pooled coastal-branch + tree-branch + dense-loop, MATURE, neural top1)

Arm A pooled mean: 0.6354
Arm B pooled mean: 0.7014
Pooled diff (Arm B - Arm A, neural): 6.60 pp
Pooled diff (Arm B - Arm A, hybrid): 7.29 pp

### Per-family MATURE neural top1 difference (Arm B - Arm A)

| family | diff (pp) | improved |
|---|---|---|
| coastal-branch | +10.42 | True |
| tree-branch | +4.17 | True |
| dense-loop | +5.21 | True |

### Paired bootstrap (2,000 resamples, 90% interval, bootstrap seed 20260815)

MATURE neural top1 (Arm B - Arm A): observed +0.0660, 90% CI [+0.0278, +0.1076], n=144
MATURE hybrid top1 (Arm B - Arm A): observed +0.0729, 90% CI [+0.0312, +0.1181]
EARLY neural top1 (Arm B - Arm A): observed -0.0139, 90% CI [-0.0486, +0.0208]
MATURE MRR (Arm B - Arm A): observed +0.0441, 90% CI [+0.0179, +0.0703]

CI lower bound > 0 (MATURE neural top1): **True**
Improved on >= 2 of 3 unseen families: **True** (coastal-branch, tree-branch, dense-loop)
No unseen-family regression worse than 5.0pp: **True** (worst: 0.00pp)
Directionally consistent across all 3 seeds (per-seed pooled MATURE diff all >= 0): **True** (['+0.0729', '+0.0729', '+0.0521'])

## Calibration (B_DEPTH_AWARE, alpha=0.1, representative seed 31874)

| arm | known-family marginal coverage | known-family mean set size | unseen-transfer marginal coverage | unseen-transfer EARLY coverage |
|---|---|---|---|---|
| ARM_A | 0.955 | 2.089 | 0.747 | 0.625 |
| ARM_B | 0.833 | 3.491 | 0.827 | 0.771 |

Known EARLY conditional-coverage limitation (carried forward from M8.7): preserved as a documented limitation, not claimed solved by either arm.

## Comparison with M7 (historical context only -- Section 14, not perfectly paired)

M7's SEQUENTIAL EXPANDED arm (`reports/evaluation/hydrocore-v5/m7-summary.md`) found tree-branch regression, weak dense-loop gain, and no robust unseen-topology generalization, using a different corpus vintage/depth grid ((3, 25) only) and the pre-AGE_FIX_ONLY representation. M9.0's INTERLEAVED_MULTI_FAMILY (Arm B, this document) shows tree-branch improving and improvement/no-regression on all three unseen families under the matched M9.0 protocol -- qualitatively different from M7's own sequential-curriculum pattern. This is reported as context; the primary causal claim remains Arm A vs Arm B under THIS document's own matched protocol, not a claim that M9.0 and M7 are a paired experiment.

## FINAL M9.0 DECISION

    INTERLEAVED_TOPOLOGY_TRAINING_REJECTED

M9 RECIPE: **AGE_FIX_ONLY + SINGLE_FAMILY_CURRENT_TRAINING**

- relative-time representation promoted: NO
- cadence-diverse training promoted: NO
- PCGrad enabled: NO
- B_DEPTH_AWARE retained: YES
- alpha: 0.1
- M9 capacity study scientifically unblocked: YES

locked tests opened: before=False, after=False. No model promoted to production. No safety/authority semantics changed. No model-size change. No PyTorch Geometric introduced.
