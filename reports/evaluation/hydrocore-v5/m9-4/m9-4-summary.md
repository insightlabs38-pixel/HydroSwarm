# Milestone 9.4 summary: source-representative, exchangeability-corrected re-evaluation

Frozen-checkpoint re-evaluation only. No training, no tuning. Follows up `reports/evaluation/hydrocore-v5/m9-3/m9-3-closure.json` (EVAL_MAX_SOURCES=4 truncation root cause).

**M9_4_LEGACY_REPRODUCTION**: PASS
**Representativeness audit passed**: True

## Macro-family (equal-weight) MATURE neural Top-1, unseen development families, full source population

ARM_A: 0.6745370370370369
ARM_B2: 0.737962962962963
Macro delta (bootstrap point estimate): 0.06342592592592593
90% paired-bootstrap CI: [0.028240740740740743, 0.10002314814814814]
Legacy (M9.0a, EVAL_MAX_SOURCES=4) pooled gain was +6.6pp -- VALID FOR THE LEGACY EVALUATED SOURCE SUBSET, NOW REASSESSED ON FULL SOURCE SUPPORT above.

## Gates

Predictive-generalization gate passed: **True**
Known-family guardrails passed: **True**
Calibration gate passed: **False**

## M9_4_DECISION: B (INTERLEAVED_PREDICTIVE_GAIN_CONFIRMED_CALIBRATION_FAILS)

Predictive generalization gate and known-family guardrails passed, but the B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH calibration gate (marginal coverage >= 0.85 for every required trained-family/seed) still fails on the source-representative calibration/development population.

Loop-grid J1 diagonal / row total (ARM_B2, all seeds/depths): 15/84
Hard source-pair counts (aggregate): {'J1->J7': 18, 'J1->J8': 39, 'J7->J1': 0, 'J8->J1': 0}

Provisional best HydroCore-S recipe: None
Optimizer-step-parity confirmation still required: False

locked tests opened: before=False, after=False. No model promoted to production. No safety/authority semantics changed.
