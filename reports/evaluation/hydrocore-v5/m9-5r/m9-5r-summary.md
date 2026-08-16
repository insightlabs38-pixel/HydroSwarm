# Milestone 9.5R summary: independent, one-shot calibration confirmation

Independent, one-shot confirmation of HydroCore-S calibration at the already-predeclared adequate support level (20 independent calibration incidents/source), using fresh, disjoint calibration and development populations. Does NOT reinterpret or overwrite M9.5, which remains formally closed as `M9_5_DECISION=E` (`REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER`).

**Representativeness audit passed**: True
**Corrected sanity/implementation gate**: PASS

## CURRENT control (ARM_A, golden-reference)

All 3 seeds pass >=0.85: **True**
Per-seed coverage: {'20260814': {'marginal_coverage': 0.9, 'passes_operational_floor_0_85': True}, '31874': {'marginal_coverage': 0.9178571428571428, 'passes_operational_floor_0_85': True}, '20260815': {'marginal_coverage': 0.9089285714285714, 'passes_operational_floor_0_85': True}}

## INTERLEAVED confirmation (ARM_B2, 3 trained families x 3 seeds)

All 9 cells pass >=0.85: **True**

## Candidate-set-size guard

Pathological full-set behavior detected: **False**

## M9_5R_DECISION: A (INDEPENDENT_CALIBRATION_CONFIRMATION_PASS)

Representativeness audit passed, corrected sanity/implementation gate passed, CURRENT control passes all 3 seeds, INTERLEAVED passes all 9 trained-family/seed cells, and the candidate-set guard passes (no pathological full-set behavior) -- M9.5's favorable calibration evidence is independently confirmed under a fresh, disjoint population.

Provisional best HydroCore-S recipe: CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + STEP_MATCHED_INTERLEAVED_MULTI_FAMILY
Next recommended milestone: M9.6_EXACT_COMPUTE_PARITY_CONFIRMATION

locked tests opened: before=False, after=False. No model promoted to production. No safety/authority semantics changed. No architecture/training/calibration-method change performed. No field-performance claim made.
