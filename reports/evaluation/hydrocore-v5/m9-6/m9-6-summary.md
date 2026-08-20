# Milestone 9.6 summary: exact-compute-parity final HydroCore-S confirmation

Freshly trained, exactly-matched 1350-optimizer-step CURRENT and INTERLEAVED HydroCore-S arms, evaluated on a fresh source-representative development population and calibrated using the independently confirmed M9.5R calibration policy.

**Training parity passed**: True
**Development representativeness passed**: True
**Calibration representativeness passed**: True

## Predictive generalization (unseen macro-family MATURE Top-1, ARM_B - ARM_A)

Delta: **0.03287037037037037**, 90% CI: [0.008787037037037029, 0.056296296296296296]
Gate passed: **True**

## Known-family guardrails (golden-reference)

Passed: **True** (EARLY regression 4.1666666666666625pp, MATURE regression 0.2083333333333437pp, MRR regression 0.01649305555555547)

## Calibration gate

CURRENT control 3/3 pass: **True**
INTERLEAVED 9/9 pass: **True**
Candidate-set guard pass: **True**

## M9_6_DECISION: A (HYDROCORE_S_STATUS=FROZEN)

Exact compute parity, representativeness, predictive generalization gate, known-family guardrails, and the full calibration gate (CURRENT control, INTERLEAVED 9/9, candidate-set guard) all passed. HydroCore-S architecture and interleaved training recipe are confirmed.

Selected recipe: CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING + B_DEPTH_AWARE_CALIBRATION + ALPHA_0_1 + SOURCE_REPRESENTATIVE_CALIBRATION_SUPPORT_20_PER_SOURCE
Next recommended milestone: Not started here -- recommended: remaining system-level validation (OOD/fusion, Scout, Strategist, trajectory/planning, safety/authority) OR a separately governed HydroCore-M capacity experiment IF all prerequisite gates are complete AND meaningful headroom remains (not assumed automatically).

locked tests opened: before=False, after=False. No model promoted to production. No field-performance claim made.
