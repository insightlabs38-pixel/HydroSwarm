# Milestone 9.8 summary: HydroCore-S vs HydroCore-M capacity comparison

Preregistered in M9.7, checkpoint policy corrected in M9.7A. Executed exactly as frozen -- no architecture, seed, threshold, or statistical-procedure change.

**Training parity passed**: True
**Development representativeness passed**: True
**Calibration representativeness passed**: True

## Primary endpoint (unseen-topology MATURE neural Top-1, M - S)

Delta: **-0.0034259259259259277**, 90% CI: [-0.012777777777777779, 0.0062962962962962955], threshold: +0.02
Guardrail A (primary effect) passed: **False** (clean_fail=True, borderline=False)

## Family / seed consistency

Guardrail B (family consistency) passed: **True** -- improved ['coastal-branch', 'tree-branch'], worst regression -1.8055555555555558pp
Guardrail C (seed consistency) passed: **True** -- per-seed deltas {'20260814': -0.0033333333333333335, '31874': -0.016944444444444446, '20260815': 0.01}

## Known-family retention (golden-reference)

Guardrail D passed: **True** (EARLY regression 1.388888888888895pp, MATURE regression 0.6249999999999978pp, MRR regression 0.0019290123456788821)

## Calibration

Guardrail E passed: **True** (S control 3/3: True, M 9/9: True, candidate-set guard: True)

## Engineering cost (descriptive only)

Parameter ratio M/S: 3.328. Checkpoint size ratio: 3.324. Median inference latency ratio: 1.590.

## M9_8_DECISION: B (HYDROCORE_M_NO_MEANINGFUL_CAPACITY_GAIN)

Selected predictor after M9.8: **S**. HydroCore-L authorized: **False**.

This milestone reports a preregistered capacity-only comparison. It does not make field-performance, production-readiness, or locked-test claims, and does not itself authorize any further capacity scaling.

locked tests opened: before=False, after=False. No model promoted to production.
