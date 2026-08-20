# Milestone 8.6 summary: representation invariance and temporal-feature-usage audit

Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED (4182612 parameters, checkpoint sha256=44a2721394d95985...)
Networks tested: golden-reference (J1-J4, R1, T1), dev-grid-25 (M8's own deterministic grid generator).

## Sections 3-6: node-order, edge-order, sensor-order permutation and node-ID relabeling

Predeclared structural tolerance: max abs posterior diff <= 0.0001 (a conservative margin above float32 summation-order noise for a correctly-implemented equivariant graph model).

| transformation | max abs diff observed (all cases/seeds) | top1 identity preserved everywhere | result |
|---|---|---|---|
| node-order permutation | 1.45337e-07 | True | PASS -- exact equivariance/invariance confirmed |
| edge-order permutation | 4.77697e-08 | True | PASS -- exact equivariance/invariance confirmed |
| sensor-order permutation | 0 | True | PASS -- exact equivariance/invariance confirmed |
| node-ID relabeling (full pipeline) | 1.23635e-07 | True | PASS -- exact equivariance/invariance confirmed |

## Section 7: timestamp-origin translation

Predeclared tolerance: max abs posterior diff <= 0.01 (looser than Sections 3-6, reasoned out BEFORE running -- see module docstring's a priori float32-epoch-precision hypothesis, which Section 10 below shows was NOT actually the mechanism).

| case | +1h max abs diff | +24h max abs diff | +7d max abs diff | any offset failed |
|---|---|---|---|---|
| golden-reference:early_N2 | 1.34876e-07 | 3.43137e-06 | 3.21005e-05 | False |
| golden-reference:mid_N8 | 2.55853e-08 | 7.08542e-07 | 7.18471e-06 | False |
| golden-reference:mature_full | 7.12465e-09 | 2.6833e-07 | 3.30583e-06 | False |
| golden-reference:no_event_mature | 0.000455364 | 0.0116292 | 0.0906047 | True |
| dev-grid-25:early_N2 | 0.00149351 | 0.0313151 | 0.0514614 | True |
| dev-grid-25:mid_N8 | 0.000883006 | 0.0164979 | 0.0233134 | True |
| dev-grid-25:mature_full | 0.000101864 | 0.00204562 | 0.00584128 | False |
| dev-grid-25:no_event_mature | 0.000221686 | 0.00355734 | 0.0071634 | False |

**Largest posterior discrepancy across the entire invariance matrix: 0.0906047** (timestamp-origin translation; every other transformation class stayed within float32 noise, <=1.5e-7).

## Section 10: failure triage

Failing transformation class: **timestamp_origin_translation**

Classification: **ABSOLUTE_TIME_ORIGIN_LEAKAGE**

Smallest reproducer: src/hydroswarm/preprocessing/builder.py:155 -- `age = now - series.timestamps_seconds[-1] if series else now`. For any node with NO sensor coverage at all (e.g. every reservoir/tank, or any unmonitored junction), the `else` branch falls back to the raw `now` value (the incident's own elapsed-time-so-far) instead of a fixed, origin-independent 'never observed' sentinel. `now = max(item.timestamps_seconds[-1] for item in sensor_series)` shifts by the exact same constant as every real observation's timestamp, so this column is NOT translation-invariant for unobserved nodes, unlike every other temporal quantity in this builder (all computed as a genuine elapsed DIFFERENCE, not a bare absolute/elapsed value).

Confirmed NOT the cause:
- SIGCHLD/process reaping -- not applicable, this is a pure Python/NumPy computation, no subprocess involved.
- float32 absolute-Unix-epoch-timestamp precision loss (the module docstring's a priori hypothesis) -- REFUTED by direct measurement: this benchmark's `timestamps_seconds` values are already small, incident-relative elapsed seconds (matching production's own `hydroswarm.api.app.sensor_series` convention, `(observed_at - detected_at).total_seconds()`), not large absolute epoch values, so no meaningful float32 rounding occurs at this magnitude; and directly neutralizing the explicit `timestamps` batch key (Section 8 arm B) left the discrepancy essentially unchanged (0.09060 -> 0.09060), proving the explicit TemporalEncoder timestamp pathway is NOT the source.
- DATA_BUILDER_ORDER_DEPENDENCE / NODE_INDEX_MAPPING_DEFECT / EDGE_AGGREGATION_ORDER_DEPENDENCE / SENSOR_SERIALIZATION_DEPENDENCE / LITERAL_IDENTIFIER_LEAKAGE -- all directly ruled out by Sections 3/4/5/6 passing at float32-noise-floor magnitude.

Materiality: Effect size is state-dependent: negligible under confident/mature event-positive evidence (max abs diff <=6e-5 across all golden-reference/dev-grid-25 event-positive mature cases), largest under low-confidence/high-uncertainty evidence (no-event state, or sparse early/mid evidence on the sparsely-sensed 25-node grid) where the posterior is closer to uniform and more sensitive to any nuisance input. Also grows with how far the injected origin shift departs from the training-typical range production timestamps actually occupy (+1h: negligible everywhere; +24h/+7d: the offsets that actually trip the predeclared tolerance).

Why not fixed in M8.6: The fix (replacing the `else now` fallback with a fixed, origin-independent sentinel) would change the numeric value HydroCore's frozen node_encoder receives for every unobserved node relative to what the FROZEN Milestone-1 checkpoint was actually trained on -- this is exactly the kind of feature-computation change Section 1 of this milestone explicitly prohibits ('Do NOT alter... feature dimensions, normalization statistics') without retraining, since the model's weights encode an implicit expectation of this column's current (buggy) distribution. Correcting it blind, without retraining, could plausibly make frozen-model behavior WORSE, not better, on real incidents whose unobserved-node ages happen to already resemble what training saw.

## Section 8: temporal-feature-usage counterfactuals

Temporal pathways identified by inspection (not assumed from prior reports):
- **Explicit timestamp pathway**: `HydroBatch['timestamps']` -> `TemporalEncoder`/`QualityEncoder`'s `timestamps` argument (sinusoidal phase added to the projected sequence).
- **Derived age-feature pathway**: `node_features[..., 9]` (`measurement_age`, static per-node scalar), `temporal_features[..., 2]`, `quality_features[..., 3]` (per-timestep age/86400 channels).

| arm | mean L1 | mean JS divergence | top1 change rate | top3-set change rate | mean entropy delta |
|---|---|---|---|---|---|
| EXPLICIT_TIMESTAMP_NEUTRALIZED | 0.000833 | 1.60802e-06 | 0 | 0 | -0.00328971 |
| AGE_FEATURES_NEUTRALIZED | 0.0328824 | 0.00286907 | 0.1 | 0.1 | 0.0909455 |
| ALL_TEMPORAL_NEUTRALIZED | 0.032473 | 0.00278437 | 0.1 | 0.1 | 0.086114 |

By evidence maturity (AGE_FEATURES_NEUTRALIZED mean JS divergence -- effect grows with maturity; EXPLICIT_TIMESTAMP_NEUTRALIZED stays near zero at every maturity):

| maturity | AGE_FEATURES_NEUTRALIZED JS | EXPLICIT_TIMESTAMP_NEUTRALIZED JS |
|---|---|---|
| early_N2 | 1.16371e-06 | 4.30879e-06 |
| early_N3 | 1.00195e-05 | 2.5299e-06 |
| mid_N8 | 0.00152488 | 1.11939e-06 |
| no_event_mature | 0.00150212 | 2.01833e-09 |
| mature_full | 0.0113072 | 8.00065e-08 |

## Section 9: REPRESENTATION_SENSITIVITY_COUNTERFACTUAL (same reports, different elapsed spacing)

Diagnostic only -- not a real-trajectory accuracy claim. Tight (10-20min) vs wide (10-20hr) inter-report spacing, same concentration values/report count/order, through the normal feature-generation path.

| case | report count | max abs diff (tight vs wide) | top1 preserved |
|---|---|---|---|
| golden-reference:early_N2 | 2 | 1.95776e-06 | True |
| golden-reference:early_N3 | 3 | 9.7996e-05 | True |
| dev-grid-25:early_N2 | 2 | 0.00533957 | True |
| dev-grid-25:early_N3 | 3 | 0.0299543 | True |

golden-reference (all nodes observed, no unobserved-node age-fallback confound) shows near-zero spacing sensitivity (~2e-6 at N=2), directly confirming the module docstring's TemporalEncoder analysis: its elapsed-time normalization (`elapsed - elapsed[:,:1]`, then divided by the window's own span) collapses to the same relative phase pattern regardless of the actual absolute gap size. dev-grid-25's larger spacing sensitivity is confounded by the SAME Section 7/10 unobserved-node age-fallback defect (changing spacing changes `now`, which changes unobserved nodes' age value on that sparsely-sensed network) rather than demonstrating a separate, genuine spacing-sensitivity effect.

## Large-network consistency (golden-reference vs dev-grid-25)

Node-order/edge-order/sensor-order/relabeling: identical (exact-pass) behavior on both networks. Timestamp-origin: same underlying mechanism on both (confirmed via Section 10's triage), differing only in magnitude/threshold-crossing because dev-grid-25 has a much higher unobserved-node fraction. This is a representation-correctness comparison only -- not an unseen-topology generalization claim.

## Verdicts

**PRIMARY VERDICT: REPRESENTATION_CORRECTION_REQUIRES_RETRAINING**

**Temporal usage classification: TEMPORAL_FEATURE_USAGE_WEAK_OR_PARTIAL**

The explicit timestamp/positional-encoding pathway (TemporalEncoder/QualityEncoder's `timestamps` argument) is measurably INERT: neutralizing it changes top1 predictions in 0% of tested cases and moves the posterior by a mean JS divergence of order 1e-6, at every evidence maturity tested (including mature/full-window evidence, where the effect should be largest if this pathway carried real information). The derived age-feature pathway (measurement_age / per-timestep age channels) IS measurably used and its influence GROWS with evidence maturity (mean JS divergence ~1e-6 at N=2 vs ~0.011 at full 25-report maturity) -- so temporal information overall is not unused, but one of the two designed pathways is carrying essentially none of that signal. This is exactly the 'only one pathway is materially used while another expected pathway is inert' criterion this milestone's own instructions give as a reason to warrant M8.7.

**M8_7_TEMPORAL_REPRESENTATION_EXPERIMENT_WARRANTED: YES**

locked tests opened: before=False, after=False. No model retrained. No architecture, calibration, alpha/K, OOD logic, planning, or safety/authority semantics changed.
