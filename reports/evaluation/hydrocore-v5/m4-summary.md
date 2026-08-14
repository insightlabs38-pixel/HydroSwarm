# Milestone 4 summary: robust planning under source uncertainty

Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED
Calibration: B_DEPTH_AWARE (Milestone 3 frozen scheme, refit identically here).
K (max verified candidate sources) = 3 (hydroswarm.simulation.wrapper.MAXIMUM_EVALUATION_HYPOTHESES -- the existing hard ceiling, equal to production's maximum_planning_candidates default; never relaxed).

## Scope limitation

ROBUSTLY_VERIFIED in Milestone 4 means exact verification across every plausible source-location candidate in the conformal region, holding the remaining modeled incident-profile assumptions fixed. It is not a claim of robustness over all uncertain contamination parameters. The WeightedSourceHypothesis set varies candidate source node identity only; start_minute, duration_minutes, and relative_strength are held at IncidentSourceProfile's governed defaults for every hypothesis (matching production's own _runtime_evaluation_context), never jointly sampled or varied across the hypothesis set. This milestone does not model uncertainty in source timing, duration, or injection strength.

## Ground-truth safety audit methodology (Milestone-4 correction)

Every eligible incident where control or robust selected a plan is exact-WNTR verified against the real held-out `IncidentSourceProfile`, unconditionally -- including incidents where the true source falls OUTSIDE the conformal candidate set (a calibration-coverage miss). The prior revision only ran this check when the true source was inside the candidate set, so coverage misses were silently reported as false_safe=False without ever being tested. That gate has been removed.

## Per-bucket results (development_holdout, one representative depth/bucket, n<=40/bucket)

| bucket | depth | n | eligible | exceeds-K | coverage hit | coverage miss | control verified | robust verified | control false-safe (unconditional) | robust false-safe (unconditional) |
|---|---|---|---|---|---|---|---|---|---|---|
| EARLY | 2 | 40 | 0.55 | 0.45 | 18 | 4 | 0.91 | 0.91 | 0 | 0 |
| MID | 4 | 40 | 1.00 | 0.00 | 33 | 7 | 0.95 | 0.95 | 0 | 0 |
| MATURE | 12 | 40 | 0.90 | 0.00 | 36 | 0 | 0.94 | 0.94 | 0 | 0 |

## False-safe breakdown by policy (unconditional / conditional-on-coverage-hit / coverage-miss-only), overall

| policy | selected | selected on hit | selected on miss | false-safe (unconditional) | rate | false-safe (coverage hit) | rate | false-safe (coverage miss) | rate |
|---|---|---|---|---|---|---|---|---|---|
| control | 92 | 83 | 9 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 |
| robust | 92 | 83 | 9 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 |

## Overall (n=120)

- Planning eligible (region size 1-3): 0.817
- Region exceeds K (not reachable without a future architecture decision): 0.150 -- fails closed under both policies, no actionability claimed.
- Candidate-set coverage: 87 hit / 11 miss (true source outside the conformal candidate set) among eligible incidents -- coverage misses are now included in the unconditional false-safe audit below, not skipped.
- Control (naive single top-1-hypothesis) verified rate of eligible: 0.939
- Robust (whole-region multi-hypothesis) verified rate of eligible: 0.939
- Genuinely multi-candidate incidents (size > 1, the only population where control and robust could structurally disagree -- size-1 regions make robust degenerate to control by construction): 24 of 120 total (24 shown for cross-check).
- Decision disagreement rate among genuinely multi-candidate incidents: 0.0 (0 of 24) -- **on this held-out sample, robust whole-region verification never changed the verify/reject decision relative to naive top-1-only verification.** This is an honest negative finding for the mechanism's empirically demonstrated safety value-add on this topology, not evidence the mechanism is unnecessary in general: it reflects that here, action-template plan safety (pressure/service constraints) was largely source-location-invariant, not that whole-region verification is redundant by design (the K=3 architectural guarantee against false-safe holds regardless).
- Control false-safe count (naive-verified plan actually unsafe against the real held-out source): 0 (rate of control-verified: 0.0)
- Robust false-safe count (robustly-verified plan actually unsafe against the real held-out source): 0 (rate of robust-verified: 0.0)
- Mean exact simulator calls/incident: control=6.0, robust=6.918367346938775
- Mean wall-clock seconds/incident (control+robust combined): 0.727516220163199

**Zero robust authority invariant violations: True.** Control (naive) authority invariant violations present: False.

**Material actionability gain: False** -- K equals production's existing maximum_planning_candidates threshold, so eligibility is identical between arms; this experiment could not and does not claim a reachable-incident actionability increase (see module docstring for why, and the exceeds-K rate above for how much traffic a future K-relaxation would need to address).

**Exit decision: MECHANISM_VALIDATED_NO_MATERIAL_ACTIONABILITY_GAIN_AT_K3**
