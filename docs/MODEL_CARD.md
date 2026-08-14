# HydroCore-v4 Model Card

## Current shipped model

HydroSwarm ships the frozen `hydrocore-v4` **small** variant: 4,182,612
parameters (4.18M), with model SHA-256
`a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7`.
Its feature-schema hash is
`7ec97775e5f01f87ae62669146a7eb70958f99b1162a356614eb87220e9ddd09`;
normalization hash is
`e0808f21579b693f66e4edb5900e561bcf9c521e850d5c9d2428cb0db0fa1114`.
The self-contained release bundle and [FINAL_SYSTEM.md](FINAL_SYSTEM.md) are
the current identity source of truth.

The architecture combines structural/temporal/sensor-quality encoders,
forward-only local graph layers, bounded latent attention, a feature-only
classical prior, and task heads. It is an advisory component in a hybrid
classical/neural pipeline; exact WNTR verification and human approval remain
outside the model and authoritative.

## Runtime promotion and authority

Runtime-enabled outputs are `source_node`, `event_presence`, `event_cause`,
`evidence_sufficiency`, `next_step`, and `relative_strength`. Other trained
heads (including candidate reduction, sampling, fault, plan-value, and
consequence proxies) are not promoted. The learned OOD head is excluded: it
received no real training-split gradient and must not influence live authority.

Source probabilities are fused with a classical signature posterior. The
shipping split-conformal calibration artifact has alpha 0.1, hash
`cf06c2000ead772d7de2d8cdcf00b7cb45e59b325f44be61114982531a4fa4d1`, and
held-out calibration coverage 91.4%. Unknown topology, invalid calibration,
OOD, broad candidate regions, high disagreement, or insufficient evidence
can suppress planning. No model output can verify, approve, or execute a plan.

Calibration was refit (same alpha, same calibration split) after capability
remediation changed the canonical structural-identity hash for the governed
topology families; see [Capability remediation](evaluation/CAPABILITY_REMEDIATION.md)
for the old/new hash pairs. Model weights, feature schema, normalization, and
signature policy did not change.

## Measured evidence

The frozen validation metrics are source top-1 72.1–73.3%, top-3 86.8–87.6%,
MRR 0.811–0.817, conformal coverage 91.4%, event-presence F1 0.895, and
supported event-cause macro F1 0.698; see [Phase 13](../reports/results/v4/phase13-metrics-and-baselines.md).

The frozen robustness-scale characterization sampled 168 governed validation/
development rows. Nominal replay top-1 was 76.2%; the existing unseen-topology
population was 27.8% top-1 and was 100% planning-suppressed. All predeclared
OOD rows were calibration-inapplicable and planning-suppressed; no authority
invariant failure occurred. This is limited synthetic development evidence,
not field validation. Details: [robustness-scale evaluation](evaluation/ROBUSTNESS_SCALE_EVALUATION.md).

The separate LIVE robustness characterization ran 264 real API trajectories
through dynamic fusion, live OOD, active sampling, persistence, planning, and
exact WNTR verification. It recorded 252 safe suppressions, 9 safe
no-usable-evidence abstentions, and 3 verified loop-grid plans; no measured
authority invariant failed. It also found an unresolved OOD truthfulness issue
on an unvalidated topology and a repeated-sampling recommendation issue.
Those findings are not hidden by the favorable safety-boundary result; see
[the LIVE evaluation](evaluation/LIVE_ROBUSTNESS_EVALUATION.md).

## Limitations and locked status

- The model and all reported evaluation data are synthetic; no utility-scale
  or field-performance claim is supported.
- Calibration is marginal and topology/applicability-specific, not a
  per-incident guarantee or cross-topology validation.
- LIVE evidence covers only 6--9-node governed/development networks; it does
  not establish utility-scale performance. **ROB-LIVE-01 and ROB-LIVE-02 are
  remediated**: the post-remediation 264-run replay of the frozen LIVE matrix
  recorded 0 repeated-observed sampling recommendations (was 27) and correct
  `CAUTION`/topology-novelty 1.0 on the unvalidated coastal network (was a
  false `NORMAL`). See
  [LIVE robustness post-remediation](evaluation/LIVE_ROBUSTNESS_POST_REMEDIATION.md).
- **CAP-REM-01** (causal-prefix training-distribution limitation): the
  controlled 1/2/3/6/25-step causal curve (~15/50/45/80/80% top-1) reproduces
  after remediation and is a training-distribution limitation, not a
  parameter-capacity bottleneck. No retraining occurred on the remediation
  branch; a separate causal-prefix HydroCore-v5 experiment is the predeclared
  follow-up. See [Capability remediation](evaluation/CAPABILITY_REMEDIATION.md).
- **CAP-REM-02** (active-sampling scope limitation): a 40-incident paired
  development experiment found EIG reduces posterior entropy but did not beat
  random valid-unsampled selection on samples-to-actionability (37.5% vs
  45.0% actionable within three samples); broad conformal candidate regions,
  not evidence insufficiency, were the dominant blocker. Active sampling
  remains advisory/experimental pending revisit after causal-prefix training.
- Capability remediation changed serving-layer behavior only (canonical
  structural identity, calibration/OOD applicability, causal telemetry
  history, missing-health semantics, the 25-step feature window, calibration
  group selection, and sampling timing); it did not retrain or alter
  HydroCore-v4 weights.
- The locked final evaluation remains unopened (`locked_test_opened: false`).

## Legacy S/M/L research record

The previous S/M/L-generation architecture, its historical locked evaluation,
and promotion decision are retained in [EVALUATION.md](EVALUATION.md) and
historical reports. Those results are not claims about the current v4 runtime.
