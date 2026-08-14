# Capability remediation finalization

This branch remediated the demonstrated serving defects without changing the
HydroCore-v4 weights: canonical structural identity, calibration/OOD
applicability, causal telemetry history, missing-health semantics, the
25-step feature window, calibration group selection, and sampling timing.
Historical diagnostic and pre-remediation evidence remain unchanged.

## Scientific identity

The final campaigns ran against code commit
`41aae8795cd2d25894f9521f5ac7ea17d04256a5`. HydroCore-v4 remains
`a501ad87…`; feature schema, normalization, signature policy, fusion, and
authority thresholds are unchanged. Calibration was previously refit solely
from the designated calibration split after the canonical topology identity
change. The locked test and locked topology test remain unopened.

The governed canonical golden hash is `5508d272…`; programmatic and EPANET
paths resolve the same identity. Supported topology observations are calibrated
and OOD-NORMAL; unseen coastal is calibration-invalid, topology-novel
(1.0), OOD-CAUTION, and cannot plan.

## Causal evidence and model decision

The causal temporal curve remains: top-1 0.15/0.50/0.45/0.80/0.80 at
1/2/3/6/25 steps, versus 0.20 for final latest-only evidence. Controlled
validation reproduced top-1 0.7205, top-3 0.8680, and MRR 0.8113.

CAP-REM-01 is therefore a **causal-prefix training-distribution limitation**.
It does not establish a parameter-capacity bottleneck. No retraining was
performed here; the appropriate follow-up is a separate causal-prefix
HydroCore-v5 experiment.

## CAP-REM-02: active sampling

The final paired development experiment used 40 canonical golden incidents,
three causal steps, 50% initial sensor coverage, three samples, 30-minute
acquisition delay, and matched seeded 0.05 mg/L noise.

EIG had median realized entropy reduction 0.9572 bits and positive realized
reduction in every EIG round, with expected/realized Spearman 0.544. It did
not demonstrate better operational actionability: EIG was actionable within
three samples in 0.375 of incidents; random valid-unsampled was 0.450.

The post-sample decomposition found no stale analysis, calibration-group,
timestamp, or API/runtime divergence. In localization-correct-but-suppressed
states, conformal candidate breadth was the dominant blocker (48 states),
followed by learned model-evidence insufficiency (20). The candidate-region
blocker was the sole blocker in 45 states. Thresholds and alpha were not
retuned.

CAP-REM-02 is consequently scoped as a **current product limitation**:
active sampling is evidence guidance that can reduce uncertainty, but current
development evidence does not show that it reduces samples-to-actionability
versus random in the sparse causal regime. It remains advisory/experimental
until revisited after causal-prefix training.

## Full production-path LIVE campaign

The frozen 264-run API-driven LIVE matrix completed with zero authority
invariant failures. A separate six-run clean supported-topology control slice
then covered branched-loop and loop-grid without altering that frozen matrix.
Together the 270 API-path runs cover nominal, missingness, sensor health,
sensor coverage, measurement noise/bias, hydraulic shifts, ambiguity, unseen
topology, scale, and all three governed supported networks. Overall analyzed
outcomes were top-1 0.9087, top-3 0.9722, MRR 0.9472; supported nominal was
0.8889/0.9444/0.9278. Canonical supported runs had no false
calibration-invalid result and were OOD-NORMAL; coastal remained
calibration-invalid, topology-novel, OOD-CAUTION, and ineligible to plan.

The combined campaign generated 367 plans, 185 VERIFIED results, 182
ABSTAINED plan results, and 367 exact WNTR calls. It acquired 23 samples; no
already-observed recommendation occurred. ROB-LIVE-01 and ROB-LIVE-02 are
remediated.

The frozen matrix does not define cumulative `actionable within 1/2/3`
metrics: it retains final authority and selected post-sample states, not a
complete initial-to-round authority trajectory for every incident. Those rates
are therefore intentionally null in `live-capability.json`; the 40-incident
paired sampling campaign is the authoritative measurement of actionability
within a sample budget.

See the machine-readable final record in
`reports/evaluation/capability-remediation/`, notably `sampling.json`,
`sampling-blockers.json`, `full-live-results.json`,
`supported-topology-controls.json`, `full-live-summary.json`,
`safety-regressions.json`, `validation.json`, and `summary.json`.

## Local release validation

The final local validation record is published in `validation.json`:
1,136 Python tests passed with one documented historical skip in 657.36
seconds; the strict self-test passed; and frontend lint, typecheck, format,
test (29 files/162 tests), and build gates all passed. The focused remediation
suite also passed 31 tests. Following the Docker CI regression, its focused
helper/authority regression set passed 24 tests.

## Remaining limitation and next step

Do not claim EIG operational superiority or treat an uncommissioned topology's
planning suppression as a localization failure. The branch includes only the
design for network commissioning in
[NETWORK_COMMISSIONING_DESIGN.md](../NETWORK_COMMISSIONING_DESIGN.md).

The next model work, if authorized, is a separate causal-prefix training
experiment—not a capacity increase or architecture rewrite.
