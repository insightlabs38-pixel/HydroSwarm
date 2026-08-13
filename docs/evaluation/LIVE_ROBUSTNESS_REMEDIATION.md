# LIVE Robustness Remediation

This remediation preserves the original pre-fix LIVE artifacts unchanged.

## ROB-LIVE-01 — sampling authority

**Reproduction/root cause.** The API fell back from a current live analysis to
persisted candidates and could return a candidate-node recommendation without
a current `REQUEST_SAMPLE` action or real active-sampling result.

**Correction.** `/samples/recommend` now returns 409 unless the current
`IncidentAnalysisResult` has `control_action == REQUEST_SAMPLE`, a non-stopped
real sample result, and a recommended node not represented in current
evidence. Audit events are appended only after that validation.

**Post-fix result.** Focused API/sampling tests pass; demo/persisted candidate
fallbacks are now fail-closed.

## ROB-LIVE-02 — topology OOD identity

**Reproduction/root cause.** Calibration used `network_sha256(network)` while
live OOD received `simulator.state_hash()` and a default empty topology allow
list. Thus topology novelty could not agree with calibration applicability.

**Correction.** Production V4 constructs `OODReference` from
`calibration.validated_topology_hashes`; the pipeline passes the same
structural `network_sha256(network)` identity to OOD. State hash remains
provenance only.

**Post-fix result.** All 24 coastal targeted rows are calibration-inapplicable,
have topology novelty 1.0, OOD `CAUTION`, planning disabled, and no invariant
failure. Loop-grid structural hash remains in the allow-list.

## Golden-reference investigation

**Classification: GOLD-EXPECTED.** `golden_network.inp` has structural hash
`decd4ac707a3817115c7307940861dafac6d3fad34a3358179f20c7830a2cc79`, absent
from calibration's validated structural hashes. The live harness imported its
actual bytes correctly. `network_sha256` deliberately includes graph links
and static hydraulic link attributes, including roughness; scenario hydraulic
randomization can therefore change the support identity. Existing calibration
comments and corpus metadata define this as complete calibration-domain
structural configuration, not connectivity alone. Roughness variation should
therefore change calibration applicability under the current frozen design.
No golden product correction is required.

## Identity audit

| Identity | Meaning | Consumers |
| --- | --- | --- |
| Network file SHA | imported-byte integrity | API import/verification binding |
| Structural topology hash | graph plus static hydraulic support identity | calibration, OOD novelty, model-input signatures |
| Simulator state hash | live hydraulic/evidence-state provenance | signatures, verification provenance |
| Signature artifact hash | generated classical signature identity | inference provenance/cache |

Model, calibration, feature schema, normalization, signature policy, alpha,
thresholds, and WNTR policy are unchanged. No additional finding was found.
