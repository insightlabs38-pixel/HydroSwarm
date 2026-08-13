# LIVE Robustness Post-Remediation Comparison

## Scope and provenance

This compares the preserved pre-remediation 264-row discovery campaign with a
new 264-row run of the exact frozen condition matrix. The original artifacts
remain unchanged. The post-remediation runner recorded study baseline
`e45f72cf730d3f12c13dbcb9403c64f185510173` separately from runtime commit
`2c7060fa7dd5149a05b9828124ddcd5f0d8b8bec`.

Model, calibration, feature schema, normalization, signature policy, alpha,
OOD thresholds, fusion/disagreement/evidence/planning thresholds, and WNTR
policy are unchanged. `locked_test_opened` was false before and after.

## Findings

| Finding | Pre-remediation | Post-remediation | Status |
| --- | --- | --- | --- |
| ROB-LIVE-01 | 27 repeated-observed recommendations | 0 | REMEDIATED |
| ROB-LIVE-02 coastal OOD | novelty 0.0; `NORMAL` in 24/24 | novelty 1.0; `CAUTION` in 24/24 | REMEDIATED |

Post-fix sampling endpoint behavior is fail-closed when current authority is
not `REQUEST_SAMPLE`, when no current real result exists, when the result is
stopped, or when its node is already represented in evidence. Expected
information gain remains an expectation: acquired samples can still increase
realized posterior entropy.

## Aggregate comparison

| Metric | Pre | Post |
| --- | ---: | ---: |
| Runs | 264 | 264 |
| Top-1 | 0.318 | 0.306 |
| Top-3 | 0.882 | 0.847 |
| MRR | 0.586 | 0.571 |
| Planning-eligible rate | 0.012 | 0.012 |
| Invariant failures | 0 | 0 |
| Exact simulator calls | 4 | 4 |
| Plans / VERIFIED / ABSTAINED | 4 / 3 / 1 | 4 / 3 / 1 |
| Sampling rounds | 333 | 309 |

The small localization differences are retained. OOD novelty now contributes
to live trust/fusion and invalid recommendations are removed, so post-fix
trajectories need not be numerically identical; this is not interpreted as an
accuracy improvement claim.

## Controls and limitations

Validated loop-grid had topology novelty 0.0. Golden-reference remained
structurally outside calibration support, calibration-inapplicable, topology
novel, `CAUTION`, and planning-suppressed: GOLD-EXPECTED remains unchanged.
The valid network-size range is still 6--9 nodes and no field-performance or
utility-scale claim is supported.

Raw post-remediation records are `post-remediation-results.json`,
`post-remediation-results.csv`, and `post-remediation-summary.json`.
