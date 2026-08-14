# Capability remediation

This branch begins the capability-remediation phase from merged main
`dec954c7dbc3408469d1dbc412ad4be83d310585`. The diagnostic evidence remains
immutable: `git diff` against that base shows no changes under
`reports/evaluation/capability-diagnostic/`,
`docs/evaluation/CAPABILITY_DIAGNOSTIC.md`, or
`docs/evaluation/CAPABILITY_DIAGNOSTIC_PROTOCOL.md`.

## Product and evidence contract

The authoritative product contract is
[`PRODUCT_CAPABILITY_CONTRACT.md`](../PRODUCT_CAPABILITY_CONTRACT.md). Runtime
accepts one or more causal reports per sensor, retains all reports available at
the decision time, and bounds HydroCore-v4 features to the latest 25 causal
report steps. It does not invent history or consume future observations.

## Implemented corrections

- CAP-DATA-01: Structural identity now serializes static physical link values
  at explicit deterministic precision. It makes programmatic and EPANET `.inp`
  representations of a governed network match, while demand/tank state remains
  simulator provenance rather than topology identity.
- CAP-CAL-01 / CAP-OOD-01: Calibration and OOD use that same canonical identity.
  Calibration was refit from only the 712-example designated calibration split
  at alpha 0.1; coverage is 0.9143 and the model weights are unchanged.
- CAP-PARITY-01 / CAP-PARITY-02: Missing readings contribute zero effective
  health, and runtime explicitly supplies a 25-step feature window.
- CAP-TEMPORAL-01: The LIVE harness now submits all reports available by its
  decision time rather than one final report per sensor.
- CAP-CAL-02: Runtime selects calibration by canonical governed family and a
  deterministic evidence-derived condition, recording NETWORK_SPECIFIC,
  CONDITION_SPECIFIC, GLOBAL, or INAPPLICABLE selection in the result.
- CAP-SAMPLE-01: Production EIG receives a real decision time and ranks the
  measurement expected after each candidate's collection delay. It does not
  use a future trajectory peak.
- CAP-DATA-02: permutation support now covers sensor and quality masks.

## Measured post-fix slice

`live-capability-results.json` is a new, causal, four-run canonical
golden-reference development slice. It records top-1/top-3/MRR of 1.0, initial
planning eligibility of 0.75, and zero authority-invariant failures. This is
not a full replacement for the required paired EIG-vs-random experiment,
early-time checkpoint curve, or topology transfer campaign; those remain
necessary before treating the branch as a release candidate.

The locked test was false before and after the calibration and LIVE stages.
