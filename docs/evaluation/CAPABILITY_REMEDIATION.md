# Capability remediation

This remediation starts from merged main
`dec954c7dbc3408469d1dbc412ad4be83d310585`. The diagnostic reports remain
unchanged. All results here use development-only data; the locked final and
locked topology tests were not opened.

## Corrected product behavior

- Governed network identity is one canonical, explicit nine-significant-digit
  structural serialization. It makes programmatic construction and canonical
  EPANET parsing agree without an allow-list. Mutable demand and tank state
  remain simulator-state provenance, not structural identity.
- Production now carries all causal reports through analysis and explicitly
  supplies HydroCore-v4's 25-step maximum feature window. Missing readings
  produce effective health 0.0.
- Runtime uses canonical governed family plus deterministic evidence condition
  to select Mondrian calibration and exposes both calibration source and group
  in analysis and the immutable audit event.
- Sampling ranks the predicted measurement at recommendation time plus the
  reported per-node collection delay. The API returns that delay, and the
  development evaluator materializes the matching delayed, seeded-noisy
  measurement (sigma 0.05 mg/L).

The product and evidence contract is [PRODUCT_CAPABILITY_CONTRACT.md](../PRODUCT_CAPABILITY_CONTRACT.md).
Network compatibility, model/calibration applicability, and operational
readiness remain independent concepts.

## Measurements

The complete machine-readable record is in
`reports/evaluation/capability-remediation/`.

- Calibration was refit only on the designated 712-example calibration split,
  preserving alpha 0.1. Coverage is 0.9143 and mean candidate size is 2.8006.
  Model, schema, and normalization hashes did not change.
- Canonical golden, branched-loop, and loop-grid are all calibrated with OOD
  NORMAL rate 1.0 in the development topology study. Coastal unseen is
  calibration-invalid, has OOD NORMAL rate 0.0, and has planning rate 0.0.
- The causal-prefix study (n=20 golden development incidents) gives top-1 of
  0.15/0.50/0.45/0.80/0.80 at 1/2/3/6/25 steps, while final latest-only is
  0.20. This confirms the runtime no longer silently acts latest-only, but
  also demonstrates weak genuinely early evidence.
- The API-driven nominal LIVE slice (n=4) has top-1/top-3/MRR 1.0/1.0/1.0,
  calibration-valid rate 1.0, OOD NORMAL rate 1.0, planning eligibility 0.75,
  and no authority invariant failures.
- The sparse 50%-coverage paired sampling slice (n=20, three-step prefix,
  three-sample budget) has EIG median realized entropy reduction 0.7818 bits
  versus 0.0 for random; however actionable within three samples is 0.0 for
  both and EIG final top-1 is 0.70 versus random 0.75. Safety thresholds were
  not adjusted to mask that result.
- Full train/serve parity passes across all three governed networks and clean
  and degraded conditions, including node/edge/temporal-quality masks,
  classical prior, ordering, and signatures.

## Findings and model decision

CAP-REM-01: early causal prefixes remain materially weak (top-1 0.15 at one
step and 0.45 at three) while later causal evidence is strong. This is a
training-distribution finding, not parameter-count evidence.

CAP-REM-02: after semantic alignment, EIG reduces entropy but does not
materially outperform random on the primary actionability metric in the
predeclared sparse development slice. This requires a focused follow-up;
no safety or fusion tuning was performed here.

Therefore the model decision is:

> CAUSAL-PREFIX RETRAINING JUSTIFIED, CAPACITY INCREASE NOT YET JUSTIFIED

The branch is **not ready for a PR** until CAP-REM-02 is resolved or scoped as
a product limitation and the broader required LIVE/safety campaign is run.
