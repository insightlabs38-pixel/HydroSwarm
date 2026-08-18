# HydroCore v5 M10.5B calibration materialization amendment

This amendment authorizes one deterministic release-artifact operation and no
calibration development: serialize the exact calibrator constructed by
`scripts/hydrocore_v5/m10_4_common.py::fit_frozen_calibrator` for the already
frozen M10.5A deployment identity.

## Frozen inputs

- selected seed: `20260814`
- selected checkpoint SHA-256:
  `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`
- constructor: `m10_4_common.fit_frozen_calibrator`
- invocation: load the canonical model for seed 20260814, then call the
  constructor with its real hash and `trained_family_topology_hashes()`.
- support: `reports/evaluation/hydrocore-v5/m9-6/m9-6-canonical-calibration.jsonl`,
  arm `ARM_B_M9_6`
- alpha: `0.1`; minimum group size: `10`; grouping: existing
  `family:depth_bucket` network grouping from the frozen support rows.
- feature schema: `DEFAULT_FEATURE_SCHEMA.fingerprint`; fusion identity:
  `DYNAMIC_TRUST_FUSION_CONFIG`.

The runner must not generate, modify, select, or inspect calibration data
beyond this frozen support; it must not inspect locked evaluations. It invokes
the existing M10.4 helper exactly twice solely to demonstrate deterministic
materialization. It does not call `SplitConformalCalibrator.fit` itself or
implement a scoring formula.

Pass requires equal independent artifacts, save/load structural equality,
byte checksum validation, and equal selected group/threshold source/candidate
set on every arm-filtered support row. The resulting artifact schema remains
`hydroswarm-calibration-v1`.
