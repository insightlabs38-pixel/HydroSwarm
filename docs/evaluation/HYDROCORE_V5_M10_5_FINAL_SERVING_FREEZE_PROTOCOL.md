# M10.5 resumed serving-freeze protocol

This additive resumed M10.5 protocol follows committed M10.5A deployment
identity `6829c676bf9aa074dbc9e62150b256efd0475335`: seed 20260814,
canonical M9.6 `FINAL_STEP_1350` SHA-256
`de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`.

The release must load that identity and the exact frozen M10.4 calibration
semantics without re-fitting calibration.  It must preserve M10.4 feature
semantics, remove untrained `next_step` from any v5 allowlist, and never fall
back silently to v4.

Preflight finding: `M10_4_PipelineFactory.__init__` obtains its selected-seed
calibrator by calling `fit_frozen_calibrator`, whose implementation calls
`SplitConformalCalibrator.fit(...)`.  M10.4 reports no immutable per-seed
calibration artifact that the normal v5 runtime can load.  Creating one now
would perform the forbidden calibration fit/reconstruction; changing the
calibration path would violate M10.4 parity.  Therefore this protocol freezes
the implementation blocker and forbids bundle creation/default rewiring.
