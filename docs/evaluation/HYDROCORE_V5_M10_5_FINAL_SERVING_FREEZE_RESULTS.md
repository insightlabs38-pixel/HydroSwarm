# M10.5 resumed serving-freeze results

## Result: `M10_5_SERVING_FREEZE_BLOCKED_IMPLEMENTATION`

M10.5A validly selected seed 20260814 in a separately committed,
performance-independent amendment.  Resumed M10.5 cannot create the required
immutable serving bundle because the exact M10.4 per-seed calibration state is
not stored as a loadable artifact.  The evaluation factory rebuilds it through
`SplitConformalCalibrator.fit`; doing that in release packaging would refit
calibration, expressly prohibited by M10.5.

No release bundle, default serving change, calibration rewrite, v4 fallback,
or parity experiment was performed.  The historical v4 release and every
prior artifact remain unchanged.  A separately authorized amendment must
provide an immutable, hash-verified selected-seed calibration artifact derived
without a refit before M10.5 can resume.
