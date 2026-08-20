# HydroCore v5 M10 completion

M10 is complete through `M10_5_SERVING_FREEZE_PASS`. The normal application
now loads the immutable `models/hydrocore-v5-release` bundle for selected seed
`20260814`; it does not rebuild calibration and cannot fall back to v4 on a
v5 asset failure. The release uses the exact M10.5B materialized M10.4
calibration object and keeps deterministic OOD, Scout, Strategist, WNTR/EPANET
verification, human approval, and no-actuation authority boundaries.

`next_step` is suppressed because it was not supervised by canonical M9.6.
The M10.4-tested unobserved-age runtime feature behavior is retained, with its
known training/serving deviation disclosed rather than changed. Historical
blocked M10.5 closures remain immutable evidence and are superseded only by
the additive current-status index.
