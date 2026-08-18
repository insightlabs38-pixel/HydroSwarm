# HydroCore-v5 M11.5 full-validation results

Closure: `M11_5_FULL_VALIDATION_FAIL`. The exact M11.2 finalist identity was
unchanged, all non-locked scientific/runtime rows passed, and locked access
remained prohibited. The matrix is nevertheless not green because the
software/release-quality row is hard-gating and failed.

## Matrix result

The 14-row matrix contains 13 hard rows and one descriptive limitations row.
It recorded 12 PASS, 1 FAIL, and 1 DESCRIPTIVE result. Rows A--M passed or
remained descriptive as frozen: exact clean-process identity/release loading,
closed predictive/calibration/robustness/OOD/Scout/planning/end-to-end evidence,
fail-closed/no-v4-fallback behavior, and output governance all matched the
M11.2 certificate. Every authority/safety counter was zero except the matrix
summary's `invariant_failures=1`, which records the failing software row.

Row N failed for two independently recorded reasons:

1. Full Python pytest returned 1 failure, 1758 passed, and 3 skipped after
   1327.99 seconds. The failure was
   `tests/scientific/test_m11_2_finalist_freeze.py::test_m11_2_artifacts_freeze_without_authorizing_locked_evaluation`.
2. Docker build could not start because the Docker daemon socket was absent:
   `unix:///var/run/docker.sock` was unavailable. This is an environment
   limitation, but Docker is an established required release gate and is not
   treated as PASS.

Pyright passed with zero errors/warnings; changed-file ruff passed; strict
v5 self-test passed; frontend lint, typecheck, tests (162 tests), and production
build all passed. No repair, rerun-for-selection, tuning, or release mutation
was performed after the failures.

## Safety, identity, and lock state

The finalist remains `HydroCore-v5 M10 frozen release`, with the same selected
seed, checkpoint, release manifest, calibration artifact, feature semantics,
fusion, sentinel-only tasks, and governed output allowlist. Deterministic OOD,
Scout, planning, WNTR/EPANET, and human approval authority were preserved;
learned OOD/Scout/Strategist remained non-authoritative and autonomous actuation
remained absent.

`locked_test_opened_before=false`, `locked_test_opened_after=false`, and
`locked_evaluation_authorized=false`. The seven known limitations remain
carried forward without a claimed resolution.

## Consequence

M11.6 prerequisites are **not** satisfied: its M11.5-green-matrix prerequisite
is false. M11.6 is not authorized and must not be run. Any future remediation
of the failing software gates requires separate governance; it is outside this
M11.5 execution.
