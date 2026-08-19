# HydroCore-v5 M11.5 full-validation results

Closure: `M11_5_FULL_VALIDATION_PASS`. The exact M11.2 finalist identity was
unchanged, all non-locked scientific/runtime rows passed, the
software/release-quality row passed, and locked access remained prohibited.
The matrix is green and M11.6 prerequisites are satisfied, but the locked
evaluation is **not** authorized and requires explicit human authorization.

This document supersedes the earlier `M11_5_FULL_VALIDATION_FAIL` record (kept
in git history at commit `f6283d8`). The only hard-gate blocker in that record
was the software/release-quality row; the software/CI gate has since been
corrected and verified green on PR #15 head `dc715a16`.

## Matrix result

The 14-row matrix contains 13 hard rows and one descriptive limitations row.
It recorded 13 PASS and 1 DESCRIPTIVE result; `matrix_green=true`. Rows A--M
passed or remained descriptive as frozen: exact clean-process
identity/release loading, closed predictive/calibration/robustness/OOD/Scout/
planning/end-to-end evidence, fail-closed/no-v4-fallback behavior, and output
governance all matched the M11.2 certificate. Every authority/safety counter
was zero, including `invariant_failures=0`.

Row N (software / release quality) passed, backed by the verified GitHub
Actions CI on PR #15 head `dc715a16`:

- HydroSwarm CI: `python-quality` (ubuntu-latest, 3.12) PASS; `python-quality`
  (windows-latest, 3.12) PASS (pytest + simulator smoke); `frontend-quality`
  PASS; `frontend-e2e` PASS.
- HydroSwarm Docker Build & Runtime Gate (PR): `docker-verify` linux/amd64 and
  linux/arm64 PASS (hardened startup, frozen hashes, EPANET smoke, real live
  workflow, persistence, offline/no-external-network check).
- HydroSwarm Native Cross-Platform Verification: Windows x86_64, Linux
  x86_64, Linux arm64, macOS x86_64, macOS arm64 all PASS.

The full per-run evidence is recorded in `m11-5-software-gates.json`
(`all_required_pass=true`, 11 SUCCESS check runs with their Actions URLs). The
earlier local full-pytest failure
(`test_m11_2_artifacts_freeze_without_authorizing_locked_evaluation`) was
fixed by making the M11.2 freeze check future-milestone safe, and the earlier
local Docker build could not run only because no Docker daemon socket was
present in that environment; the Docker PR gate now passes on both
architectures.

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

M11.6 prerequisites are satisfied: the M11.5-green-matrix prerequisite is
true. M11.6 is nevertheless **not** authorized and must not be run without
explicit human authorization. The stop boundary is: **M11.5 COMPLETE. M11.6
PRECONDITIONS SATISFIED. LOCKED EVALUATION IS NOT AUTHORIZED. STOPPING FOR
EXPLICIT HUMAN AUTHORIZATION.**
