# Baseline test and lint gate results

Recorded at commit `5697f912667fa236ece784a98f141c8162ff6bf8` (main, before branch creation),
verified again immediately after creating `agent/gcp-multitopology-v3` (identical tree, zero diff).
No source file was changed before these results were recorded.

| Command | Result | Notes |
|---|---|---|
| `python -m pytest -q` (PYTHONPATH=src) | **1 failed, 97 passed** | Pre-existing failure: `tests/frozen/test_frozen_artifacts.py::test_frozen_manifest_matches_checked_in_inputs` — asserts `data/frozen/golden_scenario.json` is 490 bytes per `data/frozen/manifest.json`, actual size is 469 bytes. This predates this run (present on `main` at the starting commit, before any edits). Recorded as a pre-existing blocker, not fixed yet since Phase 0 forbids source edits before baseline recording; a fix will be proposed as an independent, separately-tested task once Phase 0 closes, without touching locked test data. |
| `python -m ruff check src tests scripts` | **Pass** | 0 issues |
| `python -m pyright` | **Pass** | 0 errors, 0 warnings, 0 informations |
| `npm ci` (frontend) | **Pass** | node_modules were absent at session start; installed 306 packages, 0 vulnerabilities. `EBADENGINE` warning for `@mapbox/jsonlint-lines-primitives` (wants node>=22, have 20.19.2) — non-fatal. |
| `npm run lint` (frontend) | **Pass** | eslint, `--max-warnings 0` |
| `npm run test -- --run` (frontend) | **Pass** | 2 test files, 4 tests (store.test.ts, App.test.tsx incl. axe accessibility check) |
| `npm run build` (frontend) | **Pass** | `tsc -b && vite build`, 687 modules, no errors |

## Checkpoint safety verification

`DefaultPipelineFactory.trained_assets_ready` was exercised directly against the promoted
runtime checkpoint (no code changes involved):

```
trained_assets_ready: True
fallback_reason: None
param_count: 4040645
```

This confirms the promoted `models/hydrocore-s-learning-v1.safetensors` still loads with a
matching SHA-256, a matching feature-schema fingerprint, and a validated calibration
artifact, and its parameter count (4,040,645) matches the metadata's `parameter_count` and
the plan's documented ~4.04M HydroCore-S size. Nothing in Phase 0 modified this checkpoint,
its metadata, or the calibration artifact.

## Gate not yet run

`npm run test:e2e` (Playwright) and `python -m build --wheel --no-isolation` are Phase 9
final-validation gates per the plan and were not run during Phase 0 baseline recording;
they will be run again at end-of-run alongside the rest of Phase 9.
