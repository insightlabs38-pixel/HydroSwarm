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

`python -m build --wheel --no-isolation` is a Phase 9 final-validation gate per the plan and
was not run during Phase 0 baseline recording; it will run at end-of-run alongside the rest
of Phase 9.

## Update after Bundle C (commit `2067571`)

| Command | Result | Notes |
|---|---|---|
| `python -m pytest -q` | **270 passed** | up from 98 at baseline (1 pre-existing failure fixed, 172 new tests added across Bundle A/B) |
| `python -m ruff check src tests scripts` | **Pass** | |
| `python -m pyright` | **Pass** | |
| `npm run lint` (frontend) | **Pass** | |
| `npm run test -- --run` (frontend, vitest) | **Pass** | 24 tests, up from 4 (api.ts mode logic, ModelGovernanceTable, identifier-independence, failure injection) |
| `npm run build` (frontend) | **Pass** | |
| `npx playwright test` (frontend, e2e, real Chromium) | **Pass** | 10 tests, up from 1 -- run and verified in this session (browsers were pre-cached at `~/.cache/ms-playwright`); 2 real screenshot baselines committed under `frontend/tests/e2e/visual-regression.spec.ts-snapshots/` |
| HydroCore-S checkpoint load | **Pass** | re-verified unchanged: same hash, feature-schema, calibration |

## Update after Bundle D (commit `d0a0b42`)

| Command | Result | Notes |
|---|---|---|
| `python -m pytest -q` | **349 passed** | up from 270 after Bundle C (79 new tests across Tasks 4.0-4.6: 11 prior_mode + 17 incident_pooling + 15 message_direction + 12 event_control_heads + 12 auxiliary_heads + 11 consequence_prescreening + minor updates to pre-existing targets_v2/losses tests for the new governed categories/tasks) |
| `python -m ruff check src tests scripts` | **Pass** | |
| `python -m pyright` | **Pass** | |
| `npm run lint` / `npm run test -- --run` / `npm run build` / `npx playwright test` (frontend) | **unchanged** | Bundle D touched no frontend code; not re-run this update, no reason to expect a change |
| HydroCore-S checkpoint load | **Pass** | re-verified after *every* Task 4.x commit (7a5492d, 0bddd71, 6c2d712, f4974b8, 6474a69, d0a0b42), not just once at the end -- each new architecture flag's default value leaves `DefaultPipelineFactory().trained_assets_ready` True with the same hash/schema/calibration match |

## Update after Bundle E (commit `0606586`)

| Command | Result | Notes |
|---|---|---|
| `python -m pytest -q` | **355 passed** | up from 349 after Bundle D (6 new tests: 1 development_holdout split test, 1 label_audit multi-topology regression test, 1 Trainer custom-collate_fn test, 2 target-mask-companion tests, 1 evidence_sufficiency shape/loss test) |
| `python -m ruff check src tests scripts` | **Pass** | |
| `python -m pyright` | **Pass** | |
| `npm run lint` / `npm run test -- --run` / `npm run build` / `npx playwright test` (frontend) | **unchanged** | Bundle E touched no frontend code |
| HydroCore-S checkpoint load | **Pass** | re-verified after every Bundle E commit; the `evidence_sufficiency` output-shape fix changes a forward-pass convention only, no parameter shapes |
| Real end-to-end training (new this bundle) | **Pass** | 6 real `Trainer.fit()` runs (E0/E3/E4/E9-none/E9-feature_only/E9-logit_only) against real Cycle A data: finite loss throughout, every present supervised head received nonzero gradient, every run resumed correctly from its own checkpoint, every exported checkpoint reloaded with all-finite weights and passed `verify_architecture_compatibility` against its own recorded config. See `reports/results/v3/architecture-smoke-jobs.json`. |
