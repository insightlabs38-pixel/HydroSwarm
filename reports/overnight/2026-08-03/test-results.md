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

## Update after Task 3.2 + Cycle B corpus landing (commit `bd944d5`)

| Command | Result | Notes |
|---|---|---|
| `python -m pytest -q` | **363 passed** (occasionally 362 passed / 1 failed) | 4 new backend contract tests for `GET /incidents/{id}/view` on top of Bundle E's 355+4=359. The intermittent failure, `tests/scientific/test_scout_labels.py::test_information_gain_is_nonnegative_within_tolerance`, is a pre-existing test-infra issue: it seeds via Python's built-in `hash()` on a string, which is salted per-interpreter-invocation unless `PYTHONHASHSEED` is fixed (it isn't), so the exact scenario/seed combination it exercises differs every `pytest` process regardless of any code change. Confirmed unrelated to this session's changes: passes reliably in isolation (`pytest tests/scientific/test_scout_labels.py` and `pytest tests/scientific/`), and passed on 2 of 3 consecutive full-suite runs with identical code. Not modified -- out of scope for Task 3.2, and altering a physics-adjacent test's tolerance without being asked risks exactly the kind of safety-boundary weakening this run is instructed to avoid. |
| `ruff check` (touched files: `api/app.py`, `api/state.py`, `inference/pipeline.py`, `inference/__init__.py`, new test file) | **Pass** | 0 issues |
| `pyright` (touched files) | **Pass** | 0 errors, 0 warnings, 0 informations |
| `npx tsc --noEmit` | **Pass** | narrower project scope than `tsc -b` |
| `npm run build` (`tsc -b && vite build`) | **Pass** (after 1 fix) | `tsc -b`'s wider scope (includes `tests/`) caught a stale `IncidentView` object literal in `IdentifierIndependence.test.tsx` missing the new `provenance`/`selectedPlanId`/`recommendedPlanId`/`counterfactuals` fields -- `tsc --noEmit` alone did not catch this. Fixed in `bd944d5`. |
| `npm run lint` | **Pass** | |
| `npm run format:check` (prettier) | 11 pre-existing failures, 0 new | Confirmed via `git stash`: the same 11 files (none touched this session) were already unformatted before any Task 3.2 work began. `src/api.ts` was the only file this session's edits newly affected, and `prettier --write` was applied to it specifically (not the other 10 pre-existing files, which are out of scope). |
| `npm run test -- --run` (vitest) | **Pass** | 25 tests, up from 24 (api.test.ts's 2 stale stub-throw tests replaced with 3 tests exercising the real `/view` contract: full mapping, fallback-still-LIVE, malformed-response-falls-back) |
| `npx playwright test` (e2e, real Chromium) | **Pass**, 10/10 | one 1920x1080 visual-regression test needed a retry (~0.01% pixel diff at chart-marker anti-aliasing pixels); confirmed flaky and unrelated to this session's changes by rerunning it alone immediately afterward (passed) |
| HydroCore-S checkpoint load | **Pass** | unaffected -- this bundle touched API/frontend code and the data corpus, not the model or checkpoint |

## Update after Counterfactuals.tsx fix + Stage 2/3 training (commit `ddf6fd9`)

| Command | Result | Notes |
|---|---|---|
| `npm run test -- --run` (vitest) | **Pass** | 30 tests, up from 25 (+5 `Counterfactuals.test.tsx`: real names render not "PLAN A"/"PLAN B", no crash with 0 or 1 plans, recommended styling follows `plan.status` not position, plan order can change without breaking which branch is marked recommended) |
| `npm run lint` / `npx tsc --noEmit` / `npm run build` | **Pass** | |
| `npx playwright test --update-snapshots` | **Pass**, 10/10 | 2 committed screenshot baselines regenerated (now show all 3 demo plans instead of silently dropping the 3rd) and visually reviewed before committing |
| `ruff check` / `pyright` (`scripts/run_stage3_finalist_training.py`) | **Pass** | 0 issues, 0 errors |
| Stage 2 architecture screening | **Pass**, 9/9 | see `training-jobs.md` for full detail |
| Stage 3 script smoke verification | **Pass** | 1-epoch, 64/32/32/32-example scratch run (not committed) exercised the full training -> calibration-fit -> validation/dev-holdout/OOD-eval path without error before the real multi-hour job was launched |
