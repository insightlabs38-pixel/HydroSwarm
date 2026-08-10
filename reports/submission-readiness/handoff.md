# Submission-readiness implementation — handoff report

Authoritative spec: `/workspace/submission.txt` ("HydroSwarm Hackathon-Winning
Submission Readiness Implementation Plan"). Background context (already-completed
prior passes, not this plan's own scope): `overnight-plan.txt`, `core-issues.txt`,
`core-issues2.txt`, `core-issues3.txt`, `core-issues5.txt`.

Per operating instructions for this session: **stop after SUB-0/SUB-1**. This report
covers those two phases only; SUB-2 onward have not been started.

## Branch / baseline

- Branch: `feature/submission-readiness-v1`.
- Branched from: `feature/ui-mission-control-v1` @ `828393edcd20cf307cd2569c7f7a7e539c171e82`
  ("docs(handoff): record UI-11.1 and pre-merge polish pass completion"), **not** from
  `origin/main`.

### Baseline discrepancy — recorded per SUB-0's "record any pre-existing gap" requirement

submission.txt §"Recommended branch / delivery strategy" states the baseline is
"main after merge of the mission-control UI PR". At session start, `origin/main` was
**40 commits behind** `feature/ui-mission-control-v1`
(`git merge-base --is-ancestor origin/main feature/ui-mission-control-v1` → true;
`git rev-list --left-right --count origin/main...feature/ui-mission-control-v1` → `0	40`).
The mission-control UI PR has not actually been merged to `main` yet. Since the entire
submission-readiness plan assumes the mission-control shell, `DEMO_FALLBACK` semantics,
and the frozen-`hydrocore-v4`-by-default runtime already exist (all present only on
`feature/ui-mission-control-v1`, confirmed by inspecting its commit log and current
source), branching from stale `main` would silently discard that work and contradict
the plan's own premise. I branched from `feature/ui-mission-control-v1` instead and am
recording this explicitly rather than guessing silently.

**Decision for a human to confirm:** either (a) merge `feature/ui-mission-control-v1`
into `main` via PR before or alongside this branch's own PR, or (b) treat
`feature/ui-mission-control-v1` as the de facto integration branch for this phase of
work. This session did not merge anything to `main` (per the "do not work on or push to
main" restriction) and took no action beyond recording the discrepancy.

## Environment (SUB-0 baseline record)

- Commit at branch point: `828393e`
- OS: Linux 6.17.0-1022-gcp, `aarch64` (ARM64) — GCP VM
- CPU: 16 vCPU, RAM: 62.7 GiB, free disk: 190 GB / 242 GB
- Python: 3.12.13 (repo `.venv`, pre-existing)
- PyTorch: 2.13.0+cu130 (CPU-only usage in this repo; the CUDA build tag is just what
  resolved locally — the Dockerfile explicitly forces the CPU wheel, unaffected)
- WNTR: 1.5.0
- Node: v20.19.2, npm 9.2.0 (repo `frontend/node_modules` pre-existing)
- ruff: 0.16.1, pyright: 1.1.411
- No `docker` binary available in this sandbox (see "Known limitations" below).

## Baseline gate results (before any SUB-1 change; `reports/submission-readiness/baseline-*.log`)

| Gate | Result |
|---|---|
| `pytest -q` (881 tests) | **881 passed**, 0 failed, 599s wall time |
| `ruff check src tests scripts` | **All checks passed** |
| `pyright` | **0 errors, 0 warnings** |
| `npm run lint` (frontend) | pass |
| `npm run typecheck` (frontend) | pass |
| `npm run format:check` (frontend) | pass |
| `npx vitest run` (frontend, 107 tests / 18 files) | **107 passed** |
| `npm run build` (frontend) | pass |

No pre-existing failures. Clean starting point confirmed before any source file was
touched (matches SUB-0's "no source file has been changed before baseline recording").

## Freeze-invariant verification (SUB-0 + ongoing through SUB-1)

- `reports/results/v4/architecture-freeze.json`: `status: "FROZEN"`,
  `locked_test_opened: false`, `locked_evaluation_status: "NOT PERFORMED -- awaiting
  separate explicit authorization"`.
- No `final-selection.json` anywhere in the repository.
- `git status --porcelain -- models/ data/learning-v1 reports/results/v4` was clean at
  session start and remains clean now (SUB-1 touched no scientific asset — see below).
- No model weights, calibration, normalization, or signature-policy files were modified,
  regenerated, or re-hashed by this session's changes.

## Docs/file inventory snapshot

`reports/submission-readiness/baseline-docs-inventory.txt` — 25 entries. Notably absent
(expected; out of SUB-0/SUB-1 scope, left for SUB-8/SUB-9):
`docs/README.md`, `docs/FINAL_SYSTEM.md`, `docs/diagrams/`.

## SUB-1 — packaging safety (relocatable V4 bundle, container-safe)

### Root cause found

1. **Two independent, silently-divergent path computations.**
   `hydroswarm.api.app` computed the V4 release bundle directory as
   `Path(__file__).resolve().parents[3] / "models" / "hydrocore-v4-release"` (source-tree
   relative). `hydroswarm.cli.run_self_test` independently computed
   `Path.cwd() / "models" / "hydrocore-v4-release"`. These agree only for an
   editable/source checkout run from the repo root. For a **non-editable** `pip install .`
   (exactly what the Dockerfile does), the installed package lives under
   `site-packages/hydroswarm/...`, so `parents[3]` from there is *not* the repository
   root — the app would silently resolve to a nonexistent directory, fail closed to the
   classical-safe fallback, and report `ok: true` / a plausible-looking self-test with no
   obvious error. This is exactly the "appears healthy while actually degraded" failure
   class the plan calls out as unacceptable.
2. **The Dockerfile never copied the bundle into the image at all** — confirmed by
   reading the pre-change `Dockerfile`; no `COPY models/...` existed, and `.dockerignore`
   excluded the entire `models/` directory unconditionally (`models` with no negation),
   so even a naive `COPY models/ models/` would have copied nothing.
3. **`HYDROSWARM_DATA_DIR` was declared but never read.** The Dockerfile/
   `docker-compose.yml` set `HYDROSWARM_DATA_DIR=/data` (a writable named volume,
   `read_only: true` elsewhere in the container), but no application code referenced that
   variable — `hydroswarm.storage.database.default_database_path()` only checked
   `HYDROSWARM_DB_PATH` (unset) and otherwise defaulted under `Path.cwd()` (`/app`, the
   read-only root). As shipped, the container's first database write would have raised
   `OSError: Read-only file system`. Found during SUB-1 validation, in scope per the
   plan's "a concrete bug uncovered during validation" backend-change allowance.
   `V4PipelineFactory`'s classical-signature cache had the identical problem
   (`project_root / "data" / "generated" / "signatures"`).

### Changes made

- **New module `src/hydroswarm/runtime/paths.py`**: `resolve_v4_bundle_dir()` and
  `resolve_data_dir()`. Both follow the same 2–3 tier priority: explicit env var
  (`HYDROSWARM_V4_BUNDLE_DIR` / `HYDROSWARM_DATA_DIR`) → project-root-relative default →
  (bundle resolver only) cwd-relative development fallback. Anchored to `paths.py`'s own
  file location, not the caller's, so every caller that omits `project_root` resolves
  identically regardless of which module/cwd it's called from.
- `hydroswarm/runtime/__init__.py`: exports the two new functions.
- `hydroswarm/api/app.py`: `DEFAULT_V4_RELEASE_BUNDLE_DIR` now computed via
  `resolve_v4_bundle_dir(_PROJECT_ROOT)` instead of inline concatenation.
- `hydroswarm/cli.py` (`run_self_test`): now calls `V4PipelineFactory(resolve_v4_bundle_dir())`
  — the exact same resolver as `app.py`, not an independently-computed cwd-relative path.
  Also added `bundle_dir` to the `trained_assets` block of the machine-readable self-test
  output, for container debuggability.
- `hydroswarm/runtime/v4_defaults.py`: `V4PipelineFactory`'s signature-cache directory now
  goes through `resolve_data_dir(project_root)` instead of a raw
  `project_root / "data" / "generated" / "signatures"` concatenation.
- `hydroswarm/storage/database.py` (`default_database_path`): now honors
  `HYDROSWARM_DATA_DIR` (after `HYDROSWARM_DB_PATH`, which still wins if both are set)
  before falling back to the pre-existing `cwd`-relative default. This is what makes the
  Dockerfile's pre-existing `HYDROSWARM_DATA_DIR=/data` declaration actually do something.
- `Dockerfile`:
  - `ENV HYDROSWARM_V4_BUNDLE_DIR=/app/models/hydrocore-v4-release` added.
  - `COPY models/hydrocore-v4-release/ models/hydrocore-v4-release/` added (after the
    frontend/configs copy, before the `chown`/`USER` switch).
  - A build-time self-test gate: `RUN python -c "..."` that calls
    `hydroswarm.cli.run_self_test()` inside the built image and fails the `docker build`
    (non-zero exit) unless `trained_assets.ready is True` **and** `frontend_assets ==
    "built"`. This is stricter than a bare `hydroswarm self-test` invocation would be —
    that command always exits 0 (`"ok": True` is unconditional; it reports
    `trained_assets.ready: false` without failing), which would not have caught this
    exact class of bug. Placed after `USER hydroswarm` so it runs with the same
    filesystem permissions the production process will have.
- `.dockerignore`: changed the blanket `models` exclusion to
  `models/*` + `!models/hydrocore-v4-release` + `!models/hydrocore-v4-release/**`, so the
  release bundle is included in the build context while every other `models/` subtree
  (training checkpoints, `cycle-b2-candidates/controls`) stays excluded — keeps the image
  small and never bundles unpromoted/non-frozen weights.
- New tests: `tests/unit/test_runtime_paths.py` (10 cases for the resolver priority
  rules), `tests/unit/test_storage_database.py` (env-var priority for the DB path),
  `tests/unit/test_dockerfile_v4_bundle_packaging.py` (static Dockerfile/.dockerignore
  content assertions + confirms the committed bundle has every file
  `REQUIRED_BUNDLE_FILES` needs, so the `COPY` has something real to copy).

### Verified

- `models/hydrocore-v4-release/` confirmed **not** LFS-tracked (`git lfs ls-files` empty
  for it; only `cycle-b2-candidates/controls/**/*.safetensors` are LFS-filtered per
  `.gitattributes`) and **not** gitignored (`git check-ignore` confirms) — the Docker
  build context will contain real bytes, not LFS pointers, so no LFS-hydration step is
  needed before `docker build` for this specific COPY.
- `tests/integration/test_production_runtime_wiring.py` (all 7 tests, including the
  3 marked `real_simulation` that load the actual frozen checkpoint and drive a real
  incident through the production `app` object) — **all pass** after the resolver
  change, confirming `hydroswarm.api.app:app` and `hydroswarm.cli.run_self_test` still
  agree and both still resolve the real frozen `hydrocore-v4` identity
  (`model_sha256 = a501ad87...`).
- `tests/integration/test_v4_pipeline_factory.py`, `test_v4_release_bundle.py`: pass.
- New unit tests: pass (see file list above).
- `ruff check` + `pyright` on every touched file: clean (`sub1-ruff.log`, `sub1-pyright.log`).
- Full repository `pytest -q` re-run after all SUB-1 changes:
  **895 passed, 0 failed** (881 baseline + 14 new tests added this phase), 618.63s
  wall time — zero regressions. See `reports/submission-readiness/sub1-pytest.log`.

### Known limitations / not verified this session

- **No `docker` binary is available in this sandboxed environment.** The Dockerfile,
  `.dockerignore`, and build-time self-test gate were implemented per spec and
  statically tested (structural assertions in
  `test_dockerfile_v4_bundle_packaging.py`), but an actual `docker build` /
  `docker run` / `hydroswarm self-test` inside a real container was **not** executed.
  This must be verified manually or in CI (`.github/workflows`, currently has no Docker
  step at all — adding one is SUB-3's `release.yml` scope) before the Docker path can be
  marked "Tested" anywhere in submission docs. Do not claim Docker verification beyond
  what is stated here.
- `DefaultPipelineFactory` (the legacy, no-longer-default hydrocore-v3 path) was
  deliberately **not** touched — it has its own explicit "do not touch" convention
  documented in-repo from the pre-freeze pass, and it is out of scope for a runtime that
  no longer uses it by default.
- SUB-2 through SUB-12 have not been started.

## Files changed this session

```
 .dockerignore                                       | modified
 Dockerfile                                           | modified
 src/hydroswarm/api/app.py                            | modified
 src/hydroswarm/cli.py                                | modified
 src/hydroswarm/runtime/__init__.py                   | modified
 src/hydroswarm/runtime/paths.py                      | new
 src/hydroswarm/runtime/v4_defaults.py                | modified
 src/hydroswarm/storage/database.py                   | modified
 tests/unit/test_runtime_paths.py                     | new
 tests/unit/test_storage_database.py                  | new
 tests/unit/test_dockerfile_v4_bundle_packaging.py    | new
 reports/submission-readiness/                        | new (this report + gate logs)
```

## Exact continuation commands

```bash
cd /workspace/HydroSwarm
git checkout feature/submission-readiness-v1

# re-run full gates
source .venv/bin/activate
python -m pytest -q
python -m ruff check src tests scripts
python -m pyright
cd frontend && npm run lint && npm run typecheck && npm run format:check && npx vitest run && npm run build && cd ..

# next phase per submission.txt §75 / §81:
# SUB-2 -- native setup scripts (setup_hydroswarm_linux.sh / _macos.sh / _windows.ps1)
#          and platform launchers (start_hydroswarm_linux.sh / _macos.sh / _windows.ps1)
```

## Next phase (not started)

SUB-2 — native setup/launcher scripts per platform, as specified in submission.txt §15–17.
