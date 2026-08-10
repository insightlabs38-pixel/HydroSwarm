# Submission-readiness implementation — handoff report

Authoritative spec: `/workspace/submission.txt` ("HydroSwarm Hackathon-Winning
Submission Readiness Implementation Plan"). Background context (already-completed
prior passes, not this plan's own scope): `overnight-plan.txt`, `core-issues.txt`,
`core-issues2.txt`, `core-issues3.txt`, `core-issues5.txt`.

This report originally covered SUB-0/SUB-1 only (that session's operating instructions
were to stop there). It is now continued by a later autonomous session (2026-08-10)
that synced this branch onto merged `main` and completed SUB-2; see the dated sections
below. Earlier sections are left as originally written except where explicitly marked
RESOLVED/updated.

## Branch / baseline

- Branch: `feature/submission-readiness-v1`.
- Branched from: `feature/ui-mission-control-v1` @ `828393edcd20cf307cd2569c7f7a7e539c171e82`
  ("docs(handoff): record UI-11.1 and pre-merge polish pass completion"), **not** from
  `origin/main`.

### Baseline discrepancy — RESOLVED 2026-08-10

submission.txt §"Recommended branch / delivery strategy" states the baseline is
"main after merge of the mission-control UI PR". At the *original* SUB-0/SUB-1 session
start, `origin/main` was 40 commits behind `feature/ui-mission-control-v1` and the PR
had not been merged; branching from `feature/ui-mission-control-v1` directly (as
recorded further down, still historically accurate) was the correct call at that time.

**Update (this session, 2026-08-10):** `git fetch origin` picked up
`561a442 Merge pull request #2 from insightlabs38-pixel/feature/ui-mission-control-v1`
— the mission-control UI PR is now merged into `origin/main`. `origin/main`'s tip
(`c1fc9cc`, the prior research/training-track head) was already an ancestor of
`feature/ui-mission-control-v1` at merge time, so PR #2 was a fast-forward-shaped merge
that introduced no new tree conflicts. This branch (`feature/submission-readiness-v1`)
was synced onto that new `origin/main` via `git merge --no-ff origin/main` (not rebase,
to avoid force-pushing the two commits — `ce992e4`, `22fadad` — already pushed to
`origin/feature/submission-readiness-v1`). The merge was content-empty (this branch's
tree was already a superset of `origin/main`'s, since it descended from the same UI-branch
tip origin/main just merged) and is recorded as commit `33bb332`. Gates re-verified green
after the sync (see below). No human decision is outstanding on this point anymore.

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

## 2026-08-10 session — branch sync + SUB-2

### Sync onto merged main

- `git fetch origin` revealed `origin/main` had advanced to `561a442` (PR #2, merging
  `feature/ui-mission-control-v1`). See "Baseline discrepancy — RESOLVED" above.
- `git merge --no-ff origin/main` into `feature/submission-readiness-v1` → commit
  `33bb332`, zero content diff, pushed.
- Post-sync gate re-check: `ruff check src tests scripts` clean, `pyright` 0/0/0. Full
  `pytest -q` re-run (`reports/submission-readiness/sub2-post-merge-pytest.log`,
  pre-SUB-2 tree): **895 passed, 0 failed, 546.75s** — identical to the SUB-1 baseline
  count, confirming the merge was content-empty as expected and introduced zero
  regressions. A second full run including the SUB-2 test additions is captured in
  `reports/submission-readiness/sub2-full-pytest.log`.

### SUB-2 — native setup/launcher scripts (commit `388384d`)

Implements submission.txt §15–17.

- `scripts/setup_common.py`: shared Python helper (`check-python`, `verify-bundle`,
  `frontend-status`, `self-test` subcommands) used by all three platform scripts so
  the verification logic can't drift between them the way the pre-SUB-1 bundle-path
  resolution did.
- `setup_hydroswarm_linux.sh` / `_macos.sh` / `_windows.ps1`: create `.venv`, install
  CPU-only deps into it exclusively (never global, never `sudo`/apt/yum/pacman/brew
  invocation — only printed instructions), verify the frozen V4 bundle via the SUB-1
  resolver, build the frontend only if no prebuilt `frontend/dist` exists (detects
  Node 22+), and gate "setup complete" on a green self-test. macOS additionally
  verifies the WNTR/EPANET import and does not assume Rosetta. Windows uses
  PowerShell (not batch), does not run the full real-simulator suite, and explains
  the Docker/WSL2 production-latency path per §15.4.
- `start_hydroswarm_linux.sh` / `_macos.sh` / `_windows.ps1`: explicit `.venv`
  interpreter (never ambient `python`), fail closed with a clear message if `.venv`
  is missing, run a readiness check before binding, bind loopback by default, print
  the URL.
- `start_hydroswarm.sh` / `.bat`: converted to thin OS-detecting compatibility
  wrappers delegating to the new platform launchers (previously invoked ambient
  `python` directly — exactly the gap submission.txt §3.3 called out).
- `cli.py`: added `hydroswarm self-test --human` rendering the §17 readiness
  checklist (`render_self_test_report`). Purely additive — default `hydroswarm
  self-test` JSON output (relied on by CI/tests/Dockerfile build gate) is unchanged.
- Tests: `tests/unit/test_native_setup_scripts.py` (46 cases, structural/static —
  venv-only install, fail-closed launchers, no system-package-manager mutation,
  readiness gating present) + new `test_cli.py` cases for `--human`.

**Live-verified in this sandbox** (not just statically): ran
`./setup_hydroswarm_linux.sh` end-to-end against the real repo state — bundle
verified, self-test printed `READY`. Then ran `./start_hydroswarm_linux.sh` in the
background and confirmed `curl http://127.0.0.1:8765/api/health` returned
`{"status":"ok","offline":true,...}` before stopping it. macOS and Windows scripts
were **not** executable in this Linux sandbox and were only statically/structurally
verified — flag this if a judge/user reports a macOS- or Windows-specific setup
failure; that path has real execution risk this session could not retire.

### Docker availability update

Unlike the SUB-0/SUB-1 session, **`docker` (29.7.2) is now available** in this
sandbox. SUB-3 (multiarch Docker/release packaging) is next and will include a real
`docker build`/`docker run` verification this time, not just the static Dockerfile
assertions SUB-1 was limited to.

### SUB-3 — multiarch Docker/release packaging (commit `c1664b6`) — NOT marked complete

Implemented per submission.txt §18–22: `.github/workflows/release.yml` (tag-triggered
only; verify-frozen-bundle → buildx amd64+arm64 GHCR push → per-platform
pull-and-self-test matrix → RELEASE_MANIFEST.json + runtime zip attached to the GitHub
Release), `docker-compose.release.yml` (pinned GHCR tag, no local build), and two
generator scripts (`scripts/build_release_manifest.py`,
`scripts/build_release_bundle.py`) — both real-tested locally against the actual repo
state, hashes sourced from `models/hydrocore-v4-release/runtime_manifest.json` (never
hand-typed).

**This session could not execute an actual `docker build`/`docker run` at all**,
neither amd64 nor arm64 (this host is aarch64, so arm64 would have been native).
Root-caused, not just observed: this sandbox's own container has `CAP_SYS_ADMIN`
stripped and blocks `unshare` even for an *unprivileged* user namespace
(`unshare --user --map-root-user --mount` → `Operation not permitted`, despite
`/proc/sys/kernel/unprivileged_userns_clone` = `1`) — a seccomp/LSM-level restriction
on this sandbox, not a fixable Docker/kernel config gap. Full diagnostic trail:
`reports/submission-readiness/sub3-docker-sandbox-limitation.md`.

**Per this session's explicit gate ("do not mark Docker/release packaging complete
until a real amd64 build/run and an arm64 build/smoke have executed successfully"),
SUB-3 is NOT being marked done.** What remains, for whoever picks this up next:

1. Push a `v*` tag (or run `workflow_dispatch`) so `release.yml` runs on a real GitHub
   Actions runner (has the privileges this sandbox withholds), **or**
2. On any machine with normal (non-sandboxed) Docker privileges:
   ```bash
   docker build -t hydroswarm:smoke .
   docker run --rm hydroswarm:smoke hydroswarm self-test --human
   docker buildx build --platform linux/amd64,linux/arm64 -t hydroswarm:multiarch-smoke .
   ```
3. Only after (1) or (2) produces a real green result should Docker be described as
   "Tested"/"Recommended judge path" anywhere in README/docs (SUB-7 onward) — do not
   let later phases quietly promote Docker to "verified" on the strength of this
   session's static-only verification.

### SUB-5/SUB-6 — progressive reference UI + experience-state separation (commit `952815d`)

Backend (`e542739`) + frontend (`952815d`). New `REFERENCE` runtime mode,
`reference/` module (types/mapMilestone/useReferenceIncident), first-launch
gateway, `?experience=reference|live|fallback` routing, ModeBanner controls.
See that commit message for full detail. Frontend gates green (lint,
typecheck, format:check, 125 vitest, `npm run build`). Live end-to-end
smoke test: real server + curl of `/api/health` and `/api/reference-demo`.

**Not done / left for a later pass:** no visual/screenshot verification was
possible in this sandbox (no browser). Playwright visual-regression
baselines (`frontend/tests/e2e/visual-regression.spec.ts`) were **not**
updated or re-run for the new gateway/REFERENCE banner -- do this before
claiming the reference experience is visually correct, not just
type-correct and unit-tested. `evidenceHistory`/`audit`/`benchmarks` are
left empty (`[]`) in the REFERENCE IncidentView mapping -- real data exists
in the golden result (explanations, event ledger) but wasn't threaded
through the SUB-4 artifact schema; empty is honest (not fabricated) but is
a known richness gap, not a bug, if a future pass wants a fuller experience.

## Exact continuation commands

```bash
cd /workspace/HydroSwarm
git checkout feature/submission-readiness-v1
git pull origin feature/submission-readiness-v1

# re-run full gates
source .venv/bin/activate
python -m pytest -q
python -m ruff check src tests scripts
python -m pyright
cd frontend && npm run lint && npm run typecheck && npm run format:check && npx vitest run && npm run build && cd ..

# check the post-sync full-suite background run this session started:
cat reports/submission-readiness/sub2-post-merge-pytest.log

# SUB-3 status: implemented but Docker execution UNVERIFIED -- this sandbox
# cannot run any container at all (CAP_SYS_ADMIN stripped; see
# reports/submission-readiness/sub3-docker-sandbox-limitation.md). Verify on
# a real GitHub Actions runner or unsandboxed machine before treating SUB-3
# as done -- see the SUB-3 section above for exact commands.

# next phase per submission.txt task list SUB-7:
# SUB-7 -- README judge-first restructure
```

## Next phase (not started as of end of 2026-08-10 session)

SUB-7 — judge-first README rebuild (submission.txt §3.1 / task list SUB-7):
one-line value prop, product screenshot, problem, operator workflow, why
different, strongest measured results, try-it, final-system architecture,
technical depth links, limitations, research/evaluation details pushed
deeper. Five-minute judge test must pass. Then SUB-8 (docs IA), SUB-9
(technical docs/diagrams), SUB-10 (Devpost), SUB-11 (release tests),
SUB-12 (release candidate).
