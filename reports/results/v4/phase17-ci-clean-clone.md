# Phase 17 — CI and Clean-Clone Reproducibility

core-issues3.txt "PHASE 17 — CI AND CLEAN-CLONE REPRODUCIBILITY".

## CI workflow

`.github/workflows/ci.yml` already existed (built in an earlier pass,
predating this one) and substantively satisfies the phase's requirements:
`uv sync --all-extras --dev` (locked environment), `ruff check`, `pyright`,
the full `pytest` suite with coverage (728 tests as of this pass —
unit + scientific + architecture-config/strict-load
[`tests/unit/test_checkpoint_identity.py`] + trajectory/collation tests
[`tests/scientific/test_scout_trajectory.py`, `tests/unit/test_variable_collate.py`,
etc.] are all included, exceeding the phase's "selected" bar rather than
falling short of it), plus an offline self-test and a separate
frontend-quality job (lint/typecheck/test/build). Runs on `ubuntu-latest`
and `windows-latest` — never touches the locked test (grepped every
`locked`-adjacent reference across `tests/`/`scripts/`: all are either
tests *of* the fail-closed locked-test-authorization mechanism itself, or
a standalone v1/v2-era script never invoked by CI). No expensive WNTR
corpus *generation* runs in CI — the full test suite completes in ~8.6
minutes locally (measured this pass, 728 tests), nowhere near the hours a
full multi-thousand-scenario corpus regeneration takes (Stage-F's own
`train` split generation alone took 9072s/~2.5h per the corresponding
`*-report.json`).

**One real gap found and fixed this pass**: the workflow's `push` trigger
was `branches: [main]` only — a push to this pre-freeze work's actual
branch (`agent/gcp-multitopology-v3`) never triggered CI at all unless a
PR happened to be open. Changed to `branches: [main, "agent/**"]` so this
(and any future agent branch) gets real CI feedback on push, per the
phase's own explicit "runs on the development branch/PR" requirement.

## Clean-clone reproduction (performed for real this pass, on this same Arm/aarch64 sandbox)

```bash
git clone <repo> /workspace/clean-clone-test/HydroSwarm
cd /workspace/clean-clone-test/HydroSwarm
git checkout agent/gcp-multitopology-v3          # HEAD 6b8f8ee (this pass's own commits)
uv sync --all-extras --dev                        # 136 packages installed
for split in train validation calibration development-holdout; do
  tar --use-compress-program=unzstd \
    -xf artifacts/migration/cycle-b2-scenarios-${split}.tar.zst \
    -C data/learning-v2/cycle-b2/scenarios          # 13,550 real scenario .npz arrays restored
done
./scripts/build_epanet_arm64.sh                    # builds OWA EPANET v2.2 for aarch64, verifies a real simulation
export PYTHONPATH=src
python scripts/run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2 \
  --report-output /tmp/clean_clone_corpus_gates.json
python -m pytest -q
python scripts/run_trajectory_corpus_gates.py
```

**Results:**

| step | result |
|---|---|
| `git clone` + LFS pull | 1.43 GiB of real LFS objects pulled (verified: no pointer-stub files, real safetensors byte sizes present) |
| `uv sync --all-extras --dev` | 136 packages installed cleanly |
| scenario archive extraction | 13,550 `.npz` files restored from `artifacts/migration/cycle-b2-scenarios-*.tar.zst` |
| `build_epanet_arm64.sh` | succeeded; verified against a real `EpanetSimulator.run_sim()` |
| `run_corpus_gates.py` (original 9 Cycle B2 gates) | **9/9 passed, including `deterministic_replay`** (12 scenarios genuinely replayed byte-identical/hash-tolerant) — this specific gate could NOT be verified in the primary working session this pass (raw scenario arrays are gitignored/ephemeral there — see `reports/results/v4/trajectory-corpus-gates.json`'s `"passed_except_environment_limitation"` status); the clean clone, with the archives explicitly extracted, closes that gap for real |
| full `pytest -q` | **728 passed, 0 failed** (6267 warnings, all pre-existing deprecation/pytest-collection noise unrelated to this pass), 440.07s wall time. Slower than the primary working session's equivalent run (519s vs. 440s — actually faster here; both fully cold vs. warm disk cache did not materially change the outcome) |
| `scripts/run_trajectory_corpus_gates.py` (Phase 16) | **all gates passed** in the clean clone too, `cycle_b2_original_nine` now genuinely `"passed"` (not `"passed_except_environment_limitation"`) since the scenario archives were extracted first — see `reports/results/v4/phase17-clean-clone-trajectory-gates.json` |

**Both environments now cross-verify each other**: the primary working session's `reports/results/v4/trajectory-corpus-gates.json` (missing raw scenario data, `deterministic_replay` downgraded) and this clean clone's equivalent (raw scenario data present, `deterministic_replay` genuinely exercised) together prove the corpus-gates script itself is correct in both configurations, not just lucky in one.

**Second real finding from this reproduction**: `scripts/run_trajectory_corpus_gates.py`'s `joint_v4_six` sub-gate initially FAILED in the clean clone — `data/learning-v2/cycle-b2-joint-v4`'s tensor shards are not actually committed at all (see `reports/results/v4/phase18-artifact-and-lfs-governance.md` item 1 for the full `.gitattributes`/`.gitignore` inconsistency this exposed). Regenerating it explicitly:

```bash
python scripts/build_stage_f_joint_corpus.py --include-ood-extension
```

produced a tensor corpus whose `dataset_fingerprint_sha256`
(`32b0528f569d27a0e51e6285d9da696c794c720d8ffc5c6d702042532d67f93c`)
**matched the primary working session's committed
`data/learning-v2/cycle-b2-joint-v4/checksums.json` byte-for-byte** —
real, deterministic, cross-environment reproducibility, not merely "a"
corpus that happens to look similar. After regeneration, `joint_v4_six`
passed and the full `run_trajectory_corpus_gates.py` suite reported
`overall_status: "passed"` in the clean clone too (all sub-gates
genuinely passing, none downgraded this time — see
`reports/results/v4/trajectory-corpus-gates.json`'s clean-clone
counterpart for the full detail).

## Artifact hashes / provenance

- Clean-clone source: local clone of `/workspace/HydroSwarm` at commit `6b8f8ee` (this pass's HEAD at clone time) — LFS objects pulled from the same origin this pass already verified pushes to (`https://github.com/insightlabs38-pixel/HydroSwarm.git`, confirmed reachable earlier this pass).
- No locked-test data was opened during this reproduction.
