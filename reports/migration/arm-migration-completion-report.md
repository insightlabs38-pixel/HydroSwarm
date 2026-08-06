# x86 -> Arm VM migration: completion report

Branch: `agent/gcp-multitopology-v3`. Source Git SHA: `6fea9f220f1a63bbf10c85fa241e7716d52477df`
(end of `reports/results/v3/phase3-completion-report.md`'s Phase 3 work). Scout/
Strategist implementation was not started, per instruction. **The locked final test
was not opened, inspected, copied, or archived at any point during this migration.**

## Commits pushed (3)

| Commit | Type | Summary |
|---|---|---|
| `3b2c1a7` | `build(migration)` | Standard-Git config/metadata: `.gitattributes`, `.gitignore`, `docs/ARM_MIGRATION.md`, `scripts/verify_migration_artifacts.py`, all `models/cycle-b2-{candidates,controls}/*/*.json` sidecars, `artifacts/migration/cycle-b2-scenarios-manifest.json`, `reports/migration/{source-environment,arm-migration-inventory}.json`. No binary payload. |
| `a898d5d` | `data(migration)` | The full LFS payload: 124 objects (Cycle B2 tensor corpus, 4 finalist checkpoints, 2 calibration artifacts, 2 control checkpoints, 4 scenario archives). |
| `6b9bf22` | `docs(migration)` | One-line follow-up: documents that `uv sync --frozen --extra dev` (not plain `--frozen`) is needed for the full corpus-gate suite, found during the clean-clone test below. |

All three pushed to `origin/agent/gcp-multitopology-v3`. Final working tree: clean
(`git status --short` empty; `git lfs status` shows nothing pending).

## Byte and object totals

From `reports/migration/arm-migration-inventory.json`'s `summary` (computed from the
actually-staged files, not an estimate):

| | Entries | Bytes |
|---|---|---|
| Standard Git | 73 | 42,229,190 (~40 MiB) |
| Git LFS | 124 | 426,950,843 (~407 MiB) |
| **Total** | **197** | **469,180,033 (~447 MiB)** |

`git lfs ls-files` on both the source repo and the clean clone independently confirm
124 LFS objects.

## Included artifacts

- **Cycle B2 tensor corpus** (110 shards: 55 raw `tensors/` + 55 `tensors-normalized/`,
  every split including both OOD categories) -- verified against each split's
  `manifest.json` sha256 before every commit and again in the clean clone.
- **Four finalist checkpoints** (`models/cycle-b2-candidates/{E1,E0}-seed{20260810,
  20260811}/`): `model.safetensors`, `optimizer_state.pt`, `trainer_state.json`,
  `config.json`, `metadata.json`, `summary.json`, each finalist's Stage 3
  `fixed_weight_fusion` calibration (`stage3-fixed-weight-calibration.json`), and a
  synthesized `architecture_config.json` (variant/overrides/schema hashes/topology
  hashes/manifest hashes/normalization hashes -- these runs predate
  `promote_checkpoint.py`'s architecture-config-sidecar convention, so this file did
  not previously exist).
- **Two real dynamic-fusion calibration artifacts** (`E1-seed20260810`,
  `E0-seed20260811` -- the higher-val-top1 seed per finalist, matching Phase 3 item 18's
  own scope): `calibration-dynamic-fusion.json` + `.sha256`.
- **Two no-adapter control checkpoints** (`models/cycle-b2-controls/no-adapter-seed
  {20260810,20260811}/`): final `model-export.safetensors` (renamed `model.safetensors`
  for naming consistency with candidates -- verified this is the file matching the
  specified hash, not `checkpoints/checkpoint-0016/model.safetensors`, which differs),
  plus the same metadata/architecture-config treatment as candidates. No optimizer
  state or periodic checkpoints, per instruction.
- **Four deterministic scenario archives** (`artifacts/migration/cycle-b2-scenarios-
  {train,validation,calibration,development-holdout}.tar.zst`): raw per-scenario
  `.npz` + `.parquet` arrays, one archive per split, sorted/zeroed-metadata tar +
  zstd -19. `development-holdout` includes the `UNSEEN_TOPOLOGY`/`SEVERE_MISSINGNESS`
  OOD scenarios (they live physically inside that split's directory). Verified
  byte-for-byte identical to the source directory via a full extraction + `diff -rq`
  before committing. Combined archive size: ~5.6 MiB (raw content compresses roughly
  16-30x, well under any plausible LFS quota concern).
- **Governance docs**: `reports/migration/{arm-migration-inventory,source-environment}.json`,
  `docs/ARM_MIGRATION.md`, `scripts/verify_migration_artifacts.py`, plus four
  `reports/migration/_build_*.py` provenance scripts recording exactly how each
  artifact set was constructed from the source run directories.

## Intentionally omitted artifacts

Recorded with reasons in `arm-migration-inventory.json`'s `omitted` array:

- Stage 2 screening checkpoints (diagnostic only, superseded by Stage 3).
- Periodic (non-final) checkpoints under Stage 3 (`checkpoint-0004/0008/0012`) and all
  of Stage 4's periodic checkpoints/optimizer state -- only each finalist's final
  `checkpoint-0016` and each control's final `model-export.safetensors` are preserved.
- `best-model.safetensors`/`model-export.safetensors` duplicates under Stage 3 (the
  specified candidate hashes match `checkpoint-0016/model.safetensors` specifically,
  confirmed by hash comparison against all three candidate files before choosing).
- `experiments/cache/signatures/` (regenerable `SignatureArtifact` cache; already
  `.gitignore`d, untouched by this migration).
- `data/learning-v2/cycle-a` and `cycle-b` (old, preserved historical corpora) --
  out of scope; `.gitignore`'s narrow exceptions apply to `cycle-b2` only.
- **The locked final test** -- never opened, listed, hashed, read, or archived by any
  command run during this migration.
- No LFS-quota overflow occurred (total LFS payload ~407 MiB), so nothing was omitted
  for quota reasons.

## Every verified checksum

All verified via `scripts/verify_migration_artifacts.py` (semantic
`CalibrationArtifact.artifact_hash` for calibration, plain sha256 for everything else),
both before the first commit and again against the fresh clone:

| Artifact | Verified hash |
|---|---|
| E1-seed20260810 `model.safetensors` | `051cfd94dec4a7ec61e559a1268b66acaada2d6248bda8c976846f9064ef3a23` |
| E1-seed20260811 `model.safetensors` | `4ae71f3b31c3e7d4e10667126aad5343d64dad513aa48c626c4e3fa42a5dd63a` |
| E0-seed20260810 `model.safetensors` | `04ada898f994c8cd54e12d65a7997256d80e5d6fb4c96a003f56e3492ad43580` |
| E0-seed20260811 `model.safetensors` | `c8f6a5e62a09264f653eec90854ca4934581348e05aa3c86cefc65cb5eee65df` |
| E1-seed20260810 dynamic-fusion calibration | `639384e86ce3c6ad30fb73914b08b8aa302337d77feb3472481e00c6d6cf040d` |
| E0-seed20260811 dynamic-fusion calibration | `548009981c74a1d1c66c28936c1e66d65eca670638329c737101dad3d22a922f` |
| no-adapter-seed20260810 `model.safetensors` | `fe2bd18b6849d680beae3c4274481797d873f752388ca412ee0a9965b4bb0e3b` |
| no-adapter-seed20260811 `model.safetensors` | `f9fa30883cfe33c7ac1f272daaed62d9e263f743760fc7717e00e354703f5d2e` |
| node-normalization.json | `4dcf22a68839a8630e83b0e90f47ac3400b176b576e76d8bee5662221d238691` |
| edge-normalization.json | `3e715d707475d81eba90de6609246f51bb0eee8a94c58eab4958f4370fca514d` |
| All 110 tensor shards (raw + normalized) | verified against `manifest.json` per-shard sha256; see `arm-migration-inventory.json` for the full per-file list |
| 4 scenario archives | verified via extraction + `diff -rq` against source, plus recorded sha256 in `cycle-b2-scenarios-manifest.json` |

## Clean-clone result

Fresh clone to `/tmp/HydroSwarm-arm-migration-check` (removed after the test) from
`origin/agent/gcp-multitopology-v3` at `a898d5d` (before the doc-fix commit) --
`git clone` auto-pulled LFS content; `git lfs pull` afterward was a confirmed no-op.

- `git lfs ls-files | wc -l` → 124 (matches source).
- `find . -name "*.safetensors" -size -1k` → no output (no un-pulled pointer files).
- `scripts/verify_migration_artifacts.py --corpus-dir data/learning-v2/cycle-b2` →
  all five checks (`shard_manifests`, `checkpoints`, `calibration`, `controls`,
  `normalization`) passed with zero problems.
- One normalized training shard loaded via `ShardedScenarioDataset` (9,000 examples,
  a real example's `node_features` tensor shape confirmed).
- All four finalist checkpoints and both control checkpoints loaded via
  `HydroCore.from_variant(**architecture_config)` + `load_state_dict(strict=False)`
  with zero missing/unexpected keys.
- All four scenario archives extracted; `development_holdout` produced exactly 2,550
  `.npz` files (1,750 regular + 800 OOD), matching the source corpus's own count.
- Full `run_corpus_gates.py` suite: 8/9 passed on the first run
  (`mask_round_trip` failed only because `uv sync --frozen` does not install the dev
  dependency group `pytest` is in); installing it via
  `uv sync --frozen --extra dev` and re-running produced **9/9 passed**,
  `overall_status: "passed"` -- confirmed a real environment gap, not an artifact
  defect, before writing it up (see the `6b9bf22` doc-fix commit).
- Locked test confirmed absent: no path matching `*locked*` or `final-selection.json`
  anywhere in the clone; the only match for "final-selection" text was
  `configs/evaluation_policy_v3.json`'s own policy-schema reference to the concept, not
  an opened result.

## Exact clone/setup commands for the new Arm VM

See `docs/ARM_MIGRATION.md` for the complete, copy-pasteable walkthrough (clone + LFS
pull, environment recreation, archive extraction, checkpoint-loading smoke test, full
artifact verification, corpus gates, and a training-resume example). Quick summary:

```bash
git clone --branch agent/gcp-multitopology-v3 https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm
git lfs install && git lfs pull
uv sync --frozen --extra dev   # --extra dev needed for the full corpus-gate suite
export PYTHONPATH=src
python scripts/verify_migration_artifacts.py --corpus-dir data/learning-v2/cycle-b2
python scripts/run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2   # needs scenario archives extracted first (see docs/ARM_MIGRATION.md step 3)
```

## Remaining local-only files (not migrated, still on the source x86 VM)

- `experiments/runs/cycle-b2-stage2/` (~232 MiB), `cycle-b2-stage3/` (~870 MiB),
  `cycle-b2-stage4/` (~432 MiB) -- full per-epoch training run directories
  (periodic checkpoints, `metrics.jsonl`, `epoch_summary.json`, etc.). `.gitignore`d
  by design (`experiments/runs/`, not broadened by this migration); only each
  finalist/control's final export was extracted into the migrated `models/` trees.
- `experiments/cache/signatures/` (~132 KiB) -- regenerable `SignatureArtifact` cache.
- `data/learning-v2/cycle-a/`, `data/learning-v2/cycle-b/` (old) -- preserved
  historical corpora, out of this migration's scope.
- `reports/migration/_build_*.py`, `_inventory-raw-commands.txt`,
  `_experiments-runs-cycle-b2-files.txt` -- these ARE committed (standard Git, in
  `3b2c1a7`), not local-only; listed here only to note they are one-off provenance
  scripts, not part of the ongoing product surface.

None of the above is required to continue Scout/Strategist/OOD/auxiliary-head
training, rerun corpus gates, or reproduce Phase 3's corrected results on the new VM --
everything needed for those is in the two migration commits.

## Confirmation

The locked final test was not opened, inspected, copied, archived, or referenced by
path in any command run during this migration, on either the source repository or the
clean-clone verification. `final-selection.json` does not exist anywhere in the
migrated tree.
