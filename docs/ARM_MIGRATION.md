# Arm VM migration (from the x86 E2 GCP CPU VM)

This document is the exact setup path for continuing `agent/gcp-multitopology-v3`
(Cycle B2 corpus regeneration, finalist training, and calibration -- see
`reports/results/v3/phase3-completion-report.md`) on a new Arm VM. It does not cover
Scout/Strategist implementation (`core-issues2.txt`, not started) or opening the locked
final test (still unopened; `final-selection.json` still does not exist).

See `reports/migration/source-environment.json` for the exact source-VM versions this
migration was produced from, and `reports/migration/arm-migration-inventory.json` for
the complete artifact-by-artifact manifest (path, size, sha256, Git vs. LFS, source run
and seed, required/recommended/omitted).

## 1. Clone and pull LFS objects

```bash
git clone --branch agent/gcp-multitopology-v3 https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm
git lfs install
git lfs pull
```

`git lfs pull` downloads every `.safetensors`/`.pt`/`.tar.zst` object tracked per
`.gitattributes` (the Cycle B2 tensor corpus, the four finalist checkpoints, the two
control checkpoints, and the four scenario archives). Confirm nothing is left as a
pointer file:

```bash
git lfs ls-files | wc -l          # expect all Cycle B2 shard/checkpoint/archive objects
find . -name "*.safetensors" -size -1k   # expect no output -- a real LFS object is
                                          # never this small; an un-pulled pointer file is
```

## 2. Recreate the Python environment

```bash
# Confirm an Arm-compatible wntr wheel exists for this Python/OS before proceeding --
# wntr wraps EPANET, which ships prebuilt per-platform binaries inside the wheel.
# Every corpus-gate, calibration, and training script in this repo calls into it.
uv sync --frozen

# The mask_round_trip corpus gate shells out to `pytest` (it runs
# tests/unit/test_variable_collate.py as a secondary check) -- `uv sync --frozen`
# alone does not install the dev dependency group. Add it if running the full gate
# suite (step 5 below):
uv sync --frozen --extra dev
```

torch on the source VM was `2.13.0+cu130` (a CUDA build, unused -- every run in this
repository's history is CPU-only; see `source-environment.json`). Installing a CPU-only
or Arm-native `torch>=2.5` build is correct and expected to differ from the source
version string; it does not need to match exactly.

## 3. Extract the raw scenario archives (optional -- only if you need raw scenario
   replay, not just the tensor corpus)

The governed tensor corpus (`data/learning-v2/cycle-b2/tensors{,-normalized}/`) is
pulled directly via LFS in step 1 and is sufficient for training/calibration/gates.
Raw per-scenario `.npz`/`.parquet` arrays (needed only for scenario replay --
`run_corpus_gates.py`'s `deterministic_replay` gate, or `fit_dynamic_fusion_calibration.py`)
are compressed archives, extracted like this:

```bash
for split in train validation calibration development-holdout; do
  tar --use-compress-program=unzstd \
    -xf artifacts/migration/cycle-b2-scenarios-${split}.tar.zst \
    -C data/learning-v2/cycle-b2/scenarios
done
```

Each archive's member paths are already `<split>/<file>`, so extracting into
`data/learning-v2/cycle-b2/scenarios/` reproduces the original layout exactly (verified
byte-for-byte against the source directory before this migration was committed).
`development-holdout` includes the OOD-holdout scenarios (`UNSEEN_TOPOLOGY`,
`SEVERE_MISSINGNESS`) -- they live physically inside that split's scenario directory,
not a separate one. See `artifacts/migration/cycle-b2-scenarios-manifest.json` for each
archive's sha256, byte size, and file count.

## 4. Checkpoint-loading smoke command

```bash
export PYTHONPATH=src
python -c "
import json
from pathlib import Path
from safetensors.torch import load_file
from hydroswarm.model import HydroCore

for label in ('E1-seed20260810', 'E1-seed20260811', 'E0-seed20260810', 'E0-seed20260811'):
    root = Path('models/cycle-b2-candidates') / label
    config = json.loads((root / 'architecture_config.json').read_text())
    model = HydroCore.from_variant(config['variant'], **config['overrides'])
    missing, unexpected = model.load_state_dict(load_file(root / 'model.safetensors'), strict=False)
    assert not missing and not unexpected, (label, missing, unexpected)
    print(label, 'OK', model.parameter_count(), 'params')

for label in ('no-adapter-seed20260810', 'no-adapter-seed20260811'):
    root = Path('models/cycle-b2-controls') / label
    config = json.loads((root / 'architecture_config.json').read_text())
    model = HydroCore.from_variant(config['variant'], **config['overrides'])
    missing, unexpected = model.load_state_dict(load_file(root / 'model.safetensors'), strict=False)
    assert not missing and not unexpected, (label, missing, unexpected)
    print(label, 'OK', model.parameter_count(), 'params')
"
```

`architecture_config.json` (one per candidate/control directory) was synthesized during
this migration from the experiment registry's `opened` record plus the checkpoint's own
`model.safetensors` -- these training runs predate `promote_checkpoint.py`'s
`architecture_config` sidecar convention, so no such file existed at training time. It
records `variant`, `overrides` (e.g. `prior_mode=feature_only` for E1), the feature/
target schema hashes, topology hashes, corpus manifest hashes, and the exact
normalization artifact hashes the checkpoint was trained against.

## 5. Full artifact verification (checksums, calibration identity, corpus gates)

```bash
export PYTHONPATH=src
python scripts/verify_migration_artifacts.py --corpus-dir data/learning-v2/cycle-b2
```

Verifies: every tensor shard (raw and normalized) against its `manifest.json` sha256
and that raw/normalized split inventories match; all four finalist checkpoint hashes;
both dynamic-fusion calibration artifact identities (`CalibrationArtifact.artifact_hash`
-- a semantic property, not a file-bytes hash; the script loads and asks the artifact
directly, matching how runtime code checks it); both control checkpoint hashes; and both
normalization artifact hashes. Pass `--skip-shards` to skip the (slower) full shard
checksum pass if only checkpoint/calibration/normalization identity needs confirming.

Then the full governed corpus gate suite (core-issues.txt Phase 3 item 15; requires the
scenario archives extracted per step 3, for the `deterministic_replay` gate):

```bash
python scripts/run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2
```

If a full run is too expensive on first boot, the cheapest informative subset is
`shard_checksums`, `target_mask_validation`, `topology_provenance`, and
`normalization_ownership` -- these don't require the scenario archives and complete in
well under a minute; `deterministic_replay` and `mask_round_trip` are the two gates that
call into real WNTR simulation and cost more.

## 6. Resume or continue training

Point any of the Stage 2-4 scripts at the pulled corpus exactly as on the source VM
(paths are unchanged -- this migration deliberately preserves the existing directory
structure so no script edits are needed):

```bash
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

# Example: continue/replicate finalist training from the corrected corpus
python scripts/run_stage3_finalist_training.py \
  --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
  --finalists E1 \
  --run-root experiments/runs/cycle-b2-stage3 \
  --registry experiments/registry/cycle-b2-stage3-E1.jsonl \
  --output reports/results/v3/cycle-b2-stage3-E1.json
```

To load a preserved finalist checkpoint directly (e.g. as a starting point for the next
phase's Scout/Strategist/OOD/auxiliary head training -- not started; see
`core-issues2.txt`), use `models/cycle-b2-candidates/<label>/model.safetensors` with its
sibling `architecture_config.json` per the smoke command in step 4, rather than any path
under `experiments/runs/` (periodic/intermediate checkpoints there are not migrated).

## What is NOT included in this migration

- `experiments/runs/cycle-b2-*`'s periodic checkpoints (`checkpoint-0004/0008/0012`),
  Stage 2 screening checkpoints, and `best-model.safetensors`/`model-export.safetensors`
  duplicates -- only each finalist's final `checkpoint-0016` (candidates) and each
  control's final `model-export.safetensors` (controls) are preserved. `experiments/runs/`
  itself remains untracked and un-migrated (see `.gitignore`); re-run the relevant stage
  script if a periodic checkpoint is needed.
- The locked final test. It was not opened, inspected, copied, or archived at any point
  during this migration, and remains so.
- `.venv`, wheel caches, `node_modules`, credentials, and service-account files -- never
  captured by this migration's tooling and excluded by `.gitignore` regardless.

## Locked-test discipline

This migration touched only: `.gitattributes`, `.gitignore`, `data/learning-v2/cycle-b2/
tensors{,-normalized}/`, `models/cycle-b2-{candidates,controls}/`, `artifacts/migration/`,
`reports/migration/`, this file, and `scripts/verify_migration_artifacts.py`. No locked
test path was read, listed, hashed, archived, or referenced by any command run during
this migration.
