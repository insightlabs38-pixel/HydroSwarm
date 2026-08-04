# Changed files

## Phase 0 (baseline)

No files under `src/`, `tests/`, `scripts/`, `frontend/src/`, `configs/`, `data/`, or `models/`
were changed. Only new files were added, all under `reports/overnight/2026-08-03/`:

- `environment.json`
- `resource-usage.json`
- `test-results.md`
- `completed-tasks.json`
- `failed-tasks.json`
- `changed-files.md` (this file)
- `experiment-plan.md`
- `experiment-registry.json`
- `training-jobs.md`
- `artifact-inventory.json`
- `follow-up.md`
- `summary.md`

## Bundle A (Tasks 0.2-0.8), commits f67b0a6..1c3b55b

New source modules under `src/hydroswarm/training/`: `registry.py`, `job_runner.py`,
`sharded_data.py`, `label_audit.py`, `sampler.py`, `split_policy.py`; a small addition to
`data.py` (`load_scenario_examples_jsonl`); a small SIGTERM-handling addition to
`trainer.py`. New module `src/hydroswarm/preprocessing/schema.py` additions
(`NormalizationStats.save/load/fingerprint`, `NODE_FEATURE_SEMANTICS`/
`EDGE_FEATURE_SEMANTICS`). New scripts: `scripts/audit_labels.py`,
`scripts/fit_normalization.py`. New config: `configs/evaluation_policy_v3.json`. New doc:
`docs/EVALUATION_V3_POLICY.md`. New governed artifacts: `reports/results/v3/label-audit-learning-v1.json`,
`reports/results/v3/normalization/{node,edge}-normalization.json(.sha256)`. Plus one
pre-existing-bug fix: `data/frozen/manifest.json` (commit `19468ac`). 65 new tests across
7 new test files plus additions to `tests/unit/test_preprocessing.py` and
`tests/scientific/test_training_smoke.py`. No file under `data/learning-v1/`, `models/`, or
any historical `reports/results/*.json` (pre-existing, non-v3) was modified.

This section will be appended to (not rewritten) as later bundles land, grouped by commit.
