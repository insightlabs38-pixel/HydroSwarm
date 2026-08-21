# Reproducibility

HydroSwarm separates **reproducible software/artifact verification** from **reopening the final held-out test**. The final M11.6 lock should not be rerun.

## Reproducibility target

A reviewer should be able to verify:

1. the exact frozen finalist identity;
2. the calibration identity and applicability policy;
3. the serving factory/allowlists;
4. the generated final-test design/materialization provenance;
5. the one authorized/opened execution;
6. the immutable metrics/gate/safety/closure chain;
7. the current source's V5 self-test and non-locked test behavior.

## Frozen identity checklist

| Artifact | Expected identity |
|---|---|
| V5 checkpoint | `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5` |
| release manifest | `f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34` |
| calibration file | `8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d` |
| calibration artifact hash | `f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd` |
| selected seed | `20260814` |
| parameters | 4,182,612 |
| release schema | `hydroswarm-v5-release-v1` |
| trained task | `sentinel` |
| learned runtime outputs | 5 frozen Sentinel outputs |

Compare [runtime manifest](../models/hydrocore-v5-release/runtime_manifest.json) with [M11.2 finalist freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json).

## Verify the current runtime

From the current source checkout:

```bash
hydroswarm self-test --strict
```

The machine-readable trained-assets block should report V5 release identity and the frozen checkpoint hash above. The self-test also performs bounded learned inference, a real WNTR smoke run, SQLite/resource/frontend/reference checks.

## Current V5 container reproduction

```bash
docker compose build
docker compose up
```

The current Dockerfile includes the V5 bundle and runs the strict self-test during image build.

Do not use `docker-compose.release.yml` to verify the final V5 system yet: it is pinned to the intended `v0.2.0` release image, but as of this pre-tag commit that tag has not been published, so the pull will fail.

## Verify the one-time lock without reopening it

The final evidence chain is:

1. [M11.2 finalist freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json)
2. [M11.6 materialization manifest](../data/locked/m11-6/m11-6-materialization-manifest.json)
3. [M11.6 opened record](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-opened-record.json)
4. [raw incident evidence](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-raw-incidents.jsonl)
5. [metrics](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-metrics.json)
6. [gate](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-gate.json)
7. [safety counters](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-safety-counters.json)
8. [post-run governance](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-post-run-governance.json)
9. [closure](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-closure.json)

The opened record binds the run to checkpoint, calibration, release manifest, design freeze, evaluator, code-under-test, and materialization manifest identities.

The post-run governance confirms:

- 125 rows complete;
- exactly one opening;
- authorization consumed;
- no retry/resume;
- no locked rerun;
- no post-lock tuning;
- no manifest/dataset changes;
- no code/evaluator changes.

## Why rerunning M11.6 would be the wrong reproduction

A one-time final holdout is evidence precisely because it is not repeatedly observed. Running it again after results are known would not “make the result more reproducible”; it would consume the same held-out data again and weaken the governance story.

To reproduce behavior:

- rerun ordinary unit/integration/scientific tests;
- run self-test;
- use development/golden fixtures;
- inspect/recompute statistics from already-recorded final evidence where allowed;
- verify hashes and artifact relationships.

Do **not** reopen or regenerate the final locked population as a fresh performance attempt.

## Non-locked checks

From an installed current checkout:

```bash
python -m pytest
python -m pyright
python -m ruff check src tests scripts
```

Historical artifacts record their own exact full-suite counts at each milestone; current counts can naturally change with later documentation/test-only work. Passing current tests does not rewrite the frozen M11.6 scientific result.

## Training provenance

The selected training record is [ARM_B_M9_6 seed 20260814](../reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed20260814.json). It records:

- 600 physical train scenarios;
- three equal-weight trained topology families;
- 20 epochs;
- 1,350 actual optimizer steps;
- `FINAL_STEP_1350` checkpoint policy;
- canonical checkpoint hash matching the final release.

The M9.6 manifest preserves the other seeds and calibration/development evaluation row identities.

## Known reproducibility caveats

- Early V5 M1 corpus generation records cross-environment WNTR/NumPy RNG divergence relative to historical `cycle-b2` replay; the generated V5 corpus is therefore identified by its own manifests rather than assumed to reproduce old scenario IDs.
- The final M9.6 train/serve unobserved-age semantic deviation is frozen and documented.
- Release packaging is not yet aligned: the pinned public release-compose image is older than the V5 source (a new immutable image tag must be published before `docker-compose.release.yml` can point at it).

These caveats should be fixed only in a separate authorized code/packaging pass, not silently rewritten into the frozen evidence.

## Evidence integrity principle

Reproducibility here means: **same artifacts, same hashes, same protocol identities, same recorded complete one-time execution, independently inspectable code and non-locked checks**. It does not mean treating a final holdout as a benchmark that can be rerun until convenient.
