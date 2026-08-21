# Model artifacts

## `hydrocore-v5-release/` — promoted runtime model (current default)

The frozen **HydroCore-v5 M10 frozen release** bundle (`small`, 4,182,612 parameters,
selected seed `20260814`), loaded by `V5PipelineFactory` and wired as the default
`pipeline_factory` for the normal `hydroswarm.api.app:app` production entry point. `V5PipelineFactory`
never falls back to the historical V4 bundle below.

Frozen identity (matches `runtime_manifest.json` in this directory, the
[M11.2 finalist freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json),
and [Final system](../docs/FINAL_SYSTEM.md) exactly):

- model SHA-256: `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`
- release manifest SHA-256: `f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34`
- calibration file SHA-256: `8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d`
- calibration artifact hash: `f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd`
- runtime-enabled outputs: `event_cause`, `event_presence`, `evidence_sufficiency`,
  `relative_strength`, `source_node`

Every file's content hash is independently verified against `runtime_manifest.json`'s own
`files` mapping by `V5PipelineFactory` on every load — a corrupted or hand-edited file
fails closed, never loads silently, and never falls back to the historical V4 bundle.

## `hydrocore-v4-release/` — historical, no longer the default

A self-contained V4 inference release bundle (`hydroswarm.runtime.v4_inference_bundle`),
loaded by `V4PipelineFactory`. This was the promoted runtime model prior to the V5 freeze
(UI-11.1) and remains preserved as historical evidence -- no current runtime path (the
production app, native setup verification, strict self-test, or the Docker image) depends
on it any more, and it is not packaged into the current release ZIP or container image.
This is an exact byte-for-byte copy of
`experiments/runs/v4-release-bundle/no_adapters-seed20260810/` (an ephemeral, gitignored
path — see `experiments/runs/`'s own convention) at the moment
`hydrocore-v4-architecture-freeze` was tagged, except for the documented
capability-remediation calibration refit. Model weights and normalization remain unchanged.

Frozen identity (matches the `hydrocore-v4-architecture-freeze` git tag and
`reports/results/v4/architecture-freeze.json` exactly):

- architecture: `hydrocore-v4`, variant `small` (HydroCore-S), `use_adapters=false`,
  `prior_mode=feature_only`, seed `20260810`
- model SHA-256: `a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7`
- calibration artifact hash: `cf06c2000ead772d7de2d8cdcf00b7cb45e59b325f44be61114982531a4fa4d1`
  (FITTED, alpha=0.1, empirical coverage 91.43%)
- normalization hash: `e0808f21579b693f66e4edb5900e561bcf9c521e850d5c9d2428cb0db0fa1114`
- runtime-enabled outputs: `event_cause`, `event_presence`, `evidence_sufficiency`,
  `next_step`, `relative_strength`, `source_node`

Every file's content hash is independently verified against `SHA256SUMS` (cross-checked
against `artifact-manifest.json`) by `load_v4_inference_bundle()` on every load — a
corrupted or hand-edited file fails closed, never loads silently.

To reproduce this exact bundle from its real training checkpoint (not required — this
directory already IS that output, copied verbatim):

```
python scripts/build_v4_inference_release_bundle.py \
  --checkpoint-dir experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810 \
  --calibration-artifact experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/calibration.json \
  --output-dir experiments/runs/v4-release-bundle/no_adapters-seed20260810
```

## Legacy V3 artifacts (preserved, no longer the default)

`hydrocore-s-learning-v1.safetensors` was the promoted runtime model prior to UI-11.1 and
remains loadable via `DefaultPipelineFactory` (`hydroswarm.runtime.defaults`), which is
untouched by this change; `hydromono-s-learning-v1.safetensors` is the equal-budget
baseline; `hydrocore-m-learning-v1-partial.safetensors` is a two-epoch feasibility
artifact, not a converged model. The completed fixed-budget HydroCore-M candidate and its
full provenance are preserved under `experiments/learning-v2/hydrocore_m/`; it was
evaluated but not promoted. Adjacent metadata files contain SHA-256, schema, corpus, size,
parameter, and training provenance. Optimizer pickle state is deliberately excluded.

Local optimizer checkpoints, signature caches, and optional exported runtimes remain
excluded from source control. Published artifacts must carry
a SHA-256 digest, architecture/configuration identifier, dataset-manifest digest, license,
and measured calibration/evaluation metadata. The application remains functional in its
classical-safe mode when no learned checkpoint is present.
