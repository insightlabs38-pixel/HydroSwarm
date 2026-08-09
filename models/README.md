# Model artifacts

## `hydrocore-v4-release/` — promoted runtime model (current default)

A self-contained V4 inference release bundle (`hydroswarm.runtime.v4_inference_bundle`),
loaded by `V4PipelineFactory` and wired as the default `pipeline_factory` for the normal
`hydroswarm.api.app:app` production entry point (UI-11.1). This is an exact byte-for-byte
copy of `experiments/runs/v4-release-bundle/no_adapters-seed20260810/` (an ephemeral,
gitignored path — see `experiments/runs/`'s own convention) at the moment
`hydrocore-v4-architecture-freeze` was tagged; nothing in this directory was regenerated,
retrained, or recalibrated to produce this commit.

Frozen identity (matches the `hydrocore-v4-architecture-freeze` git tag and
`reports/results/v4/architecture-freeze.json` exactly):

- architecture: `hydrocore-v4`, variant `small` (HydroCore-S), `use_adapters=false`,
  `prior_mode=feature_only`, seed `20260810`
- model SHA-256: `a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7`
- calibration artifact hash: `829c167b267b3ce32f55559f3aec4b4933e337f3358e22e1f792a26b402f68fa`
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
