# Model artifacts

Published safe-weight checkpoints live here. `hydrocore-s-learning-v1.safetensors` is the
promoted runtime model; `hydromono-s-learning-v1.safetensors` is the equal-budget baseline;
`hydrocore-m-learning-v1-partial.safetensors` is a two-epoch feasibility artifact, not a
converged model. The completed fixed-budget HydroCore-M candidate and its full provenance
are preserved under `experiments/learning-v2/hydrocore_m/`; it was evaluated but not
promoted, so HydroCore-S remains the runtime default. Adjacent metadata files contain
SHA-256, schema, corpus, size, parameter, and training provenance. Optimizer pickle state
is deliberately excluded.

Local optimizer checkpoints, signature caches, and optional exported runtimes remain
excluded from source control. Published artifacts must carry
a SHA-256 digest, architecture/configuration identifier, dataset-manifest digest, license,
and measured calibration/evaluation metadata. The application remains functional in its
classical-safe mode when no learned checkpoint is present.
