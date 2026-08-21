# Changelog

## Unreleased

- Rebased the release/setup/packaging surface onto the frozen HydroCore-v5
  M10 finalist: native setup verification, self-test wording, the release
  manifest generator, the runtime ZIP builder, and the Docker image now all
  resolve exclusively to `models/hydrocore-v5-release/` and its M11.2/M11.6
  evidence, with no current-path dependency on the historical
  `models/hydrocore-v4-release/` bundle (which remains in the repository as
  historical evidence). No model weights, calibration values, or M11.6
  results/gates/authorization changed; M11.6 was not rerun.
- Added a frozen robustness/scale characterization protocol, deterministic
  locked-test-excluding runner, raw results, summary, and documentation guard.
- Reconciled current HydroCore-v4 dataset/model/public-submission documentation
  with the frozen release bundle and characterization evidence.
- This is development metadata (`0.1.0`); no release/tag is cut by this entry.

## 0.2.0 - 2026-08-03

- Added deterministic bounded swarm orchestration and specialist agents.
- Added secure persistent EPANET imports and authoritative WNTR verification.
- Added graph-local/global HydroCore variants, native feature preprocessing, training,
  signatures, calibration, active sampling, response revision, explanations, and governed
  scenario generation.
- Added simulator caching, timeouts, exact-run budgets, extended consequences, and replay.
- Added a governed 1,320-scenario learning corpus, canonical feature schema v2, trained
  HydroCore-S and HydroMono-S checkpoints, partial HydroCore-M, calibration, and held-out
  bootstrap evaluation.
- Added default trained-asset loading with checksum/schema validation and classical-safe
  fallback; validated the promoted checkpoint through the full hybrid pipeline.
- Completed a fixed-budget 17-epoch HydroCore-M run with corrected profile label spaces,
  ordinal-aware objectives, calibration-only refitting, and a locked 2,000-bootstrap test.
  M did not pass promotion, so HydroCore-S remains the default.
- Added a genuinely different seven-junction EPANET topology experiment and unseen-hash
  OOD policy; low transfer coverage correctly produces `CAUTION` and suppresses planning.

## 0.1.0 - 2026-08-03

- Initial HydroSwarm scientific and local-application foundation.
