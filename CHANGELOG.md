# Changelog

## Unreleased -- targeting `0.2.0` / future tag `v0.2.0`

Product minor release: material system-level changes since `0.1.x`, not a
patch. Highlights:

- HydroCore-v5 final serving identity: the API, native setup, self-test,
  release manifest, runtime ZIP, and Docker image all resolve exclusively
  to the frozen `models/hydrocore-v5-release/` bundle, with no current-path
  dependency on the historical `models/hydrocore-v4-release/` bundle
  (retained in the repository as historical evidence only).
- Completed the governed M10/M11 evaluation and freeze lifecycle, including
  the M11.6 locked evaluation (executed exactly once; `M11_6_LOCKED_EVALUATION_PASS`).
- Revised learned/deterministic authority boundaries: deterministic OOD,
  Scout, and planning retain operational authority; a learned output alone
  cannot mark a plan `VERIFIED`.
- Substantially upgraded operator console (frontend).
- V5-only setup/runtime/release path across native scripts, Docker, and CI.
- Release-manifest schema v2 and removal of V4 from current release
  dependencies.
- Revised native support matrix (Linux x86_64/ARM64, Windows x86_64, macOS
  Apple Silicon ARM64; macOS Intel is not supported -- no upstream
  `torch>=2.5` wheel exists for it).
- Fixed a release-workflow publication-order defect: the final image tag
  and `latest` are now only ever promoted (retagged, never rebuilt) from a
  digest that has already passed the strict amd64/arm64 container
  self-test, instead of being pushed before that test ran.

**Limitations:** all reported results are simulation-based measurements,
not field-validated; there is no claim of real-world utility performance.
Native macOS support is Apple Silicon only (no Intel Mac). Native Linux
ARM64 setup passes `self-test --strict`, but the Live Example endpoint
needs a manual `scripts/build_epanet_arm64.sh` step for real EPANET
water-quality simulation until that is wired into the setup script
directly -- see docs/INSTALLATION.md; the Docker image is unaffected.
HydroSwarm has no autonomous actuation path -- every plan requires a
separate human approval event.

Also in this cycle:

- Added a frozen robustness/scale characterization protocol, deterministic
  locked-test-excluding runner, raw results, summary, and documentation guard.
- Reconciled current HydroCore-v4 dataset/model/public-submission documentation
  with the frozen release bundle and characterization evidence.

No model weights, calibration values, or M11.6 results/gates/authorization
changed; M11.6 was not rerun. `pyproject.toml`/frontend package version is
now `0.2.0`; no `v0.2.0` tag or GitHub Release has been cut by this entry yet.

## 0.2.0 (internal, 2026-08-03) -- superseded, never tagged or published

> This entry predates the `vX.Y.Z-hackathon` tagging scheme (only
> `v0.1.0`-`v0.1.3-hackathon` were ever tagged/published; `pyproject.toml`'s
> version was reset to and stayed at `0.1.0` afterward -- see the
> `## 0.1.0` entry below, dated the same day). It is kept as historical
> development metadata, not as evidence of a real "0.2.0" release. The
> product version `0.2.0` first introduced in this repository's actual
> `pyproject.toml`/tag is the one under `## Unreleased` above, first
> targeted at future tag `v0.2.0`.

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
