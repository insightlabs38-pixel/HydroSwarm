# HydroSwarm recording manifest — v2 editorial pacing pass

## Capture record

- Git HEAD used for capture: `5e942edc1fec105ee6f5d092729fba2132531d3e` (`demo/video-footage-v1` baseline)
- Relationship to release: baseline product is equivalent to `v0.1.3-hackathon` → `c7e0a6bbd3281b28e72b2fcffba33cc491d4757e`; its only later main changes remain documentation-only.
- Product change check: no frontend application, backend/runtime, scientific/model, reference artifact, test, threshold, or release-content change was made for this pass.
- Backend start method: `.venv/bin/hydroswarm start --host 127.0.0.1 --port 8765`
- Strict self-test before capture: **READY** — frozen HydroCore-v4 bundle/hash, normalization, fitted calibration, WNTR/EPANET, frontend assets, reference artifact, SQLite, and port availability passed.
- API provenance: no API traffic was mocked or intercepted. The runtime-only recorder has no Playwright route handler and uses the local production endpoint directly.
- Capture format: Chromium / Playwright, 1920×1080, deviceScaleFactor 1, 25 fps, silent WebM VP8, no browser chrome or DevTools.
- Cursor: the v1 neutral 26 px runtime-only presentation cursor; eased movement, subtle click compression/ring, and hidden during static proof holds.
- QC: every file was fully decoded frame-by-frame after capture, passed the recorder’s horizontal-overflow check, and has first usable / midpoint / final usable frames. Primary proof images for 02, 03, and 05 were visually inspected.

`v2 is an editorial pacing pass. No scientific result, application state, reference artifact, model output, or product behavior was changed relative to v1.`

## Takes

| File | Authored start → end | Total / usable in-point / usable duration | SHA-256 | Editing purpose / QC |
| --- | --- | --- | --- | --- |
| `raw/01_reference_uncertainty_v2.webm` | 1 `initial_uncertainty` → 2 `evidence_insufficient` | 26.68s / 3.82s / 22.86s | `a5ec420f820c87174f34a672a4395bc72369a8955e8beb924e2715560cc1e29f` | Source uncertainty then withheld planning; PASS |
| `raw/02_reference_sampling_to_posterior_v2.webm` | 3 `sample_recommended` → 5 `posterior_contracted` | 32.52s / 5.20s / 27.32s | `f64744265ba52623199bb52b27e3323c2a66e670eb27ef225cbf049f97c69302` | Causal sample → evidence → J2 99% posterior; PASS, posterior proof visually inspected |
| `raw/03_reference_unsafe_plan_rejected_v2.webm` | 6 `plans_generated` → 7 `unsafe_plan_rejected` | 37.36s / 7.14s / 30.22s | `d0ac771ee40d9d3100ed9e2ee4a5942da90b4229888e0828d24e6b5f07b1f88e` | Feeder closure then deterministic `REJECTED` / `PRESSURE_BELOW_MINIMUM`; PASS, rejection proof visually inspected |
| `raw/04_reference_verified_alternative_v2.webm` | 8 `safe_plan_verified` → 8 `safe_plan_verified` | 29.40s / 8.51s / 20.89s | `3b80d3e4865c145e3e8e06841f805e56fdb86df73916bf773e2e3ca7e2f51af8` | Verified J4 flush alternative and safe consequence comparison; PASS |
| `raw/05_reference_human_approval_v2.webm` | 9 `human_approval_boundary` → 10 `completed` | 32.60s / 8.98s / 23.62s | `3748d95c91a8b973969444d6e56e5bd6dfbe9b65a2a36f98090647300c1e2cde` | Exact verification, human approval boundary, and no infrastructure actuation; PASS, pre-approval proof visually inspected |
| `raw/reference_master_take_v2.webm` | Gateway → 10 `completed` | 106.08s / 0.15s / 105.93s | `1ef283906d37447203263b76813b7e9baaa99a706f5e1ae1bd34f743052cbbe4` | Long-form authored replay backup, including both replay-only actions; PASS |

## QC files

- Standard evidence: `qc/<take>-first.png`, `qc/<take>-mid.png`, `qc/<take>-final.png`, and `qc/<take>-decode-terminal.png`.
- Primary proof frames: `qc/02_reference_sampling_to_posterior_v2-posterior-proof.png`, `qc/03_reference_unsafe_plan_rejected_v2-rejection-proof.png`, `qc/05_reference_human_approval_v2-approval-boundary-proof.png`.
- All v2 assets are separate from v1 under `demo-recordings/v2/`; no v1 raw file, QC image, hash, or manifest was modified.
