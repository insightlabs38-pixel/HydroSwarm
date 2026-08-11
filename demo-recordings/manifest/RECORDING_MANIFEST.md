# HydroSwarm recording manifest

## Capture environment

- Git HEAD: `fe8a09952735dfbf1d6f80da4904627c3cccc898` (`main`)
- Release tag: `v0.1.3-hackathon` → `c7e0a6bbd3281b28e72b2fcffba33cc491d4757e`
- Product equivalence check: post-tag changes are only `docs/DEVPOST.md` and `docs/SUBMISSION_CHECKLIST.md`; no frontend, backend/runtime, model/scientific, reference artifact, or API changes.
- Application start method: `.venv/bin/hydroswarm start --host 127.0.0.1 --port 8765`
- Pre-LIVE strict self-test: **READY** — frozen HydroCore-v4 bundle/hash, normalization, fitted calibration, WNTR/EPANET, frontend assets, and reference artifact passed.
- LIVE capture used the real loopback production backend: **yes**. No route interception or mocked API routing exists in the harness.
- Format: native Playwright Chromium WebM / VP8, 1920×1080, 25 fps source, device scale factor 1, silent.
- The archived `raw/` WebMs are the final byte-identical native Playwright captures. `Trim start` is the clean usable in-point in that raw file.
- H.264 MP4: **not created**. System `ffmpeg` is unavailable; the pre-existing Playwright-bundled ffmpeg can decode VP8/emit QC PNGs but has no H.264 encoder. No packages were installed.
- QC: `qc/` contains first usable, midpoint, and final usable PNGs for every clip. Resolution, non-empty/decodeable input, no horizontal overflow, provenance labels, and required DOM state were checked by the local harness.

## Clips

| File | Mode | Authored start → end | Duration / usable in-point | Visible action / cursor | Intended use | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `00_gateway_to_reference.webm` | REFERENCE | Gateway → 0 `alert` | 10.84s / 0.14s | Run Reference Incident; virtual cursor | Solution introduction | `6558e9bb641ea1e98caa9b706bc4a3ceb2b4eaf4e76f9dcab9e2c4fdcf1e11cd` |
| `01_reference_uncertainty.webm` | REFERENCE | 1 `initial_uncertainty` → 2 `evidence_insufficient` | 15.68s / 3.74s | Source, then Next; virtual cursor | Four candidates / planning withheld | `1cfe81b45dcb16392e50006c9b608632c256a42088cf6e7e259a42e1a4166504` |
| `02_reference_sampling_to_posterior.webm` | REFERENCE | 3 `sample_recommended` → 5 `posterior_contracted` | 19.88s / 5.00s | Replay sample collection, then Next; virtual cursor | Adaptive sampling / posterior contraction | `c9b2da97e6033d51da591151d6eb50b0a42c47f034cd2832561a167bc43d282d` |
| `03_reference_unsafe_plan_rejected.webm` | REFERENCE | 6 `plans_generated` → 7 `unsafe_plan_rejected` | 17.16s / 6.76s | Select feeder closure, then Next; virtual cursor | Safety proof: `REJECTED`, `PRESSURE_BELOW_MINIMUM`, −8.2 m margin | `8ba2049e338eba2154fce3cb50d57c6bdb6f630266ce6a5db82558b75e8999cc` |
| `04_reference_verified_alternative.webm` | REFERENCE | 8 `safe_plan_verified` → 8 | 18.72s / 8.21s | Select J4 flush; virtual cursor | Verified alternative / consequences | `46c355a62c5aa41c5e4039d2d621f1c85408af1b80dab57c729b496a542347ff` |
| `05_reference_human_approval.webm` | REFERENCE | 9 `human_approval_boundary` → 10 `completed` | 21.20s / 8.72s | Replay operator approval; virtual cursor | VERIFIED ≠ APPROVED; no infrastructure actuation | `37f885f19f580f6e912dbf4ff15166693b3dd1d7456d4a10abdf522084ac72d8` |
| `06_live_computation_proof.webm` | LIVE | Gateway → real `awaiting_approval` | 21.56s / 0.14s | Run Live Example; collect real sample; virtual cursor | LIVE production-computation proof | `c0a632294295ca80267152c589787dd246626fdaba6bb8aabda9494a1ef8d51c` |
| `07_completed_incident.webm` | REFERENCE | 10 `completed` → 10 | 18.40s / 7.89s | Static; cursor hidden | Closing b-roll | `d0e76c2fb050957f2697fe62167bfd2a490405dc59ce1beb3c6106760845a4fa` |
| `08_model_authority_broll.webm` | REFERENCE | 8 `safe_plan_verified` → 8 | 17.40s / 7.24s | Open Model & Authority; virtual cursor then hidden | Optional authority B-roll | `ab3081c72fddbd0e30ef1447e07dc2a63ea9f5ab43952d634c76a10c12901466` |
| `reference_master_take.webm` | REFERENCE | Gateway → 10 `completed` | 32.84s / 0.14s | Full authored replay, including both replay-only actions; virtual cursor | Transition backup/master | `da66e6044e627b375c5b58a49c2a841ad53fbf192dbc20c94a59c6ea7c134e47` |

## Editing notes

- The first seconds before each listed usable in-point are deliberate silent setup. Trim them in the editor; no product behavior was bypassed.
- The master take completed successfully, using `Replay sample collection` at milestone 3 and `Replay operator approval` at milestone 9.
- No dead computation time was fabricated in LIVE. The real local run reached the human-approval pause quickly; the final stable hold is intentional edit material.
- The virtual presentation cursor is runtime-only DOM injection: 26 px translucent neutral circle, high z-index, pointer-events none, eased 320–650 ms moves, and short 0.85-scale/ring clicks. It is hidden during static proof holds.
