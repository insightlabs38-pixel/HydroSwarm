# Submission checklist

- [x] Public repository is accessible and license is correct. (GitHub confirms public
      visibility and Apache-2.0; LICENSE is present.)
- [x] Native verification passed on hosted Linux x86-64, Linux ARM64, Windows x86-64,
      macOS ARM64, and macOS Intel (`macos-15`). Linux ARM64 additionally completed the
      clean runtime-ZIP setup → strict self-test → loopback-health release-path check.

| Release-path platform              | Actually verified status                                                                                                                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux ARM64 native                 | Verified: clean runtime ZIP extraction → setup → strict self-test → loopback `/api/health`; hosted native CI also passed                                                           |
| Linux x86-64 native                | Verified: hosted native CI passed                                                                                                                                                  |
| Windows x86-64 native              | Verified: hosted native CI passed, including strict self-test and real simulator smoke                                                                                             |
| macOS ARM64 / Intel native         | Verified: hosted native CI passed on both architectures (`macos-15` Intel)                                                                                                         |
| Docker linux/amd64 and linux/arm64 | Verified: hosted hardened-runtime gate passed natively on both architectures, including EPANET smoke, LIVE workflow, exact verification, approval, persistence, and offline checks |

- [x] `hydroswarm self-test` and the golden scenario pass with networking disabled
      (`tests/integration/test_offline_runtime_audit.py`, this session -- mechanically
      blocks every outbound socket connect() and re-runs both).
- [x] Python and frontend quality gates pass from a clean install (ruff, pyright, full
      pytest, npm lint/typecheck/format:check/vitest/build all green throughout this
      session; see `reports/submission-readiness/`).
- [x] Evaluation results were regenerated and match README/report tables. (This session
      additionally corrected a real staleness gap: the README/MODEL_CARD/EVALUATION docs
      cited only the superseded v3-era S/M/L benchmark with no mention that HydroCore-v4
      is the actual frozen default -- fixed; see docs/FINAL_SYSTEM.md.)
- [x] Frozen scenario outputs are computed, checksummed, and not embedded UI claims.
- [x] Technical report PDF visually reviewed; figures, tables, limitations, references present.
- [ ] 3:30-4:30 demo video has captions, clear audio, visible real outputs, and no cuts
      that imply false causality. **Not started** -- intentionally, per SS23: no
      placeholder video URL is presented as finished.
- [x] Fresh Playwright screenshots show the first-launch gateway, reference sampling,
      approval boundary, and LIVE V4 proof start without sensitive paths/data; all 53
      Playwright tests passed. README uses these as its primary
      visual story rather than the old fallback-only screenshot.
- [ ] Devpost write-up, built-with list, repository, video, and report links work
      anonymously. Write-up and built-with list updated this session
      (`docs/DEVPOST.md`); only the video link remains intentionally pending.
- [x] AI-assistance disclosure uses the approved wording.
- [x] No secrets, private data, restricted datasets, unreviewed checkpoints, or oversized
      caches are committed.
- [x] Safety boundary and absence of autonomous control are visible in README and UI
      (footer, ModeBanner, Approval workspace, docs/FINAL_SYSTEM.md's authority-boundaries
      section). **Not yet visible in a video** -- no video exists yet (see above).
- [x] A local `v0.1.0-hackathon` runtime ZIP was built, extracted into a clean Linux
      ARM64 directory, and passed setup → strict self-test → loopback launch. It includes
      checksums, the frozen V4 bundle, built frontend, and reference artifact. The
      `v0.1.3-hackathon` GitHub Release publishes the current runtime ZIP and
      `RELEASE_MANIFEST.json`; its GHCR multiarch image passed strict self-tests for both
      published architectures.
- [x] Hardened Docker verification (`--read-only`, non-root, dropped capabilities,
      no-new-privileges, `/data` persistence, `/tmp` tmpfs) passed on hosted native
      AMD64 and ARM64. This is not a Docker Desktop host-integration claim.
- [x] Final claims are limited to measured synthetic/reference-network results.

## What a human needs to do before this submission is truly ready

1. Record the demo video, showing the REFERENCE
   INCIDENT as the primary walkthrough.
