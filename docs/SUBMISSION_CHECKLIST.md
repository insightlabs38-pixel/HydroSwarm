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
- [x] Current scientific documentation is rebased to the frozen HydroCore-v5 finalist and
      one-time M11.6 locked result. Historical V3/V4/M9/M10 artifacts remain preserved and
      explicitly separated from current claims; see [Final system](FINAL_SYSTEM.md),
      [Scientific evidence](SCIENTIFIC_EVIDENCE.md), and [Claims and evidence](CLAIMS_AND_EVIDENCE.md).
- [x] Frozen scenario outputs are computed, checksummed, and not embedded UI claims.
- [x] Technical report PDF visually reviewed; figures, tables, limitations, references present.
- [x] Demo video recording exists. Captions, audio, visible real outputs, and no cuts
      implying false causality remain a final-release review task; its public submission
      URL must be inserted manually on submission day.
- [x] Fresh Playwright screenshots show the first-launch gateway, reference sampling,
      approval boundary, and a LIVE proof start without sensitive paths/data; all 53
      Playwright tests passed. The screenshots predate the V5 documentation rebase and are
      retained as UI/workflow evidence, not as proof of the current finalist model identity.
- [ ] Submission-day/manual task: verify the Devpost write-up, built-with list,
      repository, report links, and final public video URL anonymously.
- [x] AI-assistance disclosure uses the approved wording.
- [x] No secrets, private data, restricted datasets, unreviewed checkpoints, or oversized
      caches are committed.
- [x] Safety boundary and absence of autonomous control are visible in README, UI, and
      the recorded demo. Final-release review must confirm the final exported video still
      shows that boundary.
- [x] Historical release artifacts are preserved as historical. The existing
      `v0.1.0-hackathon`/`v0.1.3-hackathon` release paths contain the frozen V4-era bundle;
      they are not presented as the current HydroCore-v5 serving identity. A new reviewed
      release/tag must be cut from the eventual final submission commit before claiming a
      published V5 image or runtime ZIP.
- [x] Hardened Docker verification (`--read-only`, non-root, dropped capabilities,
      no-new-privileges, `/data` persistence, `/tmp` tmpfs) passed on hosted native
      AMD64 and ARM64. This is not a Docker Desktop host-integration claim.
- [x] Final claims are limited to measured synthetic/reference-network results.

## What a human needs to do before this submission is truly ready

1. Submission day: insert the final public video URL and verify playback anonymously.
2. Final-release task: cut a new reviewed release/tag from the eventual final submission
   commit; do not relabel the historical `v0.1.3-hackathon` release.
