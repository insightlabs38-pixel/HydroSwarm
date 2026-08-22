# Submission checklist

- [x] Public repository is accessible and license is correct. (GitHub confirms public
      visibility and Apache-2.0; LICENSE is present.)
- [x] Native verification passed on hosted Linux x86-64, Linux ARM64, Windows x86-64, and
      macOS ARM64 (Apple Silicon). Linux ARM64 additionally completed the clean runtime-ZIP
      setup → strict self-test → loopback-health release-path check. Native macOS Intel/x86_64
      is **not supported** (no upstream `torch>=2.5` wheel exists for it); this is a stated
      platform boundary, not a gap in verification coverage.

| Release-path platform              | Actually verified status                                                                                                                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux ARM64 native                 | Verified: clean runtime ZIP extraction → setup → strict self-test → loopback `/api/health`; hosted native CI also passed                                                           |
| Linux x86-64 native                | Verified: hosted native CI passed                                                                                                                                                  |
| Windows x86-64 native              | Verified: hosted native CI passed, including strict self-test and real simulator smoke                                                                                             |
| macOS ARM64 (Apple Silicon) native | Verified: hosted native CI passed                                                                                                                                                   |
| macOS Intel/x86_64 native          | Not supported: no upstream `torch>=2.5` wheel; `setup_hydroswarm_macos.sh` fails early with this explanation rather than failing obscurely mid-install                            |
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
- [x] Final demo video (3:23) is complete: burned-caption master and clean master both
      completed. Public Vimeo URL is available:
      [vimeo.com/1220385465](https://vimeo.com/1220385465?share=copy&fl=sv&fe=ci#t=0). The
      project owner verified the public Vimeo URL loads and plays back in an incognito/private
      browser session; this is an owner-performed check, not an independent third-party
      verification.
- [x] Fresh Playwright screenshots show the first-launch gateway, reference sampling,
      approval boundary, and a LIVE proof start without sensitive paths/data; all 53
      Playwright tests passed. The screenshots predate the V5 documentation rebase and are
      retained as UI/workflow evidence, not as proof of the current finalist model identity.
- [x] Devpost write-up, built-with list, and report/link fields are complete.
- [x] Final Devpost anonymous-render QA (confirming the published Devpost page renders and
      links correctly when viewed anonymously) is complete.
- [x] AI-assistance disclosure uses the approved wording.
- [x] No secrets, private data, restricted datasets, unreviewed checkpoints, or oversized
      caches are committed.
- [x] Safety boundary and absence of autonomous control are visible in README, UI, and
      the recorded demo.
- [x] Historical release artifacts are preserved as historical. The `v0.1.0-hackathon`
      through `v0.1.3-hackathon` release paths contain the frozen V4-era bundle and are not
      presented as the current HydroCore-v5 serving identity. The current published
      HydroCore-v5 serving identity is release [`v0.2.1`](https://github.com/insightlabs38-pixel/HydroSwarm/releases/tag/v0.2.1)
      (`ghcr.io/insightlabs38-pixel/hydroswarm:v0.2.1`), a patch release over `v0.2.0` that
      fixes the governed LIVE sampling-abstention frontend path with no
      model/calibration/scientific change; see [CHANGELOG](../CHANGELOG.md).
- [x] Hardened Docker verification (`--read-only`, non-root, dropped capabilities,
      no-new-privileges, `/data` persistence, `/tmp` tmpfs) passed on hosted native
      AMD64 and ARM64. This is not a Docker Desktop host-integration claim.
- [x] Final claims are limited to measured synthetic/reference-network results.
