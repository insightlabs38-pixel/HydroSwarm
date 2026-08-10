# Submission checklist

- [ ] Public repository is accessible and license is correct. (LICENSE present and
      Apache-2.0; GitHub visibility/access settings not verifiable from this session.)
- [x] Clean Linux native installation commands pass (`./setup_hydroswarm_linux.sh` +
      `./start_hydroswarm_linux.sh` live-smoke-tested end to end this session: bundle
      verified, self-test READY, `/api/health` responded). **Windows/macOS not verified**
      -- no such environment was available this session; scripts are statically/
      structurally tested only (`tests/unit/test_native_setup_scripts.py`).
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
- [ ] Screenshots show the built application and contain no sensitive paths/data. The
      existing `docs/screenshots/operator-overview.png` (mission-control shell,
      DEMO_FALLBACK) is current, but **no new screenshot exists for the first-launch
      gateway or the REFERENCE INCIDENT experience added this session** -- no browser
      was available in this sandbox to capture one.
- [ ] Devpost write-up, built-with list, repository, video, and report links work
      anonymously. Write-up and built-with list updated this session
      (`docs/DEVPOST.md`); video/report links remain intentionally pending.
- [x] AI-assistance disclosure uses the approved wording.
- [x] No secrets, private data, restricted datasets, unreviewed checkpoints, or oversized
      caches are committed.
- [x] Safety boundary and absence of autonomous control are visible in README and UI
      (footer, ModeBanner, Approval workspace, docs/FINAL_SYSTEM.md's authority-boundaries
      section). **Not yet visible in a video** -- no video exists yet (see above).
- [ ] Release archive installs offline after dependencies/assets are included and
      contains checksums. `scripts/build_release_bundle.py` exists, is unit-tested
      (correct contents, no `__pycache__`, SHA256SUMS integrity), and was run locally
      producing a real, correct zip -- **but no actual tagged release archive has been
      cut and extract-and-run tested end to end.**
- [ ] Docker recommended judge path (`docker compose -f docker-compose.release.yml up`)
      actually builds/runs. **Blocked in this sandbox**: confirmed root cause is
      `CAP_SYS_ADMIN` stripped and `unshare` blocked even for an unprivileged user
      namespace -- see `reports/submission-readiness/sub3-docker-sandbox-limitation.md`.
      `.github/workflows/release.yml` is written and ready to verify this for real on a
      GitHub Actions runner or an unsandboxed machine; that execution has not happened.
- [x] Final claims are limited to measured synthetic/reference-network results.

## What a human needs to do before this submission is truly ready

1. Verify Docker on a real machine or GitHub Actions (`docker build`/`docker run`, then
   `docker buildx build --platform linux/amd64,linux/arm64`) -- see the two unchecked
   Docker/release items above and `reports/submission-readiness/sub3-docker-sandbox-limitation.md`.
2. Verify the native setup/launch scripts on real Windows and macOS machines (only Linux
   was live-tested this session).
3. Run the Playwright visual-regression suite on a real CI runner and confirm the
   baseline screenshots still match after this session's first-launch-gateway routing
   fix (`?experience=fallback` instead of bare `/` -- same DOM output expected, but
   unverified without a real browser).
4. Take new screenshots of the first-launch gateway and REFERENCE INCIDENT experience.
5. Record the demo video once the above are confirmed working, showing the REFERENCE
   INCIDENT as the primary walkthrough.
6. Cut an actual version tag once ready, and let `.github/workflows/release.yml` produce
   the real multiarch image, `RELEASE_MANIFEST.json`, and runtime zip on real
   infrastructure.
