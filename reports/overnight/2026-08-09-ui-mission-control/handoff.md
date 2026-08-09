# UI mission-control implementation — handoff report

Authoritative spec: `/workspace/ui-work.txt` ("HydroSwarm UI Master Implementation Plan").
This report is kept current throughout the run per the operator instructions; it is
distinct from `reports/overnight/2026-08-03/` (the backend/model overnight plan).

## Branch / commit ledger

- Branch: `feature/ui-mission-control-v1` (pre-existing, was 0 commits ahead of `main`
  at session start; **not** pushed to or worked on `main`).
- Baseline commit (session start, matches ui-work.txt §0's stated baseline exactly):
  `c1fc9cc326a9d8ba8f6573fe5d2c200a10ca89cd`
- Commits this session:
  1. `6cce3a5` — `fix(frontend): align live view models with governed backend data` (UI-0)
  2. `09d69c8` — `docs(handoff): start UI mission-control run handoff report`
  3. `2868795` — `feat(ui): introduce HydroSwarm mission-control shell` (UI-1)
  4. `37f956e` — `docs(handoff): record UI-1 completion and visual-verification finding`
  5. `3dc2b49` — `feat(ui): rebuild synchronized operational network map` (UI-2)
  6. `7db5ab2` — `docs(handoff): record UI-2 completion`
  7. `f173a65` — `feat(ui): add governed source-localization workspace` (UI-3)
  8. `83c72d2` — `docs(handoff): record UI-3 completion`
  9. `7381d90` — `feat(ui): add deterministic evidence-value sampling workspace` (UI-4)
  10. `15538af` — `docs(handoff): record UI-4 completion`
  11. `bf77697` — `feat(ui): add verified response-plan decision workspace` (UI-5)
  12. `54a52b3` — `docs(handoff): record UI-5 completion and exact-run-budget scope decision`
  13. `4d50e37` — `feat(ui): implement guarded human plan-approval workflow` (UI-6)
  14. `ab347f5` — `docs(handoff): record UI-6 completion and demo-fixture invariant lesson`
  15. `8c4f38b` — `feat(ui): add synchronized technical evidence dock` (UI-7)
  16. `51c7ff5` — `docs(handoff): record UI-7 completion`
  17. `8d7cba7` — `feat(ui): add deterministic replay and fail-closed demo states` (UI-8)
  18. `f218379` — `UI-9: Network and Model & Authority utility workspaces`
  19. `bcf398a` — `docs(handoff): record UI-9 completion and UI-10 scoping note`
  20. `0ccdaf3` — `UI-10: harden accessibility, responsive layout, and reduced-motion`
  21. `20ae245` — `docs(handoff): record UI-10 completion and UI-11 scoping note`
  22. `a73d034` — `fix(ui): consolidate mission-control visual hierarchy before final qa` (UI-10.5)

Commits 1-20 were pushed to `origin/feature/ui-mission-control-v1` in the session that
produced them. Commit 22 (this session, UI-10.5) has **not** been pushed — see "GitHub
push" note in the UI-10.5 section below.

## Environment

- Node v20.19.2, npm 9.2.0 (frontend `node_modules` was not present at session start;
  ran `npm install` — see package-lock.json diff, which is npm-version metadata churn
  only, not a dependency change).
- Frontend gate commands: `npm run lint`, `npm run typecheck` (new script, `tsc -b`),
  `npm test -- --run` (vitest), `npm run build`. All four pass as of the latest commit.
- `npm run format:check` (prettier): **now passes on the whole repo** — the pre-existing
  gap noted after UI-0 (6 files failing on `main` before this session touched them) was
  fixed incidentally in the UI-1 commit when `prettier --write` ran over all of `src/`;
  diffs verified formatting-only before committing.
- Playwright/chromium was installed this session (`npx playwright install --with-deps
  chromium`) and used to manually verify the UI-1 shell renders correctly in a real
  browser (`vite build` + `vite preview` + a throwaway screenshot script, not committed)
  at 1920x1080 and 1440x900 — this caught two real bugs (an aria-required-children
  violation and a CSS grid layout bug in the audit list) that the automated gate
  (vitest + jsdom + axe) did not catch. **Recommendation: do this manual visual check
  after every phase with layout/CSS changes**, not just UI-0/UI-1 — jsdom has no real
  layout engine, so CSS grid/flex bugs are invisible to `npm test`.
- Backend (Python) test suite was **not** re-run this session (no `src/hydroswarm/**`
  files were touched by any commit so far — 0 backend files changed). The most recent
  commit on `main`/this branch's own history (`c1fc9cc` and its immediate ancestors)
  records "874 passed, 0 failed" locally. If a future phase needs a backend-exposing
  change, run the backend gate (`pytest -q`, `ruff check`, `pyright`) before committing
  it, per ui-work.txt §3's "covered by Python CI" requirement for any backend touch.
- No WNTR/EPANET native build was needed this session (no live simulation was run;
  UI-0 only consumed already-defined Pydantic/TypeScript contracts by inspection).

## Backend API surface (confirmed present, read-only research this session)

All the endpoints ui-work.txt §9 requires the frontend to consume **already exist** in
`src/hydroswarm/api/app.py` (single-file router) with matching Pydantic models in
`src/hydroswarm/api/state.py` / `src/hydroswarm/domain/schemas.py`:

- `GET /api/incidents/{id}/view` → `IncidentView` (state.py:221)
- `GET /api/incidents/{id}/authority` → `list[DecisionCertificate]` (app.py:484)
- `GET /api/incidents/{id}/evidence-certificate` → `EvidenceCertificate` (app.py:544)
- `GET /api/incidents/{id}/frontier` → `list[ParetoFrontierEntryView]` (app.py:503)
- `GET /api/incidents/{id}/explanations/{intent}` → `GroundedExplanation` (app.py:1519)
- `POST /api/incidents/{id}/plans/{plan_id}/verify` (app.py:874)
- `POST /api/incidents/{id}/plans/{plan_id}/approve` (app.py:983, fails closed on stale
  context with 409, matches ui-work.txt §9.7 exactly)
- `POST /api/incidents/{id}/replay` (app.py:1038)
- `WS /ws/incidents/{id}` (app.py:1523)
- `GET/POST /api/networks*` (app.py:380-410)

No model-governance/benchmark API route exists — `ModelGovernanceTable.tsx` reads a
committed static `frontend/public/model-governance.json` instead (its own `"note"`
field says a live endpoint is future work). UI-9 (utilities) should keep using that
file unless/until a live endpoint is added.

**No OpenAPI/TS codegen exists** — `frontend/src/api.ts`'s hand-maintained
`ApiIncidentView` interface is the only thing keeping the TS and Pydantic shapes in
sync, and it can drift. Every phase that adds new fields to a workspace should
double-check the Pydantic model, not just extrapolate from existing TS.

## UI-0 — data integrity and CI baseline: COMPLETE

See commit `6cce3a5` message for full detail. Summary: `NetworkLink.flow/concentration`,
`Plan.exposureReduction`, and `recommendedSample.delayMinutes/cost` are now
`number | null` and genuinely `null` in LIVE mode instead of a fabricated `0`. `Plan`
now carries a fully-typed `PlanVerificationView` (decision, simulator/version, state
hash, consequences, worst-case consequences, evaluation provenance, rejection codes,
`verificationStatus: CURRENT|STALE`, context hash) and typed `PlanAction[]`, both
mirroring the backend Pydantic models field-for-field. `EvidencePanel` now shows real
`evidence_history` rounds instead of a fabricated before/after probability delta.
`HydraulicChart` takes the incident and only renders a real time series in
DEMO_FALLBACK/REPLAY (LIVE/ERROR get a real sensor-snapshot bar chart or an empty
state). `OperationalMap` now `fitBounds()`s to real node geometry instead of a
hard-coded demo center, and disables Flow/Concentration layers + their checkboxes with
"data unavailable" when no link carries real data (true for all LIVE incidents today,
since the backend doesn't expose per-link flow/concentration yet — see gap below).

No backend files were changed. Zero regressions: lint/typecheck/tests/build all green.

### Known gaps carried forward (not blockers, recorded per instructions)

1. **Per-link flow not exposed.** `HydraulicSimulator` computes it but
   `NetworkLinkView` (state.py) doesn't carry it onto `/view` yet. This is a legitimate
   candidate for a narrowly-additive backend exposure (ui-work.txt §3: "Any backend
   change made only to expose an existing field... must be additive, typed, tested")
   if UI-2 (map rebuild) wants a real Flow layer instead of a permanently-disabled one.
   Not done in UI-0 to keep that phase's diff bounded to typing/data-integrity only.
2. **No no-response exposure baseline.** `Plan.exposureReduction` has no backend
   comparator computing it against doing nothing; stays `null` in LIVE indefinitely
   until that exists. DEMO_FALLBACK fixture still shows illustrative numbers (it's a
   hand-authored, clearly-labeled complete trace, not a live-capability stand-in).
3. **No sample delay/cost fields** on `SampleRecommendationView` — same treatment.

## UI-1 — design tokens and shell: COMPLETE

See commit `2868795` message for full detail. Summary: new `styles/tokens.css` with the
exact ui-work.txt §7 token set (legacy `--ink`/`--muted`/`--panel-2`/`--line` aliased
onto it so unmigrated pages keep working); `store.ts` rebuilt to the full
`ConsoleUiState` contract from §10 (workspace navigation replaces the old flat page-tab
model; only layout prefs persist to localStorage); new `shell/` components
(MissionHeader, ModeBanner, WorkflowRail, WorkspaceToolbar, DecisionInspector,
TechnicalDock) assembled into the exact header/rail/toolbar+workspace/inspector/dock
layout from §5. `WorkflowRail` derives every stage's status (complete/current/waiting/
blocked/caution/unavailable) from real `IncidentView` fields — workspaces without a real
implementation yet (Source/Sampling/Response/Approval/Replay/Network/Model & Authority)
are honestly `unavailable`/placeholder rather than faking readiness or content.
`TechnicalDock`'s six tabs are wired to real data already on `IncidentView` (nothing new
from the backend): Timeline/Evidence/Hydraulics reuse UI-0's components; Verification/
Provenance/Audit are new. `Overview.tsx` had its now-redundant hydraulic/timeline/
evidence panels removed (the dock owns that content now) but keeps everything without a
dedicated workspace home yet (map, source candidates, sample, sensors, counterfactuals,
plan table, explanation).

Two small narrowly-additive TS-mirror-only field exposures (`IncidentView.generatedAt`,
`.runtimeAnalysisMode`) — both fields already exist in the backend's `/view` JSON
response (`generated_at`, `runtime_mode`), just weren't mapped by the hand-maintained TS
interface before. No backend files changed.

`AuditPage.tsx`/`TopologyPage.tsx` are temporarily unrouted (not deleted) — their content
either moved into the dock (Audit) or awaits its own phase (Topology/debug, not yet
placed in the new IA). Revisit if a UI-9 utilities decision is needed for Topology.

No backend files were changed in either UI-0 or UI-1. Zero regressions: lint/typecheck/
format:check/tests/build all green, verified visually in a real browser at two viewports.

## UI-2 — map rebuild: COMPLETE

See commit `3dc2b49` message for full detail. Summary: removed the old static
`NetworkLink.action` field (a fake per-link property disconnected from which plan was
actually selected) in favor of rendering the real overlay from whichever plan is
selected (falling back to recommended), resolved against live geometry, styled per
ActionType. Added click-to-select on nodes/links (updates the shared store + a visible
highlight layer + a themed popup via `setDOMContent`, never string-built HTML). Wired
`WorkspaceToolbar`'s previously-disabled "Fit network"/"Layers" buttons to the
already-defined `mapLayers`/`mapFitRequestedAt` store fields (this was UI-1's own
explicit deferral note). All new effects update sources/filters/viewport only, never
rebuild the map (rebuild still keys only on `incident`/hasFlowData/hasConcentrationData).

**Real bug caught only by browser verification, not the automated gate** (which mocks
`maplibre-gl` entirely, so it can't catch real async-timing bugs): `map.on('load', ...)`
fires asynchronously in a real browser, so the plan-overlay-update effect could run and
bail out (style not loaded yet) before 'load' fired, and never got a second chance —
silently leaving the default-selected plan's action overlay empty on first render, with
zero automated test noticing. Fixed by seeding the action sources with real data at
creation time. **This continues to validate the UI-1 handoff note**: any phase touching
map/canvas/layout should get a real Playwright screenshot pass before being called done,
not just a green `npm test`.

## UI-3 — Source/Sentinel workspace: COMPLETE

See commit `f173a65` message for full detail. Summary: restructured `src/api.ts` into
`src/api/{client,incident}.ts` (only now that a second module needs the shared
`request()` helper — not preemptively) and added `src/api/authority.ts` consuming the
real `GET /incidents/{id}/authority` endpoint. Confirmed exact certificate names
(`source_localization`, `scout_recommendation`, `ood_state`,
`plan_consequence:{plan_id}`) by reading `hydroswarm/inference/authority.py` directly
rather than guessing. **Important discovery**: the backend's `/view` response already
computes grounded explanations for *every* `ExplanationIntent` up front (see
`get_incident_view` in app.py) — the frontend was just discarding all but `WHY_SOURCE`'s
text into a single string. Wiring "Why this source?" for real needed zero new network
calls, just keeping `IncidentView.explanations: GroundedExplanation[]` around. The same
applies to "Why this sample?"/"Why plan rejected?"/"What uncertainty remains?" — those
questions' data is already sitting in `incident.explanations`, ready for UI-4/UI-5/UI-6
to wire up without any new backend work.

New `SourceWorkspace.tsx` (real primary workflow-rail content, not a placeholder):
ranked candidates, calibration/coverage, classical-neural disagreement, OOD, the
`source_localization` authority certificate (new reusable `AuthorityBadge`/
`ApplicabilityBadge` components), and the real grounded explanation. `DecisionInspector`'s
Source case now shows a real query-free summary. DEMO_FALLBACK gets a hand-authored
`demoAuthorityCertificates` set (since `/authority` only exists for a real LIVE
incident) — clearly attributable to the existing DEMO_FALLBACK banner, never used in
LIVE mode. Verified visually via Playwright.

## UI-4 — Sampling/Scout workspace: COMPLETE

See commit `7381d90` message for full detail. Summary: `src/api/sampling.ts` +
`fetchEvidenceCertificate()` against the real `GET /incidents/{id}/evidence-certificate`
endpoint; new `SamplingWorkspace.tsx` showing evidence status (with the deterministic
`scout_recommendation` authority badge, reusing UI-3's `fetchAuthorityCertificates`),
sample-budget state, the next-sample recommendation with real EIG/candidate-reduction/
accessibility/alternatives, and the real grounded "Why this sample?" explanation.
"No further sampling recommended" is an explicit labeled empty state, never a blank
card. `demoEvidenceCertificate` added for DEMO_FALLBACK, consistent with the rest of the
fixture. Verified visually via Playwright.

## UI-5 — Response/Strategist workspace: COMPLETE

See commit `bf77697` message for full detail (the largest single phase so far). Summary:
`src/api/planning.ts` + `fetchParetoFrontier()` against the real
`GET /incidents/{id}/frontier` endpoint (confirmed exact `FrontierMode` literal and
field shapes from `hydroswarm/planning/pareto.py`); new `ParetoFrontier.tsx` (2D ECharts
scatter + dense table, two **entirely separate** chart+table pairs for EXPOSURE_AWARE vs
HYDRAULIC_ONLY — never merged, per ui-work.txt 15) and `PlanActionSequence.tsx` (full
ordered action list, not just a count); new `ResponseWorkspace.tsx` assembling
verification detail, action sequence, the existing `PlanTable`, the frontier, and
"Compare plans?"/"Why plan rejected?" explanations. `WorkspaceToolbar` gained a real
Response-specific "Posterior-weighted / Worst-case" frontier-context toggle
(`store.frontierMode`). Verified visually — this run caught and fixed one real
consistency bug (the frontier chart wasn't highlighting the recommended-plan fallback
the rest of the workspace uses).

**Deliberate scope decision, not an oversight**: the "exact-run budget" gap flagged
after UI-3 (`IncidentState.remaining_epanet_budget`/`plans_exactly_verified`/
`exact_simulation_cache_hits` exist on the backend but not on `IncidentView`) was
**not** closed with a backend exposure in this phase. Reasoning: doing so would be this
session's first Python file change, requiring a full backend gate run (pytest/ruff/
pyright, ~874 tests) per ui-work.txt §3's "covered by Python CI" requirement for any
backend touch, and would context-switch away from the frontend-focused momentum of this
run. It remains a legitimate, still-open, narrowly-additive candidate for a future
session — recorded here explicitly rather than silently dropped.

## UI-6 — guarded human plan-approval workflow: COMPLETE

See commit `4d50e37` message for full detail. Summary: `api/client.ts` gained
`requestJson()` + a real `ApiError` class carrying the HTTP status and the backend's
actual `{"detail": "..."}` message (previously discarded — several of this backend's
error strings, like the stale-verification 409, *are* the exact operator copy
ui-work.txt specifies). New `api/approval.ts` + `ApprovalWorkspace.tsx`: the required
SIMULATOR VERIFIED → CURRENT CONTEXT → OPERATOR REVIEW → HUMAN APPROVED ladder, full
plan/consequences/verification-context display, a form requiring both a non-empty
operator ID and an explicit review checkbox before Approve enables (no one-click
approval), a real receipt on success, fail-closed on 409 with the real backend reason
shown. DEMO_FALLBACK never performs a real mutation (no live UUID to record against) —
disabled with an explicit reason.

**Found and fixed a real data bug this feature exposed**: `demoFixture.ts`'s
`selectedPlanId: 'B'` (meaning "already approved") coexisted with `approvalPending:
true` — contradictory, and something no prior phase's UI surfaced clearly enough to
notice. Real backend behavior guarantees these two fields can't both be true
simultaneously (`approve_plan` always sets `approval_pending: False` in the same
transaction as `selected_plan_id`), so this was a genuine fixture bug, not a design
choice — fixed to `selectedPlanId: null`. Worth remembering for future demo-fixture
edits: **cross-check new fixture fields against the invariants implied by real backend
transactions**, not just against what individually "looks plausible."

## UI-7 — technical dock: COMPLETE

See commit `8c4f38b` message for full detail. As predicted in the UI-6 handoff note,
this phase was smaller than its number suggests: the dock frame/tabs/resize/collapse
and all six tabs' content already existed from UI-1. The real remaining work was
cross-panel synchronization (ui-work.txt §22) that nothing had audited yet: the Audit
tab and Timeline tab were entirely independent of each other (fixed bidirectionally),
and the Verification tab only reacted to an explicit plan click instead of falling back
to the recommended plan the way every workspace since UI-3 does (fixed to match). Also
added axe coverage for the Source/Sampling/Response/Approval workspace bodies (UI-1 only
ever checked the default Incident view) — all pass with zero violations.

## UI-8 — replay/failure/demo: COMPLETE

See commit `8d7cba7` message for full detail. Confirmed by reading the backend first
(per the note left in the UI-7 handoff): `/replay`'s `state` field is the incident's
*current* raw state, not a historical snapshot, so there really is no historical
map/chart data to replay — this backend only supports real event-ledger replay, and the
new `ReplayWorkspace` never pretends otherwise. Also closed the "exact-run budget" gap
(flagged after UI-3/UI-5) at zero backend cost, since `/replay`'s state already carries
those fields. **`mode: 'REPLAY'` remains genuinely unset by any code path** — there is
no feature today that constructs a full historical `IncidentView` (nodes/links/
candidates/plans) from a stored trajectory, and fabricating one just to exercise
`ModeBanner`'s REPLAY branch would violate ui-work.txt 34's "fabricate replay state".
That branch stays defined-but-dormant until (if ever) a real "load a stored trajectory"
feature is built — this is a deliberate honesty choice, not a gap to close later.

## UI-9 — utilities workspaces (Network, Model & Authority): COMPLETE

See commit `f218379` message for full detail. Validation and Benchmarks were already
migrated into the shell in UI-1 (mounting the pre-existing `ValidationPage`/
`BenchmarkPage`), so this phase's real scope was the two remaining utilities:

- **Network workspace** (new `api/networks.ts` + `NetworkWorkspace.tsx`): the first
  frontend consumer of `GET /api/networks` / `POST /api/networks/import` anywhere in the
  app. Import is a real local-file-only multipart/form-data POST (raw `fetch()` with a
  `FormData` body — deliberately not `requestJson()`, since the browser must set the
  multipart boundary itself, so no `Content-Type` header is set manually; confirmed via a
  test asserting `init.headers?.['Content-Type']` is `undefined`). Network detail shows
  real node/link counts, sha256, validation errors, and a coordinate-coverage percentage
  computed from real topology metadata (`null`/"not measured" when metadata carries no
  nodes, never fabricated). `TopologyPreview` renders a Cytoscape graph built only from
  the real imported node/link list — empty metadata correctly yields an honest
  `EmptyState`, not a placeholder graph.
- **Model & Authority workspace** (new `AuthorityWorkspace.tsx`, reusing UI-3's
  `fetchAuthorityCertificates`/`demoAuthorityCertificates`): a governance table showing
  every Decision Certificate together (`source_localization`, `scout_recommendation`,
  `ood_state`, `plan_consequence:*`) instead of just the one UI-3 needed, plus the
  seven-level authority ladder (`UNAVAILABLE`..`HUMAN_APPROVED`) as a real `<ol>` so a
  lower-authority result can never be visually confused with a higher one.

Both wired into `App.tsx`, `WorkflowRail` (network/authority stage status no longer
hardcoded `'unavailable'` — network reflects current/complete regardless of incident
mode since it's not incident-scoped; authority stays `'unavailable'` in ERROR mode since
there's no analysis to certify), and `DecisionInspector` (real inline summaries, no new
queries, following the same query-free pattern as the `validation`/`benchmarks` cases).
The `NOT_YET_MIGRATED_DETAIL`/`NOT_YET_IMPLEMENTED` placeholder maps are now empty
(defensive fallback only) since every workspace in the rail is implemented.

Gate: tsc/eslint/prettier clean, 81/81 tests (14 files), production build succeeds
(`NetworkWorkspace` is its own lazy chunk, 448.63 kB / 143.57 kB gzip — includes
cytoscape, matching the existing per-workspace code-splitting pattern). Verified visually
via a real Playwright screenshot against a `vite preview` server: zero console errors,
both workspaces render correctly (including the Network workspace's honest "Networks
unavailable / HydroSwarm API 500" state against the actually-unreachable local backend —
confirms the error path is real, not mocked). `AuditPage.tsx`/`TopologyPage.tsx` remain
intentionally unrouted from UI-1 — Topology's content now has a real, better home
(`NetworkWorkspace`'s per-network `TopologyPreview`), and Audit's content already moved
into the technical dock in UI-1/UI-7, so neither page needs its own route. No backend
files changed (0 files under `src/hydroswarm/` this phase, as in every phase so far).

## UI-10 — accessibility, responsive, and reduced-motion hardening: COMPLETE

See commit `0ccdaf3` message for full detail. Started with an explicit audit against
ui-work.txt §24-26 rather than assuming gaps — this found the shell already had strong
foundations from earlier phases (skip link, semantic landmarks, correct ARIA tabs, a
keyboard-accessible dock splitter with arrow-key resize, real map/chart text equivalents
generated from the same real data as the visual, status badges already distinguished by
icon shape and not just color, stable TanStack Query keys, memoized GeoJSON transforms in
`OperationalMap`, and a single persistent MapLibre instance updated in place rather than
recreated — all confirmed by reading the actual component code, not assumed). The phase's
real work was the concrete gaps that audit surfaced:

- **Skip link didn't move keyboard focus.** `<a href="#main-content">` scrolled the
  target into view but never shifted actual keyboard focus there (the `<main>` had no
  `tabIndex`), so a keyboard user activating it saw no change to subsequent Tab order —
  a real, well-known WCAG skip-link pitfall. Fixed with `tabIndex={-1}` on `#main-content`.
  (jsdom doesn't implement a real browser's native fragment-focus behavior, so this is
  only unit-tested for its precondition — `tabindex="-1"` present — with the actual
  post-activation focus move confirmed via a real Playwright/Chromium run instead.)
- **Responsive breakpoints didn't match ui-work.txt §25's tiers.** The rail was collapsing
  at 1099px instead of 1439px; the inspector was just narrowing in place at 768-1099px
  instead of becoming an overlay drawer. Corrected both, and made the inspector a real
  `position: absolute` drawer within `.mission-shell-body` (now `position: relative`) at
  that tier. The plan table's redundant Actions-count column (the full action list has
  its own dedicated `PlanActionSequence` view) is hidden at ≤1439px — deliberately not
  any safety-relevant column (exposure reduction, pressure violations, service
  availability, status all stay visible at every width), consistent with this session's
  standing rule against ever suppressing safety-relevant data for cosmetic reasons.
- **Real horizontal page-overflow bug at narrow widths**, found only by measuring in a
  real browser, not by reading the CSS: the mission header's status badge row
  (`flex: 0 0 auto`, no wrap) forced the whole document wider than the viewport once the
  badges stopped fitting next to the brand/incident context. Confirmed programmatically
  before the fix (`document.documentElement.scrollWidth` 643px vs `clientWidth` 600px at
  a 600px viewport) and after (600px/600px, exact match). Fixed by letting the badge row
  shrink and scroll horizontally within its own bounded, hidden-scrollbar container
  instead of forcing page-level scroll or hiding any individual badge.
- **ECharts animations ignored the app's `reducedMotion` toggle.** The existing
  `.reduced-motion` CSS class and `prefers-reduced-motion` media query (both already
  present since UI-1) only govern CSS transitions/animations — ECharts renders to canvas
  and animates independently of CSS, so `HydraulicChart` and `ParetoFrontier`'s
  exposure-aware scatter kept animating regardless of the toggle. Threaded
  `animation: !reducedMotion` into both `setOption` calls.

**Verified with a real Playwright/Chromium pass**, per this session's standing rule for
any phase touching layout/canvas: four viewports (1920, 1300, 900, 600px) screenshotted
to confirm each responsive tier renders as intended, plus a keyboard-only navigation run
(Tab into the rail, Enter to activate a stage, confirmed the Source workspace rendered
with real data) — not just the automated jsdom-based gate, which can't see real layout or
real async focus/scroll behavior (the same category of gap that caught real bugs in UI-1
and UI-2).

Added regression coverage: `ReducedMotionCharts.test.tsx` (2 tests, asserting the real
`animation` flag ECharts receives under each toggle state — required upgrading the shared
`echarts/core` test mock's `init` to a real `vi.fn()` in `tests/setup.ts` so its calls
become inspectable, since it was previously a plain arrow function with no call history),
plus an `App.test.tsx` assertion for the skip-link fix's precondition.

Gate: tsc/eslint/prettier clean, 84/84 tests (15 files, up from 81/14), production build
succeeds (chunk sizes unchanged from UI-9 — no new dependencies this phase). No backend
files changed.

**Deliberately not changed**: the plan table's other columns, the dock's bottom-drawer
positioning (already full-width/tabbed/collapsible below `.mission-shell-body` in the
column flex layout, already satisfying "tabbed bottom drawer" with no code change
needed), and the `.status-icon` shapes (audited and found already shape-distinct per tone
— filled circle/filled square/filled diamond/hollow ring — not the color-only gap it
initially looked like from the token list alone; a proposed "fix" here was written, found
to be a no-op against the real CSS, and reverted rather than left in as dead code).

## UI-10.5 — visual consolidation before final QA: COMPLETE

Spec: `/workspace/ui-work2.txt`. This phase was inserted between UI-10 and UI-11 by a
newer session prompt; it does not appear in ui-work.txt §31's own phase list.

Commit: `a73d034` — `fix(ui): consolidate mission-control visual hierarchy before final qa`
(6 files changed, 356 insertions / 361 deletions: `frontend/src/pages/Overview.tsx`,
`frontend/src/shell/DecisionInspector.tsx`, `frontend/src/styles.css`,
`frontend/src/workspaces/ResponseWorkspace.tsx`, `frontend/src/workspaces/SourceWorkspace.tsx`,
`frontend/tests/App.test.tsx`). Not yet pushed to `origin` — see "GitHub push" note below.

### Pre-existing uncommitted work found at session start

Before touching anything, `git status` showed uncommitted changes to
`frontend/tests/e2e/operator.spec.ts`, `frontend/tests/e2e/visual-regression.spec.ts`, and
two of that spec's PNG snapshots, plus two untracked ad-hoc smoke scripts
(`scripts_scratch_smoke.py`, `frontend/screenshot_live_smoke.mjs`). These were real UI-11
work-in-progress (a genuine Playwright rewrite of the "normal demo flow" from ui-work.txt
§32, predating ui-work2.txt's insertion of this UI-10.5 phase) — not something to discard.
Since ui-work2.txt §14 explicitly forbids touching visual-regression baselines in this
phase ("Do NOT yet regenerate/approve the final Playwright visual regression baselines"),
and this phase's CSS/layout changes would have made those in-progress baselines stale
immediately anyway, it was stashed rather than committed or discarded:

```
git stash list
# stash@{0}: On feature/ui-mission-control-v1: pre-existing uncommitted UI-11 work
#   (operator.spec.ts, visual-regression.spec.ts, snapshots) found at UI-10.5 session start
```

**Before starting UI-11, run `git stash pop` first** — that work is a real head start on
UI-11's own scope (turning ad-hoc screenshot scripts into a committed E2E suite) and should
be reviewed/continued, not redone from scratch. The two ad-hoc scratch scripts were moved to
this session's scratchpad directory (outside the repo) rather than committed or deleted.

### Real visual inspection (before editing anything)

Built the frontend and ran it under real Chromium via `vite preview` (not `vite build`
alone) with `VITE_API_BASE` pointed at a deliberately-unreachable port so the app takes its
real network-failure→DEMO_FALLBACK code path deterministically, independent of whatever
backend process happens to be running in this shared sandbox (one was: a leftover `uvicorn
hydroswarm.api.app:app` on 127.0.0.1:8765 from an earlier session, still running throughout
this one — left untouched). This surfaced two real, previously-undetected findings no CSS
reading or jsdom test could have caught:

1. **A real crash**: `Overview.tsx` (the Incident workspace) unconditionally read
   `incident.candidates[0].nodeId`. With a real backend reachable but no `VITE_INCIDENT_ID`
   configured, the app correctly enters ERROR mode with genuinely empty `candidates: []`
   (per `errorIncidentView`'s "never a fabricated value" contract) — and Overview crashed
   on that combination, producing a blank page instead of the intended ERROR UI. Fixed by
   guarding on `leading` throughout (see commit).
2. **The visual problems ui-work2.txt predicted**: at 1366x768 with the dock at its 240px
   default, the map was clipped to a small sliver; Source and Response each stacked a local
   `.right-rail` beside the map that duplicated fields already in the global
   `DecisionInspector` at equal prominence (e.g. Source's "Calibrated candidate set" panel
   vs. the inspector's identical candidate-set-size/conformal-target/calibration block).

### What changed (ui-work2.txt §2–§8)

- **Source / Response local `.right-rail` removed.** Both workspaces now stack the map
  (full-width, `wide-panel`) above their full-detail panels instead of a 2-column
  map+sidebar grid. `DecisionInspector.tsx`'s `SourceSummary` gained a compact
  authority/applicability badge pair (queries the same `['authority', incident.id]`
  TanStack Query key the workspace already populates — confirmed no double-fetch, this
  dedup pattern was already used between Source/Response before this phase) and the
  previously-missing "held-out measured coverage" field (ui-work.txt §13.2 requires it;
  the old local rail had it, the inspector never did — this phase's rail removal would
  have silently dropped it from the Source workspace entirely had it not been added here).
  `ResponseSummary` gained the selected plan's decision/CURRENT-STALE badges, simulator
  identity, pressure/service margins, numerical sensitivity, and rejection/abstention
  reason. Full hash-level provenance and worst-case consequences stay PRIMARY in
  `TechnicalDock > Verification`, which already covered every one of those fields.
- **Incident workspace (`Overview.tsx`) simplified.** Removed: the full `PlanTable`, the
  full `Counterfactuals` grid, the long "Verified explanation" section with 4
  always-disabled "not yet connected to the live API" buttons (stale — Approval (UI-6) and
  real grounded explanations (UI-3/UI-5) already exist; these were never updated), and the
  per-sensor health list (already color/shape-coded on the map, per ui-work.txt §11).
  Replaced with: a trimmed compact strip, a full-width dominant map, and three small
  Source/Sampling/Response panels that navigate to their dedicated workspace via real
  `setWorkspace()` calls (Zustand) instead of repeating that workspace's content. The two
  disabled fake buttons ("Review sample request", "Review … approval") are now real
  navigation to Sampling/Approval respectively — the first genuine fix of the "wire or
  remove nonfunctional controls" gap ui-work.txt §3.4 named, which had survived UI-1..UI-10
  untouched because Overview.tsx was never revisited after UI-0.
- **`Counterfactuals` moved from Overview to `ResponseWorkspace`** (its natural home — a
  no-response plan-comparison view, not incident-overview content) rather than deleted.
  Its eyebrow was `"SYNCHRONIZED AT 08:40"`, a literal hard-coded string with no data
  binding — found while moving it; replaced with `"NO-RESPONSE COMPARATOR"`.
- **Map sizing.** `.map-shell` was `height: 530px` unconditionally. Now
  `clamp(420px, 62vh, 720px)` — substantially larger at 1920x1080 (the phase's explicit
  hard acceptance target per ui-work2.txt §4) with the dock open at its 240px default,
  confirmed via real screenshots, not computed. At 1366x768 the map still requires scrolling
  to see in full — this is arithmetic, not a missed optimization: header(52) + mode
  banner(~34) + toolbar(38) + dock(240, must not shrink below default per explicit
  instruction) + footer(~30) already consumes 394px of 768 before any workspace content, and
  ui-work2.txt §4's hard acceptance bar is stated only for 1920x1080. Not a regression
  either way — the old fixed 530px map didn't fit at 768px height any better.
- **Legacy gradient/radius flattening.** `.panel` (was a 145° diagonal gradient, 7px
  radius) and `.decision-banner` (was a 100° gradient) now use flat
  `var(--panel)`/`var(--panel-raised)` and the `--shell-panel-radius` token (5px). The
  decision-banner's `<h1>` (global scale up to 2.15rem/34px, sized for a full-page
  document title, shared with Validation/Benchmark page headings) was oversized for a
  "compact incident strip" and caused 2-line wrapping at 1440px width; scoped to
  `.decision-banner h1 { font-size: 1.2rem }` rather than changing the shared global rule.

### Explicitly NOT changed (per ui-work2.txt §9 and general scope discipline)

- No change to `OperationalMap.tsx`'s per-workspace mount architecture (each of
  Incident/Source/Response still mounts its own `<OperationalMap>` instance). Verified via
  real navigation clicks between workspaces that this doesn't produce visible flashing or
  lost selection state beyond what already existed pre-phase; ui-work2.txt §9 explicitly
  forbids the larger single-persistent-map refactor unless real testing shows a problem,
  and none was found.
- Sampling, Approval, and Replay workspaces' local `.right-rail` usage — ui-work2.txt §2
  names Source and Response specifically ("especially Source and Response"); Sampling's
  local rail content (Stop Certificate, Sample budget) doesn't duplicate the global
  inspector's sparser summary at equal prominence, and Approval has no map to free space
  beside.
- No backend files touched (0 files under `src/hydroswarm/**` this phase, as in every
  phase this session).

### Verification

`npm run lint && npm run typecheck && npm run format:check && npm test -- --run && npm run build`
all pass (84/84 tests — 3 assertions in `App.test.tsx` updated because they depended on
content intentionally relocated to a different workspace, or now legitimately duplicated
by design between the compact inspector and the full workspace panel; see the commit body
for the exact list). Then real Chromium/Vite-preview screenshots (DEMO_FALLBACK, forced via
an unreachable `VITE_API_BASE` as described above) across:

- Incident/Source/Sampling/Response/Approval @ 1920×1080
- Incident/Response @ 1440×900, Source @ 1440×900
- Incident @ 1366×768
- Rail + inspector + dock all collapsed simultaneously (Response)
- Selecting a second plan in the plan table (confirms toolbar breadcrumb / inspector /
  Pareto-frontier highlight all stay synchronized — ui-work.txt §22, unaffected by this
  phase's refactor)
- The ERROR-mode crash fix (`?failure=no_valid_plan` with no real backend reachable)

Zero console/page errors beyond the deliberately-unreachable API port's network failure in
every case. No horizontal overflow observed at any tested width. No panel overlap, no
duplicated second right sidebar, no unusable map height at 1920×1080.

**Confirmed**: no backend files changed; frozen model/calibration identities untouched (no
model/calibration/checkpoint files or config touched); locked final test remains unopened;
`reports/results/v3/final-selection.json` was not created; the architecture-freeze tag was
not moved.

### GitHub push

This session did not have working GitHub authentication (unlike the session that produced
the earlier UI-0..UI-10 commits, which explicitly notes "GitHub auth is working this
session" above) — push was not attempted since no push command exists yet in this
transcript to report success or failure against; the commit exists locally on
`feature/ui-mission-control-v1` at `a73d034`. **Next session: attempt
`git push origin feature/ui-mission-control-v1` and update this section with the result;
if it fails, continue locally per the standing instruction to not block on GitHub auth
errors.**

## Remaining phases (UI-11): NOT STARTED

Tracked in ui-work.txt §31 order (UI-10.5 above was inserted ahead of it by ui-work2.txt,
not part of that original ordering).

Next up: UI-11 (demo hardening / final QA). **Start with `git stash pop`** to recover the
in-progress Playwright E2E work described above. Per ui-work.txt's own UI-11 scope: lock
down the deterministic demo flow, finish turning the ad-hoc screenshot scripts used
throughout this session into a real committed Playwright E2E suite, confirm zero console
errors and zero external runtime calls in the production build (ui-work.txt §26's "no
Google Fonts / Mapbox cloud styles / CDNs / remote icon packs / OpenAI / Anthropic /
internet-hosted scripts" — worth an explicit grep-based check of `dist/` and `index.html`,
not just an assumption), regenerate/approve the Playwright visual-regression baselines now
that UI-10.5 has stabilized the layout (ui-work2.txt §14 deferred this here explicitly), do
an integrated backend smoke test if a real backend can be started in this environment
(confirm first whether WNTR/EPANET and the Python venv are available here before assuming —
note a `uvicorn hydroswarm.api.app:app` process was observed already running on
127.0.0.1:8765 in this sandbox during UI-10.5, left untouched), and finalize docs. This is
the last phase per ui-work.txt §31's own ordering.

## Continuation commands

```
cd /workspace/HydroSwarm
git checkout feature/ui-mission-control-v1
git stash pop   # recovers in-progress UI-11 Playwright work found before UI-10.5
cd frontend
npm install   # if node_modules is missing (ephemeral sandbox)
npm run lint && npm run typecheck && npm run format:check && npm test -- --run && npm run build
```

No long-running/background jobs are active from this session as of this report.
