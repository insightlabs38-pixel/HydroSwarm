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
`frontend/tests/App.test.tsx`). Pushed to `origin` — see "GitHub push" note below.

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

`git push origin feature/ui-mission-control-v1` succeeded this session
(`20ae245..ed942d9  feature/ui-mission-control-v1 -> feature/ui-mission-control-v1`),
carrying both the UI-10.5 commit (`a73d034`) and this handoff-report commit (`ed942d9`) to
`origin`.

## Pre-UI-11 fixes (workflow progression + simulator budget): COMPLETE

Two operator-requested fixes, landed before UI-11 as instructed.

Commit: `2e299ea` — `fix(ui): align workflow progression and expose simulator budget`
(14 files: `frontend/src/workflow.ts` new, `frontend/tests/workflow.test.ts` new, plus
`Overview.tsx`/`DecisionInspector.tsx`/`WorkflowRail.tsx`/`types.ts`/`demoFixture.ts`/
`api/incident.ts` on the frontend and `api/app.py`/`api/state.py` +
`tests/integration/test_incident_view_contract.py` on the backend). Pushed.

**A. Workflow progression.** `Overview.tsx` and `DecisionInspector.tsx` each
independently derived "next step" from retained artifacts (`recommendedSample`,
`plans`, `selectedPlanId`) instead of the authoritative `incident.status`, and
`WorkflowRail` derived rail-item status the same way — these could disagree, and the
shipped demo fixture is exactly the reported contradiction (status=APPROVAL,
approvalPending=true, a still-populated `recommendedSample` left over from the earlier
SAMPLING stage): both surfaces said "Collect recommended sample" while a plan was
actually awaiting human approval. Added `frontend/src/workflow.ts`
(`deriveWorkflowProgression`), the one shared authority on `incident.status` for
Source/Sampling/Response/Approval rail status and next-step text, per the operator's
exact SAMPLING/PLANNING/APPROVAL/CLOSED table. All three call sites now use it. 9 new
unit tests cover all four controller states, ERROR mode, the `noValidPlan`-blocks-
response case, and the exact reported contradiction scenario.

**B. Simulator budget.** `IncidentState` already tracked `exact_simulations_used`,
`plans_exactly_verified`, `exact_simulation_cache_hits`, `remaining_epanet_budget`;
`/replay` already exposed them (embeds the full `IncidentState`), but the normal live
`/view` response did not. Added `SimulatorBudgetView` (`src/hydroswarm/api/state.py`) as
a new required field on `IncidentView`, populated in `get_incident_view()` directly from
`record.state` — no recomputation, no altered budget semantics, purely additive
passthrough. Extended the existing `/view` contract test with budget assertions
cross-checked directly against `GET /api/incidents/{id}`'s authoritative `IncidentState`.
Frontend: matching `SimulatorBudget` type + `simulatorBudget` field on `IncidentView`
(null only in ERROR mode; DEMO_FALLBACK carries a labeled fixture value like every other
field on that fixture), a typed adapter, and compact (not KPI-card) rendering — one
"Remaining exact simulation budget" line on the Incident inspector, the full four-field
breakdown on the Response inspector.

Gates at this commit: frontend eslint/tsc/prettier clean, 93/93 vitest (up from 84);
backend ruff/pyright clean on touched files, **full pytest suite 874/874 passing**
(unchanged count from the pre-existing baseline — zero regressions), full-tree pyright
clean.

## UI-11 — final demo hardening / QA: COMPLETE

Commit: `dab17d0` — `test(ui): lock mission-control demo and interaction regression
suite` (18 files: `operator.spec.ts`, `visual-regression.spec.ts`, 9 new baseline PNGs
+ 2 stale ones removed, `api/incident.ts`, `geometry.ts` new, `geometry.test.ts` new,
`api-incident.test.ts`, `index.html`). Pushed.

### Recovering the stash

`git stash show -p stash@{0}` (from the UI-10.5 handoff above) was inspected before
writing anything new. Its test *logic* was real, valuable UI-11 work-in-progress
(a genuine Playwright rewrite of the ui-work.txt §32 demo flow); its *screenshots* were
explicitly not reused (`git stash pop` brought them back, but they were immediately
superseded by fresh baselines — see below). Two assertions in the recovered
`operator.spec.ts` were stale against the current UI and fixed:

- `getByRole('heading', { name: 'Source candidates' })` — Overview's panel is now
  titled "Source" ("UI-10.5" 3.C condensed it).
- `getByLabel('Calibrated candidate set')` — that panel no longer exists in the Source
  workspace body; its content moved into the global Decision Inspector ("UI-10.5" 2).
  Rewritten to query `getByRole('complementary', { name: 'Decision inspector' })`.

`visual-regression.spec.ts`'s `waitForOverviewLoaded()` helper waited for
`.table-plan-button`, which no longer exists on Overview (the plan table moved to
Response in "UI-10.5") — fixed to wait for the map canvas instead. The
"selected-plan synchronization" test moved from Overview to Response for the same
reason, and was extended to prove the toolbar breadcrumb, Decision Inspector, and
Verification dock tab all update together on plan selection (ui-work.txt §22), not just
the table's own highlighted row.

### New E2E scenarios

Two new deterministic DEMO_FALLBACK variants added to `api/incident.ts`
(`?demo=ood_suppressed`, `?demo=stale_verification`, mirroring the existing
`?failure=<category>` mechanism) so these real, already-wired governed states are
reachable end-to-end without a live backend — every field they set is a real,
already-typed `IncidentView` field, no new fabricated shape:

- **OOD / suppressed planning**: `MissionHeader`'s "OOD OUTSIDE_VALIDATED_RANGE" badge
  and "DEGRADED" readiness render; the real plan-comparison table shows zero rows while
  planning is suppressed (found and had to explicitly scope around: the Pareto frontier
  panel below it renders its own separate, mode-keyed illustrative dataset unrelated to
  this incident's real plans — a pre-existing, deliberate demo-fixture design, not a bug
  introduced here).
- **Stale verification blocks approval**: the "Verification is stale." empty-state
  renders, and the approval hierarchy ladder never advances past "Simulator verified" —
  proves `ApprovalWorkspace`'s existing `canReview = isVerified && isCurrent` gate
  actually holds against a real STALE plan, without touching that gate.

### Fresh visual baselines (all 9 required views)

`incident-1920x1080`, `source-1920x1080`, `sampling-1920x1080`, `response-1920x1080`,
`approval-1920x1080`, `replay-1920x1080`, `incident-1440x900`, `response-1440x900`,
`incident-1366x768` — all regenerated from the current UI (`--update-snapshots`), not
carried over from the stash. The two stale `overview-*` baselines from before the
Overview→Incident naming were deleted.

### Offline audit

Grepped the production `dist/` bundle (`assets/*.js`, `assets/*.css`, `index.html`) for
Google Fonts, Mapbox cloud styles, any CDN host, remote icon packs, OpenAI/Anthropic,
analytics/telemetry SDKs — zero matches except three inert strings inside vendored
library code (a MapLibre error message referencing a GitHub issue URL, MapLibre's
default attribution HTML — dead code since `attributionControl: false` is set, and
React's own production-error-decoder URL — never auto-fetched, only ever a clickable
link if a React invariant fails). Then, not trusting a static grep alone, ran a real
Playwright pass against a built `vite preview` server capturing every actual network
`request` event while visiting every workspace: the only non-origin URL seen was a
`blob:` object URL (MapLibre's own in-browser worker script — not a network call).
Clean.

### Skip-link duplicate (found during hardening, fixed)

`index.html` carried a static `<a class="skip-link" href="#main-content">Skip to
incident workspace</a>`, outside React's `#root`, left over from the pre-mission-control
dashboard. `App.tsx` grew its own real, tested skip-link ("Skip to main content") in
UI-1 and nothing ever removed the old one — both existed simultaneously in the live DOM
the whole time since UI-1, with the stale static one first in tab order and no working
focus target of its own. Removed; added a regression test asserting exactly one
`.skip-link` element exists.

### Integrated backend/frontend smoke test

A real `uvicorn hydroswarm.api.app:app` instance was started against the actual
promoted checkpoint (`models/hydrocore-s-learning-v1.safetensors`,
sha256 `85715fbe061a30131b39b717137d2522c3870d674d262f4717ef7541731d5423` — unchanged
from before this session). Imported `data/frozen/golden_network.inp`, created and
`/analyze`d a real incident (id `a553054d-963a-4f81-843a-361384fc08e2`), confirmed:

- `GET /view` returns `data_mode: LIVE`, a real `controller_state`, and the new
  `simulator_budget` field populated with real (all-zero, honestly — this network's
  calibration doesn't validate, so `/plans/generate` correctly 409s
  `PLANNING_SUPPRESSED`/`CALIBRATION_INVALID_OR_MISSING` before any simulation runs)
  values, not fabricated ones.
- `GET /authority` (3 certificates), `GET /evidence-certificate`, and
  `GET /frontier` all respond 200 with real content.
- The locked final evaluation was never opened.

Then pointed a real production build at this live incident (`VITE_INCIDENT_ID` set,
default `VITE_API_BASE` relative to the `vite preview` proxy) and loaded it in real
Chromium: header shows a real "LIVE" badge (no DEMO_FALLBACK banner), the Decision
Inspector echoes `LIVE`, the workflow-progression fix renders correctly on genuinely
live data ("next: Continue evidence collection" while `status: SAMPLING`, matching Part
A's table), and the simulator-budget fix renders correctly on genuinely live data
("Remaining exact simulation budget: 3").

**This smoke test found a real, previously-unknown bug**: the page threw
`pageerror: Invalid LngLat latitude value: must be between -90 and 90` on load.
`golden_network.inp`'s raw EPANET coordinates (arbitrary engineering units, Y up to
1450 — typical for a hand-designed test network with no real-world georeference) were
being passed straight through to MapLibre as `[lng, lat]`, which requires latitude in
`[-90, 90]` and throws synchronously otherwise. Any real backend incident built on a
network like that would have crashed the map. Added `frontend/src/geometry.ts`
(`normalizeMapCoordinates`): coordinates already within a safe geographic range pass
through unchanged (never distorts genuinely georeferenced data — confirmed against the
demoFixture's real `-80.xx, 35.xx` North Carolina coordinates); out-of-range local
coordinates are rescaled together across the whole network with one uniform scale
factor for both axes, so real relative shape/angles are preserved, into a small
MapLibre-safe window. Frontend-only rendering-geometry fix — no backend, model, or
scientific-authority code touched. Rebuilt and reloaded against the same live incident
afterward: zero console/page errors, map renders the real (now-valid-range) network
topology correctly. 6 new unit tests lock the fix down (real vs. out-of-range input,
shape-preservation via uniform scale, single-node/empty-input edge cases).

Backend and preview processes were stopped after the smoke test completed.

### Final gates

Frontend: `npm run lint`, `npm run typecheck`, `npm run format:check` all clean;
`npm test -- --run` → **102/102** (up from 84 at session start: +9 workflow, +6
geometry, +3 demo-variant, +4 assorted assertions across existing files); `npm run
build` succeeds; **Playwright E2E → 26/26** (`npx playwright test tests/e2e/`), zero
unexpected console errors anywhere in this session's real-browser runs (informational
`net::ERR_UNSAFE_PORT`/`ECONNREFUSED` lines from the deliberately-unreachable API used
to force DEMO_FALLBACK for screenshotting are expected, not unexpected).

Backend: ruff and full-tree pyright re-confirmed clean at the end of this session; the
**full pytest suite (874/874)** was run once, at the Part A/B commit, since Part A/B was
the only backend-touching change this session (Part C/UI-11 touched zero backend files
— confirmed via `git status` before committing) and nothing backend-side changed after
that run. Did not attempt to replicate CI-only jobs unrelated to this change (Windows
portability, Git-LFS hydration) — out of scope for a frontend-focused change with zero
backend diff in this commit.

### Safety/governance invariants (re-verified at session end)

- `git tag -l | grep freeze` → `hydrocore-v4-architecture-freeze`, still pointing at
  `fcd2011...` (last moved 2026-08-09 01:04:28, before this session started) — untouched.
- `models/hydrocore-s-learning-v1.safetensors` sha256 unchanged (`85715fbe...`, verified
  both via `git status`/`git diff --stat` showing no changes and by direct `sha256sum`,
  and cross-checked against the live `/view` response's `provenance.model_checkpoint_hash`
  during the smoke test).
- `find . -iname "final-selection.json"` → no matches. Still absent.
- Locked final evaluation: not opened this session (confirmed no relevant script/command
  was ever run against it).
- No `src/hydroswarm/**` files changed in the UI-11 commit; the two backend files changed
  in the pre-UI-11 commit (`api/app.py`, `api/state.py`) are additive-only (one new typed
  response field, populated from already-computed state) and fully gated (874/874
  pytest).

## UI-11.1 — final bounded submission-integrity pass: COMPLETE

A separate, later session. The visual architecture was frozen going in (no redesign);
this pass fixed remaining runtime/data-integrity/demo/CI issues only.

- **§1 — frozen V4 wired into production**: `hydroswarm.api.app`'s module-level `app`
  and `hydroswarm.cli.run_self_test` composed `DefaultPipelineFactory` (legacy
  `hydrocore-v3`/`models/hydrocore-s-learning-v1.safetensors`) instead of the frozen
  `hydrocore-v4` candidate, despite `V4PipelineFactory` already existing. Fixed by
  copying the already-built, hash-verified release bundle from
  `experiments/runs/v4-release-bundle/no_adapters-seed20260810/` into a git-tracked
  `models/hydrocore-v4-release/` and switching exactly the two call sites
  (`app.py`, `cli.py`) to `V4PipelineFactory(...)` — no V4 loading logic reimplemented.
  New `tests/integration/test_production_runtime_wiring.py` (7 tests) locks the frozen
  identity, output governance, fail-closed behavior, and exact-WNTR-verification
  decoupling down; a real end-to-end incident analysis and a real
  `./start_hydroswarm.sh` + live-Chromium smoke both confirmed the production app
  actually serves `model_sha256 a501ad87bc...16c7` / `calibration_hash 829c167b26...68fa`.
- **§2 — null-vs-0 in the frontend**: sensor pressure/concentration, node concentration,
  and fusion disagreement were still `?? 0`-coerced (unmeasured silently became a
  scientifically meaningful 0). Made `number | null` end-to-end, rendered as "not
  measured", with regression coverage distinguishing null from genuine 0 at both the
  mapping and rendered-DOM layers.
- **§3 — suppressed-planning Pareto/plan-comparison leak**: `?demo=ood_suppressed`
  correctly set `plans: []`, but the Pareto frontier and "compare plans" panels still
  drew `demoParetoFrontier`/a stale explanation authored for the populated base
  scenario. All three panels now key off `plans.length === 0` and show one governed
  suppressed state; Playwright asserts both an empty plan table and an empty frontier.
- **§4 — demo fixture provenance honesty**: `demoFixture.ts` carried a fabricated model
  name (`HydroSwarm-M 0.9.2`) and hand-authored 64-hex-character strings shaped like
  real SHA-256 hashes. Replaced with honest empty/unavailable values; the one
  genuinely real value (`calibrationVersion: hydroswarm-calibration-v1`, the real
  `CALIBRATION_SCHEMA_VERSION` constant) was kept, not removed.
- **§5 — Playwright in CI**: added a dedicated `frontend-e2e` ubuntu-latest job
  (Chromium install, build, `npm run test:e2e`), separate from `frontend-quality`. The
  Linux Chromium visual baselines had never actually run in CI before this.
- **§6 — final regression gates**: backend ruff/pyright clean, full pytest 881/881;
  frontend lint/typecheck/format/build clean, 107/107 unit tests, 26/26 Playwright;
  real documented-launch smoke test (network import, incident create/analyze, real
  EPANET map coordinates, simulator-budget fields, zero console errors, zero external
  network calls, genuine unforced planning-suppression governance observed twice).
- **§7 — real PR + CI**: opened PR #2 (`feature/ui-mission-control-v1` → `main`, not
  merged). First real CI run caught a genuine environment-fidelity issue in the new
  `frontend-e2e` job: baselines regenerated in this session's local sandbox did not
  pixel-match GitHub Actions' actual Ubuntu/Chromium font rendering (a whole-page
  sub-pixel "ghosting" diff, not a content regression). Fixed by regenerating the 9
  affected baselines from inside the real CI environment itself (a throwaway
  push-triggered workflow ran `playwright test --update-snapshots` on `ubuntu-latest`
  and uploaded the result; downloaded, committed, workflow deleted) — never by loosening
  the pixel tolerance. All 4 required checks (`frontend-quality`, `frontend-e2e`,
  `python-quality` Ubuntu + Windows) passed on the next real run.
- **§8 — governance invariants re-confirmed**: `hydrocore-v4-architecture-freeze` tag
  unmoved and still an ancestor of HEAD; `architecture-freeze.json`'s
  `locked_test_opened: false`, `final_selection_json_exists: false`,
  `locked_evaluation_status: "NOT PERFORMED"` all unchanged (file untouched by this
  branch); no `final-selection.json` anywhere in the tree.

Commits (`792afac`..`e1e9d78`): `792afac` feat(runtime) wire V4 · `099ce06` §2 null-vs-0
· `a4cf0ce` §3 suppressed-Pareto fix · `e4b662d` §4 demo provenance honesty · `937b555`
§5 CI Playwright job · `877f312`/`23c020c` temp baseline-regen tooling (deleted) ·
`e1e9d78` real-CI baseline fix.

## Pre-merge polish pass: COMPLETE

A final bounded pass requested ahead of merge — no redesign, no backend/model/scientific
behavior change:

1. **README screenshot**: `docs/screenshots/operator-overview.png` was from a pre-
   "UI-10.5" layout (old right rail, `HydroSwarm-M 0.9.2` header text). Replaced with a
   real 1920×1080 Incident-workspace capture from this branch's current build; caption
   wording updated to match.
2. **DEMO_FALLBACK semantics**: `ModeBanner`'s label changed from "DETERMINISTIC DEMO
   FALLBACK" to "ILLUSTRATIVE DEMO / DEMO_FALLBACK"; `demoFixture.ts`'s
   `runtimeAnalysisMode` (`'FULL_HYBRID'` → `null`) and `runtimeMs` (`438` → `0`) no
   longer imply a real inference run occurred or a real latency was measured — both
   reuse this codebase's existing "0/null means don't render the badge" convention
   (`MissionHeader`), not a new one; `modeReason` now says "hand-authored", not
   "simulator-derived". All 14 test references to the old banner string updated.
3. **Stale docs**: `hydroswarm.runtime.v4_defaults`'s module docstring said "nothing in
   this module is wired into the live production entry point" — written before UI-11.1
   §1 wired it in. Updated to state current reality while preserving the still-true
   distinction from the separate, still-unopened locked test. This handoff doc and PR
   #2's description updated to the current final SHA/CI-green state (this section).

9 Playwright baselines affected by the visible banner/badge text changes regenerated
from real CI again, same throwaway-workflow method as UI-11.1 §7. Final local gates:
frontend lint/typecheck/format/build clean, 107/107 unit tests; backend ruff/pyright
clean, the 14 tests touching `v4_defaults`/production wiring re-run and green. Real PR
CI (`frontend-quality`, `frontend-e2e`, `python-quality` Ubuntu + Windows) green again
on the final pushed SHA.

## Final summary

- **Branch**: `feature/ui-mission-control-v1`
- **Final SHA**: `a5c7cbcb50fd7edb0070b1a1290f0bbd00d5a096`
- **PR**: [#2](https://github.com/insightlabs38-pixel/HydroSwarm/pull/2)
  (`feature/ui-mission-control-v1` → `main`), open, mergeable, all 4 required GitHub
  Actions checks passing on the final SHA above. **Not merged** (never attempted, per
  every phase's own instruction not to merge automatically).
- **This session's commits** (chronological, all pushed to `origin`):
  1. `a73d034` — `fix(ui): consolidate mission-control visual hierarchy before final qa` (UI-10.5)
  2. `ed942d9` — `docs(handoff): record UI-10.5 completion and stashed UI-11 work-in-progress`
  3. `63eba84` — `docs(handoff): record successful GitHub push for UI-10.5`
  4. `2e299ea` — `fix(ui): align workflow progression and expose simulator budget` (pre-UI-11 fixes A+B)
  5. `dab17d0` — `test(ui): lock mission-control demo and interaction regression suite` (UI-11)
  6. `792afac` — `feat(runtime): serve the frozen HydroCore-v4 architecture by default` (UI-11.1 §1)
  7. `099ce06` — `UI-11.1 §2: stop fabricating 0 for unmeasured sensor/node/disagreement data`
  8. `a4cf0ce` — `UI-11.1 §3: stop showing an illustrative Pareto frontier when planning is suppressed`
  9. `e4b662d` — `UI-11.1 §4: stop the demo fixture's provenance from looking like a real identity`
  10. `937b555` — `UI-11.1 §5: add a dedicated Playwright/E2E job to CI`
  11. `e1e9d78` — `fix(ci): regenerate visual baselines from the real GitHub Actions environment`
  12. `4cdd6b0` — `chore(polish): pre-merge README screenshot, DEMO_FALLBACK semantics, and stale docs`
  13. `a5c7cbc` — `fix(ci): regenerate Playwright baselines for the DEMO_FALLBACK polish pass`

UI phase (UI-0 through UI-11.1, plus this pre-merge polish pass) is complete per
ui-work.txt §31's own ordering, the operator-inserted UI-10.5/pre-UI-11 fixes, and the
two later bounded integrity/polish passes. PR #2 is open with real, currently-green
GitHub Actions CI. Recommend the next step be human review of the PR and, if approved,
merging it — not something any session should do unprompted.

## Continuation commands

```
cd /workspace/HydroSwarm
git checkout feature/ui-mission-control-v1
git pull
cd frontend
npm install   # if node_modules is missing (ephemeral sandbox)
npm run lint && npm run typecheck && npm run format:check && npm test -- --run && npm run build
npx playwright install --with-deps chromium   # if not already installed
npx playwright test tests/e2e/
```

To re-run the integrated backend smoke test:

```
cd /workspace/HydroSwarm
source .venv/bin/activate
uvicorn hydroswarm.api.app:app --host 127.0.0.1 --port 8765 &
# then POST /api/networks/import (data/frozen/golden_network.inp), POST /api/incidents,
# POST /api/incidents/{id}/analyze, GET /api/incidents/{id}/view
```

No long-running/background jobs are active from this session as of this report.
