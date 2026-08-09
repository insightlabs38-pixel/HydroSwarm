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

Pushed to `origin/feature/ui-mission-control-v1` after each commit (GitHub auth is
working this session).

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

## Remaining phases (UI-10, UI-11): NOT STARTED

Tracked as tasks #12–#13 in this session's task list, in ui-work.txt §31 order. Each
phase's own gate must pass and get its own commit before moving to the next, per
ui-work.txt §36 ("Do not implement all phases in one mega-change"). No shortcuts,
placeholders, or "TODO" UI states are to be committed as if finished.

Next up: UI-10 (accessibility, responsive layout, and performance hardening). Axe checks
already exist per-workspace (all passing, zero violations) from UI-1/UI-7/UI-9, so UI-10's
real remaining scope is: a full keyboard-only navigation pass across the shell (rail →
toolbar → workspace → inspector → dock, including the map's click-to-select and the
approval form's required checkbox), responsive behavior below the desktop breakpoint the
shell currently assumes (ui-work.txt's layout spec is desktop-first; verify or explicitly
scope-note what happens narrower), and a performance pass on the two largest chunks
(`OperationalMap` at ~1.06 MB / 287 kB gzip, `index` at ~819 kB / 270 kB gzip) — confirm
whether further code-splitting is warranted or whether current lazy-loading already keeps
initial paint reasonable. Then UI-11 (demo hardening / final QA, including turning the
manual Playwright screenshot scripts used ad hoc all session into a real committed E2E
suite, per the plan's own UI-11 scope).

## Continuation commands

```
cd /workspace/HydroSwarm
git checkout feature/ui-mission-control-v1
cd frontend
npm install   # if node_modules is missing (ephemeral sandbox)
npm run lint && npm run typecheck && npm test -- --run && npm run build
```

No long-running/background jobs are active from this session as of this report.
