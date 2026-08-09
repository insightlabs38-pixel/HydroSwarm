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

## Remaining phases (UI-3 through UI-11): NOT STARTED

Tracked as tasks #5–#13 in this session's task list, in ui-work.txt §31 order. Each
phase's own gate must pass and get its own commit before moving to the next, per
ui-work.txt §36 ("Do not implement all phases in one mega-change"). No shortcuts,
placeholders, or "TODO" UI states are to be committed as if finished.

Next up: UI-3 (Source/Sentinel workspace) — candidate inspector, calibration/coverage,
disagreement, OOD, wiring the already-confirmed-present `GET /api/incidents/{id}/authority`
endpoint (see "Backend API surface" above — `DecisionCertificate`/`AuthorityLevel`/
`ApplicabilityStatus`), and a real grounded "Why this source?" via
`GET /api/incidents/{id}/explanations/WHY_SOURCE`. This is the first phase that needs a
genuinely new frontend API module (`api/authority.ts` per ui-work.txt §9/§27) rather than
just consuming fields already on `IncidentView`.

## Continuation commands

```
cd /workspace/HydroSwarm
git checkout feature/ui-mission-control-v1
cd frontend
npm install   # if node_modules is missing (ephemeral sandbox)
npm run lint && npm run typecheck && npm test -- --run && npm run build
```

No long-running/background jobs are active from this session as of this report.
