# Follow-up / next actions

Updated live throughout the run. As of Bundle C completion (commit `2067571`):

## Immediate next steps (in dependency order)

1. **Task 3.2 — complete incident-view API contract.** Not fully implemented this run.
   `fetchIncident()` currently throws `LiveViewIncompleteError` naming exactly which fields
   are missing (nodes, links, recommendedSample, evidence, plans, benchmarks, explanation)
   rather than silently substituting fixture content -- this closed the dangerous bug, but
   means LIVE mode can never actually be reached yet, even against a running, healthy
   backend. To close this: add a `GET /api/incidents/{id}/view` FastAPI endpoint (see
   `src/hydroswarm/api/app.py`, `src/hydroswarm/api/state.py::IncidentRuntime`) that
   assembles a complete response from the runtime's existing `state`/`plans`/`verifications`/
   `analysis`, plus the network's stored `geojson` (already on `NetworkRecord`) for map node
   coordinates. Then update `frontend/src/api.ts::fetchIncident()` to call it and remove the
   `LiveViewIncompleteError` throw. Add matching Pydantic (backend) and TypeScript (frontend)
   contract tests per the plan's Task 3.2 requirements.
2. **Bundle D — configurable HydroCore architecture (Tasks 4.0-4.6).** Can proceed
   independently of Task 3.2 (disjoint code: model vs. frontend/API).
3. Phase 5 (Cycle A/B/C corpus generation) is unblocked on the *code* side -- Bundle B is
   complete (variable-topology, lazy-loading, target-schema, and label-audit-capable
   infrastructure all exist and are tested) -- but has not been attempted: it requires
   actually generating multiple genuinely-different training topologies (only the single
   reference network has been used in tests so far) and running real multi-topology WNTR
   generation at scale, which is a substantial, long-running (likely many-hour) undertaking
   best run as a background job via `hydroswarm.training.job_runner` once Bundle D lands.

## Exact command to resume

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q                        # expect 270 passed
cd frontend && npm ci && npm run test -- --run   # expect 24 passed
npx playwright install --with-deps chromium && npx playwright test  # expect 10 passed
```

## Scope assessment for whoever picks this up next

Phase 0, Bundle A (0.2-0.8), Bundle B in full (1.1-1.5, 2.1-2.6), and Bundle C (3.1, 3.3
partial, 3.4 partial, 3.5, 3.6, 3.7, 3.8; 3.2 given an interim treatment) are complete,
tested, and committed -- 33 commits, 217 new tests (172 backend + 45 frontend), all gates
green, zero baseline artifacts touched, every new scenario/label-generation test runs
against the real reference network with real WNTR simulation (no simulator mocks), every
new frontend test runs against real rendered output (JSDOM for unit, real Chromium for e2e,
no snapshot-without-verification). This is substantive, real progress covering the entire
code/infrastructure half of the plan plus the frontend correctness half. What remains is
Task 3.2's full backend contract, Bundle D (model architecture config), and especially
Phases 5-9, which require actually generating multiple genuinely-different topologies and
running real multi-topology WNTR data generation at 10k-40k scenario scale plus real CPU
training runs lasting many hours each -- substantially larger in wall-clock terms than
anything completed so far, and the natural next phase for long-running background jobs via
the job runner built in Bundle A.

## Open questions / risks noted so far

- The plan's GCP section describes a persistent `/srv/hydroswarm/` layout with systemd/tmux
  job control on a dedicated VM the agent owns exclusively. This session is running inside a
  container on a GCP host (europe-west1-b, correct hardware shape) but without that
  `/srv/hydroswarm` layout pre-provisioned. The job runner (Task 0.3) will target paths
  under the repository (e.g. `experiments/`, `reports/`) rather than assuming `/srv/hydroswarm`
  exists, and will document this deviation rather than silently creating root-level
  directories outside the repo.
- No remote push has occurred and none is planned unless explicitly requested; the plan only
  requires not pushing to main, and commits will stay local on `agent/gcp-multitopology-v3`.
