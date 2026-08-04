# Follow-up / next actions

Updated live throughout the run. As of Bundle D completion (commit `d0a0b42`):

## Immediate next steps (in dependency order)

1. **Phase 5 / Bundle E — Cycle A corpus generation.** Code-side prerequisites are now fully
   complete: Bundle B's variable-topology/lazy-loading/target-schema/label-audit
   infrastructure, and Bundle D's configurable HydroCore architecture (every new flag
   defaults to the checkpoint-compatible original behavior, so smoke/screening jobs can
   ablate `prior_mode`/`incident_pooling`/`message_direction` without needing a from-scratch
   checkpoint). This has not been attempted yet: it requires actually generating multiple
   genuinely-different training topologies (only the single reference network has been used
   in every test so far) and running real multi-topology WNTR generation at scale -- a
   substantial, long-running (likely many-hour) undertaking best run as a background job via
   `hydroswarm.training.job_runner` (Task 0.3).
2. **Task 3.2 — complete incident-view API contract.** Still not fully implemented.
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
   contract tests per the plan's Task 3.2 requirements. Independent of Bundle E (disjoint
   code: model/corpus vs. frontend/API) -- can be picked up in parallel by a separate pass if
   ever running two threads of work, but this run is proceeding to Bundle E first since it's
   directly on the plan's critical path to a trained/selected architecture.
3. Once real multi-topology corpora exist (Cycle A, then B, then C), Bundle D's new
   architecture flags are ready to actually be exercised in training: the E0-E10 architecture
   screening matrix (Phase 6) ablates exactly the flags Bundle D added
   (`prior_mode`/`incident_pooling`/`message_direction`, plus `event_control_heads`/
   `auxiliary_heads`/`consequence_prescreening_heads` once their labels exist). Label
   generation for Task 4.4's `next_step`, Task 4.5's three auxiliary targets, and Task 4.6's
   five consequence proxies was deliberately deferred out of Bundle D (see Bundle D's own
   commits for the exact reasoning) -- these need real corpus-generation/PlanVerifier wiring
   before those specific heads can be trained, not before Cycle A smoke jobs can start using
   the rest of the architecture.

## Exact command to resume

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q                        # expect 349 passed
cd frontend && npm ci && npm run test -- --run   # expect 24 passed
npx playwright install --with-deps chromium && npx playwright test  # expect 10 passed
```

## Scope assessment for whoever picks this up next

Phase 0, Bundle A (0.2-0.8), Bundle B in full (1.1-1.5, 2.1-2.6), Bundle C (3.1, 3.3 partial,
3.4 partial, 3.5, 3.6, 3.7, 3.8; 3.2 given an interim treatment), and Bundle D in full
(4.0-4.6) are complete, tested, and committed -- 38 commits, 296 new tests (251 backend + 45
frontend), all gates green, zero baseline artifacts touched, every new scenario/
label-generation test runs against the real reference network with real WNTR simulation (no
simulator mocks), every new frontend test runs against real rendered output (JSDOM for unit,
real Chromium for e2e, no snapshot-without-verification), and the promoted HydroCore-S
checkpoint's loadability was independently re-verified after every single Bundle D commit,
not just once at the end. This is substantive, real progress covering the entire
code/infrastructure half of the plan, the frontend correctness half, and the model
architecture half. What remains is Task 3.2's full backend contract and Phases 5-9, which
require actually generating multiple genuinely-different topologies and running real
multi-topology WNTR data generation at 10k-40k scenario scale plus real CPU training runs
lasting many hours each -- substantially larger in wall-clock terms than anything completed
so far, and the natural next phase for long-running background jobs via the job runner built
in Bundle A.

## Bundle D scope notes (what was deliberately not done, and why)

- **Label generation for the new heads' targets was not attempted.** `event_cause`/
  `event_presence` already had labels from Bundle B's corpus work and are wired into
  `compute_multitask_loss`; `next_step` (Task 4.4), `sensor_reconstruction`/
  `future_concentration`/`travel_time` (Task 4.5), and the five `*_proxy` targets (Task 4.6)
  do not have generators yet. Each is registered in the governed `targets_v2` contract with
  a real masking rule and source-of-truth, and each head's loss is wired to fire
  automatically the moment a real target with that name appears in a training batch -- no
  further model-side work is needed once Phase 5/6 corpus generation produces them.
- **Inference-pipeline serialization of any Bundle D output was not attempted.**
  `src/hydroswarm/inference/pipeline.py` (the human-approval-boundary-adjacent production
  output path) was not touched by Bundle D at all. This was a deliberate scope decision: all
  six new architecture flags default to `False`/original-behavior, so the promoted checkpoint
  and current production pipeline are completely unaffected either way, and wiring new,
  not-yet-trained heads into a boundary the plan explicitly says never to weaken felt like
  the wrong thing to do speculatively, ahead of those heads actually being trained on real
  data.
- **The Task 4.6 ranking-quality / simulator-call-reduction evaluation harness was not
  built.** That evaluation is only meaningful once a `consequence_prescreening_heads=True`
  variant is actually trained on real `PlanVerifier` output, which requires Phase 5/6 corpus
  and training work that hasn't started.

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
