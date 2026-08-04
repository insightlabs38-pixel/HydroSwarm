# Follow-up / next actions

Updated live throughout the run. As of Bundle B completion in full (commit `63e060b`):

## Immediate next steps (in dependency order)

1. **Bundle C — frontend live/demo data integrity (Tasks 3.1-3.8).** In progress now.
   Start with Task 3.1 (explicit LIVE/REPLAY/DEMO_FALLBACK/ERROR runtime modes) since the
   rest of Bundle C builds on having that mode concept in place. Inspect `frontend/src/api.ts`
   and `frontend/src/App.tsx` first (per the plan's flagged concern: the API adapter overlays
   partial live data onto `demoIncident`, which can make fixture content appear live).
2. Bundle D — configurable HydroCore architecture (Tasks 4.0-4.6). Can proceed in parallel
   with Bundle C since they touch disjoint code (frontend vs. model).
3. Phase 5 (Cycle A/B/C corpus generation) is now unblocked on the *code* side -- Bundle B is
   complete (variable-topology, lazy-loading, target-schema, and label-audit-capable
   infrastructure all exist and are tested) -- but has not been attempted: it requires
   actually generating multiple genuinely-different training topologies (only the single
   reference network has been used in tests so far) and running real multi-topology WNTR
   generation at scale, which is a substantial, long-running (likely many-hour) undertaking
   best run as a background job via `hydroswarm.training.job_runner` once Bundle C/D land.

## Exact command to resume

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q   # expect 270 passed
```

## Scope assessment for whoever picks this up next

Phase 0, Bundle A (0.2-0.8), and Bundle B in full (1.1-1.5, 2.1-2.6) are complete, tested,
and committed -- 27 commits, 172 new tests, all gates green, zero baseline artifacts
touched, every new scenario/label-generation test runs against the real reference network
with real WNTR simulation (no simulator mocks). This is substantive, real progress covering
the entire code/infrastructure half of the plan. What remains is Bundle C (frontend), Bundle
D (model architecture config), and especially Phases 5-9, which require actually generating
multiple genuinely-different topologies and running real multi-topology WNTR data
generation at 10k-40k scenario scale plus real CPU training runs lasting many hours each --
substantially larger in wall-clock terms than anything completed so far, and the natural
next phase for long-running background jobs via the job runner built in Bundle A.

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
