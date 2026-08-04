# Follow-up / next actions

Updated live throughout the run. As of Bundle B schema/infra completion (commit `541c72c`):

## Immediate next steps (in dependency order)

1. **Task 2.2 — generate Sentinel labels** (source node/region, event presence/cause,
   timing, strength, sensor fault, evidence sufficiency; required normal/fault-only
   examples). This is the next task and the first one that needs real simulation-backed
   label *generation* rather than schema/infrastructure work. Before writing code:
   investigate how `data/learning-v1` was actually generated
   (`scripts/rebuild_canonical_tensors.py` and whatever produced
   `data/learning-v1/scenarios/`) so the new generator extends that pipeline instead of
   duplicating or conflicting with it. This investigation was not yet done in this run.
2. Tasks 2.3-2.4 — Scout and Strategist label generation (deeper integration with
   `sampling`/`planning`/WNTR verification; expensive, WNTR-call-heavy).
3. Only after 2.2-2.4 pass their tests and label audits: Bundle C (frontend live/demo
   integrity) and Bundle D (configurable HydroCore architecture) can proceed; Cycle A corpus
   generation (Phase 5) is gated on all of Bundle B per the plan ("Do not begin large
   dataset generation until variable-topology, lazy-data-loading, target-schema, and
   label-audit tests pass" — variable-topology and lazy-data-loading are satisfied; the
   target-schema contract (2.1) is satisfied; the label-audit tooling (0.5) exists and
   generalizes to targets_v2, but has not yet been run against real Sentinel/Scout/
   Strategist labels because none exist yet).

## Exact command to resume

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q   # expect 242 passed
```

## Scope assessment for whoever picks this up next

Phase 0, Bundle A (0.2-0.8), and Bundle B's schema/infrastructure scope (1.1-1.5, 2.1, 2.5,
2.6) are complete, tested, and committed -- 22 commits, 144 new tests, all gates green, zero
baseline artifacts touched. This is real, substantive progress, but it is also the more
tractable half of the plan: schema definitions, data-structure extensions, and
infrastructure that could be built and verified with synthetic fixtures. The remaining work
(Tasks 2.2-2.4, all of Bundles C/D, and especially Bundles E-I which require actual
multi-topology WNTR data generation at 10k-40k scenario scale and real CPU training runs
lasting many hours each) is substantially larger in wall-clock terms and requires either
long-running background jobs (now supported via `hydroswarm.training.job_runner`) or
multiple further sessions. Treat this run's output as a solid foundation, not a finished
implementation of the full plan.

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
