# Follow-up / next actions

Updated live throughout the run. As of Tasks 1.1-1.5 completion (commit `7bdf4f5`):

## Immediate next steps (in dependency order)

1. Task 2.1 — define the targets_v2 contract (schema + docs for Sentinel/Scout/Strategist/
   control targets, event-cause and next-step class enums, masking rules for missing
   targets, schema-version compatibility checks).
2. Task 2.2 — generate Sentinel labels (source node/region, event presence/cause, timing,
   strength, sensor fault, evidence sufficiency) including required normal/fault-only
   examples. This is the first task that needs real simulation-backed label generation,
   not just schema/infrastructure work.
3. Tasks 2.3-2.4 — Scout and Strategist label generation (deeper integration with
   sampling/planning/WNTR verification).
4. Task 2.5 — OOD/abstention category definitions.
5. Task 2.6 — trajectory-state serialization.
6. Only after 2.1-2.6 pass their tests: Bundle C (frontend live/demo integrity) and Bundle D
   (configurable HydroCore architecture) can proceed; Cycle A corpus generation (Phase 5) is
   gated on all of Bundle B per the plan ("Do not begin large dataset generation until
   variable-topology, lazy-data-loading, target-schema, and label-audit tests pass" — the
   first three are now satisfied; target-schema/label-audit-for-targets_v2 is Task 2.1-2.2).

## Exact command to resume

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q   # expect 209 passed
```

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
