# Follow-up / next actions

Updated live throughout the run. As of Phase 0 completion:

## Immediate next steps

1. Investigate and fix the pre-existing frozen-artifact size mismatch
   (`tests/frozen/test_frozen_artifacts.py`) as its own small commit, before or alongside
   Bundle A, since it currently blocks a fully-green `pytest -q` baseline.
2. Begin Bundle A (Task 0.2 experiment registry) — everything else in the dependency order
   is blocked on having governed run provenance.

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
