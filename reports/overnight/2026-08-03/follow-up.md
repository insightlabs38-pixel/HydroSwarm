# Follow-up / next actions

Updated live throughout the run. As of Bundle E completion (commit `0606586`):

## Immediate next steps (in dependency order)

1. **Phase 5 / Bundle F — Cycle B corpus generation and full updated S training.** Cycle A
   proved the pipeline end to end (generation, sharding, label audit, real training loop, all
   with real bugs found and fixed along the way -- see "Bundle E findings" below), so Bundle F
   can proceed with confidence the underlying mechanics are sound. Cycle B needs 8,000-12,000
   train / 1,000 validation / 1,000 calibration / 1,500-2,000 development_holdout / 300-500
   per OOD shift category, across 3 training topologies + 1 development-OOD topology (Cycle A
   only used 2 topologies total; Bundle F needs at least 2 more genuinely different networks
   -- none exist in the repo yet beyond the golden reference and branched-loop, so new .inp
   files need to be authored or imported before generation can start). Cycle B also requires
   the full distribution-stratification (12 dimensions) and required-hard-negatives lists the
   plan specifies for Cycle B specifically (Cycle A deliberately did not attempt these -- it
   is a smoke corpus). This is a substantially longer-running job than Cycle A's ~3 minutes;
   a real candidate for `hydroswarm.training.job_runner`-supervised background execution.
   After Cycle B: run the full E0-E10 experiment matrix (not just the 4 Bundle E
   smoke-screened), select the strongest updated S configuration(s), run source-only vs.
   full-multitask diagnostics, fit calibration.
2. **Task 3.2 — complete incident-view API contract.** Still not fully implemented (see prior
   handoff updates for detail). Independent of Bundle F -- disjoint code (model/corpus vs.
   frontend/API) -- can be picked up in parallel if ever running two threads of work, but this
   run has been proceeding through the model/data critical path first.
3. Bundle G (updated HydroCore-M training) is explicitly gated on Bundle F's S-architecture
   selection completing first -- "only after selecting strongest S architecture" per the
   plan -- so it cannot start early no matter how much idle capacity exists.

## Exact command to resume

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q                        # expect 355 passed
cd frontend && npm ci && npm run test -- --run   # expect 24 passed
npx playwright install --with-deps chromium && npx playwright test  # expect 10 passed
```

To inspect Cycle A or re-run the Bundle E smoke sweep:

```bash
export PYTHONPATH=src
python scripts/generate_cycle_a_corpus.py --output data/learning-v2/cycle-a   # already run; will refuse to overwrite
python scripts/run_architecture_smoke_jobs.py   # already run; ~2m24s, writes reports/results/v3/architecture-smoke-jobs.json
```

`experiments/runs/bundle-e-smoke/` (~1.3GB of disposable smoke-run checkpoints) is safe to
delete once this report has been reviewed; it is gitignored and not referenced by anything
downstream.

## Scope assessment for whoever picks this up next

Phase 0, Bundle A (0.2-0.8), Bundle B in full (1.1-1.5, 2.1-2.6), Bundle C (3.1, 3.3 partial,
3.4 partial, 3.5, 3.6, 3.7, 3.8; 3.2 given an interim treatment), Bundle D in full (4.0-4.6),
and Bundle E in full (Cycle A generation + Stage 1 smoke screening for E0/E3/E4/E9) are
complete, tested, and committed -- 44 commits, 302 new tests (257 backend + 45 frontend), all
gates green, zero baseline artifacts touched. This is substantive, real progress covering the
entire code/infrastructure half of the plan, the frontend correctness half, the model
architecture half, and now a first real (if intentionally small) proof that the full
generate -> shard -> audit -> train -> resume -> reload pipeline works end to end. What
remains is Task 3.2's full backend contract and Phases 5 (Cycle B/C)-9, which require
authoring or importing at least 2 more genuinely different network topologies, running real
multi-topology WNTR data generation at 10k-40k scenario scale, and real CPU training runs
lasting many hours each -- substantially larger in wall-clock terms than anything completed
so far.

## Bundle E findings (what was caught, and why it mattered)

Cycle A's stated purpose is exactly what it delivered: "variable-topology pipeline
validation, target coverage validation... shape and memory tests." Three real correctness
bugs were found and fixed only because this was the first time real multi-topology data and
a real model actually flowed through the full training loop together (every prior test used
either synthetic single-topology fixtures or hand-matched shapes on both sides of a boundary):

1. `label_audit._sensor_fault_prevalence` assumed one shared node count per split
   (`torch.stack` across the whole split) -- crashes the instant two topologies with
   different junction counts are mixed. Fixed (commit `76cb9a8`).
2. `compute_multitask_loss` never read targets_v2's `f"{task}_mask"` companions at all, so
   `corpus.py`'s placeholder 0 labels for NORMAL/SENSOR_FAULT_ONLY scenarios (~30% of Cycle A)
   were silently trained against as real labels. `source_region` was also found to have no
   loss wired to it at all. Fixed (commit `470a042`).
3. `HydroCore.evidence_head`'s output was never squeezed (`[batch,1]` vs. the real `[batch]`
   target), crashing the first real forward-pass-into-loss run. Fixed (commit `0606586`).

**Implication for whoever runs Cycle B/C**: these fixes are now in place and covered by
regression tests, so Cycle B/C training should not hit the same issues. But this is a strong
signal that *other* untested interactions between real multi-topology data and the training
loop may still exist -- treat Cycle B's own Stage 1 smoke screening (before the full 3-6 epoch
Stage 2 architecture screening) as a real gate, not a formality, precisely because it is what
caught all three of the above.

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
- Bundle F needs genuinely new network topologies (at least 2 more beyond golden-reference and
  branched-loop) authored or imported before Cycle B generation can start -- this is real
  authoring work (a valid, hydraulically-sane EPANET INP file), not a parameter tweak, and has
  not been scoped yet.
