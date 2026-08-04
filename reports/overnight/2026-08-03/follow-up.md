# Follow-up / next actions

Updated live throughout the run. As of this update (commit `bd944d5`; Stage 2 screening
still running in the background):

## Immediate next steps (in dependency order)

1. **Bundle F Stage 2 architecture screening (E0-E8) -- running now, not yet complete.**
   Launched against the real Cycle B corpus (12,750 scenarios landed this session -- see
   `summary.md`'s "Datasets generated"). E0 completed in ~21.7 minutes; E1-E8 remain, ~3-3.5
   hours total at the observed per-experiment pace. See `training-jobs.md` for exact
   monitor/resume commands. **This is the actively blocking item** -- everything else in Bundle
   F depends on its ranking.
2. **After Stage 2 completes**: read `reports/results/v3/stage2-architecture-screening.json`'s
   `ranking`, select the strongest S configuration(s) by the predeclared score (declared before
   any run, in the script's own docstring -- do not retune after seeing results), then run the
   full E0-E10 matrix if the plan calls for entries beyond E0-E8, fit calibration artifacts
   against the winner, and proceed to development-holdout/OOD evaluation.
3. **Task 3.2 — complete incident-view API contract.** Done this session (backend endpoint +
   schema, frontend wiring, contract tests) -- see `summary.md`'s "Task 3.2" section for full
   detail. No longer a follow-up item.
4. Bundle G (updated HydroCore-M training) is explicitly gated on Bundle F's S-architecture
   selection completing first -- "only after selecting strongest S architecture" per the
   plan -- so it cannot start early no matter how much idle capacity exists.
5. **Phase 8 (frontend UX, Tasks 8.1/8.3/8.4) is a candidate independent thread not yet
   scoped.** Task 8.2 (model-governance view) already exists (`ModelGovernanceTable.tsx`).
   Tasks 8.1 (simplified decision-rail workspace layout), 8.3 (validated-vs-unseen-topology
   comparison view), and 8.4 (a scripted "RUN VERIFIED INCIDENT DEMONSTRATION" guided judge
   mode) do not appear implemented yet (checked via grep for their plan-specified strings/
   concepts -- none found in `frontend/src`). These are genuinely independent of Bundle F/G's
   training results (they consume the same `IncidentView`/demo-fixture data already wired up),
   so they're a legitimate next thread if picked up before Bundle F training finishes -- not
   started this session because they're substantial new UI features (new components, new
   interaction flows, likely new visual-regression baselines) that deserve their own scoping
   pass rather than a rushed addition alongside active job supervision.

## Exact command to resume

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q                        # expect 363 passed (see test-results.md re: 1 flaky pre-existing test)
cd frontend && npm ci && npm run test -- --run   # expect 25 passed
npx playwright install --with-deps chromium && npx playwright test  # expect 10 passed
```

To check on or resume the actively-running Stage 2 screening job, see `training-jobs.md`'s
exact commands (status check, relaunch-from-scratch, mark-finished + read ranking).

To inspect Cycle A/B or re-run the Bundle E smoke sweep:

```bash
export PYTHONPATH=src
python scripts/generate_cycle_a_corpus.py --output data/learning-v2/cycle-a   # already run; will refuse to overwrite
python scripts/generate_cycle_b_corpus.py --output data/learning-v2/cycle-b   # already run; will refuse to overwrite
python scripts/run_architecture_smoke_jobs.py   # already run; ~2m24s, writes reports/results/v3/architecture-smoke-jobs.json
```

`experiments/runs/bundle-e-smoke/` (~1.3GB of disposable smoke-run checkpoints) is safe to
delete once this report has been reviewed; it is gitignored and not referenced by anything
downstream. Do not delete `experiments/runs/bundle-f-stage2/` or `experiments/jobs/bundle-f-stage2/`
until the Stage 2 job (see `training-jobs.md`) has completed and its report has been reviewed.

## Scope assessment for whoever picks this up next

Phase 0, Bundle A (0.2-0.8), Bundle B in full (1.1-1.5, 2.1-2.6), Bundle C in full (3.1-3.8,
including Task 3.2's full backend contract, completed this session), Bundle D in full (4.0-4.6),
and Bundle E in full (Cycle A generation + Stage 1 smoke screening for E0/E3/E4/E9) are
complete, tested, and committed -- 48 commits, 306+ new tests (261 backend + 45 frontend), all
gates green (modulo one pre-existing, unrelated, intermittent test -- see test-results.md), zero
baseline artifacts touched. Bundle F's Cycle B corpus (12,750 scenarios across 4 topologies) has
also landed this session, and Stage 2 architecture screening (E0-E8) is running now against it.
This covers the entire code/infrastructure half of the plan, the entire frontend correctness
half (including the one item that had only an interim treatment before), the model architecture
half, and now real multi-topology corpus generation at the plan's target scale. What remains is
the rest of Bundle F (Stage 2 completion, S finalist selection/training, calibration) and
Phases 6-9, which require real CPU training runs lasting many hours each -- substantially larger
in wall-clock terms than anything completed so far, and gated on Stage 2 finishing.

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
- This session pushed to `origin/agent/gcp-multitopology-v3` on GitHub (never to `main`) after
  each independently validated milestone, per this run's operating instructions to push to
  GitHub occasionally in addition to local commits.
- Bundle F's two additional topologies (loop-grid for training, coastal-branch for
  development-OOD) were authored and are already committed (`1a7cf15`); Cycle B generation used
  them successfully. No longer an open item.
