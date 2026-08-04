# Training / long-running job log

Updated live as jobs are launched, and finalized (best checkpoint, elapsed time) as they
complete or are paused for handoff.

## Bundle E Stage 1 smoke/failure screening (completed)

Ran synchronously in the foreground (each run completes in ~20-25s; no background job
supervision needed at this scale -- `hydroswarm.training.job_runner`, Task 0.3's resumable
background-job infrastructure, is reserved for the genuinely long Cycle B/C training runs in
Bundle F/G). All six runs recorded in `experiments/registry/bundle-e-smoke.jsonl` via
`ExperimentRegistry`; full detail in `reports/results/v3/architecture-smoke-jobs.json`.

| Run ID | Purpose | Status | Checkpoint | Best val loss (resumed) |
|---|---|---|---|---|
| `20260804T185412Z-d5040873` | E0 (baseline) | success | `experiments/runs/bundle-e-smoke/` (gitignored) | 5.32 |
| `20260804T185435Z-6f3d9552` | E3 (source_conditioned pooling) | success | ditto | see report |
| `20260804T185458Z-eedf2666` | E4 (dual_gated message channels) | success | ditto | see report |
| `20260804T185523Z-c5c147a6` | E9-none (prior_mode=none) | success | ditto | see report |
| `20260804T185546Z-77b59d2c` | E9-feature_only (prior_mode=feature_only) | success | ditto | see report |
| `20260804T185608Z-5d77d169` | E9-logit_only (prior_mode=logit_only) | success | ditto | see report |

Checkpoints are still on disk at handoff time under
`experiments/runs/bundle-e-smoke/<run>/checkpoints/`, but these are disposable smoke
artifacts (each run's own script already includes a resume check as part of what it verifies)
-- not meant to be preserved into or resumed from for Bundle F's real training. Safe to
delete (`rm -rf experiments/runs/bundle-e-smoke/`, ~1.3GB) once this report has been reviewed.

To re-run the whole Stage 1 sweep from scratch:

```bash
cd /workspace/HydroSwarm
export PYTHONPATH=src
python scripts/run_architecture_smoke_jobs.py
```

## No job currently active

Nothing is running in the background at handoff time. Bundle F (Cycle B corpus generation,
8,000-12,000 train scenarios across 3 topologies + 1 dev-OOD topology) is the next
long-running item and is the first candidate for `hydroswarm.training.job_runner`-supervised
background execution, per the plan's guidance for genuinely long jobs.
