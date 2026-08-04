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

## Bundle F Stage 2 architecture screening (E0-E8) -- RUNNING

First real use of `hydroswarm.training.job_runner` (Task 0.3's resumable background-job
supervisor) for a genuinely long job, per the plan's guidance. Launched against the full Cycle B
corpus (9,000 train examples, 3 topologies), 4 epochs, batch size 16, seed `20260805`, one run
per E0-E8 configuration, ranked by the predeclared score in
`scripts/run_stage2_architecture_screening.py`'s docstring.

| Field | Value |
|---|---|
| Run dir | `experiments/jobs/bundle-f-stage2/` |
| Status file | `experiments/jobs/bundle-f-stage2/status.json` |
| Log | `experiments/jobs/bundle-f-stage2/job.log` |
| PID file | `experiments/jobs/bundle-f-stage2/job.pid` |
| Registry | `experiments/registry/bundle-f-stage2.jsonl` (per-experiment `ExperimentRegistry` records) |
| Final report (on completion) | `reports/results/v3/stage2-architecture-screening.json` |
| Started | 2026-08-04T21:36:19Z |
| Max runtime per experiment | 7200s (2 hours); no overall job-level cap -- 9 experiments sequentially in one process |
| Progress at handoff | E0 (baseline) completed successfully in 1300.4s (~21.7 min); E1 in progress |

**To check status:**

```bash
cd /workspace/HydroSwarm
python3 -c "import json; print(json.load(open('experiments/jobs/bundle-f-stage2/status.json'))['state'])"
tail -20 experiments/jobs/bundle-f-stage2/job.log
```

**To resume/relaunch if the job dies before completing** (the script itself has no
per-experiment checkpointing -- a restart reruns all 9 experiments from scratch, since it writes
its final report only once at the end; total observed pace is ~22 min/experiment so a full
restart costs on the order of 3-3.5 hours, not prohibitive at this corpus size):

```bash
cd /workspace/HydroSwarm
export PYTHONPATH=src
python3 -c "
import sys; sys.path.insert(0, 'src')
from hydroswarm.training import job_runner
command = [sys.executable, '-u', 'scripts/run_stage2_architecture_screening.py']
handle = job_runner.launch(command, run_dir='experiments/jobs/bundle-f-stage2', workdir='.',
    min_free_disk_gb=5.0, resume_command=command, env={'PYTHONPATH': 'src'})
print(handle.pid)
"
```

Once `state` is `COMPLETED`, mark it finished and inspect the ranking:

```bash
python3 -c "
from hydroswarm.training import job_runner
job_runner.mark_finished('experiments/jobs/bundle-f-stage2', exit_code=0)
"
python3 -c "
import json
report = json.load(open('reports/results/v3/stage2-architecture-screening.json'))
print(report['ranking'])
print(report['failures'])
"
```

Next steps after this job completes: select the strongest S configuration(s) by the
predeclared score, run source-only (E7) vs. full-multitask (E8) diagnostics already included in
this sweep, fit calibration artifacts against the winner, then proceed to Bundle F's remaining
evaluation phases per the plan.
