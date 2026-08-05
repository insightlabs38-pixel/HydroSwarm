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

## Bundle F Stage 2 architecture screening (E0-E8) -- COMPLETED

Launched against the full Cycle B corpus (9,000 train examples, 3 topologies), 4 epochs, batch
size 16, seed `20260805`, one run per E0-E8 configuration, ranked by the predeclared score in
`scripts/run_stage2_architecture_screening.py`'s docstring. All 9 runs succeeded, zero failures.

| Field | Value |
|---|---|
| Run dir | `experiments/jobs/bundle-f-stage2/` (status/log committed at `e6cc544`) |
| Registry | `experiments/registry/bundle-f-stage2.jsonl` |
| Final report | `reports/results/v3/stage2-architecture-screening.json` |
| Started / ended | 2026-08-04T21:36:19Z / 2026-08-05T01:07:10Z (~3h31m) |
| Ranking (predeclared score) | E2 (0.7794) > E0 (0.7783) > E1 (0.7768) > E3 (0.7742) > E6 (0.7727) > E4 (0.7709) > E8 (0.7696) > E7 (0.7680) > E5 (0.7647) |

Per-experiment wall time ranged 1263-1514s (~21-25 min); no maximum-runtime or disk-budget
trip occurred. Top four scores span only 0.0052 -- noise-level at one seed/4 epochs, which is
exactly why Stage 3 (below) carries three finalists forward instead of trusting a single pick.
E7 (source-only weights) and E8 (full-multitask heads), the plan's required diagnostic pair, both
scored below baseline E0 at this scale. Disposable per-experiment checkpoints remain gitignored
under `experiments/runs/bundle-f-stage2/`.

To re-run from scratch: `python3 scripts/run_stage2_architecture_screening.py` (no per-experiment
resume; a full restart costs ~3.5h at the observed pace).

## Bundle F Stage 3 finalist training (E2/E0/E1, 2 seeds each) -- RUNNING

Trains the top three Stage 2 finalists with two seeds each, to early stopping (patience=3) or a
documented 2-hour-per-run ceiling, then fits calibration per checkpoint (calibration split only)
and evaluates on validation/development_holdout/both OOD-holdout categories. See
`scripts/run_stage3_finalist_training.py`'s module docstring for full methodology and scope
notes (full-trajectory eval and the live OODDetector pipeline are explicitly out of scope, not
silently skipped).

| Field | Value |
|---|---|
| Run dir | `experiments/jobs/bundle-f-stage3/` |
| Status file | `experiments/jobs/bundle-f-stage3/status.json` |
| Log | `experiments/jobs/bundle-f-stage3/job.log` |
| Registry | `experiments/registry/bundle-f-stage3.jsonl` |
| Final report (on completion) | `reports/results/v3/stage3-finalist-training.json` |
| Started | 2026-08-05T01:13:50Z |
| Runs | 6 total: {E2, E0, E1} x seeds {20260810, 20260811} |
| Max epochs / early stopping / per-run ceiling | 16 epochs / patience 3 / 7200s (2h) |
| Verification before launch | scratch smoke run (1 epoch, 64/32/32/32-example subsets, not committed): training, calibration fitting, and validation/dev-holdout/OOD evaluation all completed cleanly; OOD top-1 (27%) sensibly far below in-distribution top-1 (92%) |

**To check status:**

```bash
cd /workspace/HydroSwarm
python3 -c "import json; print(json.load(open('experiments/jobs/bundle-f-stage3/status.json'))['state'])"
tail -30 experiments/jobs/bundle-f-stage3/job.log
```

**To resume/relaunch if the job dies before completing** (no per-run resume across the whole
script; a restart reruns all 6 finalist/seed combinations from scratch since the report is only
written once at the end -- at up to 2h/run this could cost up to ~12h worst case, though early
stopping should make the real total considerably less; check `job.log` for which
finalist/seed combinations already logged `OK` before deciding whether a full restart or a
smaller manual rerun of just the missing combinations is more appropriate):

```bash
cd /workspace/HydroSwarm
export PYTHONPATH=src
python3 -c "
import sys; sys.path.insert(0, 'src')
from hydroswarm.training import job_runner
command = [sys.executable, '-u', 'scripts/run_stage3_finalist_training.py']
handle = job_runner.launch(command, run_dir='experiments/jobs/bundle-f-stage3', workdir='.',
    min_free_disk_gb=5.0, resume_command=command, env={'PYTHONPATH': 'src'})
print(handle.pid)
"
```

Once `state` is `COMPLETED`, mark it finished and inspect the results:

```bash
python3 -c "
from hydroswarm.training import job_runner
job_runner.mark_finished('experiments/jobs/bundle-f-stage3', exit_code=0)
"
python3 -c "
import json
report = json.load(open('reports/results/v3/stage3-finalist-training.json'))
print(json.dumps(report['failures'], indent=2))
for finalist, seeds in report['results'].items():
    for seed, r in seeds.items():
        print(finalist, seed, r['validation_metrics']['source_top1'], r['development_holdout_metrics']['source_top1'], r['stopped_early'])
"
```

Next steps after this job completes: per the plan's Stage 6 ("Calibration and final
selection"), compare the six finalist/seed results (validation + development_holdout + OOD
top-1/ECE/coverage), select the single strongest S architecture, and only then proceed to
Bundle G (HydroCore-M, explicitly gated on this selection) and the locked-test track.
