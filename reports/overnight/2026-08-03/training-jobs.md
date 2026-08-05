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

## Bundle F Stage 3 finalist training (E2/E0/E1, 2 seeds each) -- COMPLETED

Trained the top three Stage 2 finalists with two seeds each, to early stopping (patience=3) or a
documented 2-hour-per-run ceiling, then fit calibration per checkpoint (calibration split only)
and evaluated on validation/development_holdout/both OOD-holdout categories. All 6 runs
succeeded, zero failures.

| Field | Value |
|---|---|
| Run dir | `experiments/jobs/bundle-f-stage3/` (status/log committed at `23024dd`) |
| Registry | `experiments/registry/bundle-f-stage3.jsonl` |
| Final report | `reports/results/v3/stage3-finalist-training.json` |
| Selection recommendation | `reports/results/v3/finalist-selection-recommendation.md` |
| Started / ended | 2026-08-05T01:13:50Z / 2026-08-05T13:16Z (~12.0h) |
| Runs | 6 total: {E2, E0, E1} x seeds {20260810, 20260811} -- every run hit the 2h ceiling (`stop_reason: runtime_budget`) at 11-13 epochs, none early-stopped |
| Recommendation | **E1** (prior_mode=feature_only) -- best OOD-unseen-topology generalization (top1 0.530 vs E2's 0.495, calibrated coverage 0.848 vs 0.823), trading <1pp in-distribution accuracy to E2. See the recommendation doc for the full trade-off and its "not yet covered" caveats. |

To re-run from scratch: `python3 scripts/run_stage3_finalist_training.py` (no per-run resume; a
full restart costs ~12h at the observed pace -- check `job.log` for which finalist/seed
combinations already logged `OK` before deciding whether a full restart or a smaller manual
rerun of just the missing combinations is more appropriate).

## Bundle F Stage 4 control training (no-adapter HydroCore-S / HydroMono-S, 2 seeds) -- RUNNING

Trains the one Stage 4 control this repo can meaningfully distinguish, under the identical
budget as Stage 3's finalists. See `scripts/run_stage4_controls_training.py`'s module docstring
for why HydroMono-S and no-adapter HydroCore-S are the same control here (verified
bit-identical), why "current architecture baseline" (E0) and "classical-only baseline" need no
new training.

| Field | Value |
|---|---|
| Run dir | `experiments/jobs/bundle-f-stage4/` |
| Status file | `experiments/jobs/bundle-f-stage4/status.json` |
| Log | `experiments/jobs/bundle-f-stage4/job.log` |
| Registry | `experiments/registry/bundle-f-stage4.jsonl` |
| Final report (on completion) | `reports/results/v3/stage4-controls-training.json` |
| Started | 2026-08-05T13:30:19Z |
| Runs | 2 total: no-adapter-S x seeds {20260810, 20260811} (same seeds as Stage 3) |
| Max epochs / early stopping / per-run ceiling | 16 epochs / patience 3 / 7200s (2h) -- identical to Stage 3 |
| Verification before launch | scratch smoke run (1 epoch, 64/32/32/32-example subsets, not committed): completed cleanly, same pattern as Stage 3's pre-launch check |
| Classical-only baseline (already computed, no training) | val top1 0.625, dev_holdout top1 0.639, OOD-unseen-topology top1 0.236, OOD-severe-missingness top1 0.592 |

**To check status:**

```bash
cd /workspace/HydroSwarm
python3 -c "import json; print(json.load(open('experiments/jobs/bundle-f-stage4/status.json'))['state'])"
tail -30 experiments/jobs/bundle-f-stage4/job.log
```

**To resume/relaunch if the job dies before completing** (no per-run resume; a full restart
costs up to ~4h worst case at 2 runs x 2h):

```bash
cd /workspace/HydroSwarm
export PYTHONPATH=src
python3 -c "
import sys; sys.path.insert(0, 'src')
from hydroswarm.training import job_runner
command = [sys.executable, '-u', 'scripts/run_stage4_controls_training.py']
handle = job_runner.launch(command, run_dir='experiments/jobs/bundle-f-stage4', workdir='.',
    min_free_disk_gb=5.0, resume_command=command, env={'PYTHONPATH': 'src'})
print(handle.pid)
"
```

Once `state` is `COMPLETED`, mark it finished and inspect the results:

```bash
python3 -c "
from hydroswarm.training import job_runner
job_runner.mark_finished('experiments/jobs/bundle-f-stage4', exit_code=0)
"
python3 -c "
import json
report = json.load(open('reports/results/v3/stage4-controls-training.json'))
print(json.dumps(report['failures'], indent=2))
for control, seeds in report['results'].items():
    for seed, r in seeds.items():
        print(control, seed, r['validation_metrics']['source_top1'], r['development_holdout_metrics']['source_top1'], r['stopped_early'])
"
```

Next steps after this job completes: fold the no-adapter control's results into
`finalist-selection-recommendation.md` alongside E0's Stage 3 numbers and the classical-only
baseline (all four Stage 4 comparison points then complete), confirm E1 (or revise the
recommendation if a control surprisingly outperforms it), and proceed to Stage 5 (updated
HydroCore-M, explicitly gated on this selection) and Bundle G.
