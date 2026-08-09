# Stage-F `no_adapters` third-seed repeatability check (seed 20260813)

Branch `agent/gcp-multitopology-v3`. This is a **stability/repeatability
check only**, not a checkpoint-selection sweep. It does not replace the
predeclared seed-20260810 selected checkpoint, and it does not change
`runtime_enabled_outputs` or any promotion decision.

## What was run

Third seed (`20260813`) of the already-selected `HydroCore-S` /
`no_adapters` Stage-F finalist configuration, using the exact same corpus,
split ownership, normalization, model config, training budget, task
weights, optimizer/scheduler, and evaluation protocol as
`no_adapters`-seeds `20260810`/`20260811` (`scripts/run_stage_f_training.py`,
unmodified `SHARED_MODEL_CONFIG`/`BATCH_SIZE`/`MAX_EPOCHS`/
`EARLY_STOPPING_PATIENCE`/`MAXIMUM_RUNTIME_SECONDS`):

```bash
export PYTHONPATH=src
python scripts/run_stage_f_training.py \
  --arms no_adapters --seeds 20260813 \
  --run-root experiments/runs/stage-f \
  --registry experiments/registry/stage-f.jsonl \
  --output reports/results/v4/stage-f-no-adapters-seed20260813.json
```

Ran to completion cleanly: 16/16 epochs, `stop_reason=maximum_epochs`
(matching both prior seeds — no early stopping in any of the three runs),
no exceptions, no non-finite validation loss (the training script itself
raises `RuntimeError` on a non-finite `best_validation_loss`, so a
completed report is itself proof of finiteness).

Checkpoint: `experiments/runs/stage-f/no_adapters-seed20260813/20260808T234644Z-7d902580/model-export.safetensors`
(SHA-256 `e63c836a35982647abc244fc1e25395fb1daa57ed3bce97864889272fa890ea1`,
gitignored/ephemeral, matching this project's established
`experiments/runs/` convention — see the `hydroswarm_checkpoint_persistence`
memory record). Full results:
`reports/results/v4/stage-f-no-adapters-seed20260813.json`.

## Headline comparison

| metric | seed 20260810 (selected) | seed 20260811 | seed 20260813 (this check) |
|---|---|---|---|
| best validation loss | 5.35117 | 5.42401 | 5.43492 |
| development-holdout mean loss | 8.79113 | 8.87386 | 8.90205 |
| wall time | 2812.5s | 2884.5s | 2852.3s |
| epochs completed | 16 | 16 | 16 |
| stopped early | False | False | False |
| non-finite / NaN values | none | none | none |

Seed 20260813 falls at the high (worse) end of the 3-seed spread on both
headline metrics — **not** a marginally-better result, so there is no
temptation to replace the predeclared seed-20260810 selection on this
evidence even setting the "do not replace merely for a marginally better
metric" instruction aside. The 3-seed spread is small in absolute and
relative terms: best-validation-loss spread 0.0838 (~1.6% relative to the
minimum), development-holdout spread 0.1109 (~1.3% relative to the
minimum). Runtime (2812-2884s) and epoch/stop behavior are effectively
identical across all three.

## Per-task development-holdout loss (the tasks the instruction calls out)

| task | seed 20260810 | seed 20260811 | seed 20260813 | max pairwise spread |
|---|---|---|---|---|
| `source_node` | 0.724327 | 0.725815 | 0.737651 | ~1.8% |
| `evidence_sufficiency` | 4.116838 | 4.192237 | 4.185743 | ~1.8% |
| `relative_strength` | 0.590496 | 0.584537 | 0.608177 | ~4.0% |
| `event_presence` | 0.299072 | 0.299613 | 0.305041 | ~2.0% |
| `event_cause` | 0.903612 | 0.901719 | 0.896706 | ~0.8% |
| `plan_value` | 0.004488 | 0.004495 | 0.004466 | ~0.6% |
| `plan_validity` | 0.023420 | 0.023788 | 0.023755 | ~1.6% |
| `exposure_proxy` | 0.102114 | 0.101997 | 0.102547 | ~0.5% |
| `pressure_risk_proxy` | 7.54e-06 | 5.15e-06 | 5.51e-06 | ~46%* |
| `service_loss_proxy` | 7.04e-06 | 6.80e-06 | 4.32e-06 | ~39%* |
| `containment_time_proxy` | 0.046004 | 0.049852 | 0.048191 | ~8.4% |
| `plan_regret_proxy` | 0.012904 | 0.012924 | 0.012869 | ~0.4% |
| `duration` | 0.796422 | 0.796250 | 0.790868 | ~0.7% |
| `source_region` | 0.372927 | 0.368976 | 0.370652 | ~1.1% |
| `start_time` | 0.798456 | 0.811609 | 0.815345 | ~2.1% |
| `sensor_fault` | 4.13e-05 | 3.67e-05 | 3.30e-05 | ~25%* |

\* `pressure_risk_proxy`, `service_loss_proxy`, and `sensor_fault` have
absolute losses at or below ~1e-5-1e-4 in every seed (the tasks are
already near-saturated). Percent spreads on values this small are noise,
not a stability signal — flagged so this isn't misread as instability
by a percentage figure alone; the absolute magnitudes are two to four
orders of magnitude below every other task's loss and below any operating
threshold.

The largest *substantively meaningful* pairwise spread is
`relative_strength` (~4.0%) and `containment_time_proxy` (~8.4% on an
already-small absolute value, 0.046-0.050). Both are ordinary seed-to-seed
training variance, not a directional drift or a discontinuity — none of
`source_node`, `evidence_sufficiency`, `relative_strength`, the event
outputs, or the plan tasks show a seed that is an outlier relative to the
other two; seed 20260813 sits inside or adjacent to the range already
spanned by seeds 20260810/20260811 on every task.

## Numerical stability

No NaNs or non-finite values anywhere in `best_validation_loss`,
`development_holdout.mean_loss`, or any of the 16
`development_holdout.per_task_mean_loss` entries. Consistent with both
prior seeds.

## Runtime and memory

Wall time 2852.3s, within the 2812.5-2884.5s range spanned by the first
two seeds (no runtime regression or blowup).

Peak RSS was not recorded by either historical seed run (the training
script does not capture it), so there is no historical baseline to
compare against exactly. This session sampled RSS via periodic external
`ps` polling during the run (10-minute interval, not continuous): 1568 MB,
1891 MB, 1880 MB, 2177 MB across four samples before the process exited
between the fourth sample and completion. This is an approximate,
externally-sampled figure, not a continuously-tracked peak, and is
reported honestly as such rather than as a precise peak-RSS measurement.
It is consistent with the general memory profile expected of this
`HydroCore-S` config/batch size and shows no runaway growth.

## Decision

**Seed 20260813 is reasonably consistent with the existing finalist
behavior.** Per the governing instruction, this 3-seed evidence is
recorded and **seed 20260810 is retained as the selected checkpoint** —
no replacement, no re-fit of calibration, no rebuild of the release
bundle (the selected checkpoint has not changed, so
`experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810` and
the calibration/release-bundle artifacts built from it in Section 19
remain valid and untouched).

No material instability was found. This is not a STOP condition.
