# Phase 12 Stage G — Conditional HydroCore-M Decision

core-issues3.txt "PHASE 12 — STAGED TRAINING AND ABLATIONS", Stage G: "Run
HydroCore-M only if HydroCore-S shows a capacity-limited case: train and
validation losses both remain high; no major overfitting; persistent
underfitting on important tasks; short scaling screen improves operational
metrics; CPU/RAM cost remains acceptable."

## Pre-screen analysis of the completed Stage F `no_adapters` runs

Examined `experiments/runs/stage-f/no_adapters-seed20260810/*/metrics.jsonl`
(the full 16-epoch, real joint-v4-corpus run already completed and
reported in `stage-f-adapters-comparison.json`):

| epoch | mean train batch loss |
|---|---|
| 0 | 11.29 |
| 7 | 6.17 |
| 14 | 4.77 |
| 15 (final) | 4.64 |

Train loss was **still declining at a healthy, non-plateaued rate through
the final epoch** (epoch 14→15: −2.3%), and every one of the 21 real
per-task losses (`event_cause`, `source_node`, `relative_strength`,
`start_time`, `duration`, `evidence_sufficiency`, etc.) was still
improving meaningfully from epoch 7 to epoch 15 — none had flattened. None
of Stage F's 4 runs (2 arms × 2 seeds) early-stopped; all four hit the
16-epoch/7200s budget still improving. This is the signature of an
**epoch/time-budget-limited** regime, not a capacity-limited one — the
"persistent underfitting on important tasks" criterion this gate asks for
was not evident from the training curve alone.

## Empirical scaling screen

Per the gate's own explicit instruction to run "a short scaling screen"
rather than resting on curve-reading alone: `scripts/run_phase12_stage_g_scaling_screen.py`
(new this pass) trained `HydroCore-S` (small, 4,182,612 params) and
`HydroCore-M` (medium, 12,673,108 params — 3.0x more parameters) on the
**identical** joint-v4 corpus, task config, and a **matched** 3-epoch/
seed-20260810 budget (single seed — a screen, not a promotion-quality run;
`use_adapters=False`, Stage F's own established winner).

| variant | parameters | epochs | wall time | best_validation_loss | validation full-pass mean loss |
|---|---|---|---|---|---|
| small | 4,182,612 | 3 | 297.3s | 8.2046 | 8.2046 |
| medium | 12,673,108 | 3 | 612.0s | 8.4066 | 8.4066 |

**Medium is WORSE than small at the matched budget** (8.41 vs. 8.20), not
better — and consistently so: small beats medium on essentially every one
of the 21 per-task losses individually (`source_node` 0.824 vs. 0.856,
`event_cause` 0.551 vs. 0.579, `evidence_sufficiency` 0.133 vs. 0.153,
`next_step` 0.430 vs. 0.462, etc. — not one or two noisy tasks, a broad,
consistent pattern), while taking roughly half the wall time.

Full detail: `reports/results/v4/stage-g-scaling-screen.json`;
`experiments/registry/stage-g-scaling-screen.jsonl`.

## Gate-by-gate verdict

| criterion | result |
|---|---|
| train and validation losses both remain high | partially true (train ≈4.64, val ≈5.35 at 16 epochs) — but see below |
| no major overfitting | true (modest train/val gap, ~13%) |
| persistent underfitting on important tasks | **NOT demonstrated** — every task's loss was still actively declining, not plateaued |
| short scaling screen improves operational metrics | **FAILS directly** — medium is measurably *worse* than small at a matched budget |
| CPU/RAM cost remains acceptable | moot given the above (medium costs ~2x more per epoch for a worse result) |

## Decision: **Do not train HydroCore-M.**

The scaling screen is the gate's own tie-breaker and it is unambiguous:
more capacity does not help at this data/training-budget point — if
anything it currently hurts, most plausibly because a larger model needs
proportionally more epochs/gradient steps to reach the same point in its
own training trajectory that the smaller model reaches faster, not because
`HydroCore-S` has hit a real representational ceiling. The correct lever
for `HydroCore-S`'s still-declining training curve is more epochs/time
budget within the existing architecture, not a larger one — consistent
with core-issues3.txt's own explicit "Do not train HydroCore-M
automatically if S does not show a clear data-limited or capacity-limited
case" and "Do not add optimization complexity speculatively."

`HydroCore-S` (the `no_adapters` arm, per Stage F's own direction-consistent
2-seed screening result) remains the sole architecture-size candidate
carried into Phase 19 architecture selection. `HydroCore-L` was not
attempted, per the phase's own explicit prohibition.

No locked-test data was used to reach this decision.
