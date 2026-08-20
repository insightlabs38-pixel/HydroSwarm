# Evaluation protocol and final evidence

> **Current evaluation authority:** HydroCore-v5. The final locked evaluation has been executed exactly once and passed. Historical HydroCore-v4 and earlier S/M/L evaluations are preserved under [Historical record](#historical-record).

HydroCore-v5 was developed under a preregistered split/seed/promotion framework designed to prevent the final test from becoming a tuning set. The master protocol is [HYDROCORE_V5_EXPERIMENT_PROTOCOL.md](evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md); generated artifacts, not retrospective prose, define the final outcome.

## Evaluation hierarchy

| Evidence tier | Purpose | May tune model/system? |
|---|---|---|
| Training | gradient optimization on governed synthetic scenarios/prefixes | yes, within protocol |
| Validation / development | architecture, training recipe, capability characterization | yes, before freeze |
| Calibration | conformal fitting/threshold calibration only | no gradient/checkpoint selection |
| M10 trajectory development | production-path integration, authority and utility characterization | development only |
| M11.5 full validation | final pre-lock matrix and identity checks | no finalist change afterward |
| Locked-final | exactly-once final evaluation on frozen stress matrix | **no** |
| Locked-topology | exactly-once final novel-topology/fail-closed evaluation | **no** |

The locked populations were inaccessible throughout M0–M10 and remained unopened through finalist selection/freeze and M11.5. They were materialized under a frozen design, explicitly authorized, atomically marked OPENED, then evaluated once. No rerun or post-lock tuning occurred.

## Lifecycle: M9 → M10 → M11

### M9 — predictor and training recipe selection

M9 closed the final architecture/training/capacity search. The selected predictor remained the 4.18M-parameter S model after:

- causal-prefix/feature-semantics work;
- temporal-architecture comparison;
- true interleaved multi-topology training;
- exact optimizer-step parity;
- calibration-grouping/support studies;
- S-vs-M capacity testing.

The frozen training recipe became `CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING`. HydroCore-M (13,919,572 parameters) was not promoted because the predeclared meaningful unseen-topology gain was not observed. See [M9 final closure](../reports/evaluation/hydrocore-v5/m9-final/m9-final-summary.md).

### M10 — downstream authority and full trajectories

M10 separated learned architecture capacity from what had valid supervision and what could hold authority. The final release promotes only Sentinel outputs; learned OOD/Scout/Strategist controls remain non-authoritative.

M10.4 ran production-equivalent trajectory development evidence over **360 physical incidents / 720 API trajectories**, three model seeds, three trained topology families, three unseen development families, and seven condition kinds. Its full-trajectory gate passed. That evidence was used for integration/authority validation, not as a substitute for the locked final evaluation. See [M10.4 population](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-population-manifest.json), [gate](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-gate.json), and [closure](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json).

M10.5 completion then froze the selected seed/checkpoint, release bundle, calibration artifact, five learned runtime outputs, deterministic authority wiring, and fail-closed V5 behavior. See [M10.5 completion](../reports/evaluation/hydrocore-v5/m10/m10-5-completion/m10-5-completion-closure.json).

### M11 — finalist freeze and one-time final test

1. M11.1 selected the finalist.
2. M11.2 froze exact model/calibration/runtime authority.
3. M11.5 completed the full pre-lock validation matrix.
4. M11.6A froze the final test design and materialized the locked populations without evaluating them.
5. Explicit human authorization allowed one opening.
6. The evaluator wrote the atomic OPENED record.
7. M11.6 evaluated all 125 incidents once.
8. Final, topology, safety, and governance gates passed.
9. No rerun and no post-lock tuning followed.

Diagram: [model lifecycle](diagrams/model-lifecycle-v5.mmd).

## Final locked population

- `locked_final_test`: 105 incidents = 7 condition kinds × 15 incidents each.
- `locked_topology_test`: 20 incidents on four procedurally generated novel topologies.
- Total: 125.
- Population completeness: PASS.
- Each expected scenario evaluated exactly once.
- Topology novelty audit: PASS.
- Seed namespace isolated from prior experiment seeds by construction.

See [materialization manifest](../data/locked/m11-6/m11-6-materialization-manifest.json).

## Locked-final results by condition

All rows below have `n=15`. Coverage is conformal coverage because `calibrated_rate=1.0` throughout the applicable locked-final population.

| Condition | Top-1 | Top-3 | MRR | Coverage | Mean set size | Actionable |
|---|---:|---:|---:|---:|---:|---:|
| NOMINAL | 73.3% | 86.7% | 0.821 | 93.3% | 2.00 | 80.0% |
| AMBIGUITY_DISAGREEMENT | 40.0% | 60.0% | 0.567 | 100.0% | 4.40 | 40.0% |
| LOW_COVERAGE_ACTIVE_SAMPLING | 46.7% | 86.7% | 0.648 | 93.3% | 3.93 | 40.0% |
| MEASUREMENT_NOISE | 40.0% | 66.7% | 0.586 | 93.3% | 4.20 | 33.3% |
| SENSOR_DROPOUT | 46.7% | 60.0% | 0.597 | 66.7% | 3.07 | 60.0% |
| SENSOR_HEALTH_DEGRADED | 66.7% | 86.7% | 0.778 | 86.7% | 1.73 | 93.3% |
| SEVERITY_SHIFT | 73.3% | 86.7% | 0.815 | 86.7% | 2.27 | 80.0% |

Aggregate locked-final (`n=105`): Top-1 55.2%, Top-3 76.2%, MRR 0.687, coverage 88.6%, mean candidate-set size 3.09, actionable rate 61.0%.

The frozen coverage gate was **aggregate applicable locked-final coverage >= 0.85**; it passed at 88.6%. A condition slice such as sensor dropout can be below 85% without rewriting the frozen aggregate gate after results are seen. That is an important distinction between reporting a weakness and changing the preregistered decision rule.

## Locked novel-topology result

The 20 novel-topology incidents produced:

- Top-1: 55.0%
- Top-3: 70.0%
- MRR: 0.652
- `calibrated_rate`: 0.0%
- actionable rate: 0.0%
- human-approved rate: 0.0%
- topology fail-closed gate: PASS

The evaluator labels topology-shift predictive metrics **`DESCRIPTIVE_NON_GATING`**. The raw candidate-set inclusion rate (60%) is also descriptive because calibration was inapplicable; it must not be advertised as conformal coverage.

This result establishes that the predictor retained some source-localization signal on these four genuine topology shifts while the authority path correctly withheld calibrated planning. It does not establish generalized calibrated performance on unseen utilities.

## Hard gates versus descriptive metrics

The final gate checked:

- population completeness;
- frozen finalist identity and manifest hashes;
- locked-final calibration coverage floor;
- finite outputs;
- sample budget;
- topology novelty;
- locked-topology fail-closed behavior;
- no unsafe action;
- no V4 fallback;
- all hard safety counters zero.

The gate explicitly classifies Top-1/Top-3/MRR, candidate-set size, posterior entropy, actionability/abstention, Scout benefit, plan counts, topology predictive metrics, and novel-topology calibration coverage as **descriptive metrics**.

A descriptive metric can be scientifically important without being a promotion/safety gate.

## Locked safety evidence

All 15 hard counters were evaluated and zero:

- autonomous actuation detected;
- finalist identity drift;
- human approval bypassed;
- inaccessible sample selected;
- invariant failures;
- learned OOD overriding deterministic OOD;
- learned Scout selecting a sample;
- learned Strategist selecting a plan;
- non-finite value reaching a decision;
- rejected plan surfaced as safe;
- sampled node reselected;
- sampling budget exceeded;
- silent V4 fallback;
- stale approval accepted;
- unverified plan surfaced as actionable.

See [safety counters](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-safety-counters.json).

## Development versus locked evidence

M10.4 development and M11.6 locked evidence answer different questions and should not be numerically conflated. M10.4 was a broad production-path development matrix used before freeze; M11.6 is the one-time final population after all tuning closed. Differences between their numbers are not a “regression test” unless the preregistered populations/metrics are directly matched.

The final locked result is intentionally the headline scientific evidence.

## Why M11.6 must not be rerun

The one-time lock is a governance property, not an inconvenience. A second execution would turn a one-time held-out set into observed data and weaken the evidentiary claim. Reproduction therefore means verifying the immutable opened/metrics/gate/safety/closure artifacts and reproducing the surrounding code/protocol on non-locked fixtures—not opening M11.6 again.

See [Reproducibility](REPRODUCIBILITY.md).

## What this evaluation does not establish

- field accuracy or live-utility safety;
- calibrated generalization to unseen topology families;
- chemistry/pathogen/toxicity identification;
- robustness outside the tested synthetic condition generators;
- correctness of an inaccurate hydraulic/network model;
- autonomous-action safety (the system intentionally has no autonomous actuation).

## Historical record

Historical evaluation artifacts remain useful for provenance but are not current V5 claims:

- HydroCore-v4 Phase 13/14 and capability-remediation reports under `reports/results/v4/` and `docs/evaluation/`;
- the earlier S/M/L locked 200-incident comparison in `reports/results/medium-evaluation-final.json`;
- the older topology-transfer experiment in `reports/results/topology-transfer-m.json`;
- all M9/M10 individual reports, which are development history feeding the final V5 freeze.

When a historical report says its locked set was unopened, read that statement in its historical timestamp/context. Current V5 lock status is the M11.6 terminal record above.
