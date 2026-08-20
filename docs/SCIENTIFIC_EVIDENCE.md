# Scientific evidence dossier

This page is the detailed scientific evidence record for the **final HydroCore-v5 M10 frozen release**. It separates development evidence from the one-time locked evaluation and distinguishes predictive measurements from hard authority/safety gates.

## Frozen identity

| Item | Frozen value |
|---|---|
| Variant / parameters | `small` / 4,182,612 |
| Selected seed | `20260814` |
| Checkpoint SHA-256 | `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5` |
| Release manifest SHA-256 | `f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34` |
| Calibration SHA-256 | `8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d` |
| Calibration artifact hash | `f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd` |
| Calibration | alpha 0.1, `B_DEPTH_AWARE` |
| Trained task family | `sentinel` |
| Runtime learned outputs | 5 Sentinel outputs |
| Deterministic authorities | OOD / Scout / planning |
| Physical authority | WNTR/EPANET |
| Human approval | required |

Sources: [M11.2 freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json), [runtime manifest](../models/hydrocore-v5-release/runtime_manifest.json).

## Evidence chronology

### Development: M9

M9 selected the final S-scale predictor/training recipe without access to the locked final test. The selected checkpoint policy was final optimizer step 1350. A larger HydroCore-M arm did not meet the predeclared meaningful capacity-gain criterion and was not promoted.

### Development: M10

M10 made the model/output-governance boundary explicit and tested the integrated production path. M10.4 covered 360 physical incidents / 720 API trajectories over trained and unseen development families and seven condition kinds. The full trajectory gate passed, with development Top-1 `0.8194` in its own population. This number is not the final held-out headline because M10 was still development evidence.

A disclosed M10.4 gate was vacuous: a selected-plan-vs-NO_ACTION comparison could not be positively characterized because NO_ACTION did not appear in the generated candidate set for that population. The artifact reports that limitation rather than claiming a positive non-harm result.

Sources: [M10.4 gate](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-gate.json), [population](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-population-manifest.json), [closure](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json).

### Freeze: M10.5 / M11.2 / M11.5

M10.5 completion froze the selected seed, release bundle/calibration, runtime output allowlist, deterministic authority, and no-V4-fallback behavior. M11.2 froze the finalist identity. M11.5 made the full pre-lock validation matrix green. No final-test result was used to make those choices.

## Locked population and governance

M11.6A froze/materialized the final population:

- 105 locked-final incidents;
- 20 locked-topology incidents;
- 125 total;
- four novel topology files;
- 9–12 junctions in the generated novel topologies;
- 125 unique canonical scenario hashes;
- zero within-set collisions;
- a locked seed namespace disjoint from prior seed namespaces by construction.

Materialization explicitly occurred while the lock was still unopened.

After explicit authorization, M11.6 recorded one atomic OPENED transition, executed all 125 cases, and closed with:

- `authorized_openings = 1`;
- `locked_open_count = 1`;
- authorization consumed;
- no retry/resume;
- no locked rerun;
- no post-locked tuning;
- no manifest/dataset changes;
- no code/evaluator changes.

Sources: [materialization manifest](../data/locked/m11-6/m11-6-materialization-manifest.json), [opened record](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-opened-record.json), [post-run governance](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-post-run-governance.json).

## Locked-final predictive matrix

Each condition has `n=15`.

| Condition | Top-1 | Top-3 | MRR | Mean posterior entropy | Coverage | Mean set size | Actionable |
|---|---:|---:|---:|---:|---:|---:|---:|
| NOMINAL | 73.3% | 86.7% | 0.821 | 0.555 | 93.3% | 2.00 | 80.0% |
| AMBIGUITY_DISAGREEMENT | 40.0% | 60.0% | 0.567 | 1.608 | 100.0% | 4.40 | 40.0% |
| LOW_COVERAGE_ACTIVE_SAMPLING | 46.7% | 86.7% | 0.648 | 1.317 | 93.3% | 3.93 | 40.0% |
| MEASUREMENT_NOISE | 40.0% | 66.7% | 0.586 | 1.425 | 93.3% | 4.20 | 33.3% |
| SENSOR_DROPOUT | 46.7% | 60.0% | 0.597 | 1.184 | 66.7% | 3.07 | 60.0% |
| SENSOR_HEALTH_DEGRADED | 66.7% | 86.7% | 0.778 | 0.587 | 86.7% | 1.73 | 93.3% |
| SEVERITY_SHIFT | 73.3% | 86.7% | 0.815 | 0.783 | 86.7% | 2.27 | 80.0% |

Aggregate (`n=105`):

| Metric | Result |
|---|---:|
| Top-1 | 55.2% |
| Top-3 | 76.2% |
| MRR | 0.687 |
| Mean posterior entropy | 1.066 |
| Applicable conformal coverage | 88.6% |
| Mean candidate-set size | 3.09 |
| Actionable rate | 61.0% |
| Human-approved rate | 40.0% |
| No-safe-plan rate | 21.0% |

Source: [M11.6 metrics](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-metrics.json).

### Interpretation

The nominal subset is materially stronger than the aggregate stress matrix. The final evidence therefore supports a claim of **measured synthetic robustness with significant stress degradation**, not “73% final accuracy” without qualification.

Sensor dropout is the clearest applicable-calibration weakness: its 15-case coverage was 66.7%. The frozen hard gate, however, was aggregate applicable locked-final coverage `>=0.85`; the aggregate was 88.6% and passed. Reporting the weak slice does not justify redefining the gate retrospectively.

## Sampling evidence

Across the 105 locked-final incidents:

- 37.1% requested at least one sample;
- mean samples per incident: 0.771;
- mean entropy reduction per sample: 0.136 bits;
- mean true-source rank change per sample: 0.0247.

These are descriptive final measurements. The sampling authority itself is deterministic `rank_sample_locations`; the locked counter `learned_scout_selected_sample` remained zero.

## Planning evidence

Across locked-final:

- mean generated candidates per incident: 0.819;
- mean WNTR-verified candidates: 0.400;
- human-approved rate: 40%;
- no-safe-plan rate: 20.95%.

The final evaluator's hard safety tests, not the number of approvals, determine whether unverified/rejected plans improperly crossed the authority boundary.

## Novel-topology evidence

For `locked_topology_test` (`n=20`):

| Metric | Result | Status |
|---|---:|---|
| Top-1 | 55.0% | descriptive/non-gating |
| Top-3 | 70.0% | descriptive/non-gating |
| MRR | 0.652 | descriptive/non-gating |
| Raw candidate inclusion | 60.0% | descriptive; **not calibrated coverage** |
| `calibrated_rate` | 0.0% | fail-closed authority evidence |
| Actionable rate | 0.0% | fail-closed authority evidence |
| Human-approved rate | 0.0% | fail-closed authority evidence |
| Generated plan candidates | 0 | fail-closed authority evidence |
| WNTR-verified plan candidates | 0 | fail-closed authority evidence |

The topologies are genuinely novel under the frozen materialization audit. Predictive signal survives, but calibrated operational authority does not. This is the intended honest distinction: **prediction under shift is not automatically permission to act under shift**.

No external baseline is used to label 55% Top-1 “strong”; the repository simply reports the measured value and the enforced authority behavior.

## Final hard gate

`m11-6-gate.json` records every hard check as passing:

- evaluation population complete;
- finalist identity;
- applicable locked-final calibration coverage;
- locked-final complete;
- locked-topology complete;
- locked-topology fail closed;
- manifest hashes;
- no unsafe action;
- no V4 fallback;
- finite outputs;
- zero safety counters;
- sample budget;
- topology novelty.

`global_pass = true`, `locked_final_pass = true`, `locked_topology_pass = true`.

## All 15 hard safety counters

| Counter | Locked count |
|---|---:|
| autonomous_actuation_detected | 0 |
| finalist_identity_drift | 0 |
| human_approval_bypassed | 0 |
| inaccessible_sample_selected | 0 |
| invariant_failures | 0 |
| learned_ood_overrode_deterministic | 0 |
| learned_scout_selected_sample | 0 |
| learned_strategist_selected_plan | 0 |
| nonfinite_value_reached_decision | 0 |
| rejected_plan_surfaced_as_safe | 0 |
| sampled_node_reselected | 0 |
| sampling_budget_exceeded | 0 |
| silent_v4_fallback | 0 |
| stale_approval_accepted | 0 |
| unverified_plan_surfaced_as_actionable | 0 |

This is strong evidence about the tested software authority invariants. It is not a guarantee of real-world safety outside the modeled/tested conditions.

## Terminal result

- `M11_6_LOCKED_FINAL_PASS`
- `M11_6_LOCKED_TOPOLOGY_PASS`
- `M11_6_LOCKED_EVALUATION_PASS`
- no finalist change allowed
- no retry after fail
- no rerun
- no post-lock tuning

Source: [M11.6 closure](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-closure.json).

## What is established

Within the frozen synthetic test design, the evidence establishes:

- exact finalist/model/calibration identity remained stable;
- the applicable locked-final aggregate conformal coverage floor passed;
- the complete 125-case population ran once;
- all final hard authority/safety counters were zero;
- novel-topology calibration/action authority failed closed;
- predictive performance under stress/topology shift is measured and reported, including weaknesses.

## What is not established

The evidence does **not** establish:

- field accuracy;
- utility-scale calibrated topology transfer;
- public-health safety;
- chemistry/pathogen identification;
- real-world action safety;
- absence of all software failure modes;
- a per-incident conformal guarantee.

For claim-level wording, see [Claims and evidence](CLAIMS_AND_EVIDENCE.md).
