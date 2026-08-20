# Milestone 8.5 summary: hydraulic backend diagnostic

> **SUPERSEDED by Milestone 8.5a** (`reports/evaluation/hydrocore-v5/m8-5a-summary.md`,
> `m8-5a-execution.json`, `m8-5a-scale.json`). This document's own headline finding below correctly identified
> the fork-based timeout wrapper (not the solver/demand model) as the actual scalability driver, but its
> **SIGCHLD/process-reaping speculation was tested directly by M8.5a and REFUTED**: the child's real `/proc` OS
> state during every observed false timeout was `S` (sleeping/blocked-on-IO), never `Z` (zombie). The actual
> mechanism is a `multiprocessing.Queue` "join before drain" IPC deadlock in `HydraulicSimulator.
> _run_with_timeout`, now fixed. The **`PDD_SCALABILITY_BLOCKER_REMAINS` decision token below is superseded** by
> M8.5a's `WRAPPED_EXECUTION_BLOCKER_RESOLVED` -- corrected measurements show no genuine PDD/WNTR/EPANET solver
> scalability limitation through N=250, with numerical parity to direct/unwrapped execution.

WNTR version: 1.5.0

**Headline finding: CONFIRMED: at least one arm succeeds in milliseconds when called directly (unwrapped) but times out through HydraulicSimulator._run_with_timeout (wrapped) at the same size. This means the process-completion detection in the fork-based timeout wrapper -- not the solver engine or demand model -- is the actual scalability driver Milestone 8 observed. The apparent 'PDD bottleneck' is very likely this wrapper artifact, not a genuine PDD/engine performance limitation. Root-causing the exact OS/signal-handling mechanism (this sandbox's SIGCHLD/process-reaping behavior is the leading suspect, given the unrelated zombie-process accumulation already observed in this same environment across Milestones 7B/8) is out of scope for this diagnostic milestone.**

## Section 1: capability matrix

| backend | supported | notes |
|---|---|---|
| WNTR_NATIVE_PDD | True |  |
| EPANET_PDD | True |  |
| WNTR_NATIVE_DDA | True |  |
| EPANET_DDA | True |  |
| WNTR_NATIVE_WATER_QUALITY | False | no `quality` result key; native WNTRSimulator does not simulate water quality |
| EPANET_WATER_QUALITY | True | production's own simulate_incident/_run_epanet path already uses this; re-verified in Section 7. |

## Section 2/3: benchmark sweep

| arm | target N | status | warmup ms | deterministic repeats | min pressure | max pressure |
|---|---|---|---|---|---|---|
| CURRENT_WNTR_PDD | 10 | OK | 54.35 | True | 106.52 | 110.34 |
| EPANET_PDD | 10 | OK | 46.49 | True | 106.52 | 110.34 |
| WNTR_DDA | 10 | OK | 48.23 | True | 106.52 | 110.34 |
| EPANET_DDA | 10 | OK | 47.76 | True | 106.52 | 110.34 |
| CURRENT_WNTR_PDD | 25 | FAILED (SimulationTimeoutError: CURRENT_WNTR_PDD exceeded the 60-second timeout) | | | | |
| EPANET_PDD | 25 | OK | 54.33 | True | 106.51 | 111.33 |
| WNTR_DDA | 25 | FAILED (SimulationTimeoutError: WNTR_DDA exceeded the 60-second timeout) | | | | |
| EPANET_DDA | 25 | OK | 52.50 | True | 106.51 | 111.33 |
| CURRENT_WNTR_PDD | 50 | FAILED (SimulationTimeoutError: CURRENT_WNTR_PDD exceeded the 60-second timeout) | | | | |
| EPANET_PDD | 50 | FAILED (SimulationTimeoutError: EPANET_PDD exceeded the 60-second timeout) | | | | |
| WNTR_DDA | 50 | FAILED (SimulationTimeoutError: WNTR_DDA exceeded the 60-second timeout) | | | | |
| EPANET_DDA | 50 | FAILED (SimulationTimeoutError: EPANET_DDA exceeded the 60-second timeout) | | | | |
| CURRENT_WNTR_PDD | 100 | FAILED (SimulationTimeoutError: CURRENT_WNTR_PDD exceeded the 60-second timeout) | | | | |
| EPANET_PDD | 100 | FAILED (SimulationTimeoutError: EPANET_PDD exceeded the 60-second timeout) | | | | |
| WNTR_DDA | 100 | FAILED (SimulationTimeoutError: WNTR_DDA exceeded the 60-second timeout) | | | | |
| EPANET_DDA | 100 | FAILED (SimulationTimeoutError: EPANET_DDA exceeded the 60-second timeout) | | | | |
| CURRENT_WNTR_PDD | 250 | FAILED (SimulationTimeoutError: CURRENT_WNTR_PDD exceeded the 60-second timeout) | | | | |
| EPANET_PDD | 250 | FAILED (SimulationTimeoutError: EPANET_PDD exceeded the 60-second timeout) | | | | |
| WNTR_DDA | 250 | FAILED (SimulationTimeoutError: WNTR_DDA exceeded the 60-second timeout) | | | | |
| EPANET_DDA | 250 | FAILED (SimulationTimeoutError: EPANET_DDA exceeded the 60-second timeout) | | | | |
| CURRENT_WNTR_PDD | 500 | FAILED (SimulationTimeoutError: CURRENT_WNTR_PDD exceeded the 60-second timeout) | | | | |
| EPANET_PDD | 500 | FAILED (SimulationTimeoutError: EPANET_PDD exceeded the 60-second timeout) | | | | |
| WNTR_DDA | 500 | FAILED (SimulationTimeoutError: WNTR_DDA exceeded the 60-second timeout) | | | | |
| EPANET_DDA | 500 | FAILED (SimulationTimeoutError: EPANET_DDA exceeded the 60-second timeout) | | | | |

## CRITICAL METHODOLOGICAL FINDING: wrapper confound check

**Confound detected: True**

CONFIRMED: at least one arm succeeds in milliseconds when called directly (unwrapped) but times out through HydraulicSimulator._run_with_timeout (wrapped) at the same size. This means the process-completion detection in the fork-based timeout wrapper -- not the solver engine or demand model -- is the actual scalability driver Milestone 8 observed. The apparent 'PDD bottleneck' is very likely this wrapper artifact, not a genuine PDD/engine performance limitation. Root-causing the exact OS/signal-handling mechanism (this sandbox's SIGCHLD/process-reaping behavior is the leading suspect, given the unrelated zombie-process accumulation already observed in this same environment across Milestones 7B/8) is out of scope for this diagnostic milestone.

| size | arm | unwrapped status | unwrapped ms | wrapped status | wrapped ms | confound |
|---|---|---|---|---|---|---|
| 25 | CURRENT_WNTR_PDD | OK | 47.954 | FAILED |  | True |
| 25 | EPANET_PDD | OK | 21.102 | OK | 58.063 | False |
| 25 | WNTR_DDA | OK | 29.789 | FAILED |  | True |
| 25 | EPANET_DDA | OK | 15.383 | OK | 55.891 | False |
| 50 | CURRENT_WNTR_PDD | OK | 87.439 | FAILED |  | True |
| 50 | EPANET_PDD | OK | 24.839 | FAILED |  | True |
| 50 | WNTR_DDA | OK | 208.855 | FAILED |  | True |
| 50 | EPANET_DDA | OK | 18.907 | FAILED |  | True |

## Section 4/5: hydraulic equivalence (CURRENT_WNTR_PDD vs EPANET_PDD)

Overlap sizes (both arms completed): [10]

### Benign case

| n | max abs pressure diff (m) | max rel demand diff | below-min sets match | NUMERICALLY_CLOSE | SAFETY_DECISION_EQUIVALENT |
|---|---|---|---|---|---|
| 10 | 0.0000 | 0.0001 | True | True | True |

### Stressed case (every reservoir base_head scaled to 40% of nominal)

| n | max abs pressure diff (m) | max rel demand diff | below-min sets match | NUMERICALLY_CLOSE | SAFETY_DECISION_EQUIVALENT |
|---|---|---|---|---|---|
| 10 | 0.0000 | 12.5146 | True | True | True |

## Section 6: plan-verification semantic check (N=10)

Plan rejection agreement rate: 1.000

| plan | decision A | decision B | rejection codes A | rejection codes B | agree |
|---|---|---|---|---|---|
| NO_ACTION | VERIFIED | VERIFIED | [] | [] | True |
| LOW_IMPACT | VERIFIED | VERIFIED | [] | [] | True |
| STRESSFUL | REJECTED | REJECTED | ['PRESSURE_BELOW_MINIMUM'] | ['PRESSURE_BELOW_MINIMUM'] | True |

## Section 7: water-quality/incident path check

production's own simulate_incident already runs EPANET_PDD's exact engine+demand-model combination; this checks whether that already-shared path succeeds at the sizes where PDD hydraulics-only (Section 2) succeeds.

| n | status | elapsed ms | has concentration output | source node represented |
|---|---|---|---|---|
| 10 | OK | 51.74 | True | True |

## Section 9: decision

Largest CURRENT_WNTR_PDD network completed: 10
Largest EPANET_PDD network completed: 25

| criterion | met |
|---|---|
| 1_materially_better_scalability | True |
| 2_meaningfully_larger_network | False |
| 3_hydraulically_close_on_overlap | True |
| 4_safety_decision_equivalent_on_plans | True |
| 5_acceptable_under_stress | True |
| 6_water_quality_capability_preserved | True |
| 7_no_safety_threshold_weakened | True |

**Decision: PDD_SCALABILITY_BLOCKER_REMAINS**

M9 scientifically unblocked: True. M8/M8.5 diagnose a solver-PERFORMANCE limitation, not a data-integrity or model-input correctness issue -- HydroCore's own inputs/training/calibration are untouched and unaffected by this finding at the network sizes M9's capacity study is expected to use. M8.5's own confound check further shows the limitation most likely sits in HydraulicSimulator._run_with_timeout's fork-based process-completion detection (this execution sandbox specifically), not in PDD or any specific solver engine -- an important correction to M8's own attribution, but still a backend-engineering question orthogonal to HydroCore's architecture/training/calibration. Engineering remediation of the hydraulic backend (if pursued) remains separate from M9.

Production source unchanged: True. locked tests opened: before=False, after=False.
