# Milestone 8.5a summary: corrected wrapped simulation execution and scaling attribution

**Supersedes** the PDD-scalability-blocker attribution in `m8-summary.md` and the `PDD_SCALABILITY_BLOCKER_REMAINS` decision token in `m8_5-summary.md`. See `reports/evaluation/hydrocore-v5/m8-5a-execution.json` and `m8-5a-scale.json` for full raw data.

## What M8/M8.5 got right and wrong

1. M8 correctly observed a failure in the measured wrapped execution path at 25-49 grid junctions.
2. M8 did NOT validly establish that PDD/WNTR/EPANET itself was the scaling bottleneck -- that attribution is **RETRACTED** by this milestone.
3. M8.5 correctly demonstrated that direct/unwrapped PDD completed in milliseconds on cases where the wrapped path failed, and correctly flagged (without confirming) that the wrapper's own process-completion detection, not the solver, was implicated.
4. M8.5 explicitly speculated a SIGCHLD/zombie-reaping mechanism. **This milestone tested that hypothesis directly and REFUTES it**: the child's real `/proc` OS state during every observed false timeout was `S` (sleeping/blocked-on-IO), never `Z` (zombie).
5. The actual mechanism, established with root_cause_established=YES (`m8-5a-execution.json:root_cause`): `HydraulicSimulator._run_with_timeout` called `process.join(timeout)` BEFORE ever reading `result_queue`. A child's return value only finishes crossing the OS pipe once the parent drains it; nothing drained it until join() returned, and join() could not return until the child exited, which it could not do until the pipe drained -- Python's own documented "joining processes that use queues" deadlock. Once a real PDD result's pickled size crosses the pipe's buffered capacity (empirically ~60-100KB on this host; a real 25-node grid's PDD result is ~68KB), the child blocks mid-write and is misreported as hung.
6. Previously reported wrapped-path timeout values from M8/M8.5 must NOT be interpreted as solver-performance measurements -- they measured this IPC defect, not PDD/WNTR/EPANET performance.

## Phase 1: root-cause diagnostic

Classification: **IPC_BLOCKING_ON_UNDRAINED_QUEUE_PIPE** (root cause established: YES)

Payload-size threshold where join-first first falsely times out: 100000 bytes.
Zombie OS state ever observed during a false timeout: False.
Running/blocked OS state observed during a false timeout: True.

| payload bytes | join-first timed out | join-first OS state | drain-first timed out | confound |
|---|---|---|---|---|
| 1000 | False | GONE | False | False |
| 16000 | False | GONE | False | False |
| 60000 | False | GONE | False | False |
| 100000 | True | S | False | True |
| 300000 | True | S | False | True |
| 1000000 | True | S | False | True |
| 5000000 | True | S | False | True |

Real-simulation reproduction (ABCD minimal comparison, C=wrapped production method, D=direct/unwrapped):

| N | C status | C wall ms | D status | D wall ms | confound reproduced |
|---|---|---|---|---|---|
| 10 | OK | 52.176 | OK | 22.502 | False |
| 25 | FAILED | 8036.399 | OK | 38.655 | True |

Pre-fix baseline (current/unfixed wrapper, N=50, 30 runs, 5.0s timeout): 30/30 falsely timed out; leaked children detected: False.

## Phase 2: corrected wrapped execution path

`HydraulicSimulator._run_with_timeout` (src/hydroswarm/simulation/wrapper.py) now drains `result_queue` WHILE waiting (bounded polling `result_queue.get(timeout=...)` against the deadline, with an `is_alive()` liveness check on `queue.Empty` to still fail fast on a genuine crash-without-a-result), instead of `process.join(timeout)` followed by `result_queue.get_nowait()`. The hard timeout, terminate/kill escalation, exception propagation, and IPC-resource cleanup are all preserved; a genuine hang still raises `SimulationTimeoutError` within the configured `timeout_seconds`. Regression tests added to `tests/scientific/test_simulator_extended.py` (large-result transfer, child exception, genuine timeout, repeated-run leak check, and a real N=25 PDD run at the exact size that previously falsely timed out).

## Phase 3: numerical parity

Predeclared thresholds: max abs pressure diff <= 1e-06 m, max abs demand diff <= 1e-09 m3/s.

**Overall parity: PASS**

| N | max abs pressure diff (m) | max abs demand diff (m3/s) | pass |
|---|---|---|---|
| 10 | 2.842e-14 | 0.000e+00 | True |
| 25 | 2.842e-14 | 0.000e+00 | True |
| 50 | 2.842e-14 | 0.000e+00 | True |

## Phase 4: scale characterization (corrected wrapper)

| N | direct WNTR ms (mean) | wrapped WNTR ms (mean) | direct EPANET ms (mean) | wrapped EPANET ms (mean) | solver ms | IPC ms | reaping ms | wrapper overhead ms |
|---|---|---|---|---|---|---|---|---|
| 10 | 18.549 | 48.959 | 8.978 | 41.414 | 20.336 | 2.718 | 14.616 | 25.754 |
| 25 | 39.304 | 73.253 | 11.680 | 45.987 | 42.599 | 3.064 | 14.818 | 27.514 |
| 50 | 106.988 | 279.543 | 16.059 | 52.668 | 240.531 | 3.140 | 18.252 | 30.959 |
| 100 | 237.700 | 413.128 | 24.139 | 63.261 | 367.416 | 3.428 | 18.816 | 36.052 |
| 250 | 782.150 | 974.445 | 48.209 | 92.623 | 828.586 | 5.013 | 20.227 | 40.452 |

Repeated-run leak check (corrected wrapper, N=50, 15 runs): 0 runs left a leaked child.

## Phase 5: environment sensitivity

Tested in the single execution environment available to this milestone (see VALIDATION note in `m8-5a-execution.json.environment`); a second, independently provisioned environment was not stood up per this milestone's own scope constraints ("do not spend substantial time provisioning unrelated infrastructure"). The mechanism identified (a general Python `multiprocessing.Queue` "join before drain" deadlock, gated by the OS pipe buffer size) is documented CPython behavior on any POSIX "fork"-context host, not specific to a sandboxing detail of this container -- but this milestone did not empirically confirm that in a second environment, so the conclusion is reported as MIXED_WRAPPER_AND_ENVIRONMENT_INTERACTION: a portable wrapper defect (confirmed here), whose exact failure threshold (pipe buffer size) is itself an environment-dependent parameter.

## Phase 7: decision

**WRAPPED_EXECUTION_BLOCKER_RESOLVED**

All four arms (direct/wrapped x WNTR-native/EPANET-backed PDD) completed successfully at every tested size up to N=250 with the corrected wrapper, and repeated runs left zero leaked children. Milestone 8's original PDD-scalability-blocker attribution is retracted: the corrected measurements show no genuine solver scaling limitation at the sizes actually tested; the original ceiling was the wrapped-execution IPC defect M8.5a fixed.

LARGE_NETWORK_EXPERIMENTS_REOPENED: YES

PYG_NOT_JUSTIFIED: retained unchanged -- this milestone is a wrapper/IPC correction, not a neural-layer or graph-batching finding; M8's own synthetic neural-inference-scaling evidence is untouched.

locked tests opened: before=False, after=False.
