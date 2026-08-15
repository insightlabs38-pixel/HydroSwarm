# Milestone 8 summary: backend scalability and memory stability

> **SUPERSEDED IN PART by Milestone 8.5a** (`reports/evaluation/hydrocore-v5/m8-5a-summary.md`,
> `m8-5a-execution.json`, `m8-5a-scale.json`). The FAILED rows below at N>=25 were real observations of the
> measured wrapped execution path, but this document's implicit attribution of that failure to PDD/WNTR/EPANET
> solver scalability is **RETRACTED**. Milestone 8.5a root-caused the failure to a `HydraulicSimulator.
> _run_with_timeout` IPC defect (`process.join(timeout)` called before ever draining `result_queue`, causing a
> real Python `multiprocessing.Queue` "joining processes that use queues" deadlock once a PDD result's pickled
> size exceeded the OS pipe's buffered capacity -- around 25-49 junctions, exactly this table's own failure
> threshold) and fixed it; corrected measurements complete successfully through N=250 with numerical parity to
> direct/unwrapped execution. **The FAILED timings/timeouts below must NOT be read as PDD/WNTR/EPANET
> performance measurements.** `PYG_NOT_JUSTIFIED` (this document's own neural-inference-scaling finding) is
> unaffected and stands.

Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED (4182612 parameters, checkpoint sha256=44a2721394d95985...)

## 8.1 Network-size scaling

| target N | actual N | status | import ms | sig-lib build ms | feature ms | neural ms | classical ms | fusion ms | calib ms | sampling ms | plan ms | verify ms | total incident ms | RSS delta MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 10 | OK | 67.35 | 505.10 | 2.98 | 333.32 | 0.64 | 0.13 | 0.0474 | 0.92 | 0.21 | 131.23 | 469.48 | 53.19 |
| 25 | - | FAILED (SimulationTimeoutError: hydraulics exceeded the 60-second timeout) | | | | | | | | | | | |
| 50 | - | FAILED (SimulationTimeoutError: hydraulics exceeded the 60-second timeout) | | | | | | | | | | | |
| 100 | - | FAILED (SimulationTimeoutError: hydraulics exceeded the 60-second timeout) | | | | | | | | | | | |
| 250 | - | FAILED (SimulationTimeoutError: hydraulics exceeded the 60-second timeout) | | | | | | | | | | | |
| 500 | - | FAILED (SimulationTimeoutError: hydraulics exceeded the 60-second timeout) | | | | | | | | | | | |

**Primary finding:** HydraulicSimulator._prepared_network() hard-codes demand_model=PDD for every hydraulic/incident simulation; PDD convergence in WNTR's native solver becomes impractical (exceeds the timeout) around 25-49 junctions on ANY topology tested (line/tree/grid), while the same solver handles 500 nodes in ~1.3s under plain DDA. This is a classical-hydraulics-layer bottleneck, not a HydroCore/neural bottleneck.

### Neural inference in isolation (synthetic batches, decoupled from the PDD hydraulics bottleneck above)

| n nodes | neural inference ms |
|---|---|
| 10 | 14.267 |
| 25 | 16.830 |
| 50 | 17.831 |
| 100 | 24.310 |
| 250 | 29.146 |
| 500 | 42.534 |
| 1000 | 100.605 |
| 2000 | 134.717 |

## 8.2 Long-lived process test

200 sequential incidents on the fixed golden-reference network; 200 iterations per isolated repeated-stage loop.

| loop | n | post-warmup slope (MB/iter) | first-20 mean MB | last-20 mean MB | peak MB | plateaus |
|---|---|---|---|---|---|---|
| full_sequential_incidents | 200 | -0.0006 | 1107.29 | 778.19 | 1107.37 | True |
| repeated_neural_inference | 200 | 0.0005 | 782.48 | 782.87 | 782.87 | True |
| repeated_wntr_simulation | 200 | 0.0000 | 782.87 | 782.88 | 782.88 | True |
| repeated_import | 200 | 0.0000 | 782.88 | 782.89 | 782.89 | True |
| repeated_sample_analysis | 200 | -0.0000 | 782.89 | 782.89 | 782.89 | True |

## 8.3 Caching benchmark

Material-improvement bar (predeclared): warm/cold latency ratio <= 0.5.

| cache | exists | cold ms | warm ms | speedup ratio | verdict |
|---|---|---|---|---|---|
| network parsing | False | 0.573 | 0.392 | n/a | NO_CACHE_EXISTS |
| static features (HydraulicContextCache) | True | 72.089 | 0.129 | 0.002 | MATERIAL_IMPROVEMENT |
| hydraulic states (SimulationResultCache) | True | 53.278 | 1.528 | 0.029 | MATERIAL_IMPROVEMENT |
| signature libraries (SignatureCache) | True | 216.928 | 1.226 | 0.006 | MATERIAL_IMPROVEMENT |

## 8.4 Framework decision

Primary scalability finding (not itself a PyG question -- see below): HydraulicSimulator._prepared_network() hard-codes demand_model=PDD for every hydraulic/incident simulation; PDD convergence in WNTR's native solver becomes impractical (exceeds the timeout) around 25-49 junctions on ANY topology tested (line/tree/grid), while the same solver handles 500 nodes in ~1.3s under plain DDA. This is a classical-hydraulics-layer bottleneck, not a HydroCore/neural bottleneck.

Most PyG-relevant component: hydroswarm.model.layers.EdgeAwareGraphConv / DualChannelGraphConv (hand-rolled per-batch-item Python message-passing loop; no torch_geometric import exists anywhere in this codebase today).
Neural-inference scaling ratio (largest/smallest N tested): 9.442457694425828
Node-count ratio (largest/smallest N tested): 200.0
Superlinear neural-inference scaling: False
Long-lived-process memory plateaus: True
Caching materially helps anywhere: True

**Decision: KEEP_CURRENT_IMPLEMENTATION_NO_MEASURED_PROBLEM**

No measured problem with variable graph batching, graph-op scalability, custom message-passing performance, or memory stability was found at the tested scales (synthetic neural-inference-only sweep up to 2000 nodes, decoupled from the separate PDD hydraulics bottleneck above). Per experiments.txt 8.4/12, PyTorch Geometric is not justified on this evidence; no framework migration for cleaner code alone. The real scalability priority this milestone surfaced is the classical-hydraulics PDD bottleneck, not the neural/graph layer.

locked tests opened: before=False, after=False.
