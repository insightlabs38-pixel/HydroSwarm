# Milestone 6 summary: temporal realism -- cadence, detection delay, sensor observability

Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED (/workspace/HydroSwarm/experiments/runs/hydrocore-v5-causal/A-seed31874/20260814T065947Z-44557eae/model-export.safetensors)
Calibration: B_DEPTH_AWARE (Milestone 3 frozen scheme, refit identically here).

## 6.1 Telemetry cadence

**Milestone-6 correction applied this revision** (see `correction_note` in m6-cadence.json for the full text): the matched-elapsed-time analysis had an off-by-one -- N reports spaced `cadence` apart span (N-1)*cadence minutes of elapsed time, not N*cadence. Matched-elapsed-time evidence is now built directly from real timestamps (`timestamp <= first_timestamp + checkpoint`) with explicit per-row and cross-cadence alignment assertions, never from a `depth = checkpoint // cadence` formula. Fixed-report-count elapsed-time labels now report the ACTUAL elapsed minutes read from the resulting series, never `depth * cadence`. Paired per-incident predicted-node identity analysis was added to distinguish 'aggregate top-1 accuracy is equal' from 'predictions are identical'. Predictor, calibration, alpha, K, incident pool, topology, and seeds are unchanged from the prior revision.

Practicality: used cadences [15, 30, 60] min (5 min dropped -- see `practicality_note` in m6-cadence.json for the measured 180s-timeout-vs-0.04s evidence, unchanged this revision).

### Fixed PHYSICAL elapsed-time comparison (corrected)

N incidents: 48. Reporting resolution (alignment tolerance): 15 min. Every row below passed a per-row actual-elapsed-vs-checkpoint assertion AND a cross-cadence alignment assertion before being compared (script raises, does not silently compare, on misalignment). Max cadence-sensitivity spread at a VERIFIED-ALIGNED elapsed-time boundary: **0.00pp** (bar: 10pp). Strongly cadence-sensitive: **False**.

| checkpoint (min) | actual elapsed @15min | actual elapsed @30min | actual elapsed @60min | top1 @15min | top1 @30min | top1 @60min | spread (pp) |
|---|---|---|---|---|---|---|---|
| 60 | 60.0 | 60.0 | 60.0 | 0.438 | 0.438 | 0.438 | 0.00 |
| 120 | 120.0 | 120.0 | 120.0 | 0.625 | 0.625 | 0.625 | 0.00 |
| 180 | 180.0 | 180.0 | 180.0 | 0.812 | 0.812 | 0.812 | 0.00 |
| 240 | 240.0 | 240.0 | 240.0 | 0.812 | 0.812 | 0.812 | 0.00 |
| 360 | 360.0 | 360.0 | 360.0 | 1.000 | 1.000 | 1.000 | 0.00 |

### Fixed REPORT-COUNT comparison (corrected elapsed labels)

depth=1 is a single observation and therefore spans 0 minutes of elapsed time regardless of cadence (first_timestamp == last_timestamp for every cadence). Identical behavior at depth=1 is EXPECTED and is NEVER used as evidence of report-count/elapsed-time conflation; the earliest meaningful fixed-report-count comparison in this analysis is depth >= 2.

| depth (reports) | actual elapsed @15min | actual elapsed @30min | actual elapsed @60min | top1 @15min | top1 @30min | top1 @60min | spread (pp) |
|---|---|---|---|---|---|---|---|
| 1  (depth=1: NOT evidence -- see caveat above) | 0.0 | 0.0 | 0.0 | 0.250 | 0.250 | 0.250 | 0.00 |
| 2 | 15.0 | 30.0 | 60.0 | 0.438 | 0.438 | 0.438 | 0.00 |
| 3 | 30.0 | 60.0 | 120.0 | 0.438 | 0.438 | 0.625 | 18.75 |
| 4 | 45.0 | 90.0 | 180.0 | 0.438 | 0.625 | 0.812 | 37.50 |
| 6 | 75.0 | 150.0 | 300.0 | 0.625 | 0.812 | 0.938 | 31.25 |
| 8 | 105.0 | 210.0 | 420.0 | 0.625 | 0.812 | 1.000 | 37.50 |

### Paired prediction-identity analysis (fixed report count, depth >= 2)

Aggregate top-1 accuracy equal does NOT by itself imply 'predictions are identical' -- this table verifies per-incident predicted-node identity across cadences directly. L1 = sum of absolute differences between the two probability vectors over the same node ordering.

**report_count_conflation_at_depth2 (paired, bar=0.90): True**

| depth | n matched | fraction identical predicted_node (all cadences) | 15v30 agreement | 15v60 agreement | 30v60 agreement | mean L1 dist | max L1 dist |
|---|---|---|---|---|---|---|---|
| 2 | 48 | 1.000 | 1.000 (n=48) | 1.000 (n=48) | 1.000 (n=48) | 0.0253 | 0.1011 |
| 3 | 48 | 0.812 | 1.000 (n=48) | 0.812 (n=48) | 0.812 (n=48) | 0.2690 | 1.9695 |
| 4 | 48 | 0.625 | 0.812 (n=48) | 0.625 (n=48) | 0.812 (n=48) | 0.5078 | 1.9696 |
| 6 | 48 | 0.625 | 0.812 (n=48) | 0.625 (n=48) | 0.812 (n=48) | 0.5023 | 1.9698 |
| 8 | 48 | 0.625 | 0.812 (n=48) | 0.625 (n=48) | 0.812 (n=48) | 0.5022 | 1.9694 |

## 6.2 Detection delay

N incidents: 48. Onset never fed as model input: **True**.

| delay (min) | n | top1 | top3 | mrr | mean reports available |
|---|---|---|---|---|---|
| 0 | 48 | 0.250 | 0.750 | 0.521 | 8.00 |
| 60 | 48 | 0.938 | 1.000 | 0.969 | 12.00 |
| 180 | 48 | 1.000 | 1.000 | 1.000 | 20.00 |
| 360 | 48 | 1.000 | 1.000 | 1.000 | 32.00 |

## 6.3 Irregular/asynchronous telemetry (development-only, diagnostic)

N incidents: 24 at 30min cadence, depth=6 reports.

| stress case | n | top1 | top1 degradation vs clean (pp) |
|---|---|---|---|
| CLEAN_BASELINE | 24 | 0.750 | - |
| DELAYED_REPORTS | 24 | 0.750 | 0.00 |
| GAPS | 24 | 0.708 | 4.17 |
| PARTIAL_HISTORICAL_AVAILABILITY | 24 | 0.708 | 4.17 |
| TIMESTAMP_JITTER | 24 | 0.542 | 20.83 |
| UNEQUAL_SENSOR_INTERVALS | 24 | 0.500 | 25.00 |

## 6.4 Sensor coverage and placement

N junctions: 4. Evidence depth: 3 reports (EARLY bucket). K=3.

| budget | policy | n | top1 | top3 | mean candidate size | gate-pass rate | median samples required |
|---|---|---|---|---|---|---|---|
| 1 | DEGREE_CENTRALITY | 12 | 0.500 | 0.750 | 3.75 | 0.250 | 1.0 |
| 1 | HYDRAULIC_OBSERVABILITY | 12 | 0.250 | 0.750 | 4.00 | 0.000 | 1.0 |
| 1 | RANDOM | 12 | 0.583 | 1.000 | 3.75 | 0.083 | 1.0 |
| 2 | DEGREE_CENTRALITY | 12 | 1.000 | 1.000 | 2.00 | 0.750 | 0.0 |
| 2 | HYDRAULIC_OBSERVABILITY | 12 | 1.000 | 1.000 | 2.17 | 1.000 | 0.0 |
| 2 | RANDOM | 12 | 0.833 | 1.000 | 2.42 | 0.667 | 0.0 |
| 3 | DEGREE_CENTRALITY | 12 | 1.000 | 1.000 | 1.75 | 0.750 | 0.0 |
| 3 | HYDRAULIC_OBSERVABILITY | 12 | 1.000 | 1.000 | 1.75 | 1.000 | 0.0 |
| 3 | RANDOM | 12 | 1.000 | 1.000 | 1.50 | 1.000 | 0.0 |

## 6.5 Conditional architecture change

**Decision: NEW_TEMPORAL_REPRESENTATION_EXPERIMENT_WARRANTED**

hydroswarm.model.encoders.TemporalEncoder already encodes masked histories using elapsed timestamps rather than array position (its own docstring); HydraulicFeatureBuilder.build already feeds real per-timestep age (elapsed seconds since the latest observation) into temporal_features and quality_features, plus real absolute timestamps into the batch. An explicit elapsed-time representation already exists in HydroCore's architecture -- confirmed by inspection, not assumed. This remains true regardless of the corrected cadence finding below: HydroCore already has elapsed-time-aware temporal features, so even if the problem remains, the recommended next experiment continues to be cadence-diversified causal-prefix training and/or a controlled timestamp-conditioning ablation at matched model size -- never a new temporal architecture.

(A) Corrected matched-physical-time result: max cadence-sensitivity spread at a VERIFIED-ALIGNED elapsed-time boundary is 0.00pp (bar: 10pp) -- below the predeclared bar; no material cadence-dependent behavior at matched physical time. (B) Fixed-report-count (depth=2; elapsed spans by cadence: 15min cadence=15.0min elapsed, 30min cadence=30.0min elapsed, 60min cadence=60.0min elapsed) PAIRED predicted-node identity: 1.000 of matched incidents predict the exact SAME node across every available cadence (bar: 0.90; mean L1 probability-vector distance across cadence pairs: 0.025282343158967616). This is aggregate-and-paired evidence of report-count/time conflation at the earliest meaningful depth: paired predictions, not merely aggregate accuracy, are substantially invariant. Since HydroCore's TemporalEncoder already encodes elapsed timestamps (not array position) yet at least one of the above signals still shows a cadence/report-count problem, the existing representation's GENERALIZATION -- not its presence -- is implicated: train_records/calibration_records/M1's causal-prefix corpus all use a fixed hourly reporting cadence, so the model has never actually been trained on report sequences where consecutive reports span 15 or 30 minutes rather than 60. The correctly targeted follow-up is therefore a training-distribution diversification (cadence-varied causal-prefix corpus, i.e. re-running Milestone 1's arm construction with randomized inter-report spacing) and/or a controlled ablation of the TemporalEncoder's timestamp-conditioning at matched model size -- NOT a new architecture built from scratch. Not executed in this correction (a full retrain is explicitly out of scope); recorded here as the concrete next milestone recommendation, per experiments.txt's own decision tree ('M6: is cadence/delay robustness poor? YES -> test explicit time encoding').

## Limitations

6.1/6.2/6.3 use a single canonical (golden-reference, 4-junction) topology with full sensor coverage (placement isolated to 6.4); cadence/delay behavior on larger or differently-shaped networks was not tested here (topology diversity is Milestone 7's scope). 6.4's HYDRAULIC_OBSERVABILITY placement and identifiability diagnostic are network-structural (no incident-specific information), matching Milestone 1.2's identifiability baseline's own scope limits. 6.3's stress cases are development-only diagnostics per the milestone text and are not used to select or tune anything.

## Milestone-6 correction scope

This revision recomputes ONLY 6.1 (telemetry cadence): a matched-elapsed-time off-by-one and a fixed-report-count elapsed-time mislabeling, both described above. 6.2 (detection delay), 6.3 (irregular telemetry), and 6.4 (sensor placement) are PRESERVED byte-for-byte from the prior revision -- their own timing/elapsed-time constructions were verified by inspection to already derive from real timestamps (never a depth*cadence-style formula), so no direct implementation inconsistency from the 6.1 fix carries over to them; they were not rerun. The frozen Milestone-1 predictor, Milestone-3 B_DEPTH_AWARE calibrator, alpha=0.1, K=3, topology, incident pool construction, and random seeds are all unchanged from the prior revision throughout this correction.
