# Milestone 6 summary: temporal realism -- cadence, detection delay, sensor observability

Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED (/workspace/HydroSwarm/experiments/runs/hydrocore-v5-causal/A-seed31874/20260814T065947Z-44557eae/model-export.safetensors)
Calibration: B_DEPTH_AWARE (Milestone 3 frozen scheme, refit identically here).

## 6.1 Telemetry cadence

Practicality: used cadences [15, 30, 60] min (5 min dropped -- see `practicality_note` in m6-cadence.json for the measured 180s-timeout-vs-0.04s evidence).
N incidents: 48. Max cadence-sensitivity spread at matched elapsed time: **18.75pp** (bar: 10pp). Strongly cadence-sensitive: **True**.

| elapsed (min) | top1 @15min | top1 @30min | top1 @60min | spread (pp) |
|---|---|---|---|---|
| 60 | 0.438 | 0.438 | 0.250 | 18.75 |
| 120 | 0.625 | 0.625 | 0.438 | 18.75 |
| 180 | nan | 0.812 | 0.625 | 18.75 |
| 240 | nan | 0.812 | 0.812 | 0.00 |

Report-count invariance at FIXED depth (literal reading of 'six samples != one fixed duration'; depths 1-2 only flat-spread bar: 5pp) -- **report_count_conflation_at_low_depth (depths 1-2): True**:

| depth (reports) | elapsed @15min | elapsed @30min | elapsed @60min | top1 @15min | top1 @30min | top1 @60min | spread (pp) |
|---|---|---|---|---|---|---|---|
| 1 | 15 | 30 | 60 | 0.250 | 0.250 | 0.250 | 0.00 |
| 2 | 30 | 60 | 120 | 0.438 | 0.438 | 0.438 | 0.00 |
| 3 | 45 | 90 | 180 | 0.438 | 0.438 | 0.625 | 18.75 |
| 4 | 60 | 120 | 240 | 0.438 | 0.625 | 0.812 | 37.50 |
| 6 | 90 | 180 | 360 | 0.625 | 0.812 | 0.938 | 31.25 |
| 8 | 120 | 240 | 480 | 0.625 | 0.812 | 1.000 | 37.50 |

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
| TIMESTAMP_JITTER | 24 | 0.542 | 20.83 |
| UNEQUAL_SENSOR_INTERVALS | 24 | 0.500 | 25.00 |
| DELAYED_REPORTS | 24 | 0.750 | 0.00 |
| GAPS | 24 | 0.708 | 4.17 |
| PARTIAL_HISTORICAL_AVAILABILITY | 24 | 0.708 | 4.17 |

## 6.4 Sensor coverage and placement

N junctions: 4. Evidence depth: 3 reports (EARLY bucket). K=3.

| budget | policy | n | top1 | top3 | mean candidate size | gate-pass rate | median samples required |
|---|---|---|---|---|---|---|---|
| 1 | RANDOM | 12 | 0.583 | 1.000 | 3.75 | 0.083 | 1.0 |
| 1 | DEGREE_CENTRALITY | 12 | 0.500 | 0.750 | 3.75 | 0.250 | 1.0 |
| 1 | HYDRAULIC_OBSERVABILITY | 12 | 0.250 | 0.750 | 4.00 | 0.000 | 1.0 |
| 2 | RANDOM | 12 | 0.833 | 1.000 | 2.42 | 0.667 | 0.0 |
| 2 | DEGREE_CENTRALITY | 12 | 1.000 | 1.000 | 2.00 | 0.750 | 0.0 |
| 2 | HYDRAULIC_OBSERVABILITY | 12 | 1.000 | 1.000 | 2.17 | 1.000 | 0.0 |
| 3 | RANDOM | 12 | 1.000 | 1.000 | 1.50 | 1.000 | 0.0 |
| 3 | DEGREE_CENTRALITY | 12 | 1.000 | 1.000 | 1.75 | 0.750 | 0.0 |
| 3 | HYDRAULIC_OBSERVABILITY | 12 | 1.000 | 1.000 | 1.75 | 1.000 | 0.0 |

## 6.5 Conditional architecture change

**Decision: NEW_TEMPORAL_REPRESENTATION_EXPERIMENT_WARRANTED**

hydroswarm.model.encoders.TemporalEncoder already encodes masked histories using elapsed timestamps rather than array position (its own docstring); HydraulicFeatureBuilder.build already feeds real per-timestep age (elapsed seconds since the latest observation) into temporal_features and quality_features, plus real absolute timestamps into the batch. An explicit elapsed-time representation already exists in HydroCore's architecture -- confirmed by inspection, not assumed.

6.1 found BOTH: (a) matched-elapsed-time cadence sensitivity at or above the predeclared 10pp bar (18.75pp measured), and (b) at the two EARLIEST report counts specifically (depth=1 and depth=2 -- NOT the whole EARLY 1-3 bucket: depth=3 breaks this pattern, see below), top-1 is BIT-FOR-BIT IDENTICAL across all three cadences (0.00pp spread) despite up to a 4x difference in the elapsed time those 1-2 reports actually span (15/30/60 min at depth=1; 30/60/120 min at depth=2) -- the literal, and strongest possible, signature of the milestone's own stated concern, 'a model should not accidentally equate six samples with one fixed physical duration.' This exact invariance does NOT continue at depth=3 (18.75pp spread reappears there -- real cadence sensitivity returns), so the finding is reported precisely as 'depths 1-2 only', not generalized to the full EARLY bucket. Since HydroCore's TemporalEncoder already encodes elapsed timestamps (not array position) yet still shows this pattern, the existing representation's GENERALIZATION -- not its presence -- is implicated: train_records/calibration_records/M1's causal-prefix corpus all use a fixed hourly reporting cadence, so the model has never actually been trained on report sequences where consecutive reports span 15 or 30 minutes rather than 60. The correctly targeted follow-up is therefore a training-distribution diversification (cadence-varied causal-prefix corpus, i.e. re-running Milestone 1's arm construction with randomized inter-report spacing) and/or a controlled ablation of the TemporalEncoder's timestamp-conditioning at matched model size -- NOT a new architecture built from scratch, and not simply adding more temporal-encoding complexity on top of the encoder that already exists. Not executed in this milestone (a full retrain is out of scope for an evaluation-only script against the frozen Milestone-1 predictor); recorded here as the concrete next milestone recommendation, per experiments.txt's own decision tree ('M6: is cadence/delay robustness poor? YES -> test explicit time encoding').

## Limitations

6.1/6.2/6.3 use a single canonical (golden-reference, 4-junction) topology with full sensor coverage (placement isolated to 6.4); cadence/delay behavior on larger or differently-shaped networks was not tested here (topology diversity is Milestone 7's scope). 6.4's HYDRAULIC_OBSERVABILITY placement and identifiability diagnostic are network-structural (no incident-specific information), matching Milestone 1.2's identifiability baseline's own scope limits. 6.3's stress cases are development-only diagnostics per the milestone text and are not used to select or tune anything.
