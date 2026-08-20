# HydroCore-v5 synthetic dataset card

## Scope

HydroCore-v5 is trained, calibrated, developed, and finally evaluated on governed WNTR/EPANET-generated synthetic water-network scenarios. This card distinguishes **physical scenarios**, **derived causal-prefix/evaluation rows**, **development trajectories**, and the **one-time locked populations** so counts are not accidentally compared as if they represented the same unit.

No utility telemetry or field-incident outcome data establishes the reported performance.

## Split roles

| Role | Allowed use |
|---|---|
| Training | gradient optimization |
| Validation | model/recipe comparison before freeze |
| Calibration | conformal fitting only; no gradient/checkpoint selection |
| Development holdout / OOD development | repeated characterization before freeze |
| M10 development trajectories | production-path integration/authority evaluation |
| `locked_final_test` | one-time final applicable stress evaluation |
| `locked_topology_test` | one-time novel-topology/fail-closed evaluation |

The governing split policy was frozen before the final program; physical scenarios are split before causal prefixes/augmentations are generated.

## Causal-prefix foundation

The M1 causal-prefix foundation used one `golden-reference` family with 970 physical scenarios:

| Split | Physical scenarios |
|---|---:|
| train | 600 |
| validation | 100 |
| calibration | 150 |
| development_holdout | 120 |
| total | 970 |

Each physical scenario can be viewed at causal depths `1, 2, 3, 4, 6, 12, 25` without moving across splits. See [M1 prefix dataset artifact](../reports/evaluation/hydrocore-v5/m1-prefix-dataset.json).

This M1 population is important lineage, but it is **not** the whole final training story: later M9 work established multi-topology interleaved training.

## Final selected training population

The selected M9.6 `ARM_B_M9_6` run trained on **600 physical training scenarios** across three topology families, with equal family weighting:

| Trained family | Physical training scenarios |
|---|---:|
| `golden-reference` | 200 |
| `branched-loop` | 200 |
| `loop-grid` | 200 |
| total | 600 |

The final selected run completed exactly 1,350 optimizer steps. Across 20 epochs, the training record reports equal family exposure (3,600 scenario exposures per family). See [selected training record](../reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed20260814.json).

The M9.6 evaluation manifest also includes three unseen development families—`coastal-branch`, `tree-branch`, and `dense-loop`—for generalization characterization. Its `9,660` calibration rows and `23,940` prediction rows are **derived evaluation rows across seeds/depths/families**, not unique physical incident counts. See [M9.6 manifest](../reports/evaluation/hydrocore-v5/m9-6/m9-6-manifest.json).

## Supervision scope

The causal V5 corpus provides valid targets for the Sentinel task family. The final frozen release therefore declares `trained_tasks = ["sentinel"]`.

Model architecture/configuration may contain OOD, Scout, Strategist, and consequence-control heads, but the final freeze records those operational outputs as suppressed/non-authoritative. Dataset documentation must not infer supervision merely from a nonzero task weight or the presence of a head.

## M10 development populations

M10.4 evaluated full production-path trajectories across:

- 3 model seeds;
- 3 trained families (`golden-reference`, `branched-loop`, `loop-grid`);
- 3 unseen development families (`coastal-branch`, `tree-branch`, `dense-loop`);
- 7 condition kinds;
- 360 physical incidents total;
- 720 API incidents/trajectories total.

The seven conditions were nominal, low-coverage active sampling, sensor dropout, degraded sensor health, measurement noise, severity shift, and ambiguity/disagreement.

This was development evidence used before the finalist freeze. See [M10.4 population manifest](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-population-manifest.json).

## Final locked populations

The final design was frozen before population materialization. M11.6 then materialized two isolated populations:

| Locked split | Count | Purpose |
|---|---:|---|
| `locked_final_test` | 105 | 7 applicable condition kinds × 15 incidents |
| `locked_topology_test` | 20 | genuine novel-topology characterization + fail-closed authority |
| total | 125 | exactly-once final evaluation |

The locked-final matrix contains 15 incidents each for NOMINAL, AMBIGUITY_DISAGREEMENT, LOW_COVERAGE_ACTIVE_SAMPLING, MEASUREMENT_NOISE, SENSOR_DROPOUT, SENSOR_HEALTH_DEGRADED, and SEVERITY_SHIFT.

### Locked-topology generation

The topology population uses **four procedurally generated novel topologies**, with 9–12 junctions. The materialization novelty audit checked:

- file-byte novelty;
- network-hash novelty;
- graph-signature novelty;
- within-set network/signature uniqueness;
- disjointness from the prior topology evidence it could mechanically compare against.

All four passed the frozen novelty rule.

### Seed isolation and overlap control

The locked generator derives seeds in a namespace beginning at `2**31`, while the prior governed seed namespaces are below `2**31`. The materialization report records 125 unique canonical scenario hashes, zero within-set collisions, and seed-namespace disjointness by construction.

The report explicitly does **not** claim a historical canonical-scenario-hash comparison where prior schemas were not comparable. That limitation is preferable to inventing an overlap proof the artifacts cannot support.

See [M11.6 materialization manifest](../data/locked/m11-6/m11-6-materialization-manifest.json).

## Locked-data governance

Materialization itself did not evaluate the finalist and recorded the locked test as unopened. After M11.5 passed and the finalist was frozen, explicit authorization was consumed exactly once. The M11.6 opened record then established the atomic transition to OPENED; 125/125 incidents were evaluated; no rerun and no post-lock tuning followed.

The correct reproduction of this dataset's final-test role is to verify the materialization/opened/governance artifacts, **not** regenerate or reopen the final test as an ordinary benchmark.

## Scenario content and corruption mechanisms

Across the V5 program, governed synthetic variation includes hydraulic/topology family, incident source and severity, causal evidence depth, sensor coverage/health, missingness/dropout, noise, ambiguity/disagreement, and timing effects. WNTR/EPANET supplies modeled hydraulic/water-quality behavior.

Every mechanism remains a simulation assumption. Its inclusion does not establish that real utilities have the same distribution, failure frequencies, or network-model fidelity.

## Data quality and limitations

- All labels/outcomes are simulator-derived.
- Synthetic topology diversity is finite and does not prove utility-scale transfer.
- Sensor/dropout/noise mechanisms are governed stressors, not prevalence estimates.
- The final locked sensor-dropout slice shows a real weak point: 66.7% applicable coverage in 15 incidents.
- Novel locked topology was calibration-inapplicable; predictive metrics there are descriptive only.
- The M9.6 train/serve unobserved-age semantic deviation is frozen and disclosed.
- Counts labelled “rows” in evaluation manifests may reflect derived prefix/depth/seed evaluations and must not be re-described as unique incidents.

## Storage and provenance

The repository uses JSON/JSONL manifests, safetensors model artifacts, EPANET `.inp` networks, and checksummed generated evidence. Frozen final identities are hash-bound in the V5 release manifest and M11 finalist/evaluation records.

See [Data generation](DATA_GENERATION.md), [Evaluation](EVALUATION.md), and [Reproducibility](REPRODUCIBILITY.md).

## Historical datasets

`learning-v1`, `cycle-b2-joint-v4`, and earlier V3/V4 dataset cards/reports remain historical provenance. Their counts and split semantics should not be presented as the final V5 locked population.
