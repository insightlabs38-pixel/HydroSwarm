# physics-informed-localizer-full-data-gate: pre-registration

Branch: `exp/physics-informed-localizer-full-data-gate`. Based on the
completed `exp/physics-informed-localizer-scale-validation` at
`8ccc59ccfccc5362cad432ee355a9266884d204c` (that branch's final report:
`reports/evaluation/physics-informed-localizer-scale-validation/FINAL_REPORT.md`,
classification `CANDIDATE_FOR_FULL_SCALE_VALIDATION`). **Experimental,
non-release.** No change to `models/hydrocore-v5-release`, `data/locked/`,
HydroSwarm v0.2.1, HydroCore-v5's frozen release artifacts, M11.6 locked
evidence, any hackathon claim, or any governance module
(`hydroswarm.inference.ood`, `hydroswarm.calibration.conformal`, any
actionability gate, simulator verification, or human-approval boundary).
This is a **data-scale validation gate**, not a hyperparameter search and
not yet a release/promotion training run: no GNN, no new physics features,
no capacity change, no attention-structure change, no LR/epoch tuning based
on results, no loss-weight change, no PCGrad, no calibration-threshold
change, no OOD/actionability relaxation. The completed scale-validation
branch's own reports, seed directories, manifests, results, and plan are
treated as immutable and are not modified.

## 0. What the completed studies found (recap, treated as fact)

- HydroCore-v5 source-localization failures are predominantly
  representation-limited, not purely information-limited.
- Generic topology-relative/structural feature injection failed; candidate
  conditioning without physics grounding is insufficient and can regress.
- Physics-informed candidate conditioning produces a reproducible
  unseen-topology improvement; `hop_magnitude_compatibility` (C2) is the
  dominant physics feature.
- Adding `nearest_sensor_log_concentration` (C1) to C2 preserves the C2
  Top-1 gain, significantly improves Top-3 over C2, preserves MRR, and
  causes no known-topology/calibration regression.
- On 3 fresh seeds (20260929/20261013/20261027), `C1_C2` vs `A_CONTROL`
  achieved unseen-topology Top-1 delta +5.48pp, 90% CI [+3.81, +7.14]pp,
  positive in all 3 seeds. `C2` alone has been positive across 6 seeds
  spanning two studies.
- **Every one of those runs trained on only 600 examples** (200/family,
  stratified across `golden-reference`/`branched-loop`/`loop-grid`), while
  the actual normalized Cycle-B2 `train` split contains 9000 examples
  (3000/family; verified directly against the corpus at this branch's base
  commit).

**This branch answers exactly one new question**: does `C1_C2`'s
unseen-topology advantage survive when training-data scale increases from
600 to 9000 examples, architecture and optimization budget (6 epochs,
identical optimizer/config) held fixed?

## 1. Primary hypothesis and endpoint (pre-registered, fixed before any
full-data training)

**H_primary**: the validated `C1_C2` candidate-conditioned localizer
(`nearest_sensor_log_concentration` + `hop_magnitude_compatibility`, `C3`
zeroed) retains a positive, meaningful `ood-UNSEEN_TOPOLOGY` Top-1
advantage over `A_CONTROL` when trained on the full 9000-example Cycle-B2
`train` split instead of the 600-example subsample used by every prior
validation run.

**Primary endpoint: `C1_C2` vs `A_CONTROL` on full-data-trained
`ood-UNSEEN_TOPOLOGY` Top-1, paired example-level analysis.** Secondary
endpoints: unseen-topology Top-3, MRR, true-source rank, true-source
probability, Top-1/Top-2 margin, entropy, known-topology
(`validation`/`development_holdout`) performance, calibration/OOD proxy
behavior, low-centrality and long-distance hard subgroups, ranking-shape
(Top-3 avoidance-of-regression / true-source-rank-when-Top-1-wrong /
failures-with-source-in-Top-3 / correct-Top-3-to-outside-Top-3
conversions), and the 600-vs-9000 same-seed comparison itself.

## 2. Pre-declared seed (fixed before any training; not changed after
observing results)

`SEED = 20261110` -- one new seed, disjoint from every seed used by the
completed `physics-informed-localizer-validation`
(20260814/20260901/20260915) and `physics-informed-localizer-scale-
validation` (20260929/20261013/20261027) studies.

## 3. Arms (exactly two)

| arm | `localizer_mode` | active physics columns | purpose |
|---|---|---|---|
| `A_CONTROL` | `default` | -- | existing frozen-equivalent control (unmodified `source_node_head`) |
| `C1_C2` | `candidate_conditioned` | `nearest_sensor_log_concentration` + `hop_magnitude_compatibility` | the selected physics-informed representation from the completed scale-validation branch |

`hop_arrival_time_compatibility` (C3) is zeroed/disabled. `C2`, `C_FULL`,
`C1` alone, `B_CANDIDATE_CONDITIONED`, and `A_CAPACITY_MATCHED` are **not**
re-run on this branch -- those questions are already answered by the
completed studies; re-running them would not serve this branch's narrower
data-scale question. Both arms reuse `physics_informed_localizer_
validation.run_experiment`'s `ARMS` registry, `build_model`,
`_mask_physics_columns`, `augment_batch`/`make_collate_fn`, `train_arm`
(pilot stage) / `evaluate_arm` (both stages) unmodified -- see
`scripts/hydrocore_v5_experimental/physics_informed_localizer_full_data_gate/
run_experiment.py`'s own module docstring for the exact reuse boundary.

**Verified at this branch's base commit** (`8ccc59ccfccc5362cad432ee355a
9266884d204c`), before any training:

- `A_CONTROL` parameter count: 4,044,113 -- identical to a plain
  `HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13,
  event_control_heads=True)` call with no experimental kwargs (same total,
  same per-parameter shapes), and identical to the completed scale-
  validation study's own `A_CONTROL` parameter count.
- `C1_C2` parameter count: 4,231,897 -- identical to the completed
  scale-validation study's own `C1_C2` parameter count (same
  `model_kwargs`, same `localizer_physics_feature_dim=3`, same
  `CandidateConditionedLocalizer`; ablation changes only which physics
  columns are nonzero, never the architecture).
- `C1_C2`'s active physics columns are exactly
  `{"nearest_sensor_log_concentration", "hop_magnitude_compatibility"}`;
  `hop_arrival_time_compatibility` (C3) is masked to exactly 0 for every
  example (verified by `tests/unit/
  test_physics_localizer_full_data_gate_ablation.py`, 18/18 passed).
- `_mask_physics_columns` does not mutate its input tensor.
- `A_CONTROL`'s `localizer_mode == "default"` and `model_kwargs == {}`; no
  default HydroCore code path is altered by this branch.
- Physics features (`nearest_sensor_log_concentration`,
  `hop_magnitude_compatibility`) are computed by
  `candidate_sensor_features.py`/`physics_features.py` from
  `temporal_features`/`hop_distance`/`active_sensor_mask`/`node_mask`/
  `timestamps` only -- the same unmodified functions the completed
  validation/scale-validation studies used; no label or target tensor is
  read anywhere in `augment_batch`/`compute_physics_features`.
- No file under `hydroswarm/inference/ood.py`,
  `hydroswarm/calibration/conformal.py`, any actionability/governance
  module, or `models/hydrocore-v5-release/**` is modified by this branch's
  diff.

## 4. Git / dataset provenance (pre-registered)

- Repository commit (base of this branch): `8ccc59ccfccc5362cad432ee355a
  9266884d204c` (`exp/physics-informed-localizer-scale-validation`'s final
  report commit).
- Corpus: `data/learning-v2/cycle-b2/tensors-normalized/` (unchanged from
  every prior study in this family). Verified split sizes at this commit:
  `train`=9000 (3000/family across `golden-reference`/`branched-loop`/
  `loop-grid`; 2100/family, 6300 total, carry a real source label -- the
  remaining 900/family lack one and are used for the model's other
  multitask heads only, never for source-localization metrics),
  `validation`=1000 (334/333/333), `calibration`=1000 (334/333/333),
  `development_holdout`=1750 (584/583/583), `ood-UNSEEN_TOPOLOGY`=400
  (all `coastal-branch`, the unseen family).
- Dataset manifest hashes (`index_sha256` from each split's own
  `manifest.json`, recorded before any training on this branch):

  | split | `index_sha256` |
  |---|---|
  | `train` | `a4e05a19cab4c999d6c2f98f5c2b4bd26e1349ddd85ed80ef69f9292fd63ada9` |
  | `validation` | `2f629c9c1020e2fa125be7c87a33e79f5dba655899b4dd06240c8befa8556b16` |
  | `calibration` | `a2c5a5ad1691a627ae9a4aca2bf7bb0c5deb3921396a6a657b3d29154c561a75` |
  | `development_holdout` | `1c21df5f0e39ea4659ab31fda61e60c157c7703c64cdde8c8a114efba42c4bcd` |
  | `ood-UNSEEN_TOPOLOGY` | `cb0228e96353d3352179eb6a2d8dbafaa24a7eea917f80eee63031046ede788f` |

- Git LFS: an `lfs pull` was required (none of the five splits above were
  materialized in this fresh clone). Fetched with exactly:
  `git lfs pull --include="data/learning-v2/cycle-b2/tensors-normalized/train/**,data/learning-v2/cycle-b2/tensors-normalized/validation/**,data/learning-v2/cycle-b2/tensors-normalized/calibration/**,data/learning-v2/cycle-b2/tensors-normalized/development_holdout/**,data/learning-v2/cycle-b2/tensors-normalized/ood-UNSEEN_TOPOLOGY/**"`.
  Verified afterward: all 5 splits' `.safetensors` shards materialized
  (0 remaining LFS pointers); `cycle-b2-ood-extension`,
  `cycle-b2-control-v2`, `cycle-b2-trajectories-v3`,
  `models/cycle-b2-candidates`, and `models/cycle-b2-controls` sampled and
  confirmed still-pointer (untouched). No LFS cache pruning performed.

## 5. Stages (exactly two, same seed)

### Stage 1 -- same-seed pilot anchor (600 examples)

Byte-for-byte reuse of the completed studies' own pilot protocol: 600 train
examples (200/family, stratified, filtered to examples with a real source
label), validation/development_holdout capped at 300 (same capped-index
logic), full calibration (1000) and full `ood-UNSEEN_TOPOLOGY` (400), 6
epochs, same optimizer/batch/LR/weight-decay/clip/warmup/scheduler/task
weights (read from `configs/training-v5-causal.yaml`), CPU, fp32,
deterministic. Purpose: establish the `C1_C2` vs `A_CONTROL` effect at
seed 20261110 BEFORE scaling training data -- a same-seed anchor, not a new
discovery experiment.

### Stage 2 -- full-data training (9000 examples)

The entire, unsubsampled 9000-example `train` split (no stratified cap, no
real-source filtering of the training set itself -- "the entire normalized
Cycle-B2 train split" is taken literally). Same architecture, same
optimizer/batch/LR/weight-decay/clip/warmup/scheduler/task weights, same 6
epochs (**not** the 20-epoch promotion config), CPU, fp32, deterministic.
Full (uncapped) evaluation populations: `validation`=1000,
`development_holdout`=1750, `calibration`=1000, `ood-UNSEEN_TOPOLOGY`=400.
The only two changes vs. Stage 1: training-data scale and evaluation
population size -- isolating exactly the `600 -> 9000` factor.
`checkpoint_every_epochs` is set to 1 (vs. Stage 1's end-of-run-only) for
crash recovery on the much longer run; this is an I/O-cadence change only,
never a change to the trained result, optimizer, LR, batch size, or epoch
budget.

## 6. Full-data success gate (pre-registered, fixed before training; not
relaxed after observing results)

Classify the full-data result as `PASS_FULL_DATA_GATE` only if **all** of
the following hold, evaluated on `C1_C2` vs `A_CONTROL`,
`ood-UNSEEN_TOPOLOGY` Top-1, full-data-trained models, paired
example-level analysis:

1. Top-1 delta vs `A_CONTROL` is positive.
2. Point estimate is at least +2.0 percentage points.
3. Paired 90% bootstrap CI excludes zero in the positive direction.
4. Top-3 does not show a statistically significant negative regression
   (bootstrap CI upper bound < 0).
5. MRR does not show a statistically significant negative regression (same
   test).
6. Neither `validation` nor `development_holdout` Top-1 shows a material,
   statistically supported regression (paired bootstrap CI upper bound
   < 0).
7. Calibration/OOD proxy behavior is not materially worse (calibration
   coverage drop > 5 points, ECE increase > 0.05, or any population's
   `proxy_actionable_rate` drop > 10 points vs `A_CONTROL`).
8. No governance code is altered by this branch's diff.

`INCONCLUSIVE_FULL_DATA_GATE` if the point estimate is positive but
< +2.0pp, or the CI crosses zero, or results are mixed without a clear
harmful regression. `FAIL_FULL_DATA_GATE` if the full-data Top-1 delta is
<= 0, or there is a clear harmful Top-3/MRR/known-topology regression.
`PASS_FULL_DATA_GATE` additionally earns the label
`CANDIDATE_FOR_PROMOTION_QUALITY_TRAINING` -- meaning only that the next
experiment may legitimately use the repository's full 20-epoch
promotion-quality regime, never that this branch itself promotes or
releases anything.

## 7. Critical scale analysis (pre-registered)

Beyond `C1_C2 vs A_CONTROL`, this branch's central analysis is
`600-example effect vs 9000-example effect` at the exact same seed
(20261110): pilot `A_CONTROL`/`C1_C2`/delta reported alongside full-data
`A_CONTROL`/`C1_C2`/delta, with a descriptive (not a new confirmatory
statistical test) classification of whether scale strengthens, preserves,
attenuates, or eliminates the effect.

## 8. Statistical convention (unchanged, "HydroSwarm's established
convention")

Paired bootstrap: 2000 resamples, deterministic bootstrap seed `20260826`
(same as every prior study in this family), 90% percentile interval.
Matched by `scenario_id` within a stage/seed. Identical evaluation examples
used between arms at each stage.

## 9. Hard-subgroup and ranking-shape validation (required, not optional)

Repeated on full-data-trained models: low/medium/high betweenness-
centrality terciles, short-vs-long source-to-nearest-sensor distance
(median split), each with exact `n` reported. Ranking-shape validation
(because C1's inclusion rationale is Top-3/ranking quality, not Top-1
alone): does `C1_C2` at full-data scale avoid a Top-3 regression, improve
true-source rank on examples where its own Top-1 remains incorrect,
increase the fraction of Top-1 failures with the true source still in
Top-3, and avoid converting `A_CONTROL`'s correct-Top-3 cases into
outside-Top-3 cases for `C1_C2`? No new post-hoc subgroup is introduced.

## 10. Cross-study context

Prior evidence is reported descriptively, never pooled into one misleading
combined significance test across different training scales: this
branch's own 600-example pilot anchor and 9000-example full-data result are
each clearly labeled by scale, and compared only against each other (same
seed) and descriptively against the prior 3-seed/6-seed 600-example
evidence already committed on other branches.

## 11. Safety / governance (unchanged)

Identical rationale to every prior study in this family: this pilot-scale
localization-only harness never exercises the sampling/planning/execution
control loop that produces non-zero `hard_safety_counters` at the M11.6
evaluation tier, so every arm reports all eight counters as 0 because those
code paths are never invoked here, not because they were independently
re-verified at that tier. No governance module is modified by this branch.

## 12. Runtime priority order (task-specified)

1. branch/pre-registration; 2. focused tests; 3. 600-example `A_CONTROL`;
4. 600-example `C1_C2`; 5. 9000-example `A_CONTROL`; 6. 9000-example
`C1_C2`; 7. paired/statistical analysis; 8. report; 9. broader unit tests
if remaining runtime permits. No additional arms are added after these are
complete. A negative full-data result is preserved as a negative result --
no rescue attempt, no post-hoc architecture change.

This branch is **not merged**. All work is committed and pushed only to
`exp/physics-informed-localizer-full-data-gate`.
