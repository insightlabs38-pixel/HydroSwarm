# Topology-generalization experiment plan (EXPERIMENTAL, NON-RELEASE)

Branch: `exp/topology-generalization`. Frozen `v0.2.1` (checkpoint SHA-256
`de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`), the
M11.6 locked evidence (`reports/evaluation/hydrocore-v5/m11/m11-6-final/`),
`data/locked/m11-6/`, and every other frozen/locked artifact are treated as
read-only baselines. Nothing under this plan opens, retrains against,
fine-tunes on, or overwrites locked data. No production default, factory, or
gate is changed. Written before any new training run on this branch, per
this repository's own protocol-before-results convention (e.g.
`docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md`).

## 1. What already exists (read before designing anything new)

This is a heavily pre-explored question. Reusing, not re-deriving:

- **The architecture is already a real GNN.** `HydroCore` (`src/hydroswarm/model/core.py`)
  has no one-hot node ID or persistent per-network embedding. Node order is
  an arbitrary per-example array index (alphabetical at import time,
  `src/hydroswarm/networks/importer.py`). `EdgeAwareGraphConv`/`DualChannelGraphConv`
  (`src/hydroswarm/model/layers.py`) do genuine sparse edge-indexed
  message-passing. Permutation equivariance is a first-class, tested
  capability (`src/hydroswarm/training/permutation.py`, `tests/unit/test_permutation.py`).
  A code comment states directly: "the model itself is topology-agnostic"
  (`scripts/hydrocore_v5/run_m7_topology.py:14-17`).
- **Multi-topology interleaved training is already shipped.** M9.0
  (`docs/evaluation/HYDROCORE_V5_M9_0_PROTOCOL.md`) found interleaved
  3-family training improved unseen-topology MATURE top-1 by +6.6pp
  (90% CI [+2.78,+10.76]pp) but broke known-family calibration coverage,
  and was rejected at that milestone. M9.0b tried four different Mondrian
  calibration-grouping schemes to fix the coverage break; all four failed
  (`INTERLEAVED_PREDICTOR_CALIBRATION_NOT_RESOLVED`). **M9.6** later found
  an exact-compute-parity recipe that passed every calibration/guardrail
  gate, and **this is the recipe actually shipped**: the frozen
  `models/hydrocore-v5-release/model.safetensors` is byte-identical
  (confirmed by SHA-256) to `ARM_B_M9_6-seed20260814`'s canonical export
  (`reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed20260814.json`),
  trained with `CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING`
  across golden-reference/branched-loop/loop-grid (200 scenarios/family).
- **Model-capacity scaling is already closed.** `docs/MODEL_CARD.md`: "a
  13.9M M model did not show the predeclared meaningful unseen-topology
  capacity gain." M9 closed architecture/training/capacity search with the
  4.18M S model retained.
- **The M11.6 locked result (frozen, read-only, cited not reproduced)**
  shows the real remaining gap is *not* primarily raw predictive quality:

  | split | n | top1 | top3 | MRR | calibrated_rate | actionable_rate | coverage |
  |---|---|---|---|---|---|---|---|
  | `locked_final_test` (known families) | 105 | 0.552 | 0.762 | 0.687 | 1.0 | 0.610 | 0.886 |
  | `locked_topology_test` (novel topologies) | 20 | 0.550 | 0.700 | 0.652 | **0.0** | **0.0** | 0.600 |

  (`reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-metrics.json`).
  Raw localization quality on genuinely novel topologies is close to the
  known-family number. What collapses to exactly zero is
  `calibrated_rate`/`actionable_rate`, because calibration validity and the
  deterministic `OODDetector` are gated on exact network-hash membership in
  a fixed `validated_topology_hashes`/`validated_network_hashes` allowlist
  (`src/hydroswarm/calibration/conformal.py:127-135`, "an unknown topology
  must invalidate calibration"; `src/hydroswarm/inference/ood.py:44-60,107-129`).
  This is a deliberate, tested, fail-closed safety design
  (`tests/scientific/test_ood_labels.py`, `docs/MODEL_CARD.md`'s stated
  non-intended-use: "treat an unseen topology as calibrated merely because
  a neural prediction exists"), not a bug, and this experiment does not
  attempt to loosen it. `m11-6-metrics.json` itself marks the topology
  split's predictive metrics `"topology_shift_predictive": "DESCRIPTIVE_NON_GATING"`.
- **A representation asymmetry exists and appears unexploited.**
  `src/hydroswarm/preprocessing/schema.py` already tags every node/edge
  feature `FeatureScope.ABSOLUTE` or `FeatureScope.TOPOLOGY_RELATIVE`
  (documentation only, not yet used to change how anything is normalized).
  The 19-dim `node_features`/13-dim `edge_features` tensors are normalized
  once, globally, by `NormalizationStats` fit only on the train split
  (`src/hydroswarm/preprocessing/schema.py:100-183`, applied in
  `src/hydroswarm/preprocessing/builder.py`). By contrast,
  `GraphStructuralEncoder` (`src/hydroswarm/model/encoders.py:52-89`),
  which encodes 3 graph-position scalars (`travel_time`,
  `reservoir_reachability`, `demand_centrality`), already rescales by a
  **per-example** (`dim=-2`, i.e. per-graph) max-abs statistic before its
  own linear projection — a topology-size-adaptive normalization the main
  node/edge feature path does not have. Nothing in the repository's prior
  topology work (M7, M9.0, M9.0b, M9.6, the capability diagnostic) targets
  this specific asymmetry.

## 2. Hypotheses

- **H1 (primary).** Node/edge features tagged `TOPOLOGY_RELATIVE` in
  `schema.py` (`elevation`, `base_demand`, `current_demand`,
  `arrival_time_residual`, `distance_to_reservoir`, `distance_to_sensor`
  for nodes; `length`, `estimated_travel_time`, `flow_magnitude`,
  `pipe_volume` for edges) carry a train-topology-scale-dependent global
  normalization. On a topology whose absolute scale (size, terrain,
  demand magnitude) differs from the training networks, these features
  land systematically off-distribution for `StaticFeatureEncoder`/the
  graph backbone, even under interleaved multi-family training.
  Augmenting these specific columns with an additional **per-graph
  relative-normalized** copy (same per-example max-abs convention
  `GraphStructuralEncoder` already uses) should improve raw predictive
  quality (top-1/top-3/MRR, event/evidence heads) on unseen topologies,
  without changing training-topology diversity, model capacity, or any
  calibration/OOD gate.
- **H2 (predicted negative result, stated in advance).** Because
  `actionable_rate`/`calibrated_rate` on a genuinely novel topology are
  categorically gated by exact network-hash membership (Section 1),
  H1 succeeding on raw predictive metrics is **not expected to move
  actionable_rate away from its governed floor** for topologies outside
  the validated-hash allowlist, and must not be reported as an
  actionability improvement per the task's own requirement 7. This
  experiment treats that as an expected, correctly-governed outcome, not
  a defect to fix.
- **Falsification.** H1 is rejected if the representation change does not
  improve unseen-topology top-1/top-3/MRR beyond noise (paired bootstrap
  CI including 0), or if it measurably regresses known-family calibration
  coverage/candidate-set-size guardrails (the same guardrails that killed
  M9.0's training-diversity approach).

## 3. Likely implementation points

- New file `src/hydroswarm/model/topology_normalization.py`: pure
  functions (no learnable parameters, so no checkpoint-shape risk) that
  read `NODE_FEATURE_SEMANTICS`/`EDGE_FEATURE_SEMANTICS` from
  `hydroswarm.preprocessing.schema` (reused, not duplicated) to select the
  `TOPOLOGY_RELATIVE` columns, and append a per-graph max-abs-normalized
  copy of those columns via masked reduction over the node/edge axis
  (`node_mask`/`edge_mask`-aware, so padding never contaminates the
  statistic).
- `HydroCore.__init__` (`src/hydroswarm/model/core.py`) gains one new,
  strictly additive constructor argument,
  `topology_relative_augmentation: bool = False`, following the exact
  precedent already set by `temporal_dynamics` (M9.1 preflight seam,
  `core.py:530-548`): default preserves today's exact behavior and
  checkpoint shape for every existing caller; the existing
  `node_feature_dim`/`edge_feature_dim` constructor arguments keep
  meaning "raw input width" (unchanged, so `architecture_config()`/
  checkpoint-identity semantics for the frozen release are untouched);
  when enabled, internal encoder/backbone input widths are computed as
  raw + augmented-column-count, and `forward()` calls the new
  augmentation function on `node_features`/`edge_features` before
  `node_encoder`/the backbone loop consume them.
- No change to `GraphStructuralEncoder`, `TemporalEncoder`/`QualityEncoder`,
  `EdgeAwareGraphConv`, `LatentHydraulicBlock`, any output head, the
  multitask objective, `SplitConformalCalibrator`, `OODDetector`, or any
  authority/certificate code.

## 4. Baseline

Two baselines, both read-only:

1. **Frozen v0.2.1 locked evidence** (Section 1 table) — cited, never
   reproduced or re-opened.
2. **Freshly reproduced M9.6-recipe control**, trained on this branch with
   the exact same corpus/config M9.6 used
   (`data/learning-v2/cycle-b2/`, `configs/model_small.yaml`, the
   `EXACT_..._INTERLEAVED_MULTI_TOPOLOGY_TRAINING` recipe), so the
   experimental arm can be compared against a same-session,
   same-compute-budget control rather than only a historical number. This
   is necessary because this session's compute budget will not exactly
   reproduce the full 3-seed/20-epoch/8h-per-run M9.6 campaign; both
   control and experimental arms use an identically reduced,
   explicitly-labeled budget so the comparison stays paired and honest
   (absolute numbers will not match the historical 20-epoch run; the
   paired delta is the thing being measured).

## 5. Evaluation design

- **Data**: `data/learning-v2/cycle-b2/tensors*` (train/validation/
  calibration/development_holdout/`ood-UNSEEN_TOPOLOGY`), already split
  network-disjoint with a documented zero-leakage check
  (`data/learning-v2/cycle-b2/dataset-report.json`'s `cross_split_leakage`).
  `ood-UNSEEN_TOPOLOGY` = `coastal-branch`, held out of training entirely
  (`dataset-report.json: "development_ood_topology": "coastal-branch"`).
  This is a *development*-tier unseen-topology probe, distinct from and
  never a substitute for the frozen `locked_topology_test` evidence.
- **Metrics** (both arms, every split where applicable): top-1, top-3,
  MRR, event/evidence-head losses+accuracy, calibration coverage/
  candidate-set size via the existing unmodified `SplitConformalCalibrator`
  fit per-arm on that arm's own calibration split only, actionable rate,
  abstention rate, deterministic-`OODDetector` behavior (unmodified),
  and the full existing hard-safety-counter suite reused from
  `scripts/hydrocore_v5/run_m11_6_locked_evaluation.py`'s counter
  functions where they apply outside the locked harness itself.
- **Per-topology / per-condition breakdown**: golden-reference,
  branched-loop, loop-grid (known/trained) vs. coastal-branch (unseen);
  curriculum-stage/condition breakdown reused from the existing
  `CurriculumStage`/condition taxonomy already in the corpus.
- **Statistics**: paired bootstrap CI on the arm delta (matching M9.0/M9.6's
  own methodology), reported alongside raw per-arm numbers.

## 6. Leakage risks and controls

- Train/validation/calibration/development_holdout/OOD splits are reused
  verbatim from the already-generated, already leak-checked
  `data/learning-v2/cycle-b2/` corpus — no new splitting logic is written,
  removing an entire class of leakage risk.
- `coastal-branch` is excluded from every training and calibration step
  for both arms; it is only ever read at evaluation time.
- Calibration is fit separately per arm, only on that arm's own
  calibration split, never on `development_holdout` or the OOD split,
  matching `configs/evaluation_policy_v3.json`'s split-role contract.
- No locked (`data/locked/m11-6/`) file is opened, read, or evaluated
  against by any script this experiment adds.
- All new artifacts are written under experiment-scoped paths
  (`experiments/topology-generalization/`,
  `reports/evaluation/topology-generalization/`) clearly distinct from
  any `m9-*`/`m10-*`/`m11-*` or `locked/` path, so nothing here can be
  mistaken for release evidence or accidentally overwrite it.

## 7. What this experiment will and will not claim

- Will claim, if supported: a measured, paired, statistically-qualified
  effect of the representation change on raw predictive metrics for
  known and unseen development-tier topologies, plus a check of whether
  known-family calibration guardrails still pass.
- Will not claim: any change to production defaults, gates, or the
  frozen v0.2.1 release; any actionability improvement not backed by a
  legitimately re-fit calibration procedure evaluated on its own terms;
  any generalization claim about the frozen `locked_topology_test`
  population (never opened by this work).
