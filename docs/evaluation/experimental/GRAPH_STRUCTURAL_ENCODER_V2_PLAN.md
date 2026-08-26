# Graph-structural encoder v2: experiment plan (EXPERIMENTAL, NON-RELEASE)

Branch: `exp/graph-structural-encoder-v2`. Follows `exp/failure-mode-diagnostics`
(merged conclusions only, not merged code — see "Provenance" below). Does not
modify `models/hydrocore-v5-release`, any `data/locked/` artifact, any
`m9-*`/`m10-*`/`m11-*` report, or `main`. No gate is loosened. Every new
output is experimental/non-release.

## 0. Provenance

Reused directly from `exp/failure-mode-diagnostics` (read-only reference,
not merged):

* the diagnostic report's Section 2/5/8/9 findings (this plan's motivation);
* `graph_features.py`'s centrality/hop-distance definitions (betweenness,
  closeness, hop-to-dead-end, hop-to-reservoir) — reused as the *reference
  definitions*, reimplemented per-candidate-node (not just the true source)
  in a new module, since that branch's version only ever computed features
  for the labeled source node (diagnostic use, not a model input);
* `run_pilot.py`'s train/eval harness structure (stratified family sampling,
  `OODDetector`/`SplitConformalCalibrator` reuse, proxy actionable/abstention
  metrics) — reused as the harness skeleton for this experiment's own
  `run_experiment.py`, not its `topology_normalization.py` change;
* `rerun_topology_pilot_with_logging.py`'s per-row logging convention — reused
  for this experiment's paired per-example tables.

Not reused: `topology_normalization.py` itself (the failed per-graph
max-abs rescale of already-existing scalar columns) — its own report
explicitly recommends against reusing that specific mechanism, only its
harness.

## 1. Current structural-information bottleneck (exact)

`HydroCore` (`src/hydroswarm/model/core.py`) fuses four per-node modality
embeddings before its edge-aware backbone runs (`core.py:950-982`):
`node_encoder(node_features)`, `graph_encoder(travel_time,
reservoir_reachability, demand_centrality)`, `temporal_encoder(...)`,
`quality_encoder(...)`. This experiment targets the second one,
`GraphStructuralEncoder` (`src/hydroswarm/model/encoders.py:52-89`):

1. **It never receives `edge_index`.** It is a pure per-node 4-scalar MLP
   (`travel_time`, `log1p(travel_time)`, `reservoir_reachability`,
   `demand_centrality`) with a per-graph max-abs rescale. It cannot express
   "how central is this node in the graph," only "how far (by shortest
   hydraulic path) is this node from a reservoir."
2. **`demand_centrality` is not graph centrality.** Confirmed by reading its
   source (`src/hydroswarm/simulation/wrapper.py:833-841`): it is
   `node_demand / total_network_demand` — a demand-share statistic, not a
   betweenness/closeness/degree measure. The name is misleading; there is
   currently **no real graph-topological centrality feature anywhere in the
   model.**
3. **`distance_to_sensor` (node_features column 17,
   `src/hydroswarm/preprocessing/schema.py:35`) already exists** as a raw,
   globally-normalized scalar inside the generic 19-wide `node_features`
   vector consumed by `StaticFeatureEncoder`, computed in
   `HydraulicFeatureBuilder.build` (`builder.py:175,213`) via
   `nx.multi_source_dijkstra_path_length` from the currently-reporting
   sensor set. It is one column among 19, globally normalized (not
   per-graph), and not treated specially anywhere — i.e. the model already
   has *some* observability signal, just diluted and not fed to the one
   encoder whose job is graph position.
4. **The backbone (`LatentHydraulicBlock`/`EdgeAwareGraphConv`,
   `layers.py`) already does real 1-hop mean edge-aware message passing**
   over `edge_index`, confirming the diagnostics report's conclusion that
   HydroCore is not a naive non-graph model — but this happens *after*
   modality fusion, mixed with every other task's signal across
   `num_layers` blocks, with no architectural guarantee that
   periphery/centrality information specifically survives to
   `source_node_head`.

**Bottleneck, precisely stated:** `GraphStructuralEncoder` is the one part
of the model whose stated job is "structural position," but it sees only a
hydraulic-distance-to-reservoir scalar and a demand-share scalar, never
real centrality, never explicit sensor-observability, and never
`edge_index` itself. This is the exact seam the diagnostics report
(Section 8/9, hypothesis 1) flagged as the narrowest, most-targeted next
step, ahead of a full GNN rewrite.

## 2. Proposed feature/encoder changes

All new feature computation lives in two new, pure, deterministic modules
under `scripts/hydrocore_v5_experimental/graph_structural_encoder_v2/`,
mirroring `topology_normalization.py`/`graph_features.py`'s convention of
computing additive features at the batch level (after
`collate_variable_topology`), so **no frozen preprocessing/corpus-generation
code is touched** and no existing checkpoint's input contract changes
unless a new model explicitly opts in.

* **`structural_features.py`** (Arm B — CENTRALITY): per-candidate-node,
  purely topological, computed from `edge_index`/`node_mask`/
  `source_candidate_mask` only (no sensor data, no labels):
  - normalized degree (`degree / max(node_count - 1, 1)`);
  - betweenness centrality (networkx, exact — networks are 6-9 nodes);
  - closeness centrality (networkx, exact);
  - hop-distance to nearest non-candidate node (reservoir/tank, identified
    as `node_mask & ~source_candidate_mask`, never by decoding a
    normalized `node_type` column), normalized by that graph's own
    diameter;
  - hop-distance to nearest degree-1 ("dead end") node, normalized by
    diameter (0 if the candidate itself is a dead end);
  - normalized graph position: closeness rescaled to `[0,1]` by that
    graph's own min/max (same convention as
    `exp/failure-mode-diagnostics`'s `graph_features.py`, for direct
    comparability).
* **`observability_features.py`** (Arm C — OBSERVABILITY/DISTANCE):
  per-candidate-node, relationship to the *currently reporting* sensor set
  (`sensor_mask.any(dim=time)` — the same operational definition of
  "sensor" already used to derive `distance_to_sensor` at corpus-build
  time, so this arm does not invent a new, inconsistent notion of
  "sensor"):
  - hop-distance to nearest active sensor (unweighted graph hops, distinct
    from the existing length-weighted `distance_to_sensor` column —
    reported as a genuinely new derived signal, not a copy);
  - mean / min / max hop-distance to *all* active sensors;
  - fraction of active sensors within 1-hop / 2-hop / 3+-hop radius bins;
  - local sensor coverage density: `(#active sensors within radius r) /
    (#nodes within radius r)`, `r` = graph radius // 2 (adapts to small
    graphs).
  Both modules are batched (loop over the batch's small graphs, cheap:
  6-9 nodes), permutation-equivariant by construction (recomputed from
  `edge_index`, not looked up by a fixed node identity), and covered by a
  dedicated relabeling-invariance unit test (Section 5).
* **`GraphStructuralEncoder` extension (Arm D — STRUCTURAL AGGREGATION,
  `src/hydroswarm/model/encoders.py`):** two new, fully backward-compatible
  constructor parameters, `structural_feature_dim: int = 0` and
  `use_edge_aggregation: bool = False`. When both are at their default, the
  module is **byte-for-byte identical** to today's `GraphStructuralEncoder`
  (same input width, same layers, same forward computation) — no existing
  checkpoint or caller is affected. When `structural_feature_dim > 0`, the
  encoder's input linear layer widens to accept the extra per-graph-
  normalized structural/observability columns (Arms B/C plug in here, or
  both — Arm E). When `use_edge_aggregation=True`, one lightweight
  mean-neighbor aggregation pass (a single `EdgeAwareGraphConv`-style
  linear+mean-pool, no learned edge-feature projection since this encoder
  has no edge features of its own — deliberately smaller than the
  backbone's own conv) is applied to the per-node hidden representation
  before the final projection to `d_model`, giving this specific encoder
  direct, controlled access to `edge_index` for the first time. `HydroCore`
  gains matching optional constructor passthroughs
  (`graph_structural_feature_dim`, `graph_structural_edge_aggregation`),
  and its `forward()` call site (`core.py:951-955`) passes the new optional
  tensors via `batch.get(...)` — `None` when absent, so every existing
  caller is unaffected. This is the smallest change that gives
  `GraphStructuralEncoder` real `edge_index` access without replacing or
  duplicating the backbone's own message passing (no full GNN rewrite).

## 3. Ablation arms

| Arm | Name | `GraphStructuralEncoder` input | `edge_index` in this encoder | Purpose |
|---|---|---|---|---|
| A | CONTROL | today's 4 scalars only | no | frozen-equivalent baseline (same harness as `run_pilot.py`'s own CONTROL, not the real frozen release) |
| B | CENTRALITY | + degree/betweenness/closeness/hop-to-periphery/normalized-position | no | tests centrality hypothesis in isolation |
| C | OBSERVABILITY | + sensor hop-distance/coverage features | no | tests competing observability hypothesis in isolation |
| D | STRUCTURAL_AGG | today's 4 scalars only | **yes** (1-hop mean aggregation) | tests whether `edge_index` awareness alone (no new static features) helps |
| E | COMBINED | B + C features, with `edge_index` aggregation | yes | best-justified combination; the arm capacity control (Section 8) applies here specifically |

Compute-conscious fallback (Section 8): if wall-clock forces a cut, priority
is A, B, C, E — D is the cheapest to skip since Section 2's diagnostic
report already found the backbone has message passing and the marginal
question ("does *this encoder specifically* need it too") is the least
novel of the four hypotheses. All five are attempted first; this is the
documented fallback order only if compute runs out mid-experiment.

## 4. Train/validation/test split

Identical to `run_pilot.py` (reused verbatim, not re-derived): train on
`golden-reference`/`branched-loop`/`loop-grid` (200 real-source examples
per family, seed `20260814`, `has_real_source` filter), evaluate on
`validation` (300), `calibration` (full, for conformal fitting),
`development_holdout` (300), and `ood-UNSEEN_TOPOLOGY` (`coastal-branch`,
full — 400 per `dataset-report.json`, but `run_pilot.py`'s own diagnostics
report cites 280 real-source examples after the same filter). Every arm
trains on the **exact same physical examples** (same indices, same
`stratified_indices`/`capped_indices` seeding) — architecture/features are
the only thing that varies between arms, matching this task's "paired
evaluation on identical examples" requirement by construction, not by
post-hoc matching.

## 5. Topology leakage risks and controls

* **Structural features must not memorize topology identity.** Degree/
  centrality/hop-distance values are small integers/ratios that could in
  principle let a model key off "this exact centrality value pattern ==
  golden-reference network" rather than learning a general
  centrality-outcome relationship. Mitigated by: (a) per-graph
  normalization (ratios/diameter-relative, not raw counts) so the same
  *relative* position looks similar across networks of different size;
  (b) a dedicated **relabeling/permutation test** (Section 6) verifying
  features are invariant to node renaming and recomputed purely from
  `edge_index`, never from a stored per-topology lookup table; (c) the
  unseen-topology (`coastal-branch`) evaluation itself is the direct
  empirical check — if an arm's unseen-topology metrics do not improve
  (or regress) versus CONTROL, that is read as evidence against/for a
  memorization-driven artifact, not assumed away.
* **No label leakage.** Every feature function's signature takes only
  `edge_index`/`node_mask`/`source_candidate_mask`/`sensor_mask` — never
  `source_node`/`source_node_mask`/any target tensor. Enforced by a unit
  test asserting the feature functions' signatures and by code review of
  the two new modules (they import nothing from `hydroswarm.training.targets_v2`
  or `hydroswarm.training.losses`).
  Ambiguity risk: this experiment's `sensor_mask.any(dim=time)` includes
  whichever `SENSOR_DROPOUT`/`SENSOR_HEALTH_DEGRADED` condition already
  happened in that scenario window — legitimate (it is genuinely known at
  inference time which sensors reported), not label leakage, but flagged
  here explicitly since it makes Arm C's features condition-dependent in a
  way Arm B's are not; the required subgroup analysis (known vs. unseen,
  condition-matched NOMINAL) will surface if this drives an artifact.
* **Normalization stays per-graph, not per-corpus.** Reusing a global
  training-corpus statistic to normalize a topology-relative structural
  quantity is exactly the mechanism the prior pilot's own report flagged as
  ineffective — this experiment normalizes every new feature within its own
  graph (diameter-relative hops, closeness min/max, degree by node count),
  never against a cross-topology fitted `NormalizationStats`.
* **Calibration/OOD fitting stays split-correct.** Reuses `run_pilot.py`'s
  own convention exactly: `SplitConformalCalibrator.fit` only ever sees the
  `calibration` split; `OODDetector`'s `validated_network_hashes` are built
  only from this run's own `train` topology hashes, per arm.

## 6. Primary and secondary success criteria

**Primary question (not aggregate Top-1):** does any arm materially improve
localization for peripheral/low-centrality and unseen-topology cases
*without* degrading known-topology/easy-case performance, calibration, or
governance?

Primary success (per arm, vs. CONTROL, on paired identical examples):

* Top-1 (or MRR — Section 5's error taxonomy shows 55% of failures are
  representation gaps, so MRR/true-source-rank is treated as at least as
  informative as Top-1) improves on the **low-centrality tercile** and/or
  the **condition-matched unseen-vs-known NOMINAL gap narrows**, with a
  bootstrap CI (Section 9 convention) excluding zero;
* **no regression** (CI excludes a negative direction beyond noise) on the
  high-centrality tercile, known-topology NOMINAL, calibration coverage/ECE,
  or any proxy actionable/abstention/OOD-caution rate.

Secondary/diagnostic success: Arm D/E's `edge_index` aggregation
demonstrably changes predictions beyond what B/C's static features alone
produce (paired B vs. E, C vs. E transition tables); the
centrality-vs-observability question (Section 8 of this plan) resolves in
one direction with the post-hoc stratified analysis.

Explicitly a **misleading, not a passing** result: aggregate improvement
driven only by the high-centrality tercile; any calibration/coverage
regression; any actionable-rate increase alongside a Top-1 regression
(relaxed-gate artifact); a gain attributable to Arm E's larger parameter
count rather than its features (Section 8 capacity control).

## 7. Compute-conscious training plan

Mirrors `run_pilot.py` exactly: `HydroCore.from_variant("small",
event_control_heads=True)` (not the real release variant), CPU, `fp32=True`,
`deterministic=True`, single seed (`20260814`, matching the shipped
finalist), 6 epochs (`PILOT_EPOCHS`), 200 real-source examples/family
(600 train total), batch/optimizer hyperparameters taken from
`configs/training-v5-causal.yaml`. Five arms (A-E) at this scale
(`rerun_topology_pilot_with_logging.py`'s own comment: "~20 min/arm on
CPU") is a bounded, pre-committed compute budget (~2 hours total), not an
open-ended sweep. No hyperparameter tuning per arm — every arm uses
identical optimizer/schedule settings, isolating the feature/architecture
change as the only manipulated variable.

## 8. Parameter/capacity control

Every arm's `HydroCore.parameter_report()` is recorded
(`graph_encoder`-attributable delta specifically, via
`count(self.graph_encoder)` — already itemized separately by the existing
`parameter_report()` method, `core.py:1306-1369`, unmodified). Arms B/C add
a few input columns to a small MLP (a few hundred to low-thousands of
parameters); Arm D/E's aggregation adds one small linear layer. If any
arm's total parameter delta versus CONTROL exceeds roughly 1% of total
model size, a parameter-matched non-structural control (extra hidden width
on the unchanged 4-scalar input, no new features) will be added before
attributing any gain to structure rather than capacity — assessed after
Section 2's modules are implemented and actual widths are known, not
pre-guessed here.

## 9. Statistics and reporting conventions

Paired per-example evaluation (identical indices across arms, Section 4),
bootstrap CIs computed with the same resampling convention already
established in this repository (`analyze_paired_pilot.py`'s 2000-resample
percentile bootstrap, matching `bootstrap_followup.py`'s own established
HydroSwarm convention), 2x2 transition tables (arm-correct/arm-wrong x
CONTROL-correct/CONTROL-wrong) per arm, true-source-rank deltas, Top-3
membership transitions. Exploratory (single-seed, pilot-scale, one unseen
topology) is labeled as such throughout, distinct from confirmatory M11.6
locked evidence, exactly as the diagnostics report itself insists on.

## 10. Safety/governance scope note

This experiment changes only `GraphStructuralEncoder`'s input features and
internal aggregation. It does not touch `hydroswarm.inference.ood`,
`hydroswarm.calibration.conformal`, any actionability gate, or any
human-approval path — those modules are called unmodified, exactly as
`run_pilot.py` already calls them. Unseen-topology proxy-actionability
therefore remains structurally fail-closed by construction (the same
`OODDetector.topology_level` check `run_pilot.py` already uses), not
re-verified from scratch. This pilot-scale harness does not exercise the
full sampling/planning/execution control loop that produces M11.6's 8 hard
safety counters (`human_approval_bypassed` etc.) — same limitation
`run_pilot.py` already had; the report states this explicitly rather than
fabricating counters this evaluation tier cannot produce, and confirms by
code-diff review that no governance module is edited on this branch.
