# HydroCore-v5 failure-mode diagnostics: plan (EXPERIMENTAL, NON-RELEASE)

Branch: `exp/failure-mode-diagnostics`. This is a diagnostic investigation,
not a model-improvement task: no architecture change, no retraining of
`models/hydrocore-v5-release`, no gate loosening, no hyperparameter tuning.
Frozen `v0.2.1`, `models/hydrocore-v5-release/`, the M11.6 locked evidence
(`reports/evaluation/hydrocore-v5/m11/m11-6-final/`, `data/locked/m11-6/`),
and every published hackathon claim are read-only baselines: this plan
reads them for analysis and cites their numbers, and never opens, mutates,
retrains against, or overwrites any of them. Written before any new
analysis artifact in this document's Phase 2+ was produced, following this
repository's own protocol-before-results convention.

## 0. What already exists (read before designing anything new)

- **`exp/topology-generalization`** (unmerged sibling branch, not touched
  by this work) ran a controlled pilot: `HydroCore` "small" (4.18M params),
  CONTROL vs. `EXPERIMENTAL_TOPOLOGY_RELATIVE` (per-graph relative-normalized
  copy of every `FeatureScope.TOPOLOGY_RELATIVE` column), trained fresh on
  `data/learning-v2/cycle-b2` (golden-reference/branched-loop/loop-grid,
  600 examples, 6 epochs, seed `20260814`), evaluated on the `coastal-branch`
  unseen-topology OOD split. Result (`docs/evaluation/experimental/
  TOPOLOGY_GENERALIZATION_PILOT_RESULTS.md`, cherry-picked onto this branch
  read-only): **top-1 bit-for-bit identical (0.3750) on the same 280
  real-source examples; top-3 regressed -0.025 (90% CI [-0.046,-0.004],
  excludes zero); MRR CI includes zero; calibration precision (ECE, mean
  set size) slightly worse; coverage identical; actionable/OOD behavior
  categorically unaffected (hash-gated, both arms 0% actionable on the
  unseen topology).** This is the "previous pilot" this branch is asked to
  explain, and is treated as a confirmed negative result, not re-litigated.
  Its per-example predictions were never persisted to disk (only aggregate
  summaries were), so Phase 4 below re-runs its exact protocol once, purely
  to obtain per-example rows for paired analysis (see Phase 4).
- **The M11.6 locked evaluation** (`reports/evaluation/hydrocore-v5/m11/
  m11-6-final/`) is the frozen, real v0.2.1 evidence: 125 incidents, split
  `locked_final_test` (105, 3 known families: golden-reference,
  branched-loop, loop-grid) and `locked_topology_test` (20, 4 procedurally
  generated novel topologies, 5 incidents each, all under `NOMINAL`
  condition only — **stress conditions were never tested on novel
  topologies**, a real gap in the frozen evidence itself, not something
  this branch can fill without reopening locked generation). Headline
  (cited, not reproduced): known top1/top3/MRR = 0.552/0.762/0.687 vs.
  novel-topology 0.550/0.700/0.652 — raw predictive quality is close.
  `calibrated_rate`/`actionable_rate` collapse to exactly 0.0 on
  `locked_topology_test` because `OODDetector.topology_level` and
  `SplitConformalCalibrator.validate_runtime` are deterministically
  hash-gated (`src/hydroswarm/inference/ood.py:44-67`,
  `src/hydroswarm/calibration/conformal.py:127-135`) — a deliberate,
  tested, fail-closed design, not a bug, and `m11-6-metrics.json` itself
  marks the split's predictive metrics `DESCRIPTIVE_NON_GATING`.
- **Architecture**: `HydroCore` (`src/hydroswarm/model/core.py`) has no
  persistent per-network node embedding; `EdgeAwareGraphConv`/
  `DualChannelGraphConv` do real sparse edge-indexed message passing;
  permutation equivariance is tested (`tests/unit/test_permutation.py`).
  `GraphStructuralEncoder` (`src/hydroswarm/model/encoders.py:52-89`)
  encodes only 3 scalar graph-position features (`travel_time`,
  `reservoir_reachability`, `demand_centrality`) via an MLP over
  per-example max-abs-rescaled values — it never consumes `edge_index`
  itself, despite the name.
- **Related prior diagnostic** (merged PR #12, `diag/capability-bottleneck`
  lineage): found the *live-serving* gap (not the frozen-evaluation gap
  this branch studies) was explained by evidence sparsity and a
  network-identity/`.inp`-round-trip hashing defect that defeats
  calibration/OOD gates. Not re-litigated here; cited as a precedent for
  what a "network-identity/canonicalization" failure category looks like,
  and explicitly checked for relevance to M11.6 (see Phase 5 — M11.6 uses
  one fixed, non-round-tripped `.inp` per known family, so this specific
  defect does not obviously apply to the locked-evaluation population,
  though this is not exhaustively re-audited here).
- **Incidental, out-of-scope defect** (documented by the prior pilot, not
  fixed here per this branch's own instructions): `hydroswarm.training.
  losses._cross_entropy`'s all-invalid-batch fallback can emit `NaN` when
  every candidate logit in a microbatch is the model's masked-out sentinel.
  Reproduces on stock code, unrelated to any change on this branch. Noted
  again below only where it could affect an analysis in this branch.

## 1. Available covariates

**A. From the frozen M11.6 locked evidence** (`m11-6-raw-incidents.jsonl` +
`locked_final_test|locked_topology_test/scenarios.jsonl`, joined via
`m11-6-materialization-manifest.json`'s `(split, scenario_index) ->
scenario_id` map, cross-checked on `source_node`/`topology_id` agreement):
`split` (known/novel), `network_family`, `topology_id`, `condition_kind`
(7-way categorical stress taxonomy already used by the harness: NOMINAL,
LOW_COVERAGE_ACTIVE_SAMPLING, SENSOR_DROPOUT, SENSOR_HEALTH_DEGRADED,
MEASUREMENT_NOISE, SEVERITY_SHIFT, AMBIGUITY_DISAGREEMENT), `source_node`,
`top1_correct`/`top3_correct`/`reciprocal_rank`, `posterior_entropy`,
`candidate_set_size`, `calibrated`, `conformal_truth_coverage`, `outcome`,
`control_action`, `final_status`, planning/sampling/approval counters,
`no_safe_plan`, per-incident hard-safety-counter block, plus scenario-level
condition metadata (`missing_rate`, `health_fraction`, `health_mode`,
`coverage`, `ambiguity`, `hydraulic` — populated only for the condition
kinds they apply to) and `generator_config` (`sensor_count`=4 constant
across all 125 incidents, `sensor_noise_std`, demand/strength/duration/
start-time *bin ranges*).

**B. Derived, via `networkx` over the frozen, hash-verified `.inp` topology
files** (`graph_features.py`; node-count-checked against
`dataset-report.json`/the M11.6a novelty spec before trusting a file):
`node_count`, `edge_count`, `graph_density`, `graph_diameter`,
`dead_end_count`, `reservoir_count`, and per-source-node `degree`,
`betweenness_centrality`, `closeness_centrality` (+ min-max-normalized
graph position), `hops_to_reservoir`, `hops_to_nearest_dead_end`,
`is_boundary_node` (degree 1), `eccentricity`.

**C. From the topology-relative pilot re-run** (Phase 4 only, `data/
learning-v2/cycle-b2`, a *development*-tier corpus, never a substitute for
locked evidence): full per-example softmax probability vectors (so exact
`true_source_rank`, `top1_probability`, `margin_top1_top2`,
`posterior_entropy_bits`, `n_candidates` are computable, not just top1/
top3/MRR), `topology_hash`/`network_hash`, `curriculum stage`,
`event_presence` correctness, plus the same graph-structural block as (B)
computed from each example's own `TopologyMetadata.node_ids/edge_ids` (no
`.inp` file needed there — already attached per example).

## 2. Missing covariates that would be worth deriving but are NOT available

Listed explicitly per this task's own instruction not to invent them:

- **Exact realized incident strength/duration/start-time per M11.6
  incident.** `scenarios.jsonl` records only `generator_config`'s *bin
  ranges* (e.g. `strength_bins: [0.5, 1.0, 2.0]`), not which bin (or exact
  value) a given incident actually drew. Cannot be recovered without
  re-deriving from the incident's `seed` via the generator's own RNG
  sequence, which is out of this branch's scope (would mean re-deriving,
  not just reading, locked evidence). Only `condition_kind` /
  `perturbation_level` (categorical, e.g. `"30%"`) is usable as a stress
  intensity proxy.
- **Sensor node identity/placement for M11.6 incidents.** `sensor_count=4`
  is recorded and constant across all 125 incidents (not a useful
  stratifier there), but *which* 4 nodes are sensors per incident is not
  in the locked scenario record. Source-to-sensor graph distance and
  sensor-coverage statistics are therefore **not computable for the M11.6
  population** — flagged, not fabricated. (They *are* partially
  reconstructable in principle from `data/learning-v2/cycle-b2` per-example
  tensors, which do carry a `sensor_mask`; not pursued here because Phase 4
  already treats that corpus as a separate, secondary population and this
  specific covariate was not central to either failure hypothesis below.)
- **Stress conditions on novel topologies.** `locked_topology_test` is
  100% `NOMINAL` (20/20). There is no frozen evidence of how HydroCore-v5
  behaves under, e.g., `SENSOR_DROPOUT` on an unseen topology. Any claim
  about "unseen-topology + stress interaction" cannot be evidenced from
  M11.6 and is explicitly out of scope for a confirmatory finding.
- **Event/evidence-head raw outputs for M11.6** (only aggregate
  scout/planning counters are in the raw incident record).
- **Demand-perturbation/noise magnitude as a continuous covariate** for
  M11.6 (only categorical condition labels, per above).

## 3. Proposed stratifications (Phase 3)

Primary axes, each reported with subgroup `n` and a small-sample flag at
`n < 10` (matching `SplitConformalCalibrator`'s own
`minimum_group_size=10` convention already used throughout this
repository):

1. `seen_topology` (known n=105 vs. novel n=20) — the central question.
2. `condition_kind` (7 levels, known-family only for non-NOMINAL; novel
   topology has only NOMINAL, so any known-vs-novel comparison is
   implicitly NOMINAL-only unless stated otherwise).
3. `network_family` / `topology_id` (7 distinct networks: 3 known + 4
   novel, 5-35 examples each — every novel-topology-family cell is a
   small-sample flag by construction, n=5).
4. `node_count` (6 discrete values 6-13; too few distinct networks for
   quantile binning to be meaningful, reported as exact bins).
5. `source_degree` (1-4, discrete).
6. `source_betweenness_centrality` / `source_closeness_centrality` /
   `source_normalized_graph_position` (continuous; tercile bins given
   n=125, explicitly labeled exploratory given the small population).
7. `source_hops_to_reservoir`, `source_is_boundary_node`,
   `graph_diameter`, `graph_density`.
8. (Phase 4 population only) curriculum stage, `network_id`
   (golden-reference/branched-loop/loop-grid/coastal-branch), true-source
   rank buckets.

Deliberately NOT stratifying on: exact incident strength/timing/sensor
coverage for M11.6 (unavailable, Section 2), or any subgroup crossing two
of the above at once for the M11.6 population (n=125 total does not
support 2-way crossings beyond `seen_topology x condition_kind`, which is
degenerate as noted).

## 4. Leakage risks and controls

- M11.6 diagnostics are **read-only**: no locked file is written to; the
  new diagnostic table lives under `reports/evaluation/
  failure-mode-diagnostics/`, distinct from every `locked/`/`m9-*`/`m10-*`/
  `m11-*` path.
- The Phase 4 pilot re-run reuses `run_pilot.py`'s own seeded, leak-checked
  index construction verbatim (imported, not reimplemented) so it trains/
  evaluates on exactly the same examples the original pilot did; `coastal-
  branch` is still never used for training or calibration fitting, only
  evaluation.
- No topology or split label from any frozen evidence is used to train or
  tune anything in this branch — every model this branch touches (the
  Phase 4 re-run) is evaluated, never fine-tuned on evaluation-time labels.
- Graph-structural features are derived from topology structure only
  (`.inp` connectivity / `node_ids`+`edge_ids`), never from outcome labels
  — no risk of target leakage into a covariate.

## 5. Statistical comparisons

- M11.6 subgroup tables: point estimates + `n`; no inferential test is
  claimed confirmatory given `n=125` (and `n=20` for the entire novel-
  topology population) — differences are reported descriptively, matching
  `m11-6-metrics.json`'s own `DESCRIPTIVE_NON_GATING` framing for that
  split.
- Phase 4 paired pilot analysis: 2x2 top-1 transition table (McNemar-style
  descriptive counts, both arms share the same 280 examples), per-example
  true-source-rank delta, top-3-membership transition counts, and the
  **same paired-bootstrap convention already used by `run_pilot.py`/
  `bootstrap_followup.py`/M9.0/M9.6** (2000 resamples, 90% interval) —
  reused, not reinvented, for consistency with the rest of this
  repository's own evaluation methodology.

## 6. Expected output artifacts

- `scripts/hydrocore_v5_experimental/failure_mode_diagnostics/` — all
  analysis code (`graph_features.py`, `build_m11_6_diagnostic_table.py`,
  `rerun_topology_pilot_with_logging.py`, `analyze_m11_6_failure_modes.py`,
  `analyze_paired_pilot.py`).
- `reports/evaluation/failure-mode-diagnostics/` — `m11-6-diagnostic-
  table.jsonl` (+ provenance JSON), `m11-6-subgroup-metrics.json`,
  `m11-6-error-taxonomy.json`, `pilot-rerun/*-rows.jsonl` (per-example,
  per-arm, per-population), `pilot-rerun/reproduction-check.json`,
  `paired-pilot-analysis.json`.
- `docs/evaluation/experimental/FAILURE_MODE_DIAGNOSTICS_PLAN.md` (this
  document) and `FAILURE_MODE_DIAGNOSTICS_REPORT.md` (final report).
