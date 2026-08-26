# Graph-structural-encoder-v2: report (EXPERIMENTAL, NON-RELEASE)

Branch: `exp/graph-structural-encoder-v2`. Follows the plan in
`GRAPH_STRUCTURAL_ENCODER_V2_PLAN.md` (read that first for the bottleneck
analysis, arm definitions, leakage controls, and success criteria this
report evaluates against). Diagnostic/architecture experiment only: no
change to `models/hydrocore-v5-release`, `data/locked/`, any M11.6 artifact,
or any governance module (`hydroswarm.inference.ood`,
`hydroswarm.calibration.conformal`, any actionability gate). Every result
below is a fresh, small-variant, pilot-scale training run on
`data/learning-v2/cycle-b2` — **not** a re-evaluation of the frozen v0.2.1
release, and not directly comparable in magnitude to M11.6's own locked
numbers (different model variant, different, smaller corpus slice, single
seed). Negative results are retained in full, as required by the plan.

## 1. Implementation summary

Two purely additive changes, both fully backward-compatible (verified by
the existing test suite plus new dedicated tests — 21 new unit tests, all
passing):

1. **Two new deterministic, label-free feature modules**
   (`scripts/hydrocore_v5_experimental/graph_structural_encoder_v2/`):
   `structural_features.py` (Arm B — degree, betweenness, closeness,
   hop-to-reservoir, hop-to-dead-end, normalized graph position; computed
   from `edge_index`/`node_mask`/`source_candidate_mask` only) and
   `observability_features.py` (Arm C — hop-distance to nearest/mean/max
   active sensor, 1/2/3+-hop sensor-coverage fractions, local sensor
   coverage density; computed from `edge_index`/`sensor_mask` only). Both
   are per-candidate-node (every node, not just the true source),
   permutation-equivariant by construction, and covered by relabeling-
   invariance unit tests.
2. **`GraphStructuralEncoder`/`HydroCore` extension**
   (`src/hydroswarm/model/encoders.py`, `src/hydroswarm/model/core.py`):
   two new optional constructor parameters (`structural_feature_dim`,
   `use_edge_aggregation`, plus `edge_aggregation_source` for the capacity
   control) default to the module's exact original behavior — verified
   bit-for-bit identical output when unset, and the full existing model
   test suite (`test_model.py`, `test_prior_mode.py`,
   `test_variable_collate.py`, `test_scale_safety.py`,
   `test_gradient_coverage.py`, plus integration checkpoint tests) passes
   unmodified. When opted in, the encoder's input widens to accept
   structural/observability columns and/or gains one lightweight
   mean-neighbor aggregation pass over `edge_index` — deliberately smaller
   than the backbone's own `EdgeAwareGraphConv` (no learned edge-feature
   projection).

Training/evaluation harness (`run_experiment.py`) and statistical analysis
(`analyze_results.py`) reimplement `exp/failure-mode-diagnostics`'s
`run_pilot.py` harness structure (not imported — that branch is not
merged), reusing the same corpus, seed, stratified sampling, `OODDetector`/
`SplitConformalCalibrator` calls, and bootstrap convention.

## 2. Experimental-arm definitions

| Arm | `GraphStructuralEncoder` extra input | `edge_index` used | Purpose |
|---|---|---|---|
| A_CONTROL | none (original 4 scalars) | no | baseline |
| B_CENTRALITY | +6 structural columns | no | centrality hypothesis alone |
| C_OBSERVABILITY | +7 observability columns | no | competing observability hypothesis alone |
| D_STRUCTURAL_AGG | none | **yes** (1-hop mean aggregation) | does `edge_index` awareness alone help |
| D_CAPACITY_CONTROL | none | no (self-only, same param shapes) | capacity-matched control for D |
| E_COMBINED | +13 columns (B+C) | yes | best-justified combination |

All six arms trained on the **identical** 600 real-source examples (200/family
× `golden-reference`/`branched-loop`/`loop-grid`, seed `20260814`), 6 epochs,
CPU, `fp32=True`, `deterministic=True`, `configs/training-v5-causal.yaml`
optimizer settings — differing only in `GraphStructuralEncoder`'s
configuration. Evaluated on the same `validation` (n=300), `development_holdout`
(n=300, capped/seeded), `calibration` (n=712 real-source), and
`ood-UNSEEN_TOPOLOGY` (`coastal-branch`, n=280 real-source) populations.

## 3. Exact parameter counts

| arm | total params | `graph_encoder`-inclusive `encoders` bucket | delta vs A_CONTROL |
|---|---|---|---|
| A_CONTROL | 4,044,113 | 941,376 | — |
| B_CENTRALITY | 4,045,265 | 942,528 | +1,152 (0.03%) |
| C_OBSERVABILITY | 4,045,457 | 942,720 | +1,344 (0.03%) |
| D_STRUCTURAL_AGG | 4,118,225 | 1,015,488 | +74,112 (1.83%) |
| D_CAPACITY_CONTROL | 4,118,225 | 1,015,488 | +74,112 (1.83%, **identical** to D_STRUCTURAL_AGG) |
| E_COMBINED | 4,120,721 | 1,017,984 | +76,608 (1.90%) |

D_STRUCTURAL_AGG's parameter delta exceeded the plan's ~1% capacity-control
threshold (Section 8), so D_CAPACITY_CONTROL was built as an exactly
parameter-matched non-structural control (same submodules, same shapes,
`edge_index` never read — see plan doc Section 8) and trained/evaluated
identically. `backbone`/`heads`/`adapters` parameter counts are provably
unchanged across every arm (`test_parameter_report_reflects_arm_capacity_deltas`),
so every delta above is attributable entirely to `graph_encoder`.

## 4. Reproducible commands

```
git lfs pull --include="data/learning-v2/cycle-b2/tensors-normalized/**"
python3 -m pytest tests/unit/test_graph_structural_encoder_v2_features.py tests/unit/test_graph_structural_encoder_v2_model.py -q
python3 scripts/hydrocore_v5_experimental/graph_structural_encoder_v2/run_experiment.py   # ~85 min total, 6 arms, CPU
python3 scripts/hydrocore_v5_experimental/graph_structural_encoder_v2/analyze_results.py
```

Config/seed/dataset identifiers are recorded in
`reports/evaluation/graph-structural-encoder-v2/run-manifest.json` (seed
`20260814`, 6 epochs, 200 examples/family, split sizes). Per-arm training
summaries (elapsed time, epochs completed, final validation loss) and exact
parameter reports are recorded in each `<arm>-evaluation.json`'s `training`/
`parameter_report` fields. Trained checkpoints are not committed (gitignored,
deterministically regenerable from this script + the recorded seed/config —
same convention as `exp/failure-mode-diagnostics`'s
`experiments/topology-generalization/runs/`).

## 5. Baseline vs all-arm metric table

See `reports/evaluation/graph-structural-encoder-v2/metric-table.md` for
the full table. Summary (Top-1 / Top-3 / MRR):

| population | A_CONTROL | B_CENTRALITY | C_OBSERVABILITY | D_STRUCTURAL_AGG | D_CAPACITY_CONTROL | E_COMBINED |
|---|---|---|---|---|---|---|
| validation | .693/.873/.796 | .703/.873/.803 | .693/.877/.796 | .697/.877/.799 | .693/.877/.797 | .697/.870/.798 |
| development_holdout | .690/.880/.795 | .700/.880/.801 | .693/.873/.798 | .700/.877/.801 | .700/.880/.802 | .700/.883/.802 |
| ood-UNSEEN_TOPOLOGY | .375/.757/.586 | .368/.729/.572 | .354/.704/.562 | .361/.725/.569 | .361/.729/.570 | .371/.739/.583 |

**Aggregate reading:** every arm nudges known-topology Top-1 up by ~0-1pp
and **every arm is flat-to-negative on unseen-topology Top-1 and clearly
negative on unseen-topology Top-3** relative to A_CONTROL. This already
contradicts the primary hypothesis at the aggregate level; Sections 7-9
show the subgroup/paired analysis confirms it, not just the aggregate.

## 6. Condition-matched known/unseen topology comparison

`CLEAN` (`CurriculumStage`) is used as this corpus's own least-stressed-
condition proxy — **not** the same taxonomy as M11.6's `condition_kind`
(that field does not exist anywhere in `src/hydroswarm`; see Limitations).

| arm | known-CLEAN top1 (n=111) | unseen-CLEAN top1 (n=60) | gap |
|---|---|---|---|
| A_CONTROL | 0.712 | 0.417 | 0.295 |
| B_CENTRALITY | 0.721 | 0.433 | 0.287 |
| C_OBSERVABILITY | 0.703 | 0.417 | 0.286 |
| D_STRUCTURAL_AGG | 0.712 | 0.400 | 0.312 |
| D_CAPACITY_CONTROL | 0.712 | 0.400 | 0.312 |
| E_COMBINED | 0.694 | 0.433 | **0.260** |

E_COMBINED narrows the condition-matched gap the most, but by *lowering
known-topology* performance (0.712 → 0.694) about as much as it raises
unseen (0.417 → 0.433) — on n=60, this is not distinguishable from noise
(the pooled unseen-topology paired bootstrap for E_COMBINED, Section 8,
has CI [-0.025, +0.018], excluding neither direction). This gap narrowing
is **not** treated as a confirmed effect — see Section 13.

## 7. Low/medium/high centrality subgroup results

Terciles cut on A_CONTROL's pooled betweenness centrality (identical cut
points applied to all arms, since centrality is a deterministic function of
the physical example, identical across arms). Full table:
`centrality-subgroups.json`.

| arm | low (n=350) top1 | mid (n=237) top1 | high (n=293) top1 |
|---|---|---|---|
| A_CONTROL | 0.446 | 0.696 | 0.679 |
| B_CENTRALITY | 0.451 | 0.713 | 0.672 |
| C_OBSERVABILITY | 0.446 | 0.700 | 0.659 |
| D_STRUCTURAL_AGG | 0.443 | 0.709 | 0.672 |
| D_CAPACITY_CONTROL | 0.443 | 0.705 | 0.672 |
| E_COMBINED | 0.457 | 0.696 | 0.676 |

The low-vs-high centrality gap itself **replicates strongly** in this pilot
corpus (23-25pp, consistent with the diagnostics report's M11.6 finding of
a 26pp betweenness gap) — the diagnosed hard subgroup is real and
reproduces on fresh data. **Paired bootstrap CIs specifically on the
low-centrality subgroup** (`subgroup-bootstrap.json`, arm vs A_CONTROL,
matched by scenario_id):

| arm | low-centrality Δtop1 | 90% CI | excludes zero? |
|---|---|---|---|
| B_CENTRALITY | +0.006 | [-0.009, +0.020] | no |
| C_OBSERVABILITY | 0.000 | [-0.009, +0.009] | no |
| D_STRUCTURAL_AGG | -0.003 | [-0.017, +0.011] | no |
| D_CAPACITY_CONTROL | -0.003 | [-0.017, +0.011] | no |
| E_COMBINED | +0.011 | [0.000, +0.026] | **borderline (touches zero)** |

**No arm achieves a bootstrap CI excluding zero on the primary target
subgroup.** E_COMBINED's low-centrality point estimate is the largest and
its CI's lower bound touches exactly 0.0 — the closest any arm comes to a
confirmed effect, but not a confirmed one by the plan's own pre-registered
criterion.

## 8. Observability/distance subgroup results

Split at A_CONTROL's pooled median hop-distance-to-nearest-active-sensor
(median = 0 — most true sources in this tiny-graph corpus are themselves
directly instrumented; "short_distance" = co-located with an active sensor,
"long_distance" = hop distance ≥ 1). Full table: `distance-subgroups.json`,
subgroup bootstrap: `subgroup-bootstrap.json`.

| arm | short-distance top1 (n=597) | long-distance top1 (n=283) |
|---|---|---|
| A_CONTROL | 0.688 | 0.385 |
| B_CENTRALITY | 0.697 (Δ+0.008, CI [0.000,+0.017]) | 0.382 (Δ-0.004, CI [-0.025,+0.018]) |
| C_OBSERVABILITY | 0.687 (Δ-0.002, CI [-0.008,+0.003]) | 0.371 (Δ-0.014, CI [-0.032,+0.004]) |
| D_STRUCTURAL_AGG | 0.693 (Δ+0.005, CI [-0.002,+0.012]) | 0.375 (Δ-0.011, CI [-0.032,+0.011]) |
| D_CAPACITY_CONTROL | 0.693 (Δ+0.005, CI [-0.002,+0.012]) | 0.371 (Δ-0.014, CI [-0.035,+0.007]) |
| E_COMBINED | 0.698 (Δ+0.010, CI **[0.002,+0.018], excludes zero**) | 0.375 (Δ-0.011, CI [-0.032,+0.011]) |

The only subgroup-level effect that clears the pre-registered bootstrap bar
anywhere in this experiment is **E_COMBINED on short-distance (already the
*easier* subgroup)** — a small, real, positive effect (+1.0pp, CI excludes
zero), concentrated in the already-easy short-distance group, with a flat
(non-significant) effect on the harder long-distance group. This is exactly
the "misleading result" pattern the plan warned about (Section 6: "aggregate
improvement driven only by already-easy... cases").

## 9. Paired transition tables

Full tables (2×2 top1/top3, rank/margin/entropy deltas) in
`paired-transitions.json`. Headline numbers, `ood-UNSEEN_TOPOLOGY` (n=280,
the population where the primary hypothesis is most directly tested):

| arm | top1 Δ (bootstrap) | mean rank Δ | mean margin Δ | mean entropy Δ |
|---|---|---|---|---|
| B_CENTRALITY | -0.007 [-0.029,+0.014] | **+0.129 (worse)** | -0.034 | +0.080 |
| C_OBSERVABILITY | **-0.021 [-0.039,-0.004], excludes zero** | **+0.168 (worse)** | -0.043 | +0.107 |
| D_STRUCTURAL_AGG | -0.014 [-0.039,+0.007] | +0.132 (worse) | -0.041 | +0.105 |
| D_CAPACITY_CONTROL | -0.014 [-0.039,+0.007] | +0.125 (worse) | -0.041 | +0.105 |
| E_COMBINED | -0.004 [-0.025,+0.018] | +0.039 (worse, smallest) | -0.010 | +0.026 |

**C_OBSERVABILITY produces a statistically significant Top-1 *regression*
on unseen topology** (CI excludes zero, entirely on the negative side).
Every arm's true-source rank gets *worse* on average on unseen topology
(positive mean delta = worse rank), margin shrinks (less confident), and
posterior entropy rises (more diffuse) — the same qualitative signature
`exp/failure-mode-diagnostics` found for the earlier, unrelated
topology-relative-normalization pilot ("influential but mostly noisy...
redistributes existing uncertainty" rather than adding discriminative
signal). E_COMBINED is consistently the *least bad* of the four non-control
arms on every one of these unseen-topology metrics, but never clears zero
in the favorable direction either.

On known topologies (`validation`/`development_holdout`), transitions are
small and mostly non-significant, with two exceptions that both land on
`development_holdout`: **D_STRUCTURAL_AGG and D_CAPACITY_CONTROL both show
an identical +0.010 Top-1 delta with an identical [0.0033, 0.020] CI** —
see Section 10.

## 10. Calibration / OOD / safety results

| arm | calibration coverage (target 0.90) | ECE | mean candidate-set size |
|---|---|---|---|
| A_CONTROL | 0.9073 | 0.0542 | 2.593 |
| B_CENTRALITY | 0.9073 | **0.0428 (lowest)** | 2.624 |
| C_OBSERVABILITY | 0.9073 | 0.0520 | 2.638 |
| D_STRUCTURAL_AGG | 0.9073 | 0.0526 | 2.676 |
| D_CAPACITY_CONTROL | 0.9073 | 0.0512 | 2.674 |
| E_COMBINED | 0.9073 | 0.0537 | 2.632 |

Coverage is identical across every arm (an expected property of split
conformal calibration on the same calibration split/alpha, not itself
evidence of no change). B_CENTRALITY shows the lowest expected calibration
error of any arm — a small, real calibration improvement worth noting, not
offset by any coverage or actionability regression. No arm shows reduced
coverage, reduced proxy-actionable-rate on known topologies, or increased
proxy-actionable-rate on unseen topology: **every arm's `ood-UNSEEN_TOPOLOGY`
proxy_actionable_rate is exactly 0.0 and proxy_abstention_rate is exactly
1.0** (`metric-table.md`) — the fail-closed unseen-topology gate is intact
in every arm, unsurprising since no arm touches `hydroswarm.inference.ood`
or `hydroswarm.calibration.conformal`, but empirically confirmed rather than
assumed.

All 8 hard safety counters (`human_approval_bypassed`, `invariant_failures`,
`nonfinite_value_reached_decision`, `unverified_plan_surfaced_as_actionable`,
`rejected_plan_surfaced_as_safe`, `sampled_node_reselected`,
`sampling_budget_exceeded`, `inaccessible_sample_selected`) are **exactly
zero for every arm**. Per the plan's Section 10 scope note: this pilot-scale
localization-only harness does not exercise the sampling/planning/execution
control loop that produces these counters at the M11.6 evaluation tier, so
this is reported as "the code paths that produce these counters were never
invoked, by construction" — not as an independent re-verification at that
tier. No governance module is modified on this branch (confirmed by diff
review of the branch).

## 11. Centrality-vs-observability interpretation

Post-hoc analysis on A_CONTROL's own pooled localized rows (n=880,
`centrality-vs-observability.json`) — does the model that has *neither*
feature type fail more on low-centrality nodes because centrality is
informative, or only because centrality happens to proxy poor sensor
coverage?

- **Centrality and sensor-distance are nearly uncorrelated** in this corpus
  (r = -0.079) — they are not measuring the same thing.
- Univariate logistic coefficients (standardized): centrality **+0.420**,
  distance **-0.564** (distance predicts failure somewhat more strongly
  alone).
- **Jointly, both coefficients stay large and change little**: centrality
  +0.401 (vs +0.420 univariate), distance -0.550 (vs -0.564 univariate).
  Conditioning on one does **not** make the other's association vanish.
- Stratified check: centrality's association with Top-1 persists within
  both the short-distance stratum (coefficient +0.374) and the
  long-distance stratum (coefficient +0.457) — i.e. even among nodes that
  are (or are not) close to a sensor, more-central nodes are still easier
  to localize.

**Answer to the competing hypothesis (plan doc, "Centrality-vs-observability
question"):** (A) centrality provides independent value — supported; (B)
observability/distance provides *more* value than centrality — weakly
supported (larger standardized coefficient, but not by a large margin); (D)
"the apparent centrality effect disappears after conditioning on
observability" — **not supported**, the joint-model coefficients barely
move. Both signals are real and largely independent in this data; **neither
is a redundant proxy for the other.** This makes the arms' failure to
convert that association into a significant subgroup gain (Sections 7-9)
more notable, not less — the underlying evidence for *why* centrality
matters is intact, but this experiment's specific architectural
intervention did not successfully exploit it at this compute scale.

## 12. Did `edge_index` utilization actually help? (Arm D)

**No, and this is a clean negative result, not an ambiguous one.**
D_STRUCTURAL_AGG (real `edge_index` mean-aggregation) and D_CAPACITY_CONTROL
(the exactly parameter-matched, `edge_index`-blind control) produce:

- **Identical** total parameter counts (4,118,225, by construction).
- **Identical or statistically indistinguishable** Top-1 across every
  population and every subgroup examined (Sections 5, 7, 8, 9) — most
  strikingly, an **exactly identical** paired-bootstrap delta on
  `development_holdout` (+0.010, CI [0.0033, 0.020] for *both* arms).

The small, real Top-1 gain D_STRUCTURAL_AGG shows on known topology is
**fully reproduced by its non-structural, same-capacity control** — the
gain is attributable to the extra ~74k parameters in `GraphStructuralEncoder`,
not to anything `edge_index` supplies. This is exactly the failure mode the
plan's Section 8 capacity control was designed to catch, and it caught it.

## 13. Robust vs. concentrated-in-easy-cases

Per the plan's misleading-result checklist:

- **B_CENTRALITY**: small, mostly non-significant known-topology gains;
  flat-to-negative on unseen topology; no significant low-centrality gain.
  Not concentrated in easy cases, but also not a confirmed effect anywhere.
- **C_OBSERVABILITY**: the only arm with a **significant regression**
  (unseen-topology Top-1, Section 9) and no compensating gain anywhere.
- **D_STRUCTURAL_AGG / D_CAPACITY_CONTROL**: known-topology gain is a
  **capacity artifact** (Section 12), not structural; unseen-topology is
  flat-to-negative for both, identically.
- **E_COMBINED**: the only significant *positive* effect anywhere in this
  experiment (short-distance subgroup, Section 8) is in the **already-easy**
  subgroup (top1 0.688 baseline, among the highest of any subgroup examined)
  — textbook "gains concentrated in easy cases," the exact pattern the plan
  flagged as misleading rather than a genuine win. E_COMBINED's low-centrality
  point estimate is the largest of any arm but does not clear its own
  pre-registered CI bar.

**No arm in this experiment satisfies the plan's primary success
criterion** (a subgroup-bootstrap CI excluding zero on the low-centrality
or long-distance hard subgroups, without a compensating regression
elsewhere).

## 14. Negative/null results (retained in full)

- The low/high-centrality performance gap **replicates** on fresh data
  (Section 7) — this part of the original diagnostic finding is solid.
- **No arm significantly improves the low-centrality subgroup.**
- **No arm significantly improves unseen-topology localization**; one arm
  (C_OBSERVABILITY) significantly **regresses** it.
- **Arm D's apparent gain is a capacity artifact**, not a structural one —
  demonstrated directly by a parameter-matched control, not inferred.
- Unseen-topology rank/margin/entropy degrade (mildly, consistently) across
  every non-control arm — the same "redistributes existing uncertainty
  without adding discriminative signal" pattern
  `exp/failure-mode-diagnostics` found for an entirely different, unrelated
  prior intervention (topology-relative renormalization). Two independent
  interventions on two different architectural seams have now shown the
  same failure signature on unseen-topology transfer.
- Calibration coverage is unaffected everywhere; one arm (B_CENTRALITY)
  shows a small calibration-error improvement with no offsetting cost —
  the one unambiguously positive (if minor) result in this experiment.

## 15. Limitations

- **Pilot-scale, single-seed.** 6 epochs, one seed (`20260814`), CPU,
  `small` variant — matches the compute budget of the prior pilot this
  experiment was asked to follow up on, but is not the frozen v0.2.1
  production recipe (20 epochs, 3-seed campaigns per `MODEL_CARD.md`). A
  larger-scale/multi-seed run could shift these conclusions, particularly
  the borderline E_COMBINED low-centrality CI (Section 7).
- **`CLEAN` ≠ M11.6's `condition_kind`.** This corpus's `ScenarioExample`
  only carries `CurriculumStage` (`CLEAN`/`OPERATIONAL`/`DEGRADED`/`SHIFT`/
  `ADVERSARIAL`); the richer `condition_kind` taxonomy
  (`NOMINAL`/`SEVERITY_SHIFT`/`SENSOR_DROPOUT`/etc.) that M11.6's locked
  evaluation reports does not exist anywhere in `src/hydroswarm` and was not
  reconstructed here. Section 6's "condition-matched" comparison is a
  same-corpus analogue, not a replication of M11.6's own condition-matched
  finding.
- **Small, tiny-graph corpus.** Networks are 6-9 nodes
  (`dataset-report.json`); the sensor-distance median is exactly 0 (most
  true sources are themselves instrumented), which degenerately collapsed
  the "mid-distance" tercile to n=0 in the centrality-vs-observability
  stratification (Section 11) — the short/long split remains informative,
  but a finer-grained distance analysis was not possible at this network
  scale.
- **Hard safety counters are structurally zero, not independently
  re-verified** at this evaluation tier (Section 10) — this pilot harness
  never exercises the sampling/planning/execution loop that produces them.
- **This pilot's `A_CONTROL` is not the frozen release model** — absolute
  Top-1/Top-3 numbers here (e.g. 0.69 known-topology Top-1) are not
  comparable in magnitude to M11.6's own locked 0.552, since architecture
  variant, training compute, and corpus differ; only within-experiment,
  paired, same-arm-set comparisons are valid.
- **Post-hoc logistic coefficients (Section 11) are associational, not
  causal**, fit with a small dependency-free gradient-descent
  implementation (no `scikit-learn` in this environment) — adequate for
  the directional question asked, not a rigorously validated statistical
  model.

## 16. Final recommendation: **reject** (this specific intervention), with a scoped continuation

None of arms B/C/D/E met the plan's own pre-registered primary success
criterion. One arm (C_OBSERVABILITY) produced a statistically significant
regression with no offsetting benefit. Arm D's only positive signal is
fully explained by a capacity-matched control. This specific set of
architectural changes to `GraphStructuralEncoder` — feeding it static
centrality/observability columns and/or one lightweight `edge_index`
aggregation pass, at this compute scale — is **not** validated and should
**not** be promoted or scaled up as implemented.

This is not a rejection of the underlying diagnosis. The low/high-centrality
gap replicated cleanly on fresh data, and the post-hoc analysis (Section 11)
shows centrality and observability both carry real, largely independent
signal about *why* the model fails — the diagnostic evidence motivating
this experiment remains intact. What failed is this experiment's specific
mechanism for getting that signal into the model early (a shallow encoder
with a handful of extra scalar columns and one small aggregation pass),
consistent with (and now a second independent data point for) the
diagnostics report's own "influential but mostly noisy" pattern.

**Recommended next step is continue research, not a larger-scale validation
of this exact design and not a full graph-native rewrite:**

1. E_COMBINED's borderline low-centrality result (CI lower bound exactly
   0.0) and its being the least-bad arm on every unseen-topology metric
   make it the only candidate worth a second, larger/multi-seed run before
   being fully discarded — but on its own current evidence, not yet a
   candidate for promotion.
2. Consider feeding structural/observability features **later** in the
   architecture (e.g. concatenated onto the backbone's post-message-passing
   hidden state, or into the source_node_head directly) rather than only
   through the early, low-capacity `GraphStructuralEncoder` fusion point —
   this experiment only tested the earliest injection point in the
   architecture.
3. C_OBSERVABILITY's clean, significant regression is worth understanding
   mechanistically (e.g. does adding sensor-distance features make the
   model *more* reliant on sensor placement in a way that transfers worse
   to an unseen topology's different sensor layout?) before any future
   attempt reuses raw hop-distance-to-sensor features.
4. A full graph-native topology-encoder rewrite is **not** justified by
   this evidence — the negative result here is about a small, targeted
   change's effectiveness at this compute scale, not about the current
   message-passing backbone's structural capacity being the bottleneck.

## Required questions — direct answers

- **Did structural information improve unseen-topology localization?** No.
  Flat-to-negative for every arm; one arm significantly worse.
- **Did it specifically help peripheral/low-centrality sources?** No arm
  cleared its pre-registered bootstrap CI on the low-centrality subgroup.
- **Was centrality itself useful, or was observability the real signal?**
  Both carry real, largely independent signal (near-zero correlation,
  stable coefficients after mutual conditioning) — neither dominates or
  subsumes the other.
- **Did lightweight `edge_index` aggregation add value?** No — its only
  apparent gain is reproduced identically by a parameter-matched control
  that never reads `edge_index`.
- **Were gains robust across subgroups or concentrated in easy cases?**
  Concentrated in easy cases (E_COMBINED's one significant gain is in the
  already-easier short-distance subgroup) or absent.
- **Did any arm harm known-topology performance, calibration, OOD
  behavior, or safety?** No known-topology regression; no calibration
  coverage regression (one arm improved ECE); OOD fail-closed behavior and
  all 8 hard safety counters intact in every arm. C_OBSERVABILITY harmed
  unseen-topology performance specifically.
- **Is a full graph-native/GNN rewrite now justified?** No — this evidence
  argues for a different injection point or feature formulation within the
  existing architecture before a larger rewrite, not for abandoning it.
- **What should the next experiment be?** A later-injection-point variant
  of Arm E (structural/observability features fused after backbone message
  passing, closer to `source_node_head`), evaluated with the same paired
  protocol and a second seed, prioritized over any other direction from
  this report's Section 9 hypothesis list.
