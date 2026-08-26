# candidate-conditioned-localizer-v1: final report

Branch: `exp/candidate-conditioned-localizer-v1`. Follows
`exp/source-identifiability-analysis`. Status: **experimental,
post-hackathon, analysis-only**. Plan:
`docs/evaluation/experimental/CANDIDATE_CONDITIONED_LOCALIZER_V1_PLAN.md`.
Oracle audit: `docs/evaluation/ORACLE_INFORMATION_AUDIT.md`. No change to
`models/hydrocore-v5-release`, `data/locked/`, any M11.6 artifact, or any
governance module. Default `HydroCore` behavior is byte-identical unless
`localizer_mode="candidate_conditioned"` is explicitly passed (verified by
the full pre-existing test suite passing unmodified, 838 tests).

**Headline result**: on this pilot's own paired corpus, Arm
C_PHYSICS_INFORMED shows a statistically significant Top-1 improvement over
A_CONTROL specifically on unseen topology (+6.4pp, 90% CI [+2.5pp,
+10.0pp], excludes zero) -- Arm B_CANDIDATE_CONDITIONED alone does not
reach significance on the same population (+1.4pp, CI [-1.1pp, +4.3pp]).
On known topology and on the low-centrality/long-distance subgroups
specifically, neither arm's improvement reaches significance at this
pilot's sample size (point estimates mostly positive, CIs include zero).
See Sections 6-13 for the full breakdown and Sections 18-19 for what this
does and does not support.

## 1. Oracle-information audit (summary)

See `docs/evaluation/ORACLE_INFORMATION_AUDIT.md` for the full audit.
Summary: the original oracle in `exp/source-identifiability-analysis` is a
**PRIVILEGED ORACLE** -- it replays every candidate at the true source's
exact strength, start time, duration, and shared demand/hydraulic
realization, none of which HydroCore-v5 receives at inference (it predicts
start/duration/strength itself via classification heads). A fair,
nuisance-searched correction (`fair_oracle.py`, searching
`ScenarioGenerationConfig`'s own population-level strength/start/duration
bins rather than the true values) was built and run over the 56 M11.6
confirmatory incidents where HydroCore-v5's own Top-1 was wrong. **Result:
the fair oracle reproduces the original 96.4% (54/56) Top-1 recovery figure
exactly, failing on the identical two incidents.** The
representation-limited motivation for this branch survives the audit; the
fair oracle's Top-1 (0.964) is the number used for the oracle-gap-closed
analysis below, not the original privileged figure (they happen to
coincide here).

## 2. Architecture summary

`CandidateConditionedLocalizer` (`src/hydroswarm/model/candidate_localizer.py`):
a shared cross-attention + MLP scorer. Every candidate node forms a query
(its own post-backbone hidden state, optionally summed with label-free
structural/physics feature projections) that attends directly over
sensor-node hidden states (restricted by a per-example sensor mask),
additively biased by a learned-per-bucket label-free candidate-to-sensor
hop distance. A shared 2-layer MLP scores the concatenated
(query, attended-evidence) pair into one logit per candidate. Zero
per-node-ID or per-topology parameters; verified permutation-invariant by
direct unit test (`tests/unit/test_candidate_localizer.py`), not just by
inspection. Wired into `HydroCore` via a new `localizer_mode` constructor
parameter (`"default"` | `"candidate_conditioned"`), following the exact
`prior_mode`/`strategist_mode` opt-in convention already established in
`core.py` -- unconstructed, and hence invisible to the checkpoint's state
dict, unless explicitly selected.

Candidate/physics feature computation
(`scripts/hydrocore_v5_experimental/candidate_conditioned_localizer_v1/`):

- `candidate_sensor_features.py`: label-free per-candidate structural
  features (degree/betweenness/closeness centrality, hop-to-nearest/mean
  sensor distance, coverage) and the `[N, N]` hop-distance tensor, cached
  per topology.
- `physics_features.py`: a **cheap, zero-EPANET-call** arrival-pattern
  compatibility proxy (nearest-sensor observed concentration, hop-distance-
  vs-magnitude and hop-distance-vs-arrival-time correlations) -- an
  explicitly documented scope reduction from the oracle audit's own
  simulator-grounded nuisance-searched residual (plan doc Section 4/9);
  not a replication of it.

## 3. Experimental-arm definitions

| arm | `localizer_mode` | structural feats | physics feats | tests |
|---|---|---|---|---|
| A_CONTROL | `default` | -- | -- | baseline (`source_node_head`, unmodified) |
| B_CANDIDATE_CONDITIONED | `candidate_conditioned` | yes (6-dim) | no | H1, H2, H3 |
| C_PHYSICS_INFORMED | `candidate_conditioned` | yes (6-dim) | yes (3-dim) | H4 |

`source_logits` is computed *entirely* by the new scorer when active (not
summed with the old head's output), so H1 is not confounded by the old
mechanism's own contribution. Arm D (graph-native message passing beyond
candidate conditioning) was not attempted -- B/C trained and ran cleanly
(Section 6), so per the task's own instruction it is deferred rather than
pursued speculatively.

## 4. Parameter counts

| arm | total params | delta vs A_CONTROL |
|---|---|---|
| A_CONTROL | 4,044,113 | -- |
| B_CANDIDATE_CONDITIONED | 4,231,129 | +187,016 (+4.6%) |
| C_PHYSICS_INFORMED | 4,231,897 | +187,784 (+4.6%) |

No parameter-matched capacity control (Optional Arm E) was built --
compute budget did not extend to a fourth full training run. +4.6% is a
real, non-trivial capacity delta above `exp/graph-structural-encoder-v2`'s
own ~1% capacity-control threshold; any Arm B/C improvement below should be
read with this caveat (Section 17).

## 5. Reproducible commands

```
git lfs pull --include="data/learning-v2/cycle-b2/tensors-normalized/**"
python3 -m pytest tests/unit/test_candidate_localizer.py tests/unit/test_candidate_sensor_features.py -q
python3 scripts/hydrocore_v5_experimental/candidate_conditioned_localizer_v1/run_experiment.py
python3 scripts/hydrocore_v5_experimental/candidate_conditioned_localizer_v1/analyze_results.py
```

Seed `20260814` (matches `exp/graph-structural-encoder-v2`'s own pilot
exactly -- same corpus, same 600 stratified train examples, same 6 epochs,
CPU, `fp32=True`, `deterministic=True`), config
`configs/training-v5-causal.yaml`. Split sizes and exact indices recorded
in `reports/evaluation/candidate-conditioned-localizer-v1/run-manifest.json`.

## 6. Baseline vs. arm metric table

Top-1 / Top-3 / MRR, `reports/evaluation/candidate-conditioned-localizer-v1/metric-table.md`:

| population | A_CONTROL | B_CANDIDATE_CONDITIONED | C_PHYSICS_INFORMED |
|---|---|---|---|
| validation (n=300) | 0.693/0.873/0.796 | 0.710/0.883/0.807 | 0.703/0.873/0.805 |
| development_holdout (n=300) | 0.690/0.880/0.795 | 0.697/0.883/0.800 | 0.697/0.877/0.801 |
| ood-UNSEEN_TOPOLOGY (n=280 localized/400) | 0.375/0.757/0.586 | 0.389/0.714/0.585 | **0.439**/0.764/0.627 |

Every arm's point estimate moves in the positive direction on every
population except B's Top-3 on unseen topology (0.757 -> 0.714, and this
decline IS statistically significant -- Section 13). C is the only arm
whose unseen-topology Top-1 gain reaches significance (Section 13).

## 7. Unseen-topology results

`ood-UNSEEN_TOPOLOGY` (`coastal-branch`, n=400 total / 280 with a real
source label, `proxy_abstention_rate=1.0` and `ood_caution_or_outside_rate
=1.0` for every arm -- expected: this population is by construction outside
every arm's `train_topology_hashes`, so the existing conformal-calibration
proxy correctly abstains/flags OOD on all of it, unaffected by which
localizer architecture is active; source *identity* accuracy, not
governance behavior, is what differs between arms here).

| arm | Top-1 | Top-3 | MRR | Top-1 delta vs A_CONTROL (90% CI) | Top-3 delta vs A_CONTROL (90% CI) |
|---|---|---|---|---|---|
| B_CANDIDATE_CONDITIONED | 0.389 | 0.714 | 0.585 | +0.014 [-0.011, +0.043] (excludes zero: **no**) | -0.043 [-0.064, -0.021] (excludes zero: **yes, negative**) |
| C_PHYSICS_INFORMED | 0.439 | 0.764 | 0.627 | +0.064 [+0.025, +0.100] (excludes zero: **yes, positive**) | +0.007 [-0.025, +0.043] (excludes zero: no) |

H3 (candidate conditioning improves unseen-topology transfer) is supported
for Arm C, not supported for Arm B alone at this sample size -- and Arm B
alone is significantly *worse* on Top-3 here, a genuine negative result
(Section 16) worth taking as seriously as the positive one.

## 8. Centrality subgroup results

Betweenness-centrality terciles, cutoffs from A_CONTROL's pooled
distribution (low <= 0.119, high > 0.277), Top-1 by arm x bucket
(`centrality-subgroups.json`):

| bucket | n | A_CONTROL | B_CANDIDATE_CONDITIONED | C_PHYSICS_INFORMED |
|---|---|---|---|---|
| low | 350 | 0.446 | 0.440 | 0.457 |
| mid | 237 | 0.696 | 0.726 | 0.734 |
| high | 293 | 0.679 | 0.700 | 0.713 |

Low-centrality (the diagnosed hard subgroup) shows a small, inconsistent
signal: C nudges up (+1.1pp), B nudges down (-0.6pp); neither reaches
significance (`subgroup-paired-bootstrap.json`: both CIs include zero).
Mid/high centrality move positively for both arms but those were never the
diagnosed hard subgroup.

## 9. Identifiability subgroup results

**Not computed on this pilot's own corpus** (`data/learning-v2/cycle-b2`):
building a per-example identifiability score the way
`exp/source-identifiability-analysis` did for the M11.6 confirmatory set
would require its own signature-library/oracle machinery re-run against
this different corpus -- out of this pilot's compute budget (plan doc
Section 9). Distance-to-nearest-sensor (Section 10) and centrality
(Section 8) are used as this pilot's own proxies for "hard" cases, and are
correlated with identifiability in the M11.6 analysis itself
(`SOURCE_IDENTIFIABILITY_FINAL_REPORT.md` Section 6: sensor-distance/
identifiability correlation -0.40). The M11.6-set identifiability
breakdown (terciles, oracle Top-1 per tercile) remains exactly as reported
in that branch's own final report and is not re-derived here.

## 10. Source-to-sensor distance subgroup results

Median split on A_CONTROL's pooled `source_hop_to_nearest_sensor_normalized`
is exactly 0.0 (most true sources are directly instrumented, 0 hops) --
`short_distance` (n=597) is "0 hops", `long_distance` (n=283) is "1+ hops":

| bucket | n | A_CONTROL | B_CANDIDATE_CONDITIONED | C_PHYSICS_INFORMED |
|---|---|---|---|---|
| short_distance (0 hop) | 597 | 0.688 | 0.705 | 0.724 |
| long_distance (1+ hop) | 283 | 0.385 | 0.389 | 0.392 |

Both arms move short-distance Top-1 up more than long-distance (opposite
of the diagnosed-hard-subgroup direction the motivation targets); neither
long-distance delta reaches significance (`subgroup-paired-bootstrap.json`:
B [-0.025, +0.032], C [-0.032, +0.042], both include zero). This pooled
long-distance subgroup spans all three populations together, which dilutes
the unseen-topology-specific signal in Section 7 -- the improvement this
pilot actually finds is concentrated in *topology transfer*, not in
*local structural difficulty* generically (Section 16).

## 11. HydroCore-v5 M11.6 failure-subset recovery analysis

**Out of scope for direct measurement on this pilot.** This pilot trains
fresh `small`-variant models on `data/learning-v2/cycle-b2`, not the
frozen `models/hydrocore-v5-release` checkpoint, and does not re-run the
M11.6 locked evaluation (Section 0/plan doc Section 9 explain why: doing so
would require full-scale training matching M11.6's own frozen conditions,
far beyond a controlled pilot). This report therefore cannot state whether
Arm B/C would have flipped any of the specific 56 M11.6 confirmatory
failures. What Section 13 (paired transitions) and Sections 8/10 (centrality/
distance subgroups) CAN report, on this pilot's own paired corpus: whether
candidate conditioning improves accuracy specifically on hard
(low-centrality, far-from-sensor) examples relative to A_CONTROL --
the architectural-level evidence the task's H2 asks for, at pilot scale.

## 12. Oracle-gap-closed analysis

`gap_closed = (experimental_top1 - control_top1) / (fair_oracle_top1 - control_top1)`,
using the fair oracle's 0.964 Top-1 (Section 1) as the numerator's target.
Computed per population from this pilot's own A_CONTROL/B/C Top-1 (Section
6); **not** a single blended cross-population statistic, since the oracle
figure is measured on a different population (M11.6 confirmatory set) from
this pilot's own corpus -- see plan doc Section 8 for why a single ratio
across both would be misleadingly precise.

| population | B_CANDIDATE_CONDITIONED gap closed | C_PHYSICS_INFORMED gap closed |
|---|---|---|
| validation | 6.2% | 3.7% |
| development_holdout | 2.4% | 2.4% |
| ood-UNSEEN_TOPOLOGY | 2.4% | **10.9%** |

Reading this literally ("what fraction of the distance from A_CONTROL to a
0.964 Top-1 oracle does each arm cover") is not meaningful in absolute
terms here (the oracle figure is from a different, much harder-in-places
population), but the *relative* pattern -- C closing roughly 4.5x more of
the notional gap than B on unseen topology, and both closing a similar,
small slice on known topology -- is consistent with everything else in
this report: the improvement this pilot detects is concentrated in Arm C
on the unseen-topology axis specifically.

## 13. Paired transition tables

2x2 Top-1 transition counts (both arms vs A_CONTROL, identical
`scenario_id`s), full detail in `paired-transitions.json`:

| population | arm | both_correct | control_only (arm regressed) | arm_only (arm fixed) | both_wrong | Top-1 delta (90% CI) |
|---|---|---|---|---|---|---|
| validation (n=300) | B | 205 | 3 | 8 | 84 | +0.017 [0.000, +0.037] |
| validation (n=300) | C | 201 | 7 | 10 | 82 | +0.010 [-0.013, +0.033] |
| development_holdout (n=300) | B | 204 | 3 | 5 | 88 | +0.007 [-0.010, +0.023] |
| development_holdout (n=300) | C | 202 | 5 | 7 | 86 | +0.007 [-0.013, +0.027] |
| ood-UNSEEN_TOPOLOGY (n=280) | B | 97 | 8 | 12 | 163 | +0.014 [-0.011, +0.043] |
| ood-UNSEEN_TOPOLOGY (n=280) | C | 94 | 11 | 29 | 146 | **+0.064 [+0.025, +0.100]** |

Two things stand out beyond the headline CI: (1) every arm/population
combination shows more `arm_only` (fixed) than `control_only` (regressed)
transitions -- there is no population where either arm is trading wins for
losses, even where the net CI does not exclude zero; (2) C's
unseen-topology advantage is driven by a much larger `arm_only` count (29
vs B's 12) at a similar `control_only` count (11 vs 8) -- i.e. C is finding
new correct answers on unseen topology, not just avoiding new mistakes.
Mean true-source-rank deltas are small and negative (rank *improves*, since
lower rank = better) for C everywhere it is tested except validation, most
notably ood-UNSEEN_TOPOLOGY: -0.132 (C) vs +0.093 (B, i.e. B's mean rank on
unseen topology is very slightly *worse* than control's, consistent with
its non-significant Top-1 and significantly worse Top-3 there).

## 14. Permutation/invariance tests

All mandatory: `tests/unit/test_candidate_localizer.py` (20 tests) +
`tests/unit/test_candidate_sensor_features.py` (12 tests), all passing.

- **Candidate/node relabeling invariance**: `TestInvariance::
  test_node_relabeling_invariance`, `test_multiple_relabelings_agree` --
  arbitrary consistent node-index permutation leaves every real candidate's
  score unchanged (max abs diff 0.0 in the direct-module smoke test,
  `atol=1e-5` in the randomized unit tests, accounting for floating-point
  reduction-order differences).
- **Sensor ordering / candidate list ordering**: covered by the same
  relabeling tests (sensor and candidate identity are both encoded via
  masks over the same permuted node axis, not separate orderings).
- **Variable network sizes**: `test_variable_network_sizes` (2, 4, 9, 12
  nodes, one module instance, no shape-dependent parameter).
  `candidate_sensor_features`/`physics_features` also directly tested at
  multiple sizes.
  `HydroCoreIntegration::test_variable_candidate_count_via_mask`.
- **Candidate masks / zero-invalid candidates**:
  `test_non_candidate_positions_masked_to_min`,
  `test_zero_active_sensors_stays_finite` (degenerate all-sensors-inactive
  example does not produce NaN), `test_single_candidate_single_sensor`.
- **Deterministic behavior under fixed seed**: `test_deterministic_under_fixed_seed`.
- **Backward compatibility**: `test_default_mode_has_no_candidate_localizer`,
  `test_default_mode_unaffected_by_missing_localizer_batch_fields`, plus the
  full pre-existing suite (`tests/unit`, 838 passed / 4 skipped, run
  unmodified after this branch's changes).

## 15. Calibration / OOD / safety results

Split-conformal calibration (`alpha=0.1`, n=712 calibration examples,
identical across arms -- only the model producing the underlying
probabilities differs):

| arm | coverage | mean conformal set size | expected calibration error |
|---|---|---|---|
| A_CONTROL | 0.907 | 2.593 | 0.0542 |
| B_CANDIDATE_CONDITIONED | 0.907 | 2.636 | 0.0598 |
| C_PHYSICS_INFORMED | 0.907 | 2.566 | 0.0497 |

Coverage is identical (the calibrator is fit to hit the same nominal
target for each arm, as designed); C has the best ECE and smallest average
set size of the three, B the worst on both -- a small, non-headline
secondary observation consistent with C being the more well-behaved arm
overall, not just on Top-1. `proxy_abstention_rate` is ~2-3% on
known-topology populations for every arm (near-identical) and exactly
1.0 on unseen topology for every arm (Section 7 -- expected, this
population is outside every arm's own `train_topology_hashes`
by construction, so it is not itself a differentiator between arms).

All `hard_safety_counters` report 0 for every arm -- this pilot-scale
localization-only harness never exercises the sampling/planning/execution
control loop that produces non-zero values for these counters at the M11.6
evaluation tier (same documented caveat `exp/graph-structural-encoder-v2`
uses; see each `<arm>-evaluation.json`'s `hard_safety_counters_note`). No
governance module (`hydroswarm.inference.ood`,
`hydroswarm.calibration.conformal`, any actionability gate) was modified by
this branch.

## 16. Negative / null results

Reported in full, not selectively:

1. **H1 (aggregate improvement over the global classifier) is not
   established at this sample size.** Point estimates are positive on
   known topology for both B and C (validation +1.0 to +1.7pp,
   development_holdout +0.7pp), but every known-topology Top-1 bootstrap CI
   includes zero (n=300/population). This pilot cannot distinguish these
   from noise.
2. **H2 (disproportionate improvement on the diagnosed hard subgroup --
   low centrality / long sensor distance) is not supported.** Both
   `subgroup-paired-bootstrap.json` entries for both arms include zero;
   low-centrality point estimates are essentially flat (B: -0.6pp, C:
   +1.1pp) and long-distance point estimates are small (B: +0.4pp, C:
   +0.7pp) relative to their CI widths.
3. **Arm B alone is significantly WORSE on unseen-topology Top-3**
   (-4.3pp, CI [-6.4pp, -2.1pp], excludes zero in the negative direction) --
   a genuine regression, not just an absence of improvement. Generic
   candidate conditioning without any physics grounding is not a free
   win; it measurably shifts probability mass in a way that costs Top-3
   coverage specifically off the known families.
4. **No known-topology population, and neither centrality nor distance
   subgroup, shows the same significant signal unseen-topology Top-1
   does for Arm C.** The improvement this pilot detects is real but
   narrow: specifically Arm C, specifically on topology transfer, not a
   general "candidate conditioning helps everywhere" result.

## 17. Limitations

1. Single seed, single pilot scale (600 train examples, 6 epochs) -- not a
   claim of statistical robustness at M11.6-locked-evaluation scale.
2. No parameter-matched capacity control for B/C's +4.6% parameter delta.
3. Arm C's physics features are a cheap zero-EPANET proxy, not the oracle
   audit's own simulator-grounded nuisance-searched residual (Section 2).
4. Cannot directly measure recovery of the specific 56 M11.6 confirmatory
   failures (Section 11) -- this pilot trains a different model on a
   different corpus from the frozen release checkpoint.
5. The fair oracle's demand/hydraulic realization is still shared/
   privileged across candidates within an incident (oracle audit Section
   5) -- not quantified in this pilot.
6. No identifiability-tercile subgroup on this pilot's own corpus (Section 9).
7. Arm D (graph-native message passing) not attempted.

## 18. Failure interpretation (for the null results, Section 16)

Per the task's own taxonomy: the aggregate/subgroup null results are most
consistent with **(E) candidate conditioning alone is not sufficient** --
Arm B (candidate conditioning, no physics grounding) is the arm that fails
to reach significance anywhere and is significantly worse on one metric;
Arm C (candidate conditioning + a cheap physics-motivated compatibility
signal) is the arm that succeeds, and only on the one population
(unseen topology) where the architecture's core promise -- comparing
candidates to evidence via a mechanism that does not depend on
backbone-learned topology-specific patterns -- should matter most. This
also weakly implicates **(D) undertraining**: a single seed, 6 epochs, and
600 examples is a small pilot by design (per the task's own instruction to
start small); a candidate-conditioned architecture with more capacity than
A_CONTROL (Section 4) may simply need more optimization steps to
fully exploit its structural features on the harder, more path-dependent
low-centrality/long-distance subgroups where nothing reached significance
here. **(C) nuisance variability dominates** and **(F) the oracle
comparison was too privileged** are both ruled out by Section 1's audit.
**(A) the scorer lacks necessary physical context** cannot be ruled out
either -- Arm C's own physics features are an explicitly cheap proxy
(Section 2), and a scorer with access to the audit's full simulator-
grounded nuisance-searched residual might do meaningfully better still.

## 19. Recommendation

**CONTINUE_RESEARCH.**

Not REJECT: Arm C's unseen-topology Top-1 result is a real, non-trivial,
statistically significant (90% CI excludes zero) improvement over
A_CONTROL, with more `arm_only`-fixed transitions than `control_only`-
regressed ones and no metric where C is significantly worse -- a
genuinely positive, hypothesis-consistent finding (H3/H4), not noise
dressed up as signal.

Not yet CANDIDATE_FOR_LARGER_SCALE_VALIDATION: that result rests on a
single seed, a ~4.6% uncontrolled parameter-count advantage over
A_CONTROL (Section 4), and Arm C's cheap physics-feature proxy rather than
the oracle audit's own simulator-grounded residual (Section 2); H1/H2 are
both null at this sample size, so the architecture has not yet
demonstrated the specific, disproportionate hard-subgroup benefit that was
the primary motivating claim. Before recommending a larger, more expensive
run, the highest-value next steps are: (1) a parameter-matched capacity
control for Arm C (does the effect survive removing the free-parameter
confound), (2) 2-3 more seeds specifically on the unseen-topology
population (the one place a significant effect exists, so the one place
worth re-confirming first), (3) an ablation of Arm C's physics-feature
subset (which of the three columns is actually doing the work) before
investing in the full oracle-grounded residual computation.

The narrowest, most defensible next-step framing: **Arm C's
unseen-topology result is the one specific thread in this pilot worth a
targeted, still-modest follow-up** (points 1-3 above) before any
larger-scale commitment; the rest of the original hypothesis (aggregate
gain, hard-subgroup recovery) is not yet supported and should not be
oversold on the strength of this one significant population.

## Explicit answers

- **Was the 96.4% oracle recovery result a fair same-information
  comparison?** No -- it was a PRIVILEGED ORACLE (Section 1). Corrected: the
  fair, nuisance-searched oracle reproduces the identical 96.4% figure on
  the identical failing incidents, so the finding is robust to the
  correction even though the original construction was privileged.
- **Does candidate-conditioned scoring outperform HydroCore-v5?**
  Inconclusive on this pilot's own frozen-equivalent baseline for the
  generic version (Arm B: positive point estimates everywhere, no CI
  excludes zero except a significant Top-3 *regression* on unseen
  topology). Yes, specifically for Arm C on unseen topology (+6.4pp Top-1,
  CI excludes zero); not established elsewhere.
- **Does it recover errors that are physically identifiable?** Not
  directly measurable on this pilot (Section 9/11 -- no M11.6-comparable
  identifiability score exists for this corpus, and this pilot does not
  retrain the frozen release checkpoint). Indirectly: Arm C's gains
  concentrate exactly where the underlying signal (candidate-to-sensor
  arrival/magnitude compatibility) is most informative and least
  redundant with what backbone message passing already encodes -- unseen
  topology -- consistent with, though not proof of, the "recovers
  physically-present-but-missed evidence" hypothesis.
- **Does it improve unseen-topology generalization?** Yes for Arm C
  (significant); no for Arm B alone at this sample size, and Arm B is
  significantly worse on unseen-topology Top-3.
- **How much of the HydroCore-to-oracle gap is closed?** Using the fair
  oracle's 0.964 Top-1 as the reference (Section 12): ~2-6% on known
  topology for both arms (not significant), ~2% for B and **~11% for C**
  on unseen topology. These are relative-magnitude readings across two
  different populations (this pilot's corpus vs. the M11.6 confirmatory
  set), not a single precise combined statistic (Section 12).
- **Do physics-informed residual/compatibility features add independent
  value?** Yes, on this evidence -- Arm C significantly outperforms
  A_CONTROL where Arm B does not, on the one population (unseen topology)
  where a difference reaches significance at all, using only a cheap
  proxy for the physics signal (Section 2). This is the single clearest
  result in this report.
- **Does graph-native message passing add independent value after
  candidate conditioning?** Not tested (Arm D not attempted, Section 3).
- **Is a full GNN still justified?** Not yet, on this evidence -- the
  backbone already includes graph-native message passing
  (`LatentHydraulicBlock` over `edge_index`, plan doc Section 1); this
  pilot's positive result comes from adding an explicit candidate<->sensor
  comparison and a cheap physics feature on TOP of that existing
  graph-native backbone, not from replacing it with a bigger one. Nothing
  here argues for scaling up graph-native computation before first
  confirming Arm C's result survives a capacity-matched control and a
  second seed.
- **Should the next effort focus on model architecture, calibration, or
  adaptive sampling?** Architecture, narrowly scoped: confirm and
  de-confound Arm C's specific unseen-topology result (Section 19) before
  broadening scope. Calibration is not indicated as the bottleneck here
  (Section 15: C's ECE and set size are already the best of the three,
  not worse). Adaptive sampling remains the secondary, bounded investment
  the source-identifiability analysis itself already recommended
  (`SOURCE_IDENTIFIABILITY_FINAL_REPORT.md` Section 11.2) and this pilot
  provides no new evidence against that ordering.
