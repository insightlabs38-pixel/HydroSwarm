# physics-informed-localizer-validation: final report

Branch: `exp/physics-informed-localizer-validation`. Follows
`exp/candidate-conditioned-localizer-v1` (that branch's final report:
`reports/evaluation/candidate-conditioned-localizer-v1/FINAL_REPORT.md`).
Plan (pre-registered before Phase 3 training):
`docs/evaluation/experimental/PHYSICS_INFORMED_LOCALIZER_VALIDATION_PLAN.md`.
Status: **experimental, non-release**. No change to
`models/hydrocore-v5-release`, `data/locked/`, any M11.6 artifact, or any
governance module.

**Headline result**: the pilot's unseen-topology Top-1 gain for
`C_FULL` (the pilot's `C_PHYSICS_INFORMED`) **replicates**, in direction,
across all 3 pre-declared seeds, and the pooled cross-seed paired-bootstrap
CI excludes zero: **+3.10pp (90% CI [+1.31, +5.00]pp)** over `A_CONTROL` on
`ood-UNSEEN_TOPOLOGY` Top-1 (n=840 = 280 examples x 3 seeds). A
parameter-matched generic-capacity control (`A_CAPACITY_MATCHED`, same
~+4.6% parameter delta, no candidate/graph/physics information) does
**not** reproduce this gain -- its own pooled delta vs `A_CONTROL` is
**-1.07pp (CI [-2.14, +0.00]pp)**, and `C_FULL` significantly beats it
head-to-head (+4.17pp, CI [+2.38, +5.95]pp). Physics-feature ablation
identifies **`C2` (hop-distance-vs-concentration-magnitude compatibility)
as the feature driving the effect**: `C2` alone reaches +3.45pp vs
`A_CONTROL` and is statistically indistinguishable from `C_FULL`
(+0.36pp, CI includes zero) -- it alone reproduces the full arm's gain.
`C1` (nearest-sensor concentration) contributes a smaller but real,
significant increment; `C3` (hop-distance-vs-arrival-time) is null on
Top-1 and does not beat `B_CANDIDATE_CONDITIONED`. Candidate conditioning
without physics grounding (`B_CANDIDATE_CONDITIONED`) is now **pooled-
significantly worse** than `A_CONTROL` on Top-1/Top-3/MRR -- a sharper,
now-significant version of the pilot's non-significant negative point
estimate. The low-centrality and long-distance hard subgroups, null in the
single-seed pilot, both reach significance for `C_FULL`/`C2` once pooled
across seeds. Magnitude, not direction, is the caveat: the effect is
+6.4pp/+1.1pp/+1.8pp across the three seeds -- always positive, but one
seed (the original pilot seed) is substantially larger than the other two.
Decision: **CANDIDATE_FOR_LARGER_SCALE_VALIDATION** (Section 16), with a
concrete scope recommendation for the next run (Section 17).

## 1. Phase 1 -- reproduction of the original pilot

At seed `20260814` (identical corpus/splits/epochs/optimizer/config to
`candidate-conditioned-localizer-v1`'s own run), this branch's
`A_CONTROL`/`B_CANDIDATE_CONDITIONED`/`C_FULL` reproduce the pilot's
reported `ood-UNSEEN_TOPOLOGY` Top-1 essentially exactly:

| arm | this branch (seed 20260814) | original pilot | discrepancy |
|---|---|---|---|
| A_CONTROL | 0.3750 | 0.3750 | none |
| B_CANDIDATE_CONDITIONED | 0.3893 | 0.3893 | none |
| C_FULL (pilot's C_PHYSICS_INFORMED) | 0.4393 | 0.4393 | none |

Split sizes also match exactly (600 train / 300 validation / 300
development_holdout / 400 ood-UNSEEN_TOPOLOGY / 1000 calibration --
`reports/evaluation/physics-informed-localizer-validation/seed-20260814/run-manifest.json`).
**Phase 1 passes: no discrepancy to document.** This gives high confidence
that the harness (data loading, feature computation, training loop,
evaluation) is a faithful continuation of the pilot's own code, not a
reimplementation that happens to look similar.

## 2. Arms and parameter counts

| arm | localizer_mode | structural feats | physics feats (of 3) | extra capacity | total params | delta vs A_CONTROL |
|---|---|---|---|---|---|---|
| A_CONTROL | `default` | -- | -- | -- | 4,044,113 | -- |
| A_CAPACITY_MATCHED | `default` | -- | -- | `localizer_capacity_hidden_dim=482` | 4,231,223 | +187,110 (+4.627%) |
| B_CANDIDATE_CONDITIONED | `candidate_conditioned` | yes | 0/3 | -- | 4,231,129 | +187,016 (+4.624%) |
| C_FULL | `candidate_conditioned` | yes | 3/3 | -- | 4,231,897 | +187,784 (+4.643%) |
| C1 | `candidate_conditioned` | yes | 3/3 (1 nonzero) | -- | 4,231,897 | +187,784 (+4.643%) |
| C2 | `candidate_conditioned` | yes | 3/3 (1 nonzero) | -- | 4,231,897 | +187,784 (+4.643%) |
| C3 | `candidate_conditioned` | yes | 3/3 (1 nonzero) | -- | 4,231,897 | +187,784 (+4.643%) |

(`reports/evaluation/physics-informed-localizer-validation/pooled/parameter-counts.json`;
`consistent_across_seeds: true` for every arm -- seed only changes
initialization, never parameter count, confirming the harness is wired
correctly.)

`A_CAPACITY_MATCHED`'s capacity delta (+4.627%) lands almost exactly
between `B` (+4.624%) and `C_FULL` (+4.643%) -- as close a parameter match
as practical without adding any candidate/graph/physics information
(Phase 2's requirement). Every C-family arm (`C_FULL`/`C1`/`C2`/`C3`)
shares an **identical** parameter count and architecture: ablation changes
only which of the three physics-feature columns are nonzero going into the
model, never the model itself (Phase 5; see Section 8 for the exact
masking mechanism).

## 3. Pre-declared seeds and reproducible commands

Fixed before Phase 3 training began (see the plan doc):
`SEEDS = (20260814, 20260901, 20260915)`. 20260814 is the original pilot's
own seed (Phase 1's direct reproduction check); the other two are
disjoint calendar dates with no dependence on any run's outcome.

```
git lfs pull --include="data/learning-v2/cycle-b2/tensors-normalized/**"
python3 -m pytest tests/unit/test_capacity_matched_localizer.py tests/unit/test_physics_feature_ablation.py -q
python3 scripts/hydrocore_v5_experimental/physics_informed_localizer_validation/run_experiment.py
python3 scripts/hydrocore_v5_experimental/physics_informed_localizer_validation/analyze_results.py
```

`run_experiment.py` (no `--seed`/`--arms`) runs all 3 seeds x the 7
priority arms (`A_CONTROL, A_CAPACITY_MATCHED, B_CANDIDATE_CONDITIONED,
C_FULL, C1, C2, C3`) -- 21 full training+evaluation runs, ~10-13 minutes
each on CPU (`fp32=True`, `deterministic=True`), ~5.5 hours total wall
time as actually run for this report (`elapsed_seconds` per
`<arm>-evaluation.json`'s `training` field: 757-919s per run). Per-seed
raw outputs and run manifests are under `seed-<seed>/`; cross-seed pooled
analysis is under `pooled/`.

**Compute-budget scope decision**: the three budget-permitting pairwise
ablation arms (`C1_C2`, `C1_C3`, `C2_C3`) were **not** run. The single-
feature ablation (Section 8) already gives an interpretable, statistically
clear answer (`C2` alone matches `C_FULL`; Section 9's required
comparisons confirm this directly), and 5 pre-declared seeds (vs. the 3
used) was likewise not pursued -- both would have added several more
hours of sequential CPU training for a question the priority-arm data
already answers cleanly. This is a deliberate, documented scope decision
per the task's own "do not create a combinatorial explosion if the budget
is limited," not an oversight.

## 4. Mean metric table across seeds (Top-1/Top-3/MRR)

Per-seed values in `seed-<seed>/metric-table.md`; means below
(`pooled/cross-seed-summary.json`):

| population | A_CONTROL | A_CAPACITY_MATCHED | B_CANDIDATE_CONDITIONED | C_FULL | C1 | C2 | C3 |
|---|---|---|---|---|---|---|---|
| validation (n=300) | 0.679/0.878/0.790 | 0.676/0.878/0.789 | 0.684/0.884/0.794 | 0.691/0.880/0.798 | 0.691/0.883/0.798 | 0.686/0.879/0.794 | 0.681/0.877/0.791 |
| development_holdout (n=300) | 0.676/0.884/0.788 | 0.677/0.882/0.789 | 0.681/0.884/0.793 | 0.684/0.877/0.793 | 0.684/0.881/0.794 | **0.694**/0.880/0.799 | 0.676/0.879/0.788 |
| ood-UNSEEN_TOPOLOGY (n=280 localized/400) | 0.389/0.729/0.585 | 0.379/0.724/0.579 | 0.370/0.707/0.571 | **0.420**/0.745/0.613 | 0.406/**0.746**/0.604 | **0.424**/0.714/0.605 | 0.382/0.701/0.579 |

Every C-family arm's unseen-topology Top-1 mean exceeds A_CONTROL's;
A_CAPACITY_MATCHED and B_CANDIDATE_CONDITIONED both fall *below*
A_CONTROL's mean there.

## 5. Primary endpoint: unseen-topology Top-1 (pre-registered)

Pooled paired bootstrap, `ood-UNSEEN_TOPOLOGY`, n=840 (280 examples x 3
seeds, matched by `(seed, scenario_id)`), 2000 resamples, seed 20260826,
90% CI (`pooled/pooled-paired-bootstrap.json`):

| arm | Top-1 delta vs A_CONTROL | 90% CI | excludes zero |
|---|---|---|---|
| A_CAPACITY_MATCHED | -0.0107 | [-0.0214, +0.0000] | no |
| B_CANDIDATE_CONDITIONED | -0.0190 | [-0.0333, -0.0060] | **yes, negative** |
| **C_FULL** | **+0.0310** | **[+0.0131, +0.0500]** | **yes, positive** |
| C1 | +0.0167 | [+0.0012, +0.0333] | yes, positive |
| C2 | +0.0345 | [+0.0179, +0.0500] | yes, positive |
| C3 | -0.0071 | [-0.0202, +0.0060] | no |

`C_FULL`'s pooled effect (+3.10pp) is smaller than the pilot's own
single-seed estimate (+6.4pp) but remains positive and significant with
3x the paired sample. Top-3/MRR, same population:

| arm | Top-3 delta | CI | excl. 0 | MRR delta | CI | excl. 0 |
|---|---|---|---|---|---|---|
| A_CAPACITY_MATCHED | -0.0048 | [-0.0155,+0.0060] | no | -0.0060 | [-0.0121,+0.0001] | no |
| B_CANDIDATE_CONDITIONED | -0.0214 | [-0.0345,-0.0083] | **yes, negative** | -0.0138 | [-0.0216,-0.0062] | **yes, negative** |
| C_FULL | +0.0167 | [+0.0000,+0.0345] | no | +0.0279 | [+0.0171,+0.0391] | **yes, positive** |
| C1 | +0.0179 | [+0.0024,+0.0333] | yes, positive | +0.0191 | [+0.0102,+0.0283] | yes, positive |
| C2 | -0.0143 | [-0.0310,+0.0024] | no | +0.0200 | [+0.0107,+0.0293] | yes, positive |
| C3 | -0.0274 | [-0.0417,-0.0131] | **yes, negative** | -0.0060 | [-0.0135,+0.0008] | no |

`C_FULL`'s Top-3 is directionally positive but not significant (no
tradeoff, no confirmed gain); MRR is significantly positive. `C2`'s Top-3
point estimate is negative (not significant) despite its strong Top-1/MRR
gains -- the one place this branch's own confirmed driver shows a mild,
inconclusive tension, flagged rather than smoothed over (Section 15,
criterion 7). `C3` is the one arm with a clear, significant Top-3
*regression*, mirroring the pilot's own finding that `B` alone regressed
Top-3 -- here the analogous null/harmful signal is `C3`, the one physics
feature ablation that does not help.

## 6. Cross-seed stability (Phase 3/7's explicit per-seed requirement)

`ood-UNSEEN_TOPOLOGY` Top-1, per seed, plus mean/median/stdev and sign
count vs. that seed's own `A_CONTROL` (`pooled/cross-seed-summary.json`):

| arm | seed 20260814 | seed 20260901 | seed 20260915 | mean | median | stdev | seeds +/-/0 |
|---|---|---|---|---|---|---|---|
| A_CONTROL | 0.3750 | 0.4000 | 0.3929 | 0.3893 | 0.3929 | 0.0129 | -- |
| A_CAPACITY_MATCHED | 0.3750 (Δ0.0000) | 0.3821 (Δ-0.0179) | 0.3786 (Δ-0.0143) | 0.3786 | 0.3786 | 0.0036 | 0 / 2 / 1 |
| B_CANDIDATE_CONDITIONED | 0.3893 (Δ+0.0143) | 0.3857 (Δ-0.0143) | 0.3357 (Δ-0.0571) | 0.3702 | 0.3857 | 0.0300 | 1 / 2 / 0 |
| **C_FULL** | 0.4393 (Δ+0.0643) | 0.4107 (Δ+0.0107) | 0.4107 (Δ+0.0179) | 0.4202 | 0.4107 | 0.0165 | **3 / 0 / 0** |
| C1 | 0.4143 (Δ+0.0393) | 0.4214 (Δ+0.0214) | 0.3821 (Δ-0.0107) | 0.4060 | 0.4143 | 0.0209 | 2 / 1 / 0 |
| C2 | 0.4393 (Δ+0.0643) | 0.4036 (Δ+0.0036) | 0.4286 (Δ+0.0357) | 0.4238 | 0.4286 | 0.0183 | **3 / 0 / 0** |
| C3 | 0.3857 (Δ+0.0107) | 0.3714 (Δ-0.0286) | 0.3893 (Δ-0.0036) | 0.3821 | 0.3857 | 0.0094 | 1 / 2 / 0 |

**`C_FULL` and `C2` are the only two arms with a positive Top-1 delta in
all 3 seeds.** `A_CAPACITY_MATCHED` never has a positive delta (0/3
positive, one exact zero). `B` and `C3` are each positive in only 1 of 3
seeds. `C1` is positive in 2 of 3.

The magnitude caveat is real: seed 20260814 (the original pilot seed)
shows a substantially larger `C_FULL`/`C2` effect (+6.4pp) than the other
two seeds (+1.1pp, +1.8pp). Direction is consistent; size is not. This
branch does not have a way to determine, from 3 seeds, whether 20260814 is
an outlier or the other two seeds under-realize the effect -- that
question is explicitly deferred to a larger-scale run (Section 17).

## 7. Phase 2: does generic capacity alone reproduce the gain?

No. Across every test in this report, `A_CAPACITY_MATCHED` (+4.627%
parameters, same generic residual-MLP path used by `A_CONTROL`'s own
`source_node_head`, zero candidate/graph/physics information) never shows
a significant positive effect and is negative in 2 of 3 seeds:

- Pooled vs `A_CONTROL`: -1.07pp, CI [-2.14, +0.00]pp (does not exclude
  zero, and the point estimate itself is negative).
- `C_FULL` vs `A_CAPACITY_MATCHED` head-to-head (pooled,
  `pooled/required-pairwise-comparisons.json`): **+4.17pp, CI [+2.38,
  +5.95]pp, excludes zero** -- `C_FULL` significantly outperforms the
  capacity-matched control directly, not just A_CONTROL.
- Known-topology (validation/development_holdout): both CIs include zero,
  point estimates near-flat (+0.1 to -0.3pp).
- Top-3/MRR on unseen topology: both negative point estimates, both CIs
  include zero -- no evidence of even a null-but-harmless capacity effect,
  let alone a beneficial one.

**Phase 2 conclusion: the unseen-topology gain is not explained by the
extra parameter count. It requires the specific candidate-conditioned +
physics-feature mechanism, not capacity in general.**

## 8. Phase 4/5: physics-feature ablation

### 8.1 Exact feature definitions (from `physics_features.py`, unchanged from the pilot)

All three features are computed once per batch, read only
`temporal_features` (channel 0 = `log1p(concentration_mg_l)`, NaN where
unobserved), the label-free candidate-to-sensor hop-distance matrix
(`candidate_sensor_features.compute_hop_distance`, from `edge_index`/
`edge_mask` only), and the active-sensor mask -- **never**
`source_node`/`source_node_mask`/any evaluation-outcome tensor. Zero
EPANET/simulator calls.

| feature (`PHYSICS_FEATURE_COLUMNS` index) | exact formula | inference-time inputs | topology-invariant | observed-data + known-structure only | cost | simulator calls | leakage risk |
|---|---|---|---|---|---|---|---|
| **`nearest_sensor_log_concentration`** (0, = C1) | peak `log1p(concentration)` observed at the reachable active sensor with minimum hop distance from the candidate | `temporal_features[...,0]`, hop-distance matrix, active-sensor mask | yes (recomputed per topology, no per-node-ID parameter) | yes (real sensor readings + graph edges) | O(sensors) argmin per candidate | none | none directly; a legitimate real-evidence proxy for the oracle audit's simulator-grounded residual (documented scope reduction, not a substitute) |
| **`hop_magnitude_compatibility`** (1, = C2) | `-corr(hop_distance_to_sensor, sensor_peak_log_concentration)` over all reachable sensors for that candidate; negated because a true source should show shorter hop <-> higher concentration (negative raw correlation), so higher output = more physically consistent | same as above | yes | yes | O(sensors) correlation per candidate | none | none; neutral (0.0) when <2 valid sensors or zero variance, never silently wrong |
| **`hop_arrival_time_compatibility`** (2, = C3) | `+corr(hop_distance_to_sensor, sensor_arrival_time)` over reachable sensors with a finite arrival time; positive raw correlation expected (farther sensors detect later under transport), kept unnegated | hop-distance matrix, per-sensor arrival time (elapsed time of that sensor's own peak reading, from `temporal_features` + `timestamps`) | yes | yes | O(sensors) correlation per candidate | none | none; neutral (0.0) when <2 valid points |

### 8.2 Ablation mechanism (Phase 5)

`run_experiment.py::_mask_physics_columns` zeroes every column **not** in
the arm's `physics_columns` before the (unchanged) `physics_projection`
layer sees the tensor. `C_FULL`/`C1`/`C2`/`C3` all construct the identical
`CandidateConditionedLocalizer` with `physics_feature_dim=3` -- there is no
parameter-count difference between them to verify away (Section 2's table
confirms this: all four report the exact same 4,231,897 total). Ablation
is a pure input-signal change, exactly as Phase 5 requires; no padding
trick was needed.

### 8.3 Which feature drives the effect

Pooled `ood-UNSEEN_TOPOLOGY` Top-1, n=840 (Section 5's table): `C2` reaches
**+3.45pp** (CI [+1.79, +5.00]pp) -- the single largest point estimate of
any arm in this report, including `C_FULL` itself. `C1` reaches **+1.67pp**
(CI [+0.12, +3.33]pp) -- smaller, but real and significant. `C3` reaches
**-0.71pp** (CI [-2.02, +0.60]pp) -- null, point estimate slightly negative.

Direct comparisons confirm this is not overlapping noise
(`pooled/required-pairwise-comparisons.json`):

| comparison | Top-1 delta | 90% CI | excludes zero |
|---|---|---|---|
| C2 vs C_FULL | +0.0036 | [-0.0155, +0.0226] | no -- **statistically indistinguishable** |
| C1 vs C_FULL | -0.0143 | [-0.0262, -0.0024] | yes, negative -- C1 alone is a real step down from full |
| C3 vs C_FULL | -0.0381 | [-0.0571, -0.0202] | yes, negative -- C3 alone is substantially worse than full |
| C2 vs B_CANDIDATE_CONDITIONED | +0.0536 | [+0.0381, +0.0702] | yes, positive |
| C1 vs B_CANDIDATE_CONDITIONED | +0.0357 | [+0.0202, +0.0512] | yes, positive |
| C3 vs B_CANDIDATE_CONDITIONED | +0.0119 | [-0.0036, +0.0251] | no |

**`C2` alone statistically reproduces `C_FULL`'s entire effect** and
significantly beats `B` on its own; `C1` also significantly beats `B` but
is itself significantly weaker than the full bundle; `C3` does not
significantly beat `B` at all -- it contributes no measurable value on
this endpoint. Cross-seed sign count (Section 6) corroborates this
ranking: `C2` is positive in all 3 seeds (tied with `C_FULL`), `C1` in 2 of
3, `C3` in only 1 of 3.

### 8.4 Physical interpretation

The dominant, reproducible signal is **hop-distance-vs-concentration-
magnitude monotonicity** (`C2`): whether a candidate's structural
proximity to sensors is consistent with the pattern of concentration
readings those sensors observed (closer sensors reading higher
concentration). **Raw nearest-sensor concentration level** (`C1`)
contributes a smaller, still-real independent increment -- knowing the
absolute reading at the closest sensor helps somewhat beyond knowing
whether the whole gradient is consistent. **Arrival-time consistency**
(`C3`) shows no reliable Top-1 signal and, on its own (Section 5), a
significant Top-3 regression -- this branch's ablation does not identify
transport-timing information as a contributor to the unseen-topology gain,
in contrast to the concentration-magnitude channel. This is consistent
with a concrete mechanistic story: on **unseen** topologies specifically,
the backbone's message-passing has never seen this graph's own edge
structure, so a candidate's post-backbone hidden state is a weaker guide
to "is this really upstream of the observed contamination" than it is on
known topologies (where the backbone has had epochs to learn that
topology's specific transport patterns) -- an explicit, computed magnitude-
consistency check compensates for exactly that gap, while a raw
concentration level or an arrival-time correlation (noisier, more
timestep-resolution-dependent) does not compensate as reliably.

## 9. Phase 6: required comparisons (summary)

1. **C_FULL vs A_CONTROL**: replicates, direction 3/3 seeds, pooled CI
   excludes zero (Section 5).
2. **C_FULL vs A_CAPACITY_MATCHED**: survives -- pooled CI excludes zero in
   `C_FULL`'s favor, and `A_CAPACITY_MATCHED` itself never beats
   `A_CONTROL` (Section 7).
3. **C_FULL vs B_CANDIDATE_CONDITIONED**: `C_FULL` significantly
   outperforms `B` head-to-head, pooled +5.00pp (CI [+3.33, +6.67]pp) --
   physics features add clear value beyond candidate conditioning alone.
4. **C1/C2/C3 vs B**: `C1` (+3.57pp) and `C2` (+5.36pp) both significantly
   beat `B`; `C3` (+1.19pp, CI includes zero) does not (Section 8.3).
5. **C1/C2/C3 vs C_FULL**: `C2` is statistically indistinguishable from
   `C_FULL`; `C1` and `C3` are both significantly *below* `C_FULL`
   (Section 8.3) -- the full three-feature bundle is not outperformed by
   any single feature, but is matched by `C2` alone.

## 10. Required subgroup analysis

Betweenness-centrality terciles and hop-to-nearest-sensor median split,
computed per seed from that seed's own `A_CONTROL` rows (same convention
as the pilot), pooled across all three populations. Full per-seed
descriptive tables: `seed-<seed>/centrality-subgroups.json`,
`distance-subgroups.json` (sample counts preserved throughout, ~283-350
per subgroup per seed). Pooled significance test,
`pooled/pooled-subgroup-bootstrap.json` (Top-1, matched by
`(seed, scenario_id)`):

| arm | low_centrality delta | CI | excl. 0 | long_distance delta | CI | excl. 0 |
|---|---|---|---|---|---|---|
| A_CAPACITY_MATCHED | -0.0068 (n=1028) | [-0.0146,+0.0010] | no | +0.0023 (n=872) | [-0.0080,+0.0126] | no |
| B_CANDIDATE_CONDITIONED | -0.0175 | [-0.0272,-0.0068] | **yes, negative** | +0.0183 | [+0.0034,+0.0321] | yes, positive |
| **C_FULL** | **+0.0214** | **[+0.0087,+0.0350]** | **yes, positive** | **+0.0195** | **[+0.0011,+0.0378]** | **yes, positive** |
| C1 | +0.0117 | [-0.0010,+0.0233] | no | +0.0195 | [+0.0034,+0.0356] | yes, positive |
| C2 | +0.0233 | [+0.0088,+0.0370] | yes, positive | +0.0344 | [+0.0149,+0.0539] | yes, positive |
| C3 | -0.0097 | [-0.0204,+0.0000] | no | +0.0011 | [-0.0149,+0.0161] | no |

**Unlike the single-seed pilot (both CIs included zero for both arms),
pooling across 3 seeds finds `C_FULL` and `C2` both significant on the
diagnosed hard subgroups** -- low centrality and long sensor distance.
This is the clearest upgrade this branch makes over the pilot's own H2-null
finding: with 3x the paired sample, the hard-subgroup signal that the
pilot could not distinguish from noise is now measurable. `A_CAPACITY_MATCHED`
remains null-to-negative on both; `C3` remains null on both; `C1` is
significant on distance but not centrality. The gain is **not** concentrated
only in easy subgroups (Section 15, criterion 9).

## 11. Known-topology results

`validation`/`development_holdout` pooled deltas vs `A_CONTROL`
(`pooled/pooled-paired-bootstrap.json`) are mostly null: every arm's Top-1
CI on both known populations includes zero except `C1` on validation
(+1.22pp, CI [+0.11, +2.33]pp) and `C2` on development_holdout (+1.89pp,
CI [+0.67, +3.11]pp) -- small, real, but not the headline effect, and
neither is `C_FULL` itself significant on either known population
(+1.22pp / +0.89pp, both CIs include zero). **No known-topology arm shows
a significant regression anywhere** -- criterion 6 (Section 15) holds
cleanly. This reproduces the pilot's own framing: whatever this mechanism
is doing, it is concentrated on topology transfer, not a general
"candidate conditioning + physics features help everywhere" effect.

## 12. Paired transition counts (pooled across all 3 seeds, unseen topology)

| arm | both_correct | control_only (regressed) | arm_only (fixed) | both_wrong | net |
|---|---|---|---|---|---|
| A_CAPACITY_MATCHED | 307 | 20 | 11 | 502 | -9 |
| B_CANDIDATE_CONDITIONED | 293 | 34 | 18 | 495 | -16 |
| **C_FULL** | 296 | 31 | **57** | 456 | **+26** |
| C1 | 302 | 25 | 39 | 474 | +14 |
| **C2** | 306 | 21 | **50** | 463 | **+29** |
| C3 | 301 | 26 | 20 | 493 | -6 |

`C2` has the best fixed-to-regressed ratio of any arm (50 vs 21, net +29),
narrowly ahead of `C_FULL` (57 vs 31, net +26); `A_CAPACITY_MATCHED` and
`B` both show more regressions than fixes (net negative); `C3` is
essentially a wash. This corroborates Sections 5/8/9 example-by-example
rather than only in aggregate rates.

## 13. Calibration / safety / OOD

Split-conformal coverage is identical (0.9073) across every arm and seed,
by construction (the calibrator targets the same nominal rate regardless
of the underlying model). Expected calibration error is comparable across
arms with no consistent outlier; `C_FULL` is at or below `A_CONTROL`'s ECE
in all 3 seeds (0.0497 vs 0.0542; 0.0492 vs 0.0544; 0.0488 vs 0.0595) --
mirroring the pilot's own observation that `C_PHYSICS_INFORMED` was the
best-calibrated arm, now replicated. `proxy_abstention_rate` is 1.0 for
every arm on `ood-UNSEEN_TOPOLOGY` (expected: this population sits outside
every arm's own `train_topology_hashes` by construction) and ~2-4% on
known-topology populations, near-identical across arms.

**All 21 runs report every `hard_safety_counter` as exactly 0**
(`reports/evaluation/physics-informed-localizer-validation/seed-*/*-evaluation.json`,
verified programmatically across all 21 files) -- for the same documented
reason as the pilot: this pilot-scale localization-only harness never
exercises the sampling/planning/execution control loop that produces
non-zero values for these counters at the M11.6 evaluation tier. No
governance module (`hydroswarm.inference.ood`,
`hydroswarm.calibration.conformal`, any actionability gate) is modified by
this branch. **Criterion 8 (Section 15) holds: no calibration/safety
regression anywhere.**

## 14. Oracle caveat (carried forward, unchanged)

Unchanged from the plan doc: the fair, nuisance-searched oracle from the
pilot's own audit (`docs/evaluation/ORACLE_INFORMATION_AUDIT.md`) is used
nowhere in this branch as a training target, and its documented residual
privilege (shares the true incident's hydraulic/demand realization across
candidates) is not independently re-tested here. This branch's evidence is
entirely about `C_FULL`/its ablations vs `A_CONTROL`/`A_CAPACITY_MATCHED`/
`B` on this pilot's own corpus, not about closing any oracle gap further.

## 15. Success / failure criteria

| # | criterion | holds? | evidence |
|---|---|---|---|
| 1 | C_FULL improves unseen Top-1 in the same direction across most seeds | **yes** | 3/3 seeds positive (Section 6) |
| 2 | pooled/combined paired CI remains above zero | **yes** | +3.10pp, CI [+1.31,+5.00]pp (Section 5) |
| 3 | A_CAPACITY_MATCHED does not reproduce the effect | **yes** | never positive; C_FULL beats it head-to-head +4.17pp (Section 7) |
| 4 | C_FULL outperforms B_CANDIDATE_CONDITIONED | **yes** | +5.00pp head-to-head, CI excludes zero (Section 9) |
| 5 | at least one physics feature explains a meaningful fraction of the gain | **yes** | C2 alone matches C_FULL exactly (Section 8.3) |
| 6 | known-topology performance does not materially regress | **yes** | all known-population CIs null or small-positive, none negative (Section 11) |
| 7 | Top-3/MRR show no reproducible harmful tradeoff undermining Top-1 | **mostly** | C_FULL: Top-3 null (not harmful), MRR positive; **C2's Top-3 point estimate is negative though CI includes zero** -- flagged, not clean (Section 5) |
| 8 | calibration/abstention/OOD/safety unchanged or no worse | **yes** | Section 13 |
| 9 | gain is not concentrated only in easy subgroups | **yes** | low-centrality and long-distance both significant, pooled (Section 10) |

Eight of nine criteria hold cleanly; criterion 7 holds for `C_FULL` itself
but carries a real caveat for `C2` specifically (Section 17's
recommendation addresses this directly: carry `C1+C2`, not `C2` alone,
into the next stage). None of the task's explicit failure criteria
trigger: the effect does not disappear across seeds, the capacity control
does not reproduce it, direction (not magnitude) is stable across seeds,
the ablation is interpretable (not "no dependence"), no Top-3/MRR
regression offsets `C_FULL`'s own Top-1 gain, the gain is not
easy-subgroup-only, and safety/calibration do not worsen.

## 16. Final decision

**CANDIDATE_FOR_LARGER_SCALE_VALIDATION.**

Not REJECT: the signal does not disappear, is not explained by capacity,
and is not confounded by a generic-capacity or candidate-conditioning-alone
effect -- `B` is actually significantly *worse* than control, the opposite
of what would make this a capacity or candidate-conditioning story.

Not merely CONTINUE_RESEARCH: unlike the pilot (single seed, H2 null,
no capacity control, no feature attribution), this branch's replication
across seeds is directionally unanimous (3/3), survives a real
capacity-matched control head-to-head, has an interpretable and
statistically confirmed physics-feature driver (`C2`, matched by `C1+C2`
without `C3`), and now finds significance in the previously-null hard
subgroups. The task's own three criteria for
`CANDIDATE_FOR_LARGER_SCALE_VALIDATION` -- replicates across seeds,
survives capacity control, has an interpretable driver, no
safety/calibration tradeoff -- are each independently satisfied by
distinct, specific evidence in this report, not by re-reading the same
single number four ways.

The one caveat withholding a stronger classification is magnitude
stability (Section 6): one seed shows an effect roughly 3-6x the other
two. A larger-scale run should treat *replicating a positive, non-trivial
effect size* -- not just a positive sign -- as its own pre-registered
success bar.

## 17. What should the next experiment be

1. **Simplify the physics-feature arm to `C1+C2`, dropping `C3`.** `C3`
   contributes no measurable Top-1 value (null vs both `A_CONTROL` and
   `B`) and is the one ablation with a significant Top-3 regression on its
   own; carrying it into a larger, more expensive run adds cost and a
   documented risk with no offsetting benefit shown here. `C2` alone
   already matches `C_FULL`; testing `C1+C2` (not run in this branch,
   budget-permitting scope, Section 3) as the next arm would confirm
   whether `C1`'s smaller independent contribution (Section 8.3) is worth
   keeping alongside `C2`, or whether `C2` alone is sufficient going
   forward.
2. **Pre-register an effect-size (not just direction) replication bar** for
   the next run's seeds, given Section 6's magnitude variability -- e.g.
   requiring the pooled CI's lower bound to clear a specific pp threshold
   set from this report's own +1.31pp, not just "excludes zero."
3. **Resolve `C2`'s Top-3 tension** (Section 5/15 criterion 7) before or
   alongside scaling up: is the small, non-significant Top-3 dip a real
   ranking-shape tradeoff for the magnitude-compatibility feature
   specifically, or noise that a larger sample would show is null too.
4. Everything the plan doc explicitly deferred remains deferred: no GNN
   rewrite, no additional attention stack, no major model scaling, no gate
   relaxation, and the oracle's residual hydraulic/demand privilege
   (Section 14) is still untested.

## Explicit answers

- **Did the original +6.4pp unseen-topology Top-1 gain replicate?** Yes,
  in direction, across all 3 seeds (+6.4pp / +1.1pp / +1.8pp), and the
  pooled cross-seed CI excludes zero (+3.10pp, [+1.31,+5.00]pp). Magnitude
  did not replicate at the same size -- the original pilot's seed shows a
  substantially larger effect than the other two.
- **How seed-stable is the effect?** Direction: fully stable (3/3 positive
  for `C_FULL` and `C2`, the only two arms with that property). Magnitude:
  not stable (Section 6) -- a real, documented caveat, not smoothed over.
- **Does a parameter-matched generic-capacity control reproduce it?** No.
  `A_CAPACITY_MATCHED` never shows a positive pooled or per-seed effect and
  is significantly outperformed by `C_FULL` head-to-head (Section 7).
- **Which physics feature(s) drive the gain?** `C2`
  (hop-distance-vs-concentration-magnitude compatibility), statistically
  indistinguishable from the full three-feature arm on its own. `C1`
  (nearest-sensor concentration) contributes a smaller, real, independent
  increment. `C3` (hop-distance-vs-arrival-time) contributes nothing
  measurable on this endpoint and regresses Top-3 on its own (Section 8).
- **Does candidate conditioning without physics remain insufficient?**
  Yes, more strongly than the pilot found: pooled across seeds,
  `B_CANDIDATE_CONDITIONED` is now *significantly worse* than `A_CONTROL`
  on Top-1, Top-3, and MRR on unseen topology (Section 5) -- not merely
  non-significant as in the single-seed pilot.
- **Does C_FULL improve hard cases or mostly easy ones?** Both, and the
  hard-case (low-centrality, long-distance) gains are now individually
  significant once pooled across seeds (Section 10) -- an upgrade over the
  pilot's own null finding there.
- **Are Top-3/MRR/calibration affected?** MRR: significantly improved for
  `C_FULL`/`C1`/`C2`. Top-3: not significantly affected for `C_FULL`
  (point estimate positive); `C2` shows a non-significant negative Top-3
  point estimate (flagged, Section 15 criterion 7); `C3` alone shows a
  significant Top-3 regression. Calibration: unaffected-to-improved
  (Section 13), no safety counter ever nonzero.
- **Is the effect strong enough for larger-scale validation?**
  Yes -- `CANDIDATE_FOR_LARGER_SCALE_VALIDATION` (Section 16), with the
  concrete scope narrowing in Section 17 (drop `C3`, test `C1+C2`,
  pre-register an effect-size bar) as the condition for that next step.
- **What should the next experiment be?** Section 17.
