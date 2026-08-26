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

<!-- PLACEHOLDER: filled once scripts/hydrocore_v5_experimental/candidate_conditioned_localizer_v1/run_experiment.py + analyze_results.py complete -->

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
read with this caveat (Section 14).

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

<!-- FILLED FROM metric-table.md -->

## 7. Unseen-topology results

<!-- ood-UNSEEN_TOPOLOGY population from metric-table -->

## 8. Centrality subgroup results

<!-- centrality-subgroups.json -->

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

<!-- distance-subgroups.json -->

## 11. HydroCore-v5 M11.6 failure-subset recovery analysis

**Out of scope for direct measurement on this pilot.** This pilot trains
fresh `small`-variant models on `data/learning-v2/cycle-b2`, not the
frozen `models/hydrocore-v5-release` checkpoint, and does not re-run the
M11.6 locked evaluation (Section 0/plan doc Section 9 explain why: doing so
would require full-scale training matching M11.6's own frozen conditions,
far beyond a controlled pilot). This report therefore cannot state whether
Arm B/C would have flipped any of the specific 56 M11.6 confirmatory
failures. What Section 12 (paired transitions) and Sections 8/10 (centrality/
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

<!-- FILLED FROM oracle-gap-closed.json -->

## 13. Paired transition tables

<!-- FILLED FROM paired-transitions.json -->

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

<!-- proxy_calibrated_coverage / ood_caution_or_outside_rate / hard_safety_counters from *-evaluation.json -->

All `hard_safety_counters` report 0 for every arm -- this pilot-scale
localization-only harness never exercises the sampling/planning/execution
control loop that produces non-zero values for these counters at the M11.6
evaluation tier (same documented caveat `exp/graph-structural-encoder-v2`
uses; see each `<arm>-evaluation.json`'s `hard_safety_counters_note`). No
governance module (`hydroswarm.inference.ood`,
`hydroswarm.calibration.conformal`, any actionability gate) was modified by
this branch.

## 16. Negative / null results

<!-- filled once metric table lands -->

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

## 18. Recommendation

<!-- filled once results land: REJECT / CONTINUE_RESEARCH / CANDIDATE_FOR_LARGER_SCALE_VALIDATION / CANDIDATE_FOR_FUTURE_PROMOTION -->

## Explicit answers

- **Was the 96.4% oracle recovery result a fair same-information
  comparison?** No -- it was a PRIVILEGED ORACLE (Section 1). Corrected: the
  fair, nuisance-searched oracle reproduces the identical 96.4% figure on
  the identical failing incidents, so the finding is robust to the
  correction even though the original construction was privileged.
- **Does candidate-conditioned scoring outperform HydroCore-v5?** <!-- filled -->
- **Does it recover errors that are physically identifiable?** <!-- filled -->
- **Does it improve unseen-topology generalization?** <!-- filled -->
- **How much of the HydroCore-to-oracle gap is closed?** <!-- filled -->
- **Do physics-informed residual/compatibility features add independent
  value?** <!-- filled -->
- **Does graph-native message passing add independent value after
  candidate conditioning?** Not tested (Arm D not attempted, Section 3).
- **Is a full GNN still justified?** <!-- filled -->
- **Should the next effort focus on model architecture, calibration, or
  adaptive sampling?** <!-- filled -->
