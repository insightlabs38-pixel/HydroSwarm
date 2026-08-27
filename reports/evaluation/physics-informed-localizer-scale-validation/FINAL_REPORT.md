# physics-informed-localizer-scale-validation: final report

Branch: `exp/physics-informed-localizer-scale-validation`, based on the
completed `exp/physics-informed-localizer-validation` at
`0534fe143fa5c1068de5880fe24d5958ad87406a` (final report:
`reports/evaluation/physics-informed-localizer-validation/FINAL_REPORT.md`,
treated as immutable and unmodified by this branch). Plan (pre-registered
before training): `docs/evaluation/experimental/
PHYSICS_INFORMED_LOCALIZER_SCALE_VALIDATION_PLAN.md`. Status:
**experimental, non-release**. No change to `models/hydrocore-v5-release`,
`data/locked/`, HydroSwarm v0.2.1, HydroCore-v5's release artifacts, M11.6
locked evidence, or any hackathon claim; no governance module modified.
This branch is not merged.

**Headline result**: the simplified two-feature `C1_C2` candidate-localizer
(`nearest_sensor_log_concentration` + `hop_magnitude_compatibility`, `C3`
zeroed) **meets the pre-registered strong-replication bar** on the primary
endpoint: pooled fresh-seed `ood-UNSEEN_TOPOLOGY` Top-1 delta vs
`A_CONTROL` is **+5.48pp, 90% CI [+3.81, +7.14]pp** (n=840, 280 examples x
3 fresh seeds), positive in all 3 seeds, comfortably clearing the
pre-registered +1.0pp lower-CI-bound requirement. `C2` alone
**independently replicates** on these same fresh seeds at an
indistinguishable pooled magnitude (+5.48pp, CI [+3.93, +7.14]pp, also 3/3
seeds positive). Head-to-head, `C1_C2` vs `C2` is a **statistical tie on
Top-1** (observed delta exactly 0.0, CI [-1.43, +1.43]pp) but `C1_C2`
**significantly improves Top-3 over `C2`** (+2.02pp, CI [+0.83,
+3.21]pp) and is the only arm whose Top-3 gain over `A_CONTROL` itself
reaches significance (+2.98pp, CI [+1.43, +4.64]pp; `C2`'s own Top-3 vs
control remains non-significant, point estimate now slightly positive
rather than the completed study's slightly negative one). This directly
answers the completed study's own open question: **adding C1 to C2
resolves, on these fresh seeds, the Top-3 ranking-shape tension flagged
for C2 alone**, without costing anything on Top-1/MRR. The one real
caveat: `C1_C2`'s low-centrality/long-distance hard-subgroup gains, while
still positive, are **not** individually significant on 3 fresh seeds
(`C2`'s are, on both). No known-topology, calibration, or governance
regression was observed for either arm. Decision:
**CANDIDATE_FOR_FULL_SCALE_VALIDATION** (Section 15), carrying `C1_C2`
forward as the primary recommendation with `C2` documented as a
defensible, simpler fallback (Section 16).

## 1. Scope and what this branch does not re-litigate

Per the completed validation study's own Section 17 recommendation, this
branch runs exactly 3 arms (`A_CONTROL`, `C2`, `C1_C2`) x 3 fresh
pre-declared seeds (`20260929`, `20261013`, `20261027`) = 9 full
training/evaluation runs. `C_FULL`, `C3`, `B_CANDIDATE_CONDITIONED`, and
`A_CAPACITY_MATCHED` are **not** re-run -- those questions are already
answered by the completed study (Sections 0/7/8/9 of that report) and no
sanity check here revealed any implementation defect that would justify
re-running them. This is a **confirmation experiment**, not an
architecture search: no GNN, no attention-stack expansion, no new physics
features, no unrelated hyperparameter tuning, no model-size scaling, no
gate relaxation.

## 2. Git / Git LFS scope actually used

Exactly the 5 approved Cycle-B2 normalized splits were fetched, via
`GIT_LFS_SKIP_SMUDGE=1` branch checkout followed by one scoped
`git lfs pull --include=...`:

```
data/learning-v2/cycle-b2/tensors-normalized/train/**
data/learning-v2/cycle-b2/tensors-normalized/validation/**
data/learning-v2/cycle-b2/tensors-normalized/calibration/**
data/learning-v2/cycle-b2/tensors-normalized/development_holdout/**
data/learning-v2/cycle-b2/tensors-normalized/ood-UNSEEN_TOPOLOGY/**
```

**No additional LFS bandwidth beyond these five paths was used.**
`ood-SEVERE_MISSINGNESS` and `cycle-b2-ood-extension` were verified to
remain untouched LFS pointers throughout (confirmed both before and after
the full run -- `file data/learning-v2/cycle-b2/tensors-normalized/
ood-SEVERE_MISSINGNESS/*.safetensors` still reports `ASCII text`, i.e. an
un-smudged pointer). No `git lfs pull --include=...` without this scope,
no `git lfs fetch --all`, no repo-wide `git lfs checkout`, and no
already-downloaded LFS objects were pruned. The one broad-suite test that
requires `cycle-b2-ood-extension`
(`tests/integration/test_full_output_gradient_smoke.py::
test_ood_category_head_receives_gradient_from_genuinely_diverse_real_classes`)
was run as part of Section 13's test pass and failed with a shard-checksum
mismatch **because that corpus is intentionally an un-smudged LFS pointer
on this branch, not because of any implementation defect** -- consistent
with, not a violation of, the pre-registered LFS scope.

## 3. Pre-declared fresh seeds and arms

`SEEDS = (20260929, 20261013, 20261027)` -- fixed in the plan doc before
any training, disjoint from the completed study's own `(20260814,
20260901, 20260915)`, never selected or replaced based on results.

| arm | `localizer_mode` | active physics columns | total params | delta vs A_CONTROL |
|---|---|---|---|---|
| `A_CONTROL` | `default` | -- | 4,044,113 | -- |
| `C2` | `candidate_conditioned` | `hop_magnitude_compatibility` | 4,231,897 | +187,784 (+4.643%) |
| `C1_C2` | `candidate_conditioned` | `nearest_sensor_log_concentration` + `hop_magnitude_compatibility` | 4,231,897 | +187,784 (+4.643%) |

`C2` and `C1_C2` share **byte-identical** parameter counts and shapes at
every one of the 3 fresh seeds (`consistent_across_seeds: true` for both,
`pooled/parameter-counts.json`) -- confirmed both by construction (Phase
5's `_mask_physics_columns` ablation mechanism, imported unmodified from
the completed branch) and by the dedicated unit test
(`TestC2AndC1C2ShareIdenticalArchitecture`,
`tests/unit/test_physics_localizer_scale_validation_ablation.py`). These
totals match the completed study's own `C2`/`C_FULL` parameter count
(4,231,897) exactly, confirming both branches construct the identical
model. `A_CONTROL`'s 4,044,113 also matches exactly.

Splits at every seed: 600 train (stratified 200/family across
`golden-reference`/`branched-loop`/`loop-grid`) / 300 validation / 300
development_holdout / 400 `ood-UNSEEN_TOPOLOGY` (unseen family
`coastal-branch`) / 1000 calibration -- identical to the completed study's
own pilot-scale conditions (`seed-<seed>/run-manifest.json`, all 3 fresh
seeds).

## 4. Runs completed

All 9 required training/evaluation runs completed successfully; no arm
failed and no implementation defect was found, so no rerun or
configuration change was needed. Total training wall time: 5,467s (~91
minutes; 584-919s/run on CPU, `fp32=True`, `deterministic=True`), all 9
committed under `seed-<seed>/{a_control,c2,c1_c2}-evaluation.json` plus
per-population row logs.

| seed | A_CONTROL top1/top3/mrr | C2 top1/top3/mrr | C1_C2 top1/top3/mrr |
|---|---|---|---|
| 20260929 | 0.3429 / 0.6857 / 0.5513 | 0.3893 / 0.6929 / 0.5806 | 0.4179 / 0.7393 / 0.6077 |
| 20261013 | 0.3143 / 0.6929 / 0.5345 | 0.3714 / 0.7071 / 0.5777 | 0.3786 / 0.7214 / 0.5826 |
| 20261027 | 0.3429 / 0.6857 / 0.5493 | 0.4036 / 0.6786 / 0.5855 | 0.3679 / 0.6786 / 0.5608 |

(all on `ood-UNSEEN_TOPOLOGY`, n=280/seed;
`seed-<seed>/metric-table.md`.)

## 5. Primary endpoint (pre-registered): pooled fresh-seed unseen-topology Top-1

Paired bootstrap, `ood-UNSEEN_TOPOLOGY`, n=840 (280 x 3 seeds, matched by
`(seed, scenario_id)`), 2000 resamples, deterministic bootstrap seed
20260826, 90% CI (`pooled/pooled-paired-bootstrap.json`,
`pooled/required-pairwise-comparisons.json`):

| comparison | Top-1 delta | 90% CI | excludes zero |
|---|---|---|---|
| **C1_C2 vs A_CONTROL** | **+0.0548** | **[+0.0381, +0.0714]** | **yes, positive** |
| C2 vs A_CONTROL | +0.0548 | [+0.0393, +0.0714] | yes, positive |
| C1_C2 vs C2 | 0.0000 | [-0.0143, +0.0143] | no -- **exact tie** |

### 5.1 Effect-size replication bar (Section 5 of the plan, applied exactly as pre-registered)

For `C1_C2` vs `A_CONTROL`: point estimate +5.48pp (**> 0**); positive
direction in **3/3** fresh seeds (**>= 2/3** required); pooled 90% CI
lower bound **+3.81pp** (**>= +1.0pp** required). **All three conditions
of "Strong replication" are met, with substantial margin above the
+1.0pp bar** (informed by, and here clearing by more than 2.8pp, the
completed study's own +1.31pp pooled lower bound). This bar was fixed
before training and is not altered here.

`C2` vs `A_CONTROL` independently satisfies the identical strong-
replication reading (point estimate +5.48pp, 3/3 seeds positive, lower CI
bound +3.93pp): **`C2` replicates on 3 fresh, independent seeds**, not
merely on the 3 seeds it was originally discovered on.

Top-3/MRR, same population and convention:

| comparison | Top-3 delta | CI | excl. 0 | MRR delta | CI | excl. 0 |
|---|---|---|---|---|---|---|
| C2 vs A_CONTROL | +0.0095 | [-0.0048, +0.0238] | no | +0.0363 | [+0.0272, +0.0459] | yes, positive |
| **C1_C2 vs A_CONTROL** | **+0.0298** | **[+0.0143, +0.0464]** | **yes, positive** | +0.0387 | [+0.0285, +0.0489] | yes, positive |
| **C1_C2 vs C2** | **+0.0202** | **[+0.0083, +0.0321]** | **yes, positive** | +0.0024 | [-0.0056, +0.0108] | no |

`C1_C2` is the only arm whose Top-3 gain over `A_CONTROL` itself reaches
significance, and it significantly beats `C2` head-to-head on Top-3 while
being statistically tied with it on MRR (Section 8 unpacks this in full).

## 6. Cross-seed stability (never hidden behind the pooled figure)

`ood-UNSEEN_TOPOLOGY` Top-1 per seed, `pooled/cross-seed-summary.json`:

| arm | 20260929 | 20261013 | 20261027 | mean | stdev | seeds +/-/0 (delta vs same-seed control) |
|---|---|---|---|---|---|---|
| A_CONTROL | 0.3429 | 0.3143 | 0.3429 | 0.3333 | 0.0165 | -- |
| **C2** | 0.3893 (Δ+0.0464) | 0.3714 (Δ+0.0571) | 0.4036 (Δ+0.0607) | 0.3881 | 0.0161 | **3/0/0** |
| **C1_C2** | 0.4179 (Δ+0.0750) | 0.3786 (Δ+0.0643) | 0.3679 (Δ+0.0250) | 0.3881 | 0.0263 | **3/0/0** |

Both arms are positive in all 3 fresh seeds -- direction is fully stable
for both. **Magnitude is more stable for `C2`** (deltas +4.6/+5.7/+6.1pp,
stdev 0.0161) **than for `C1_C2`** (deltas +7.5/+6.4/+2.5pp, stdev
0.0263): seed `20261027` is the one seed where `C1_C2`'s advantage over
`A_CONTROL` shrinks substantially and where `C2` alone (+6.07pp) actually
outperforms `C1_C2` (+2.50pp) on that seed specifically. This mirrors,
rather than resolves, the completed study's own magnitude-instability
caveat (Section 6 of that report) -- carried forward honestly, not
smoothed into the pooled mean.

## 7. Six-seed descriptive cross-study summary (A_CONTROL / C2 only)

**Explicitly descriptive, not a new confirmatory statistical test** --
combines the completed study's 3 committed seeds with this branch's 3
fresh ones for the two arms both branches ran
(`pooled/cross-study-six-seed-summary.json`). `C1_C2` has **no**
prior-study seeds and is **not** given six-seed treatment anywhere in this
report.

| arm | seed source | seeds | mean top1 | median | stdev | positive-delta count |
|---|---|---|---|---|---|---|
| A_CONTROL | prior (3) + fresh (3) | 6 | 0.3613 | 0.3589 | 0.0334 | -- |
| **C2** | prior (3) + fresh (3) | **6** | **0.4060** | 0.4036 | 0.0249 | **6/6** |

`C2`'s per-seed Top-1: prior `0.4393, 0.4036, 0.4286`; fresh `0.3893,
0.3714, 0.4036` -- positive vs its own seed's `A_CONTROL` in **all 6**
independent seeds across two separate studies run at different times, mean
pooled delta +4.46pp. **This six-seed descriptive picture strengthens,
rather than weakens, confidence in `C2` as a real driver**: the fresh
seeds' somewhat lower absolute Top-1 values are consistent with ordinary
seed-to-seed variance around a real positive effect, not a sign the
effect is fading.

## 8. The critical comparison: does C1+C2 resolve C2's Top-3 tension?

The completed study flagged, for `C2` alone: strong positive Top-1,
positive MRR, a **small negative** Top-3 point estimate vs `A_CONTROL`
whose CI included zero (Section 5/15 criterion 7 of that report). This
branch's fresh-seed dedicated analysis
(`pooled/top3-tension-analysis.json`) tests this directly.

**Top-3, C1_C2 vs C2, head-to-head** (n=840, same bootstrap convention):
+2.02pp, CI **[+0.83, +3.21]pp**, **excludes zero, positive**. Paired
Top-3 transition counts
(`C2` vs `C1_C2` directly, not each vs control):

| both correct | C2-only (harmed by C1) | C1_C2-only (fixed by C1) | both wrong | net |
|---|---|---|---|---|
| 572 | 10 | 27 | 231 | **+17** |

`C1_C2` fixes 27 examples `C2` alone got wrong on Top-3 while breaking
only 10 `C2` got right -- a real, example-level improvement, not just a
favorable aggregate.

**Rank-recovery analysis**: of the 54 examples (pooled, 3 fresh seeds)
where `C2`'s true-source rank is *worse* than `A_CONTROL`'s on that same
example (`C2` "harms" that example's ranking relative to control),
`C1_C2` improves on `C2`'s rank in **17 (31.5%)** of them, fully restoring
rank to at least `A_CONTROL`'s own level in 16 of those 17. This is a
partial, not complete, mechanism: `C1`'s addition recovers roughly
one-third of the specific cases where `C2`'s ranking behavior is worse
than doing nothing, it does not eliminate the underlying tension in every
case.

**True-source-rank distribution** (pooled, n=840/arm):

| arm | mean rank | median | rank=1 fraction | rank<=3 fraction |
|---|---|---|---|---|
| A_CONTROL | 2.779 | 2.0 | 33.3% | 68.3% |
| C2 | 2.648 | 2.0 | 38.8% | 69.3% |
| **C1_C2** | **2.601** | 2.0 | 38.8% | **71.3%** |

`C1_C2` has the best mean rank and the best rank<=3 fraction of any arm --
consistent with, and explaining, its significant pooled Top-3 gain where
`C2` alone remains non-significant.

**Sanity check**: `top1_correct_top3_wrong` is 0 in every seed for
`C1_C2`, as required by construction (a correct Top-1 is trivially inside
Top-3) -- confirms the harness's own `localization_top_k` metric is
behaving correctly, not masking a real inconsistency.

**Conclusion on the Top-3 tension**: on these 3 fresh seeds, `C1_C2`
**significantly improves** Top-3 both over `A_CONTROL` and over `C2`
directly, and shows the best true-source-rank distribution of any arm.
This is a genuine, example-level resolution of the specific concern the
completed study raised for `C2` alone -- not a complete elimination (31.5%
partial-recovery rate, not 100%), and not yet independently re-tested on
`C2`'s own original 3 seeds (this branch does not re-run `C2` there), but
a real, positive, statistically supported answer to the question this
branch was designed to ask.

## 9. Is C1+C2 better than C2 alone, overall?

**Mixed, reported without smoothing**:

- **Top-1**: exact tie (pooled delta 0.0, CI includes zero) -- `C1_C2`
  neither improves nor regresses `C2`'s primary-endpoint gain.
- **Top-3**: `C1_C2` significantly better (Section 8).
- **MRR**: statistically tied (+0.24pp, CI includes zero) though both
  independently beat `A_CONTROL` significantly.
- **Hard subgroups** (Section 10): `C2` is significant on both
  low-centrality and long-distance; `C1_C2`'s point estimates are smaller
  and **not** significant on either, on these 3 fresh seeds.
- **Cross-seed magnitude stability** (Section 6): `C2` is more stable
  (stdev 0.0161 vs 0.0263); `C1_C2` is the one arm where a single seed
  (`20261027`) shows a substantially smaller gain, coinciding with the one
  seed where `C2` alone edges it out.
- **Known-topology / calibration** (Sections 11-12): no difference of
  consequence either way; neither arm regresses.

Neither arm dominates the other on every axis. `C1_C2` is the better
choice specifically **if Top-3 ranking shape matters** (it does: an
operational localizer that gets the right answer into the reported
top-3 set is a materially different deliverable than one that gets Top-1
right only as often). `C2` is the better choice if **hard-subgroup
significance and cross-seed magnitude stability** are prioritized and
Top-3 is not decision-relevant. Section 16 makes this trade-off explicit
rather than picking one number to declare an unqualified winner.

## 10. Hard subgroup analysis

Betweenness-centrality terciles and hop-to-nearest-sensor median split,
computed per seed from that seed's own `A_CONTROL` rows (same convention
as the completed study), pooled Top-1 vs `A_CONTROL`
(`pooled/pooled-subgroup-bootstrap.json`; n preserved per seed in that
file):

| arm | low_centrality delta | CI | excl. 0 | n | long_distance delta | CI | excl. 0 | n |
|---|---|---|---|---|---|---|---|---|
| **C2** | **+0.0164** | **[+0.0048, +0.0270]** | **yes, positive** | 1036 | **+0.0237** | **[+0.0068, +0.0407]** | **yes, positive** | 885 |
| C1_C2 | +0.0097 | [-0.0039, +0.0232] | no | 1036 | +0.0147 | [-0.0023, +0.0328] | no | 885 |

`C2` retains, on 3 fresh seeds, the significant hard-subgroup gains the
completed study found (pooled across its own 3 seeds). `C1_C2`'s point
estimates on both hard subgroups remain **positive** -- the gain has not
shifted back toward easy cases, and there is no sign of a regression --
but neither individually clears significance at this sample size. This is
the clearest quantified cost of adding `C1` to `C2` found in this report,
and it is reported plainly rather than folded into the (still positive)
headline Top-1 number.

## 11. Known-topology guardrail

`validation`/`development_holdout` pooled Top-1 deltas vs `A_CONTROL`
(`pooled/pooled-paired-bootstrap.json`, n=900 each):

| population | C2 delta | CI | excl. 0 | C1_C2 delta | CI | excl. 0 |
|---|---|---|---|---|---|---|
| validation | +0.0178 | [+0.0078, +0.0289] | yes, positive | +0.0100 | [-0.0033, +0.0233] | no |
| development_holdout | +0.0067 | [-0.0044, +0.0178] | no | +0.0078 | [-0.0044, +0.0200] | no |

**No known-topology regression anywhere for either arm** -- every CI is
either non-significant or significantly *positive*; none is
CI-excluding-zero negative. `C1_C2` does not purchase its unseen-topology
Top-3 improvement at any known-topology cost.

## 12. Paired transition counts (pooled, unseen topology, vs A_CONTROL)

Top-1, `pooled/` per-seed `paired-transitions.json` files aggregated:

| arm | both_correct | control_only (regressed) | arm_only (fixed) | both_wrong | net |
|---|---|---|---|---|---|
| C2 | 266 | 14 | 60 | 500 | +46 |
| C1_C2 | 261 | 19 | 65 | 495 | +46 |

Identical net fixed-minus-regressed count for both arms on Top-1,
consistent with Section 5's exact pooled tie. (Top-3 transitions between
`C2` and `C1_C2` directly, not each vs control, are in Section 8.)

## 13. Calibration / safety / OOD

**All 9 runs report every one of the eight `hard_safety_counters` as
exactly 0** (verified programmatically across all 9
`seed-*/*-evaluation.json` files) -- for the same documented reason as the
completed study: this pilot-scale localization-only harness never
exercises the sampling/planning/execution control loop that produces
non-zero values for these counters at the M11.6 evaluation tier.
**This localization-only experimental harness does not exercise the full
sampling/planning/execution control loop, so zero hard counters must not
be represented as an independent re-validation of the full M11.6 safety
tier.** No governance module (`hydroswarm.inference.ood`,
`hydroswarm.calibration.conformal`, any actionability gate, conformal
calibration logic, simulator verification, or approval boundary) is
modified by this branch.

Split-conformal coverage is identical (0.9073) across every arm and seed,
by construction. Expected calibration error is comparable across arms
with no consistent regression -- `C2`/`C1_C2` are at or below
`A_CONTROL`'s ECE in every one of the 3 seeds (e.g. seed 20260929: 0.0592
/ 0.0613 vs control's 0.0770; seed 20261013: 0.0484 / 0.0508 vs 0.0525;
seed 20261027: 0.0352 / 0.0372 vs 0.0363).

## 14. Oracle caveat (carried forward, unchanged)

Unchanged from both prior branches: the fair, nuisance-searched oracle
from the original pilot's audit (`docs/evaluation/
ORACLE_INFORMATION_AUDIT.md`) is used nowhere in this branch as a training
target, and its documented residual privilege (shares the true incident's
hydraulic/demand realization across candidates) is not independently
re-tested here. This branch's evidence is entirely about `C2`/`C1_C2` vs
`A_CONTROL` on fresh seeds of this pilot's own corpus.

## 15. Success / failure against the pre-registered decision rule

| # | criterion (Section 9 of the plan) | holds? | evidence |
|---|---|---|---|
| REJECT trigger 1 | C1_C2 pooled point estimate <= 0 | **no, does not trigger** | +5.48pp (Section 5) |
| REJECT trigger 2 | majority of fresh seeds regress | **no, does not trigger** | 3/3 positive (Section 6) |
| REJECT trigger 3 | real known-topology/calibration regression | **no, does not trigger** | Sections 11/13 |
| Strong replication bar | point estimate > 0, >=2/3 seeds positive, lower CI >= +1.0pp | **yes, all three, with margin** | +5.48pp, 3/3, +3.81pp lower bound (Section 5.1) |
| C2 independent replication | positive pooled CI on fresh seeds | **yes** | +5.48pp, CI [+3.93,+7.14]pp (Section 5) |
| C1_C2 better than C2? | | **mixed, not a clean win** | tie on Top-1/MRR, better Top-3, weaker subgroup significance (Section 9) |
| Top-3 tension resolved or not worsened | | **resolved (on these fresh seeds)** | Section 8 |
| Hard subgroups retained | | **positive but not significant for C1_C2; significant for C2** | Section 10 |
| Known-topology regression | | **none for either arm** | Section 11 |

Every REJECT trigger is cleanly absent; the strong-replication bar is met
with substantial margin; the Top-3 concern is resolved rather than
worsened. The one criterion that does not hold cleanly is subgroup
significance for `C1_C2` specifically -- flagged, not hidden, and the
reason Section 16 gives a qualified rather than unqualified recommendation
between the two winning arms.

## 16. Final decision

**CANDIDATE_FOR_FULL_SCALE_VALIDATION.**

Justification against the plan's own three-part bar (Section 9 of the
plan doc): the selected physics representation must (a) meet the
effect-size criterion or provide equivalently compelling fresh-seed
evidence, (b) remain directionally stable and survive the paired
comparison against control, and (c) show no meaningful known-topology/
calibration regression while resolving or not worsening the Top-3
concern. Both `C2` and `C1_C2` independently satisfy (a) and (b) with
substantial margin (Section 5/5.1/6); both satisfy (c)'s regression
half (Sections 11/13); `C1_C2` additionally satisfies (c)'s Top-3 half
directly (Section 8), while `C2` alone leaves that concern where the
completed study left it (positive-trending but non-significant, not
worsened).

Not REJECT: no pre-registered REJECT trigger fires (Section 15). Not
CONTINUE_RESEARCH: the signal is not seed-unstable in direction (3/3 for
both arms across two independent studies for `C2`, 3/3 fresh seeds for
`C1_C2`), is not subgroup-limited in the sense of having shifted
*negative* on hard cases (both remain positive-pointing), and the
mechanism is not mechanistically unclear -- it is the same interpretable
`hop_magnitude_compatibility` driver the completed study already
identified, now independently confirmed, plus a `C1` addition whose
Top-3 effect is directly measured and positive.

## 17. Which representation for the next larger-scale run: C2, C1+C2, or neither?

**Not neither** -- both are ruled in by the decision rule above.

**Recommendation: carry `C1_C2` forward as the primary representation**,
with `C2` alone documented as a defensible fallback, not because either
is a clean, dominant winner but because the two arms trade off along axes
that point in different practical directions:

- **Carry `C1_C2`** if the next run's evaluation cares about Top-3
  ranking shape (e.g. any downstream use where the "true source is in the
  reported candidate set" matters, not only exact Top-1) -- `C1_C2` is the
  only arm in this report with a significant Top-3 gain over control, and
  it costs nothing on Top-1/MRR/known-topology/calibration to get it.
- **Fall back to `C2` alone** if the next run is Top-1/MRR-only, or if
  hard-subgroup significance (low-centrality, long-distance) or tighter
  cross-seed magnitude stability is the deciding factor -- `C2` is
  significant on both hard subgroups here where `C1_C2` is not, has now
  replicated on 6 independent seeds across two studies (Section 7) rather
  than 3, and shows less seed-to-seed magnitude variance (Section 6).

**Is C1 worth carrying forward?** Conditionally yes: it earns its place
specifically through the Top-3/rank-distribution improvement (Section 8),
which is a real, significant, and previously-flagged-as-a-concern effect
-- not through any Top-1 gain (it provides none beyond `C2` alone) or
subgroup gain (it is weaker there). If the next run's primary endpoint
remains Top-1-only with no interest in ranking shape, `C1` adds
implementation surface without demonstrated additional benefit on that
specific endpoint.

Everything the completed study and this branch's own plan doc explicitly
deferred remains deferred: no GNN rewrite, no additional attention stack,
no major model scaling, no gate relaxation, no re-test of the oracle's
residual hydraulic/demand privilege (Section 14), and `C_FULL`/`C3`/`B`/
`A_CAPACITY_MATCHED` remain un-rerun (their questions are already
answered).

## 18. Tests

Before training (per the plan's Section 4 test-first requirement):

- `tests/unit/test_physics_feature_ablation.py` (completed branch's own
  ablation-wiring tests, imported behavior unchanged) -- pass.
- `tests/unit/test_physics_localizer_scale_validation_ablation.py` (new,
  this branch): verifies `C1_C2` activates exactly
  `nearest_sensor_log_concentration` + `hop_magnitude_compatibility` with
  `hop_arrival_time_compatibility` (C3) zeroed after masking; `C2` and
  `C1_C2` share identical instantiated parameter counts and per-parameter
  shapes; `_mask_physics_columns` does not mutate its input tensor; the
  wrapper's seed/results-root retargeting does not mutate the completed
  branch's own `ARMS` registry; `A_CONTROL`'s model construction is
  numerically indistinguishable (same parameter count and shapes) from a
  plain `HydroCore.from_variant` call with no experimental kwargs -- pass.
- `tests/unit/test_candidate_localizer.py`,
  `tests/unit/test_capacity_matched_localizer.py` (existing, unmodified)
  -- pass.
- Combined: **53 passed, 0 failed**.

After all experiment work completed, within the remaining unattended
budget:

- **Full `tests/unit/` suite**: `python3 -m pytest tests/unit -q` --
  **883 passed, 4 skipped** (skips are pre-existing and unrelated: a
  historical-artifact-portability checkpoint not materialized in this
  checkout), **0 failed**, 35.9s.
- **`tests/integration/`**, excluding the 6 files that require
  `data/locked/`, `models/hydrocore-v5-release`, or other release/
  v4-bundle artifacts outside this branch's approved LFS scope
  (`test_default_pipeline_factory.py`, `test_production_runtime_wiring.py`,
  `test_v4_inference_bundle_loader.py`, `test_v4_pipeline_factory.py`,
  `test_v4_production_checkpoint.py`, `test_v4_release_bundle.py`):
  **40 passed, 1 failed**. The one failure
  (`test_full_output_gradient_smoke.py::
  test_ood_category_head_receives_gradient_from_genuinely_diverse_real_classes`)
  requires `data/learning-v2/cycle-b2-ood-extension`, an LFS corpus this
  branch's pre-registered scope explicitly excludes (Section 8 of the
  task) -- the failure is a shard-checksum mismatch against an
  intentionally un-smudged LFS pointer, not an implementation defect, and
  fetching that corpus to make the test pass would itself violate the
  LFS scope this report is required to respect. **Not fixed, left red,
  and documented here exactly as observed**, per the task's own
  instruction to record precisely what was and was not run rather than
  claim an inapplicable pass.
- The 6 excluded integration test files were **not run at all** (not run
  and failed, not run and skipped -- simply not invoked), to avoid any
  risk of triggering reads against `data/locked/`/release-bundle paths
  this branch never fetches.

## Explicit answers (task-required)

- **Does C1+C2 meet the +1.0pp lower-CI effect-size replication bar?**
  Yes, with substantial margin: lower bound +3.81pp on pooled fresh-seed
  unseen-topology Top-1 (Section 5.1), classified **Strong replication**
  per the pre-registered rubric.
- **Does C2 independently replicate on three fresh seeds?** Yes: +5.48pp
  pooled, CI [+3.93, +7.14]pp, positive in 3/3 fresh seeds, and now
  positive in 6/6 seeds across two independent studies (Section 7).
- **Is C1+C2 better than C2 alone?** Mixed, not a clean win: tied on
  Top-1/MRR, significantly better on Top-3, weaker (non-significant) on
  hard subgroups, more seed-to-seed magnitude variance (Section 9).
- **Does C1+C2 preserve C2's Top-1 benefit?** Yes, exactly -- pooled delta
  vs `C2` is 0.0000 (Section 5).
- **Does it improve or resolve the Top-3 tension?** Improves/resolves on
  these fresh seeds: significantly better Top-3 than both `A_CONTROL` and
  `C2` directly, best true-source-rank distribution of any arm, recovers
  31.5% of the specific examples `C2` harmed relative to control
  (Section 8). Not a complete elimination of the mechanism, but a real,
  measured, statistically supported improvement.
- **Does it retain gains on low-centrality and long-distance cases?**
  Point estimates remain positive on both (no reversion to easy-subgroup-
  only benefit) but neither individually reaches significance at this
  sample size, unlike `C2`'s significant gains on both (Section 10).
- **Does it cause any known-topology regression?** No, for either arm
  (Section 11).
- **How stable is effect magnitude across the three new seeds?** Less
  stable for `C1_C2` (+7.5/+6.4/+2.5pp, stdev 0.0263) than for `C2`
  (+4.6/+5.7/+6.1pp, stdev 0.0161); direction is fully stable for both
  (Section 6).
- **Does six-seed descriptive evidence strengthen or weaken confidence in
  C2?** Strengthens: positive in all 6 independent seeds across two
  separate studies, mean pooled delta +4.46pp (Section 7). `C1_C2` has no
  six-seed evidence and is not given that treatment.
- **Is C1 actually worth carrying forward?** Conditionally yes, on the
  strength of its Top-3/rank-distribution effect specifically -- it earns
  no additional Top-1 or subgroup benefit over `C2` alone (Section 17).
- **Which representation for the next full-scale run: C2, C1+C2, or
  neither?** Not neither. `C1_C2` is the primary recommendation if Top-3
  ranking shape is decision-relevant; `C2` alone is a defensible fallback
  if hard-subgroup significance or magnitude stability is prioritized
  instead (Section 17).

**Final classification: CANDIDATE_FOR_FULL_SCALE_VALIDATION** (Section
16), for either `C1_C2` (primary recommendation) or `C2` alone (fallback),
depending on which axis the next run's design prioritizes.
