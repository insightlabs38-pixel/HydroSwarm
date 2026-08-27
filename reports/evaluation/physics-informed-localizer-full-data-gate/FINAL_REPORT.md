# physics-informed-localizer-full-data-gate: final report

Branch: `exp/physics-informed-localizer-full-data-gate`, based on the
completed `exp/physics-informed-localizer-scale-validation` at
`8ccc59ccfccc5362cad432ee355a9266884d204c` (final report:
`reports/evaluation/physics-informed-localizer-scale-validation/FINAL_REPORT.md`,
classification `CANDIDATE_FOR_FULL_SCALE_VALIDATION`, treated as immutable
and unmodified by this branch). Plan (pre-registered before training):
`docs/evaluation/experimental/PHYSICS_INFORMED_LOCALIZER_FULL_DATA_GATE_PLAN.md`.
Status: **experimental, non-release**. No change to
`models/hydrocore-v5-release`, `data/locked/`, HydroSwarm v0.2.1,
HydroCore-v5's frozen release artifacts, M11.6 locked evidence, any
hackathon claim, or any governance module. This branch is not merged.

**Headline result**: the validated `C1_C2` candidate-conditioned localizer
(`nearest_sensor_log_concentration` + `hop_magnitude_compatibility`, `C3`
zeroed) **does not survive the data-scale gate**. At the pre-declared
seed 20261110, the same-seed pilot anchor (600 training examples)
reproduced the expected effect (`ood-UNSEEN_TOPOLOGY` Top-1 delta
**+6.79pp**, 90% CI [+3.57, +10.36]pp, excludes zero, positive -- squarely
within the +2.5pp to +7.5pp per-seed range every prior study in this
family observed). Training the *identical* architecture and optimization
budget on the **entire, unsubsampled 9000-example** Cycle-B2 train split
instead **reversed the point estimate to negative**: Top-1 delta
**-3.21pp**, 90% CI [-6.79, +0.36]pp (crosses zero -- not itself
statistically significant on Top-1 alone, but a full 10.0-percentage-point
swing from the same-seed pilot). Two secondary endpoints cross into
statistically significant harm at full-data scale: MRR delta **-2.70pp,
CI [-4.63, -0.79]pp (excludes zero, significant regression)**, and the
low-centrality hard-subgroup Top-1 delta **-3.94pp, CI [-5.98, -2.03]pp
(excludes zero, significant regression)**. The Top-3 ranking-shape picture
also reverses: net Top-3 conversions go from **+4** (pilot, this seed) to
**-7** (full-data, this seed) -- `C1_C2` now breaks more previously-correct
Top-3 cases than it fixes. No known-topology or calibration/OOD regression
was observed. **Classification: `FAIL_FULL_DATA_GATE`.** This is reported
and preserved as a negative result, per the pre-registered instruction not
to rescue it: no architecture change, no hyperparameter tuning, and no
additional arm was introduced after observing this outcome.

## 1. Scope and what this branch does not re-litigate

Per the completed scale-validation branch's own recommendation
(`CANDIDATE_FOR_FULL_SCALE_VALIDATION`), this branch runs exactly the
selected `C1_C2` representation against `A_CONTROL`, at exactly one new
pre-declared seed (`20261110`), at two training scales (600 examples,
identical to every prior pilot in this family; the entire, unsubsampled
9000-example split). `C2` alone, `C_FULL`, `C3`, `B_CANDIDATE_CONDITIONED`,
and `A_CAPACITY_MATCHED` are **not** re-run -- those questions are already
answered by the completed studies. This is a **data-scale validation
gate**, not a hyperparameter search: no GNN, no attention-stack expansion,
no new physics features, no unrelated hyperparameter tuning, no model-size
scaling, no gate relaxation, and no rescue attempt after the negative
full-data result was observed.

## 2. Git / Git LFS scope actually used

An `lfs pull` **was required**: none of the five permitted Cycle-B2
normalized splits were materialized in this fresh clone. Fetched via one
scoped command, exactly as pre-registered:

```
git lfs pull --include="data/learning-v2/cycle-b2/tensors-normalized/train/**,data/learning-v2/cycle-b2/tensors-normalized/validation/**,data/learning-v2/cycle-b2/tensors-normalized/calibration/**,data/learning-v2/cycle-b2/tensors-normalized/development_holdout/**,data/learning-v2/cycle-b2/tensors-normalized/ood-UNSEEN_TOPOLOGY/**"
```

Verified afterward: all 5 splits' `.safetensors` shards materialized (0
LFS pointers remaining, 93MB in `.git/lfs/objects`). Sampled and confirmed
**still un-smudged LFS pointers** throughout, both before and after this
branch's full run: `data/learning-v2/cycle-b2-ood-extension`,
`data/learning-v2/cycle-b2-control-v2`,
`data/learning-v2/cycle-b2-trajectories-v3`,
`models/cycle-b2-candidates`, `models/cycle-b2-controls`. No
`git lfs fetch --all`, no unrestricted `git lfs pull`, no repo-wide
`git lfs checkout`, and no already-downloaded LFS objects were pruned.
**No unexpected LFS object was materialized beyond the five approved
paths.** The one integration test requiring `cycle-b2-ood-extension`
(`tests/integration/test_full_output_gradient_smoke.py::
test_ood_category_head_receives_gradient_from_genuinely_diverse_real_classes`)
was run as part of Section 9's test pass and failed with a shard-checksum
mismatch **because that corpus is intentionally an un-smudged LFS pointer
on this branch**, not an implementation defect -- the same documented
exception the completed scale-validation study recorded.

## 3. Pre-declared seed and arms

`SEED = 20261110` -- fixed in the plan doc before any training, disjoint
from every prior seed in this experimental family, never selected or
replaced based on results.

| arm | `localizer_mode` | active physics columns | total params | delta vs A_CONTROL |
|---|---|---|---|---|
| `A_CONTROL` | `default` | -- | 4,044,113 | -- |
| `C1_C2` | `candidate_conditioned` | `nearest_sensor_log_concentration` + `hop_magnitude_compatibility` | 4,231,897 | +187,784 (+4.643%) |

Both totals are **byte-identical** to the completed scale-validation
study's own `A_CONTROL`/`C1_C2` parameter counts, confirmed both by
construction (`_mask_physics_columns` ablation, reused unmodified) and by
the dedicated unit tests (`tests/unit/
test_physics_localizer_full_data_gate_ablation.py`, 18/18 passed before
any training). `C3` (`hop_arrival_time_compatibility`) confirmed zeroed
for every example in both stages.

## 4. Stage 0 pre-registration and audit

Recorded before any training (full detail:
`docs/evaluation/experimental/PHYSICS_INFORMED_LOCALIZER_FULL_DATA_GATE_PLAN.md`):
repository commit `8ccc59ccfccc5362cad432ee355a9266884d204c`; dataset
manifest `index_sha256` hashes for all 5 splits; verified split sizes
(`train`=9000, 3000/family, 2100/family carrying a real source label;
`validation`=1000; `calibration`=1000; `development_holdout`=1750;
`ood-UNSEEN_TOPOLOGY`=400); verified `A_CONTROL`/`C1_C2` parameter counts
and architecture identity; verified no label/target tensor enters physics
feature construction; verified no governance module is touched by this
branch's diff.

## 5. An infrastructure bug discovered and fixed en route (in scope, documented)

Launching the full-data-9000 stage for `A_CONTROL` crashed immediately
with `FloatingPointError: non-finite multitask loss`. Root-caused by
scanning all 9000 training examples individually: `compute_multitask_loss`'s
all-masked "no valid target this batch" fallback computed the
graph-connected zero as `logits.sum() * 0.0`. The model's own masked-out
candidate positions carry a large-magnitude sentinel (observed: float32's
near-min, -3.4028e38); summing two or more of them in one batch overflows
float32's representable range to -inf, and `-inf * 0.0` is NaN. Every
prior study in this experimental family filtered training data to
`has_real_source == True` examples (the pilot protocol's own stratified
sampling), so an all-masked source_node batch was never exercised in
training before this branch; it is common (900/9000 individual examples
trigger it; batches of 2 combine two independently-drawn examples, so the
crash surfaced almost immediately) once the **entire, unfiltered** corpus
is used, exactly as this gate's own pre-registration requires.

**Fix** (`src/hydroswarm/training/losses.py`, all 6 call sites sharing
this pattern): zero every element BEFORE summing, not after
(`(logits * 0.0).sum()` instead of `logits.sum() * 0.0`). `0.0 * any
finite value` is exactly `+/-0.0` in IEEE 754 regardless of magnitude, so
summing zeros can never overflow; the result is mathematically identical
(`0.0`) to the un-overflowed case for every batch that previously computed
a finite loss, and remains graph-connected so `backward()` still runs.
Confirmed by: the existing `tests/unit/` suite (all passed, no
regression), a new targeted regression test
(`test_masked_out_classification_task_with_extreme_logits_is_finite_not_nan`,
`tests/unit/test_losses.py`) reproducing the exact failure with the
observed sentinel magnitude, and a fresh scan of all 9000 full-data train
examples under both arms after the fix (**0 non-finite, down from 900/9000
for `A_CONTROL` before the fix**). This is a pure numerical-stability
bug fix, unrelated to any architecture, hyperparameter, loss-weight, or
governance decision this branch is scoped to leave alone -- it does not
touch `CandidateConditionedLocalizer`, `_mask_physics_columns`, any task
weight value, any calibration/OOD/actionability code path, or the trained
result of any batch that did not previously crash. Full detail and
justification: commit `e85c568`.

## 6. Stage 1: same-seed pilot anchor (600 examples)

Byte-for-byte reuse of the existing 600-example protocol at seed 20261110
(200/family stratified, validation/development_holdout capped at 300, full
calibration/OOD, 6 epochs). Both arms trained successfully (A_CONTROL
435.8s, C1_C2 432.2s).

| population | A_CONTROL top1/top3/mrr | C1_C2 top1/top3/mrr |
|---|---|---|
| validation | 0.677 / 0.867 / 0.789 | 0.667 / 0.877 / 0.786 |
| development_holdout | 0.673 / 0.867 / 0.786 | 0.707 / 0.883 / 0.806 |
| **ood-UNSEEN_TOPOLOGY** | **0.336 / 0.732 / 0.557** | **0.404 / 0.746 / 0.603** |

`ood-UNSEEN_TOPOLOGY` paired bootstrap (n=280, 2000 resamples, seed
20260826, 90% CI): Top-1 delta **+6.79pp [+3.57, +10.36]pp (excludes zero,
positive)**; MRR delta +4.61pp [+2.76, +6.65]pp (excludes zero, positive);
Top-3 delta +1.43pp [-1.07, +3.93]pp (CI includes zero). This is squarely
within the per-seed magnitude range (+2.5pp to +7.5pp) every prior 600-
example study in this family observed -- **the pilot anchor reproduces the
expected effect at this seed before scaling training data**, exactly as
Stage 1 is designed to establish.

## 7. Stage 2: full-data training (9000 examples)

Entire, unsubsampled 9000-example train split (no stratified cap, no
real-source filtering of the training set itself), full uncapped
evaluation populations, identical architecture/optimizer/6-epoch budget,
per-epoch checkpointing for crash recovery only. Both arms trained
successfully after the Section 5 fix (A_CONTROL 5952.2s ~= 99.2min, C1_C2
6388.6s ~= 106.5min; ~206 minutes total training wall time for this
stage).

| population | A_CONTROL top1/top3/mrr | C1_C2 top1/top3/mrr |
|---|---|---|
| validation | 0.702 / 0.881 / 0.804 | 0.715 / 0.879 / 0.811 |
| development_holdout | 0.708 / 0.878 / 0.807 | 0.698 / 0.882 / 0.802 |
| **ood-UNSEEN_TOPOLOGY** | **0.461 / 0.800 / 0.653** | **0.429 / 0.775 / 0.626** |

Both arms' absolute performance improves substantially over the pilot
(more training data helps both arms, as expected) -- `A_CONTROL`'s own
unseen-topology Top-1 rises from 0.336 (pilot) to 0.461 (full-data), a
+12.5pp within-arm gain from 15x more training data. But **`C1_C2`'s
advantage over `A_CONTROL` does not merely shrink -- it inverts**:
`C1_C2`'s own full-data Top-1 (0.429) is now *below* `A_CONTROL`'s
(0.461).

## 8. Primary endpoint (pre-registered): full-data `ood-UNSEEN_TOPOLOGY` Top-1

Paired bootstrap, n=280 (matched by `scenario_id`), 2000 resamples,
deterministic bootstrap seed 20260826, 90% CI
(`full-data-9000/seed-20261110/paired-transitions.json`,
`gate/primary-endpoint-gate.json`):

| endpoint | delta | 90% CI | excludes zero |
|---|---|---|---|
| **Top-1** | **-3.21pp** | **[-6.79, +0.36]pp** | no (crosses zero) |
| Top-3 | -2.50pp | [-5.00, 0.00]pp | no (right at the boundary) |
| **MRR** | **-2.70pp** | **[-4.63, -0.79]pp** | **yes, negative -- significant regression** |

Paired Top-1 transition table (n=280): both_correct=105, control_only
(C1_C2 regressed relative to control)=24, arm_only (C1_C2 fixed)=15,
both_wrong=136 -- net **-9** (more examples broken than fixed), reversing
the pilot's own **+19** net at this seed (27 fixed, 8 broken, Section 6).

## 9. Full-data success gate: evaluated against the pre-registered 8-point rubric

| # | criterion | holds? | evidence |
|---|---|---|---|
| 1 | Top-1 delta vs A_CONTROL is positive | **NO** | -3.21pp (Section 8) |
| 2 | Point estimate >= +2.0pp | NO (moot, already negative) | -3.21pp |
| 3 | Paired 90% CI excludes zero, positive | NO | CI [-6.79, +0.36]pp crosses zero |
| 4 | No statistically significant Top-3 negative regression | borderline yes | CI upper bound is exactly 0.0, not < 0 |
| 5 | No statistically significant MRR negative regression | **NO** | CI [-4.63, -0.79]pp, excludes zero negative |
| 6 | No known-topology material regression | yes | validation/development_holdout CIs both include zero (Section 11) |
| 7 | Calibration/OOD proxy behavior not materially worse | yes | Section 12 |
| 8 | No governance code altered | yes | Section 5's fix is training-loss infrastructure, not governance |

Per the pre-registered decision rule (plan doc Section 6): **FAIL_FULL_DATA_GATE
triggers as soon as the full-data Top-1 delta is <= 0** -- criterion 1
alone is sufficient, independent of the CI's own significance. The
additional, statistically significant MRR regression (criterion 5) and
low-centrality subgroup regression (Section 10) independently corroborate
that this is a real degradation, not merely a null result.

**Classification: `FAIL_FULL_DATA_GATE`.**

## 10. Hard-subgroup validation (full-data-trained models)

Betweenness-centrality terciles and hop-to-nearest-sensor median split,
computed from `A_CONTROL`'s own full-data rows
(`full-data-9000/seed-20261110/subgroup-paired-bootstrap.json`):

| subgroup | n | C1_C2 delta | 90% CI | excludes zero |
|---|---|---|---|---|
| **low_centrality** | 786 | **-3.94pp** | **[-5.98, -2.03]pp** | **yes, negative -- significant regression** |
| long_distance | 746 | -1.74pp | [-4.42, +0.94]pp | no |

The low-centrality hard subgroup -- the exact population the completed
studies found `C1_C2`/`C2` most reliably helped at 600-example scale --
shows a **statistically significant harmful regression** at full-data
scale. This is the single clearest quantitative signal in this report that
the effect has not merely weakened but reversed on precisely the cases it
was previously strongest on.

(Note: subgroup `n` here pools across `validation`+`development_holdout`+
`ood-UNSEEN_TOPOLOGY` localized rows, consistent with every prior study's
own `centrality_subgroups`/`distance_subgroups` convention -- these are not
restricted to the primary `ood-UNSEEN_TOPOLOGY` population alone.)

## 11. Known-topology guardrail

`validation`/`development_holdout` Top-1 deltas vs `A_CONTROL`
(`gate/primary-endpoint-gate.json`'s `known_topology_regressions`):

| population | n | C1_C2 delta | 90% CI | material regression? |
|---|---|---|---|---|
| validation | 712 | +1.26pp | [-0.42, +3.09]pp | no |
| development_holdout | 1228 | -0.98pp | [-2.36, +0.41]pp | no |

Neither CI excludes zero on the negative side -- **no known-topology
regression**, consistent with every prior study in this family.

## 12. Calibration / OOD proxy behavior

Split-conformal coverage identical (0.9073) for both arms, by
construction. Expected calibration error: `A_CONTROL` 0.0345, `C1_C2`
0.0382 (a 0.37-point increase, below this branch's own 5-point
"materially worse" threshold). `proxy_actionable_rate`/
`proxy_abstention_rate`/`proxy_calibrated_coverage` on `validation` and
`development_holdout` are within a point or two of each other in both
directions; `ood-UNSEEN_TOPOLOGY` reports 100% abstention/caution for
*both* arms (as in every prior study -- the OOD detector correctly flags
the unseen topology family regardless of arm, so this population is never
"proxy-actionable" for anyone). **No calibration/OOD regression is
materially worse for `C1_C2`.**

## 13. Ranking-shape validation (task-required)

Because C1's inclusion rationale was specifically Top-3/ranking quality,
not Top-1 alone, this is compared directly against the pilot at the same
seed (`{pilot-600,full-data-9000}/seed-20261110/ranking-shape-analysis.json`):

| metric | pilot-600 | full-data-9000 |
|---|---|---|
| Top-3 transition net (C1_C2-only minus control-only) | **+4** (11 fixed, 7 broken) | **-7** (5 fixed, 12 broken) |
| correct-Top-3-to-outside-Top-3 conversions (control_only) | 7 | 12 |
| mean true-source rank when C1_C2's Top-1 is wrong, vs control on same examples | -0.096 (C1_C2 better) | **+0.344 (C1_C2 worse)** |
| rank improved / unchanged / worsened (of those Top-1-wrong cases) | 39 / 105 / 23 | 14 / 92 / 54 |
| fraction of Top-1 failures with source still in Top-3 | C1_C2 57.5% vs control 59.7% | C1_C2 60.6% vs control 62.9% |

**Every one of these reverses direction between the pilot and full-data
stages.** At 600 examples, `C1_C2` converted more wrong-Top-3 cases to
correct than the reverse, and improved true-source rank on average when
its own Top-1 stayed wrong. At 9000 examples, the same architecture does
the opposite on both axes. This is not a case of "the Top-1 gain
disappeared but the Top-3/ranking benefit survived" -- **the ranking-shape
benefit reverses in lockstep with Top-1**, at this seed.

## 14. Critical scale analysis: 600-example effect vs 9000-example effect (same seed)

```
pilot A_CONTROL            0.3357  (ood-UNSEEN_TOPOLOGY top1)
pilot C1_C2                0.4036
pilot delta                +0.0679  (+6.79pp)

full-data A_CONTROL         0.4607
full-data C1_C2             0.4286
full-data delta             -0.0321  (-3.21pp)
```

Swing: **-10.00 percentage points** between the two training scales at the
identical seed. Descriptive classification (per the pre-registered
scale-effect taxonomy): **eliminated** -- more precisely, the effect does
not merely shrink to zero, it **reverses sign**. `A_CONTROL` itself
improves substantially with more data (+12.5pp within-arm), consistent
with the general expectation that more training data helps a
representation-limited model; `C1_C2`'s own within-arm full-data Top-1
(0.429) is actually *slightly lower* than would be predicted by simple
extrapolation, and specifically lower than `A_CONTROL`'s full-data figure.
This is evidence against, not for, treating `C1_C2`'s pilot-scale
advantage as a stable, scale-invariant representational improvement --
it appears more consistent with an artifact of training-data scarcity
(the physics features providing a useful shortcut/regularizer specifically
when there isn't enough data for the backbone to learn source
localization well on its own) than with a durable architectural gain that
compounds with more data.

## 15. Cross-study context (descriptive only, no pooled test across scales)

Every prior study in this experimental family trained on 600 examples;
this branch is the first to report a 9000-example result for any arm.
These are reported side by side, clearly labeled, never combined into one
pooled significance test:

- **600-example evidence** (prior, immutable, committed on other
  branches): `C2` positive across 6 independent seeds spanning two
  studies; `C1_C2` positive across 3 fresh seeds (pooled +5.48pp, CI
  [+3.81, +7.14]pp) plus this branch's own pilot-600 anchor at a 4th,
  disjoint seed (+6.79pp) -- **4/4 pre-scale seeds now positive for
  `C1_C2`**, a materially larger and more consistent body of evidence than
  any single seed alone.
- **9000-example evidence** (this branch, one seed only): `C1_C2` Top-1
  delta -3.21pp, CI crosses zero; MRR and low-centrality subgroup
  significantly negative. **1/1 full-data seed run so far, and it is
  negative.**

No claim is made that the 600-example effect was ever spurious *at that
scale* -- 4 independent seeds replicating a consistent positive effect is
real evidence at that scale. The claim this report supports is narrower
and specific to this branch's own pre-registered question: **that
600-example effect does not extrapolate to the full 9000-example
corpus, at least not at this one full-data seed.**

## 16. Tests

Before training (plan doc's test-first requirement):
`tests/unit/test_physics_localizer_full_data_gate_ablation.py` (new, 18
tests: scope retargeting, C1_C2 column masking with C3 zeroed, identical
parameter count/shapes vs C2, unchanged default A_CONTROL behavior,
checkpoint-discovery helper), `tests/unit/test_physics_feature_ablation.py`
and `tests/unit/test_physics_localizer_scale_validation_ablation.py`
(existing, unmodified, reused behavior) -- **63 passed** (43 + 20
dataset-loading), 0 failed.

After the Section 5 loss-overflow fix: full `tests/unit/` suite --
**903 passed, 4 skipped** (pre-existing, unrelated: a historical-artifact-
portability checkpoint not materialized in this checkout), 0 failed
(37.8s).

After all experiment work completed: `tests/integration/`, excluding the
6 files that require `data/locked/`, `models/hydrocore-v5-release`, or
other release/v4-bundle artifacts outside this branch's approved LFS scope
(`test_default_pipeline_factory.py`, `test_production_runtime_wiring.py`,
`test_v4_inference_bundle_loader.py`, `test_v4_pipeline_factory.py`,
`test_v4_production_checkpoint.py`, `test_v4_release_bundle.py`) --
**40 passed, 1 failed**. The one failure
(`test_full_output_gradient_smoke.py::
test_ood_category_head_receives_gradient_from_genuinely_diverse_real_classes`)
requires `data/learning-v2/cycle-b2-ood-extension`, an LFS corpus this
branch's pre-registered scope explicitly excludes -- a shard-checksum
mismatch against an intentionally un-smudged LFS pointer, not an
implementation defect. **Not fixed, left red, and documented here exactly
as observed**, identical to the completed scale-validation study's own
recorded exception. The 6 excluded integration test files were not run at
all.

## Explicit answers (task-required)

- **Did C1_C2 beat A_CONTROL after training on all 9000 examples?** No.
  Full-data `ood-UNSEEN_TOPOLOGY` Top-1 for `C1_C2` (0.4286) is below
  `A_CONTROL` (0.4607) (Section 7/8).
- **What was the Top-1 delta and 90% CI?** -3.21pp, 90% CI
  [-6.79, +0.36]pp (Section 8).
- **Did it clear the +2.0pp full-data effect-size gate?** No -- the point
  estimate itself is negative, well below the +2.0pp bar (Section 9).
- **Did the Top-3 advantage survive?** No. Point estimate -2.50pp; net
  Top-3 transitions reverse from +4 (pilot) to -7 (full-data) at this
  same seed (Section 13).
- **Did MRR improve, remain neutral, or regress?** Regressed,
  significantly: -2.70pp, CI [-4.63, -0.79]pp, excludes zero (Section 8).
- **Did the effect survive on low-centrality sources?** No -- it reversed
  into a statistically significant regression: -3.94pp, CI
  [-5.98, -2.03]pp (Section 10).
- **Did it survive on long-distance sources?** Point estimate also
  negative (-1.74pp) though the CI includes zero -- not independently
  significant, but not a survival of the prior positive effect either
  (Section 10).
- **Was there any known-topology regression?** No -- validation and
  development_holdout deltas both have CIs that include zero, no
  statistically supported regression either way (Section 11).
- **Did calibration/OOD behavior materially change?** No -- coverage
  identical by construction, ECE difference (0.37pp) well below the
  5-point materiality threshold, proxy actionable/abstention rates within
  a point or two in both directions (Section 12).
- **How did the 600-example and 9000-example deltas differ at the same
  seed?** +6.79pp (600) vs -3.21pp (9000) -- a 10.0-percentage-point swing
  that crosses zero and reverses sign (Section 14).
- **Did more training data strengthen, preserve, attenuate, or eliminate
  the physics-informed advantage?** Eliminated, and reversed -- not merely
  attenuated toward zero, but past zero into a negative (and on two
  secondary endpoints, statistically significant negative) direction
  (Section 14).
- **Is C1 still worth retaining alongside C2 after full-data training?**
  This branch cannot answer that question directly -- it did not re-run
  `C2` alone at full-data scale (out of scope; see Section 1). What it can
  say: whatever mechanism made `C1_C2` beneficial at 600-example scale
  does not carry over to 9000-example scale for `C1_C2` specifically, on
  this one seed, which is reason for real caution about carrying `C1`
  (or `C2`) forward into a full-data or promotion-quality run without
  first re-testing `C2` alone at full-data scale and adding further
  full-data seeds.
- **Is the evidence now strong enough to proceed to a promotion-quality
  20-epoch training run?** No. This is exactly what the pre-registered
  gate is designed to prevent: a `FAIL_FULL_DATA_GATE` result does not
  earn `CANDIDATE_FOR_PROMOTION_QUALITY_TRAINING`. Proceeding to a
  20-epoch promotion-quality run on `C1_C2` now, on the strength of the
  600-example evidence alone, would be training past a result this
  branch's own data-scale check found does not hold up.

## Final classification

**`FAIL_FULL_DATA_GATE`.**

Per the pre-registered decision rule (plan doc Section 6): the full-data
`C1_C2` vs `A_CONTROL` `ood-UNSEEN_TOPOLOGY` Top-1 delta is <= 0 (-3.21pp),
which alone triggers `FAIL_FULL_DATA_GATE` independent of the CI's own
significance. This is corroborated, not merely permitted, by two
independently significant harmful signals (MRR, low-centrality subgroup)
and a full reversal of the Top-3 ranking-shape picture between the pilot
and full-data stages at this same seed. This gate does **not** earn
`CANDIDATE_FOR_PROMOTION_QUALITY_TRAINING`. The negative result is
preserved as observed: no architecture change, no hyperparameter re-tuning,
no additional arm, and no attempt to rescue this outcome was made after it
was seen, per the pre-registered instruction. The one code change made
during this branch (Section 5) is a pure numerical-stability bug fix in
shared, non-experimental training-loss infrastructure, made because the
full, unfiltered corpus this gate's own pre-registration required could
not otherwise be trained on at all, and is independently verified to leave
every previously-finite loss value unchanged.

**Recommended next step**: before spending further full-scale compute on
`C1_C2`, (a) re-run `C2` alone at full-data scale at this same seed (and
ideally 1-2 more) to determine whether the reversal is specific to `C1_C2`
or a broader property of physics-informed candidate conditioning at this
data scale; (b) if time/budget allows, run 1-2 additional full-data seeds
for `C1_C2` before concluding the effect is seed-representative rather
than a single unlucky draw, since this branch (by design, per its own
compute-bounded pre-registration) has only one full-data seed; (c) do
**not** proceed to a 20-epoch promotion-quality run on `C1_C2` until a
full-data result exists that clears (or a revised pre-registration
explicitly relaxes) the +2.0pp gate this branch was designed to test.
