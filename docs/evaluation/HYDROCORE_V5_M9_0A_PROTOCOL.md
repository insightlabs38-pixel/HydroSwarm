# HydroCore-v5 Milestone 9.0a protocol (frozen before Arm B2 is trained or either arm is evaluated)

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` (see its own new
Section 8.7) and `docs/evaluation/HYDROCORE_V5_M9_0_PROTOCOL.md` (referenced,
not rewritten). This document freezes the M9.0a sub-protocol BEFORE Arm B2 is
trained or evaluated. It is not altered after seeing results.

## 0. Why this milestone exists

Milestone 9.0 (`docs/evaluation/HYDROCORE_V5_M9_0_PROTOCOL.md`,
`reports/evaluation/hydrocore-v5/m9-0-summary.md`) found a pooled
unseen-topology MATURE neural top1 gain of +6.60pp (hybrid +7.29pp, paired
90% bootstrap CI entirely > 0, all three unseen families improved, all three
seeds directionally positive) for `INTERLEAVED_MULTI_FAMILY` (Arm B) over
`SINGLE_FAMILY_CONTROL` (Arm A), but rejected Arm B because its known-family
`B_DEPTH_AWARE` marginal coverage (0.833, representative seed 31874 only)
fell below the frozen 0.85 guardrail, and provisionally selected
`AGE_FIX_ONLY + SINGLE_FAMILY_CURRENT_TRAINING` for Milestone-9 capacity
scaling.

Two confounds/uncertainties remain before that fallback recipe is accepted:

1. **Optimizer-update count.** Arm A trained with `batch_size=2`,
   `gradient_accumulation_steps=4` (4 microbatches/update, 1 family). Arm B
   trained with 1 microbatch/family x 3 families = 3 microbatches/update.
   Empirically (Section 2 below), Arm A's real optimizer-step count was 1350
   and Arm B's was 1800 -- Arm B received 33% more optimizer updates and a
   different scheduler trajectory (`m9-0-runs/ARM_B_INTERLEAVED_MULTI_FAMILY-
   seed*.json` `training_summary.global_steps`; `m8-7-runs/AGE_FIX_ONLY-
   seed*.json` `training_summary.global_steps`/epoch-level `metrics.jsonl`).
   The observed unseen-topology gain cannot yet be attributed solely to
   interleaved topology diversity.
2. **Calibration seed robustness.** M9.0 trained/evaluated all three
   predictor seeds, but the promotion-blocking B_DEPTH_AWARE calibration
   number was computed using only the representative seed (31874) --
   `m9-0-calibration.json` has no per-seed breakdown for the other two
   seeds.

M9.0a resolves both: Arm B2 (`STEP_MATCHED_INTERLEAVED_MULTI_FAMILY`) is
trained with 4 microbatches/update (matching Arm A's accumulation window)
using a fixed 3-update rotation so each family still gets equal (1/3)
long-run weight, and calibration is fit/evaluated separately for all three
predictor seeds for both arms.

## 1. Arms

**Arm A -- SINGLE_FAMILY_CONTROL.** Identical definition to M9.0 Arm A
(`HYDROCORE_V5_M9_0_PROTOCOL.md` Sections 1-2): REUSES the existing
Milestone-8.7 `AGE_FIX_ONLY` checkpoints (seeds 20260814, 31874, 20260815)
verbatim, zero retraining. The M9.0 comparability verification
(architecture, representation, optimizer/schedule/task-weights, 600-scenario
budget, checkpoint-selection rule) remains valid unchanged -- no new mismatch
was discovered while writing this protocol, so Arm A is reused rather than
retrained (M9.0 Section 2, restated here as still-valid per this document's
own Section 0 preflight).

**Arm B2 -- STEP_MATCHED_INTERLEAVED_MULTI_FAMILY.** Representation
`AGE_FIX_ONLY`, IDENTICAL architecture to Arm A (~4,182,612 parameters).
Trained fresh, 3 seeds, on the SAME golden-reference/branched-loop/loop-grid
corpus M9.0 Arm B used (`run_m7_topology`'s `TRAINED_FAMILIES`,
`TRAIN_PER_FAMILY=200`, `VALIDATION_PER_FAMILY`, `CALIBRATION_PER_FAMILY`,
imported unmodified), differing from M9.0 Arm B ONLY in the per-optimizer-
step family schedule (Section 3 below) and the resulting scheduler
`total_steps` (Section 5).

## 2. Exact Arm-A optimizer-step accounting (measured, not assumed)

Derived directly from `experiments/runs/hydrocore-v5-causal-m8-7/AGE_FIX_ONLY-
seed*/*/metrics.jsonl` (per-microbatch records; `global_step` is the
optimizer-step counter at time of each record, matching
`Trainer._train_epoch`'s own semantics) for all three seeds -- identical
across all three seeds because curriculum-stage filtering is seed-independent
(deterministic round-robin over the same 5-stage enum) and Arm A's dataset
size (600) does not vary by seed:

| epoch | microbatches this epoch | optimizer steps this epoch (microbatches / 4) |
|---|---|---|
| 0 | 60 | 15 |
| 1 | 120 | 30 |
| 2 | 180 | 45 |
| 3 | 240 | 60 |
| 4-19 (16 epochs, saturated) | 300 each | 75 each |

**Arm A total optimizer steps (all 20 epochs) = 15+30+45+60+16x75 = 1350**,
identical for all three seeds (verified directly against each seed's
`metrics.jsonl`; the `training_summary.global_steps` field recorded in
`m8-7-runs/*.json` is the step count AT THE BEST EPOCH, not necessarily the
full-run total -- seed 20260815's best epoch was 18 of 19, giving a recorded
1275, but its `metrics.jsonl` confirms the same 1350-step full-run total as
the other two seeds; this document's parity target is the FULL-RUN total,
1350, not the best-epoch snapshot).

Arm A's scheduler is built by `Trainer.__init__` with `total_steps =
ceil(len(train_dataset) / batch_size / gradient_accumulation_steps) * epochs
= ceil(600/2/4) * 20 = 1500` (the STATIC full-pool estimate, using the
un-curriculum-filtered dataset length) -- this is DIFFERENT from the REAL
1350 steps actually taken; Arm A's own LR trajectory therefore never reaches
cosine-schedule fraction 1.0, stopping at real-step-1350/scheduler-total-1500
= 0.9 of the way through its own nominal schedule. This is an existing,
unmodified property of how Arm A was actually trained (M8.7, unchanged) --
Arm B2 replicates it exactly (Section 5) rather than "fixing" it.

## 3. Step-matched interleaved training design (Arm B2, frozen before training)

4 family-pure microbatches per optimizer update (matching Arm A's
`gradient_accumulation_steps=4`), using the milestone instructions' preferred
deterministic 3-update rotation so each family receives exactly 4
microbatches per 3 consecutive optimizer updates (equal 1/3 long-run
weighting, no two consecutive same-family microbatches ever adjacent across
an update boundary beyond what the rotation itself produces, and no
contiguous single-family phase):

```
update cycle position 0 (mod 3 == 0): golden, branched-loop, loop-grid, golden        (golden gets the 4th slot)
update cycle position 1 (mod 3 == 1): golden, branched-loop, loop-grid, branched-loop  (branched-loop gets the 4th slot)
update cycle position 2 (mod 3 == 2): golden, branched-loop, loop-grid, loop-grid      (loop-grid gets the 4th slot)
```

Per optimizer update:

```
optimizer.zero_grad(set_to_none=True)
for slot_family in (golden, branched, loop, extra_family_for_this_cycle_position):
    batch = NEXT micro-batch from slot_family's own per-epoch iterator
            (the "extra" slot draws a DIFFERENT, subsequent microbatch from
            that family's iterator, not a repeat of the batch already drawn
            in that same update's base slot for the same family)
    output = model(batch.inputs)
    result = compute_multitask_loss(...)
    (result.total / 4).backward()          # NOT /3 -- Section 4 below.
clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
optimizer.step()
scheduler.step()
global_step += 1
```

This is the direct 4-microbatch generalization of M9.0's own
`interleaved_optimizer_step` (`run_m9_0_arm_b.py`): same
zero_grad-then-accumulate-then-clip-then-step structure, `num_families`
generalized to "number of scheduled slots this update" (4, with one family
appearing twice), same shared single `optimizer`/`model` receiving every
slot's gradient before any `.step()` call.

One shared `torch.optim.AdamW` instance and one shared `LambdaLR` scheduler
receive every slot's gradient contribution before any `.step()` call -- there
is exactly one set of model parameters and one optimizer state for all three
families (Section 8 tests assert this directly, extending M9.0's own).

### Why this exactly matches Arm A's per-epoch step count with zero remainder

Each family's 200-scenario pool is curriculum-filtered by the SAME
`CurriculumSchedule.progressive()` fractions Arm A's 600-scenario pool is
(verified directly against a real pool before training, Section 8): 20%,
40%, 60%, 80%, 100% of the pool at epochs 0, 1, 2, 3, 4+ respectively. At
`batch_size=2`, that is 20, 40, 60, 80, 100 microbatches/family/epoch
(identical across all three families and all three seeds, matching what
M9.0's own Arm B run already exhibits per-epoch in
`m9-0-runs/ARM_B-seed*/*/metrics.jsonl`).

Targeting Arm A's own per-epoch optimizer-step counts (15, 30, 45, 60,
75x16; Section 2) with 4 microbatches/update, 4/3 average per family: each
epoch's total family-microbatch consumption is `updates_this_epoch x 4 / 3`
per family = 20, 40, 60, 80, 100 -- EXACTLY equal to that family's own
naturally-available curriculum-filtered pool size that epoch, with ZERO
remainder, and `updates_this_epoch` (15, 30, 45, 60, 75) is itself exactly
divisible by 3 (5, 10, 15, 20, 25 full 3-update rotation blocks) every
single epoch. This is not a coincidence: it falls directly out of
`TRAIN_PER_FAMILY x 3 == 600` (Arm A's own pool size) and both arms sharing
`batch_size=2` and the same curriculum fractions -- but it is verified
directly against real pools (Section 8), not assumed.

**Therefore Arm B2 achieves EXACT per-epoch and total optimizer-step parity
with Arm A (1350 steps total, matching Section 2's per-epoch table exactly,
zero deviation) while consuming its full, unmodified 200-scenario-per-family
curriculum-filtered pool every epoch -- no data volume increase, no
under-consumption, no leftover microbatches, no predeclared-nearest-parity
compromise needed.**

## 4. Gradient normalization

Each of the 4 accumulated microbatch losses is divided by 4 before
`.backward()` (`result.total / 4`), matching Arm A's own
`gradient_accumulation_steps=4` normalization exactly -- NOT by 3, unlike
M9.0's Arm B (which divided by `num_families=3`, matching ITS OWN 3-microbatch
window). Verified by a dedicated unit test (Section 8).

## 5. Scheduler parity

Arm B2's scheduler is built with the SAME `total_steps=1500` static estimate
Arm A's own `Trainer.__init__` uses (`ceil(600/2/4)*20`), NOT a per-family
recomputation (M9.0's Arm B used `ceil(200/2/1)*20=2000`, the source of its
scheduler-trajectory mismatch). Same `warmup_steps=10`, same `lr=0.0003`,
same cosine schedule, same `weight_decay=0.01`, same
`gradient_clip_norm=1.0`. At any given REAL optimizer-step index, Arm A and
Arm B2 therefore compute the identical `multiplier(step)` cosine value
(`_scheduler`'s own formula, `hydroswarm.training.trainer._scheduler`,
reused unmodified) -- both arms reach real-step-1350 out of nominal
total-steps-1500 (fraction 0.9) at the end of training, not merely
"the same total_steps configured," but the same REALIZED trajectory.

## 6. Exposure accounting

Unmodified from M9.0: TRAIN_PER_FAMILY=200/family (600 total, matching Arm
A's 600), VALIDATION_PER_FAMILY-derived 100 total, CALIBRATION_PER_FAMILY=50
(150 total). Arm B2's training data volume is NOT increased above M9.0
Arm B's; only the optimizer-step SCHEDULE consuming that same data changes.
Per-epoch/per-family microbatch and optimizer-step counts are recorded
exactly as derived in Section 3 and verified against the real training run
in `reports/evaluation/hydrocore-v5/m9-0a-budget-parity.json`.

## 7. Family-specific signature libraries, AGE_FIX_ONLY propagation

Unmodified from M9.0 (`HYDROCORE_V5_M9_0_PROTOCOL.md` Sections 7-8): three
independent `SignatureLibrary` instances, one per family, each family's
`CausalPrefixDatasetView` (and `stages_through()`) constructed with exactly
that family's own library; `unobserved_age_sentinel="fixed"`,
`include_relative_gap_feature=False` throughout dataset construction, every
epoch's `stages_through()` return value, validation, and evaluation views.

## 8. Tests (frozen list, before training)

`tests/scientific/test_step_matched_interleaving_m9_0a.py` proves, before
Arm B2 is trained for real:

1. exactly 4 microbatches contribute before `optimizer.step()`;
2. the 3-update rotation gives each family exactly 4 microbatches per 3
   consecutive updates (equal long-run family weighting);
3. each accumulated loss is normalized by `/4`, not `/3` or any other value;
4. no `zero_grad()` call occurs between microbatch contributions within one
   update (gradients genuinely accumulate -- same combined-vs-summed-solo-
   gradient equivalence test M9.0 used, generalized to 4 slots with one
   family appearing twice);
5. `scheduler.step()` fires exactly once per optimizer update, never per
   microbatch;
6. the real per-epoch/per-family microbatch counts (20/40/60/80/100) exactly
   match Section 3's predicted consumption from a real (small, fast) pool,
   with zero remainder, for at least one non-saturated and one saturated
   epoch;
7. family-specific `SignatureLibrary`s remain correct (each family uses its
   own; cross-family use raises, same backstop M9.0 already tests);
8. `AGE_FIX_ONLY` semantics survive `stages_through()` for all three
   families;
9. no future evidence enters a depth-truncated prefix;
10. the SAME model/optimizer state (not per-family copies) receives every
    slot's gradient.

M9.0's own `tests/scientific/test_interleaved_topology_m9_0.py` is retained
unmodified (it documents/protects M9.0's 3-microbatch Arm B, still a valid
historical arm) and re-run as part of Section 11's validation, not replaced.

## 9. Standard known-network evaluation, trained-family retention, primary unseen-topology evaluation

Unmodified from M9.0 (`HYDROCORE_V5_M9_0_PROTOCOL.md` Sections 11-14):
golden-reference `development_holdout`, depths 1,2,3,4,6,12,25, EARLY/MID/
MATURE aggregation, neural+hybrid metrics; Arm-B2 trained-family retention on
branched-loop/loop-grid labeled `TRAINED_FAMILY_GENERALIZATION`; primary
unseen-topology evaluation on IDENTICAL coastal-branch/tree-branch/dense-loop
held-out incidents M9.0 used (same `_generate_eval_scenarios` seeds), per-
incident rows preserved, never collapsed to pooled means only.

## 10. Calibration -- all three predictor seeds (the second confound this milestone resolves)

`B_DEPTH_AWARE`, `alpha=0.1` (unchanged). For EACH of the three predictor
seeds (20260814, 31874, 20260815) INDEPENDENTLY:

- Arm A: fit on that seed's own golden-reference calibration split (150
  scenarios), exactly the M1-M8.7 pattern -- Arm A has three distinct
  checkpoints (one per seed), so this yields three distinct Arm-A
  calibrators, unlike M9.0's evaluation script (which fit Arm A's calibrator
  once, on the representative-seed checkpoint only, and reused it for all
  three seeds' rows -- an inconsistency M9.0a corrects for calibration
  specifically, since "calibration seed robustness" is this milestone's own
  named uncertainty; M9.0a does NOT change the KNOWN-NETWORK LOCALIZATION
  metrics methodology, which continues to evaluate each seed's own
  checkpoint against the shared depth grid exactly as M9.0 did).
- Arm B2: fit on that seed's own pooled golden-reference + branched-loop +
  loop-grid calibration split (150 total, 50/family), via
  `run_m7_topology._fit_model_calibrator`'s existing pooling pattern
  (imported/reused, not reimplemented) -- same construction M9.0 used for
  its own (representative-seed-only) Arm-B calibrator.

Neither arm's calibrator, for any seed, ever sees coastal-branch/tree-branch/
dense-loop calibration examples. No redesign: no family-conditioned
calibration as a new method, no M7B adaptive calibration, no alpha change, no
quantile tuning, no fitting on unseen families.

For EACH seed report known-family marginal/EARLY/MID/MATURE coverage, mean
and median candidate-set size, singleton rate. Then report mean/min/max
across the 3 seeds. Frozen per-seed calibrators are additionally evaluated
(never refit) on the three unseen families and reported as
`UNSEEN_TOPOLOGY_CALIBRATION_TRANSFER`, per seed.

## 11. Calibration promotion interpretation (frozen, restated verbatim from the milestone instructions)

The frozen safety floor remains known-family marginal coverage >= 0.85, not
weakened. `CALIBRATION_ROBUST_PASS` only if all 3 seeds >= 0.85.
`CALIBRATION_SEED_UNSTABLE` if exactly 2/3 pass. `CALIBRATION_SYSTEMATICALLY_
INCOMPATIBLE` if 0 or 1 of 3 pass. EARLY conditional undercoverage is
preserved as a documented limitation regardless of the marginal-coverage
classification.

## 12. Known-network guardrails, primary topology-gain bar, statistical analysis

Unmodified bars from M9.0 (`HYDROCORE_V5_M9_0_PROTOCOL.md` Sections 17-21):
EARLY top1 regression <=5pp, MATURE top1 regression <=3pp, MRR regression
<=0.03, all finite, candidate sets operationally acceptable, known-family
marginal calibration coverage >=0.85 (now evaluated per Section 11's
all-seed rule), no safety/OOD/authority regressions; pooled unseen MATURE
neural-OR-hybrid top1 gain >=+5pp with paired 90% bootstrap CI lower bound
>0 (2,000 resamples, bootstrap seed 20260815, per-incident paired rows),
improvement on >=2/3 unseen families, no unseen-family regression >5pp,
non-negative pooled MATURE direction for all three seeds, trained-family
retention confirms learning, known-network guardrails pass.

## 13. Final decision logic

Exactly the five outcomes (A-E) specified in the milestone instructions
Section 22, reproduced verbatim in `m9-0a-summary.md`'s own Decision
section, not restated here to avoid two documents drifting apart.

## 14. No locked data, no capacity change, no representation change, no calibration redesign

`locked_final_test`/`locked_topology_test` unopened throughout (checked
before/after every script). Both arms remain ~4.18M parameters. No
relative-gap representation, no cadence-diverse training, no PCGrad, no
PyTorch Geometric, no alpha change, no OOD threshold change, no novelty-aware
fusion reopening, no new calibration method, no M9.0b calibration-only study
begun inside this milestone.
