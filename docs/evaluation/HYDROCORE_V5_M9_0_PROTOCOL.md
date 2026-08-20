# HydroCore-v5 Milestone 9.0 protocol (frozen before any Arm B training or evaluation result is generated)

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` (see its own new
Section 9.6). This document freezes the M9.0 sub-protocol BEFORE Arm B is
trained or either arm is evaluated. It is not altered after seeing results.

## 0. Why this milestone exists

Milestone 7 (`reports/evaluation/hydrocore-v5/m7-summary.md`,
`scripts/hydrocore_v5/run_m7_topology.py`) compared a golden-reference-only
model ("CURRENT") against a model trained on golden-reference +
branched-loop + loop-grid ("EXPANDED"). Because `CausalPrefixDatasetView`
requires one `SignatureLibrary` per view (`scenario_to_prefix_example`
raises if `junction_ids != signature_library.node_ids`) and
`collate_scenarios` requires shape-consistent micro-batches, EXPANDED's
three families could not be mixed within a batch. M7 worked around this
with a SEQUENTIAL three-phase curriculum (golden-reference epochs 0-6,
branched-loop epochs 7-13, loop-grid epochs 14-19, chained via
`Trainer.fit(resume_from=...)`). M7 found no robust unseen-topology
generalization gain, but this result is confounded by family
recency/catastrophic forgetting/optimizer-state dominated by the last
family trained. M9.0 retests the same underlying question -- does topology
diversity in training help genuinely unseen topologies? -- using TRUE
interleaving (family-pure micro-batches, cross-family gradient
accumulation within one optimizer step) so family order cannot confound
the result. This document does not revise or erase M7's own conclusions
(`reports/evaluation/hydrocore-v5/m7-summary.md` remains as-is); M9.0's
own comparison is Arm A vs Arm B under this protocol, with M7 quoted only
as historical context (Section 13 below).

M9.0 does not revisit any Milestone-8.7 representation decision. The
selected representation, `AGE_FIX_ONLY`
(`unobserved_age_sentinel="fixed"`), is frozen input to both M9.0 arms,
unchanged (`reports/evaluation/hydrocore-v5/m8-7-closure.json`,
`final_decision.selected_representation_for_future_experiments`).

## 1. Arms

**Arm A -- SINGLE_FAMILY_CONTROL.** Representation `AGE_FIX_ONLY`,
architecture `HydroCore.from_variant("small", ...)` with
`run_m8_7_arm.SHARED_MODEL_CONFIG` (~4,182,612 parameters), trained on
golden-reference only, `configs/training-v5-causal.yaml` unmodified
(`TrainingConfig.from_yaml(..., require_complete_task_weights=True)`),
`ARM_POLICIES["A"]` (full-history control) causal-prefix depth policy.

**REUSES the existing Milestone-8.7 `AGE_FIX_ONLY` checkpoints
(seeds 20260814, 31874, 20260815) verbatim, zero retraining**, per the
comparability verification in Section 2 below -- this IS "the matched
control," not an approximation of it: identical architecture, identical
representation, identical optimizer/schedule/task-weights, identical
600-golden-reference-scenario budget, identical checkpoint-selection rule
(lowest validation loss at full-history depth=25).

**Arm B -- INTERLEAVED_MULTI_FAMILY.** Representation `AGE_FIX_ONLY`,
IDENTICAL architecture to Arm A. Trained on the SAME three families M7's
EXPANDED arm trained on -- golden-reference, branched-loop, loop-grid --
using the SAME scenario-generation code M7 used
(`run_m7_topology._family_scenario_pool`, `run_m7_topology.SEED_BASES`,
`run_m7_topology.TRAINED_FAMILIES` network loaders, `TRAIN_PER_FAMILY=200`,
`VALIDATION_PER_FAMILY=(33,33,34)`, `CALIBRATION_PER_FAMILY=50`), imported
and reused unmodified -- so Arm B's training corpus is the SAME 600
train/100 validation/150 calibration scenarios (200/33-or-34/50 per family)
M7's EXPANDED arm would have used, and the only material difference from
M7's EXPANDED arm is the FAMILY SCHEDULE (interleaved here vs sequential
3-phase there), exactly as this milestone requires. Trained fresh, 3
seeds, via a new script (`scripts/hydrocore_v5/run_m9_0_arm_b.py`) because
no interleaved-family training loop exists anywhere in this repository
(verified by search; `Trainer._train_epoch` is built around exactly one
dataset/`DataLoader` with no multi-dataset extension point).

Neither arm changes model capacity (both remain the ~4.18M "small"
variant), the causal-prefix depth policy, task weights, PCGrad
(`pcgrad_enabled=false`, unchanged), or the temporal representation.

## 2. Arm A checkpoint-reuse verification (recorded before any M9.0 result)

| requirement | Arm A (M8.7 AGE_FIX_ONLY) | match |
|---|---|---|
| architecture | `HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)`, 4,182,612 params | identical to Arm B |
| representation | `unobserved_age_sentinel="fixed"`, `include_relative_gap_feature=False` | identical to Arm B |
| optimizer config | `configs/training-v5-causal.yaml` via `TrainingConfig.from_yaml(..., require_complete_task_weights=True)`, only `seed`/`gradnorm_log_every_n_batches` overridden in memory | identical to Arm B |
| scenario budget | 600 train / 100 validation / 150 calibration, golden-reference only | matches Arm B's 600/~100/150 TOTAL (Section 6) |
| optimizer-step accounting | recorded per-seed in `reports/evaluation/hydrocore-v5/m8-7-runs/AGE_FIX_ONLY-seed*.json` (`training_summary.global_steps`) | recorded, compared transparently against Arm B's own (Section 6 -- NOT forced equal, see rationale there) |
| checkpoint-selection rule | lowest validation loss at full-history (depth=25) evaluation, `early_stopping_patience=5` | identical rule applied to Arm B (Section 4) |
| epochs | 20 (`maximum_epochs` stop reason, all 3 seeds) | identical (Arm B also runs 20 epochs) |
| train/validation manifests suitable for comparison | both arms evaluated on the SAME frozen `development_holdout` set (`causal_prefix.build_scenario_pool("development_holdout", ...)`); training-scenario IDENTITY need not match between arms -- the causal question is single-family-vs-interleaved-multi-family training, not scenario-instance parity within the shared golden-reference family | valid for M9.0's comparison |

No material mismatch found. Arm A checkpoints are reused verbatim; their
hashes (`checkpoint_sha256`) are copied into `m9-0-results.json` unchanged
from `m8-7-runs/AGE_FIX_ONLY-seed*.json`.

## 3. Topology-family splits

Trained families (both TRAINED families are used by Arm B only; Arm A
never sees branched-loop/loop-grid): golden-reference, branched-loop,
loop-grid -- `run_m7_topology.TRAINED_FAMILIES`.

Unseen development families (never enter gradient updates, checkpoint
selection, hyperparameter tuning, or calibration fitting, for EITHER arm):
coastal-branch, tree-branch, dense-loop -- `run_m7_topology.UNSEEN_FAMILIES`,
reused unmodified (same network loaders/generators M7 used, including the
two locally-built topologies `build_tree_branch_network`/
`build_dense_loop_network`).

`locked_topology_test` / `locked_final_test` are never used, at any point,
for either arm, for training, evaluation, or calibration -- verified via
`hydroswarm.evaluation.live_robustness.locked_test_opened` before and
after every M9.0 script, exactly as every prior v5 milestone verifies.

Split policy: physical scenarios are split BEFORE causal-prefix views are
generated (`_family_scenario_pool` assigns `split`/`network_family` at
scenario-generation time, before any `CausalPrefixDatasetView` exists);
derived prefixes of one scenario stay in that scenario's split; the
topology-family split is at the FAMILY level (every network belonging to
one family is entirely inside one role -- trained or unseen -- never
mixed), matching `HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` Section 1.4.

## 4. Interleaved training design (Arm B, frozen before training)

One optimizer update = one micro-batch from EACH of the three trained
families, in FIXED order (golden-reference, branched-loop, loop-grid),
every single step -- the simplest form of Section 4's canonical structure
from the milestone instructions, chosen deliberately over a shuffled
multi-step cycle because it gives the smallest possible (zero) contiguous
single-family run: no two consecutive gradient contributions ever come
from the same family.

Per optimizer step:

```
optimizer.zero_grad(set_to_none=True)
for family in (golden-reference, branched-loop, loop-grid):
    batch = next micro-batch from family's own stage-filtered DataLoader
    output = model(batch.inputs)                      # SAME model/params
    result = compute_multitask_loss(output, batch.targets, task_weights=...)
    (result.total / 3).backward()                      # accumulate, no zero_grad between
clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
optimizer.step()
scheduler.step()
global_step += 1
```

This mirrors `Trainer._train_epoch`'s own existing accumulation idiom
exactly (`(result.total / gradient_accumulation_steps).backward()` called
repeatedly before one `clip + step + scheduler.step() + zero_grad`), with
the accumulation window replaced by "one micro-batch per family" (3)
instead of "N consecutive micro-batches of one family" (4, the config
default) -- a necessary, predeclared consequence of family-pure
micro-batching, not an arbitrary optimizer change. `(result.total /
3).backward()` (mean-of-families per step) is the direct analogue of
Trainer's own `/ gradient_accumulation_steps` normalization.

One shared `torch.optim.AdamW` instance and one shared `LambdaLR`
scheduler (`hydroswarm.training.trainer._scheduler`, reused unmodified)
receive every family's gradient contribution before any `.step()` call --
there is exactly one set of model parameters and one optimizer state for
all three families (Section 7's regression tests, below, assert this
directly).

Curriculum: `CurriculumSchedule.progressive()` (unchanged), applied
IDENTICALLY per family every epoch (`family_view.stages_through(stage)`,
same `unobserved_age_sentinel="fixed"` kwargs-forwarding fix M8.7
established, verified for all three per-family views). Because each
family's 200-scenario pool is generated by the SAME
`stage = stages[index % len(stages)]` round-robin
(`run_m7_topology._family_scenario_pool`) over the same 5-stage enum, all
three families' curriculum-filtered counts are equal at every epoch by
construction (verified directly against a real pool before training,
Section 25) -- so no family is ever starved relative to another within an
epoch, and the "no family may dominate long contiguous training phases"
requirement is satisfied both across steps (fixed round-robin order) and
across epochs (equal per-family growth).

Validation: one validation pass per family per epoch (mirroring
`Trainer._validate`, called three times), averaged with equal (1/3, 1/3,
1/3) weight into one scalar used for checkpoint-selection/early-stopping,
exactly analogous to Arm A's single-family validation loss.
`early_stopping_patience=5` (config default, unchanged) applies to this
averaged loss -- NOT disabled the way M7 disabled it for EXPANDED, because
Arm B's validation loss is computed consistently (same three families,
equally weighted) every single epoch, so "stale epochs" remains a
meaningful convergence signal here (unlike M7's cross-phase-boundary case,
where validation loss was measured against a DIFFERENT family's
validation set than produced the previous best).

Checkpoint/export: `save_checkpoint` every `checkpoint_every_epochs=1`
epoch (config default), best-model export on improvement of the averaged
validation loss, final `TrainingSummary` fields recorded identically to
`Trainer.fit()`'s own (`epochs_completed`, `global_steps`,
`best_validation_loss`, `best_epoch`, `stopped_early`, `stop_reason`,
`export_path`, `export_sha256`).

## 5. Family weighting

Equal: golden-reference = branched-loop = loop-grid = 1/3, both in
TRAINING (one micro-batch per family per step, Section 4) and in
CALIBRATION fitting (Section 10). No compelling numerical reason to
deviate was found before training (each family's 200-scenario pool and
identical stage distribution already guarantee equal, deterministic
per-family volume with no implicit dominance).

Actual gradient-bearing sample counts per family are recorded in
`reports/evaluation/hydrocore-v5/m9-0-results.json`
(`arm_b.per_seed.<seed>.family_exposure_counts`): each family contributes
exactly `optimizer_steps_this_epoch x batch_size(2)` examples per epoch,
identical across the three families by construction.

## 6. Total training budget accounting (frozen, not adjusted after training)

| | Arm A | Arm B |
|---|---|---|
| total physical train-scenario exposure per full epoch pass | 600 (golden-reference) | 600 (200 golden-reference + 200 branched-loop + 200 loop-grid) |
| epochs | 20 | 20 |
| batch size (per micro-batch) | 2 | 2 (per family, per micro-batch) |
| accumulation window per optimizer step | 4 micro-batches, all golden-reference (config default) | 3 micro-batches, one per family (Section 4, frozen) |
| optimizer steps/epoch once curriculum saturates (epoch >= 4) | `ceil(600/2/4) = 75` | `ceil(200/2/1) = 100` (per family; identical across all 3 families) |
| static scheduler `total_steps` estimate (mirrors `Trainer.__init__`'s own formula, `ceil(N/batch_size/accum) * epochs`) | `ceil(600/2/4) * 20 = 1500` | `ceil(200/2/1) * 20 = 2000` |
| validation/checkpoint cadence | every epoch (`checkpoint_every_epochs=1`) | every epoch, identical |

Arm B's per-epoch optimizer-step count (100) is HIGHER than Arm A's (75)
once the curriculum saturates, by construction: replacing Arm A's
4-micro-batch single-family accumulation window with Arm B's mandated
1-micro-batch-per-family x 3-family window (Section 4) necessarily changes
how many optimizer steps one full epoch pass produces, even though both
arms traverse their full family-scoped scenario pool exactly once per
epoch and train for the same 20 epochs. This is recorded transparently,
not hidden or forced into artificial equality -- Section 6 of the
milestone instructions lists "total physical training-scenario exposure"
and "epoch-equivalent training budget" as the primary matched dimensions
(both equal: 600 scenarios/epoch, 20 epochs), and explicitly permits
"approximately N total exposures" rather than exact optimizer-step parity.
Both arms' actual `global_steps` are recorded per seed in
`m9-0-results.json` and reported in `m9-0-summary.md` without adjustment.

## 7. Signature-library handling

Three independent `SignatureLibrary` instances for Arm B, one per family,
each fit via `fit_pool_signature_library(family_pools["train"])`
(unmodified), keyed by family name in a `dict[str, SignatureLibrary]`. Each
family's `CausalPrefixDatasetView` (and its `stages_through()` return
value) is constructed with exactly that family's own library; no library
is ever shared or reused across families other than trivially avoiding
redundant construction within one family. `scenario_to_prefix_example`'s
own hard assertion (`junction_ids != signature_library.node_ids ->
ValueError`) is the structural guarantee against silent cross-family
library mixing -- if wiring were ever wrong, training would crash loudly
(a shape/identity mismatch), not silently mislearn. Regression tests
(Section 25) additionally assert this positively (each family's examples
verifiably use that family's own library) rather than relying on the
crash-on-misuse property alone.

## 8. AGE_FIX_ONLY propagation

Both arms use `unobserved_age_sentinel="fixed"`,
`include_relative_gap_feature=False` throughout: initial dataset
construction, every family's `stages_through()` return value every epoch
(the exact M8.7 regression -- verified for all three of Arm B's families,
not just golden-reference), validation views, and evaluation views. Tests
in Section 25 assert this for Arm B's three families explicitly (M8.7's
own regression test already covers the single-family case for Arm A/its
reused checkpoints).

## 9. Training policy (unchanged from the frozen small-model recipe)

Representation `AGE_FIX_ONLY`; architecture HydroCore small (~4.18M);
causal training policy `ARM_POLICIES["A"]` (full-history control); task
weights and PCGrad exactly as `configs/training-v5-causal.yaml`
(`pcgrad_enabled=false`); optimizer AdamW,
`lr=0.0003`, cosine schedule, `warmup_steps=10`,
`gradient_clip_norm=1.0`, `weight_decay=0.01`, `deterministic=True`. Only
topology training distribution/scheduling differs between arms, per
Section 10 of the milestone instructions.

## 10. Seeds

Three seeds for both arms: 20260814, 31874, 20260815. Arm A reuses the
existing three M8.7 AGE_FIX_ONLY checkpoints (Section 2). Arm B trains
fresh on all three. No seed is dropped, substituted, or re-rolled.

## 11. Standard known-network evaluation

Both arms, golden-reference `development_holdout`
(`build_scenario_pool("development_holdout", network_loader=build_wntr_network)`),
depths 1, 2, 3, 4, 6, 12, 25 (`CAUSAL_PREFIX_DEPTHS`, unchanged). Metrics:
neural top1/top3/MRR/NLL/entropy, plus hybrid (hydroswarm.inference.fusion's
real, unmodified `fuse_source_probabilities`/`dynamic_classical_trust`)
top1/top3/MRR. Aggregated EARLY (1-3), MID (4-6), MATURE (12, 25), matching
every prior v5 milestone's own convention.

## 12. Trained-family retention (Arm B only, labeled TRAINED_FAMILY_GENERALIZATION)

Arm B evaluated on held-out development scenarios from branched-loop and
loop-grid (via `run_m7_topology._generate_eval_scenarios`, unmodified,
`EVAL_MAX_SOURCES=4 x EVAL_SEED_REPEATS=4` per family, full depth grid).
Reported separately from the primary unseen-topology result and clearly
labeled `TRAINED_FAMILY_GENERALIZATION` -- these families were trained on
by Arm B, so this is a sanity check that interleaving actually taught the
model the added families, not generalization evidence.

## 13. Primary unseen-topology evaluation

Both arms, both evaluated on IDENTICAL held-out incidents (same
`_generate_eval_scenarios` seeds) from coastal-branch, tree-branch,
dense-loop, full depth grid, all three seeds. Per-incident rows preserved
in `m9-0-topology-generalization.json` -- never collapsed to pooled means
only. Aggregated EARLY/MID/MATURE per family, and pooled across all three
unseen families.

## 14. Comparison with M7 (historical context only, not the primary causal claim)

`reports/evaluation/hydrocore-v5/m9-0-summary.md` includes a section
comparing M7's SEQUENTIAL EXPANDED result against M9.0's INTERLEAVED
Arm B result, explicitly marked as NOT a perfectly paired comparison
(different seed-corpus vintage for CURRENT vs Arm A's M8.7-representation
checkpoints, different evaluation-depth grids -- M7 used `(3, 25)` only,
M9.0 uses the full 7-depth grid). The primary causal claim of M9.0 remains
Arm A vs Arm B under this document's own matched protocol; M7 is reported
only to note whether interleaving qualitatively changes M7's own observed
pattern (trained-loop-grid gain, tree-branch regression, weak dense-loop
gain).

## 15. Calibration

`B_DEPTH_AWARE`, `alpha=0.1` (unchanged), fit per predictor
(`SplitConformalCalibrator.fit`, unmodified) on:

- Arm A: golden-reference calibration split only (150 scenarios, same as
  every M1-M8.7 script).
- Arm B: the POOLED calibration split across golden-reference,
  branched-loop, loop-grid (150 total, 50/family), `network_id =
  f"{family}:{bucket}"`, exactly `run_m7_topology._fit_model_calibrator`'s
  existing pooling pattern (imported/reused, not reimplemented).

Neither arm's calibrator ever sees coastal-branch/tree-branch/dense-loop
calibration examples -- by construction, those `network_id`s are never a
key in the fitted calibrator, so `SplitConformalCalibrator.selection()`
always falls back past `NETWORK_SPECIFIC` for unseen-family rows. This is
verified directly (`calibration_applicable == False` for every unseen-family
row) and reported, not assumed. The known EARLY conditional-coverage
limitation carried forward from M8.7 is preserved and reported explicitly
for both arms -- not silently resolved.

Frozen calibrators are evaluated (never refit) on the unseen development
families and reported as `UNSEEN_TOPOLOGY_CALIBRATION_TRANSFER`.

## 16. OOD / fusion

The existing, unmodified `hydroswarm.inference.fusion.
fuse_source_probabilities` / `dynamic_classical_trust` only. M7's
LOCAL-ONLY `novelty_aware_classical_trust` / `NOVELTY_TRUST_BOOST` /
Part-B gating logic is NOT reused, NOT reopened, and NOT re-run in M9.0 --
the milestone instructions explicitly forbid this. No OOD threshold,
fusion threshold, or trust formula is modified.

## 17. Statistical analysis

Paired (same development incidents/seeds across arms) comparison,
Arm B minus Arm A, for:

- pooled unseen MATURE neural top1
- pooled unseen MATURE hybrid top1
- pooled unseen overall MRR
- pooled unseen EARLY top1

Paired bootstrap: 2,000 resamples, 90% interval, bootstrap seed 20260815
(matching the M8.7 closure's own bootstrap-seed convention), resampling
per-incident (paired) rows with replacement. Per-family differences (not
only pooled) are also computed and reported. A point estimate alone is
never treated as evidence of superiority.

## 18. Predeclared regression guardrails (frozen, not relaxed after results)

INTERLEAVED_MULTI_FAMILY (Arm B) is not promoted if any hold:

- known-golden EARLY top1 regression > 5pp vs Arm A
- known-golden MATURE top1 regression > 3pp vs Arm A
- known-golden MRR regression > 0.03 vs Arm A
- Arm B fails to demonstrate it learned branched-loop/loop-grid (i.e.
  TRAINED_FAMILY_GENERALIZATION MATURE top1 is not clearly above chance
  and not clearly below Arm A's known-golden MATURE top1 by more than the
  golden guardrail itself, or any non-finite output occurs there)
- numerical instability or non-finite output anywhere (either arm)
- marginal known-family calibration coverage materially worsens (falls
  below `0.90 - 0.05 = 0.85`, the same frozen bar M8.7 used)
- candidate sets become operationally useless (mean size >
  `0.5 x node_count` for the network in question, same frozen bar)
- OOD/fail-closed behavior changes, authority semantics change, or any
  locked data is touched

## 19. Topology-diversity promotion gate (frozen, restated verbatim from the milestone instructions)

`PROMOTE_INTERLEAVED_TOPOLOGY_RECIPE` only if ALL hold: known-network
guardrails pass; pooled unseen-family MATURE top1 improves by >= +5
percentage points (hybrid OR neural, with the other representation not
materially regressing); the paired 90% bootstrap interval for the primary
pooled unseen comparison has lower bound > 0; improvement on at least 2 of
3 unseen families with no unseen-family regression worse than 5pp on the
primary MATURE metric; directionally consistent across all three seeds;
trained-family performance confirms the multi-family corpus was actually
learned; OOD/calibration/safety remain valid/fail-closed. Otherwise:
`INTERLEAVED_TOPOLOGY_GAIN_NOT_ROBUST` (point estimate improves but
statistical/family-consistency criteria fail),
`KEEP_SINGLE_FAMILY_RECIPE` (essentially no gain), or
`INTERLEAVED_TOPOLOGY_TRAINING_REJECTED` (known-network regression). No
alternative training schedule is tried within M9.0 regardless of outcome.

## 20. No locked data, no capacity change, no representation change

`locked_final_test`/`locked_topology_test` unopened throughout (checked
before/after every script). Both arms remain ~4.18M parameters -- no
width/depth change. No relative-gap representation, no cadence-diverse
training, no PCGrad, no PyTorch Geometric, no alpha change, no OOD
threshold change, no novelty-aware fusion reopening.
