# HydroCore-v5 Milestone 10.2 Scout supervision/representation refit protocol (frozen BEFORE any Level-A training result is inspected)

Amends nothing in `docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md` or `HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md`,
both of which remain frozen and unmodified. Builds directly on
`HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md`'s finding (Scout is `PRESENT_BUT_UNSUPERVISED` in every
canonical M9.6 checkpoint) and on this task's Parts 2-4 (`hydroswarm.training.gradient_coverage`,
`hydroswarm.training.scout_training_state`, `hydroswarm.training.scout_targets`).

**This document is frozen and hashed BEFORE Level A executes.** No value in it may be changed after Level-A
results are inspected. Any genuine implementation defect found after freezing must be fixed under the frozen
DESIGN (population, gate, allowlist) unchanged; a defect that would require changing the scientific design
itself is reported, not silently patched.

## 1. Scope and authorization

This protocol governs Level A (frozen backbone, Scout-specialist-side training) and, only if Level A's frozen
gate legitimately fails for a representation-capacity reason, Level B (one predeclared, bounded partial
shared-backbone unfreeze). Level C (full/joint retraining) is explicitly out of scope and forbidden; if the
evidence indicates it is required, this task stops and reports
`M10_2_SCOUT_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED` without performing it. The true M10.2
learned-vs-deterministic Scout scientific comparison is explicitly NOT performed by this protocol or its
execution -- a separately authorized later task.

## 2. Teacher checkpoint (frozen, immutable)

Every Level-A/B run starts from one of the three canonical M9.6 `ARM_B_M9_6` `FINAL_STEP_1350` checkpoints
(seeds `20260814`, `31874`, `20260815`; SHA-256 `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`
/ `527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332` /
`b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c`). These files and hashes are never
overwritten by this protocol's execution; every resulting refit checkpoint records its parent teacher SHA-256
explicitly (Section 9).

## 3. Training-state and target schema (frozen, versioned, hashed)

- **Training-state schema**: `hydroswarm.training.scout_training_state.SCOUT_TRAINING_STATE_SCHEMA_VERSION =
  "scout-training-state-v1"`. Channel wiring: `sensor_mask`/`quality_features` (already-sampled + revealed
  evidence, via a synthetic `SensorSeries` fed through the real `HydraulicFeatureBuilder` path),
  `classical_prior` (posterior, recomputed via `causal_prefix._prefix_classical_prior` over the augmented
  series), `role_features[0]` = `sampling_round / MAXIMUM_SAMPLES_BOUND`, `role_features[1]` =
  `sample_budget_remaining / MAXIMUM_SAMPLES_BOUND`, `residual_features[..., 0]` = per-node accessibility
  (`1.0` junction / `0.0` reservoir-tank). No new `HydroCore` constructor parameter -- confirmed by
  `tests/scientific/test_m10_2_scout_refit_corpus.py::
  test_role_and_residual_channels_are_populated_nowhere_else_in_the_codebase`.
- **Target schema**: `hydroswarm.training.scout_targets.SCOUT_TARGET_SCHEMA_VERSION = "scout-target-v1"`.
  Targets are `hydroswarm.training.scout_trajectory._scout_step_targets`'s governed tensors, built from
  `hydroswarm.training.scout_labels.generate_scout_label`'s real, offline, exact-simulation-derived classical-EIG
  recommendation -- never fed back into the paired training-state batch (structural leakage guard: neither
  function accepts a label/target object as an input parameter).
- Both schema version strings are distinct from `hydroswarm.evaluation.scout_state.SCOUT_EVAL_STATE_SCHEMA_VERSION`
  (`"scout-eval-state-v1"`, evaluation-only) and from
  `hydroswarm.training.checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION` (`"scout-state-v1-unbuilt"`, the still-
  accurate training-CORPUS-layout placeholder for the *original* M9.6 checkpoints, unchanged by this protocol).

## 4. Populations (development-only, disjoint from every locked split and from the future true-M10.2 population)

- **Family scope**: `golden-reference` only. Deliberately bounded, single-family pilot scope (the same scope
  M8.7's `CURRENT_CONTROL`/`ARM_A` arm used before M9.0 widened to multi-family) -- chosen for tractability
  within this task's session, not as a result-driven narrowing (frozen before any Level-A result exists). If
  Level A is accepted, a future, separately authorized amendment may widen family coverage before the true
  M10.2 comparison; this protocol does not presuppose that decision.
- **Seed namespace**: role `scout_refit_m10_2`, disjoint from every other seed range in the repository (verified
  by `grep` over `1_200_000_000`..`1_200_999_999`: zero hits before this document, matching M10.1's own
  disjointness-verification convention).
  - **Train**: `seed_base=1_200_000_000`, `count=250`, `source_round_robin=True`.
  - **Validation** (Level-A gate +, if triggered, Level-B gate): `seed_base=1_200_100_000`, `count=100`,
    `source_round_robin=True` -- offset by `100_000` from train, guaranteeing zero seed overlap.
- **Split discipline**: one physical scenario belongs to exactly one split (train xor validation) by
  construction (disjoint seed ranges); every Scout-trajectory step derived from one scenario stays in that
  scenario's split (trajectory building never crosses scenarios). `locked_final_test`/`locked_topology_test`
  are never accessed by this protocol or its execution -- asserted before and after every phase.
- **Depth**: fixed `depth=25` (this family's existing `MATURE_DEPTHS` full-history value -- `DEPTHS=(1,2,3,4,6,
  12,25)`, reused unmodified from `m10_common`/`m9_4_common`, not invented for this protocol). Scout-round
  evidence (grab samples) is layered ON TOP of this fixed full-history base via the synthetic-sensor mechanism
  (Section 3) -- depth itself is never swept or tuned.
- **Trajectory bound**: `maximum_samples=3` (within `scout_trajectory.MAXIMUM_SAMPLES_BOUND=5`'s existing
  product cap; a fixed, non-tuned pilot choice). `noise_scale_mg_l=0.5` (matches
  `hydroswarm.training.scout_labels.generate_scout_label`'s own existing default; not tuned).

## 5. Level-A trainable parameter allowlist (frozen, exact, forward-graph-traced)

Traced from `HydroCore.forward()` directly (not assumed from the task's own suggested candidate list): the
frozen Scout-state contract's `role_features`/`residual_features` channels are projected into the shared hidden
state by `self.role_projection`/`self.residual_projection` **before** `self.backbone` runs (`hydroswarm/model/
core.py` lines ~985-998, backbone loop at ~1000) -- i.e. the new round/budget/accessibility signal passes
through the (frozen) backbone before ever reaching the Scout heads. Neither projection layer ever received a
real gradient during M9.6 training either (nothing populated `role_features`/`residual_features` then), so
leaving them frozen-at-random-init would confound "the frozen backbone's representation is insufficient" with
"the injection layer for the brand-new input signal was never trained" -- an uninterpretable Level-A result.
Both are small, single `nn.Linear` layers structurally dedicated to exactly these two new channels, not part of
the shared multi-block transformer backbone itself, so including them preserves "hydraulic/Sentinel backbone
remains frozen" in substance, not merely in the letter of "only the four Scout heads."

Exact 18 trainable parameters (verified present via `model.named_parameters()` against a real, freshly
constructed M9.6-configuration model):

```
role_projection.weight, role_projection.bias
residual_projection.weight, residual_projection.bias
sample_node_head.network.0.{weight,bias}, sample_node_head.network.1.{weight,bias}
information_gain_head.0.weight, information_gain_head.1.{weight,bias}
candidate_reduction_head.0.weight, candidate_reduction_head.1.{weight,bias}
should_continue_sampling_head.network.0.{weight,bias}, should_continue_sampling_head.network.1.{weight,bias}
```

Every other parameter (encoders, backbone transformer blocks, `final_norm`, `prior_projection` -- already
trained via the wired `classical_prior` channel -- every Sentinel/OOD/Strategist head) is frozen
(`requires_grad=False`), mechanically asserted before training starts and after training ends.

## 6. Optimizer / schedule (frozen, not tuned, not swept)

Adam, `lr=1e-3`, `weight_decay=0.0`, no scheduler (constant LR). `batch_size=8`. `epochs=20` (reusing M9.6's own
"20 epochs" convention, not invented). **Checkpoint selection: final epoch** (`FINAL_STEP` policy, matching
M9.7A's own already-frozen anti-cherry-picking rule -- never best-validation). No early stopping. No
hyperparameter search of any kind; these values are fixed once, here, before any training run.

## 7. Gradient-coverage requirement (frozen; must pass before any competence metric is interpreted)

`hydroswarm.training.gradient_coverage.compute_gradient_coverage`/`require_gradient_coverage` must report
`passed=True` for all four Scout tasks (`sample_node`, `information_gain`, `candidate_reduction`,
`should_continue_sampling`) against the Section 5 allowlist, on a representative real training batch, before
Level-A's own competence gate (Section 8) is evaluated at all. A gradient-coverage failure is an
implementation/data defect, not a representation-capacity finding, and blocks interpretation of any competence
number until fixed under this same frozen design.

## 8. Level-A representation-sufficiency gate (frozen BEFORE any Level-A result is inspected)

Evaluated on the **validation** population only (Section 4), using the **final-epoch** Level-A checkpoint.
Bootstrap procedure reused unmodified from the M10 protocol's own convention (Section 7 of
`HYDROCORE_V5_M10_PROTOCOL.md`): 2,000-resample, 90% CI, bootstrap seed `20260819`.

`LEVEL_A_REPRESENTATION_SUFFICIENT` requires ALL of:

1. **Gradient coverage** (Section 7) passes for all four tasks.
2. **Support**: at least 20 validation examples with `sample_node_mask=True` (a real recommendation existed),
   at least 20 with `should_continue_sampling` defined (always true by construction), both `should_continue_sampling`
   classes present at least once.
3. **`sample_node`**: top-1 accuracy among `sample_node_mask=True` validation examples has a 90% bootstrap CI
   whose LOWER bound exceeds a naive uniform-random-eligible-candidate baseline's own top-1 accuracy on the
   SAME examples (paired resampling). This is a comparison against a NAIVE baseline, never against
   `HydroScout.deterministic_fallback`'s operational behavior -- deterministic-Scout superiority is explicitly
   reserved for the true M10.2 comparison, never used to gate Level A.
4. **`information_gain`**: MSE against the real target is lower than a constant-train-mean-prediction baseline's
   MSE, AND the Spearman rank correlation between predicted and target values (over `information_gain_mask=True`
   positions) has a 90% bootstrap CI excluding zero on the positive side.
5. **`candidate_reduction`**: same treatment as (4).
6. **`should_continue_sampling`**: accuracy's 90% bootstrap CI lower bound exceeds the majority-class baseline's
   accuracy on the same validation examples.
7. **No NaN/Inf** in any prediction or metric.

If ALL seven hold: `M10_2_SCOUT_REFIT_A_ACCEPTED`. Training stops; Level B is NOT run merely because it is
authorized.

If (1), (2), or (7) fails: this is an implementation/data defect. Report distinctly; do not label it "full
retrain required" and do not silently loosen this gate to make it pass.

If (1), (2), (7) all hold but ANY of (3)-(6) fails: proceed to Level B (Section 9), since this is consistent
with a genuine frozen-representation limitation rather than a defect.

## 9. Level B (frozen scope, defined here BEFORE Level-A results, run only if Section 8's escalation condition fires)

**Scope**: Section 5's 18 Level-A parameters **plus** `backbone[3]` (the LAST of the 4 `LatentHydraulicBlock`
modules in the `small` variant's backbone -- `len(model.backbone) == 4`, confirmed against a real model
instance) **plus** `final_norm.weight`. No other parameter. Warm-starts from the SAME canonical M9.6 teacher
checkpoint used for Level A (never from Level A's own outcome-tuned weights) -- independent comparability over
hidden sequential tuning, per this task's own explicit instruction. Same optimizer/schedule/population/gate
methodology as Level A (Sections 6/8), evaluated with the SAME bootstrap procedure.

**Level-B promotion additionally requires** (both, alongside Section 8's Scout-competence criteria evaluated
against Level B's own checkpoint):

- **A**: Level B's own Scout representation-sufficiency comparison (Section 8, criteria 3-6) materially
  improves over Level A's own validation numbers under the same paired-bootstrap procedure (Level B's 90% CI
  lower bound exceeds Level A's own point estimate for at least the criteria Level A failed).
- **B**: M9 preservation -- development/validation-only (never locked) re-evaluation of all nine
  `TRAINED_WITH_REAL_TARGETS` Sentinel tasks (Section 1 of `HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md`)
  plus source-posterior/calibration-coverage behavior, comparing Level B's checkpoint against the unmodified
  M9.6 teacher on the SAME development population, under the existing M9-frozen acceptance bounds (coverage
  floor 0.85, alpha 0.1, unchanged). No calibration refit is performed under any circumstance in this task.

If Level B fails Scout competence, or regresses M9 preservation, or would require recalibration/broader
unfreezing to pass: `M10_2_SCOUT_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED`. Level A's own (non-B) checkpoint is
retained as the task's best available artifact in that case, clearly labeled as not promoted.

## 10. Output governance (unaffected regardless of outcome)

Learned Scout remains runtime-disabled and non-authoritative in every case. `hydroswarm.inference.authority.
scout_certificate` is not modified, not called by anything this protocol executes, and continues to hardcode
`source="CLASSICAL_EIG"`/`AuthorityLevel.DETERMINISTIC` regardless of any refit outcome. No `runtime_enabled_outputs`
promotion occurs under this protocol.

## 11. Locked-test policy (restated)

`locked_final_test`/`locked_topology_test` are never accessed by this protocol's population, gate, or
checkpoint-selection logic. `locked_test_opened` is asserted `False` before and after every phase in every
execution artifact this protocol's execution produces.
