# Milestone 9.7A summary: M9.8 checkpoint-selection consistency correction

Narrowly scoped amendment to M9.7. No M9.8 execution occurred. No HydroCore-M development/predictive performance was observed. No M9.6 or M9.7 historical artifact was edited.

## Problem

M9.7's `m9-7-training-parity-plan.json` froze M9.8 checkpoint selection as "validation only," and `m9-7-m9-8-preregistration.json` said ARM_S should reuse "M9.6's own trained checkpoints" without specifying which of M9.6's two recorded checkpoints per seed. But M9.6 itself was **not** frozen on a best-validation checkpoint: `m9_6_common.CANONICAL_CHECKPOINT_POLICY = "FINAL_STEP_1350"`, both M9.6 training scripts export a separate canonical FINAL_STEP_1350 checkpoint after training, and `run_m9_6_evaluate.py`'s `_canonical_model` helper loads exactly that canonical export -- never `best_validation_export_path` -- for every number M9.6's own `M9_6_DECISION=A` rested on.

Left as originally worded, ARM_S (reused from M9.6, correctly canonical) and ARM_M (freshly trained under the ambiguous "validation only" text) could end up checkpointed by two different criteria -- breaking the intended exact-compute capacity comparison even though the optimizer-step budget was already correctly matched at 1350/1350.

**This is not a hypothetical risk.** `reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed20260814.json` (already-committed, immutable) records `best_epoch=12` vs. `canonical_epoch=19` for a real, already-trained seed -- genuinely different checkpoints, different SHA-256.

## Correction

Full machine-readable freeze: `m9-7a-amendment.json`.

- **ARM_S**: reuse M9.6's `canonical_export_path`/`canonical_export_sha256` (FINAL_STEP_1350) -- not `best_validation_export_path`. SHA-verified, same REUSED/REGENERATED transparency procedure as before.
- **ARM_M**: train exactly 1350 optimizer steps (unchanged), then apply M9.6's own Section-14 procedure verbatim: reload the final epoch-20/step-1350 checkpoint into a fresh model instance and export it as the canonical, promotion-authoritative checkpoint.
- **Best-validation checkpoints** may still be saved/reported for both arms, diagnostically -- exactly as M9.6's own records already do (`best_validation_export_path` alongside `canonical_export_path`) -- but MUST NOT determine any M9.8 guardrail, bootstrap comparison, or promotion decision.

## What did not change

Architecture/parameter counts, seeds, the 1350-optimizer-step budget, optimizer/scheduler/batch/exposure policy, dataset/source policy, the primary endpoint, the bootstrap procedure (2000 resamples, 90% CI, bootstrap seed 20260819), the +0.02 practical-effect threshold, the calibration policy, guardrails A-F, and `HYDROCORE_L_AUTHORIZED=False` are all unchanged and reverified against the original M9.7 artifacts (`m9-7a-amendment.json`'s `unchanged_and_reverified` block). Every M9.7 report artifact is byte-identical to its M9.7-closure state (`m9_7_artifacts_hash_snapshot`).

## Method

The protocol document `docs/evaluation/HYDROCORE_V5_M9_7_PROTOCOL.md` is amended via an appended, dated Amendment section (matching `HYDROCORE_V5_M9_1_PROTOCOL.md` Section 21's own established amendment-log convention) -- not edited in place. The original M9.7 JSON artifacts are not touched; this new `m9-7a/` directory is the authoritative correction going forward.

## Tests

`tests/scientific/test_m9_7a_checkpoint_policy.py` proves: (1) M9.6's canonical checkpoint policy is FINAL_STEP_1350 for every M9.6 training-run record; (2) best-validation and canonical checkpoints genuinely diverge for at least one real seed; (3) M9.6's own evaluation script loads the canonical checkpoint, not best-validation, for its authoritative numbers; (4) the M9.7A amendment freezes FINAL_STEP_1350 for both ARM_S and ARM_M and marks best-validation diagnostic-only; (5) every other M9.8-frozen parameter (architecture, seeds, step budget, endpoint, bootstrap, threshold, calibration, guardrails, L-authorization) is unchanged; (6) every M9.7 artifact's SHA-256 matches its frozen snapshot (immutability); (7) locked splits remain unopened before/after.

## Locked splits

`locked_test_opened`: **before = False, after = False**.

## M9_7A_DECISION: CHECKPOINT_POLICY_CONSISTENCY_FIXED

`M9_8_CAPACITY_EXPERIMENT_READY = true`. M9.8 itself is NOT started by this amendment.
