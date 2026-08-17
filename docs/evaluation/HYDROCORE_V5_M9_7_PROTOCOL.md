# HydroCore-v5 Milestone 9.7 protocol: HydroCore-M capacity preflight and M9.8 freeze

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` Section 6 (model-size experiments). This is a PRE-FLIGHT / PROTOCOL-FREEZE document only: it does not authorize, and this milestone does not run, the S-vs-M predictive capacity comparison itself. That comparison is Milestone 9.8, frozen by this document but executed only after this document (and the M9.8 predeclared design in Section 3 below) is committed.

M9.7 does not inspect any HydroCore-M development predictive result, does not open `locked_final_test`/`locked_topology_test`, does not tune HydroCore-M based on performance, and does not alter the already-frozen HydroCore-S recipe (M9.6, `M9_6_DECISION=A`, `HYDROCORE_S_STATUS=FROZEN`).

## 0. Inherited state

M9.6 froze `HYDROCORE_S_STATUS=FROZEN` with the recipe `CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING + B_DEPTH_AWARE_CALIBRATION + ALPHA_0_1 + SOURCE_REPRESENTATIVE_CALIBRATION_SUPPORT_20_PER_SOURCE` (`reports/evaluation/hydrocore-v5/m9-6/m9-6-closure.json`). M9.6 established exact optimizer-step parity, source-representative development/calibration representativeness, a statistically supported unseen-topology gain for the interleaved recipe, known-family guardrails, and full calibration passage, all with both locked splits unopened throughout. M9.7 inherits this recipe unchanged and asks only whether a scientifically fair HydroCore-M capacity experiment can be built on top of it.

## 1. Governance rules (frozen, restated from the M9.7 task instructions)

Never inspect or open `locked_final_test`/`locked_topology_test`. Do not train HydroCore-L. Do not run broad architecture search or reopen Graph ODE/CDE/SDE, Mamba/SSM, transformer alternatives, arbitrary pooling/graph redesign. Do not change AGE_FIX_ONLY representation, causal semantics, topology-training policy, conformal alpha=0.1, source-representative calibration policy, WNTR/EPANET authority, human-approval requirements, deterministic/classical fallback authority, or evidence-depth definitions merely to help HydroCore-M. Do not use `development_holdout` to choose any HydroCore-M architecture/training/calibration parameter. HydroCore-M is a CAPACITY experiment, not an architecture experiment. Historical artifacts/checkpoints/results are immutable. No checkpoint is promoted automatically; no production runtime default changes.

## 2. Frozen HydroCore-S architecture (reconstructed from source, not prose)

Reconstructed directly from `src/hydroswarm/model/core.py`/`encoders.py`/`layers.py`/`adapters.py`/`candidate_plan_encoder.py` and the actual M9.6 training scripts (`scripts/hydrocore_v5/run_m9_6_train_arm_a.py`, `run_m9_6_train_arm_b.py`, `run_m8_7_arm.py`'s `SHARED_MODEL_CONFIG`/`ARM_DEFINITIONS`), verified by direct instantiation in this environment (not estimated):

- `HydroCore.from_variant("small", use_adapters=False, temporal_feature_dim=6, quality_feature_dim=4, elapsed_time_normalization="window_relative", prior_mode="feature_only", event_control_heads=True, scout_control_heads=True, strategist_mode="candidate_conditioned", action_vocabulary_size=9, consequence_prescreening_heads=True, ood_category_head=True)`.
- `MODEL_VARIANTS["small"] = ModelVariant(d_model=192, nhead=6, dim_feedforward=576, num_layers=4, latent_tokens=64, modality_layers=1)`.
- Verified total parameter count: **4,182,612** (matches every historical M8.7/M9.0-line/M9.6 record verbatim).

Full inventory (module-by-module: encoders, backbone block structure, every output head, feature schema, AGE_FIX_ONLY semantics, checkpoint-identity/compatibility machinery, and which dimensions are legitimate capacity-scaling dimensions vs. frozen semantics): `reports/evaluation/hydrocore-v5/m9-7/m9-7-frozen-s-inventory.json`.

## 3. HydroCore-M: deterministic capacity selection

Target range: **12-16M trainable parameters**, predeclared center **~14M**, per the master protocol Section 6.

**Selection rule (frozen, applied before any predictive evaluation):**

1. Fix `num_layers=4`, `latent_tokens=64`, `modality_layers=1` identical to S -- preserve the same number and ordering of semantic stages.
2. Fix `head_dim = d_model / nhead = 32` and `dim_feedforward / d_model = 3.0`, both S's own exact ratios.
3. Sweep `d_model` over multiples of 32 (required for integer `nhead` at fixed `head_dim`) from 224 to 640.
4. Instantiate the real `HydroCore` for each candidate and count actual parameters (never a formula estimate).
5. Filter to the 12-16M band.
6. Select the point nearest the predeclared ~14M center.

This sweep produced **exactly one** in-range candidate: `d_model=352, nhead=11, dim_feedforward=1056` -> **13,919,572 parameters** (80,428 from the 14M center, 0.57%). No accuracy, loss, or any predictive signal was computed for any candidate at any point in this selection. Full enumeration, including every rejected mechanically-possible alternative and the reason each was rejected (including the pre-existing `MODEL_VARIANTS["medium"]`, rejected because it doubles `num_layers` -- a depth/stage-count change, not a pure capacity change, when pure widening alone reaches the target band cleanly): `reports/evaluation/hydrocore-v5/m9-7/m9-7-capacity-candidates.json`.

**Selected HydroCore-M**, registered as `MODEL_VARIANTS["small_v5_capacity_m"] = ModelVariant(352, 11, 1056, 4, 64, modality_layers=1)` in `src/hydroswarm/model/core.py` (additive; does not modify `"small"`/`"medium"`/`"large"`):

`HydroCore.from_variant("small_v5_capacity_m", use_adapters=False, **AGE_FIX_ONLY_model_kwargs, **SHARED_MODEL_CONFIG)` -> **13,919,572 parameters**, ratio vs. S = **3.328x**. Full config and parameter-report breakdown: `reports/evaluation/hydrocore-v5/m9-7/m9-7-selected-m-architecture.json`.

## 4. Capacity/semantic parity audit

`reports/evaluation/hydrocore-v5/m9-7/m9-7-semantic-parity-audit.json` and `m9-7-parameter-counts.json` confirm: M differs from S in `d_model`/`nhead`/`dim_feedforward` only. Every other dimension (`num_layers`, `latent_tokens`, `modality_layers`, all input feature widths, all output-head identities and class counts, normalization, activation, dropout, `prior_mode`, `incident_pooling`, `message_direction`, `strategist_mode`, every `SHARED_MODEL_CONFIG` flag, `use_adapters`) is identical to S. No new submodule type, loss, prior, pooling mode, or temporal mechanism was introduced.

**Conclusion: `M9_7_CAPACITY_ISOLATION_FAILED = FALSE`.** No unavoidable mechanical dimensional adjustment was required beyond the three width-related dimensions above.

## 5. M9.8 training recipe (frozen now, before any M9.8 development performance is observed)

Full machine-readable freeze: `reports/evaluation/hydrocore-v5/m9-7/m9-7-training-parity-plan.json`. Summary:

- Representation: `AGE_FIX_ONLY`, unchanged for M.
- Topology training: `EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING` (M9.6 `ARM_B` recipe), trained families `{golden-reference, branched-loop, loop-grid}`, unseen development families `{coastal-branch, tree-branch, dense-loop}`, unchanged for M.
- Optimizer-step budget: **S = 1350, M = 1350** (identical; M does not get more steps for being larger). Explicit compute-comparability check performed: `hydroswarm.training.trainer._scheduler` builds its cosine multiplier purely from `config`/`total_steps`, with no dependency on model width or parameter count -- no blocker found, none invented.
- Scheduler: cosine, `warmup_steps=10`, `scheduler_total_steps=1500`, identical trajectory semantics for S and M.
- Optimizer: `AdamW`, `lr=0.0003`, `weight_decay=0.01`, `gradient_clip_norm=1.0` -- unchanged, no size-dependent scaling exists in the frozen recipe to inherit.
- Batch/microbatch: `batch_size=2`, `gradient_accumulation_steps=4`, effective batch = 8, `microbatches_per_optimizer_update=4` (interleaved) -- unchanged for M. If a future M9.8 execution environment cannot fit HydroCore-M's physical microbatch at this size, `gradient_accumulation_steps` MUST be increased proportionally to preserve effective batch/optimizer-step/scheduler semantics -- a preregistered compensation rule, never a silent effective-batch-size change.
- Training depth: full-history depth=25 training only, unchanged.
- Seeds: `20260814, 31874, 20260815` (representative seed 31874, confirmation seed 20260815), unchanged, no reroll/substitution.
- Checkpoint selection: validation only, unchanged.
- Calibration: `B_DEPTH_AWARE`, `alpha=0.1`, `minimum_group_size=10`, source-representative support 20/source, refit independently per architecture per seed -- M never reuses S's quantiles.
- Development data: evaluation only, never optimization, for either arm.
- No locked splits at any stage.

## 6. M9.8 predeclared endpoints, statistics, and promotion rule

Full machine-readable freeze: `reports/evaluation/hydrocore-v5/m9-7/m9-7-m9-8-preregistration.json`. Summary:

**Arms**: `ARM_S` = frozen HydroCore-S + frozen M9.6 `ARM_B_M9_6` recipe (checkpoint-reuse-by-default from M9.6's own trained artifacts, SHA-256 verified, transparent REGENERATED fallback if the blob is physically absent -- mirroring the M9.1/M9.0b convention exactly). `ARM_M` = HydroCore-M + identical recipe except capacity, always trained fresh.

**Depths**: `(1, 2, 3, 4, 6, 12, 25)`; buckets `EARLY=(1,2,3)`, `MID=(4,6)`, `MATURE=(12,25)`.

**Primary endpoint**: unseen-topology MATURE neural Top-1, macro-averaged equally across the 3 unseen families (`coastal-branch`, `tree-branch`, `dense-loop`) -- identical construction to M9.6's own primary metric, applied as `ARM_M - ARM_S`.

**Statistics**: paired bootstrap, family-stratified/matched-incident, 2,000 resamples, 90% CI, resampling unit = incidents. **Bootstrap seed: 20260819** -- deterministic, verified unused anywhere in this repository at freeze time, distinct from M9.6's own bootstrap seed (20260818) and the standing confirmation seed (20260815).

**Practical-effect threshold**: no numeric size-promotion threshold is preregistered anywhere in this repository to inherit -- `/workspace/experiments.txt` Milestone 9.2's own "Promotion philosophy" gives only qualitative worked examples (+10pp judged meaningful; +1-2pp judged not meaningful), and the actual M9.2/M9.3 milestones executed on this branch are unrelated diagnostic studies, not that capacity-threshold worked-example milestone. Accordingly this document adopts the M9.7 task prompt's own proposed quantitative floor, which sits consistent with (not below) experiments.txt's own "+1-2pp is not meaningful" precedent:

HydroCore-M may replace HydroCore-S in a future M9.8 decision only if **all** of the following hold:

- **A (primary effect)**: `M - S >= +0.02` absolute on the primary endpoint, AND the paired-bootstrap 90% CI lower bound `> 0`.
- **B (family consistency)**: M improves at least 2 of the 3 unseen families, and no unseen family regresses by more than 0.03 absolute.
- **C (seed consistency)**: no catastrophic seed reversal; all three seed deltas reported individually.
- **D (known-family retention)**: EARLY Top-1 regression `<= 0.05` absolute; MATURE Top-1 regression `<= 0.03` absolute; MRR regression `<= 0.03`, all relative to S.
- **E (calibration)**: alpha remains 0.1; required coverage `>= 0.85` for every predeclared seed/family bucket per the corrected M9.6 policy; candidate-set-size guard passes.
- **F (engineering)**: no NaN/Inf/instability; no causal violation; no checkpoint/resume defect; no source-support regression.
- **G (capacity cost)**: latency/memory reported, no post-hoc cost threshold invented (none exists in the repository/master protocol to inherit).

**Predeclared M9.8 decisions**: `A: HYDROCORE_M_MEANINGFUL_CAPACITY_GAIN_VALIDATED`; `B: HYDROCORE_M_NO_MEANINGFUL_CAPACITY_GAIN` (retain S, no L); `C: HYDROCORE_M_PREDICTIVE_GAIN_BUT_GUARDRAIL_FAILURE` (retain S, no L); `D: HYDROCORE_M_INCONCLUSIVE` (retain S unless separately governed, no automatic L); `E: ENGINEERING_OR_COMPARABILITY_BLOCKER`.

## 7. HydroCore-L policy (frozen now)

`HYDROCORE_L_AUTHORIZED = FALSE`. M9.8 must not train L. A future L experiment may only be proposed if (1) M produces a validated meaningful gain over S (M9.8 Decision A), (2) M still shows measurable unresolved capacity-related headroom, (3) the remaining error is plausibly capacity-limited rather than data/topology/source-identifiability/calibration-limited or irreducible hydraulic ambiguity, and (4) a separate protocol is frozen before L training begins. No automatic S -> M -> L.

## 8. Engineering preflight

Synthetic/tiny-batch smoke tests only, never development accuracy, per `tests/scientific/test_m9_7_capacity_preflight.py`. Results: `reports/evaluation/hydrocore-v5/m9-7/m9-7-engineering-smoke.json`, `m9-7-test-results.json`. Latency/memory (engineering-only, measured only after the M architecture was frozen in Section 3 above, never used to retune it): `reports/evaluation/hydrocore-v5/m9-7/m9-7-latency-memory.json`.

## 9. Closure

Final M9.7 decision record: `reports/evaluation/hydrocore-v5/m9-7/m9-7-closure.json`. M9.7 ends once the M architecture and the M9.8 comparison design above are frozen and the engineering preflight is green -- it does not itself execute M9.8.
