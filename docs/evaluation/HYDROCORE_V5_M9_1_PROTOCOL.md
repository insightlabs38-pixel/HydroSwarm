# HydroCore-v5 Milestone 9.1 scientific protocol (frozen before any candidate architecture is trained on the full recipe or evaluated for predictive performance)

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` (see its own Section 8.10). This is the frozen SCIENTIFIC comparison protocol referenced by the M9.1 preflight (`docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md`) and preflight correction (`docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_CORRECTION.md`), both of which are unmodified by this document. It is frozen after preflight closed `M9_1_FULL_EXPERIMENT_READY = YES` (`reports/evaluation/hydrocore-v5/m9-1-preflight-correction-summary.md`), before any candidate is trained on real predictive data or evaluated on `development_holdout`. It is not altered after seeing any predictive-performance result; amendments require an explicit, dated addendum to this document (Section 21), never a silent edit of a frozen section.

**This document authorizes no execution.** Writing it does not run the experiment, train a model, or evaluate development performance. It exists so that no later step can retroactively redefine a split, metric, seed, threshold, or promotion rule to make a specific candidate look better.

Code-under-test: `exp/hydrocore-v5-causal` at commit `49058beb19cdb4c4ed51fc1afd1e77626c65f3b4` (or a later commit on the same branch that changes nothing under `src/hydroswarm/model/continuous_time.py`, `src/hydroswarm/model/core.py`'s `temporal_dynamics` seam, or `configs/training-v5-causal.yaml` -- any such change requires re-running the M9.1 preflight, not merely re-reading this document). The exact commit actually executed against must be recorded verbatim in every run's own output artifact.

## 0. Motivating question (frozen, restated from the original M9.1 milestone framing)

Does continuous-time latent evolution over actual physical elapsed time (Graph Neural ODE), evidence-driven continuous control (Graph Neural CDE), or continuous stochastic evolution (Stable Graph Neural SDE) provide a better temporal inductive bias than HydroCore's current transformer-over-history-then-mean-pool temporal pathway, at matched parameter count and matched training recipe? This document answers that question with one predeclared primary metric and one predeclared decision procedure (Section 12); it does not permit ad hoc metric selection after seeing results.

## 1. Arms (frozen, no addition, no substitution)

| arm | temporal mechanism | wiring |
|---|---|---|
| `CURRENT` | `HydroCore(temporal_dynamics=None)` -- default `TemporalEncoder`/`QualityEncoder`, unmodified | control |
| `GRAPH_ODE` | `GraphODEDynamics` | `HydroCore(temporal_dynamics=GraphODEDynamics(...))` |
| `GRAPH_CDE` | `GraphCDEDynamics` (corrected, mask-aware) | `HydroCore(temporal_dynamics=GraphCDEDynamics(...))` |
| `GRAPH_SDE` | `GraphSDEDynamics` | `HydroCore(temporal_dynamics=GraphSDEDynamics(...))` |

No fifth arm. mTAN, GRU-ODE-Bayes, Neural PDE, temporal Transformer, graph Transformer, diffusion model, latent-ODE architecture zoo, and HPO over solver/model families remain out of scope (preflight protocol Section 1), and may be reopened only by an explicit, dated addendum to this document following an unexpected, specific, mechanistically-unresolved M9.1 result -- "another model might perform better" is never sufficient reason on its own (preflight protocol Section 32).

## 2. Frozen entry recipe (inherited unchanged from M9.0b, not reopened here)

- representation: `AGE_FIX_ONLY` (`HydraulicFeatureBuilder.build(..., unobserved_age_sentinel="fixed")`, `elapsed_time_normalization="window_relative"` for CURRENT's own encoders where applicable -- the continuous-time arms use their own frozen time semantics, Section 6 below).
- topology training: `SINGLE_FAMILY_CURRENT_TRAINING` -- `network_loader` = the golden-reference network loader used throughout M1/M8.7/M9.0b's own `CURRENT_FAMILY_DEPTH` control (`NETWORK_FAMILY = "golden-reference"` in `hydroswarm.training.causal_prefix`). The validated `INTERLEAVED_MULTI_FAMILY` topology-diversity recipe remains scientifically valid but not operationally promoted (M9.0b, `CALIBRATION_SYSTEMATICALLY_INCOMPATIBLE`) and is NOT used by any M9.1 arm.
- calibration method: `B_DEPTH_AWARE`, i.e. M9.0b's Scheme A construction, `network_id = f"{family}:{depth_bucket}"` (with `family` constant `"golden-reference"` for every M9.1 arm, since training is single-family), unmodified `hydroswarm.calibration.conformal.SplitConformalCalibrator`, `minimum_group_size=10` (the library default, not overridden).
- alpha: `0.1`, fixed, unconditional, for every arm and every seed.
- PCGrad: `pcgrad_enabled=false` (matching `configs/training-v5-causal.yaml`'s own committed value -- not overridden).
- causal-prefix training policy: `hydroswarm.training.causal_prefix.ARM_POLICIES["A"]` (full-history-control, depth=25 training), matching M8.7 Section 6 exactly -- this milestone compares temporal-dynamics MECHANISM, not causal-prefix depth-weighting policy (that remains M1B/C's own, separately closed, ablation).
- WNTR/EPANET remains deterministic physical authority; no neural output bypasses it.

**CURRENT reuses M8.7's own frozen `AGE_FIX_ONLY` checkpoints -- zero retraining, by default.** `CURRENT`'s architecture/representation/topology/config (Section 2 above) is, by construction, identical to M8.7's already-closed `AGE_FIX_ONLY` arm (`reports/evaluation/hydrocore-v5/m8-7-summary.md`, `m8-7-closure.json`): same `SHARED_MODEL_CONFIG`, same `unobserved_age_sentinel="fixed"`, same golden-reference-only training, same three seeds already trained (`20260814`, `31874`, `20260815`; `reports/evaluation/hydrocore-v5/m8-7-runs/AGE_FIX_ONLY-seed{20260814,31874,20260815}.json`). Per-seed loading procedure (mirroring M9.0b Section 1's own frozen-checkpoint-reuse convention exactly):

1. Read `training_summary.export_path`/`training_summary.export_sha256` from the corresponding `m8-7-runs/AGE_FIX_ONLY-seed{seed}.json`.
2. If the file at `export_path` exists on disk: recompute its SHA-256 and compare against `training_summary.export_sha256`. On match, load it as `CURRENT`'s model for that seed -- `torch.no_grad()` throughout, `model.eval()` only, no gradient computation, exactly M9.0b's own convention for reused checkpoints. On mismatch: abort before any inference (a silent corruption/mismatch is never tolerated).
3. **If the file at `export_path` does not exist** (e.g. a prior session's ephemeral `experiments/runs/` artifact that did not persist -- `m8-7-runs/*.json` itself is git-tracked and always available, but the multi-megabyte checkpoint blob it points to is not guaranteed to be): retrain that seed from scratch, using EXACTLY the frozen M8.7 `AGE_FIX_ONLY` recipe (identical config, identical seed, identical `configs/training-v5-causal.yaml`, identical data). This retrained checkpoint is used as `CURRENT` for that seed; it is NOT required or expected to reproduce M8.7's original checkpoint SHA-256 bit-for-bit (floating-point behavior can differ across hardware/library-patch versions even under `deterministic=True`). Any such regeneration MUST be recorded explicitly and visibly in that seed's `m9-1-runs/CURRENT-seed{seed}.json` as `"current_checkpoint_source": "REGENERATED_NOT_ORIGINAL_M8_7_CHECKPOINT"` (vs. `"REUSED_M8_7_CHECKPOINT"` for the normal path) -- this is a transparency requirement, not silently absorbed into the numbers, so any future reader can immediately see whether `CURRENT`'s exact weights matched M8.7's historical run.

`GRAPH_ODE`/`GRAPH_CDE`/`GRAPH_SDE` are always trained fresh under this protocol -- Section 5's seed plan and Section 10's training configuration govern them; they have no prior checkpoint to reuse.

## 3. Splits (frozen, reused verbatim, not redefined)

`hydroswarm.training.causal_prefix.build_scenario_pool(split, network_loader=<golden-reference loader>)`, `SPLIT_SEED_RANGES = {"train": (900_000_000, 600), "validation": (901_000_000, 100), "development_holdout": (902_000_000, 120), "calibration": (903_000_000, 150)}`, golden-reference topology only, contamination events only -- the exact, unmodified machinery every M1/M8.7/M9.0-line milestone has used. Physical scenarios are split before any causal-prefix depth view is derived (`truncate_causal_prefix` is a lazy view over an already-split scenario); derived prefixes of one scenario stay in that scenario's split by construction.

`locked_final_test`/`locked_topology_test` remain unopened for the entire duration of M9.1 (screening AND confirmation stages). Checked via `hydroswarm.evaluation.live_robustness.locked_test_opened(ROOT)` before and after every script this protocol governs, recorded in every run's output JSON as `locked_test_opened_before`/`locked_test_opened_after`. `ood_development` is not used (M9.1 does not evaluate or tune OOD behavior). `development_holdout` is used freely for this milestone's own accuracy/calibration evaluation (Section 1.1 of the main experiment protocol governs this as ordinary development use, not a locked split).

## 4. Depth grid and maturity buckets (frozen, reused verbatim)

`CAUSAL_PREFIX_DEPTHS = (1, 2, 3, 4, 6, 12, 25)` (`FULL_HISTORY_DEPTH = 25`). Maturity buckets, exactly as `scripts/hydrocore_v5/run_m9_0a_evaluate.py`/`run_m9_0_evaluate.py` already define them:

```
EARLY_DEPTHS = (1, 2, 3)
MID_DEPTHS   = (4, 6)
MATURE_DEPTHS = (12, 25)
```

Training uses full-history depth=25 only (Section 2's `ARM_POLICIES["A"]`). Evaluation exercises every depth in the grid, on every `development_holdout` scenario returned by `build_scenario_pool("development_holdout", ...)` (currently 120 scenarios; this document does not hardcode that count -- whatever the deterministic pool function returns IS the evaluation population, unchanged from every prior milestone's own convention).

## 5. Seeds (frozen, no substitution, no re-rolling)

**Screening**: `20260814`, `31874` -- apply to all four arms (`GRAPH_ODE`/`GRAPH_CDE`/`GRAPH_SDE` are trained fresh at both seeds; `CURRENT` is loaded/evaluated at both seeds via Section 2's checkpoint-reuse procedure, never retrained), matching the v5 seed policy (`HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` Section 2) and this milestone's own preflight-proposed seed plan.

**Promotion confirmation**: `20260815` -- run ONLY for whichever arm(s) satisfy Section 12's `PROMOTION_CANDIDATE` criterion at the 2-seed screening stage. If zero arms satisfy it, `20260815` is not run for any arm and the milestone closes as a negative result (Section 13, Outcome D). If more than one arm satisfies it, `20260815` is run for every such arm (not only the single largest point estimate) -- the confirmation stage re-verifies each candidate independently; only after confirmation does Section 12's tie-break among confirmed candidates apply.

Per-seed results are always preserved individually in the milestone's machine-readable artifact, never collapsed to a mean before being recorded (v5 seed policy, restated). No seed is dropped, substituted, or re-rolled because its result was unfavorable. An arm/seed producing a non-finite loss or any NaN/Inf output at any point during training or evaluation is recorded as `UNSTABLE_ARM_SEED` and is NOT retried, NOT excluded from the report, and disqualifies that arm from `PROMOTION_CANDIDATE` status regardless of its other seeds' results (Section 12).

`torch`/`numpy`/`random` seeding within a training run is governed entirely by `configs/training-v5-causal.yaml`'s own `seed` field via `Trainer`/`TrainingConfig` (`deterministic=True`), overridden in memory per run exactly as `run_m1_arm.py`/`run_m8_7_*.py` already do (the committed YAML file itself is never edited). GRAPH_SDE additionally requires the Brownian seed schedule in Section 9.

## 6. Architecture configuration (frozen, taken as-is from the corrected preflight -- NOT re-searched, NOT re-tuned)

| arm | `mlp_width` | total parameters | delta vs. `CURRENT` (4,182,612) |
|---|---|---|---|
| `CURRENT` | -- | 4,182,612 | -- |
| `GRAPH_ODE` | 574 | 4,184,118 | +0.036% |
| `GRAPH_CDE` | 214 | 4,182,894 | +0.0067% |
| `GRAPH_SDE` | 464 | 4,183,540 | +0.022% |

These exact widths, taken verbatim from `reports/evaluation/hydrocore-v5/m9-1-preflight-correction-results.json`'s `parameter_matching.corrected` block, are used for every screening and confirmation run of every seed. No width is re-searched, re-tuned, or adjusted for this scientific run under any circumstance, including if a candidate under- or over-performs -- parameter width is a pure engineering-matching constant frozen at preflight time, per preflight protocol Section 12 ("Adjust candidate hidden width/depth deterministically based on PARAMETER COUNT ONLY... Do not tune width for performance").

Base model: `HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)` (`d_model=192, nhead=6, dim_feedforward=576, num_layers=4, latent_tokens=64, modality_layers=1`; `SHARED_MODEL_CONFIG` = `run_m1_arm.py`'s own frozen dict: `prior_mode="feature_only", event_control_heads=True, scout_control_heads=True, strategist_mode="candidate_conditioned", action_vocabulary_size=ACTION_TEMPLATE_COUNT, consequence_prescreening_heads=True, ood_category_head=True`), identical for all four arms, differing only in the `temporal_dynamics` constructor argument.

## 7. Solver configuration (frozen, taken as-is from the preflight -- NOT tuned for accuracy)

**GRAPH_ODE**: `torchdiffeq.odeint`, `method="dopri5"`, `rtol=1e-3`, `atol=1e-4`. **New for the scientific run** (an engineering safety bound, not present in the preflight smoke tests because they never ran enough steps to need it): `options={"max_num_steps": 2000}` on every `odeint` call, to bound worst-case adaptive-solver step count during real training/evaluation. If this bound is ever hit, that forward pass is recorded as `SOLVER_STEP_LIMIT_EXCEEDED` (treated identically to a non-finite output for Section 5's instability handling) -- it is an engineering correction if it occurs and requires investigation before the affected arm/seed's results are used for any decision; it is never silently raised to make a failing case pass.

**GRAPH_CDE**: `torchcde.linear_interpolation_coeffs` / `torchcde.LinearInterpolation`, `torchcde.cdeint(method="rk4", options={"step_size": 0.25})`. Causal cutoff (`cutoff_index`) defaults to "use every causally available step" for every real training/evaluation call (the explicit-cutoff mode exists solely for the causality unit test, Section 9.4 of the preflight protocol, and is never used during scientific training/evaluation).

**GRAPH_SDE**: `torchsde.sdeint`, `method="euler"`, `sde_type="ito"`, `noise_type="diagonal"`, `dt=0.05`, diffusion `diffusion_scale=0.1` (bounded `[0, 0.1]` via `sigmoid`). Brownian seeding for evaluation (not training -- Section 9 below) uses the frozen deterministic schedule.

No solver package, version, method, tolerance, step size, or diffusion parameterization may be changed for this scientific run. A change is permitted only to correct a demonstrated NaN/Inf/non-convergence/`SOLVER_STEP_LIMIT_EXCEEDED` failure, and any such change must be logged as an explicit, dated engineering-correction addendum (Section 21) before the affected results are used in any decision -- exactly the preflight protocol's own Section 11 rule, carried forward unchanged.

Dependency versions pinned exactly as installed and verified at preflight time: `torchdiffeq==0.2.5`, `torchcde==0.2.5`, `torchsde==0.2.6` (the `continuous-time` optional-dependency group in `pyproject.toml`). A patch-level upgrade of any of these three packages requires re-running the full preflight correctness suite (`tests/scientific/test_m9_1_preflight.py`) before it may be used for this scientific run.

## 8. Time semantics (frozen, reused verbatim from the preflight)

```
t = (timestamps_seconds - timestamps_seconds[:, :1]) / 86_400.0
```

(`compute_relative_physical_time`, `FIXED_ELAPSED_TIME_SCALE_SECONDS = 86_400.0`). Not renegotiated here.

## 9. GRAPH_SDE Monte Carlo evaluation policy (frozen)

**MC count**: exactly `4` for every screening and confirmation evaluation of every `development_holdout` incident, at every depth. Not re-searched (preflight correction Section 20: "MC4 remains the proposed full M9.1 evaluation count" -- carried forward unchanged here).

**Brownian seed schedule** (frozen, fully deterministic, no researcher discretion at run time):

```python
import hashlib

def brownian_seed(predictor_training_seed: int, incident_id: int, prefix_depth: int, mc_index: int) -> int:
    key = f"{predictor_training_seed}:{incident_id}:{prefix_depth}:{mc_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31)
```

- `predictor_training_seed`: the arm/seed's own training seed (`20260814`, `31874`, or `20260815`).
- `incident_id`: the development_holdout scenario's own deterministic generation seed, i.e. `SPLIT_SEED_RANGES["development_holdout"][0] + index * 100` for that scenario's `index` in `build_scenario_pool("development_holdout", ...)`'s returned list -- the SAME integer `build_scenario_pool` itself already computes to generate that scenario (its own `start_seed + index * 100` loop), not a newly invented ID scheme.
- `prefix_depth`: the integer causal-prefix depth being evaluated (one of `CAUSAL_PREFIX_DEPTHS`).
- `mc_index`: `0, 1, 2, 3` for the four Monte Carlo draws.

The resulting integer is passed as `GraphSDEDynamics.forward`'s existing `seed` keyword argument (`seed: int = 20260814` in the current signature) for that specific evaluation call -- no new call-site parameter is introduced, this formula only fixes WHICH value fills the already-existing argument.

Aggregation: the reported per-incident, per-depth GRAPH_SDE posterior is the MEAN softmax probability vector across the 4 MC draws (`mean(p_0, p_1, p_2, p_3)`, not a mean of logits, not a majority vote). Predictive stochastic variance (the per-class variance across the 4 draws) is reported separately alongside the mean, never folded into it. No single Brownian draw may determine an operational prediction, top1 decision, or calibration score -- every GRAPH_SDE row used anywhere in this protocol (accuracy, calibration fitting, calibration evaluation) is the 4-draw mean.

## 10. Training configuration (frozen, reused verbatim)

`configs/training-v5-causal.yaml`, loaded via `TrainingConfig.from_yaml(..., require_complete_task_weights=True)`, UNCHANGED: `learning_rate=0.0003`, cosine schedule, `warmup_steps=10`, `epochs=20`, `batch_size=2`, `gradient_accumulation_steps=4`, `gradient_clip_norm=1.0`, `early_stopping_patience=5`, `minimum_delta=0.0`, `maximum_runtime_seconds=28800`, `device="cpu"`, `fp32=True`, `deterministic=True`, `gradnorm_logging=True`, `pcgrad_enabled=False`, full `task_weights` block exactly as committed. Checkpoint selection: lowest validation loss at full-history (depth=25) evaluation, matching every prior v5 milestone. Only `seed` (Section 5) and `gradnorm_log_every_n_batches` (a compute-cost-only knob, unchanged from M1's own override) are overridden in memory; the committed YAML is never edited. Task weights (multitask objective) are identical for all four arms -- this milestone changes temporal-dynamics MECHANISM only, not multitask weighting (M2's own frozen `PCGRAD_JUSTIFIED`, `pcgrad_enabled=false` decision, unchanged here).

## 11. Metrics (frozen)

### 11.1 Primary metric (promotion-relevant, predeclared, singular)

**Pooled EARLY+MID mean top-1 accuracy**: for each `development_holdout` incident, the mean of `top1(depth)` (1.0 if the argmax of the neural `source_node` posterior equals the true source, else 0.0) over every depth in `EARLY_DEPTHS + MID_DEPTHS = (1, 2, 3, 4, 6)`, averaged per incident, then compared paired across arms (Section 12). Frozen for this specific reason: M8.7's own closure found MATURE-depth top1 already saturated (~0.99 for every representation tested, `reports/evaluation/hydrocore-v5/m8-7-summary.md`), leaving no room for a temporal-representation change to show a measurable effect at MATURE depth -- EARLY+MID is exactly the sub-saturated regime where a genuinely better temporal inductive bias would be expected to show up, and is chosen on this pre-existing evidence, not on any M9.1 result.

Only the NEURAL predictor's own `source_node` posterior is used (`softmax(source_node_logits)`, or the 4-MC mean for GRAPH_SDE per Section 9) -- never the classical or hybrid-fused posterior. Classical localization, fusion, and OOD/disagreement behavior are confirmed unchanged by the architecture seam (preflight protocol Section 12) and are out of this comparison's scope entirely.

### 11.2 Secondary/guardrail metrics (reported for every arm/seed, not independently promotion-triggering)

- EARLY-bucket mean top1, MATURE-bucket mean top1 (each depth-bucket's own value, not folded into the primary metric), overall MRR, NLL, Brier, posterior entropy, true-source rank -- per depth, matching Section 4.1 of the main experiment protocol.
- `B_DEPTH_AWARE` marginal coverage, per-bucket (EARLY/MID/MATURE) coverage, mean candidate-set size, singleton rate -- fit independently per architecture per seed from that architecture's own governed calibration-split rows (Section 2), never reusing `CURRENT`'s quantiles for a candidate (preflight protocol Section 23).
- Candidate-set-size guardrail, expressed in the scale-invariant normalized form M9.0b Section 10 already uses: `candidate_set_size / eligible_source_node_count <= 0.5` per row (golden-reference has a fixed, known node count, so this reduces to a fixed mean-size bound; the normalized form is used for exact consistency with M9.0b's own frozen bar, not because network size varies within M9.1).
- Forward/backward finiteness, per-arm/seed instability flag (Section 5).
- Model latency/memory (engineering context only, reusing the preflight's own measured ratios; not re-measured during the scientific run unless a material code change occurred since preflight, which Section 21 would govern).

## 12. Guardrails and promotion decision procedure (frozen, single predeclared rule, no post hoc discretion)

For each of `GRAPH_ODE`/`GRAPH_CDE`/`GRAPH_SDE` independently, at the 2-seed screening stage:

**Step 1 -- Guardrails (must ALL pass, using the mean across the two screening seeds unless stated otherwise).** For every "regression" below, `regression = CURRENT_value - candidate_value` (top1 in percentage points, MRR in raw units) -- a POSITIVE regression means the candidate is worse than `CURRENT`; a negative or zero regression (candidate matches or beats `CURRENT`) always passes the guardrail. This sign convention is fixed and is never inverted by a later executor:

1. `EARLY` mean top1 regression vs. `CURRENT` (mean across `CURRENT`'s own screening seeds) `<= 5.0` percentage points.
2. `MATURE` mean top1 regression vs. `CURRENT` `<= 3.0` percentage points.
3. Overall MRR regression vs. `CURRENT` `<= 0.05`.
4. `B_DEPTH_AWARE` marginal coverage `>= 0.85` for BOTH screening seeds individually (not merely on average), AND each of EARLY/MID/MATURE pooled coverage `>= 0.85` for both seeds, AND golden-reference family's own marginal coverage (trivially identical to the pooled number here, since training is single-family) `>= 0.85` -- matching M9.0b Section 9's multi-level bar exactly. A coverage failure on ANY seed disqualifies the arm at Step 1, regardless of the other seed's result or of any accuracy gain (calibration safety is never traded against capability, per the standing scientific rule -- this mirrors M9.0b's own `CALIBRATION_SYSTEMATICALLY_INCOMPATIBLE` precedent exactly).
5. Candidate-set-size guardrail (Section 11.2) satisfied for both seeds.
6. No `UNSTABLE_ARM_SEED` or `SOLVER_STEP_LIMIT_EXCEEDED` flag on either seed.
7. No OOD/safety/authority/WNTR-gating regression (confirmed by inspection -- this milestone does not modify any of those systems, per preflight protocol Section 24; a guardrail failure here would indicate an unexpected interface violation, not a tuning outcome).

An arm failing any Step-1 guardrail is `GUARDRAILS_FAILED` and is NOT promoted regardless of its primary-metric point estimate, and does NOT proceed to the third seed.

**Step 2 -- Capability gain (only evaluated for arms that passed Step 1):**

Paired-per-incident bootstrap on the primary metric (Section 11.1), arm vs. `CURRENT`, using the SAME "representative seed" convention M8.7 Section 10/closure and M9.0b's summary tables already establish for this exact codebase (seed `31874` is the standing representative-seed choice used throughout M8/M9-line milestones, not a new pick made for this document): the candidate's seed `31874` result vs. `CURRENT`'s seed `31874` result, same-seed paired per incident, 2,000 resamples, bootstrap seed `20260815` (the confirmation seed, matching the existing convention of using the third/confirmation seed for bootstrap reproducibility regardless of whether that seed's own model has been trained yet), 90% confidence interval on the mean paired difference (candidate minus `CURRENT`). The confirmation stage (Step 3 below) does NOT reuse this screening-stage estimate -- it independently re-derives its own paired bootstrap from the seed-`20260815` pairing once that seed exists for the candidate.

- If the 90% CI is entirely `> 0`: arm is `PROMOTION_CANDIDATE` (screening-stage), proceeds to the confirmation seed (Section 5).
- If the 90% CI includes zero or is entirely `<= 0`: arm is `GUARDRAILS_PASSED_NO_SIGNIFICANT_GAIN`, NOT promoted, does NOT proceed to the third seed. This is reported as a valid, non-cherry-picked negative result for that arm (main experiment protocol Section 5's own restated rule), not omitted or downplayed.

**Step 3 -- Confirmation (only for arms that reached `PROMOTION_CANDIDATE` at screening):**

Train and evaluate seed `20260815` for that arm only. `CURRENT`'s own seed `20260815` requires no new training under this protocol -- it is one of the three M8.7 `AGE_FIX_ONLY` seeds already reused per Section 2's checkpoint-reuse procedure (loaded, not trained, subject to that section's REUSED/REGENERATED transparency requirement) -- and serves directly as the pairing partner. Re-run Step 1's guardrails using the 3-seed mean (all three of `20260814`/`31874`/`20260815`) with the SAME per-seed coverage requirement (every one of the three seeds must individually satisfy the `>=0.85` calibration bars, not just the mean). Re-run Step 2's paired bootstrap using the candidate's seed `20260815` paired against `CURRENT`'s seed `20260815` (same-seed pairing, matching Step 2's own screening-stage convention exactly), same 2,000 resamples, same bootstrap seed `20260815`, 90% CI.

- Passes Step 1 (3-seed) AND Step 2 (3-seed, CI entirely `> 0`): arm is `PROMOTION_CONFIRMED`.
- Fails either at the 3-seed stage: arm is `PROMOTION_NOT_CONFIRMED` (a valid, reportable outcome -- the screening-stage significance was not robust to the third seed, exactly analogous to M9.0a's own 3-seed-parity discipline).

**Final selection**: if exactly one arm reaches `PROMOTION_CONFIRMED`, it is the M9.1 winner. If more than one arm reaches `PROMOTION_CONFIRMED`, the arm with the LARGEST 3-seed paired-bootstrap point estimate on the primary metric is selected as the M9.1 winner. In the vanishingly unlikely event of an exact tie on that point estimate, the tied arm with the narrower 90% CI wins; if still exactly tied, the fixed order `GRAPH_ODE > GRAPH_CDE > GRAPH_SDE` (Section 1's own listing order) decides -- a fully predeclared, numeric-then-fixed-order tie-break, never a qualitative judgment call made after seeing results. If zero arms reach `PROMOTION_CONFIRMED`, the outcome is `CURRENT_HYDROCORE_RETAINED` (Section 13, Outcome D) -- `CURRENT` is not "promoted" in the sense of being a new result, it simply remains the operational architecture, and this is reported as this milestone's own negative result, matching the standing "report negative results" rule.

No promotion decision is made from the 2-seed screening stage alone. No candidate is promoted for matching `CURRENT` without a statistically significant gain (non-inferiority alone never triggers promotion, per the main experiment protocol Section 6's restated rule for model-size experiments, applied identically here).

## 13. Predeclared outcomes

- **Outcome A -- ARCHITECTURE_GAIN_VALIDATED**: exactly one arm reaches `PROMOTION_CONFIRMED` (or, per Section 12's tie-break, is selected as the single winner among multiple confirmed arms). That arm's temporal-dynamics mechanism becomes the candidate recipe for future M9 capacity-scaling work, subject to whatever separate authorization future capacity-scaling milestones require -- this document does not itself authorize M9 S/M/L scaling.
- **Outcome B -- PARTIAL_GAIN_UNCERTAIN**: one or more arms reach screening-stage `PROMOTION_CANDIDATE` but none reach `PROMOTION_CONFIRMED` at the 3-seed stage. Reported honestly as inconclusive, not rounded up to a promotion.
- **Outcome C -- GUARDRAILS_BLOCKED**: one or more arms fail Step 1 guardrails (most likely calibration, per M9.0b's own precedent for this exact recipe family). Reported per-arm with the specific failing guardrail(s) named.
- **Outcome D -- CURRENT_HYDROCORE_RETAINED**: no arm reaches `PROMOTION_CANDIDATE` even at screening. This is a legitimate, fully valid milestone closure, not a failure of the milestone itself (matching Milestone 5's own `ACTIVE SAMPLING REMAINS ADVISORY` precedent for a negative capability result).

Multiple outcomes may co-occur across different arms (e.g., `GRAPH_ODE` reaches Outcome C while `GRAPH_CDE` reaches Outcome B) -- the overall milestone report states each arm's own outcome individually; there is no single "the milestone outcome" collapsing distinct per-arm results.

## 14. Statistical procedure (frozen, restated precisely)

Paired per-incident comparison: for a given depth-grid position, both arms in a comparison must have a row for the SAME `development_holdout` incident (guaranteed by construction, since both are evaluated over the identical `build_scenario_pool("development_holdout", ...)` list). Bootstrap resampling is over INCIDENTS (not over depths, not over MC draws), matching M8.7 Section 10's own unit of resampling. `numpy.random.default_rng(seed=20260815)` (or the equivalent deterministic RNG already used by every prior milestone's own bootstrap implementation -- reuse that existing helper rather than reimplementing) drives the 2,000 resamples. The reported interval is the 5th/95th percentile of the resampled mean-difference distribution (a 90% interval, matching every prior milestone's own convention).

A point-estimate difference alone never establishes superiority (main experiment protocol Section 5, restated). Every per-incident, per-seed row is preserved in the milestone's raw results JSON, never collapsed to a mean before being recorded.

## 15. Artifact and logging requirements (frozen)

For each of screening and confirmation stages, produce:

- `reports/evaluation/hydrocore-v5/m9-1-runs/{ARM}-seed{seed}.json` for `ARM` in `{GRAPH_ODE, GRAPH_CDE, GRAPH_SDE}` -- per-arm/seed training summary (epochs completed, stop reason, best epoch, best validation loss, checkpoint SHA-256, parameter count, `locked_test_opened_before`/`after`). For `CURRENT`, this file records the checkpoint-reuse outcome instead of a fresh training summary: `current_checkpoint_source` (`REUSED_M8_7_CHECKPOINT` or `REGENERATED_NOT_ORIGINAL_M8_7_CHECKPOINT`, per Section 2), the loaded/regenerated checkpoint's SHA-256, and parameter count -- never a fabricated "epochs completed" field for a run that did not occur.
- `reports/evaluation/hydrocore-v5/m9-1-results.json` -- raw per-incident, per-depth, per-seed rows for every arm (top1, MRR, NLL, Brier, entropy, true-source rank; GRAPH_SDE rows additionally record the 4 individual MC draws and their variance, per Section 9), never pre-aggregated away.
- `reports/evaluation/hydrocore-v5/m9-1-calibration.json` -- per-arm, per-seed `B_DEPTH_AWARE` fit (group-support audit: `n` per group, quantile, rank, `minimum_group_size` pass/fail -- matching M9.0b Section 7's own audit convention -- reported BEFORE any held-out coverage number), marginal/EARLY/MID/MATURE coverage with Wilson 95% intervals (matching M9.0b Section 11).
- `reports/evaluation/hydrocore-v5/m9-1-guardrails.json` -- Step 1/Step 2/Step 3 results per arm, exact pass/fail per named guardrail, bootstrap point estimate and CI per comparison.
- `reports/evaluation/hydrocore-v5/m9-1-summary.md` -- human-readable summary in the same style as `m8-7-summary.md`/`m9-0b-summary.md`, stating each arm's Section 13 outcome explicitly.
- `reports/evaluation/hydrocore-v5/m9-1-closure.json` -- final decision record (Section 12's final selection, or Outcome D), `M9_1_FINAL_DECISION`, `locked_test_opened_before`/`after` for the entire milestone.

Every artifact records: the exact executed git commit SHA, the exact seed(s), `torchdiffeq`/`torchcde`/`torchsde` versions, and `locked_test_opened` before/after. No number from `locked_final_test`/`locked_topology_test` may appear in any M9.1 artifact (main experiment protocol Section 1.3, restated).

## 16. Compute environment (frozen)

Python 3.12.13, PyTorch 2.13.0, CPU-only (`device="cpu"`, matching `configs/training-v5-causal.yaml`). `torchdiffeq==0.2.5`, `torchcde==0.2.5`, `torchsde==0.2.6` (Section 7). A change to the Python/PyTorch minor version requires re-running the M9.1 preflight correctness suite before this protocol may be executed.

## 17. Scope discipline (restated, unchanged from preflight)

No development-holdout labels are used to TRAIN (they are used only for evaluation, exactly as every prior v5 milestone already does). No calibration-split examples are used to optimize model weights. No locked data access at any point. No M9 S/M/L capacity scaling begins under this document's authorization. No architecture-family expansion beyond Section 1's frozen four without an explicit, dated addendum triggered by a specific unresolved mechanistic finding (Section 1). No safety/authority-threshold change. No alpha change. No topology-recipe change. No representation change away from `AGE_FIX_ONLY`. Conformal alpha, calibration method, and guardrail thresholds are never adjusted after seeing results, at any stage of this milestone.

## 18. Relationship to prior artifacts

This document supersedes nothing in `HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md` (architecture interface, causality, solver correctness) or `HYDROCORE_V5_M9_1_PREFLIGHT_CORRECTION.md` (the two implementation fixes) -- both remain independently frozen and historically authoritative for the engineering-correctness questions they answer. This document is authoritative for every SCIENTIFIC question (metrics, promotion, statistics) it defines; where a scientific-run script needs an engineering parameter (solver config, parameter width, time semantics), it is pulled from the preflight/correction artifacts verbatim, as cited in Sections 6-8 above, never re-derived.

## 19. Pre-execution checklist (to be satisfied, not executed, by this document's own commit)

Before the first screening run may begin: (a) this document is committed and its own commit SHA recorded as this protocol's `frozen_at_commit`; (b) `tests/scientific/test_m9_1_preflight.py` is green at that commit; (c) `locked_test_opened(ROOT)` is `False`; (d) the executing script asserts the code-under-test commit matches Section header's pinned SHA (or a Section-21-governed later one) before training anything, aborting otherwise.

## 20. Researcher-degrees-of-freedom audit (self-review, recorded per the milestone instruction to review this document for ambiguity before commit)

Every parameter a later executor could otherwise choose freely has been pinned above; specifically checked and closed:

- primary metric identity and its exact depth-bucket composition (Section 11.1) -- pinned, with a cited, pre-existing (not post hoc) justification;
- bootstrap resample count, seed, unit of resampling, and interval width (Section 14) -- pinned;
- guardrail thresholds and which seeds/buckets they apply to (Section 12 Step 1) -- pinned to the exact pre-existing M8.7/M9.0b bars, not newly invented;
- tie-break rule among multiple `PROMOTION_CONFIRMED` arms (Section 12, final selection) -- pinned to a single numeric rule;
- what happens when zero, one, or multiple arms pass at each stage (Section 12/13) -- fully enumerated;
- SDE Monte Carlo count, aggregation rule, and Brownian seed formula (Section 9) -- pinned to an executable formula, not a "conceptual form";
- what "the incident" means as a hashable ID for the Brownian seed (Section 9) -- pinned to the existing deterministic scenario-generation seed, not a new ID scheme;
- architecture width/parameter count per arm (Section 6) -- pinned to the exact preflight-correction values, explicitly forbidding re-search;
- whether `CURRENT` is retrained or reused from M8.7's own frozen checkpoints (Section 2) -- an early draft of this document left this genuinely undecided (a real degree of freedom that could silently produce two different `CURRENT` populations across runs); pinned to "reuse by default, verified by SHA-256, with an explicit, logged, non-silent fallback to retraining only if the checkpoint file is physically absent";
- solver settings and the one permitted category of mid-run change (engineering correction only, logged) (Section 7) -- pinned, with an explicit new step-count safety bound added (`max_num_steps=2000`) that the preflight smoke tests never needed but a full training run could plausibly hit;
- exact split/seed-range machinery, with no new split invented (Section 3) -- pinned by direct reference to existing code constants;
- instability handling (an unstable seed disqualifies rather than being retried or excluded) (Section 5) -- pinned;
- artifact schema and required fields (Section 15) -- enumerated explicitly so no two runs of this protocol could plausibly produce differently-shaped outputs;
- sign convention for "regression" in the accuracy/MRR guardrails (Section 12 Step 1) -- pinned explicitly (`CURRENT minus candidate`, positive = candidate worse), since an unstated sign convention is exactly the kind of silent inversion risk a later executor could get backward without anyone noticing;
- exact tie-break among multiple `PROMOTION_CONFIRMED` arms all the way down to an impossible-in-practice exact-tie case (Section 12, final selection) -- pinned through a fixed arm-listing-order fallback, so no comparison this protocol could ever produce is left without a determinate winner.

No section of this document leaves a promotion-relevant threshold, seed, metric, or tie-break to be decided at execution time.

## 21. Amendment log

This document is immutable except via a dated, explicit entry in this section.

### 2026-08-16 -- pre-execution protocol-alignment pass

Three items, closed together as one dated entry, before any full M9.1 arm is trained or `development_holdout` is inspected:

**(a) `code_under_test_commit` pin updated.** This document's own preamble (the "Code-under-test:" line immediately following the title and scope statement) pinned `exp/hydrocore-v5-causal` commit `49058beb19cdb4c4ed51fc1afd1e77626c65f3b4` as code-under-test. That commit does NOT contain Section 7's frozen `options={"max_num_steps": 2000}` bound on `GraphODEDynamics`'s `torchdiffeq.odeint` call -- it was specified in Section 7 at freeze time but not yet implemented in code, a gap this entry closes. Commit `154605180f2a950d86452cfc8ec7202990aba8cf` ("fix(v5-causal): implement frozen GRAPH_ODE max_num_steps bound") adds exactly that bound and no other model change; it is now the effective `code_under_test_commit`, per Section 19(d)'s own anticipated "or a Section-21-governed later one" mechanism -- superseding, not silently replacing, the header's original pin. Verification performed at this commit before this entry was written: `tests/scientific/test_m9_1_preflight.py` 47/47 passed; `scripts/hydrocore_v5/run_m9_1_preflight.py` re-run, all arm verdicts unchanged (`CURRENT` `BASELINE_VALID`, `GRAPH_ODE`/`GRAPH_CDE`/`GRAPH_SDE` `PREFLIGHT_PASS`); parameter counts unchanged (Section 6's table is still exact -- `max_num_steps` is a solver option, not a trainable parameter); full repo suite 1234 passed, 1 pre-existing unrelated skip, 0 failed; `ruff check` clean on the changed file (repo-wide, the same 8 pre-existing unrelated errors as at the original freeze); `pyright` 0 errors repo-wide; `locked_test_opened` `False` before and after every check in this pass. No full M9.1 arm was trained and no `development_holdout` data was inspected to perform this verification -- it is entirely a preflight-suite and static-check re-run.

**(b) "3-seed paired-bootstrap point estimate" (Section 12, final selection) clarified.** As written, this phrase is genuinely ambiguous between (i) a bootstrap computed by pooling incident rows across all three seeds, and (ii) the confirmation-stage (Step 3) bootstrap, which uses only the seed-`20260815` pairing. No procedure for (i) is defined anywhere in this document, and this amendment does not introduce one. **Clarification: the phrase means (ii) exclusively** -- "the arm with the LARGEST 3-seed paired-bootstrap point estimate" in Section 12's final-selection paragraph refers to the point estimate produced by Step 3's own already-fully-specified procedure (candidate's seed `20260815` vs. `CURRENT`'s seed `20260815`, same-seed paired per incident, 2,000 resamples, bootstrap seed `20260815`, 90% CI) for each `PROMOTION_CONFIRMED` arm. "3-seed" in that phrase describes WHICH STAGE the estimate comes from (the stage at which all three seeds' guardrails have been checked, Step 3), not how many seeds' incident rows feed the bootstrap resampling itself -- exactly one seed's rows (`20260815`) do, per Step 3 as originally written. No other part of Section 12 or Section 14 is altered by this clarification.

**(c) Runner SHA/lock-state recording requirement made explicit.** Section 15 already required "the exact executed git commit SHA" and `locked_test_opened_before`/`after` in every artifact; this entry makes explicit, for the avoidance of doubt, that "the exact executed git commit SHA" means BOTH of two distinct values, both mandatory in every artifact produced by any future M9.1 runner (screening, confirmation, or closure):

- `protocol_frozen_at_commit`: `0f05be1d47258a8c3d19e3a0d0e1122e3e560069` (the commit that added this document, `docs(v5-causal): freeze Milestone 9.1 scientific experiment protocol`) -- fixed, never changes for as long as this document itself is in force.
- `code_under_test_commit`: the exact commit of `exp/hydrocore-v5-causal` actually checked out and executed against at run time -- per (a) above, this must be `154605180f2a950d86452cfc8ec7202990aba8cf` or a later commit on the same branch that changes nothing under `src/hydroswarm/model/continuous_time.py`, `src/hydroswarm/model/core.py`'s `temporal_dynamics` seam, or `configs/training-v5-causal.yaml` (Section header's own re-preflight-trigger rule, unchanged).

Together with `locked_test_opened_before`/`locked_test_opened_after` (Section 15, unchanged), every M9.1 artifact must therefore carry all three fields. A runner that omits any of the three, or that executes against a `code_under_test_commit` other than the one this entry pins (or a later one satisfying the unchanged-files condition above), is non-compliant with this protocol and its output may not be used for any Section 12 decision.

No other section of this document is altered by this entry.

### 2026-08-17 -- M9.7 additive `core.py` change audited, `code_under_test_commit` pin re-superseded (provenance refresh, not a scientific reopening)

M9.7 (`feat(v5-causal): add governed HydroCore-M capacity config`, commit `475874d8977d0952e8fc3626eb2bd6580cc3c2f7`) modified `src/hydroswarm/model/core.py` to register a new `MODEL_VARIANTS["small_v5_capacity_m"]` entry for the M9.7/M9.8 capacity-preflight/experiment scripts. Per this document's own re-preflight-trigger rule (Section 21(a)/(c), unchanged), ANY change under `src/hydroswarm/model/core.py` correctly and automatically trips `assert_code_under_test_commit`'s frozen-path tripwire, regardless of the change's actual content -- this is by design, not a bug in the M9.1 machinery, and M9.8's own closure commit (`e33d71804c517d712724488ce9f005093cd06cb5`) disclosed the resulting `tests/scientific/test_m9_1_runner.py::test_assert_code_under_test_commit_passes_at_current_head` failure explicitly rather than silently working around it.

This entry closes that disclosed gap via a narrow, audited provenance refresh -- no full M9.1 scientific experiment (screening, confirmation, or any `development_holdout`/calibration/locked inference) is rerun, and no Section 12 decision is touched.

**Audit performed (mechanical, no predictive data inspected):**

- `git log 154605180f2a950d86452cfc8ec7202990aba8cf..475874d8977d0952e8fc3626eb2bd6580cc3c2f7 -- src/hydroswarm/model/continuous_time.py src/hydroswarm/model/core.py configs/training-v5-causal.yaml` lists exactly one commit: `475874d`. No other commit in that range touches any frozen path.
- `git diff` of that range for `continuous_time.py` and `configs/training-v5-causal.yaml` is byte-empty -- neither file changed at all.
- `git diff` of that range for `core.py` is a pure 17-line addition (0 lines removed, matching `475874d`'s own `17 insertions(+)`): one new `MODEL_VARIANTS` dict entry, `"small_v5_capacity_m": ModelVariant(352, 11, 1056, 4, 64)`. The existing `"small"`/`"medium"`/`"large"` entries are untouched (appear only as unchanged context in the diff).
- `MODEL_VARIANTS` is resolved strictly by key lookup (`MODEL_VARIANTS[variant.lower()]`, `HydroCore.from_variant`), never iterated by position or count, so an added key cannot alter `"small"`'s resolved `ModelVariant` or any GRAPH_ODE/CDE/SDE construction path (both of which live entirely in `continuous_time.py`, confirmed byte-unchanged above).
- Mechanically re-verified at `475874d` (and at every later commit through this entry's own HEAD, via the new regression test below): constructing `"small"` with this document's own frozen `SHARED_MODEL_CONFIG`/`CURRENT_MODEL_KWARGS`/`ACTION_TEMPLATE_COUNT` yields exactly `CURRENT_BASELINE_TOTAL_PARAMS` = 4,182,612 trainable parameters -- Section 6's frozen table is still exact, unchanged from the original preflight.

**Conclusion**: the M9.7 change is additive-registration-only and semantically irrelevant to every M9.1-frozen path, config, or output. It is not a change to `"small"`, to any GRAPH_ODE/CDE/SDE implementation, to temporal dynamics, or to training configuration.

**Mechanism update**: `scripts/hydrocore_v5/m9_1_common.py`'s `CODE_UNDER_TEST_COMMIT_FLOOR` (Section 19(d)/21(c)'s own `code_under_test_commit` pin) is re-superseded from `154605180f2a950d86452cfc8ec7202990aba8cf` (preserved as `CODE_UNDER_TEST_COMMIT_FLOOR_V1`) to `475874d8977d0952e8fc3626eb2bd6580cc3c2f7`, per the SAME "later commit that changes nothing under the three frozen paths" mechanism this document already defines -- not a new mechanism, not a weakened one. `FROZEN_UNCHANGED_PATHS` and the tripwire's all-or-nothing diff check are unchanged: this is a one-time, audited exception for exactly the `475874d` commit, not a rule that future `core.py` changes are presumptively fine. Any commit after `475874d` that touches any of the three frozen paths still trips the guard exactly as before, requiring the same audit-and-amend procedure.

**Regression coverage**: `tests/scientific/test_m9_1_runner.py::test_code_under_test_commit_floor_v2_audit_is_additive_only` re-runs the git-log/git-diff audit above on every test invocation (not just at authoring time), so a future history rewrite that changed what `154605180f2a950d86452cfc8ec7202990aba8cf..475874d8977d0952e8fc3626eb2bd6580cc3c2f7` actually contains would fail the test, not silently pass. `test_frozen_small_variant_param_count_unchanged_at_current_head` independently re-confirms the 4,182,612 parameter count at whatever commit the suite is run at.

**Verification performed before this entry was written**: `tests/scientific/test_m9_1_runner.py` 82/82 passed (80 pre-existing + 2 new); `tests/scientific/test_m9_1_preflight.py` 47/47 passed (unchanged); `tests/scientific/test_m9_7_capacity_preflight.py`, `test_m9_7a_checkpoint_policy.py`, `test_m9_8_capacity_comparison.py` unaffected (84/84 combined, unchanged by this entry); `pyright` 0 errors on all touched files; full repository suite re-run: **1539 passed, 1 pre-existing skip, 0 failed** (previously 1536 passed / 1 skipped / 1 failed -- the +2 delta is this entry's own 2 new tests, and the previously-disclosed `test_assert_code_under_test_commit_passes_at_current_head` failure is now a pass, for zero net regressions; see `reports/evaluation/hydrocore-v5/m9-1-provenance-refresh/m9-1-provenance-summary.md`). `locked_test_opened` `False` before and after this entire audit -- no `development_holdout`, calibration, `locked_final_test`, or `locked_topology_test` data was loaded or inspected at any point.

No other section of this document, and no M9.1 metric/guardrail/decision/artifact, is altered by this entry.
