# Phase 13 — Required Metrics and Baselines

core-issues3.txt "PHASE 13 — REQUIRED METRICS AND BASELINES". This report
consolidates every metric the phase requires: metrics already computed by
earlier phases (Stage A/B/D/E/F, control-heads, class-prevalence) are cited
by source file, and metrics that had never been computed anywhere in the
repo before this pass were computed by three new scripts written this pass:

- `scripts/run_phase13_sentinel_metrics.py` → `phase13-sentinel-classification-metrics.json`
- `scripts/run_phase13_ood_control_metrics.py` → `phase13-ood-control-metrics.json`
- `scripts/run_phase13_strategist_physical_metrics.py` → `phase13-strategist-physical-metrics.json`

All three read only already-trained, already-on-disk checkpoints
(`experiments/runs/stage-f/no_adapters-seed{20260810,20260811}`,
`experiments/runs/v4-strategist-heads-v4corpus-corrected`) against
validation/development_holdout/calibration splits of
`data/learning-v2/cycle-b2-joint-v4` and
`data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected`.
**No retraining, no new corpus generation, no locked-test access.**

## Two findings surfaced by this pass, reported up front

1. **`sensor_fault` evaluates as trivially perfect (F1=1.0) because the
   evaluated population has ZERO true negatives**, not because the head
   generalizes well. Confirmed on two independent populations: joint-v4
   validation (3,888/3,888 sensor-bearing nodes positive) AND cycle-b2's
   own base validation split directly, using the separately-trained
   Stage-A Sentinel checkpoint (also 3,888/3,888 positive). This is not an
   artifact of this pass's evaluation code (the mask -- `node_id in
   scenario.sensor_nodes`, `src/hydroswarm/training/corpus.py:424` -- is
   scoped correctly per Phase 1.3's original intent); every generated
   validation scenario in `cycle-b2` apparently has every one of its
   sensors registering at least one frozen/outage/drift/unit-mismatch
   event. **This makes `sensor_fault`'s F1/precision/recall uninformative
   for judging real discriminative power** -- there is no held-out
   negative example anywhere in this evaluation to test against. This is a
   genuine, previously-undocumented data-generation question (possibly
   deliberate oversampling for `evidence_sufficiency`/`next_step`
   training, possibly a real defect in `frozen_mask`/`drift_mask`
   computation) that needs a dedicated audit of `src/hydroswarm/data/scenarios.py`'s
   fault-injection logic -- out of this pass's Phase-13-scope; flagged
   here as a priority item, not silently reported as a clean 1.0.
2. **`ood_category_head`'s weights never received a real training
   gradient** (Stage F's `train` split has zero `ood_class` coverage by
   the current governed data-split design -- development-holdout-only
   OOD-extension populations, per restriction #9). The near-chance numbers
   below (`macro_f1=0.09`, close to the `1/11≈0.091` chance rate) are the
   expected, correct result of evaluating an effectively-randomly-initialized
   head, not a failed classifier. This is the concrete evidence Phase 14's
   promotion gate needs to keep `ood_category` out of
   `runtime_enabled_outputs`.

Both are carried into Phase 14 below.

## Sentinel

Source: `phase13-sentinel-classification-metrics.json` (this pass, Stage-F
`no_adapters` checkpoint, both seeds) + `stage-a-sentinel-training.json`
(prior pass, Stage-A `E1` checkpoint).

| metric | value (seed 20260810 / 20260811) | source |
|---|---|---|
| source top-1 (validation) | 0.7205 / 0.7331 | this pass, v4-architecture cross-check |
| source top-3 / candidate_coverage@3 | 0.8680 / 0.8756 | this pass |
| MRR | 0.8113 / 0.8172 | this pass |
| ECE (raw softmax, uncalibrated) | 0.0280 / — | this pass |
| conformal coverage (calibrated, Stage-A) | 0.9073 | `stage-a-sentinel-training.json` |
| candidate-set size (calibrated, Stage-A) | 2.29 (mean) | `stage-a-sentinel-training.json` |
| calibration ECE (calibrated, Stage-A) | 0.0213–0.0261 | `stage-a-sentinel-training.json` |
| **event-presence P/R/F1** (validation) | P=0.867 R=0.924 **F1=0.895** | this pass — first time computed |
| **event-cause macro F1** (validation, 3 supported classes) | **0.698** | this pass. AMBIGUOUS and HYDRAULIC_MISMATCH both have **zero true examples** in this corpus (confirmed via `class-prevalence.json` class index 2/3 and this pass's own `positive_support=0`) — excluded from the macro average, matching this project's own `UNSUPPORTED_OOD_CATEGORIES` convention. Per-class: CONTAMINATION F1=0.898, SENSOR_FAULT F1=0.771, NORMAL F1=0.425 (weakest — low recall, 0.31) |
| event-cause data-quality caveat | — | ~5% (633/12750) of `cycle-b2` examples carry a pre-Phase-6.4 HYDRAULIC_MISMATCH mislabel; `cycle-b2` is immutable, so this checkpoint's `event_cause` head trained on that noise |
| **profile accuracy / ordinal MAE (bins)** (validation) | start_time 0.654 / 0.515; duration 0.503 / 0.588; relative_strength 0.750 / 0.317 | this pass — first time computed. `relative_strength` (4 bins) is the strongest; `duration` (3 bins) the weakest |
| **sensor-fault macro P/R/F1** | 1.0 / 1.0 / 1.0 | this pass — **see finding #1 above; not a meaningful result** |
| topology transfer (UNSEEN_TOPOLOGY, dev-holdout, Stage-A) | source_top1 0.446–0.50 | `stage-a-sentinel-training.json`; also cross-checked this pass on the Stage-F checkpoint: source_top1 0.464–0.504, event_presence F1 0.81–0.83, profile accuracy drops ~0.15-0.2 vs. validation |
| missingness/fault robustness (SEVERE_MISSINGNESS, dev-holdout) | source_top1 0.64–0.65 (Stage-A: 0.71 pre-shift) | `stage-a-sentinel-training.json`; cross-checked this pass: event_presence F1 stays ~0.91-0.92 (robust), profile accuracy drops modestly |

## Scout

Source: `stage-d-scout-policy-comparison.json` (prior pass, real WNTR-derived
truth, no re-simulation).

| metric | value | status |
|---|---|---|
| step-0 entropy reduction, `learned_scout` | −0.219 bits (worse than `random`'s +0.007 or `fixed_order`'s +0.015) | FOUND |
| step-0 agreement with classical EIG, `learned_scout` | 0.567 | FOUND |
| median samples to resolution / resolved-within-{1,2,3} | classical_eig: 0.0 / 0.637 / 0.683 / 0.697 (random/fixed_order/classical_eig only) | **FOUND for 3 non-learned policies; `learned_scout` structurally EXCLUDED** — `HydroBatch` carries no `already_sampled`/revealed-evidence conditioning, so a genuine multi-step `learned_scout` trajectory cannot be constructed (architecture gap, not a missing eval; documented in the comparison script's own `exclusion_reason`) |
| expected/realized candidate contraction | realized, non-learned policies only, as above; expected proxy `candidate_reduction_mse=0.0213` (`scout-heads-training.json`) | PARTIAL, same root cause |
| regret vs. exact classical EIG, inaccessible-node rate, already-sampled rate, unnecessary-sample rate, budget violations, per-decision latency | — | **MISSING, same root cause** (no `already_sampled` input) — not computed this pass. Fixing requires a real `HydroBatch`/Scout-state architecture change (Phase 5/9 scope, not Phase 13); flagged for a future continuation, not attempted here to stay in scope |

## Strategist

Source: `stage-e-strategist-comparison-v4corpus-corrected.json` (prior
pass) + `phase13-strategist-physical-metrics.json` (this pass, same
checkpoint/corpus/1000-scenario validation split, real WNTR-verified
targets already stored — no re-simulation).

| metric | exact_all | deterministic_heuristic | learned_ordering | learned_prescreen | source |
|---|---|---|---|---|---|
| selected_valid_rate | 1.000 | 1.000 | 1.000 | 1.000 | prior pass |
| mean_simulator_calls | 9.0 | 3.0 | 1.0 | 3.0 | prior pass |
| mean_regret_vs_oracle | 0.0 | 0.00596 | 0.01085 | 0.00645 | prior pass |
| **exposure_reduction_vs_no_action (mg)** | 0.0263 | 0.0227 | 0.0178 | 0.0203 | this pass — first time in physical units |
| **pressure_violation_minutes** (mean, selected plan) | ≈0.0 | ≈0.0 | ≈0.0 | ≈0.0 | this pass — this candidate population essentially never causes pressure violations |
| **service_availability** (mean, selected plan) | ≈1.0 | ≈1.0 | ≈1.0 | ≈1.0 | this pass |
| **containment_time_minutes** (mean, selected plan) | 0.0204 | 0.0205 | 0.0205 | 0.0204 | this pass |
| **NDCG@3 (ranking quality vs. true plan_value)** | 1.000 (oracle) | 0.754 | **0.470** | 0.993 | this pass — first time computed. Confirms `learned_ordering`'s top-1-only policy is the weakest ranker; `learned_prescreen` (validity-gated) is nearly as good as the deterministic heuristic despite 1/3 the simulator calls of `exact_all` |
| **proxy error, physical units** (MAE / RMSE, n=6629 valid candidates) | exposure 0.233mg/0.328mg; pressure_risk 0.00198/0.00288 min; service_loss 0.00282/0.00402; containment_time 0.154/0.273 s; plan_regret 0.0677/0.121 | — | — | — | this pass — first time in physical units (previously only normalized-scale MSE existed) |
| latency (mean forward pass) | 0.0016s/scenario (batch 32) | — | — | — | this pass |

## OOD / control

Source: `control-heads-training.json` (prior pass, `next_step`/
`evidence_sufficiency`) + `phase13-ood-control-metrics.json` (this pass,
Stage-F `no_adapters`-seed20260810 checkpoint).

| metric | value | source |
|---|---|---|
| evidence_sufficiency accuracy/F1/ECE | 0.95 / 0.946 / 0.0085 | `control-heads-training.json` |
| next_step accuracy / macro F1 | 0.82 / 0.658 | `control-heads-training.json`. Weakest class: INSPECT_FAULTY_SENSOR (recall 0.137 — the class this project's own module docstrings already flag as needing a live controller branch, Phase 8 item 8) |
| policy_agreement (deterministic vs. control-head next_step) | 0.787 | `control-heads-training.json` |
| unsafe_non_abstention_count | 10 / 1000 (1.0%) | `control-heads-training.json` |
| **`ood_category` macro F1 / accuracy** (4 real-labeled categories, n=1600) | **0.095 / 0.105** | this pass — **near-chance; see finding #2 above, expected given zero training-gradient exposure** |
| **`ood_category` per-category recall** | EXTREME_DEMAND 0.018, FROZEN_DRIFTING_SENSOR 0.008, ROUGHNESS_MISMATCH 0.048, TANK_STATE_SHIFT 0.348 | this pass |
| **false-normal rate, learned head** (all 6 real OOD populations, n=2400) | 0.046 | this pass — misleadingly low-looking; a near-chance 11-way classifier rarely predicts any one specific class including NONE, so a low false-normal rate here does NOT imply safety, only that the head rarely says "normal" for ANY reason |
| **plan-suppression correctness, deterministic `OOD_CATEGORY_BEHAVIOR`** | **1.000** (2400/2400) | this pass — the number that actually matters for runtime safety: independent of the learned head, the governed behavior table correctly forbids planning for every real non-NONE category evaluated |
| **deterministic-vs-learned disagreement** (JS divergence, validation, n=1000) | mean 0.193; rate ≥0.5 threshold: 4.5% | this pass — first time computed over a real corpus (previously only a single golden-scenario example existed) |

## System

| metric | value | source |
|---|---|---|
| checkpoint resume | PASS (smoke-scale: best_val_loss 15.50→11.69 after resume, global_steps=6) | `stage-f-joint-corpus-gates.json` |
| strict checkpoint reload / fail-closed | 23 unit tests passing | `tests/unit/test_checkpoint_identity.py` |
| exact simulator calls per incident | 1.0–9.0 depending on policy | `stage-e-strategist-comparison-v4corpus-corrected.json` |
| Sentinel mean/p50/p95 inference latency | 0.0034–0.0045s / 0.0029–0.0034s / 0.0058–0.0098s per example (batch 16, CPU) | this pass, across validation+dev_holdout+2 OOD populations, both seeds |
| Sentinel peak RSS during evaluation | 835–901 MB | this pass |
| Strategist mean forward latency | 0.0016s/scenario (batch 32) | this pass |
| **complete incident success rate, end-to-end runtime-bundle load, live fail-closed tests** | **NOT COMPUTED — deferred to Phase 15** | No v4 runtime-bundle loader exists yet (`output_governance`/checkpoint-identity machinery is not wired into `runtime/defaults.py`/`inference/pipeline.py`); measuring this requires the Phase 15 runtime-integration work itself, not a standalone Phase 13 script. Only the legacy v3 path has a live loader + fail-closed tests today (`tests/integration/test_default_pipeline_factory.py`) |

## Genuinely deferred (documented, not silently dropped)

1. Scout regret-vs-EIG, inaccessible/already-sampled/unnecessary-sample
   rates, budget violations, per-decision latency — blocked on a real
   `already_sampled` input to `HydroBatch` (architecture change, out of
   Phase 13 scope).
2. `sensor_fault` real discriminative-power evaluation — blocked on
   understanding why the evaluated populations carry zero true negatives
   (data-generation audit, out of Phase 13 scope; flagged as priority).
3. System: complete-incident success rate and a live v4 runtime-bundle
   loader — this IS Phase 15's job; not duplicated here.
4. `ood_category` real (non-near-chance) classification metrics — would
   require a new, separately-authorized OOD-extension `train` population
   (Phase 6 data-generation scope), not attempted here since it would mean
   using development-holdout-labeled data outside its governed role.

## Artifact hashes / provenance

- Sentinel checkpoints: `experiments/runs/stage-f/no_adapters-seed{20260810,20260811}/.../model-export.safetensors` — SHA-256 recorded in `experiments/registry/stage-f.jsonl`.
- Strategist checkpoint: `experiments/runs/v4-strategist-heads-v4corpus-corrected/20260808T023023Z-df9549a5/checkpoints/checkpoint-0010/model.safetensors`.
- Corpus: `data/learning-v2/cycle-b2-joint-v4` (manifest/merge-report already committed), `data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected`.
- No locked-test data was opened to produce any number in this report.
