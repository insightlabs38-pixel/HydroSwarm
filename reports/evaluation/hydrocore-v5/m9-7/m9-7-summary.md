# Milestone 9.7 summary: HydroCore-M capacity preflight and M9.8 freeze

PRE-FLIGHT / PROTOCOL-FREEZE milestone. No HydroCore-M predictive/development performance was observed, computed, or inspected anywhere in this milestone. The S-vs-M capacity comparison itself is Milestone 9.8, not run here.

Full protocol: `docs/evaluation/HYDROCORE_V5_M9_7_PROTOCOL.md`.

## Inherited state

M9.6 closed `M9_6_DECISION=A`, `HYDROCORE_S_STATUS=FROZEN`, recipe `CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING + B_DEPTH_AWARE_CALIBRATION + ALPHA_0_1 + SOURCE_REPRESENTATIVE_CALIBRATION_SUPPORT_20_PER_SOURCE`. M9.7 inherits this unchanged.

## Frozen HydroCore-S (reconstructed and verified)

`HydroCore.from_variant("small", ...)` -- **4,182,612** trainable parameters, verified by direct instantiation against the actual M9.6 training scripts' construction call (not estimated). Full inventory: `m9-7-frozen-s-inventory.json`.

## Selected HydroCore-M

Deterministic, parameter-count-only sweep (fixed `num_layers=4`, `latent_tokens=64`, `modality_layers=1`, `head_dim=32`, `dim_feedforward/d_model=3.0`, all identical to S; `d_model` swept over multiples of 32) produced **exactly one** candidate inside the 12-16M target band: **`d_model=352, nhead=11, dim_feedforward=1056` -> 13,919,572 parameters** (80,428 from the predeclared 14M center, 0.57%). No predictive signal was computed for any candidate. Registered as `MODEL_VARIANTS["small_v5_capacity_m"]` in `src/hydroswarm/model/core.py` (additive; `"small"`/`"medium"`/`"large"` unmodified). Full enumeration and every rejected alternative (including the pre-existing `"medium"` variant, rejected for doubling `num_layers`): `m9-7-capacity-candidates.json`, `m9-7-selected-m-architecture.json`.

**Parameter ratio M/S: 3.328x.**

## Capacity/semantic parity

M differs from S in `d_model`/`nhead`/`dim_feedforward` only -- every other dimension (depth, latent tokens, modality layers, input/output schemas, AGE_FIX_ONLY, all `SHARED_MODEL_CONFIG` flags) is identical. **`M9_7_CAPACITY_ISOLATION_FAILED = FALSE`.** Details: `m9-7-semantic-parity-audit.json`, `m9-7-parameter-counts.json`.

## M9.8 freeze

Full training-parity plan (`m9-7-training-parity-plan.json`) and scientific preregistration (`m9-7-m9-8-preregistration.json`) committed in this milestone, before any HydroCore-M development performance exists:

- **Optimizer steps: S=1350, M=1350** (identical). Compute-comparability check: `_scheduler`'s cosine trajectory is a pure function of `(config, total_steps)`, independent of model width -- no blocker found.
- **Seeds**: 20260814, 31874, 20260815 (unchanged from M9.6).
- **Primary endpoint**: unseen-topology MATURE neural Top-1, macro-averaged across coastal-branch/tree-branch/dense-loop, ARM_M - ARM_S.
- **Statistics**: paired bootstrap, 2000 resamples, 90% CI, resampling unit=incidents, **bootstrap seed 20260819** (verified unused in this repository at freeze time).
- **Practical-effect threshold**: no repo-preregistered numeric threshold exists (only `/workspace/experiments.txt` Milestone 9.2's qualitative worked examples); this document adopts `M-S >= +0.02` absolute AND 90% CI lower bound `>0`, plus guardrails B-F (family/seed consistency, known-family retention, calibration >=0.85, engineering). Consistent with, not weaker than, the qualitative precedent.
- **HydroCore-L**: `HYDROCORE_L_AUTHORIZED = FALSE`. Not authorized by any outcome of M9.8 alone.

## Engineering preflight

All 22 required checks pass (`m9-7-engineering-smoke.json`, `m9-7-test-results.json`): instantiation, parameter count, output-shape/feature-schema parity, variable topology, permutation equivariance, causal masking, AGE_FIX_ONLY preservation, full-gradient wiring, head-dimension parity, interleaved microbatching + gradient accumulation (real `step_matched_interleaved_optimizer_step`, unmodified), 1350-step/scheduler representability, checkpoint save/load + resume identity, calibration-interface acceptance (unmodified `SplitConformalCalibrator`), and locked-split state unopened before/after. `pyright`: 0 errors on all touched files. Targeted regression suites (model, permutation, split policy, M9.4/M9.5R/M9.6 scientific tests, calibration): 155 passed. Full-repository collection: 1475 tests, 0 collection errors.

## Latency/memory (engineering only, measured after M was frozen)

`m9-7-latency-memory.json`: M/S parameter ratio 3.33x; median CPU forward-latency ratio ~1.13x; checkpoint-size ratio ~3.32x. No cost threshold imposed (none exists in the repository to inherit).

## Locked splits

`locked_test_opened`: **before = False, after = False**, throughout.

## M9_7_DECISION: A (M9_8_CAPACITY_EXPERIMENT_READY)

Deterministic capacity selection, full semantic/capacity isolation, all engineering preflight checks, and the complete M9.8 scientific/training freeze are all green. HydroCore-L remains unauthorized. No HydroCore-M development performance was observed. M9.8 is NOT started by this document.
