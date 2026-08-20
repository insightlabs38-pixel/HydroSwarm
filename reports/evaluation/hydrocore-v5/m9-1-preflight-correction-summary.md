# Milestone 9.1 preflight correction summary

Correction record: `docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_CORRECTION.md`. Original frozen protocol (`docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md`) is unmodified. Original preflight results (`m9-1-preflight-summary.md`, `m9-1-preflight-results.json`, `m9-1-architecture-seam.md`, `m9-1-solver-feasibility.json`, `m9-1-parameter-matching.json` as committed at `9c7fc06`) are preserved verbatim, not overwritten or deleted -- this document records what changed and why, on top of that history.

Starting SHA for this correction: `9c7fc06926b2bfa5eb7feedf46345dd1298a2ac3` (working tree clean at start, matching the original preflight's own final SHA).

## Historical sequence

1. original preflight protocol frozen (`784d8b2`);
2. preflight implementation, tests, results committed (`9c7fc06`);
3. post-preflight code review identified two implementation discrepancies against the already-frozen protocol -- before any predictive M9.1 experiment, using no predictive data;
4. this correction (this document + the code/test changes it describes).

## Issue 1: ODE/SDE initial state now uses first-valid evidence, not a whole-prefix pool

**Original**: `_PooledEvidenceInitialState` masked-mean-pooled every causally-available temporal/quality observation across the whole prefix into `h0`, for both `GraphODEDynamics` and `GraphSDEDynamics` -- a mismatch against the frozen protocol's own stated intent ("encode the FIRST available observation... integrate through the causal prefix's actual observation timestamps").

**Corrected**: `_FirstValidEvidenceInitialState`, used identically by both arms. For each batch item / node / modality independently, uses ONLY that modality's own first causally valid observation (earliest step index where `sensor_mask`/`quality_mask` is `True`) -- never array position 0 blindly, never a pooled/averaged combination, never a later step's value. The two modalities' first-valid indices may differ and are resolved independently. A node with no valid observation in a modality gets a deterministic zero vector for that component.

**Tests added** (`tests/scientific/test_m9_1_preflight.py`), all passing:

| test | proves |
|---|---|
| `test_first_valid_selects_first_valid_step_not_array_position_zero` | first-VALID selection, not blind array position 0 -- one node's step 0 is invalid and must be skipped, another node's step 0 is valid and must be used directly |
| `test_first_valid_all_missing_modality_gives_deterministic_zero` | a modality with no valid step gets an exact, finite zero vector |
| `test_initial_state_later_values_do_not_change_h0` | perturbing an invalid step or a later valid-but-not-first step leaves h0 unchanged; perturbing the actual first-valid step DOES change h0 (non-vacuousness check) |
| `test_initial_state_temporal_and_quality_first_valid_may_differ` | temporal first-valid at step 1 and quality first-valid at step 3 are resolved independently, each modality sensitive only to its own first-valid step |
| `test_ode_initial_state_uses_first_valid_evidence_end_to_end` | the same property holds through `GraphODEDynamics.initial_state` directly, not just the standalone helper |
| `test_sde_shares_first_valid_initial_state_semantics_with_ode` | `GraphODEDynamics` and `GraphSDEDynamics` both construct `_FirstValidEvidenceInitialState` -- identical initial-state semantics, differing only in post-h0 evolution |

Parameter count: unchanged for both arms (`_FirstValidEvidenceInitialState.projection` is the same `Linear(temporal_feature_dim + quality_feature_dim, d_model)` shape `_PooledEvidenceInitialState.projection` was) -- confirmed directly, no re-matching was needed or performed for GRAPH_ODE/GRAPH_SDE.

## Issue 2: GRAPH_CDE control path is now mask-aware

**Original**: `GraphCDEDynamics._build_path_values` accepted `sensor_mask`/`quality_mask` as arguments but never used them -- the control path `X_t` represented a genuinely-observed zero identically to a missing, zero-filled position.

**Corrected**: two explicit validity channels, `sensor_valid`/`quality_valid` (`0.0`/`1.0` per `[batch, time, node]`, via the same `_valid_mask` convention already used elsewhere in the module), appended to `X_t` alongside `relative_time`/`temporal_features`/`quality_features`. `input_channels = 1 + temporal_feature_dim + quality_feature_dim + 2`. No other channel was added.

**Tests added**, all passing:

| test | proves |
|---|---|
| `test_cde_control_path_includes_sensor_and_quality_valid_channels` | the last two channels of `_build_path_values`'s output equal `sensor_mask`/`quality_mask` exactly |
| `test_cde_observed_zero_differs_from_missing_zero` | two cases with numerically identical (0.0) feature values but different validity produce identical feature channels but DIFFERENT validity channels, and a finite nonzero difference in the CDE's actual output -- proving the distinction is load-bearing, not merely plumbed |
| `test_cde_missing_intermediate_report_stays_finite_and_causal` | a valid/missing/valid path produces finite interpolation coefficients, finite forward output, finite backward gradients, a correctly-marked missing knot, and unbroken future-evidence causality |
| `test_cde_depth_one_preserves_evidence_and_validity_exactly` | the depth-1 synthetic two-point expansion never promotes an invalid single observation to valid |

The pre-existing mandatory causality test (`test_cde_causality_future_observation_cannot_affect_earlier_cutoff`, including its non-vacuousness check) still passes unmodified with the new channels present.

## Parameter matching (re-run, mechanical count-only search, no predictive data)

| arm | original mlp_width | original params | corrected mlp_width | corrected params | corrected delta % |
|---|---|---|---|---|---|
| CURRENT (baseline) | -- | 4,182,612 | -- | 4,182,612 (unchanged) | -- |
| GRAPH_ODE | 574 | 4,184,118 | 574 (unchanged) | 4,184,118 (unchanged) | +0.036% |
| GRAPH_CDE | 242 | 4,183,082 | **214** | **4,182,894** | **+0.0067%** |
| GRAPH_SDE | 464 | 4,183,540 | 464 (unchanged) | 4,183,540 (unchanged) | +0.022% |

Only GRAPH_CDE's width changed (its input-channel count grew by 2, changing `GraphCDEField`'s weight shapes); GRAPH_ODE/GRAPH_SDE were re-verified, not re-searched, since `_FirstValidEvidenceInitialState` has the identical parameter shape as `_PooledEvidenceInitialState`. Every novel arm remains within ±5% (in fact within ±0.04%).

## Physical-time / structural / SDE regression retests (all still passing)

- ODE/CDE/SDE timestamp-origin invariance (+1h/+24h/+7d, SDE with the same Brownian seed for both cases): `<=1e-4` -- unchanged from original preflight, re-verified.
- ODE physical-gap plumbing (10min/1hr/10hr produce different results, no per-window normalization) and depth-1 zero-duration identity-on-h0: re-verified.
- ODE/CDE node-permutation invariance, SDE drift/diffusion-function permutation equivariance and explicit-noise-permuted full-path equivariance: re-verified.
- SDE Test A (fixed-seed reproducibility, exact match), Test B (different-seed stochasticity, finite nonzero), Test C (zero-diffusion collapse to the manual deterministic-drift reference, `<=1e-4`): re-verified. `diffusion_scale=0.1`, `dt=0.05`, `method="euler"` unchanged.
- CURRENT baseline equivalence (`HydroCore(temporal_dynamics=None)` unaffected; `CurrentTemporalDynamics` wrapper matches vanilla `HydroCore` within `<=1e-6`): re-verified, unchanged.

## Forward/backward finiteness and mini-overfit sanity (re-run, loss only, no accuracy)

| arm | forward finite | backward finite | grads nonzero | initial loss | final loss (20 steps) |
|---|---|---|---|---|---|
| CURRENT | True | True | True | 0.6828 | 0.0056 |
| GRAPH_ODE | True | True | True | 0.5544 | 0.0023 |
| GRAPH_CDE | True | True | True | 0.6900 | 0.0021 |
| GRAPH_SDE | True | True | True | 0.9198 | 0.0029 |

No top1/MRR/development-accuracy metric was computed for any arm.

## Latency refresh (engineering only, same predeclared thresholds, no new thresholds)

| arm | forward median (s) | ratio to CURRENT | feasibility |
|---|---|---|---|
| CURRENT | 0.0161 | 1.00x | -- |
| GRAPH_ODE | 0.0269 | 1.68x | PRACTICAL |
| GRAPH_CDE | 0.1216 | 7.57x | EXPENSIVE_BUT_TESTABLE |
| GRAPH_SDE (single path) | 0.0569 | 3.54x | EXPENSIVE_BUT_TESTABLE |

Same predeclared thresholds as the original preflight (PRACTICAL `<=3x`, EXPENSIVE_BUT_TESTABLE `3x-15x`, PREFLIGHT_BLOCKED beyond) -- not redefined for this correction. GRAPH_CDE's ratio increased modestly (6.82x -> 7.57x) from the two additional control channels, still comfortably `EXPENSIVE_BUT_TESTABLE`, nowhere near `PREFLIGHT_BLOCKED`.

## SDE Monte Carlo policy: unchanged, MC=4 remains the recommendation

MC1/4/8 stabilization was re-measured (not re-searched): mean latent summary 0.28891 (MC1) -> 0.28884 (MC4) -> 0.28879 (MC8), stable to 3 significant figures by MC4, variance growing only modestly from MC4 to MC8 -- the same pattern the original preflight found. Per Section 20 of the correction milestone instructions, the MC-count search was not reopened, and MC=4 remains the proposed full-M9.1 evaluation count.

**Recorded requirement for the full M9.1 scientific protocol** (per Section 20): before training/evaluation begins, that protocol must freeze an explicit deterministic Brownian seed schedule, conceptually `brownian_seed = stable_hash(predictor_training_seed, incident_id, prefix_depth, mc_index)` -- not implemented or exercised here, since doing so would require touching development-scoped incident identifiers, out of this correction's scope.

## Validation

- Correction-targeted tests (the 10 new tests listed above): 10/10 passed.
- Full `tests/scientific/test_m9_1_preflight.py` (original 37 + new 10): **47/47 passed**.
- `python3 -m pytest tests/ -q` (full repo suite): see final validation numbers below.
- `ruff check` on every changed/new file (`src/hydroswarm/model/continuous_time.py`, `tests/scientific/test_m9_1_preflight.py`, `scripts/hydrocore_v5/run_m9_1_preflight.py`): clean. Repo-wide `ruff check`: the same 8 pre-existing errors as the original preflight, in the same untouched files -- not opportunistically fixed, per Section 23.
- `pyright`: repo-wide, 0 errors, 0 warnings, 0 informations.
- `locked_test_opened`: `False` before and after every script/test run in this correction. `development_holdout` was never loaded. No M9.1 localization/accuracy number exists anywhere in this correction's artifacts.

## Corrected arm verdicts (Section 24 gates)

| arm | verdict |
|---|---|
| CURRENT | BASELINE_VALID |
| GRAPH_ODE | **PREFLIGHT_PASS** |
| GRAPH_CDE | **PREFLIGHT_PASS** |
| GRAPH_SDE | **PREFLIGHT_PASS** |

All gates listed in Section 24 of the correction milestone instructions pass for every arm, including the two new correction-specific gates (first-valid initial-state semantics for ODE/SDE; mask-aware control path with proven observed-zero-vs-missing-zero distinction for CDE).

## FINAL CORRECTION DECISION

    M9_1_PREFLIGHT_CORRECTED = YES
    M9_1_FULL_EXPERIMENT_READY = YES

M9_1_ARMS = CURRENT, GRAPH_ODE, GRAPH_CDE, GRAPH_SDE. No further fundamental representation/correctness problem was uncovered. No architecture family added. No predictive-performance number was computed or inspected for any arm at any point in this correction.

locked tests opened: before=False, after=False. No model promoted. No safety/authority semantics changed. No S/M/L capacity scaling begun. No scientific M9.1 comparison training begun.
