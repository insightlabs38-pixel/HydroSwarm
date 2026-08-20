# Milestone 9.1 preflight summary: continuous-time architecture testbed feasibility

> **SUPERSESSION / CORRECTION NOTE:** post-preflight code review (after this document and the commit it describes, `9c7fc06`, were already committed) found that the shipped implementation did not fully match this document's own frozen protocol on two points: (1) GRAPH_ODE/GRAPH_SDE's initial state pooled the whole causal prefix instead of using first-valid evidence only, and (2) GRAPH_CDE's control path did not represent `sensor_mask`/`quality_mask`, so it could not distinguish an observed zero from a missing one. Both were fixed with no predictive data inspected. The measurements below (in particular GRAPH_CDE's parameter count, `mlp_width`, and latency, which changed because its control path gained two validity channels) are the ORIGINAL, PRE-CORRECTION numbers -- preserved verbatim below for historical record, not edited in place. See `docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_CORRECTION.md` and `reports/evaluation/hydrocore-v5/m9-1-preflight-correction-summary.md` for the corrected implementation, corrected measurements, and the final (still `M9_1_FULL_EXPERIMENT_READY = YES`) decision.

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md`. Architecture-seam basis: `reports/evaluation/hydrocore-v5/m9-1-architecture-seam.md`. This is an ENGINEERING-CORRECTNESS milestone only -- it establishes that CURRENT/GRAPH_ODE/GRAPH_CDE/GRAPH_SDE are correctly implemented, causal, numerically stable, and parameter-matched. It computes and reports **zero** localization/accuracy numbers for any candidate and is structurally unable to reveal which architecture performs best.

Starting SHA: `c7f7bddba9513e748185cd53fde6c003e7213c79` (`exp/hydrocore-v5-causal`, matching M9.0b's own closing commit, working tree clean at start).

## Environment / dependencies

- Python 3.12.13, PyTorch 2.13.0 (CPU; `configs/training-v5-causal.yaml`'s own `device: cpu`, `fp32: true`, `deterministic: true` convention followed throughout).
- `torchdiffeq==0.2.5` (MIT), `torchcde==0.2.5` (Apache-2.0), `torchsde==0.2.6` (Apache-2.0) -- all installed and functional; all imported lazily inside `src/hydroswarm/model/continuous_time.py` (try/except), added as a new, non-default `continuous-time` optional-dependency group in `pyproject.toml`. Normal production `import hydroswarm` is unaffected by their absence.

## Architecture seam and wiring

`HydroCore.__init__` gained exactly one new, strictly additive constructor argument: `temporal_dynamics: TemporalDynamicsBase | None = None`. Default `None` (every existing caller) reproduces today's exact `temporal_encoder`/`quality_encoder` behavior; `test_hydrocore_default_construction_unaffected_by_new_parameter` and `test_current_wrapper_matches_vanilla_hydrocore` verify byte-level equivalence directly. When supplied, `forward()` routes `edge_index`/`edge_features`/`edge_mask`/`node_mask` into the new module (previously read only inside the spatial backbone) so the continuous-time vector fields are graph-aware, unlike the current per-node-only `TemporalEncoder`/`QualityEncoder`. No other module changed. Full detail in the architecture-seam report.

## Frozen technical choices actually used

- **Time semantics**: `t = (timestamps_seconds - timestamps_seconds[:, :1]) / 86_400.0` (`compute_relative_physical_time`, reusing `encoders.py`'s existing `FIXED_ELAPSED_TIME_SCALE_SECONDS` constant).
- **GRAPH_ODE**: `torchdiffeq.odeint`, `method="dopri5"`, `rtol=1e-3`, `atol=1e-4`. Graph-aware drift via a shared `GraphVectorField`-style scatter-mean-over-`edge_index` aggregator (same convention as `EdgeAwareGraphConv`).
- **GRAPH_CDE**: `torchcde.linear_interpolation_coeffs` / `LinearInterpolation` (frozen for strictly local support -- the only interpolation choice that makes the causality gate provable rather than approximately true), solved via `torchcde.cdeint(method="rk4", options={"step_size": 0.25})`. Causal cutoff (`cutoff_index`) filters the control path BEFORE interpolation coefficients are built, never after. Depth-1: a single observation is held constant across a synthetic two-point path (finite, deterministic).
- **GRAPH_SDE**: `torchsde.sdeint`, `method="euler"`, `sde_type="ito"`, `noise_type="diagonal"`, seeded `torchsde.BrownianInterval`. Diffusion `diffusion_scale * sigmoid(diffusion_net(h))`, diagonal/node-local, bounded and non-negative by construction. Drift reuses the same `GraphVectorField`-style mechanism as GRAPH_ODE.
- **Parameter-matching search**: deterministic integer-width sweep on each candidate's shared MLP width, minimizing `abs(candidate_total - baseline_total)` -- a pure counting search using no predictive data.

## Parameter matching (measured, not tuned)

| arm | mlp_width | total params | delta vs. baseline |
|---|---|---|---|
| CURRENT (baseline) | -- | 4,182,612 | -- |
| GRAPH_ODE | 574 | 4,184,118 | +0.036% |
| GRAPH_CDE | 242 | 4,183,082 | +0.011% |
| GRAPH_SDE | 464 | 4,183,540 | +0.022% |

Every novel arm within ±5% (in fact within ±0.04%, comfortably inside the ±2% preferred bound).

## Correctness / invariance / causality tests

`tests/scientific/test_m9_1_preflight.py`: **37/37 passed**, all on synthetic/hand-built tensors only (no `development_holdout`/`calibration`/locked data). Covers, per arm as applicable: relative-physical-time translation invariance and duration-preservation; ODE physical-gap plumbing and deterministic repeatability; **CDE future-evidence causality (mandatory gate) -- passed at `<=1e-6`**, with an explicit non-vacuousness check confirming the SAME appended future observation DOES change the uncapped ("evaluate now") output by `>1e-4`, proving the causality pass is not merely because the feature is ignored; CDE depth-1 degenerate-control finiteness; SDE Tests A (fixed-seed reproducibility, exact match), B (different-seed stochasticity, finite nonzero difference), and C (zero-diffusion collapse to a manually-computed deterministic-drift reference, `<=1e-4`); SDE permutation-equivariance of drift/diffusion under consistently-permuted noise; timestamp-origin invariance (+1h/+24h/+7d) for ODE/CDE/SDE at `<=1e-4` (SDE using the same Brownian seed for both cases); node-permutation invariance for ODE/CDE; parameter-count guardrails; irregular-timestamp robustness (jittered, unequal gaps, duplicated timestamp, large gap) with no NaN/Inf for all three arms; forward+backward finiteness with nonzero gradient reaching both the temporal-dynamics parameters and the output heads, for CURRENT/ODE/CDE/SDE end-to-end through real `HydroCore`.

Independently reproduced (not merely accepted from the implementation run): `pytest tests/scientific/test_m9_1_preflight.py -v` re-run directly, 37/37 passed; `ruff check` and `pyright` both clean on every new/changed file, and repo-wide `pyright` returns 0 errors/warnings.

## Forward/backward finiteness and mini-overfit sanity (loss only, no accuracy)

| arm | forward finite | backward finite | grads nonzero (dynamics + heads) | initial loss | final loss (20 steps) |
|---|---|---|---|---|---|
| CURRENT | True | True | True | 0.7433 | 0.0117 |
| GRAPH_ODE | True | True | True | 0.5377 | 0.0023 |
| GRAPH_CDE | True | True | True | 0.5939 | 0.0035 |
| GRAPH_SDE | True | True | True | 0.9322 | 0.0040 |

All four arms' loss decreases over 20 optimizer steps on a tiny synthetic batch, confirming training mechanics function. No top1/MRR/development-accuracy metric was computed for any arm, per protocol Section 9.

## Latency / memory (CPU, engineering feasibility only -- never a promotion criterion)

| arm | forward median (s) | forward+backward median (s) | ratio to CURRENT | feasibility |
|---|---|---|---|---|
| CURRENT | 0.0175 | 0.0428 | 1.00x | -- |
| GRAPH_ODE | 0.0291 | 0.0791 | 1.67x | PRACTICAL |
| GRAPH_CDE | 0.1194 | 0.3833 | 6.82x | EXPENSIVE_BUT_TESTABLE |
| GRAPH_SDE (single path) | 0.0546 | 0.0895 | 3.12x | EXPENSIVE_BUT_TESTABLE |

Classification thresholds are the ones predeclared in the frozen protocol (Section 11): PRACTICAL `<=3x`, EXPENSIVE_BUT_TESTABLE `3x-15x`, PREFLIGHT_BLOCKED beyond that. No arm approaches the BLOCKED threshold; GRAPH_CDE's adaptive-control-path solve (rk4, fixed small step size over the interpolation's own index axis) is the most expensive single-path arm but remains comfortably testable for a three-seed scientific run.

## SDE Monte Carlo stabilization (engineering only, variance-vs-cost, never accuracy-tuned)

| MC count | runtime (s) | mean latent summary | variance estimate |
|---|---|---|---|
| 1 | 0.0309 | 0.151163 | 0.0 (undefined at N=1) |
| 4 | 0.1122 | 0.151311 | 0.0001337 |
| 8 | 0.2472 | 0.151297 | 0.0001536 |

The mean latent summary is already stable to 3 significant figures by MC=4 (0.151311 vs. 0.151297 at MC=8, a 0.01% relative change), and the variance estimate grows only modestly (~15% relative) from MC=4 to MC=8. **Recommended full-M9.1 SDE Monte Carlo count: 4** -- the smallest tested count giving mechanically stable aggregation, per protocol Section 11's "smallest count that gives mechanically stable aggregation" rule, chosen on this variance/cost evidence alone, never on predictive accuracy (none was computed).

## Arm verdicts (Section 29 gate, frozen criteria)

| arm | verdict |
|---|---|
| CURRENT | BASELINE_VALID |
| GRAPH_ODE | **PREFLIGHT_PASS** |
| GRAPH_CDE | **PREFLIGHT_PASS** |
| GRAPH_SDE | **PREFLIGHT_PASS** |

All ten Section-29 criteria (parameter count, forward/backward finiteness, gradient flow to intended dynamics parameters, timestamp-origin invariance, graph structural invariance, causal-prefix behavior, physical-time plumbing, functional solver dependency, feasible compute, preserved safety interfaces) pass for every novel arm, plus GRAPH_CDE's additional future-evidence-causality gate and GRAPH_SDE's additional fixed-seed/different-seed/zero-diffusion gates. No arm is `PARAMETER_MATCH_BLOCKED` or `PREFLIGHT_BLOCKED`.

## Calibration / OOD / fusion / authority compatibility (checked, not exercised)

Every candidate emits the same `[batch, nodes, d_model]`-shaped latents feeding the SAME, unmodified `source_node_head`/`modality_fusion`/output-head graph as CURRENT -- the `source_node` probability-vector shape `B_DEPTH_AWARE` requires is unchanged for every arm. No calibrator was fit. Classical localization, neural/classical fusion, disagreement calculation, OOD scoring, `PlanVerifier`, WNTR exact verification, sampling/planning authority, and every fail-closed gate are untouched -- no architecture required changing any of them merely to function.

## Validation

- `tests/scientific/test_m9_1_preflight.py`: 37/37 passed.
- `python3 -m pytest tests/ -q` (full repo suite): **1224 passed, 1 skipped** (the pre-existing, unrelated `test_capability_diagnostic.py` PR#12 skip), 0 failed. One genuine, unrelated-to-architecture finding surfaced and fixed during this milestone: a repo-wide broken-relative-link check (`tests/unit/test_release_docs_link_audit.py`) flagged a false-positive link pattern (a closing bracket immediately followed by an opening paren around the word `hidden`) inside a code span in the draft architecture-seam report's Markdown table -- a pre-existing regex limitation in that checker (it does not respect backtick code spans), triggered by this milestone's own new doc, not a defect in any other file; fixed by rewording the offending table cell, re-verified green.
- `ruff check` on every new/changed M9.1 file: clean. Repo-wide `ruff check`: 8 pre-existing errors, all in files this milestone did not touch (`scripts/hydrocore_v5/evaluate_m2.py`, `run_m2_arm.py`, `run_m5_sampling.py`, `run_m8_5_hydraulic_backends.py`, `run_m8_scaling.py`, `src/hydroswarm/training/causal_prefix.py`) -- pre-existing repo lint debt, out of this milestone's scope, left untouched rather than opportunistically fixed.
- `pyright`: repo-wide, **0 errors, 0 warnings, 0 informations**.
- `locked_test_opened`: `False` before and after every script/test run in this milestone (checked via `hydroswarm.evaluation.live_robustness.locked_test_opened`, reading `reports/results/v4/architecture-freeze.json`). `development_holdout` was never loaded; no coastal-branch/tree-branch/dense-loop topology data was touched. No M9.1 localization score table exists anywhere in this milestone's artifacts.

## Proposed full M9.1 configuration (frozen NOW, not executed in this milestone)

- **Arms**: CURRENT, GRAPH_ODE, GRAPH_CDE, GRAPH_SDE.
- **Parameter counts**: CURRENT 4,182,612; GRAPH_ODE 4,184,118 (+0.036%); GRAPH_CDE 4,183,082 (+0.011%); GRAPH_SDE 4,183,540 (+0.022%).
- **Time units**: `t = (timestamps_seconds - timestamps_seconds[:, :1]) / 86_400.0` (days), identical convention for all three continuous-time arms.
- **Solver configurations**: GRAPH_ODE -- `torchdiffeq.odeint`, `dopri5`, `rtol=1e-3`, `atol=1e-4`. GRAPH_CDE -- `torchcde.cdeint`, linear interpolation, `rk4`, `step_size=0.25`, causal cutoff filtering before interpolation. GRAPH_SDE -- `torchsde.sdeint`, `euler`, `ito`, diagonal diffusion `diffusion_scale * sigmoid(diffusion_net(h))`.
- **SDE Monte Carlo evaluation count**: 4 (fixed), fixed seed schedule matching the screening/promotion seeds below, averaged predictive probability/posterior with stochastic variance reported separately.
- **Representation**: `AGE_FIX_ONLY` (frozen at M8.7, unchanged; `unobserved_age_sentinel="fixed"`).
- **Topology recipe**: `SINGLE_FAMILY_CURRENT_TRAINING` (frozen at M9.0b; golden-reference-only training, matching CURRENT's own established recipe -- the interleaved multi-family predictor remains scientifically valid but not operationally promoted pending calibration resolution).
- **Calibration method**: `B_DEPTH_AWARE`, fit independently per architecture per predictor seed from that architecture's own governed calibration probabilities -- never reusing CURRENT's quantiles for a candidate.
- **Alpha**: 0.1 (frozen, unconditional).
- **Seed plan**: screening seeds `20260814`, `31874` for all four arms; promotion-confirmation seed `20260815` required only for whichever arm(s) the screening pass provisionally selects, matching every prior v5 milestone's own seed discipline. No predictive promotion threshold is frozen here -- that belongs in the full M9.1 protocol, written only after this preflight, following the same freeze-before-evaluation discipline this document itself followed.
- **Expected training budget / latency estimates**: at these measured single-forward-pass ratios (CURRENT 1.00x, GRAPH_ODE 1.67x, GRAPH_CDE 6.82x, GRAPH_SDE 3.12x single-path / ~4x expected at MC=4 averaging), and given `configs/training-v5-causal.yaml`'s existing 20-epoch/`batch_size=2`/`gradient_accumulation_steps=4` CPU training budget (the same budget every prior v5 arm has used), GRAPH_ODE and GRAPH_SDE remain within a modest multiple of CURRENT's own established training wall-clock; GRAPH_CDE is the most expensive arm and should be budgeted accordingly in the full M9.1 protocol's own compute planning (still classified EXPENSIVE_BUT_TESTABLE, not blocked).

## Research scope

Frozen to exactly CURRENT / GRAPH_ODE / GRAPH_CDE / GRAPH_SDE. mTAN, GRU-ODE-Bayes, Neural PDE, temporal Transformer, graph Transformer, diffusion model, latent-ODE architecture zoo, and HPO over solver/model families remain explicitly out of scope, per the frozen protocol Section 1.

## FINAL M9.1 PREFLIGHT DECISION

    M9_1_FULL_EXPERIMENT_READY = YES

M9_1_ARMS = CURRENT, GRAPH_ODE, GRAPH_CDE, GRAPH_SDE. No arm blocked. No architecture-family expansion. No predictive-performance number was computed or inspected for any arm at any point in this milestone.

locked tests opened: before=False, after=False. No model promoted. No safety/authority semantics changed. No S/M/L capacity scaling begun. No scientific M9.1 comparison training begun.
