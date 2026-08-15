# HydroCore-v5 Milestone 9.1 preflight protocol (frozen before any candidate architecture is evaluated for predictive performance)

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md`. This document freezes the M9.1 PREFLIGHT sub-protocol -- architecture interface, continuous-time semantics, causality requirements, solver configuration, parameter-count bounds, reproducibility requirements, smoke-test data policy, and latency/memory reporting -- BEFORE any candidate architecture is trained on real predictive objectives or evaluated for localization/capability performance. It is not altered after seeing any predictive-performance number, because none is generated during preflight.

**This is NOT the scientific M9.1 comparison protocol.** It answers only "can we run the experiment correctly?", never "which architecture is best?". The scientific M9.1 protocol (arms, promotion gates, statistical comparison procedure) is written and frozen separately, after this preflight closes, following the same "freeze before evaluation" discipline this document itself follows relative to predictive performance.

## 0. Why this milestone exists

M9.0b (`docs/evaluation/HYDROCORE_V5_M9_0B_PROTOCOL.md`, `reports/evaluation/hydrocore-v5/m9-0b-summary.md`) closed with `INTERLEAVED_PREDICTOR_CALIBRATION_NOT_RESOLVED`: the validated +6.6pp interleaved topology-transfer predictor remains scientifically valid but is not operationally promoted, because its trained-family conformal coverage could not satisfy the frozen 0.85 safety gate under any of the four tested Mondrian grouping schemes. M9.0b's own recorded M9.1 recipe (representation `AGE_FIX_ONLY`, topology training `SINGLE_FAMILY_CURRENT_TRAINING`, calibration `B_DEPTH_AWARE`, `alpha=0.1`, `M9_1_SCIENTIFICALLY_UNBLOCKED: YES`) is the frozen entry state for M9.1.

M9.1 will eventually compare four continuous-time-latent-evolution architectures for the shared HydroCore backbone: CURRENT (baseline), Graph Neural ODE, Graph Neural CDE, Stable Graph Neural SDE. Before that comparison can be trusted, the testbed itself -- the interface, the time semantics, the causality guarantees, the solver configuration, and the parameter matching -- must be shown correct independent of any predictive-performance signal. That is this document's and this milestone's entire scope.

## 1. Research scope freeze

M9.1 architecture candidates are frozen to exactly:

- `CURRENT` -- the existing HydroCore temporal pathway, unmodified.
- `GRAPH_ODE` -- Graph Neural ODE.
- `GRAPH_CDE` -- Graph Neural CDE.
- `GRAPH_SDE` -- Stable Graph Neural SDE.

Explicitly out of scope, for M9.1 and this preflight: mTAN, GRU-ODE-Bayes, Neural PDE, temporal Transformer, graph Transformer, diffusion model, latent-ODE architecture zoo, HPO over solver/model families. This scope may reopen only if the full M9.1 experiment produces a specific unexpected result that leaves a mechanistic question genuinely unresolved -- "another model might perform better" is never sufficient reason on its own.

## 2. Architecture seam (frozen before implementation)

Full detail in `reports/evaluation/hydrocore-v5/m9-1-architecture-seam.md`. Summary: HydroCore's existing temporal pathway (`TemporalEncoder`/`QualityEncoder` in `src/hydroswarm/model/encoders.py`) runs once per example, per-node, with no cross-node graph interaction, entirely BEFORE the graph-aware spatial backbone (`LatentHydraulicBlock` x `num_layers`, in `src/hydroswarm/model/layers.py`) ever executes -- the backbone itself has no time axis. This is the smallest clean seam: replace only the mechanism that turns a per-node measurement history into a single `[batch, nodes, d_model]` latent, and leave the node encoder, graph-position encoder, `modality_fusion`, the spatial backbone, and every output head byte-identical.

`HydroCore.__init__` gains exactly one new, strictly additive, opt-in constructor argument: `temporal_dynamics: TemporalDynamicsBase | None = None`. When `None` (every existing caller, including every promoted checkpoint's reconstruction path), `HydroCore` builds and uses `self.temporal_encoder`/`self.quality_encoder` exactly as before -- zero behavior change, verified directly (Section 13 below). When provided, `HydroCore.forward()` calls `self.temporal_dynamics(...)` once, in place of the two separate encoder calls, and receives back `(temporal_latent, quality_latent)`, each `[batch, nodes, d_model]`, fed into the same, unmodified `modality_fusion` call. No architecture-specific output heads are introduced. Source-node target definition, multitask labels, classical-prior semantics, fusion semantics, OOD semantics, and planning semantics are unchanged for every arm.

`TemporalDynamicsBase` implementations live in `src/hydroswarm/model/continuous_time.py` (an experiment-scoped module with lazy, try/except imports of `torchdiffeq`/`torchcde`/`torchsde` -- absent by default from normal production `import hydroswarm`, matching the existing lazy-optional-dependency convention in `src/hydroswarm/simulation/wrapper.py`). No experimental architecture is wired into production runtime selection or any default factory path; no production model is promoted by this milestone.

## 3. Physical time representation (frozen)

```
t = (timestamps_seconds - timestamps_seconds[:, :1]) / FIXED_ELAPSED_TIME_SCALE_SECONDS
```

reusing the exact `FIXED_ELAPSED_TIME_SCALE_SECONDS = 86_400.0` constant already defined and already validated in `src/hydroswarm/model/encoders.py` (the M8.7 `fixed_scale`/`AGE_FIX_ONLY`-era convention), not a newly invented magnitude. `timestamps_seconds` are already incident-relative elapsed seconds by the time they reach the model (`HydraulicFeatureBuilder`'s own convention, confirmed in M8.6's triage -- not raw Unix epoch).

Required properties, all satisfied by construction:

1. Translating every timestamp by a constant does not change `t` (the `[:, :1]` subtraction happens first).
2. A 10-minute interval and a 10-hour interval remain physically different in `t` (fixed divisor, never `elapsed.abs().amax(...)`-style per-window rescaling).
3. No per-sequence duration normalization.
4. No raw Unix/epoch magnitude ever reaches the model.
5. This is the SAME mechanism that already fixed the M8.6 `ABSOLUTE_TIME_ORIGIN_LEAKAGE` finding for the `fixed_scale` encoder path -- not a new, unvalidated convention.

A single shared helper, `compute_relative_physical_time(timestamps) -> Tensor`, is the sole place this formula is implemented; every continuous-time variant calls it, so there is exactly one origin/scale convention to audit.

## 4. Graph Neural ODE (frozen design)

`dh_i/dt = f_theta(h_i, aggregate_neighbors(h_j, edge_ij), static_context_i)`, graph-aware by construction: `aggregate_neighbors` reuses the same scatter-mean-over-`edge_index` pattern already established by `EdgeAwareGraphConv` in `src/hydroswarm/model/layers.py` (mean aggregation over valid edges, gathered via `edge_index`/`edge_features`/`edge_mask`), not a flattened whole-graph MLP. Deterministic; no stochasticity. Initial latent state encodes the first available observation (or a learned zero-state absent any observation); integration proceeds through the causal prefix's actual observation timestamps in physical-time units from Section 3, evaluated at the final available timestamp to match HydroCore's existing "now" semantics.

Solver: `torchdiffeq.odeint`, `method="dopri5"`, `rtol=1e-3`, `atol=1e-4` (adaptive, standard default; not `odeint_adjoint` -- plain backprop-through-solver is adequate and simpler at this parameter scale). Solver settings may be adjusted only to correct NaN/Inf, non-convergence, or pathological runtime -- never to change predictive output quality -- and any such adjustment is recorded as an engineering correction in the results artifact, not silently applied.

## 5. Graph Neural CDE (frozen design)

`dh_t = f_theta(h_t, G) dX_t`, where `X_t` is the control path built ONLY from causally available evidence. Interpolation: **linear** (`torchcde.linear_interpolation_coeffs` / `torchcde.LinearInterpolation`), frozen specifically because linear interpolation has strictly local support -- a knot only influences its immediately adjacent interval -- unlike natural cubic splines, whose coefficients have global support and would let a future observation alter the reconstructed path at an earlier query time purely through the spline-fitting step, independent of solver behavior. This choice is made on correctness/numerical-feasibility grounds alone, per Section 7 of the milestone instructions, not by comparing interpolation methods on predictive accuracy.

Solver: `torchcde.cdeint(..., method="rk4")`, fixed-step (linear interpolation's derivative kinks at knots make fixed-step solvers the standard, better-behaved pairing over adaptive solvers).

**Strict causality (mandatory gate, Section 16 of the milestone instructions):** for a query/cutoff time `T`, the control path is built using only observations with `timestamp <= T` -- filtered BEFORE `linear_interpolation_coeffs` is called, never by truncating a full-sequence interpolation after the fact. Appending an observation at `T_future > T` must not change the output evaluated at `T`, to within `<=1e-6` preferred / `<=1e-5` maximum. This is verified directly in Section 12 below and in `tests/scientific/test_m9_1_preflight.py`; if it fails, the interpolation/filtering implementation is fixed before the arm is considered further -- the tolerance is never relaxed to make a failing implementation pass.

Depth-1 (a single available observation) is an explicitly defined, tested, deterministic degenerate-control case, not an unhandled edge case.

## 6. Stable Graph Neural SDE (frozen design)

`dh_t = f_theta(h_t, G, context) dt + g_phi(h_t, context) dW_t`. The drift `f_theta` is structurally the same `GraphVectorField`-style graph-aware mechanism as the ODE arm's drift, so the SDE arm's only qualitatively new mechanism relative to ODE is the diffusion term -- the SDE is not given a materially more expressive deterministic backbone than the ODE.

Diffusion structure: **diagonal / node-local** (`noise_type="diagonal"` in `torchsde` terms) -- never a dense or all-node-coupled covariance. Parameterization: a small per-node network producing `diffusion_scale * sigmoid(diffusion_net(h))`, bounded in `[0, diffusion_scale]` (non-negative, finite by construction), with `diffusion_scale` a small fixed engineering constant chosen only for finite dynamics/finite gradients/solver compatibility/reasonable stochastic magnitude relative to typical latent activation scale -- never tuned against predictive performance. The exact constant used is recorded in `reports/evaluation/hydrocore-v5/m9-1-preflight-results.json`.

Solver: `torchsde.sdeint`, `method="euler"`, `sde_type="ito"`, fixed step size `dt` (documented in the results artifact), Brownian path via `torchsde.BrownianInterval` with an explicit, threaded `seed` argument so fixed-seed reproducibility and different-seed stochasticity are both mechanically controllable and testable.

**Process uncertainty vs. observation noise (explicit distinction, Section 9 of the milestone instructions):** the SDE diffusion term represents latent/process uncertainty in how the hydraulic state evolves, not per-sensor measurement error -- sensor noise, dropout, and bias remain the conceptually separate domain of `quality_features` and the existing sensor-fault machinery, unchanged by this milestone. The diffusion term is never described as directly modeling every measurement error, and existing sensor-simulation semantics are not altered to make the SDE arm easier to train.

## 7. Parameter matching (frozen procedure, not a tuned result)

Baseline: `HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)` (the same config every prior v5 milestone has used, identical to the shipped v4 architecture) -- exactly 4,182,612 trainable parameters. For each of GRAPH_ODE/CDE/SDE, a deterministic integer-width search minimizes `abs(candidate_total_params - baseline_total_params)` where `candidate_total_params` is the FULL HydroCore parameter count with only `temporal_encoder`+`quality_encoder` replaced by that candidate's continuous-time module. This is a pure parameter-counting search -- it uses no predictive data of any kind, and Section 12 of the milestone instructions explicitly permits it for exactly this reason. Preferred bound: within ±2% of baseline. Absolute maximum: within ±5%. A candidate that cannot be brought within ±5% without destroying its basic architecture is reported `PARAMETER_MATCH_BLOCKED`, not forced to fit by enlarging CURRENT.

## 8. Numerical tolerances (predeclared here, not relaxed after results)

- Graph structural invariance (node-order permutation, node-ID relabeling), all three continuous-time arms: max abs latent/posterior discrepancy `<=1e-4`, matching the existing structural-invariance convention (`tests/unit/test_permutation.py`, M8.6/M8.7 Sections 3-6).
- Timestamp-origin invariance (+1h / +24h / +7d), all three continuous-time arms: max abs discrepancy `<=1e-4`. For SDE, both the original and translated case use the SAME Brownian seed/path.
- CDE future-evidence causality: `<=1e-6` preferred, `<=1e-5` maximum (Section 5 above).
- SDE zero-diffusion collapse toward the deterministic drift-only solution: tolerance predeclared and justified at implementation time (solver-method differences between the SDE integrator and the ODE integrator may prevent bit-exact equality even at zero diffusion); not loosened after seeing the actual gap.
- Determinism (CURRENT, ODE, CDE in eval mode): identical outputs across repeated calls with identical weights/inputs/timestamps/solver config, to numerical (float32) noise floor.

## 9. Smoke-test data policy (frozen)

Preflight code and tests use ONLY: synthetic/hand-constructed tensors, tiny hand-built graphs, and `train`-split examples via `hydroswarm.training.causal_prefix.build_scenario_pool("train", ...)` (golden-reference topology, disjoint seed range from every other split by construction). Preflight code and tests NEVER use `development_holdout`, `calibration`, `ood_development`, `locked_final_test`, or `locked_topology_test`, and never touch coastal-branch/tree-branch/dense-loop topology data. No M9.1 localization score table, top1/MRR/candidate-set number, or any other predictive-performance metric is computed or reported for any arm during preflight -- the preflight is structurally unable to reveal which architecture performs best, because it never computes a performance number for any of them.

`hydroswarm.evaluation.live_robustness.locked_test_opened(ROOT)` is checked and recorded `False` both before and after every preflight script run, exactly as every prior v5 milestone script already does.

## 10. Reproducibility requirements

- CURRENT, GRAPH_ODE, GRAPH_CDE in eval mode: bitwise-stable (to float32 noise floor) across repeated forward calls with identical weights/inputs/config.
- GRAPH_SDE: reproducible under a fixed Brownian seed/path (TEST A); demonstrably different (finite, nonzero) under a different seed with diffusion enabled (TEST B); collapses toward the shared deterministic drift solution at zero diffusion, within the Section 8 tolerance (TEST C).
- The mini-overfit sanity check (Section 11) uses the already-governed training learning rate from `configs/training-v5-causal.yaml` unless solver numerics make it literally nonfunctional, in which case the deviation is recorded as an engineering correction, not a tuning choice.

## 11. Latency / memory / SDE Monte Carlo reporting (frozen policy, thresholds predeclared before measurement)

Latency/memory is measured as an engineering feasibility check only, never as a promotion criterion. On the same representative small TRAIN/synthetic batch and hardware (CPU-only, per `configs/training-v5-causal.yaml`'s `device: cpu`): median and p90 forward latency, forward+backward latency, and whatever memory signal is measurable in a CPU-only sandbox (wall-clock-based proxy where a hardware memory counter is unavailable), normalized against CURRENT. Feasibility classification, predeclared before measurement:

- `PRACTICAL`: within a small constant factor of CURRENT (approximately <=3x), no special accommodation needed for a three-seed scientific run.
- `EXPENSIVE_BUT_TESTABLE`: materially slower (roughly 3x-15x CURRENT) but a three-seed run remains clearly achievable within available compute.
- `PREFLIGHT_BLOCKED`: cost is pathological enough (very roughly >15x CURRENT single-path, or SDE Monte-Carlo aggregation cost that would make even a single seed impractical) that the planned scientific run is not realistically achievable. An architecture is never rejected merely for being slower than CURRENT -- only for cost pathological enough to threaten the planned run.

SDE Monte Carlo policy: fixed counts 1, 4, 8 are measured on synthetic/TRAIN examples for stabilization of the mean latent summary, a variance estimate, and runtime -- chosen by variance-vs-cost engineering measurement only, never by predictive accuracy, per Section 22 of the milestone instructions. The recommended count for the full M9.1 protocol is the smallest of {1, 4, 8} that gives mechanically stable aggregation; 8 is not exceeded during preflight absent a compelling numerical reason. The proposed full-experiment semantics are: fixed MC count, fixed seed schedule, averaged predictive probability/posterior, with predictive stochastic variance reported separately -- no single Brownian draw ever determines an operational prediction.

## 12. Calibration and safety-interface compatibility (checked, not exercised)

No calibrator is fit during preflight. Every candidate architecture is confirmed to emit the same kind of `source_node` probability vector shape that `B_DEPTH_AWARE` (the frozen M9.0b calibration method) requires, so the full M9.1 protocol can fit `B_DEPTH_AWARE` independently per architecture per predictor seed, using that predictor's own governed calibration probabilities -- never reusing CURRENT HydroCore's quantiles for ODE/CDE/SDE. The candidate interface is confirmed not to require changing classical localization, neural/classical fusion, disagreement calculation, OOD scoring, `PlanVerifier`, WNTR exact verification, sampling authority, planning authority, or any fail-closed gate; if an architecture required such a change merely to function, that would be reported as a scope/interface problem, not implemented.

## 13. Baseline-preservation requirement

`CURRENT_HYDROCORE` (a vanilla `HydroCore(...)` instance built with `temporal_dynamics=None`, the default for every existing caller) must remain byte/functionally equivalent to today's shipped-architecture behavior. This milestone does not alter `HydroCore`'s default forward-pass output for any existing caller. Where a CURRENT-arm wrapper is used for uniformity in preflight scripts/tests, its output is verified against vanilla `HydroCore` output within `<=1e-6` on representative inputs -- this milestone must not accidentally change the scientific control that every future M9.1 arm is compared against.

## 14. Arm verdicts (frozen criteria, Section 29 of the milestone instructions)

`CURRENT` receives `BASELINE_VALID` by construction (Section 13). Each of `GRAPH_ODE`/`GRAPH_CDE`/`GRAPH_SDE` receives an independent `PREFLIGHT_PASS` or `PREFLIGHT_BLOCKED` verdict, passing only if: (1) parameter count within ±5% baseline; (2) forward/backward finite; (3) gradients reach the intended dynamics parameters; (4) timestamp-origin invariance passes; (5) graph structural invariance passes; (6) causal-prefix behavior passes; (7) physical-time plumbing passes; (8) the solver dependency is functional; (9) compute is feasible enough for the planned three-seed scientific run; (10) the architecture preserves every existing safety interface (Section 12). GRAPH_CDE additionally requires the future-evidence-causality gate (Section 5). GRAPH_SDE additionally requires fixed-seed reproducibility, different-seed stochasticity, structured (diagonal, bounded) diffusion, and the zero-diffusion controlled-collapse test. Predictive accuracy is never a preflight criterion for any arm.

If all four arms are ready, `M9_1_FULL_EXPERIMENT_READY = YES` with arms `{CURRENT, GRAPH_ODE, GRAPH_CDE, GRAPH_SDE}`. If exactly one novel arm is blocked for a fundamental engineering reason, the remaining arms may still proceed if the causal ladder (CURRENT vs. deterministic continuous evolution vs. observation-driven continuous control vs. stochastic continuous evolution) remains scientifically interpretable without it -- documented explicitly, never silently backfilled with an out-of-scope architecture family. If the architecture seam itself is shown wrong (multiple arms blocked for a shared structural reason), `M9_1_FULL_EXPERIMENT_READY = NO` and the milestone stops for review rather than proceeding on a compromised testbed.

## 15. Scope discipline (restated)

No development-holdout or locked-data access, no full training, no S/M/L capacity scaling, no architecture-family expansion beyond Section 1's frozen four, no solver/hidden-width/diffusion tuning against predictive quality, no alpha change, no topology-recipe change, no representation change away from `AGE_FIX_ONLY`, no safety/authority-threshold change, no architecture promotion, no scientific M9.1 training run begun in this milestone. This preflight answers only "can we run the experiment correctly?" -- never "which architecture is best?".
