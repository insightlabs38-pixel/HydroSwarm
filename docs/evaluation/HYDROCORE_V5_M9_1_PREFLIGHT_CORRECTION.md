# HydroCore-v5 Milestone 9.1 preflight correction (post-preflight implementation review, before any scientific evaluation)

Amends nothing in `docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md`, which remains historically frozen and unmodified. That document was correctly committed before implementation began and records the intended architecture semantics; this document records a subsequent implementation-review finding that the shipped `src/hydroswarm/model/continuous_time.py` did not fully match two of those already-frozen semantics, and closes the gap.

Historical sequence, all preserved and visible:

1. original preflight protocol frozen (`HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md`, commit `784d8b2`);
2. preflight implementation, tests, and results committed (`9c7fc06`);
3. post-preflight code review (this document) identified two implementation discrepancies against the already-frozen protocol, before any predictive M9.1 experiment was run;
4. this correction aligns the implementation with the already-frozen scientific intent -- it changes no frozen decision, only fixes code that did not yet correctly implement one.

No predictive data was inspected at any point while identifying or deciding these fixes: both issues were found by re-reading `continuous_time.py` against the protocol's own text, not by observing any accuracy/localization signal. No solver, hyperparameter, or model-family search was performed to arrive at either fix.

## Issue 1: ODE/SDE initial state pooled the whole prefix instead of using first-valid evidence

**Frozen protocol intent** (Sections 4/6): GRAPH_ODE's latent evolves continuously from an initial condition that "encode[s] the FIRST available observation (or a learned zero-state absent any observation)"; GRAPH_SDE shares the same initial-state semantics as GRAPH_ODE, with structured diffusion as its only qualitatively new mechanism.

**What was actually implemented**: `_PooledEvidenceInitialState` masked-mean-pooled EVERY causally-available temporal/quality observation across the whole prefix into `h0`, for both GRAPH_ODE and GRAPH_SDE. This is a different architecture than the frozen one -- it lets the "continuous evolution" arms consume the entire evidence history before integration even begins, which blurs the intended ODE-vs-CDE distinction (Section 6 of the milestone instructions: "ODE: latent dynamics evolve continuously" from a point condition, vs. "CDE: latent dynamics are continuously controlled by the observed evidence path" throughout) -- as implemented, ODE/SDE's initial condition already summarized the same evidence CDE is supposed to be the only arm to continuously consume.

**Correction**: `_FirstValidEvidenceInitialState`, used identically by both `GraphODEDynamics` and `GraphSDEDynamics`. For each batch item / node / modality independently, finds the earliest step index where that modality's own validity mask (`sensor_mask` for temporal, `quality_mask` for quality) is `True`, and uses exactly that step's feature vector -- never an array-position-0 default, never a pooled/averaged combination, never a value from any later step. The two modalities' first-valid indices may legitimately differ (e.g. temporal first-valid at step 1, quality first-valid at step 3) and are resolved independently. A node with no valid observation in either modality gets a deterministic zero vector for that modality's component -- no learned missing-state token is introduced (kept minimal, matching Section 2's "do not invent learned missing-state tokens in this correction").

## Issue 2: GRAPH_CDE's control path could not distinguish "observed zero" from "missing, zero-filled"

**Why this matters**: `HydraulicFeatureBuilder`/`pad_graph_batch` NaN-replace-with-zero every invalid temporal/quality position at corpus-generation time (`src/hydroswarm/preprocessing/batching.py`'s own documented convention -- see its `PaddedGraphSample.sensor_mask`/`quality_mask` docstring: re-deriving validity from `torch.isfinite()` after this point would find no NaNs left and silently produce an all-`True` mask), so `sensor_mask`/`quality_mask` (`[batch, steps, nodes]` bool, `True`=valid) are the ONLY remaining signal distinguishing a genuine zero reading from an unobserved position. `GraphCDEDynamics._build_path_values` accepted these masks as arguments but never included them in the constructed control path `X_t` -- so the CDE's control path represented "missing" and "observed exactly zero" identically, which is exactly the distinction M9.1's own irregular-telemetry hypothesis needs the CDE arm to be able to represent.

**Correction**: the control path gains two additional explicit validity channels, `sensor_valid` and `quality_valid` (each a `0.0`/`1.0` scalar per `[batch, time, node]`, derived via the same `_valid_mask` convention already used elsewhere in this module -- `finite(values) & supplied_mask`, never treating a padded/invalid position as valid by omission). `input_channels = 1 (relative_time) + temporal_feature_dim + quality_feature_dim + 2 (sensor_valid, quality_valid)`. No other channel is added. The underlying feature values at invalid positions remain numerically `0.0` (unchanged); what changes is that the validity channels now let the CDE's own learned vector field, if it chooses to, treat "feature=0.0, valid=1.0" (a genuine zero reading) differently from "feature=0.0, valid=0.0" (missing) -- proven, not merely plumbed, by a dedicated non-vacuousness test (Section 8 below).

## Scope discipline (unchanged from the original preflight, restated)

This correction:

- does not compute or inspect top1/top3/MRR/NLL/calibration-coverage/candidate-set-size/any scientific OOD performance number for any arm;
- does not load `development_holdout`, `ood_development`, or any coastal-branch/tree-branch/dense-loop development evaluation data;
- does not open `locked_final_test` or `locked_topology_test`;
- does not change `torchdiffeq`/`dopri5`/`rtol`/`atol` (GRAPH_ODE), `torchcde`/linear interpolation/`rk4`/`step_size=0.25` (GRAPH_CDE), or `torchsde`/euler/Itô/diagonal diffusion/`diffusion_scale=0.1`/`dt=0.05` (GRAPH_SDE);
- does not search over interpolation methods, solver settings, or diffusion parameterizations;
- does not add a fifth control channel beyond `sensor_valid`/`quality_valid`;
- does not give ODE/SDE any re-reading of later evidence VALUES after `h0` is constructed (later timestamps still legitimately determine integration duration -- Section 3 of this milestone's instructions draws this distinction explicitly and Section 4's Test A verifies it directly);
- does not add a new architecture family, does not run HPO, does not begin full M9.1 training or M9 capacity scaling;
- re-runs the exact same purely-mechanical, no-predictive-data integer-width parameter-matching search the original preflight used (Section 7 of the original protocol), applied only to arms whose trainable parameter count actually changed as a result of these two fixes.

This correction answers only: **does the implementation now match the already-frozen M9.1 preflight design?** It does not reopen, and does not answer, "which architecture is best?".
