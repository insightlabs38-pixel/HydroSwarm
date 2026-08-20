# HydroCore-v5 Milestone 10.2 Scout refit results (Level A, executed under the frozen protocol)

Amends nothing in `HYDROCORE_V5_M10_2_SCOUT_REFIT_PROTOCOL.md`, which remains frozen exactly as written before
Level A executed (`protocol_hash` `99d2c31f419960dddd83be2665b47cebaf5d3bffc84d939d174cd61b3e16cece`, unchanged).
This document records the result and the one amendment made before any competence metric was inspected.

## Amendment 1 (before any competence metric existed): validation population enlarged

The first Level-A run's validation population (`VALIDATION_COUNT=100`, as frozen) produced only 17 examples with
a real Scout recommendation (`sample_node_mask=True`) -- below the frozen gate's own `GATE_MIN_SUPPORT=20`. This
was observed from corpus SIZE statistics alone, before any accuracy/MSE/correlation number was computed for
either split. `scripts/hydrocore_v5/m10_2_refit_protocol.py`'s `VALIDATION_COUNT_AMENDMENT_1=300` extends
(never replaces or reorders) the same seed range (`VALIDATION_SEED_BASE=1_200_100_000`) to `300` scenarios --
the original 100 are an exact prefix. No other protocol value changed. This raised support to 65 real-
recommendation examples (360 total validation examples).

## Result: `M10_2_SCOUT_REFIT_A_ACCEPTED`

All three predictor seeds independently pass every one of the frozen gate's 7 criteria
(`docs/evaluation/HYDROCORE_V5_M10_2_SCOUT_REFIT_PROTOCOL.md` Section 8) -- no cherry-picking, all three
reported:

| Seed | sample_node top-1 (model / naive) | information_gain MSE (model / baseline), Spearman | candidate_reduction MSE (model / baseline), Spearman | should_continue_sampling accuracy (model / majority) |
|---|---|---|---|---|
| 20260814 | 0.800 / 0.296 (diff CI [0.411, 0.587]) | 0.681 / 1.129, ρ=0.288 (CI [0.241, 0.333]) | 0.060 / 0.104, ρ=0.436 (CI [0.397, 0.475]) | 0.861 / 0.819 (CI [0.831, 0.889]) |
| 31874 | 0.754 / 0.296 (diff CI [0.364, 0.546]) | 0.687 / 1.129, ρ=0.270 (CI [0.220, 0.314]) | 0.060 / 0.104, ρ=0.467 (CI [0.429, 0.504]) | 0.864 / 0.819 (CI [0.833, 0.894]) |
| 20260815 | 0.769 / 0.296 (diff CI [0.383, 0.560]) | 0.672 / 1.129, ρ=0.264 (CI [0.215, 0.312]) | 0.059 / 0.104, ρ=0.426 (CI [0.383, 0.467]) | 0.889 / 0.819 (CI [0.861, 0.914]) |

n=65 (`sample_node`), n=1359 (`information_gain`/`candidate_reduction`, per-node-per-example valid positions),
n=360 (`should_continue_sampling`). Full per-seed detail:
`reports/evaluation/hydrocore-v5/m10/m10-2-refit/m10-2-refit-level-a-gate.json`.

Gradient-coverage certificates (`m10-2-refit-gradient-coverage.json`) pass for all four Scout tasks, all three
seeds: real, finite, nonzero gradient reached every one of the 18 allowlisted parameters, and parameter values
actually changed after a controlled optimizer step. Parameter-allowlist exactness was mechanically reasserted
after training (`{name for name, p in model.named_parameters() if p.requires_grad} == LEVEL_A_PARAMETER_ALLOWLIST`)
for all three seeds.

**Interpretation**: the frozen M9.6 hydraulic/Sentinel representation DOES contain enough information for a
genuinely, correctly trained Scout specialist (four heads plus the two small injection projections for the new
round/budget/accessibility signal) to learn its four supervised tasks well above naive baselines, using only
frozen-backbone features. This is a representation-SUFFICIENCY finding for SUPERVISED-LABEL competence -- it is
explicitly NOT a finding that this refit Scout beats or should be promoted over
`HydroScout.deterministic_fallback` operationally; that comparison is the true M10.2 milestone's own job,
deliberately not performed here (Section 8's own naive-baseline-only rule).

## Level B: not triggered

Per the frozen protocol, Level B executes only if Level A passes gradient coverage/support/no-NaN but fails the
Scout-competence criteria for a legitimate representation-capacity reason. Level A passed every criterion for
every seed, so Level B was correctly never run. No `m10-2-refit-level-b.json` or Level-B checkpoint artifact
exists, consistent with the task's instruction not to fabricate Level-B results.

## Checkpoint / provenance

Three new refit checkpoints (`reports/evaluation/hydrocore-v5/m10/m10-2-refit/checkpoints/level-a-seed{20260814,
31874,20260815}/`), each with its own `checkpoint_identity.json` recording: parent M9.6 teacher SHA-256, refit
level (`"A"`), the exact 18-parameter trainable allowlist, frozen/trainable parameter counts, training-state/
target schema versions, train/validation manifest hashes, seed, optimizer-config hash, `"FINAL_EPOCH"`
checkpoint-selection policy, gradient-coverage-certificate hash, git commit, and the refit model's own SHA-256.
`identity["never_call_this_m9_6"] = true` -- these are never described as "the M9.6 checkpoint."

All three original M9.6 teacher checkpoint SHA-256 hashes verified unchanged after this pass
(`teacher_checkpoints_unchanged` in `m10-2-refit-closure.json`).

## Output governance (unaffected)

Learned Scout remains runtime-disabled and non-authoritative. `hydroswarm.inference.authority.scout_certificate`
was not modified, not called anywhere in this refit's execution, and continues to hardcode
`source="CLASSICAL_EIG"`/`AuthorityLevel.DETERMINISTIC` unconditionally. No `runtime_enabled_outputs` promotion
occurred. Strategist and OOD/`next_step` parameters were never made trainable (mechanically confirmed --
`tests/unit/test_m10_2_level_a_refit.py::test_strategist_and_ood_parameters_remain_frozen_under_level_a_allowlist`)
and are unaffected by this task, per the supervision amendment's own explicit scope limit.

## Readiness

`M10_2_SCOUT_REFIT_A_ACCEPTED`. The true M10.2 learned-vs-deterministic Scout scientific comparison may now be
separately authorized, using these Level-A refit checkpoints (never the original M9.6 checkpoints, whose raw
Scout heads remain untrained, per `HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md`'s own finding). This task does
not execute that comparison and does not proceed to M10.3.
