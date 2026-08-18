# HydroCore-v5 M10.5 protocol: governed serving-path freeze / release identity

## Status and purpose

This protocol was frozen on branch `exp/hydrocore-v5-causal` from source
commit `8a4752bf39f82542735c21dd44981f164ed2b849`, before any M10.5 serving
bundle or default-startup modification.  Its sole purpose is to determine
whether the exact M10.4-passing system can be assigned a release identity
without retrospectively selecting a favorable model seed.

M10.4 is the required parent: closure `M10_4_FULL_TRAJECTORY_PASS`, protocol
SHA-256 `cd0ac1f2d5a12a771cc441b4ea19bf0d76c672809b35d3d178f8893b768a177c`.
No locked evaluation is authorized; `locked_test_opened` must remain false.

## Frozen selection rule audit

The authoritative M9.6 records to inspect are its closure, manifest,
protocol, and three `ARM_B_M9_6` training records.  The only known frozen
checkpoint rule is `FINAL_STEP_1350`: it selects the final-step export *for
each seed*.  This protocol forbids treating that per-seed export rule as a
rule for choosing between seeds.

The candidate identities are exactly the M9.6 canonical HydroCore-S ARM_B
exports for seeds `20260814`, `31874`, and `20260815`, with the hashes recorded
in M10.4.  A serving release may proceed only if the inspected frozen records
already provide exactly one of the following:

1. one named serving/deployment seed;
2. a deterministic, non-performance-based seed selector; or
3. an explicitly governed all-three-checkpoint ensemble/replicated release.

M10.4 performance, M10.4 per-seed results, M10.5 results, and any new
accuracy measurement are forbidden selection inputs.  Absence of one of the
three mechanisms is a material governance blocker with principal closure
`M10_5_SERVING_FREEZE_BLOCKED_SELECTION_IDENTITY`.

## Serving-path and output-governance audit scope

The pre-modification serving path is traced from `hydroswarm.api.app` through
`V4PipelineFactory` and `resolve_v4_bundle_dir`; it is recorded only to
establish the required future replacement scope.  No v4 artifact may be
modified.

For any future selected release, the allowlist must be reconciled with the
M9.6 supervision record: `next_step` is not a supervised M9.6 output and
must not be release-enabled merely because its parameters may exist.  Learned
OOD, Scout, and Strategist controls remain non-authoritative; deterministic
`rank_sample_locations`, `generate_response_plans`, exact WNTR/EPANET
verification, human approval, and no autonomous actuation are immutable
authority boundaries.

The M10.4-tested runtime feature behavior, including the known unobserved-age
semantic difference from recorded M9.6 training kwargs, is frozen as a
documented release identity property.  It must not be changed under M10.5.

## Allowed work and closure criteria

Before a unique release identity exists, M10.5 may add only diagnostic
protocol/report/test artifacts.  It may not create a v5 serving bundle,
redirect default startup, alter a model/calibration/configuration, or perform
parity/release-load experiments that presuppose a selected checkpoint.

If selection is unresolved, record the blocker, the candidate hashes, the
absence evidence, the unchanged default v4 path, the output-governance finding,
and locked-test/historical-immutability status; commit and stop.  A later
amendment must define a selection/ensemble rule independently of M10.4 results
before serving freeze/parity can resume.
