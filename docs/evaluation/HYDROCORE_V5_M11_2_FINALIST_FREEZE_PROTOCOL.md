# HydroCore-v5 M11.2 finalist-freeze protocol

Status: frozen before M11.2 identity verification. M11.2 consumes the M11.1
selection record unchanged; it does not select a candidate, run a performance
experiment, tune a system component, execute M11.5, or authorize M11.6.

## Parent and exact finalist

The parent selection must be `M11_1_FINALIST_SELECTED` with M11.1 protocol
SHA-256 `52d911b86b37c7de095643cf02415601e5e1b198cf17c849239b97da8e94264d`.
The only finalist is `HydroCore-v5 M10 frozen release` at
`models/hydrocore-v5-release`, selected seed `20260814`, HydroCore-S,
checkpoint SHA-256 `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`,
release-manifest SHA-256
`f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34`,
serialized calibration SHA-256
`8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d`,
and calibration artifact hash
`f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd`.

Its recipe is `CLASSICAL_HYDROCORE_S`, `AGE_FIX_ONLY`, and
`EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING`; its calibration is
`B_DEPTH_AWARE` at alpha 0.1. The model remains sentinel-only. The release
allowlist is exactly `event_cause`, `event_presence`, `evidence_sufficiency`,
`relative_strength`, and `source_node`; `next_step` remains suppressed.

## Identity and authority freeze

The finalist identity manifest binds release bytes, checkpoint identity,
calibration bytes/artifact, runtime manifest/configuration, feature-schema and
feature-semantic identities, fusion identity, output governance, and the
runtime/authority modules that load and enforce those values. Any difference
means a different finalist.

Deterministic `OODDetector`, `rank_sample_locations`, and
`generate_response_plans` remain authoritative. Learned OOD, Scout, and
Strategist stay non-authoritative. Exact WNTR/EPANET remains physical
authority; explicit human approval is required; autonomous actuation is
prohibited. Normal serving resolves to the v5 bundle and never substitutes v4.

The M10.4-tested `incident_elapsed` behavior for never-observed-node age is
frozen, including its known difference from the M9.6 fixed-age training record.
It is not repaired in M11.2.

## Verification and no-change rules

M11.2 verifies M11.1 parent hashes, the M10 closure, all frozen identities,
clean-process bundle load, default-serving resolution, and historical
immutability. In-memory identity mutations must be rejected for checkpoint,
calibration, schema, fusion, release manifest, selected seed, trained tasks,
outputs, and authority fields; adding `next_step`, enabling a learned control,
removing human approval, or enabling actuation must invalidate identity.

No architecture/checkpoint/model/calibration/threshold/feature/control-policy
tuning or change is permitted for this finalist. `tuning_closed=true` means
that any such change produces a different candidate requiring a separately
governed revalidation path.

M11.2 must assert the locked-test guard before and after work. It must not
read, enumerate, or derive a metric from either locked split, and it must not
call the locked-test authorization function. `locked_evaluation_authorized`
remains false.

## Closure vocabulary

Success is `M11_2_FINALIST_FROZEN`. Parent selection failure is
`M11_2_FINALIST_FREEZE_BLOCKED_SELECTION_IDENTITY`; identity drift is
`M11_2_FINALIST_FREEZE_BLOCKED_IDENTITY_DRIFT`; a required system change is
`M11_2_FINALIST_FREEZE_BLOCKED_REQUIRES_SYSTEM_CHANGE`; reproducibility
failure is `M11_2_FINALIST_FREEZE_BLOCKED_REPRODUCIBILITY`.

M11.2 records `finalist_selected=true`, `finalist_frozen=true`,
`tuning_closed=true`, and `locked_evaluation_authorized=false`. It stops
before M11.5; M11.6 remains unavailable without later prerequisites and
explicit human authorization.
