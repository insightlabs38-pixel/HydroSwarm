# HydroCore-v5 M11.2 finalist-freeze results

Closure: `M11_2_FINALIST_FROZEN`. The M11.1-selected HydroCore-v5 serving
system is now one immutable finalist identity for subsequent M11 validation.
The M11.2 protocol SHA-256 is
`1974f8d1b40d1318d3c80ec9763ef7ff20b8e6a90d607a917a2aca38162e9d1e`.

## Parent and identity verification

M11.1's `M11_1_FINALIST_SELECTED` closure, selected finalist, transitional
flags, selection-record hash, and protocol SHA-256
`52d911b86b37c7de095643cf02415601e5e1b198cf17c849239b97da8e94264d`
all matched. M10 remains complete through `M10_5_SERVING_FREEZE_PASS`.

The frozen finalist is `HydroCore-v5 M10 frozen release`, seed `20260814`, at
`models/hydrocore-v5-release`:

- Checkpoint SHA-256: `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`
- Release-manifest SHA-256: `f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34`
- Calibration SHA-256: `8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d`
- Calibration artifact hash: `f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd`

The finalist identity manifest binds release assets, model configuration,
feature/schema and serving semantics, fusion configuration, output governance,
authority policy, and directly relevant runtime source hashes. It freezes the
M10.4-tested `incident_elapsed` unobserved-age behavior, including its known
difference from M9.6 fixed-age training behavior; it does not repair it.

## Authority and reproducibility

The system remains sentinel-only with runtime outputs `event_cause`,
`event_presence`, `evidence_sufficiency`, `relative_strength`, and
`source_node`; `next_step` is `SUPPRESSED_UNSUPERVISED`.

Deterministic `OODDetector`, `rank_sample_locations`, and
`generate_response_plans` remain authoritative. Learned OOD, Scout, and
Strategist remain non-authoritative. WNTR/EPANET is final physical authority,
human approval remains mandatory, autonomous actuation is prohibited, and
normal serving resolves only to v5 without v4 fallback.

A fresh Python process loaded the exact release with no fallback and verified
checkpoint, calibration artifact, feature schema, fusion, trained-task, and
output identities. Fourteen deliberate in-memory identity mutations were all
rejected, covering checkpoint/calibration/release/schema/fusion/seed/task/output
and authority drift.

## Freeze state

Historical M0--M10 and M11.1 artifacts remain unchanged. The current status is
`reports/evaluation/hydrocore-v5/m11/m11-current-status.json`.

- `finalist_selected=true`
- `finalist_frozen=true`
- `tuning_closed=true`
- `locked_evaluation_authorized=false`
- `locked_test_opened_before=false`
- `locked_test_opened_after=false`

The frozen limitations remain: vacuous M10.4 NO_ACTION Gate E evidence,
modest sampling benefit without an approved-action change, limited development
unseen-topology evidence, the retained train/serve age-semantic deviation, and
non-promotion of learned OOD, Scout, and Strategist. M11.5 is the next
authorized milestone and was not executed here.
