# HydroCore-v5 M11.1 finalist-selection results

Closure: `M11_1_FINALIST_SELECTED`. This additive selection synthesis used
only the closed, non-locked evidence listed in
`reports/evaluation/hydrocore-v5/m11/m11-1/m11-1-evidence-manifest.json`.
The frozen protocol SHA-256 is
`52d911b86b37c7de095643cf02415601e5e1b198cf17c849239b97da8e94264d`.

## Readiness and locked-test guard

The M10 index reports `m10_complete=true`; its authoritative closure is
`M10_5_SERVING_FREEZE_PASS`; M10.5A and M10.5B report respectively
`M10_5A_DEPLOYMENT_SELECTION_FROZEN` and
`M10_5B_CALIBRATION_ARTIFACT_MATERIALIZED`. The selected v5 bundle,
checkpoint, manifest, serialized calibration, calibration artifact, runtime
allowlist, authority policy, default-v5 wiring, no-v4-fallback behavior, and
retained M10.4 feature semantics all verified.

`locked_test_opened_before=false` and `locked_test_opened_after=false`. No
locked final or topology source was read, and no locked-evaluation
authorization was requested.

## Eligible candidate set and decision

Two complete frozen system identities were eligible: the M10 v5 release and
the v4 frozen incumbent. HydroCore-M, HydroCore-L, continuous-time variants,
M10.2 Scout refits, M10.3 Strategist refits, and learned OOD/fusion variants
were excluded because their own governed closure did not promote them.

Both eligible systems passed the gate-based predictive, robustness,
end-to-end, safety/authority, release-readiness, complexity, and limitation
review. The protocol's pre-frozen completeness/reproducibility tie-break then
selected the current v5 normal-serving identity: it alone has the complete
closed M9-to-M10 system-level development evidence and M10.5 immutable release
freeze. This is not a newly computed capability score or a locked-test result.

The selected finalist is `HydroCore-v5 M10 frozen release`:

- Bundle: `models/hydrocore-v5-release`
- Checkpoint SHA-256: `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`
- Manifest SHA-256: `f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34`
- Calibration SHA-256: `8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d`
- Calibration artifact hash: `f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd`

The final-selection record is
`reports/evaluation/hydrocore-v5/m11/m11-1/final-selection.json`. It records
`finalist_selected=true`, `finalist_frozen=false`, and
`locked_evaluation_authorized=false`. M11.2 is the next authorized milestone;
it has not been performed here.

## Retained limitations

M10.4's selected-plan-versus-NO_ACTION Gate E evidence was vacuous because
NO_ACTION was absent from the bounded candidate set. Deterministic active
sampling modestly improved localization but did not change the final approved
action in the M10.4 population. Development unseen-topology evidence remains
limited and appropriately suppresses calibration/actionability when unsupported.
The M10.4 incident-elapsed unobserved-age behavior differs from M9.6 fixed-age
training behavior and remains intentionally frozen, not resolved. Learned OOD,
Scout, and Strategist components were not promoted.
