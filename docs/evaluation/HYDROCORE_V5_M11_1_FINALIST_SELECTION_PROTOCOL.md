# HydroCore-v5 M11.1 finalist-selection protocol

Status: frozen before the M11.1 comparative synthesis. This additive protocol
implements the master v5 protocol's M11.1 finalist-selection step. It does
not freeze the finalist (M11.2), run the M11.5 validation matrix, or authorize
the one-time M11.6 locked evaluation.

## Scope and locked-data prohibition

The upstream identity is the M10 current-status index and its authoritative
`M10_5_SERVING_FREEZE_PASS` closure. The selected v5 release must be exactly
`models/hydrocore-v5-release`, seed `20260814`, checkpoint
`de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`,
release-manifest SHA-256
`f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34`,
serialized calibration SHA-256
`8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d`,
and calibration artifact hash
`f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd`.

M11.1 must assert the locked-test guard before and after its work. It must not
read, enumerate, score, or derive a metric from `locked_final_test` or
`locked_topology_test`, and it must not call `authorize_locked_final_test`.
Every evidence path is checked for those tokens. Both recorded values are
false.

## Candidate eligibility, frozen now

The candidate set is deliberately system-level, not a set of checkpoints:

1. `HydroCore-v5 M10 frozen release` is eligible only when the M10 index,
   M10.5/M10.5A/M10.5B closures, release bytes, allowlist, authority policy,
   and default-v5/no-v4-fallback wiring all verify.
2. `HydroCore-v4 frozen incumbent` is eligible as a reference candidate only
   because its architecture-freeze manifest and tracked v4 release bundle
   jointly supply a selected checkpoint, calibration, schema, authority
   policy, and reproducible bundle identity.

An experimental checkpoint is ineligible unless a closed, system-level record
explicitly promotes it. This excludes M9 capacity variants, continuous-time
arms, M10.2 Scout refits, M10.3 Strategist refits, and learned OOD variants.
Derived checkpoints cannot be substituted or ensembled.

## Permitted evidence

Only already-closed, non-locked, tracked evidence is permitted: the v4
architecture-freeze manifest; M0 baseline; M9 final closure and capacity
closure; M10.0--M10.5 closures and their directly referenced development-only
trajectory, safety, release, calibration, and parity artifacts. M11 outputs,
new experiments, newly fitted calibration, newly generated metrics, and any
locked source are forbidden.

## Gate-based selection rubric

No weighted score is used. Each eligible candidate must pass the following
gates: (A) predictive/incident-intelligence evidence and calibrated
actionability; (B) development robustness and fail-closed behavior; (C)
end-to-end decision utility with exact physical verification; (D) deterministic
safety/authority and human approval; and (E) reproducible release readiness.
The comparison also records complexity/scientific justification and known
limitations; a more complex candidate requires meaningful governed evidence,
and negative learned-component results favor retaining deterministic
authority.

Disqualifiers are unresolved safety/authority failure, non-reproducible
identity, a locked-data dependency, an ungoverned output or checkpoint, or a
system change required to make the candidate acceptable. Applicable numerical
claims retain the originating study's per-seed results and confidence-interval
interpretation; M11.1 makes no new statistical estimate and does not treat a
point estimate as superiority.

If more than one candidate passes all gates, the pre-frozen tie-break is: use
the candidate with the most recent complete, non-locked, system-level governed
development evidence on the normal serving identity, provided it has no
unresolved hard safety failure and does not require a post-selection change.
This rule is a completeness/reproducibility priority, not a capability-weight
or a new numerical optimization. If it cannot resolve the candidates, close
`M11_1_FINALIST_SELECTION_BLOCKED_AMBIGUOUS`.

## Closure vocabulary

The only successful state is `M11_1_FINALIST_SELECTED`. Upstream identity
failure closes `M11_1_FINALIST_SELECTION_BLOCKED_UPSTREAM_IDENTITY`; a required
system change closes `M11_1_FINALIST_SELECTION_BLOCKED_REQUIRES_SYSTEM_CHANGE`;
and an unresolved eligible comparison closes
`M11_1_FINALIST_SELECTION_BLOCKED_AMBIGUOUS`.

Selection records `finalist_selected=true`, `finalist_frozen=false`, and
`locked_evaluation_authorized=false`. M11.2 alone may freeze the selected
finalist. The retained v5 limitations are carried forward: vacuous M10.4
selected-plan-vs-NO_ACTION evidence, modest sampling improvement with no final
approved-action change, limited development unseen-topology evidence with
appropriate suppression, the intentionally retained M9.6-fixed versus
M10.4-incident-elapsed unobserved-age deviation, and non-promotion of learned
OOD, Scout, and Strategist components.
