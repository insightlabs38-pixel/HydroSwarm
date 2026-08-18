# HydroCore-v5 M10.5 results: serving-path freeze / release identity

## Result: `M10_5_SERVING_FREEZE_BLOCKED_SELECTION_IDENTITY`

M10.5 cannot safely package or serve a single canonical M9.6 checkpoint.
M9.6 froze `FINAL_STEP_1350` as the export policy **within each of its three
seeds**, but none of its authoritative closure, manifest, protocol, or shared
governance code names a deployment seed, a deterministic non-performance
selector, or an all-three-checkpoint serving ensemble.  M10.4 evaluated all
three seeds; selecting one after those results would violate the preregistered
non-cherry-picking rule.

The M10.4 parent closure and protocol hash were verified.  The three canonical
hashes were rechecked, and no M10.4 performance data were used by this audit.
The default app remains untouched on the historical v4 release rather than
being redirected to an arbitrary v5 checkpoint.

## Additional release-governance finding

M10.4's evaluation identity recorded `next_step` as runtime-enabled, while
the M9.6 supervision record does not include it.  No v5 allowlist is frozen
because no release identity exists; any later, separately authorized serving
freeze must suppress `next_step` (and preserve the retained deterministic
OOD/Scout/Strategist/WNTR/human-approval authority path).

The known M10.4 feature-semantic deviation is also unchanged: the
M10.4-tested production unobserved-age feature path was not altered.

## Required next action

A separate amendment must freeze a deployment seed, a deterministic
non-result-based selector, or a governed ensemble rule before release bundle
creation, default-path rewiring, parity, or release-load failure tests may be
performed.  No locked test was opened and no M10.6/later work was started.
