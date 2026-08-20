# Milestone 2 summary: multitask interference and objective design

Exit decision: **PCGRAD_JUSTIFIED**

## Arm comparison vs Arm A (full governed multitask)

| arm | 3-step gain (pp) | 6-step gain (pp) | mature delta (pp) | meets gain | no mature regression |
|---|---|---|---|---|---|
| B | -3.33 | -0.83 | 0.00 | False | True |
| C | -0.83 | 0.83 | -0.83 | False | True |

PCGrad precondition (frequent negative primary-primary conflict) met: **True** (pairs: ['MATURE:sensor_fault|source_node', 'MATURE:source_node|sensor_fault', 'MID:sensor_fault|source_node', 'MID:source_node|sensor_fault'])
PCGrad fully justified (precondition AND measurable source_node degradation, top1 < 0.9): **True** (evidence: ['MID:sensor_fault|source_node (Arm A top1=0.887)', 'MID:source_node|sensor_fault (Arm A top1=0.887)'])

Scope note: only Sentinel-family tasks have real supervision in this corpus (see reports/evaluation/hydrocore-v5/m2-conflict.json's scope_limitation); Scout/Strategist/OOD interference is not measured this session and is not part of this exit decision.

**Caveat on the PCGRAD_JUSTIFIED flag (recorded honestly, not acted on):** the
sensor_fault|source_node negative-conflict pair is flagged in BOTH the MID and
MATURE depth buckets, but the degradation threshold (source_node top1 < 0.90)
only trips in MID, driven almost entirely by depth=4's 0.800 top1 -- the very
next depth in the same bucket (depth=6) already recovers to 0.975, and MATURE
(where the same conflict pair also appears) sits at 0.988. If this conflict
pair genuinely caused primary-task degradation, MATURE would be expected to
show it too; it does not. This is more consistent with ordinary evidence-
scarcity variance at depth=4 than with a real gradient-conflict-caused
regression, and this session's evidence is too thin (single seed, N=120
development_holdout) to disentangle the two causally. Per experiments.txt's
"do not enable PCGrad merely because a mechanical trigger fires," this
session's actual recommendation is: **do not enable PCGrad** based on this
evidence -- the mechanical PCGRAD_JUSTIFIED label above is reported for
transparency, not as an actionable recommendation.
