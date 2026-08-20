# HydroCore-v5 TRUE Milestone 10.2 results (executed under the frozen protocol)

Amends nothing in `HYDROCORE_V5_M10_2_TRUE_EVALUATION_PROTOCOL.md`, which remains frozen exactly as written
before execution (`protocol_hash` `5dba094406aa6df7b85f50a6becdd4a092c171a7a766f75877584e7885548ad8`,
unchanged, reconfirmed identical in every artifact this run produced). No population, threshold, or metric was
changed after any result was inspected.

## Result: `M10_2_LEARNED_SCOUT_NOT_PROMOTED_DETERMINISTIC_RETAINED`

The frozen promotion rule's primary-metric consistency bar (Section 9, criterion 2) was not met: the primary
metric (`actionable_within_budget`, learned minus deterministic) had a **negative or zero** point estimate in
**all three** seeds (required: positive in all three), and a 90% CI excluding zero in **zero** of three seeds
(required: at least two of three). This is a mechanical, frozen-rule outcome, not a judgment call --
`reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-statistics.json`'s `promotion_decision` records the full
mechanical trace.

| Seed | actionable_within_budget (D / L) | diff (CI 90%) | never_actionable (D / L) | source_top1_final (D / L) |
|---|---|---|---|---|
| 20260814 | 0.81 / 0.79 | -0.020 [-0.040, 0.000] | 0.19 / 0.21 | 0.89 / 0.87 |
| 31874 | 0.80 / 0.80 | 0.000 [0.000, 0.000] | 0.20 / 0.20 | 0.89 / 0.87 |
| 20260815 | 0.71 / 0.70 | -0.010 [-0.030, 0.000] | 0.29 / 0.30 | 0.87 / 0.86 |

n=100 paired incidents per seed (300 total), family `golden-reference`, budget 3, all from
`reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-aggregate-metrics.json`.

No regression criterion was CI-confidently violated in any seed either (`never_actionable_fraction`'s CI lower
bound is exactly `0.0`, not `>0.0`, in every seed that shows any point-estimate increase; `source_top1_final_round`'s
CI never excludes zero) -- the block is driven entirely by the primary-metric consistency bar, not a safety/
regression failure.

## Why: a real, disclosed ceiling effect in this population, not a Scout-policy defect

`samples_to_actionability_both_resolved` -- computed only over incidents BOTH arms eventually resolved -- has
mean `0.0` **rounds** for both arms in all three seeds: among the ~70-80 incidents both arms resolve, the
typical resolution round is round 0, i.e. **before either policy takes a single Scout sample**. The
`golden-reference` network has only 4 junctions, and the frozen population uses the full `depth=25` (MATURE)
causal-prefix history as the STARTING evidence for every incident (Scout-round evidence layers ON TOP of this,
per the frozen protocol's own Section 3/5) -- so a large majority of incidents are already source-resolved from
their initial evidence alone, leaving genuinely little headroom for either Scout policy to demonstrate value on
`actionable_within_budget` specifically. This is a property of the frozen population design (itself inherited,
unmodified, from Level A's own accepted training/evaluation scope -- never adjusted after this observation),
not evidence of a defect in either policy.

## A genuine, non-cherry-picked secondary finding: learned Scout is dramatically more sample-efficient at equivalent outcomes

`stopping_quality.budget_exhaustion_rate_arm_D` is `1.0` in **all three seeds** -- `HydroScout.
deterministic_fallback`'s `1.0/len(candidate_region) < 0.01` stop threshold essentially never fires for a
4-junction region (`1/4 = 0.25 >> 0.01`, per the protocol document's own Finding 1), so ARM D always spends the
full 3-sample budget regardless of whether the incident is already resolved. `budget_exhaustion_rate_arm_L` is
`0.06`, `0.06`, and `0.00` -- the trained `should_continue_sampling` head makes a genuine, mostly-early stop
decision instead. `stopping_quality.false_stop_rate` (ARM L stopped before ARM D's own full-budget trajectory on
the SAME incident later proved further sampling would have helped) is low in every seed (`0.02`, `0.00`, `0.01`)
-- ARM L's early stops are rarely wrong by this frozen, paired, non-fabricated counterfactual check. Net effect:
learned Scout reaches essentially the SAME final localization quality as deterministic Scout
(`source_top1_final_round` within 1-2pp, CI-indistinguishable in every seed) while taking dramatically fewer
samples on average. This is scientifically interesting and worth carrying into a future, separately authorized
milestone's design (e.g. a population with more headroom below the ceiling, or a primary metric that credits
sample efficiency directly) -- but it is explicitly NOT grounds to promote under THIS milestone's own frozen
primary-metric rule, which this document does not retroactively loosen or substitute.

## Safety / governance hard gates

All pass, all three seeds, mechanically asserted (not merely observed) during execution --
`reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-safety-audit.json`: zero invalid/inaccessible node selections,
zero already-sampled reselections, zero budget-exceeded events, zero non-finite Scout outputs, zero fail-closed
masking violations, across all 600 trajectories (300 incidents x 2 arms) x 3 seeds.

## Checkpoint / provenance

All three approved Level-A refit checkpoint SHA-256 hashes verified against the authorizing task's approved
values before any evaluation ran (`m10-2-checkpoint-verification.json`) -- no regeneration was needed, all
three weight files were already present locally with matching hashes. All three parent M9.6 teacher checkpoint
hashes independently verified unchanged. The SAME Level-A refit checkpoint (never the original M9.6 teacher,
whose raw Scout heads remain untrained per `HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md`) was used for BOTH arms
in every incident/round, mechanically confirmed by `tests/scientific/test_m10_2_true_evaluation.py::
test_same_level_a_refit_checkpoint_object_used_for_both_arms`.

## Output governance (unaffected)

Learned Scout remains runtime-disabled and non-authoritative. `hydroswarm.inference.authority.scout_certificate`
was not modified, not called anywhere in this evaluation's execution, and continues to hardcode
`source="CLASSICAL_EIG"`/`AuthorityLevel.DETERMINISTIC` unconditionally. No `runtime_enabled_outputs` promotion
occurred -- this milestone determines scientific eligibility only, and the result (not promoted) means no
runtime-promotion question even arises this cycle.

## Readiness

`M10_2_LEARNED_SCOUT_NOT_PROMOTED_DETERMINISTIC_RETAINED`. Deterministic Scout remains the sole runtime-
authoritative policy. This is TRUE M10.2's own valid, complete, negative scientific result -- not a blocked or
incomplete evaluation. M10.3 (Strategist) is not addressed by this milestone and requires its own separately
authorized supervision/candidate-schema amendment first, per `HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md`.
