# physics-informed-localizer-validation: plan (pre-registration)

Branch: `exp/physics-informed-localizer-validation`. Follows
`exp/candidate-conditioned-localizer-v1` (that branch's final report:
`reports/evaluation/candidate-conditioned-localizer-v1/FINAL_REPORT.md`).
**Experimental, non-release.** No change to `models/hydrocore-v5-release`,
`data/locked/`, any M11.6 artifact, or any governance module. This is a
confirmation-and-ablation study, not a new architecture: no GNN rewrite,
no additional attention stack, no major model scaling, no gate relaxation.

## 0. What the pilot found (recap)

- A candidate-conditioned localizer without physics features
  (`B_CANDIDATE_CONDITIONED`) did not significantly improve unseen-topology
  Top-1 (+1.4pp, 90% CI [-1.1, +4.3]pp) and significantly REGRESSED
  unseen-topology Top-3 (-4.3pp, CI [-6.4, -2.1]pp).
- The physics-informed candidate-conditioned arm
  (`C_PHYSICS_INFORMED`, here renamed `C_FULL`) improved unseen-topology
  Top-1 from 0.375 to 0.439 (+6.4pp, 90% CI [+2.5, +10.0]pp, excludes
  zero).
- Known-topology / low-centrality / long-distance subgroup gains were
  mostly null (CIs include zero).
- `C_FULL` had ~4.6% more parameters than `A_CONTROL`; `B` and `C_FULL` have
  almost identical parameter counts (4,231,129 vs 4,231,897), so the
  B-vs-C_FULL comparison is not confounded by capacity, but neither arm was
  ever compared against a capacity-matched, non-structural control.
- The oracle audit found the original oracle privileged (shares the true
  incident's exact strength/start/duration/hydraulic realization); a
  nuisance-searched correction reproduced the same 96.4% recovery figure on
  the same failing incidents, but the corrected oracle still shares the
  true hydraulic/demand realization -- a documented residual privilege, not
  re-tested here (see "Oracle caveat" below).

**This is not yet sufficient for larger-scale validation.** This branch
asks: is the unseen-topology gain (1) reproducible across seeds, (2) not
explained by parameter count alone, (3) attributable to a specific
physically meaningful feature, (4) robust enough to justify a larger run.

## 1. Primary hypothesis and endpoint (pre-registered)

**H_primary**: physics-informed candidate-conditioned compatibility
features produce a reproducible improvement in unseen-topology source
localization beyond generic extra model capacity.

**Primary endpoint (fixed before any arm beyond A_CONTROL/B/C_FULL was
re-run on this branch): unseen-topology (`ood-UNSEEN_TOPOLOGY`) Top-1
accuracy.** Success is not redefined post hoc based on whichever secondary
metric or subgroup happens to move.

Secondary endpoints: unseen-topology Top-3/MRR; known-topology
Top-1/Top-3/MRR; low-centrality subgroup; long source-to-sensor-distance
subgroup; calibration/abstention/OOD proxy behavior; all hard safety
counters (all reported as 0 for the same documented reason as the pilot --
Section 6 below).

## 2. Pre-declared seeds (Phase 3)

`SEEDS = (20260814, 20260901, 20260915)` -- fixed in
`run_experiment.py` before any Phase-3 arm beyond seed 20260814 was
trained. `20260814` is the original pilot's own seed (direct Phase-1
reproduction check, reused rather than re-derived); the other two are
disjoint calendar dates chosen with no dependence on any run's outcome.
Three seeds, not five: see "Compute budget" (Section 7) for why five was
not pursued given the sequential-CPU-training constraint this branch
inherits from the pilot.

## 3. Arms

All arms share HydroCore-v5's `small` variant, `event_control_heads=True`,
identical training config (`configs/training-v5-causal.yaml`, 6 epochs,
`fp32=True`, `deterministic=True`, CPU), identical corpus
(`data/learning-v2/cycle-b2/tensors-normalized`), and identical
per-seed train/validation/development-holdout/calibration/OOD splits
across every arm at that seed (`run_experiment.py::stratified_indices`/
`capped_indices`, seeded once per seed and reused for every arm).

| arm | `localizer_mode` | structural feats | physics feats (of 3) | extra generic capacity | purpose |
|---|---|---|---|---|---|
| A_CONTROL | `default` | -- | -- | -- | baseline, unmodified `source_node_head` |
| A_CAPACITY_MATCHED | `default` | -- | -- | `localizer_capacity_hidden_dim=482` (+~4.6% params) | Phase 2: does generic capacity alone reproduce the gain? |
| B_CANDIDATE_CONDITIONED | `candidate_conditioned` | yes (6-dim) | 0 of 3 | -- | pilot's own arm, unchanged |
| C_FULL | `candidate_conditioned` | yes (6-dim) | 3 of 3 | -- | pilot's `C_PHYSICS_INFORMED`, unchanged |
| C1 | `candidate_conditioned` | yes (6-dim) | 1 of 3 (nearest-sensor concentration) | -- | Phase 4 ablation |
| C2 | `candidate_conditioned` | yes (6-dim) | 1 of 3 (hop-vs-magnitude) | -- | Phase 4 ablation |
| C3 | `candidate_conditioned` | yes (6-dim) | 1 of 3 (hop-vs-arrival-time) | -- | Phase 4 ablation |
| C1_C2 / C1_C3 / C2_C3 | `candidate_conditioned` | yes (6-dim) | 2 of 3 | -- | Phase 4 pairwise, budget-permitting |

**Every C-family arm (C_FULL, C1, C2, C3, and the pairwise combinations)
uses IDENTICAL model construction** -- `localizer_physics_feature_dim=3`
always, so `CandidateConditionedLocalizer`'s `physics_projection` layer and
every other parameter is byte-identical across all seven. Ablation is
implemented by zeroing the non-selected columns of
`physics_features.compute_physics_features`'s output BEFORE it reaches the
model (`run_experiment.py::_mask_physics_columns`), never by resizing the
model. This satisfies Phase 5's requirement directly (same architecture,
same parameterization, only the physics-feature INPUT ablated) rather than
via a padding trick -- there is no parameter-count difference to verify
away in the first place.

`A_CAPACITY_MATCHED`'s `localizer_capacity_hidden_dim=482` was selected by
direct enumeration over hidden widths (see
`src/hydroswarm/model/core.py`'s `CAPACITY_MATCHED_HIDDEN_DIM` comment in
`run_experiment.py`) as the value whose resulting total parameter count
(4,231,223) lands closest to Arm B/C's own delta (4,231,129 / 4,231,897)
without adding candidate conditioning, candidate-to-sensor structure,
physics features, graph information, or source-specific topology
information -- a residual MLP block (`CapacityMatchedProjection`,
`src/hydroswarm/model/adapters.py`) reading only the same per-node hidden
state `source_node_head` already consumes.

## 4. Priority and compute budget

Priority order (task's own): `A_CONTROL, A_CAPACITY_MATCHED,
B_CANDIDATE_CONDITIONED, C_FULL, C1, C2, C3`. The three pairwise arms
(`C1_C2`, `C1_C3`, `C2_C3`) are budget-permitting, run only if the priority
grid finishes with time to spare -- consistent with the task's own "do not
create a combinatorial explosion if the budget is limited."

Each (arm, seed) pair is one full from-scratch training run: 6 epochs over
600 stratified training examples, CPU, `fp32=True`,
`deterministic=True` -- ~10-13 minutes each based on the pilot's own
recorded `elapsed_seconds` (636-766s for A_CONTROL/C_PHYSICS_INFORMED).
7 priority arms x 3 seeds = 21 runs, ~4 hours of sequential CPU training in
total; this is the reason 3 (not 5) pre-declared seeds and budget-permitting
(not mandatory) pairwise ablation arms were chosen.

## 5. Statistical convention

Paired bootstrap: 2000 resamples, seed 20260826, 90% percentile interval
-- identical to `candidate_conditioned_localizer_v1/analyze_results.py`
and `exp/source-identifiability-analysis`'s own `stats_utils.py`
("HydroSwarm's established convention"), unchanged here. Per-seed paired
comparisons match by `scenario_id` within a seed; the pooled cross-seed
bootstrap concatenates every seed's own paired (control, arm) values
matched by `(seed, scenario_id)`, never mixing an example from one seed's
split with a different seed's differently-sampled split under the same ID.

Both individual per-seed results AND the pooled cross-seed summary are
always reported together (`analyze_results.py`'s per-`seed-<n>/` outputs
plus `pooled/`) -- never only the pooled figure, per the task's explicit
instruction not to hide seed disagreement behind an aggregate.

## 6. Safety / governance (unchanged from the pilot)

Identical to `candidate_conditioned_localizer_v1`: this pilot-scale
localization-only harness never exercises the sampling/planning/execution
control loop that produces non-zero `hard_safety_counters` at the M11.6
evaluation tier, so every arm reports all eight counters as 0 -- because
those code paths are never invoked here, not because they were
independently re-verified at that tier. No governance module
(`hydroswarm.inference.ood`, `hydroswarm.calibration.conformal`, any
actionability gate) is modified by this branch. Experimental localizers
remain non-release and opt-in (`localizer_mode`/
`localizer_capacity_hidden_dim` both default to values that reproduce
pre-existing `HydroCore` behavior byte-for-byte). Prediction (source
identity) remains distinct from permission to act; nothing here is wired
into planning/operational authority.

## 7. Oracle caveat (carried forward, not re-tested)

The fair, nuisance-searched oracle from the pilot's own audit
(`docs/evaluation/ORACLE_INFORMATION_AUDIT.md`) remains the branch's only
conceptual motivation and approximate upper bound -- it is not used as a
training target anywhere in this branch, and its documented residual
privilege (shares the true incident's hydraulic/demand realization across
candidates) is not independently tested here. Any oracle-gap-closed
reading in the final report inherits that caveat unchanged.

## 8. Decision rule (fixed before results)

- **REJECT** if the +6.4pp effect disappears across seeds, a
  capacity-matched control reproduces a similar gain, effect direction
  varies wildly by seed, physics ablations show no interpretable feature
  dependence, Top-3/MRR regress enough to offset the Top-1 gain, the gain
  is concentrated only in already-easy subgroups, or safety/calibration
  behavior worsens.
- **CONTINUE_RESEARCH** if the signal persists but remains seed-sensitive,
  subgroup-limited, or mechanistically unclear.
- **CANDIDATE_FOR_LARGER_SCALE_VALIDATION** only if the gain replicates
  across seeds, survives the capacity-matched control, has an
  interpretable physics-feature driver, and shows no meaningful
  safety/calibration tradeoff.
