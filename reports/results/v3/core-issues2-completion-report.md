# core-issues2.txt completion report — Scout/Strategist/OOD/control/auxiliary expansion

Branch: `agent/gcp-multitopology-v3`. Starting commit `4bf7756` (end of the Arm
migration). Ending commit `a9e4a1c`. 20 commits, all pushed, working tree clean.
See `reports/results/v3/core-issues2-handoff.md` for the running narrative this
report summarizes.

**The locked final test was not opened, listed, hashed, or referenced by any command
run during this pass. `final-selection.json` does not exist.**

## Scope and status

core-issues2.txt asks for governed Scout/Strategist/OOD/control/auxiliary
supervision plus a full trajectory corpus. Delivered in this pass: every label
generator (Phases 1-6), the trajectory-assembly integration layer and a real,
full-scale generated corpus (Phase 7), and Phase 8's Stage 1 (smoke and failure
screening) verified end-to-end against that real corpus. **Not delivered**: a
sequence-aware training loop for Scout/Strategist (their targets are sequential,
not flat per-example tensors, and this pass's scope did not extend to building a
new training loop shape for them), and therefore Phase 8's remaining stages
(2 through 7), calibration, and promotion gates. This is stated plainly, not
implied by omission — see "Remaining work" below.

Also required and completed first: this session began on a freshly migrated
aarch64 GCP VM (previous sessions ran on x86). Three real environment defects
(none anticipated by core-issues2.txt itself, but blocking all of it) were found
and fixed before any of the above could begin — see "Arm environment fixes."

## Arm environment fixes (prerequisite)

| Defect | Fix | Commit |
|---|---|---|
| wntr has zero linux-arm64 EPANET support (`toolkit.py`'s Linux branch has no arch check, unlike its darwin branch) | Built EPANET 2.2 from source (`OpenWaterAnalytics/EPANET` tag `v2.2`) via `scripts/build_epanet_arm64.sh`, installed over wntr's hardcoded library path. Must be re-run after every `uv sync`. | `f540b54` |
| pyarrow was an undeclared dependency (`scenarios.py` unconditionally writes `.parquet`, but pyarrow was never in `pyproject.toml`) | Added `pyarrow>=17`, relocked. | `c851daf` |
| Cross-architecture signed-zero non-determinism (`np.maximum(0.0, -0.0)` is IEEE-754 implementation-defined) failed the `deterministic_replay` corpus gate when replaying the x86-generated Cycle B2 corpus on this Arm VM | Normalized `-0.0` to `+0.0` in the generator; hardened the gate to fall through to its existing semantic array-equality check instead of hard-failing on a hash mismatch alone. Neither the corpus's stored data nor recorded hashes were altered. | `672dce7` |

Post-fix: `python scripts/run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2`
— 9/9 passed. Full test suite passing throughout this pass (513 tests at the end,
up from 436 at the start of this session).

## Target definitions and generation methods (Phase 1)

Full head/target/loss/runtime-authority audit (two parallel research passes).
Key findings, each acted on:

- Three silent loss-key/targets_v2-name mismatches (`"action"`/`"action_pointer"`/
  `"ood"` vs. the governed `"action_template"`/`"target_pointer"`/`"ood_class"`) —
  any correctly generated governed target for these three heads would have
  silently trained nothing. Fixed, `882761f`.
- Scout/Strategist/OOD label-generation *logic* already existed
  (`scout_labels.py`, `strategist_labels.py`, `trajectory_v2.py`'s
  `TrajectoryState`/`FullTrajectory`), just wasn't wired into a corpus builder —
  this materially reduced Phase 2/3/7's scope.
- `evidence_sufficiency`'s corpus label was documented as only the sensor-health
  subset of the full governed rule; the full rule needs live controller-loop
  state, which now exists (Phase 2-4's trajectory/OOD infrastructure) — partially
  closed in Phase 5 (see below).
- `next_step` had no dormant implementation anywhere — built from scratch in
  Phase 5, derived from `agents/controller.py`'s FSM transition logic.

`validate_targets_v2` extended to fail closed on all six items Phase 1 item 3
requires (missing required masks; invalid class ranges; invalid graph-local
indices; incorrect plan dimensions; non-finite regression values; disagreement
between target metadata and topology metadata), `eb3fa71`. All labels derived from
deterministic simulation, policy, or exact verification — no manually invented
labels introduced anywhere in this pass (item 4).

## Task-specific results: what each label generator does and how it's supervised

| Target category | Module | Source of truth | Status |
|---|---|---|---|
| `ood_class` | `training/ood_labels.py` | Deterministic thresholds on `GeneratedScenario`'s own recorded generation metadata, wide margin above the corpus's documented in-distribution ranges | 6/11 governed categories reproducible now (see below); real, generated, validated against 13,150 real records |
| `evidence_sufficiency` (extended) / `next_step` | `training/control_labels.py` | Sensor health + classical posterior entropy + OOD-category calibration validity; `next_step` mirrors the live FSM's `EVIDENCE_CHECK` branch order exactly, plus one reasoned extension (`INSPECT_SENSOR`) | Generated for all 13,150 records |
| `sample_node`/`information_gain`/`candidate_reduction`/`should_continue_sampling` (Scout) | `training/scout_trajectory.py` | `generate_scout_label` (classical EIG ranking over a fitted `SignatureArtifact`) looped with a growing `already_sampled` set | Generated for all 13,150 records (JSONL only, not merged into tensor shards — see below) |
| `action_template`/`target_pointer`/`plan_validity`/`plan_value`/`consequence_vector` (Strategist) | `training/strategist_trajectory.py` | `generate_strategist_labels` (bounded deterministic plan templates, exact WNTR verification via `PlanVerifier`) | Generated for all 13,150 records (JSONL only) |
| `sensor_reconstruction`/`future_concentration`/`travel_time` | `training/auxiliary_labels.py` | The scenario's own unmasked simulated concentration / `HydraulicSimulator.build_dynamic_graph`'s travel-time edge weight | Generated and merged into tensor shards for all 13,150 records |

## Target and mask counts, class balance, missingness

`data/learning-v2/cycle-b2-trajectories/dataset-report.json` has the complete
numbers. Summary:

- **13,150 incident trajectories** across train (9,000) / validation (1,000) /
  calibration (1,000) / development_holdout (2,150), **zero generation errors**
  in any split.
- OOD category totals: NONE 12,591; SEVERE_MISSINGNESS 533; FROZEN_DRIFTING_SENSOR
  26. `development_holdout`'s SEVERE_MISSINGNESS count (346/400, 86.5%) is a real
  validation signal: it directly confirms the classifier correctly identifies most
  of the corpus's own deliberately-generated distribution-shift examples using
  their *realized* missingness, not their generation config.
- Scout sub-trajectory length is adaptive, not maxed out uniformly: ~47% of
  incidents resolve after Scout's first recommendation across every split; the
  remainder continue up to the 5-sample bound. This is evidence of real
  per-scenario evidence-sufficiency behavior.
- Strategist produced exactly 4 plan labels (3 prescreened + the guaranteed
  NO_ACTION comparator) for every one of the 13,150 records without exception —
  directly observed, consistent with `prescreen_top_plans`'s documented cap, not
  an anomaly (every scenario's network had enough candidate templates to fill it).

## Trajectory counts and lengths

Scout: 1-5 steps (bounded by `MAXIMUM_SAMPLES_BOUND=5`), hash-chained and
integrity-checked via `FullTrajectory`. Strategist: exactly 1 step (single
decision point per incident state, per Phase 3's own spec — no multi-round plan
revision attempted in this pass).

## Hard-negative counts

None deliberately curated (core-issues2.txt Phase 2/3's explicit hard-case list —
misleading sensors, near-tied plans, exposure-vs-pressure tradeoffs — requires
deliberate scenario-pair construction not attempted here, consistent with Cycle
B2's own equivalent limitation). Strategist's label sets do organically include
rejected/invalid plans as hard negatives, since `plan_validity` is read from real
WNTR verification and not every candidate template is accepted for every network
state.

## Every experiment configuration and task-specific/operational results

One configuration run this pass: `event_control_heads=True, auxiliary_heads=True`
(`scripts/run_event_control_smoke_screening.py`, Phase 8 Stage 1 — smoke and
failure screening, not architecture ranking). **Result: PASSED**, at real scale
(200 train / 50 validation examples drawn from the full 9,000/1,000-example
corpus): all 8 new targets (`event_presence`, `event_cause`,
`evidence_sufficiency`, `next_step`, `ood_class`, `sensor_reconstruction`,
`future_concentration`, `travel_time`) receive nonzero gradient; loss stays
finite; training resumes correctly past its first checkpoint; the exported
checkpoint reloads bit-for-bit finite under its own recorded architecture config.
Full results: `reports/results/v3/event-control-smoke-screening.json`,
`experiments/registry/event-control-smoke.jsonl`.

No architecture-ranking runs (Stage 2+), calibration, or full-trajectory
evaluation were performed this pass — see "Remaining work."

## Deterministic-baseline comparisons / ablations

Not performed this pass. Every comparison core-issues2.txt Phase 2/3/4 asks for
(Scout vs. random/fixed/classical-EIG baselines; Strategist vs. deterministic
templates/full-WNTR; OOD macro-F1/false-normal rate) requires a *trained* Scout/
Strategist/OOD head to compare against, which this pass's scope did not reach
(see "Remaining work"). The classical baselines themselves (random accessible
sampling, classical EIG ranking, deterministic template ordering) already exist
and are unchanged; only the "vs. learned" side of each comparison is pending.

## Which heads were enabled or disabled at runtime

No promoted checkpoint's runtime configuration changed in this pass.
`hydroswarm/tasks.py`'s `RUNTIME_TASKS`/`CORPUS_SUPERVISED_TASKS` and every
existing promoted checkpoint's `trained_tasks`/`validated_tasks` are exactly as
they were at the start of this session (`sentinel` only). The Stage 1 smoke
checkpoint produced this pass is a smoke artifact (`experiments/runs/
event-control-smoke/`, gitignored, not promoted) — it was never intended to be,
and per the plan's promotion gates (Phase 10), no head may gain runtime authority
without passing checkpoint-metadata validation and a documented promotion
decision, neither of which this pass performed.

## Safety-gate results

- WNTR/EPANET remains authoritative throughout: `plan_validity` is read only from
  `PlanVerifier`'s own decision in every one of the 13,150 Strategist label sets
  generated this pass (never from a template's predicted score) — confirmed by
  `test_strategist_trajectory.py`'s independent re-verification test and carried
  through unchanged into the full corpus generation run.
- No autonomous infrastructure control was added or modified.
- Human approval requirements are unchanged.
- OOD/calibration/abstention boundaries were not weakened — `classify_ood_category`
  and `classify_evidence_sufficiency` both fail closed by construction (any
  non-NONE OOD category invalidates calibration per `OOD_CATEGORY_BEHAVIOR`'s
  existing fail-closed table; `evidence_sufficiency` requires calibration validity
  as one of its three gating conditions, all of which must pass).
- Locked test: not opened. Confirmed by review of every command executed this
  pass; no path under a locked-test directory was read, listed, hashed, or
  referenced.

## Remaining unsupported targets / OOD categories

**OOD categories not yet reproducible** (of 11 governed): `UNSEEN_SENSOR_LAYOUT`
(the signature-artifact design has no "trained sensor layout" concept to violate
yet), `VALVE_PUMP_MISMATCH` (scenarios.py's `valve_telemetry_incorrect` is
currently a label, not a real simulated hydraulic perturbation),
`TIMING_OUTSIDE_TRAINING_RANGE` (would break `scenario_to_example`'s ordinal-bin
lookup), `UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_CALIBRATION` (a calibration-
artifact-validation concern, not a scenario property). `EXTREME_DEMAND`/
`TANK_STATE_SHIFT`/`ROUGHNESS_MISMATCH` are classifiable by the code but this
corpus's source scenarios (Cycle B2) never generated examples with those specific
knobs pushed out of range, so zero real examples of them exist in this corpus —
the classifier is ready, the underlying scenario generation is not. `UNSEEN_
TOPOLOGY` (coastal-branch) scenarios exist in `development_holdout`'s manifest but
were not processed — coastal-branch is not one of the three training topologies
this pass's generator loads.

**Scout/Strategist targets**: generated (JSONL) but not yet trainable — no
sequence-aware training loop exists to consume them (see "Remaining work").

## Checkpoint, normalization, calibration, topology, and manifest hashes

- Validated topology hashes (all three training topologies, unchanged from Cycle
  B2): `0b1817cd6c28d42f98b1a1a74cb0234d619ee2985b1c7cf70cba4f274094b056`
  (golden-reference), `0e9cfc042e0876f34a8ecbf9435bcbee3c2d840462a274e5ca831c3b40e4fe88`
  (branched-loop), `628a6dccfeff1af5a81a41d7374f8408085611ddf5ac925ff01e7b809c89464e`
  (loop-grid).
- Normalization: unchanged from Cycle B2 (`node_normalization_sha256=
  4dcf22a68839a8630e83b0e90f47ac3400b176b576e76d8bee5662221d238691`,
  `edge_normalization_sha256=3e715d707475d81eba90de6609246f51bb0eee8a94c58eab4958f4370fca514d`)
  — this pass only added targets, never touched input feature normalization.
  `merge_trajectory_targets.py` preserves the source shards' `.inputs` verbatim.
- No new checkpoint was promoted; the Stage 1 smoke checkpoint's own hash is
  recorded in `experiments/registry/event-control-smoke.jsonl` (a smoke artifact,
  not a promotion candidate).
- Calibration: not refit this pass (existing Cycle B2 calibration artifacts,
  `E1-seed20260810`/`E0-seed20260811`, are unchanged and untouched).

## Exact reproduction and resume commands

```bash
export PYTHONPATH=src

# Environment (Arm hosts only)
./scripts/build_epanet_arm64.sh

# Regenerate the trajectory corpus from data/learning-v2/cycle-b2 (idempotent --
# already-processed scenario_ids are skipped)
for split in train validation calibration development_holdout; do
  python scripts/generate_trajectory_corpus.py \
    --corpus-dir data/learning-v2/cycle-b2 \
    --output data/learning-v2/cycle-b2-trajectories \
    --split "$split"
done

# Merge into trainable tensor shards
for split in train validation calibration development_holdout; do
  python scripts/merge_trajectory_targets.py \
    --tensor-shard-dir "data/learning-v2/cycle-b2/tensors-normalized/${split}" \
    --trajectory-jsonl "data/learning-v2/cycle-b2-trajectories/${split}.jsonl" \
    --output "data/learning-v2/cycle-b2-trajectories/tensors-enriched/${split}" \
    --split "$split"
done
# development_holdout's OOD-holdout scenarios live in a separate source shard dir:
python scripts/merge_trajectory_targets.py \
  --tensor-shard-dir data/learning-v2/cycle-b2/tensors-normalized/ood-SEVERE_MISSINGNESS \
  --trajectory-jsonl data/learning-v2/cycle-b2-trajectories/development_holdout.jsonl \
  --output data/learning-v2/cycle-b2-trajectories/tensors-enriched/ood-SEVERE_MISSINGNESS \
  --split development_holdout

# Stage 1 smoke and failure screening (already passed; reproduce or re-verify)
python scripts/run_event_control_smoke_screening.py \
  --corpus-root data/learning-v2/cycle-b2-trajectories/tensors-enriched --tensors-dirname ""

# Full test suite
python -m pytest -q   # 513 passed at the end of this pass
```

## Remaining work (for the next pass)

In priority order:

1. **A sequence-aware training loop for Scout/Strategist.** Their targets are
   inherently per-step/per-plan (not flat per-example tensors), so
   `Trainer`/`compute_multitask_loss`/`collate_variable_topology` as they exist
   today cannot consume `data/learning-v2/cycle-b2-trajectories/*.jsonl`'s
   `scout`/`strategist` fields. This is the single largest remaining piece —
   everything else in Phase 8-10 that touches Scout/Strategist is gated on it.
2. **Phase 8 stages 2-7**: architecture screening with the new heads at real
   scale (this pass only ran Stage 1's smoke/failure check, not a ranking
   sweep), Scout supervision, Strategist supervision, learned OOD, validated
   auxiliary-objective ablation, joint multitask fine-tuning.
3. **Deterministic-baseline comparisons and ablations** (Phase 2/3/4's explicit
   evaluation requirements) — blocked on (1)/(2) producing a trained head to
   compare against.
4. **Calibration and promotion** (Phase 8 Stage 6, Phase 10's gates) — blocked on
   (2).
5. **The 5 OOD categories with zero real examples** in this corpus (see
   "Remaining unsupported targets" above) — would need either new scenario-
   generation knobs (`VALVE_PUMP_MISMATCH`, real hydraulic perturbation) or
   simply running the existing classifier against a corpus that varies
   demand/tank/roughness knobs out of range (`EXTREME_DEMAND`/`TANK_STATE_SHIFT`/
   `ROUGHNESS_MISMATCH` — the classifier is ready today, only the source
   scenarios are missing).

Locked-test discipline carries forward unchanged: do not open it until every
Phase 8-10 condition in overnight-plan.txt's Stage 6/7 is satisfied and
`final-selection.json` exists.
