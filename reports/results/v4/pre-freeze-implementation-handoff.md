# core-issues3.txt pre-freeze implementation — live handoff report

Branch: `agent/gcp-multitopology-v3`. Living document, updated after each
major phase. See `reports/results/v4/pre-freeze-audit.md` for the Phase 0
audit this pass started from, and `/workspace/core-issues3.txt` for the full
specification.

**The locked final test has not been opened. `final-selection.json` does not
exist.** No work has occurred on `main`. `data/learning-v2/cycle-b2` (existing
contents), all promoted checkpoints, and every existing v3 result artifact
remain untouched.

## Status summary

| Phase | Task | Status |
|---|---|---|
| 0 | Audit current HEAD | **DONE** |
| 1 | Reconstruct exact scenario hydraulic state | **DONE** (code + tests) |
| 2 | Governed signature-artifact policy | **DONE** (documented + wired; approximation error unmeasured, flagged) |
| 3 | Repair Strategist label semantics | **DONE** (code + tests) |
| 4 | Candidate-conditioned Strategist | **DONE** (architecture + tests; not yet wired to real training data) |
| 5 | Closed-loop Scout states | not started; scoped below |
| 6 | OOD taxonomy / event-cause | not started |
| 7 | Auxiliary objectives / regression losses | not started |
| 8 | Second-pass calibrated control targets | not started |
| 9-20 | Architecture v4, training, gates, selection | not started |

Corpus regeneration (`data/learning-v2/cycle-b2-trajectories-v2/`) running;
`validation`/`calibration` complete with 0 errors, `train`/
`development_holdout` in progress — see its own section below.

## Commits this pass (newest last)

1. `5f459e7` audit(pre-freeze): record current architecture and artifact gaps
2. `76ec631` fix(data): reconstruct exact scenario hydraulic contexts
3. `ce82bde` fix(data): report unsupported-topology skips instead of silently continuing
4. `5372075` fix(data): tolerate sub-float32-rounding noise on negligible-strength scenarios
5. `bb7698e` feat(classical): wire the signature registry into trajectory generation
6. `93d5bc3` fix(data): use stored scenario data for trajectories, not regenerated arrays
7. `668092b` docs(handoff): publish pre-freeze implementation handoff after Phase 1-2
8. `bfaf676` fix(strategist): derive exact consequence values and semantic targets
9. `849accf` docs(handoff): update after Phase 3 completion and corpus regeneration restart
10. `7628714` feat(model): add candidate-conditioned strategist architecture v4

All pushed to `origin/agent/gcp-multitopology-v3`. Working tree clean apart
from the in-progress `data/learning-v2/cycle-b2-trajectories-v2/` and its
`experiments/jobs/cycle-b2-trajectories-v2-*/` job directories (uncommitted
until generation finishes with 0 errors on every split).

## Phase 1: reconstruct exact scenario hydraulic state — DONE

**Root cause confirmed**: `scripts/generate_trajectory_corpus.py` built exactly
one pristine WNTR network + `FeatureContext` per topology family and reused
both for every scenario in that family, discarding each scenario's own
randomized demand regime, roughness perturbation, tank-level variation, and
pipe-outage state. `build_incident_trajectory`, `scenario_to_example`, and
`build_strategist_trajectory` themselves already correctly accept and use a
per-scenario `network`/`feature_context` (fixed in the earlier
`core-issues.txt` repair pass) — the defect was isolated to this one caller,
narrowing the fix's scope considerably.

**Fix**: `hydroswarm/training/scenario_reconstruction.py`'s
`reconstruct_scenario_network()` is now the single canonical replay/
reconstruction function (used by both `run_corpus_gates.py`'s
`deterministic_replay` gate and `generate_trajectory_corpus.py`, per Phase 1
item 3). It replays a stored `ScenarioManifest` against its pristine topology
and returns the exact randomized network, its derived `FeatureContext`, and
identity hashes (topology/network-state/hydraulic-state), failing closed
(`ScenarioReconstructionError`) when the replay doesn't semantically match.

**Three real defects found running this at real scale against
`data/learning-v2/cycle-b2` (13,150 scenarios), not by inspection alone**:

1. A regression in my own first-draft refactor of the gate's replay-
   verification logic: short-circuited the array-level comparison whenever
   the *recorded* manifest hash matched, which doesn't detect tampering that
   only touches the raw `.npz` file. Caught by the existing
   `test_deterministic_replay_gate_fails_closed_on_tampered_artifact` test.
   Fixed: the array comparison is now unconditional whenever an `original`
   scenario is supplied.
2. Sub-float32-rounding noise (~2.7e-19 on ~1.5e-8-magnitude
   `NEGLIGIBLE_STRENGTH_MG_MIN` concentrations) between two independently-
   deterministic reconstructions and the originally-stored corpus array —
   same class of cross-environment nondeterminism as the already-documented
   signed-zero case, just below `np.array_equal`'s resolution instead of at
   it. Fixed with an `atol=1e-6` `np.allclose` fallback (three orders of
   magnitude below `quantization_step`, the smallest physically meaningful
   resolution anywhere else in the system) — verified this does NOT mask a
   real difference (a hand-tampered 0.01 mg/L difference still raises).
3. `development_holdout` mixes plain-curriculum scenarios with two
   OOD-holdout helpers (`generate_cycle_b_corpus.py`'s
   `_generate_ood_holdout_for_training_topology`) that use a materially
   different, hardcoded degradation-probability formula
   (`missing_probability=0.45` vs. the curriculum formula's `<=0.08`) —
   already known and documented: `run_corpus_gates.py`'s own
   `deterministic_replay` gate excludes this exact split because "which
   formula was used ... cannot be distinguished from the manifest alone."
   This caused a genuine ~23% spurious failure rate on `development_holdout`
   (218/950 scenarios in the run that surfaced it). Fixed: retries with the
   array-level check skipped (only for this split, only when the full check
   fails) and records `weak_verification: true` on the affected rows —
   visible, not silently smoothed over. `replay_sha256` (seed/source/network/
   stage/split identity) is still verified either way.

Also found and fixed while wiring: `build_incident_trajectory` was being
called with `reconstruction.scenario` (a freshly regenerated copy, produced
only to support the verification check) instead of the original,
already-ground-truth scenario loaded from disk. There is no reason to prefer
regenerated data over data the corpus already has stored correctly. Now
always builds from the original `scenario`; only `reconstruction.network`/
`.feature_context` (genuinely not stored anywhere else) come from
reconstruction.

Also fixed (Phase 1 item J): unsupported-topology scenarios (currently
`development_holdout`'s coastal-branch/unseen-topology subset — no governed
signature artifact exists for it yet, see Phase 2) were silently `continue`d
past. Now counted and reported in both the per-split `report.json`
(`skipped_unsupported_topology_this_run`) and a printed warning.

**6 + 2 = 8 new regression tests**
(`tests/scientific/test_scenario_reconstruction.py`): different seeds produce
different network/hydraulic-state hashes; reconstruction matches original
semantic replay; fails closed on a manifest that doesn't match the given
topology; travel-time labels change with hydraulic state; reconstructed
network is not the pristine object; a direct proof the old shared-context
shape would have collapsed two scenarios' travel-time labels; negligible-
magnitude float noise does not fail reconstruction; a real-magnitude
difference still fails closed.

**521 tests passing** (was 513 at session start), ruff/pyright clean, 9/9
corpus gates.

**Not done in this pass** (Phase 1 item 6, explicitly deferred, not silently
dropped): extending topology loading to `development_holdout`'s coastal-
branch/unseen-topology scenarios. Phase 2 items 5/R forbid fitting a
topology-specific signature artifact from development-holdout incidents, so
processing these requires the governed "Scout/Strategist unavailable,
abstain/fall back" path Phase 2 item 5 describes, which doesn't exist yet.
Tracked, reported per-run, not resolved.

## Phase 2: governed signature-artifact policy — DONE

**Audit finding**: `hydroswarm/classical/signature_registry.py`'s
`SignatureRegistry` (train-only-fitting guard, deterministic cache-key
digests, fail-closed `require()`) already existed from the earlier
`overnight-plan.txt` Task 1.2 repair pass, with its own full test suite
(`tests/scientific/test_signature_registry.py` — cache-key completeness,
no-cross-state-hit, deterministic rebuild, separate topology hashes) — but
was never actually called anywhere. `generate_trajectory_corpus.py` fit
artifacts straight against `SignatureCache`, bypassing the registry layer.

**Policy documented and named**: `TOPOLOGY_WIDE_REGIME_HASH` — every consumer
today fits exactly one signature artifact per topology from that topology's
full train-split population, regardless of demand regime or other in-corpus
hydraulic variation. This is Phase 2's "Option B: bucketed hydraulic-regime
artifacts" at the coarsest possible bucket boundary (one bucket per
topology, not per regime). Not a behavior change — `register()` only adds
governed bookkeeping on top of the same underlying artifact-fitting calls.

**Remaining work, explicitly flagged**: approximation error against exact
per-scenario state-specific artifacts is unmeasured. Runtime has no
signature-artifact consumption path yet to compare against training's
policy for equality (grepped `src/hydroswarm/runtime/defaults.py` and
`inference/`: zero references) — there is nothing on the runtime side to
wire the registry into yet, so "runtime/training policy equality" (Phase 2
item 7's last bullet) has no runtime half to test against.

## Phase 3: repair Strategist label semantics — DONE

Three real defects fixed in `strategist_labels.py` (full detail in commit
`bfaf676`'s message):

1. **Selection bias (3.1)**: training-label generation only verified plans
   the old heuristic prescreener selected (`prescreen_top_plans`,
   `predicted_validity >= 0.5`, top-3-by-predicted-score) plus `NO_ACTION` —
   using a heuristic to decide which candidates receive labels would have
   made it structurally impossible for a learned prescreener to ever beat
   it. `generate_strategist_labels` now exactly WNTR-verifies the FULL
   bounded candidate set (up to the canonical 9 templates).
2. **Ungoverned plan_value (3.2/3.3)**: was
   `proposal.predicted_value * proposal.predicted_validity` — the old
   heuristic's own unverified score. New
   `hydroswarm.planning.plan_value_policy` module (versioned,
   `PLAN_VALUE_POLICY_VERSION`) derives `plan_value`/`regret` and all five
   previously-defined-but-never-populated consequence-proxy targets
   (`exposure_proxy`, `pressure_risk_proxy`, `service_loss_proxy`,
   `containment_time_proxy`, `plan_regret_proxy`) from exact WNTR
   `ConsequenceMetrics` only. 6 monotonicity tests pass (lower exposure/
   fewer pressure violations/greater service availability/shorter
   containment time cannot reduce `plan_value`; best plan in a pool has
   zero regret; `NO_ACTION` is scored identically to every other
   candidate, never given an automatic free pass).
3. **Broken target-pointer semantics (3.4)**: `target_pointer` was computed
   via `sorted(network.junction_name_list)` at label-generation time — the
   same node-space bug class already found and fixed for Scout/Sentinel
   (junction-only order silently disagrees with the canonical node space
   every other node-indexed target uses), and link targets were silently
   dropped entirely (`positions.get()` returned `None` for link names,
   since `positions` only ever contained junction names).
   `StrategistLabel` now carries semantic identity
   (`primary_target_id`/`primary_target_type: NONE|NODE|LINK`); index
   resolution against the scenario's own canonical `node_ids`/`edge_ids`
   now happens only at tensor-building time
   (`strategist_trajectory._resolve_target_pointer`).

Also (3.5): centralized the canonical 9-template vocabulary into
`hydroswarm.planning.action_templates`; raised `generate_response_plans`'
`maximum_plans` cap from 8 to 9 (it structurally excluded the 9th template —
`ALTERNATE_VALVE_CUT` — whenever the other 8 were eligible, a second,
independent instance of the known 8-vs-9 mismatch).

**Attempted and reverted**: raising `HydroCore.action_vocabulary_size`'s
default from 8 to 9 to match. The promoted checkpoint does not pin this in
its recorded architecture config, so it silently relies on the constructor
default despite never training the action head — raising it broke strict
reload of that checkpoint (`action_head`'s saved weight shape is `[8, ...]`),
caught immediately by the full test suite
(`test_default_pipeline_factory.py`). Reverted with a comment explaining
why; deferred to Phase 9's architecture-v4 contract, where config
completeness is meant to be strictly validated rather than left to a
constructor default. **This is a real, generalizable lesson for the rest of
this pass**: any shared model-constructor default change needs a full test
run before being treated as safe, even when the head in question appears
untrained everywhere the audit checked.

15 new/rewritten tests. 536 tests passing (was 513 at session start),
ruff/pyright clean.

## Trajectory corpus regeneration (data/learning-v2/cycle-b2-trajectories-v2/)

Running as 4 resumable background jobs (`experiments/jobs/
cycle-b2-trajectories-v2-{train,validation,calibration,development_holdout}`),
launched via `hydroswarm.training.job_runner`, polled at a 10-minute interval.

**Regenerated from scratch after Phase 3** (second restart): the first
complete run (Phase 1+2 fixes only, `validation`/`calibration` reached 0
errors, `development_holdout` reached 0 errors with 327/2150 rows using the
documented `weak_verification` fallback, `train` was ~18% through) predated
the Phase 3 Strategist fix — every row written under that run had the old,
buggy `plan_value`/target-pointer semantics. Rather than ship a corpus with
inconsistent Strategist-label semantics across rows, all four splits' output
was deleted (confirmed via `git status` that nothing had been committed yet)
and regeneration restarted clean against commit `bfaf676`. In progress as of
this report.

The old `data/learning-v2/cycle-b2-trajectories/` (built with the pristine-
context bug) is left untouched and remains marked provisional/invalid per
restriction #5 — not used for any of this pass's work.

Exact resume commands (idempotent — skips already-processed scenario_ids):

```bash
export PYTHONPATH=src
for split in train validation calibration development_holdout; do
  python scripts/generate_trajectory_corpus.py \
    --corpus-dir data/learning-v2/cycle-b2 \
    --output data/learning-v2/cycle-b2-trajectories-v2 \
    --split "$split"
done
```

Job status: `cat experiments/jobs/cycle-b2-trajectories-v2-<split>/status.json`
(note: `state` is only updated by an explicit `mark_finished()` call — check
`pid` liveness directly with `kill -0 <pid>` for ground truth while a job is
running, since nothing auto-reconciles `status.json` when a plain script
process exits on its own).

## Phase 4: candidate-conditioned Strategist architecture — DONE (architecture only)

`hydroswarm.model.candidate_plan_encoder.CandidatePlanEncoder` (commit
`7628714`) replaces the anonymous learned plan-query representation with one
built from each candidate plan's own template/target/features. Wired into
`HydroCore` behind a new `strategist_mode` flag
(`"anonymous_queries"` default / `"candidate_conditioned"`), following the
existing net-new-parameters-gated-behind-a-flag convention
(`event_control_heads`/`auxiliary_heads`/`consequence_prescreening_heads`):
zero new parameters and zero behavior change in default mode. Added to
`architecture_config()`/`verify_architecture_compatibility()`.

**Real regression found and fixed during this phase, not just the intended
work**: raising `action_vocabulary_size`'s default from 8 to 9 (attempted as
part of Phase 3, reverted) broke the promoted checkpoint's strict reload.
Learned from that: before committing Phase 4, re-ran
`test_default_pipeline_factory.py` specifically (the file that caught it)
in addition to the full suite — both clean.

**Not done in this pass** (explicitly deferred, not silently dropped):
wiring real training data into this path. That requires Phase 10's
Strategist collator (variable candidate count per incident, none of which
exists yet) and the runtime top-K-exact-verification integration Phase 15
describes. This commit delivers and tests the architecture component and
its `HydroCore` integration; nothing trains through it yet.

## Restrictions honored

No work on `main`. `data/learning-v2/cycle-b2`'s existing contents, all
promoted checkpoints, and every existing v3 result artifact are untouched.
Locked test not opened; `final-selection.json` does not exist. No destructive
git/filesystem commands used. No sudo, no credential exposure. All commits
pushed to `origin/agent/gcp-multitopology-v3`.

## Next steps

1. Let the 4-split regeneration finish (`train`/`development_holdout` were
   at ~1150/9000 and ~1300/2550 respectively as of this report, ~0.75-0.8
   scenarios/s; `validation`/`calibration` already complete with 0 errors),
   verify each split's `report.json` shows `errors_this_run: 0`, then
   commit the new corpus's JSONL/manifests/reports (this stage only
   produces JSONL, not tensor shards — no Git LFS needed).
2. **Phase 5: real closed-loop Scout states** — the largest remaining
   labeled-data gap. Concrete scope for the next pass:
   - The core missing capability is **arbitrary-node simulated truth**:
     `GeneratedScenario` only stores concentration for its originally-chosen
     sensor subset, but Scout must be able to sample any accessible
     candidate node. Phase 1's `reconstruct_scenario_network` now makes this
     tractable — it returns the exact randomized network a scenario was
     simulated against, which can be re-run through
     `HydraulicSimulator.simulate_incident` reading out concentration at
     every candidate node, not just the original sensor subset.
   - Each Scout step must reveal a genuinely NEW observation (deterministic
     measurement seed = `scenario_id + step_index + node_id`, degraded
     through the same governed noise/quantization policy `_degrade` already
     uses) and fold only that into the NEXT state's evidence -- never
     re-rank against the same fixed base observations the current
     `generate_scout_label`/`build_scout_trajectory` do (see
     `scout_labels.py`'s own module docstring, which already documents this
     exact simplification and names Phase 7/5 as where it gets fixed).
   - Add explicit cutoff assertions (no input timestamp beyond the current
     state's cutoff, no future sample outcome visible before it's selected)
     -- Phase 5 item 5.3's own required tests.
   - Accessibility/hard-case generation (best-EIG-node inaccessible,
     already-sampled, near-tied EIG, exhausted budget) can follow once
     incremental revelation itself works; do not attempt both in the same
     change.
   - This is comparable in size to Phase 3's Strategist repair. Budget a
     dedicated pass for it rather than a partial attempt appended to
     another phase's work.
3. Phase 6: split OOD category (11-class) from deterministic severity
   (3-class) — currently conflated.
4. Phase 7: masked-regression helper honoring target masks (currently the
   generic MSE path ignores mask companions for several targets).
5. Phase 9: architecture-v4 contract — `strategist_mode` and every other
   Phase-4.x flag this pass and prior passes added are already individually
   recorded in `architecture_config()`, but Phase 9 wants a single strictly-
   validated contract (trained/validated/runtime-enabled output sets, not
   role-level gating) rather than the current per-flag compatibility checks.
