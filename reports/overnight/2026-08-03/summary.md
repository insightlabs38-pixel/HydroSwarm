# Overnight HydroSwarm Handoff

Plan: `overnight-plan.txt` (2026-08-03 v3), multi-topology GCP execution and training run.

## Repository state

- Branch: `agent/gcp-multitopology-v3`
- Starting commit: `5697f912667fa236ece784a98f141c8162ff6bf8` (main, "Complete HydroCore-M
  evaluation and topology transfer study")
- Ending commit: `0606586` (Bundle E complete; still running)
- Commits created: 44 (see `git log --oneline main..agent/gcp-multitopology-v3`)
- Working tree: clean at branch creation

## Completed tasks

- [x] Phase 0 — baseline verification and branch creation (Task 0.1)
- [x] Phase 0.2-0.8 (Bundle A: registry, job runner, sharded lazy data, label audit, normalization, sampler, split policy)
- [x] Phase 1.1-1.5 (Bundle B: variable-topology corpus architecture — topology metadata, per-topology
  signature registry, variable-size collation, scenario-specific hydraulic context caching, permutation
  equivariance). This was the plan's explicitly highest-priority technical change.
- [x] Phase 2.1-2.6 (Bundle B complete: targets_v2 contract, Sentinel/Scout/Strategist label generation
  against the real simulator/classical/planning/WNTR stack, OOD categories, trajectory serialization)
- [x] Phase 3 / Bundle C (frontend live/demo data integrity, Tasks 3.1, 3.3(partial), 3.4(partial), 3.5,
  3.6, 3.7, 3.8 complete; Task 3.2 given a narrower interim treatment -- see Follow-up)
- [x] Phase 4 / Bundle D (Tasks 4.0-4.6: architecture versioning + compatibility contract,
  configurable classical-prior injection, source-conditioned incident pooling, dual hydraulic
  message channels, event/next-step control heads, three auxiliary objectives, non-authoritative
  plan consequence prescreening -- every new flag defaults to exactly reproducing the promoted
  checkpoint's original behavior, verified after each task)
- [x] Bundle E (Cycle A corpus: 2,550 scenarios across 2 genuinely different topologies; Stage 1
  smoke/failure screening for E0/E3/E4/E9 -- 6 real training runs, all passed: no NaNs, every
  supervised head received gradients, every run resumed correctly)
- [ ] Cycle B — next up
- [ ] S architecture screening
- [ ] S finalist training
- [ ] M training/evaluation
- [ ] calibration/OOD evaluation
- [ ] full-trajectory benchmark
- [ ] final selection
- [ ] locked final test

## Tests (current, after Bundle E)

See `test-results.md` for full detail.

| Command | Result | Notes |
|---|---|---|
| `pytest -q` | 355 passed | started at 1 failed/97 passed; pre-existing failure fixed in `19468ac`; 257 new tests added across Bundle A + Bundle B + Bundle D + Bundle E |
| `ruff check src tests scripts` | pass | |
| `pyright` | pass | |
| `npm run lint` | pass | |
| `npm run test -- --run` (vitest) | pass (24 tests, up from 4) | unchanged since Bundle C; Bundles D/E touched no frontend code |
| `npm run build` | pass | |
| `npx playwright test` (e2e) | pass (10 tests, up from 1) | real Chromium, includes 2 committed screenshot baselines |
| HydroCore-S checkpoint load | pass | hash, feature-schema, and calibration all validate; re-verified after every Task 4.x and Bundle E commit |

## Datasets generated

**Cycle A** (`data/learning-v2/cycle-a/`, ~2,550 scenarios): 1,750 train / 250 validation / 250
calibration / 300 development_holdout, split evenly across two genuinely different topologies
(golden reference: 4 junctions/1 reservoir/1 tank/one loop; branched-loop: 7 junctions/1
reservoir/no tank/a different loop, already committed at
`data/topology-transfer/branched-loop.inp`). Generated in ~3 minutes. Manifests, signatures,
`dataset-report.json`, and `label-audit.json` are committed (~4.5MB); raw scenario arrays and
tensor shards (~55MB, regenerable) are gitignored. See `data/learning-v2/cycle-a/dataset-report.json`
for full counts/balance/leakage detail.

## Experiments completed

**Bundle E Stage 1 smoke/failure screening** (6 real training runs against a 200-train/50-validation
Cycle A subset): E0 (baseline), E3 (source-conditioned pooling), E4 (dual-gated message channels),
and three of E9's four internal prior/fusion comparisons (none/feature_only/logit_only -- the
fourth, feature_and_logit, is E0 itself). Every run: validation loss decreased between the initial
2-epoch pass and a resumed 3rd epoch, zero NaNs anywhere (loss, gradients, or exported checkpoint
weights), every supervised head present in the batch received a nonzero gradient, every resume
correctly advanced global_steps and epochs_completed. Full sweep: ~2m24s CPU time. Report:
`reports/results/v3/architecture-smoke-jobs.json`. Provenance: `experiments/registry/bundle-e-smoke.jsonl`
(real `ExperimentRegistry` open/close records with git commit, manifest hashes, resolved config,
host/CPU/RAM). Run checkpoints (~1.3GB, disposable smoke artifacts) are gitignored under
`experiments/runs/bundle-e-smoke/`.

## Metrics obtained

None for final claims yet (baseline HydroCore-S hybrid governed result remains the current
reference: 96.0% top-1, ECE 0.0269, 8.94ms latency, per the plan's existing governed results
table). Bundle E's smoke-job validation losses (`reports/results/v3/architecture-smoke-jobs.json`)
are explicitly *not* comparable architecture-selection metrics -- 200-example subset, 2-3 epochs,
Cycle A is a smoke corpus by design -- they only prove the training loop itself works correctly
for each configuration.

## Important findings

- Bundle E's real-corpus, real-training-loop work surfaced three genuine correctness bugs that
  every prior synthetic-fixture test had missed, each caught only by actually running the full
  pipeline end to end against real multi-topology data for the first time:
  1. `label_audit._sensor_fault_prevalence` unconditionally `torch.stack`ed every example's
     `sensor_fault` target across a whole split, which assumes one shared node count -- true for
     every corpus that existed before Cycle A, false the moment a split genuinely mixes two
     topologies. Fixed by computing the overall rate via concatenation and per-node rates grouped
     by `network_id`.
  2. `compute_multitask_loss` never read any of targets_v2's `f"{task}_mask"` companion tensors at
     all, so `corpus.py`'s placeholder values (0) for `source_node`/`source_region`/`start_time`/
     `duration`/`relative_strength` on NORMAL/SENSOR_FAULT_ONLY scenarios were silently trained
     against as if they were real labels -- roughly 30% of Cycle A's examples. Fixed with
     `_apply_target_mask`, folding the mask into the `-100` "ignore this position" sentinel the
     loss functions already understood. `source_region` was also found to have no loss wired to it
     at all (a real head, a real corpus-emitted target, zero supervision) and is now wired.
  3. `HydroCore.evidence_head`'s output was never squeezed (`[batch, 1]`) unlike the otherwise
     structurally identical `event_presence_logits` (`[batch]`, correctly squeezed), so the first
     real forward-pass-into-loss run crashed with a shape mismatch the moment a real corpus target
     was used instead of a hand-matched synthetic one.
  All three are fixed, each with a dedicated regression test proving both the original failure
  mode and the fix (see commits `76cb9a8`, `470a042`, `0606586`). This is exactly Cycle A's stated
  purpose ("target coverage validation... shape and memory tests") working as intended -- these
  bugs would otherwise have silently corrupted Cycle B/C training at 8,000-40,000-scenario scale.
- HydroCore-S is empirically permutation-equivariant: no non-equivariant input feature
  was found across 6 random node-order permutations (source-logit differences <1e-4,
  predictions always agree). This is a positive, load-bearing result for multi-topology
  training since it means node-array-position cannot leak into predictions.
- A real correctness bug was found and fixed in the Task 1.1 work: `source_node`'s local
  index must map through the full `node_ids` space, not a possibly-shorter
  `source_candidate_ids` subset. Caught by design (writing a regression test with a
  genuine subset) before any real multi-topology data existed to expose it silently.
- The existing codebase already had more of the "variable topology" foundation than the
  plan's description suggested: `pad_graph_batch`, `HydraulicSimulator.state_hash()`, and
  `SignatureCacheKey` were already doing much of the padding/masking and per-network-state
  hashing correctly. Tasks 1.2-1.4 ended up being targeted extensions (a lookup registry,
  additional cache-key fields, a bridging collate function) rather than new subsystems.
- The label-audit tool run against the real learning-v1 corpus found zero data-quality
  issues (0 duplicates, 0 impossible labels, 0 leakage) and a classical-signature-prior
  sanity baseline of 81-85%, consistent with the existing governed 91.5% classical result.
- Tasks 2.2-2.4 (Sentinel/Scout/Strategist label generation) turned out to build on
  substantially more existing, already-tested infrastructure than expected:
  `hydroswarm.sampling.active.rank_sample_locations` and
  `hydroswarm.planning.response.generate_response_plans`/`prescreen_top_plans` already
  implemented nearly everything those tasks describe. The work was writing the glue
  connecting them to governed scenario/corpus data, not building new subsystems, and every
  test in this stretch ran against the real reference network with real WNTR simulation
  (no mocked simulator).
- Found and fixed a second real bug during Task 2.2 verification: the forced sensor-fault
  injection for SENSOR_FAULT_ONLY scenarios could write a finite value into a slot the
  missingness mask had already marked absent, violating the corpus's finite/missing
  invariant. Caught by a targeted high-missingness regression test before it could
  corrupt generated data.
- Bundle C's core P0 bug (frontend silently presenting partial live data as fully live) was
  real and exactly as the plan described: `fetchIncident()` spread the entire demo fixture
  and overwrote only ~8 of ~20 IncidentView fields with live data while labeling the result
  `source: 'api'`. Fixed by refusing to claim LIVE until the API response is genuinely
  complete (Task 3.2's future work), rather than partially patching the merge.
- Found and fixed real Playwright test flakiness while building Task 3.7's coverage:
  `locator.count()` does not auto-wait the way `toBeVisible()` does, so checking it
  immediately after only the (Suspense-external) header banner appeared raced against
  lazy-loaded map/chart chunks in a real browser. This would have made any future e2e test
  relying on the plan table intermittently fail for reasons unrelated to the code under
  test if not caught here.
- The project's own governed model results (HydroCore-S hybrid promoted, 96.0% top-1) were
  being actively contradicted by the frontend's validation/benchmark pages ("NOT RUN" /
  "no trained checkpoint included") -- likely stale copy left over from before the model was
  trained. Fixed with real, source-hash-verified numbers (Task 3.6).
- Bundle D (Task 4.4 in particular) surfaced a real design constraint the plan implies but
  doesn't spell out: Tasks 4.1-4.3 (prior_mode/incident_pooling/message_direction) could each
  default to reproducing the promoted checkpoint's exact original behavior because each made
  an *already-existing* pathway configurable. Task 4.4's event/next-step heads and Task 4.5's
  auxiliary heads and Task 4.6's consequence proxies are net-new parameters with no prior
  existence at all -- unconditional construction was tested and directly confirmed to break
  `DefaultPipelineFactory().trained_assets_ready` (strict `load_state_dict` fails on missing
  keys). All three were gated behind their own `*_heads: bool = False` flag instead, keeping
  every Task 4.x flag's default exactly checkpoint-compatible; verified after every single
  commit in Bundle D, not just at the end.
- Auxiliary (Task 4.5) and consequence-prescreening (Task 4.6) label generation from real
  corpus/PlanVerifier data, next_step's deterministic-controller-driven labels (Task 4.4),
  inference-pipeline serialization of the new heads, and the ranking-quality evaluation
  harness Task 4.6 calls for are all deliberately deferred to Phase 5/6/7 -- see follow-up.md.
  The architecture/config/loss-wiring/checkpoint-compatibility work is complete and tested;
  only the data/evaluation half of these three tasks remains, gated behind corpus generation
  that hasn't started yet regardless.

## Remaining blockers

- None currently. The one pre-existing test failure was fixed (see below).
- Task 3.2 (a complete `/incidents/{id}/view` backend contract) was not implemented in full;
  `fetchIncident()` instead documents and throws on exactly the fields the current API
  doesn't provide. This means LIVE mode cannot actually be reached yet even against a
  running backend -- DEMO_FALLBACK (or ERROR, if an incident ID is configured but the API
  itself is unreachable/misconfigured) is always what renders today. See follow-up.md.

## Exact commands to continue

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q   # expect 355 passed
cd frontend && npm run test -- --run   # expect 24 passed
npx playwright test                     # expect 10 passed
```

See `follow-up.md` for the specific next task (Phase 5 / Bundle F: Cycle B corpus generation,
or completing Task 3.2's full backend contract).

## Every training job's status

Bundle E's 6 smoke-screening jobs (E0/E3/E4/E9-none/E9-feature_only/E9-logit_only) all
completed successfully; see `training-jobs.md` and
`reports/results/v3/architecture-smoke-jobs.json`. None are still running. No Bundle F/G/H
training has been launched yet.

## Locked final test

Not opened. Will only be opened once `reports/results/v3/final-selection.json` exists and
every Stage 6 condition in the plan is satisfied.

---

This file, and the rest of `reports/overnight/2026-08-03/`, will be kept current after every
independently validated milestone for the remainder of the run.
