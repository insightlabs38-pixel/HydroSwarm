# Overnight HydroSwarm Handoff

Plan: `overnight-plan.txt` (2026-08-03 v3), multi-topology GCP execution and training run.

## Repository state

- Branch: `agent/gcp-multitopology-v3`
- Starting commit: `5697f912667fa236ece784a98f141c8162ff6bf8` (main, "Complete HydroCore-M
  evaluation and topology transfer study")
- Ending commit: `ddf6fd9` (Cycle B corpus landed, Task 3.2 complete, a real hard-coded-plan
  bug found and fixed, Stage 2 architecture screening completed, Stage 3 finalist training
  running in the background; still running)
- Commits created: 52 (see `git log --oneline main..agent/gcp-multitopology-v3`)
- Working tree: clean except the actively-running Stage 3 job's own `experiments/jobs/` and
  `experiments/registry/bundle-f-stage3.jsonl` (uncommitted until the job finishes -- see
  "Every training job's status" below)
- Pushed to `origin/agent/gcp-multitopology-v3` on GitHub through commit `ddf6fd9`

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
- [x] Task 3.2 — complete `/incidents/{id}/view` API contract (backend Pydantic schema +
  endpoint, frontend TS type + `viewFromApi` mapping + wiring, backend contract tests driving a
  real `HybridInferencePipeline` through the HTTP API, frontend unit tests, full
  build/lint/typecheck/vitest/playwright all green). Previously only had the interim
  "throw rather than fake it" treatment; now fully implemented. See "Task 3.2" section below.
- [x] Bundle F, Cycle B corpus generation (`data/learning-v2/cycle-b/`): 12,750 scenarios across
  3 training topologies + 1 development-OOD topology. See "Datasets generated" below.
- [x] fix(frontend): `Counterfactuals.tsx` derived its comparison branches from `plans[0]`/
  `plans[1]` position and hard-coded "PLAN A"/"PLAN B" labels -- a real Task 3.3 violation
  ("hard-coded recommended branch") with zero prior test coverage, found while scoping Task
  8.1. Fixed, tested, screenshots regenerated and reviewed. See changed-files.md.
- [x] Bundle F, Stage 2 architecture screening (E0-E8): all 9 configurations completed
  successfully. Ranking: E2 > E0 > E1 > E3 > E6 > E4 > E8 > E7 > E5 (see training-jobs.md for
  full scores). Top four scores span only 0.0052 -- noise-level at this budget.
- [ ] Bundle F, Stage 3 finalist training (E2/E0/E1, 2 seeds each) — **running now** in the
  background; see "Every training job's status" below for exact resume/monitoring commands.
- [ ] M training/evaluation
- [ ] calibration/OOD evaluation
- [ ] full-trajectory benchmark
- [ ] final selection
- [ ] locked final test

## Tests (current, after Task 3.2 + Cycle B landing)

See `test-results.md` for full detail.

| Command | Result | Notes |
|---|---|---|
| `pytest -q` | 363 passed | 4 new Task 3.2 backend contract tests added on top of Bundle E's 359. One unrelated scientific test (`test_information_gain_is_nonnegative_within_tolerance`) intermittently fails when the full suite runs -- confirmed pre-existing and unrelated: it seeds via Python's `hash()` on a string, which is salted per-interpreter-invocation (not fixed via `PYTHONHASHSEED`), so its pass/fail is randomized run-to-run regardless of any code change. Always passes in isolation and passed on 2 of 3 full-suite reruns during this session. Not touched -- weakening its tolerance would risk exactly the kind of physics-boundary weakening this run is instructed not to do; the real fix (seed it deterministically) is a pre-existing test-infra issue outside Task 3.2's scope. |
| `ruff check` (touched files) | pass | |
| `pyright` (touched files) | pass, 0 errors | |
| `npm run lint` | pass | |
| `npx tsc --noEmit` / `npm run build` (`tsc -b && vite build`) | pass | note: `tsc -b` (build mode, includes tests/) caught a stale test fixture `--noEmit` alone did not, since its default project scope is narrower |
| `npm run test -- --run` (vitest) | pass (25 tests, up from 24) | |
| `npx playwright test` (e2e) | pass (10/10) | one 1920x1080 visual-regression test needed a retry (~0.01% pixel diff, chart-marker anti-aliasing) -- confirmed flaky/unrelated by rerunning it alone immediately after |
| `npm run format:check` (prettier) | 11 pre-existing failures untouched, 0 new | baseline already had 11 unformatted files before this session (confirmed via `git stash`); `src/api.ts` was the only file this session's changes newly affected, and it is now formatted correctly |
| HydroCore-S checkpoint load | pass | unchanged this session |

## Datasets generated

**Cycle A** (`data/learning-v2/cycle-a/`, ~2,550 scenarios): 1,750 train / 250 validation / 250
calibration / 300 development_holdout, split evenly across two genuinely different topologies
(golden reference: 4 junctions/1 reservoir/1 tank/one loop; branched-loop: 7 junctions/1
reservoir/no tank/a different loop, already committed at
`data/topology-transfer/branched-loop.inp`). Generated in ~3 minutes. Manifests, signatures,
`dataset-report.json`, and `label-audit.json` are committed (~4.5MB); raw scenario arrays and
tensor shards (~55MB, regenerable) are gitignored. See `data/learning-v2/cycle-a/dataset-report.json`
for full counts/balance/leakage detail.

**Cycle B** (`data/learning-v2/cycle-b/`, 12,750 scenarios, generated in ~15.5 minutes via
`scripts/generate_cycle_b_corpus.py`): 9,000 train / 1,000 validation / 1,000 calibration / 1,750
development_holdout across 3 training topologies (golden-reference, branched-loop, loop-grid --
loop-grid newly authored for this cycle) plus 400 examples each for the `UNSEEN_TOPOLOGY` and
`SEVERE_MISSINGNESS` OOD-holdout categories against a 4th development-OOD topology
(coastal-branch). Label audit is clean across every split (0 duplicate scenario IDs, 0 finite-value
violations, 0 impossible labels; a handful of near-duplicate groups flagged but not treated as
errors, consistent with Cycle A). Documented, non-blocking limitations (full detail in
`data/learning-v2/cycle-b/dataset-report.json`'s `limitations` list): split counts sit toward the
low-to-mid end of the plan's stated ranges rather than the maximum; only 2 of ~10 governed
`OODCategory` values were generated this pass; no `ood_class` per-example target exists yet
(pre-existing generator gap, same category as bugs Bundle E found); the plan's required
hard-negative list (hydraulically similar sources, graph symmetries, etc.) was not attempted this
pass. Manifests/signatures/reports (~200KB) are committed; raw scenario arrays and tensor shards
(~200MB, regenerable) are gitignored.

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

## Task 3.2 (completed this session)

`GET /api/incidents/{incident_id}/view` now exists end to end:

- **Backend**: `IncidentView` Pydantic schema (`hydroswarm/api/state.py`) with every field the
  plan lists -- incident/network IDs, runtime mode, data mode, model/checkpoint version+hash,
  calibration version+hash, network hash, simulator+version, controller state, candidates with
  coverage target/calibration validity, full map topology (nodes/links, sourced only from an
  imported network's real metadata, never guessed), sensor health, sample recommendation,
  evidence-round history, plans with verifications, selected (derived from the audit ledger's
  `PLAN_APPROVED` event, not guessed) vs. recommended plan, per-plan counterfactual consequences,
  all seven grounded explanation intents, audit events, and stage-level runtime metrics. Fails
  closed: 409 if the incident hasn't completed a real hybrid analysis (DEMO_FALLBACK can never
  back this endpoint), 503 if the network lacks full topology metadata (the legacy manual
  `/validate` path). Every nested model is `extra="forbid"`.
- **Backend tests**: `tests/integration/test_incident_view_contract.py` drives a real
  `HybridInferencePipeline` (same fixture pattern as `tests/e2e/test_iterative_pipeline.py`)
  through the actual HTTP API -- incident creation, two real reanalysis rounds, plan
  generation/verification/approval -- then asserts every field of the response against the real
  provenance hashes, imported topology, and audit trail. Plus two fail-closed tests (409 before
  analysis, 503 for incomplete topology metadata).
- **Frontend**: `viewFromApi()` in `frontend/src/api.ts` replaces the old
  `LiveViewIncompleteError` stub, mapping the live response into the UI's `IncidentView` shape.
  `IncidentView`/`demoFixture.ts` extended with `provenance`, `selectedPlanId`,
  `recommendedPlanId`, `counterfactuals` to mirror the backend schema field-for-field.
  `PlanStatus` gained a `PENDING` state for not-yet-verified plans. Two fields are honestly left
  unfilled rather than fabricated (each with an inline comment): per-link flow (not yet threaded
  through `IncidentAnalysisResult`) and `exposureReduction` (no no-response baseline is computed
  server-side yet -- same pre-existing gap as `EvidenceBundle.exposure_reduction_mg`).
  `benchmarks` stays empty in LIVE mode -- confirmed via `ModelGovernanceTable` that it's a
  distinct, not-yet-live data source unrelated to this endpoint, not something this task covers.

Commits: `580185e` (backend), `ebd260c` (frontend), `bd944d5` (test-fixture fix caught by
`tsc -b`/`npm run build`, which checks `tests/` in a wider scope than `tsc --noEmit` alone).

## Remaining blockers

- None. The one intermittent test failure (`test_information_gain_is_nonnegative_within_tolerance`)
  is pre-existing, unrelated to this session's changes, and explained above under "Tests".
- Stage 2 architecture screening (E0-E8) is running now in the background and has not finished;
  see "Every training job's status" below for how to check on or resume it.

## Exact commands to continue

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q   # expect 363 passed (occasionally 362/1 -- see Tests section)
cd frontend && npm run test -- --run   # expect 30 passed (up from 25: +5 Counterfactuals.test.tsx)
npx playwright test                     # expect 10 passed (retry 1920x1080 visual-regression if it flakes)
```

See `follow-up.md` and `training-jobs.md` for the active Stage 3 finalist-training job's exact
monitoring/resume commands.

## Every training job's status

Bundle E's 6 smoke-screening jobs (E0/E3/E4/E9-none/E9-feature_only/E9-logit_only) all
completed successfully; see `training-jobs.md` and
`reports/results/v3/architecture-smoke-jobs.json`.

**Bundle F Stage 2 architecture screening (E0-E8) completed successfully** -- all 9
configurations, zero failures, ranking E2 > E0 > E1 > E3 > E6 > E4 > E8 > E7 > E5. See
`training-jobs.md` and `reports/results/v3/stage2-architecture-screening.json`.

**Bundle F Stage 3 finalist training is running now** under `hydroswarm.training.job_runner`,
training the top three Stage 2 finalists (E2, E0, E1) with two seeds each. Run directory:
`experiments/jobs/bundle-f-stage3/` (status.json/job.log/job.pid). See `training-jobs.md` for
the exact monitor/resume commands and current progress at handoff time.

## Locked final test

Not opened. Will only be opened once `reports/results/v3/final-selection.json` exists and
every Stage 6 condition in the plan is satisfied.

---

This file, and the rest of `reports/overnight/2026-08-03/`, will be kept current after every
independently validated milestone for the remainder of the run.
