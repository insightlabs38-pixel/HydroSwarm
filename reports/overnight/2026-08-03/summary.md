# Overnight HydroSwarm Handoff

Plan: `overnight-plan.txt` (2026-08-03 v3), multi-topology GCP execution and training run.

## Repository state

- Branch: `agent/gcp-multitopology-v3`
- Starting commit: `5697f912667fa236ece784a98f141c8162ff6bf8` (main, "Complete HydroCore-M
  evaluation and topology transfer study")
- Ending commit: `541c72c` (Bundle B schema/infra scope complete; still running)
- Commits created: 22 (a3ccd25, 19468ac, f67b0a6, ace2808, 9cf1d98, ecd34ac, ad8b256, b2bba5d,
  1c3b55b, 7d5d84a, 22bae5d, 60949b1, 80c1aeb, 14e4224, dfa5b60, 7bdf4f5, 17643ab, 9dd7942,
  ca955a8, 541c72c, and 2 report-refresh commits)
- Working tree: clean at branch creation

## Completed tasks

- [x] Phase 0 — baseline verification and branch creation (Task 0.1)
- [x] Phase 0.2-0.8 (Bundle A: registry, job runner, sharded lazy data, label audit, normalization, sampler, split policy)
- [x] Phase 1.1-1.5 (Bundle B: variable-topology corpus architecture — topology metadata, per-topology
  signature registry, variable-size collation, scenario-specific hydraulic context caching, permutation
  equivariance). This was the plan's explicitly highest-priority technical change.
- [x] Phase 2.1, 2.5, 2.6 (Bundle B: targets_v2 contract, OOD categories, trajectory serialization —
  schema/governance/infrastructure scope)
- [ ] Phase 2.2-2.4 (Bundle B: actual Sentinel/Scout/Strategist label *generation* against the real
  simulator/classical/planning stack) — next up, see Follow-up
- [ ] Phase 3 (Bundle C: frontend live/demo integrity)
- [ ] Phase 4 (Bundle D: configurable HydroCore)
- [ ] Cycle A
- [ ] Cycle B
- [ ] S architecture screening
- [ ] S finalist training
- [ ] M training/evaluation
- [ ] calibration/OOD evaluation
- [ ] full-trajectory benchmark
- [ ] final selection
- [ ] locked final test

## Tests (current, after Bundle A)

See `test-results.md` for full detail.

| Command | Result | Notes |
|---|---|---|
| `pytest -q` | 242 passed | started at 1 failed/97 passed; pre-existing failure fixed in `19468ac`; 144 new tests added across Bundle A + Bundle B schema/infra scope |
| `ruff check src tests scripts` | pass | |
| `pyright` | pass | |
| `npm run lint` | pass | |
| `npm run test -- --run` | pass (4 tests) | |
| `npm run build` | pass | |
| HydroCore-S checkpoint load | pass | hash, feature-schema, and calibration all validate |

## Datasets generated

None yet (Bundle A is data-infrastructure; Cycle A/B/C generation is Bundle E/F/Phase 5).

## Experiments completed

None yet (no training runs launched; Bundle A built the registry/job-runner/sampler
infrastructure those runs will use).

## Metrics obtained

None yet (baseline HydroCore-S hybrid governed result remains the current reference:
96.0% top-1, ECE 0.0269, 8.94ms latency, per the plan's existing governed results table).

## Important findings

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

## Remaining blockers

- None currently. The one pre-existing test failure was fixed (see below).

## Exact commands to continue

```bash
cd /workspace/HydroSwarm
git -c safe.directory=/workspace/HydroSwarm checkout agent/gcp-multitopology-v3
export PYTHONPATH=src
python -m pytest -q   # expect 242 passed
```

See `follow-up.md` for the specific next task (2.2: Sentinel label generation) and what
investigation it needs before implementation.

## Every training job's status

None launched yet; see `training-jobs.md`.

## Locked final test

Not opened. Will only be opened once `reports/results/v3/final-selection.json` exists and
every Stage 6 condition in the plan is satisfied.

---

This file, and the rest of `reports/overnight/2026-08-03/`, will be kept current after every
independently validated milestone for the remainder of the run.
