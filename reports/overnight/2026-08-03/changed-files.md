# Changed files

## Phase 0 (baseline)

No files under `src/`, `tests/`, `scripts/`, `frontend/src/`, `configs/`, `data/`, or `models/`
were changed. Only new files were added, all under `reports/overnight/2026-08-03/`:

- `environment.json`
- `resource-usage.json`
- `test-results.md`
- `completed-tasks.json`
- `failed-tasks.json`
- `changed-files.md` (this file)
- `experiment-plan.md`
- `experiment-registry.json`
- `training-jobs.md`
- `artifact-inventory.json`
- `follow-up.md`
- `summary.md`

## Bundle A (Tasks 0.2-0.8), commits f67b0a6..1c3b55b

New source modules under `src/hydroswarm/training/`: `registry.py`, `job_runner.py`,
`sharded_data.py`, `label_audit.py`, `sampler.py`, `split_policy.py`; a small addition to
`data.py` (`load_scenario_examples_jsonl`); a small SIGTERM-handling addition to
`trainer.py`. New module `src/hydroswarm/preprocessing/schema.py` additions
(`NormalizationStats.save/load/fingerprint`, `NODE_FEATURE_SEMANTICS`/
`EDGE_FEATURE_SEMANTICS`). New scripts: `scripts/audit_labels.py`,
`scripts/fit_normalization.py`. New config: `configs/evaluation_policy_v3.json`. New doc:
`docs/EVALUATION_V3_POLICY.md`. New governed artifacts: `reports/results/v3/label-audit-learning-v1.json`,
`reports/results/v3/normalization/{node,edge}-normalization.json(.sha256)`. Plus one
pre-existing-bug fix: `data/frozen/manifest.json` (commit `19468ac`). 65 new tests across
7 new test files plus additions to `tests/unit/test_preprocessing.py` and
`tests/scientific/test_training_smoke.py`. No file under `data/learning-v1/`, `models/`, or
any historical `reports/results/*.json` (pre-existing, non-v3) was modified.

## Tasks 1.1-1.5 (variable-topology corpus architecture), commits 22bae5d..7bdf4f5

`src/hydroswarm/training/data.py` gained `TopologyMetadata`, `manifest_entry()`,
`resolve_source_node_id()` and a `ScenarioExample.topology` field (optional, backward
compatible). New modules: `training/variable_collate.py`, `training/permutation.py`,
`classical/signature_registry.py`, `simulation/context_cache.py`. Small additions to
`preprocessing/batching.py` (`node_scalar_features` on `GraphSample`,
`NODE_SCALAR_FEATURE_KEYS`). `sharded_data.py`'s `_IndexEntry` extended to carry topology
metadata through sharded storage. One correctness fix in `data.py`
(`source_node_id_for_local_index`). 63 new tests across 5 new test files plus additions to
`test_sharded_data.py` and `test_topology_metadata.py`. No file under `data/learning-v1/`,
`models/`, or any historical `reports/results/*.json` was modified.

## Tasks 2.1, 2.5, 2.6 (targets_v2 schema/governance scope), commits 9dd7942..541c72c

New modules: `training/targets_v2.py` (TARGETS_V2 contract, EventCause/NextStep enums),
`training/ood_categories.py` (OODCategory taxonomy, expected behavior, abstention
classification), `training/trajectory_v2.py` (TrajectoryState/FullTrajectory, reusing
`domain.IncidentState`/`OperationalAction`/`PlanVerification`). 33 new tests across 3 new
test files. No file under `data/learning-v1/`, `models/`, or any historical
`reports/results/*.json` was modified. Tasks 2.2-2.4 (the actual label *generation* against
the simulator/classical/planning stack, as opposed to the schema work here) are not started.

## Tasks 2.2-2.4 (Sentinel/Scout/Strategist label generation, closing Bundle B), commits dff8122..63e060b

`src/hydroswarm/data/scenarios.py` gained `EventType` and generator support for
normal/sensor-fault-only scenarios (negligible-strength injection, forced fault
injection). `src/hydroswarm/training/corpus.py` gained `assign_source_regions()` and
Sentinel target derivation (event_presence/event_cause/evidence_sufficiency/masking) in
`scenario_to_example()`. New modules: `training/scout_labels.py`,
`training/strategist_labels.py`. 28 new tests across 3 new test files plus additions to
`test_scenario_generation.py`, all exercised against the real reference network with real
WNTR simulation. No file under `data/learning-v1/`, `models/`, or any historical
`reports/results/*.json` was modified. Bundle B (Tasks 1.1-1.5, 2.1-2.6) is now complete.

This section will be appended to (not rewritten) as later bundles land, grouped by commit.
