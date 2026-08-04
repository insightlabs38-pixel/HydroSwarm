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

## Bundle C (frontend live/demo data integrity), commits eb066d7..2067571

`frontend/src/api.ts` (rewritten: RuntimeMode-aware fetch, LiveViewIncompleteError,
IncidentUnavailableError, failure injection), `frontend/src/types.ts` (RuntimeMode,
IncidentView.mode/modeReason/calibrationValid/measuredCoverage), `frontend/src/App.tsx`
(mode badge, ERROR/REPLAY banners, Retry action), `frontend/src/demoFixture.ts` (mode
fields, corrected benchmark row), `frontend/src/pages/Overview.tsx`,
`frontend/src/components/{EvidencePanel,OperationalMap}.tsx`,
`frontend/src/pages/TopologyPage.tsx` (dynamic identifiers, disabled unwired controls),
`frontend/src/pages/{ValidationPage,BenchmarkPage}.tsx` (governance table, corrected
claims/limitations), `frontend/src/pages/AuditPage.tsx` (disabled unwired controls, honest
chain-status text). New: `frontend/src/components/ModelGovernanceTable.tsx`,
`frontend/public/model-governance.json`. New tests: `frontend/tests/api.test.ts`,
`frontend/tests/IdentifierIndependence.test.tsx`, `frontend/tests/ModelGovernanceTable.test.tsx`,
`frontend/tests/e2e/visual-regression.spec.ts` (+ 2 committed screenshot baselines). 45 new
frontend tests (24 unit + 10 e2e, up from 4 unit + 1 e2e). No backend Python file changed in
this bundle.

## Bundle D (configurable HydroCore architecture, Tasks 4.0-4.6), commits 7a5492d..d0a0b42

`src/hydroswarm/model/core.py`: `ARCHITECTURE_VERSION`, `architecture_config()`,
`verify_architecture_compatibility()` (4.0); `prior_mode` (4.1); `incident_pooling` +
`_attention_pool()` (4.2); `message_direction` threaded into the backbone (4.3);
`event_control_heads` gating `event_presence_head`/`event_cause_head`/`next_step_head`
(4.4); `auxiliary_heads` gating `sensor_reconstruction_head`/`future_concentration_head`/
`travel_time_head` (4.5); `consequence_prescreening_heads` gating
`consequence_proxy_heads` (5 proxies) (4.6). `HydroOutput` TypedDict grew a matching key
per new output. `parameter_report()` updated to account for every conditionally-constructed
head. `src/hydroswarm/model/layers.py`: `EdgeAwareGraphConv._aggregate(reverse=...)`
refactor, new `DualChannelGraphConv` (4.3). `src/hydroswarm/model/__init__.py`: exports for
`MessageDirection`/`MESSAGE_DIRECTIONS`/`DualChannelGraphConv`.
`src/hydroswarm/training/targets_v2.py`: registered `sensor_reconstruction`/
`future_concentration`/`travel_time` under a new `"auxiliary"` category (4.5), and
`exposure_proxy`/`pressure_risk_proxy`/`service_loss_proxy`/`containment_time_proxy`/
`plan_regret_proxy` under `"strategist"` (4.6). `src/hydroswarm/training/losses.py`:
wired `event_cause`/`event_presence` (4.4), renamed the pre-existing unused
`reconstruction`/`reconstruction_prediction` placeholder to
`sensor_reconstruction`/`sensor_reconstruction_prediction` and confirmed
`future_concentration`/`travel_time` already matched (4.5), added the five `*_proxy`
regressions (4.6); added `AUXILIARY_TASKS`/`AUXILIARY_TASK_DEFAULT_WEIGHT` and the reduced
default-weight logic in `compute_multitask_loss()` (4.5). New test files:
`tests/unit/test_prior_mode.py` (11), `tests/unit/test_incident_pooling.py` (17),
`tests/unit/test_message_direction.py` (15), `tests/unit/test_event_control_heads.py` (12),
`tests/unit/test_auxiliary_heads.py` (12), `tests/unit/test_consequence_prescreening.py`
(11); small additions to `tests/unit/test_targets_v2.py` and `tests/unit/test_training.py`
for the new governed categories/tasks. 79 new/changed tests total. No file under
`data/learning-v1/`, `models/`, or any historical `reports/results/*.json` was modified; the
promoted checkpoint's loadability under `DefaultPipelineFactory` was re-verified after every
single commit in this bundle, not just once at the end. No frontend file changed in this
bundle.

## Bundle E (Cycle A corpus + architecture smoke jobs), commits `4d75a6b..0606586`

`src/hydroswarm/data/scenarios.py`: added `DatasetSplit.DEVELOPMENT_HOLDOUT`.
`src/hydroswarm/training/data.py`: `ScenarioExample.split` validator accepts it.
`scripts/generate_cycle_a_corpus.py` (new): generates the 2,550-scenario, 2-topology Cycle A
corpus into `data/learning-v2/cycle-a/` (manifests/signatures/reports committed, ~4.5MB; raw
scenario/tensor binaries gitignored, ~55MB). `src/hydroswarm/training/label_audit.py`: fixed
`_sensor_fault_prevalence` to handle mixed node counts across topologies (real bug, found by
actually running Cycle A generation). `src/hydroswarm/training/trainer.py`: `Trainer` accepts
an optional `collate_fn` (default unchanged), needed for `collate_variable_topology` against
genuinely multi-topology batches. `src/hydroswarm/training/losses.py`: `compute_multitask_loss`
now applies targets_v2's `f"{task}_mask"` companions via a new `_apply_target_mask` helper (real
bug: masked placeholder labels were being silently trained against), and wires `source_region`
to a loss for the first time (a real head/target pair that had never been connected).
`src/hydroswarm/model/core.py`: squeezed `evidence_sufficiency`'s output shape to match its real
scalar-per-example target (real bug: `[batch, 1]` vs. `[batch]` crashed the first real
model-output-into-loss run). `scripts/run_architecture_smoke_jobs.py` (new): Stage 1
smoke/failure screening for E0/E3/E4/E9 -- 6 real `Trainer.fit()` + resume + checkpoint-reload
runs, all passed. New/changed test files: `tests/scientific/test_scenario_generation.py`,
`tests/unit/test_label_audit.py`, `tests/scientific/test_training_smoke.py`,
`tests/unit/test_training.py`. `experiments/registry/bundle-e-smoke.jsonl` (real provenance
ledger) and `reports/results/v3/architecture-smoke-jobs.json` committed;
`experiments/runs/bundle-e-smoke/` (~1.3GB checkpoints) gitignored. No file under
`data/learning-v1/`, `models/`, or any historical `reports/results/*.json` was modified; the
promoted checkpoint's loadability was re-verified after every commit in this bundle. No frontend
file changed in this bundle.

This section will be appended to (not rewritten) as later bundles land, grouped by commit.
