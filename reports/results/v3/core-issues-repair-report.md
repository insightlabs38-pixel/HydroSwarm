# core-issues.txt repair pass — completion report (Phase 1 + Phase 2)

Branch: `agent/gcp-multitopology-v3`. All work below is committed and pushed; the
working tree is clean. **Phase 3 (rebuild and retrain) has not started. The locked
final test set has not been opened.** Per the user's instruction, this pass now
pauses for approval before any further work.

## Commits made (12, one per repair item, each with its own tests)

| Item | Commit | Summary |
|---|---|---|
| 1 | `b1977ba` | Preserve real observation masks through variable-topology collation |
| 2 | `9d281d2` | Make `source_region_logits` a real 3-way incident classification (+ checkpoint migration path) |
| 3 | `e207882` | Mask `sensor_fault` to real sensors; cover the full fault taxonomy (frozen/comms/drift/unit-mismatch) |
| 4 | `5af3839` | Build each scenario's hydraulic features from its own randomized network |
| 5 | `3d2dcec` | Populate real `TopologyMetadata` on every generated example |
| 6 | `0fffa25` | Thread governed normalization identity everywhere; fixed a real v2-checkpoint regression found along the way |
| 7 | `2924300` | Correct target audits to respect every `*_mask` companion |
| 8 | `c16895b` | Gate untrained Scout/Strategist/OOD heads out of runtime decisions |
| 9 | `b30850d` | Standardize checkpoint identity; runtime instantiates the model from checkpoint metadata, not hardcoded defaults |
| 10 | `42a0582` | Fit conformal calibration on the fused hybrid predictor, not raw neural probabilities |
| 11 | `0b7b8dd` | Fix training artifact records (selected checkpoint, export hash) and sparse GradNorm logging |
| 12 | `27c5152` | Keep lazy loading lazy (`ShardedScenarioDataset` used directly); verify shard checksums before training |

Full test suite after item 12: **415 passed**, 0 failed (started this pass at 387).
`ruff check` and `pyright` are clean on every file touched across all 12 items.

## Defects fixed, by item

1. **Mask preservation**: `pad_graph_batch`/`GraphSample` were re-deriving
   `sensor_mask`/`quality_mask`/`node_mask`/`edge_mask` from already-`nan_to_num`'d
   tensors via `torch.isfinite`, silently marking every padded/imputed position as
   valid. Real per-example masks now flow `HydraulicFeatureBuilder → ScenarioExample →
   shard → collator` unchanged.
2. **Source-region head**: was `RoleHead(d_model, 1)` applied per node position
   (never a real 3-way classification — effectively an accidental node index). Now
   `RoleHead(d_model, 3)` applied to `incident_context`. `ARCHITECTURE_VERSION` bumped
   to `hydrocore-v3`; a narrow, fail-closed migration path
   (`load_state_dict_with_v2_migration`) loads the one now-incompatible head fresh
   while re-raising on any other mismatch. **This is the regression that broke the
   real promoted checkpoint's load** (`trained_assets_ready` went `False` mid-repair);
   caught by end-to-end verification against the real checkpoint, not just synthetic
   tests, and fixed within the same item.
3. **Sensor-fault supervision**: unsensored/padded nodes' `sensor_fault=0.0`
   placeholder was trained against and counted as a real "healthy" observation.
   Added `sensor_fault_mask` (true only for real sensor nodes); BCE and audits now
   exclude masked positions. The label itself was also incomplete — only
   frozen/comms-outage faults were ever set despite the target's own definition
   naming drift and unit-mismatch too; both are now included.
4. **Hydraulic context reuse**: every scenario generated from one topology shared a
   single static feature context built once, discarding each scenario's own
   randomized demand/roughness/tank/pipe-outage state.
   `WNTRScenarioGenerator.generate_with_network()` now returns the exact randomized
   model alongside the scenario; corpus generation builds a fresh feature context
   per scenario from it.
5. **Topology metadata**: `TopologyMetadata` was never populated (`topology=None`
   on every example). Every example now carries a real topology hash, network hash,
   node/edge/candidate IDs, hydraulic-state hash, signature-library hash, and schema
   versions.
6. **Normalization**: no governed normalization artifact has ever existed or been
   fit; the gap wasn't tracked identity, so a real train/runtime mismatch could not
   have been detected. Added `NormalizationStats` fingerprinting
   (`HydraulicFeatureBuilder.normalization_fingerprint`, defaulting to an explicit
   `NO_NORMALIZATION_SENTINEL`), threaded through `BuiltHydroBatch`,
   `CalibrationArtifact`, and runtime provenance, with fail-closed
   `validate_runtime` checks.
7. **Label audits**: histograms, source balance, baselines, and sensor-fault
   prevalence all counted masked placeholder values as real labels; a masked
   `source_node` pointing at a masked candidate was flagged as an impossible label
   when it was actually a well-formed placeholder;
   `cross_split_leakage`'s topology-hash check was stale text claiming topology
   hashes didn't exist. All now honor `*_mask` companions and use real topology
   hashes (item 5).
8. **Untrained heads**: Scout (`sample_node`/`information_gain`), Strategist
   (`plan_value`/`plan_validity`), and the learned OOD head have never received a
   real training label (corpus generation only ever writes the 9 `sentinel`-category
   targets) — yet their random-initialization outputs were still feeding real
   runtime decisions (active-sampling ranking, plan-template ranking, OOD energy).
   `HybridInferencePipeline` now gates these out per a `trained_tasks` declaration;
   deterministic sampling/planning/OOD logic remain authoritative. Checkpoint
   metadata now declares `trained_tasks`/`validated_tasks` (`["sentinel"]` for every
   checkpoint promoted so far — accurate for all three).
9. **Checkpoint identity**: the runtime always built `HydroCore.from_variant("small")`
   with every other constructor argument left at its hardcoded default, regardless
   of what a checkpoint actually declared — silently ignoring a comprehensive
   compatibility checker (`verify_architecture_compatibility`) that already existed
   but was never wired into the runtime load path. The runtime now instantiates the
   model from the checkpoint's own `architecture_config` (variant, `use_adapters`,
   `prior_mode`, pooling, message direction, optional heads) and calls the checker as
   defense-in-depth; a checkpoint missing this declaration now fails closed instead
   of silently loading into a possibly-mismatched model. `promote_checkpoint.py` now
   requires these flags explicitly and verifies the checkpoint actually loads with
   them before promoting.
10. **Calibration fit on the wrong quantity**: every calibration-fitting code path
    (`run_stage3_finalist_training.py`'s `_predict_rows`, reused by Stage 4;
    `evaluate_learning.py`) fit `SplitConformalCalibrator` on raw
    `softmax(source_node_logits)`, never touching the classical posterior the
    deployed pipeline fuses in at runtime. Added `fixed_weight_fusion` (the same
    approximation `evaluate_medium.py`'s locked-test comparison already used
    successfully for this purpose) and fit/evaluate on the fused vector everywhere.
    `CalibrationArtifact` gained `fusion_config_hash` and `validated_topology_hashes`,
    checked fail-closed at runtime (backward compatible: both default to "unset",
    skipping the check, so no existing artifact/test is spuriously invalidated).
    **Real, verified consequence**: the currently-promoted checkpoint's own
    `calibration.json` was fit before this fix existed, with no fusion step at all —
    live analysis against it now honestly reports `calibrated: False` (see
    Limitations).
11. **Training artifact records**: the real Stage 3 registry ledger
    (`experiments/registry/bundle-f-stage3.jsonl`) shows every finalist run closing
    with `exit_status: "success"` and `selected_checkpoint: ""` — every run hit the
    2-hour runtime ceiling, and the scripts were passing `summary.final_checkpoint`
    (empty by design when a run is cut short) as the selected checkpoint instead of
    `summary.export_path` (unconditionally populated). Fixed in all three stage
    scripts; `TrainingSummary` gained `export_sha256` and `last_resumable_checkpoint`
    so a cut-short run never loses its periodic checkpoint reference. GradNorm
    diagnostics (7-9 extra backward passes per batch by default) now run every 25th
    batch in all three stage scripts instead of every batch.
12. **Lazy loading**: `_load_dataset`/`_load_ood_dataset` opened a lazy
    `ShardedScenarioDataset` and immediately materialized every example into a
    resident Python list purely to satisfy a `GovernedScenarioDataset` type hint.
    Added a structural `ScenarioDatasetView` Protocol so `Trainer` and the stage
    scripts accept either type; the loaders now return the lazy dataset directly, and
    prediction loops materialize one batch at a time. `ShardedScenarioDataset`
    gained `verify_shard_checksums()` (manifest already recorded real per-shard
    sha256; nothing ever checked it back) — called once, explicitly, before
    training, not folded into construction (which must stay metadata-only per an
    existing, deliberately-tested property).

## Corpus, topology, and normalization identity

No corpus was regenerated in this pass (Phase 3 is out of scope). Cycle A/B on disk
are unchanged from before this pass. Governed normalization has never been fit
(`scripts/fit_normalization.py` has never been run against real data) — every
normalization identity in this pass is the explicit `NO_NORMALIZATION_SENTINEL`
(`f9022e6322dfa58fdc8434a78c2c624b68acb4041e359684665eb444f5912cf7`), consistently
on both the runtime feature builder and the promoted calibration artifact.

## Checkpoint and calibration identities (the 3 promoted checkpoints)

`models/hydrocore-s-learning-v1.safetensors` (the currently deployed checkpoint),
`models/hydrocore-m-learning-v1-partial.safetensors`, and
`models/hydromono-s-learning-v1.safetensors` were **not retrained or re-promoted** —
their weights and `sha256` are byte-identical to before this pass. Their
`.metadata.json` sidecars were patched (JSON-only, no weight bytes touched) to add:

- `architecture_config` (reconstructed by actually loading each real checkpoint into
  the corresponding `HydroCore` configuration and confirming it loads via
  `load_state_dict_with_v2_migration` — verified, not guessed)
- `normalization_hash`: `NO_NORMALIZATION_SENTINEL` for all three (accurate — no
  normalization artifact has ever existed)
- `trained_tasks` / `validated_tasks`: `["sentinel"]` for all three (accurate — the
  corpus has never generated Scout/Strategist/OOD-class labels)
- `training_provenance`: `null` for all three — see Limitations; not fabricated.

`hydrocore-s-learning-v1`'s checkpoint hash: `85715fbe061a30131b39b717137d2522c3870d674d262f4717ef7541731d5423`.
Its architecture config: `{variant: small, use_adapters: true, prior_mode:
feature_and_logit, incident_pooling: mean, message_direction: forward_only,
event_control_heads: false, auxiliary_heads: false, consequence_prescreening_heads:
false}` — this is `E0` (the baseline), matching
`finalist-selection-recommendation.md`'s own documentation of which architecture the
promoted checkpoint uses.

Its calibration artifact (`reports/results/hydrocore-s-calibration.json`) was **not
refit** in this pass (refitting on the exact hybrid predictor with real trust
features requires regenerated data with preserved live estimator state — Phase 3
item 18). Verified via `scripts/validate_trained_pipeline.py` against the real
golden-network fixture: `trained_assets_ready: true`, `calibrated: false` (was `true`
before this pass — see Limitations for why this is the correct, intended
consequence, not a regression to hide).

## Comparison with preserved provisional Stage 3/4 results

`reports/results/v3/{stage2-architecture-screening,stage3-finalist-training,
stage4-controls-training}.json` and `finalist-selection-recommendation.md` are
**untouched** — no numbers in them were regenerated or edited. They were already
self-flagged as provisional pending this exact repair pass
("This entire Stage 3/4 result set ... is now flagged provisional, pending a repair
pass ... that fixes confirmed defects in the training/data pipeline"). This pass
confirms every defect that document listed was real (variable-topology masks,
source-region head shape, sensor-fault mask, per-scenario hydraulic context reuse,
topology metadata, calibration fit against fused vs. neural-only output) and fixes
all of them at the code level. None of their reported val/dev-holdout/OOD accuracy
numbers should be treated as final; they were produced by the pre-repair pipeline
and Phase 3 (items 13-18) is what regenerates comparable, correct numbers.

## Remaining known limitations

- **Scout, Strategist, and learned-OOD target generators do not exist.** Per
  core-issues.txt's explicit scope, this pass gates their outputs out of runtime
  decisions (item 8) rather than implementing them.
- **`calibrated: False` for the live, currently-promoted checkpoint.** Its
  `calibration.json` was fit pre-repair with no fusion step at all — item 10's fix
  correctly, honestly reports this rather than hiding it. This is graceful
  degradation (the pipeline still runs `FULL_HYBRID`, still falls back to a
  credible-node heuristic instead of split-conformal candidate sets), not a crash,
  and is resolved once Phase 3 refits calibration on the exact hybrid pipeline.
- **`fixed_weight_fusion` (used to fit calibration in this pass's fixed scripts) is
  an approximation**, not the exact deployed `fuse_source_probabilities` dynamic-
  trust fusion — the latter needs live `HydraulicStateEstimator` residual/
  uncertainty state that governed `ScenarioExample` tensors do not preserve
  post-hoc. `fusion_config_hash` makes this gap visible and enforced (a calibration
  fit with the approximation will not silently validate against the live dynamic
  fusion) rather than papering over it.
- **`training_provenance` (seed/git_commit/manifest_hashes) is `null`** on all 3
  promoted checkpoints. The current `experiments/registry/*.jsonl` ledgers do not
  contain a run whose `checkpoint_hashes` match these checkpoints' real sha256 —
  this pass does not fabricate that provenance. `promote_checkpoint.py` now supports
  `--registry-path`/`--registry-run-id` to link it for any future promotion.
- **The 7 "additional implemented runtime fixes"** (exact-simulation budget per
  incident, killable WNTR-verification subprocess, analysis persistence across
  restart, null-vs-zero frontend values, ERROR-mode rendering guard, `/view`
  response validation, fail-closed provenance hashes) are **not done** in this pass
  — explicitly deferred pending your review of the above, per "pause, don't
  continue without my approval."
- One test (`tests/scientific/test_scout_labels.py::
  test_information_gain_is_nonnegative_within_tolerance`) failed once under a
  full-suite run and passed cleanly in isolation and on a full-suite rerun —
  pre-existing flakiness (consistent with previously documented occasional WNTR
  numerical instability), not caused by this pass's changes.

## Exact commands to resume or reproduce

```bash
# Full test suite (what every commit in this pass was gated on)
export PYTHONPATH=src && python -m pytest -q

# Lint/typecheck (run against the specific files touched per item; whole tree also works)
ruff check src/ scripts/ tests/
pyright src/ scripts/ tests/

# Verify the real promoted checkpoint against this pass's fixes
export PYTHONPATH=src && python scripts/validate_trained_pipeline.py

# Review this pass's 12 commits
git log --oneline b1977ba~1..27c5152
```

Phase 3 (regenerate Cycle B, run corpus gates, screen E0/E1/E2, select S, fully
train, fit calibration on the exact hybrid pipeline, re-run the HydroMono/no-adapter
control, decide on HydroCore-M) has explicit prerequisites and gates spelled out in
core-issues.txt items 13-20 and is not started. **The locked final test set has not
been opened.**
