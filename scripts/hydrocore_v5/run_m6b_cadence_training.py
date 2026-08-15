"""Milestone 6B (experiments.txt): does cadence-diverse TRAINING fix the
Milestone-6 post-onset report-count invariance, before touching architecture?

M6's own corrected evidence (reports/evaluation/hydrocore-v5/m6-cadence.json):
matched-physical-time comparisons showed ~0.00pp spread (no material cadence
sensitivity), but the post-onset-anchored fixed-report-count diagnostic
(the AUTHORITATIVE M6 signal, `post_onset_paired_by_n`) showed near-total
predicted-node INVARIANCE across 15/30/60-minute cadences at a fixed report
COUNT: N=2 -> 100% identical, N=3 -> 93.8% identical. HydroCore already has
an elapsed-time-aware `TemporalEncoder` (M6's own architecture note). Every
Milestone-1-6 training run, however, only ever trained on ONE fixed
(implicit hourly) reporting cadence -- `build_wntr_network()`'s
`report_timestep=3600s`, confirmed this session (25 timestamps, 3600s
apart) -- so this milestone tests whether the invariance is a
TRAINING-DISTRIBUTION artifact (the model was never shown what a report
actually means at a non-hourly interval) rather than a capacity/
architecture problem, before any capacity/architecture work is considered.

Two arms, matched-size (~4.18M) HydroCore, IDENTICAL architecture/task
weights/optimizer/scenario family (golden-reference only)/total budget
(600 train / 100 validation / 150 calibration scenarios, 20 epochs,
batch_size=2, gradient_accumulation_steps=4 -- all from the frozen
configs/training-v5-causal.yaml, unchanged):

  A -- FIXED_CADENCE_CONTROL. Reproduces the Milestone-1 winning recipe
  (arm A, full-history) as faithfully as possible: rather than retraining a
  similar-but-not-identical copy, this REUSES the exact frozen Milestone-1
  arm-A checkpoints (both seeds, still present on disk this session) --
  byte-identical to what M3-M7 already froze, matching Milestone-7's own
  "do not retrain what is already exactly reproducible" precedent. Milestone
  7's EXPANDED topology checkpoint is explicitly NOT used as the control
  (per this milestone's own instruction) -- M7 found no generalizing benefit
  from topology diversity, so the pre-M7 Milestone-1 recipe remains the
  correct baseline for a training-DISTRIBUTION question orthogonal to
  topology.

  B -- CADENCE_DIVERSE. Same architecture/config/seed/total scenario count/
  epoch budget, but each TRAINING batch's evidence is drawn from ONE of
  three real reporting cadences (15/30/60 minutes; the pathological 300s/
  5-minute WNTR path M6 already discovered and rejected is never used),
  sampled uniformly per micro-batch (mirroring `CausalPrefixDatasetView`'s
  own established "one draw per BATCH, not per example" batching-mechanics
  pattern -- a batch mixing two report counts would have two incompatible
  time-axis lengths for `collate_scenarios` to stack). Depth policy is kept
  identical to arm A's own recipe (`full_history_policy`: always ALL
  available reports at whatever cadence was drawn for that batch) --
  "same causal-prefix formulation" per this milestone's own instruction;
  the only new axis of variation is cadence, not depth. Cadence
  diversification comes from REAL timestamped evidence: training scenarios
  are simulated once at 15-minute resolution (`build_high_resolution_network`,
  reused from `run_m6_temporal.py`, already verified there not to hit the
  5-minute pathological path) and 30/60-minute views are produced by
  SUBSAMPLING that one shared high-resolution trajectory (stride 2/4) --
  never by rewriting timestamp features after tensorization, and never by
  re-simulating (so "physically identical incidents at different cadences"
  is exact by construction, matching M6's own convention). Physical
  scenarios are split (via the existing, unmodified `SPLIT_SEED_RANGES`
  seed assignment in `build_scenario_pool`) BEFORE any cadence view is
  drawn, so no cadence view of one physical incident can ever cross a
  train/calibration/development split boundary -- cadence views are lazy,
  per-batch VIEWS over an already-split scenario, exactly like depth views
  already are.

  Validation during arm B's OWN training uses the STANDARD, UNCHANGED
  native-hourly golden-reference validation pool/signature library/
  `full_history_policy`/`CausalPrefixDatasetView` (all completely
  unmodified) -- exactly matching arm A's own validation criterion, so
  checkpoint selection is directly comparable between arms (mirrors
  Milestone 1's own "checkpoint-selection validation is always evaluated at
  one consistent criterion regardless of what the arm trains on").

Optional arm C (input-level timestamp-conditioning ablation) is SKIPPED --
see SKIPPED_ARM_C_REASON below for why a clean, non-invasive, parameter-
matched ablation is not available within this milestone's scope.

Evaluation directly REUSES `run_m6_temporal.py`'s own, already-validated
functions against each arm's model (never reimplemented): `run_cadence_experiment`
(covers BOTH this milestone's 3A matched-physical-time AND 3B post-onset
fixed-report-count diagnostics -- the same call produces
`cadence_sensitivity_at_matched_elapsed_time` for 3A and
`post_onset_paired_by_n`/`post_onset_n2_identical_fraction`/
`post_onset_n3_identical_fraction` for 3B, the milestone's own PRIMARY
endpoint) and `run_irregular_telemetry_experiment` (the important M6
irregular-telemetry stress diagnostics: TIMESTAMP_JITTER,
UNEQUAL_SENSOR_INTERVALS, DELAYED_REPORTS, GAPS, PARTIAL_HISTORICAL_AVAILABILITY),
both run against ONE SHARED incident pool so arm A and arm B are always
compared on identical incidents. Calibration reuses `run_m6_temporal.py`'s
own `_fit_frozen_calibrator` (same frozen B_DEPTH_AWARE scheme, alpha=0.1,
refit per-arm on that arm's own calibration pool -- never loaded from a
serialized artifact, matching every M3-M7 script). The standard EARLY/MID/
MATURE regime check (section 4) is new but small, reusing the frozen
`scenario_to_prefix_example`/`CAUSAL_PREFIX_DEPTHS`/`DEPTH_BUCKET_OF`/M6's
own `_row_metrics` unchanged.

No production code, K, alpha, or authority/safety threshold is touched.
Locked test/topology data is never opened (asserted before and after).

Writes:
  reports/evaluation/hydrocore-v5/m6b-temporal-training.json
  reports/evaluation/hydrocore-v5/m6b-summary.md
  reports/evaluation/hydrocore-v5/m6b-runs/CADENCE_DIVERSE-seed{seed}.json
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator, classify_runtime_condition  # noqa: E402
from hydroswarm.data.scenarios import EventType  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.preprocessing import HydraulicFeatureBuilder, SensorSeries  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    EVENT_CAUSE_INDEX,
    STAGE_MAP,
    CausalPrefixDatasetView,
    ScenarioRecord,
    TopologyMetadata,
    _event_cause,
    _evidence_sufficiency,
    _fault_any_within_depth,
    _hydraulic_state_hash,
    assign_source_regions,
    build_scenario_pool,
    fit_pool_signature_library,
    full_history_policy,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402
from hydroswarm.training.data import ScenarioExample  # noqa: E402
from hydroswarm.training.targets_v2 import TARGETS_V2_SCHEMA_VERSION  # noqa: E402
from hydroswarm.training.trainer import Trainer, set_deterministic_seed  # noqa: E402

from run_m1_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m3_calibration import DEPTH_BUCKET_OF  # noqa: E402
from run_m6_temporal import (  # noqa: E402
    _fit_frozen_calibrator,
    _generate_cadence_incident_pool,
    _row_metrics,
    _slice_series,
    build_high_resolution_network,
    run_cadence_experiment,
    run_irregular_telemetry_experiment,
)

OUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m6b-temporal-training.json"
OUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m6b-summary.md"
RUNS_ROOT = ROOT / "experiments" / "runs" / "hydrocore-v5-causal"
M6B_RUNS_SUMMARY_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m6b-runs"

SEEDS: tuple[int, ...] = (20260814, 31874)
#: 15/30/60-minute cadences, expressed as strides over a shared 15-minute-
#: resolution trajectory -- matches run_m6_temporal.py's own
#: CADENCES_MINUTES/STRIDE_OF exactly (never the pathological 5-minute path).
CADENCE_STRIDES: tuple[int, ...] = (1, 2, 4)  # 15min, 30min, 60min.

#: Predeclared promotion bars (Section 6) -- fixed BEFORE any result is
#: inspected, per this milestone's own "predeclare the decision before
#: inspecting final results" instruction.
PRIMARY_INVARIANCE_REDUCTION_BAR_PP = 10.0
STANDARD_REGRESSION_GUARDRAIL_PP = 5.0
CALIBRATION_MIN_ACCEPTABLE_COVERAGE = 0.85  # alpha=0.1 target is 0.90; matches M3's own EARLY-bucket tolerance order.

SKIPPED_ARM_C_REASON = (
    "SKIPPED_WITH_REASON: elapsed-time information enters HydroCore through at least two structurally "
    "distinct paths -- a dedicated `timestamps` tensor consumed directly by "
    "`hydroswarm.model.encoders.TemporalEncoder.forward(timestamps=...)`'s own sinusoidal encoding, AND "
    "separate age-derived scalar feature columns already baked into `temporal_features`/`quality_features` "
    "upstream in `HydraulicFeatureBuilder.build` (confirmed by inspection this session: `age = now - "
    "series.timestamps_seconds[-1]` feeds directly into both feature arrays, independent of the `timestamps` "
    "tensor). A clean ablation that neutralizes 'only' the timestamp-conditioning path without collaterally "
    "distorting the age-derived feature columns' own learned scale/semantics is not a small, structurally-fair, "
    "non-invasive change within this milestone's scope (Section 2's own explicit skip condition: 'would require "
    "invasive production/model changes or make the comparison structurally unfair'). The primary A-vs-B "
    "comparison already directly tests the milestone's real question (does training-distribution diversity fix "
    "the invariance, with zero architecture change) without needing this diagnostic; Section 7's own "
    "interpretation guidance treats arm C as conditional on being run at all."
)


# ---------------------------------------------------------------------------
# Arm A: FIXED_CADENCE_CONTROL -- the exact frozen Milestone-1 arm-A
# checkpoint, per seed (bypassing run_m3_calibration._freeze_predictor's
# single-seed selection, mirroring Milestone 7's own per-seed loading so
# both screening seeds get genuine, independent results).
# ---------------------------------------------------------------------------


def _current_arm_checkpoint(seed: int) -> str:
    path = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-runs" / f"A-seed{seed}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    export_path = record["training_summary"]["export_path"]
    if not Path(export_path).exists():
        raise FileNotFoundError(
            f"FIXED_CADENCE_CONTROL's frozen Milestone-1 seed-{seed} checkpoint is missing on disk "
            f"({export_path}) -- experiments/runs/ is gitignored and does not survive across sessions; "
            f"re-run scripts/hydrocore_v5/run_m1_arm.py --arm A --seed {seed} first."
        )
    return export_path


def _load_model(export_path: str) -> HydroCore:
    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Arm B: CADENCE_DIVERSE training example construction. A NEW, additive
# parallel to `hydroswarm.training.causal_prefix.scenario_to_prefix_example`
# (never modified) that subsamples a shared high-resolution trajectory to a
# real cadence BEFORE truncating to "full history at that cadence" -- rather
# than repurposing `scenario_to_prefix_example`'s own `depth` argument
# (which means "first N NATIVE-resolution reports", not "first N reports at
# a coarser real cadence": feeding it a 15-minute-resolution scenario with
# depth=25 would just mean "the first 25 native 15-minute points",
# i.e. always native-cadence, never actually diversifying cadence at all).
# ---------------------------------------------------------------------------


def _cadence_full_history_prefix_classical_prior(
    signature_library, node_ids: Sequence[str], full_series: Sequence[SensorSeries], selected_indices: Sequence[int],
) -> dict[str, float]:
    """Cadence-aware counterpart of `causal_prefix._prefix_classical_prior`:
    identical construction (a fixed-length template grid matching the
    signature library's own native-resolution shape, reference slots left
    genuinely unobserved/NaN), except the OBSERVED rows are exactly
    `selected_indices` (the real strided report positions this cadence
    would actually have delivered) instead of `range(k)` -- so the
    classical_prior feature never leaks finer-than-this-cadence evidence
    that the corresponding SensorSeries evidence window does not itself
    contain."""

    if not full_series:
        return {node_id: 0.0 for node_id in node_ids}
    n = len(full_series[0].timestamps_seconds)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    values = np.full((n, len(node_ids)), np.nan, dtype=np.float32)
    valid = np.zeros((n, len(node_ids)), dtype=bool)
    for item in full_series:
        column = positions.get(item.node_id)
        if column is None:
            continue
        for row in selected_indices:
            value = item.concentration_mg_l[row]
            if value is not None and not item.missing[row] and np.isfinite(value):
                values[row, column] = float(value)
                valid[row, column] = True
    vector = signature_library.posterior_from_observations(values, valid)
    return dict(zip(signature_library.node_ids, map(float, vector), strict=True))


def cadence_prefix_example(
    scenario, network, signature_library, *, cadence_stride: int,
    feature_context=None, node_normalization=None, edge_normalization=None,
) -> ScenarioExample:
    """Cadence-diverse counterpart of `scenario_to_prefix_example`: full
    available history AT `cadence_stride` (matching arm A's own
    `full_history_policy` -- "same causal-prefix formulation", the only new
    axis of variation is cadence). Target/topology construction is
    IDENTICAL to `scenario_to_prefix_example`'s (copied, not re-derived, to
    avoid any risk of silently drifting from the governed target schema);
    the only real difference is the evidence-construction step (strided
    subsample instead of a native first-`depth` truncation) and threading
    `native_index_bound` (the real native-resolution index the strided
    selection reaches) into `_fault_any_within_depth` in place of a raw
    `depth`, since that helper indexes directly into the scenario's own
    native-resolution fault-mask arrays -- see its own docstring."""

    junction_ids = tuple(sorted(network.junction_name_list))
    if junction_ids != signature_library.node_ids:
        raise ValueError("scenario network nodes do not match the signature library")
    context = feature_context or build_feature_context(network)
    full_series = build_sensor_series(scenario, context)
    n = len(full_series[0].timestamps_seconds) if full_series else 0
    selected_indices = list(range(0, n, cadence_stride)) or [0]
    native_index_bound = selected_indices[-1] + 1
    series = [_slice_series(item, selected_indices) for item in full_series]
    target_timestamps = series[0].timestamps_seconds if series else ()
    prior = _cadence_full_history_prefix_classical_prior(signature_library, junction_ids, full_series, selected_indices)
    built = HydraulicFeatureBuilder(
        node_normalization=node_normalization, edge_normalization=edge_normalization
    ).build(
        network, context.graph, context.state, series,
        classical_prior=prior, window_steps=len(target_timestamps),
    )
    node_ids = built.node_ids
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    start_bins = (0, 60, 120, 240)
    duration_bins = (30, 60, 120)
    strength_bins = (0.5, 1.0, 2.0)
    split = scenario.manifest.split.value

    event_presence = scenario.manifest.event_type == EventType.CONTAMINATION.value
    cause = _event_cause(scenario)
    regions = assign_source_regions(network)
    if event_presence:
        source_node_id = scenario.manifest.incident.source_nodes[0]
        source = positions[source_node_id]
        source_region = regions[source_node_id]
        start_time = start_bins.index(scenario.manifest.incident.start_minute)
        duration = duration_bins.index(scenario.manifest.incident.duration_minutes)
        relative_strength = strength_bins.index(scenario.manifest.incident.relative_strength)
    else:
        source = source_region = start_time = duration = relative_strength = 0

    edge_ids = tuple(
        (network.get_link(name).start_node_name, network.get_link(name).end_node_name)
        for name in sorted(network.link_name_list)
    )
    source_candidate_ids = tuple(node_id for node_id in node_ids if node_id in network.junction_name_list)
    topology_metadata = TopologyMetadata(
        topology_hash=scenario.manifest.network_sha256,
        network_hash=HydraulicSimulator(network).state_hash(),
        node_ids=node_ids,
        edge_ids=edge_ids,
        source_candidate_ids=source_candidate_ids,
        hydraulic_state_hash=_hydraulic_state_hash(context.state),
        signature_library_hash=signature_library.manifest_hash,
        target_schema_version=TARGETS_V2_SCHEMA_VERSION,
        feature_schema_version=built.feature_schema_version,
    )

    targets = {
        "source_node": torch.tensor(source),
        "source_node_mask": torch.tensor(event_presence),
        "source_region": torch.tensor(source_region),
        "source_region_mask": torch.tensor(event_presence),
        "start_time": torch.tensor(start_time),
        "start_time_mask": torch.tensor(event_presence),
        "duration": torch.tensor(duration),
        "duration_mask": torch.tensor(event_presence),
        "relative_strength": torch.tensor(relative_strength),
        "relative_strength_mask": torch.tensor(event_presence),
        "event_presence": torch.tensor(event_presence),
        "event_cause": torch.tensor(EVENT_CAUSE_INDEX[cause]),
        "evidence_sufficiency": torch.tensor(_evidence_sufficiency(series)),
        "sensor_fault": torch.tensor([
            float(
                _fault_any_within_depth(scenario.frozen_mask, scenario.sensor_nodes, node_id, native_index_bound)
                or _fault_any_within_depth(scenario.communication_outage_mask, scenario.sensor_nodes, node_id, native_index_bound)
                or _fault_any_within_depth(scenario.drift_mask, scenario.sensor_nodes, node_id, native_index_bound)
                or _fault_any_within_depth(scenario.unit_mismatch_mask, scenario.sensor_nodes, node_id, native_index_bound)
            )
            for node_id in node_ids
        ]),
        "sensor_fault_mask": torch.tensor([node_id in scenario.sensor_nodes for node_id in node_ids]),
    }

    return ScenarioExample(
        scenario_id=str(scenario.manifest.scenario_id),
        network_id=scenario.manifest.network_id,
        split=split,
        seed=scenario.manifest.seed,
        seed_family=f"{scenario.manifest.network_family}:{scenario.manifest.seed_family}",
        stage=STAGE_MAP[scenario.manifest.stage.value],
        inputs={key: value.squeeze(0) for key, value in built.batch.items()},
        targets=targets,
        topology=topology_metadata,
    )


class CadenceCausalPrefixDatasetView:
    """Cadence-diverse counterpart of `CausalPrefixDatasetView`: draws ONE
    cadence stride per MICRO-BATCH (never per example, for the same
    batching-mechanics reason `CausalPrefixDatasetView` itself documents --
    `collate_scenarios` requires every example in one micro-batch to share
    identical tensor shapes, and different cadences produce different
    window lengths). Always full-history at that cadence (no separate depth
    axis -- matches arm A's own `full_history_policy`)."""

    def __init__(
        self, records: Sequence[ScenarioRecord], *, expected_split: str, signature_library,
        cadence_strides: Sequence[int], base_seed: int, batch_size: int = 2,
    ) -> None:
        if not records:
            raise ValueError("cadence-diverse dataset view cannot be empty")
        wrong = [r.scenario.manifest.scenario_id for r in records if r.scenario.manifest.split.value != expected_split]
        if wrong:
            raise ValueError(f"records belong to the wrong split: {wrong}")
        self.expected_split = expected_split
        self._records = tuple(records)
        self._signature_library = signature_library
        self._cadence_strides = tuple(cadence_strides)
        self._base_seed = base_seed
        self._batch_size = max(1, batch_size)
        self._draw = 0
        self._current_batch_index: int | None = None
        self._current_stride: int | None = None

    def __len__(self) -> int:
        return len(self._records)

    def _stride_for_current_batch(self) -> int:
        batch_index = self._draw // self._batch_size
        if batch_index != self._current_batch_index:
            seed_material = f"{self._base_seed}:{self.expected_split}:{batch_index}"
            seed_int = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
            rng = random.Random(seed_int)
            self._current_stride = rng.choice(self._cadence_strides)
            self._current_batch_index = batch_index
        assert self._current_stride is not None
        return self._current_stride

    def __getitem__(self, index: int) -> ScenarioExample:
        record = self._records[index]
        stride = self._stride_for_current_batch()
        self._draw += 1
        return cadence_prefix_example(
            record.scenario, record.network, self._signature_library,
            cadence_stride=stride, feature_context=record.feature_context,
        )

    @property
    def manifest_hash(self) -> str:
        entries = sorted(
            (
                {
                    "scenario_id": str(r.scenario.manifest.scenario_id),
                    "seed_family": f"{r.scenario.manifest.network_family}:{r.scenario.manifest.seed_family}",
                    "stage": r.scenario.manifest.stage.value,
                }
                for r in self._records
            ),
            key=lambda entry: entry["scenario_id"],
        )
        payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def stages_through(self, stage) -> "CadenceCausalPrefixDatasetView":
        selected = [r for r in self._records if STAGE_MAP[r.scenario.manifest.stage.value] <= stage]
        if not selected:
            selected = list(self._records[:1])
        return CadenceCausalPrefixDatasetView(
            selected, expected_split=self.expected_split, signature_library=self._signature_library,
            cadence_strides=self._cadence_strides, base_seed=self._base_seed, batch_size=self._batch_size,
        )


# ---------------------------------------------------------------------------
# Arm B training: a single Trainer.fit() call (no multi-phase curriculum
# needed here, unlike Milestone 7 -- every cadence view of golden-reference
# shares the SAME junction/node count, only the time axis length differs
# per micro-batch, which CadenceCausalPrefixDatasetView's own per-batch
# stride draw already keeps shape-consistent within each batch).
# ---------------------------------------------------------------------------


def _train_cadence_diverse_arm(seed: int) -> dict[str, Any]:
    config_path = ROOT / "configs" / "training-v5-causal.yaml"
    config = TrainingConfig.from_yaml(str(config_path), require_complete_task_weights=True)
    config = replace(config, seed=seed, gradnorm_log_every_n_batches=5)
    set_deterministic_seed(config.seed, deterministic=config.deterministic)

    # Training pool: SAME SPLIT_SEED_RANGES seed assignment as arm A's own
    # golden-reference pool (build_scenario_pool is reused completely
    # unmodified), just simulated at 15-minute resolution instead of the
    # standard 60-minute network -- so every physical incident (source,
    # onset, duration, strength, sensor placement, degradation stage) is
    # IDENTICAL to arm A's own training pool, differing only in simulated
    # time-resolution (matching M6's own "physically identical incidents,
    # different resolution" convention).
    train_records_hires = build_scenario_pool("train", network_loader=build_high_resolution_network)
    cadence_library = fit_pool_signature_library(train_records_hires)
    train_view = CadenceCausalPrefixDatasetView(
        train_records_hires, expected_split="train", signature_library=cadence_library,
        cadence_strides=CADENCE_STRIDES, base_seed=config.seed, batch_size=config.batch_size,
    )

    # Validation: STANDARD, UNMODIFIED native-hourly pool/library/
    # full_history_policy/CausalPrefixDatasetView -- identical to arm A's
    # own validation criterion (see module docstring).
    standard_train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    standard_library = fit_pool_signature_library(standard_train_records)  # deterministic; identical to arm A's own library.
    validation_records = build_scenario_pool("validation", network_loader=build_wntr_network)
    validation_view = CausalPrefixDatasetView(
        validation_records, expected_split="validation", signature_library=standard_library,
        depth_policy=full_history_policy, base_seed=config.seed, batch_size=config.batch_size,
    )

    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    run_root = RUNS_ROOT / f"CADENCE_DIVERSE-seed{seed}"
    started = time.time()
    trainer = Trainer(model, train_view, config=config, run_root=run_root, validation_dataset=validation_view)
    summary = trainer.fit()
    wall_seconds = time.time() - started

    record = {
        "schema_version": 1,
        "purpose": "Milestone 6B: CADENCE_DIVERSE training arm.",
        "arm": "CADENCE_DIVERSE",
        "seed": seed,
        "cadence_strides": list(CADENCE_STRIDES),
        "cadence_minutes": [stride * 15 for stride in CADENCE_STRIDES],
        "depth_policy": "full_history_policy (identical to arm A -- only cadence varies)",
        "train_scenario_count": len(train_records_hires),
        "validation_scenario_count": len(validation_records),
        "training_config": asdict(config),
        "training_config_source": str(config_path.relative_to(ROOT)),
        "train_manifest_hash": train_view.manifest_hash,
        "validation_manifest_hash": validation_view.manifest_hash,
        "cadence_signature_library_manifest_hash": cadence_library.manifest_hash,
        "standard_signature_library_manifest_hash": standard_library.manifest_hash,
        "wall_seconds": wall_seconds,
        "training_summary": asdict(summary),
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    M6B_RUNS_SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    (M6B_RUNS_SUMMARY_ROOT / f"CADENCE_DIVERSE-seed{seed}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return {
        "export_path": summary.export_path,
        "standard_library": standard_library,
        "record": record,
    }


# ---------------------------------------------------------------------------
# Section 4: standard EARLY/MID/MATURE causal-prefix development regime
# (new, but small -- reuses the frozen scenario_to_prefix_example/
# CAUSAL_PREFIX_DEPTHS/DEPTH_BUCKET_OF/M6's own _row_metrics unchanged).
# ---------------------------------------------------------------------------


def _standard_regime_evaluation(model: HydroCore, library, development_records: list[ScenarioRecord]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for depth in CAUSAL_PREFIX_DEPTHS:
            for record in development_records:
                example = scenario_to_prefix_example(
                    record.scenario, record.network, library, depth, feature_context=record.feature_context,
                )
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                node_ids = list(example.topology.node_ids)
                truth_index = int(example.targets["source_node"].item())
                truth_node = node_ids[truth_index]
                metrics = _row_metrics(probs, node_ids, truth_node)
                rows.append({"depth": depth, "bucket": DEPTH_BUCKET_OF[depth], **metrics})

    by_bucket: dict[str, Any] = {}
    for bucket in ("EARLY", "MID", "MATURE"):
        group = [row for row in rows if row["bucket"] == bucket]
        if not group:
            continue
        by_bucket[bucket] = {
            "n": len(group),
            **{metric: statistics.fmean(row[metric] for row in group) for metric in ("top1", "top3", "mrr", "nll", "brier")},
        }
    by_depth = {}
    for depth in CAUSAL_PREFIX_DEPTHS:
        group = [row for row in rows if row["depth"] == depth]
        by_depth[str(depth)] = {
            "n": len(group),
            **{metric: statistics.fmean(row[metric] for row in group) for metric in ("top1", "top3", "mrr", "nll", "brier")},
        }
    return {"by_bucket": by_bucket, "by_depth": by_depth}


# ---------------------------------------------------------------------------
# Section 5: calibration. `_fit_frozen_calibrator` (imported from
# run_m6_temporal.py, unmodified) supplies the ACTUAL frozen-scheme
# calibrator; this section separately re-collects the same CalibrationExample
# construction ONLY to report extra summary statistics (singleton rate,
# per-bucket coverage) that `_fit_frozen_calibrator`'s own return value
# does not expose -- the calibrator itself is never refit differently.
# ---------------------------------------------------------------------------


def _collect_calibration_examples(model: HydroCore, library, calibration_records: list[ScenarioRecord]) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    with torch.no_grad():
        for depth in CAUSAL_PREFIX_DEPTHS:
            for record in calibration_records:
                scenario = record.scenario
                example = scenario_to_prefix_example(scenario, record.network, library, depth, feature_context=record.feature_context)
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                truth = int(example.targets["source_node"].item())
                full_series = build_sensor_series(scenario, record.feature_context)
                truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                condition = classify_runtime_condition(truncated_series)
                bucket = DEPTH_BUCKET_OF[depth]
                examples.append(CalibrationExample(
                    probabilities=tuple(probs), true_index=truth, condition=condition,
                    network_id=f"{scenario.manifest.network_id}:{bucket}",
                ))
    return examples


def _calibration_summary(calibrator: SplitConformalCalibrator, examples: list[CalibrationExample]) -> dict[str, Any]:
    covered, sizes, singleton = [], [], []
    by_bucket_covered: dict[str, list[bool]] = {}
    for example in examples:
        candidate = calibrator.candidate_set(example.probabilities, condition=example.condition, network_id=example.network_id)
        hit = example.true_index in candidate
        covered.append(hit)
        sizes.append(len(candidate))
        singleton.append(len(candidate) == 1)
        bucket = example.network_id.rsplit(":", 1)[-1]
        by_bucket_covered.setdefault(bucket, []).append(hit)
    return {
        "n": len(examples),
        "coverage": statistics.fmean(covered) if covered else None,
        "mean_set_size": statistics.fmean(sizes) if sizes else None,
        "median_set_size": statistics.median(sizes) if sizes else None,
        "singleton_rate": statistics.fmean(singleton) if singleton else None,
        "coverage_by_bucket": {bucket: statistics.fmean(values) for bucket, values in by_bucket_covered.items()},
    }


# ---------------------------------------------------------------------------
# Per-arm evaluation orchestration. Uses the STANDARD (native-hourly,
# 25-point) `standard_library`/`target_timestamps` for BOTH arms -- matching
# run_m6_temporal.py's OWN convention exactly: `_infer`'s
# `model_input_classical_prior`/`aligned_observations_from_series` aligns
# WHATEVER resolution `series` happens to be onto this FIXED reference grid
# by nearest-time matching (never literal indexing), so cadence-varying
# evidence (15/30/60-minute incidents from `_generate_cadence_incident_pool`)
# is handled correctly regardless of which arm's model is under test. Arm
# B's OWN separate 15-minute cadence-TRAINING library
# (`cadence_signature_library_manifest_hash`, recorded in its own run
# record) is used ONLY during its own training and never here.
# ---------------------------------------------------------------------------


#: Per-scenario row-level arrays from run_m6_temporal.py's own reports
#: (already committed at ~1.6MB for a SINGLE arm/seed there) -- trimmed
#: here to keep this milestone's combined 2-arm x 2-seed JSON a reasonable
#: size. Every AGGREGATE statistic this milestone's promotion decision or
#: summary reads (by_cadence_and_report_count, paired_prediction_analysis_by_depth,
#: post_onset_*, by_stress_case, etc.) is untouched; only the raw per-row
#: detail is dropped, replaced with its own count for provenance. Fully
#: reproducible by re-running this script (deterministic seeds).
_HEAVY_ROW_KEYS: dict[str, tuple[str, ...]] = {
    "cadence": ("depth_rows", "checkpoint_rows", "post_onset_rows"),
    "irregular": ("rows",),
}


def _trim_heavy_rows(report: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    trimmed = dict(report)
    for key in keys:
        if key in trimmed and isinstance(trimmed[key], list):
            trimmed[f"{key}_count_only"] = len(trimmed.pop(key))
    return trimmed


def _evaluate_arm(
    arm_label: str, model: HydroCore, standard_library, target_timestamps, incidents, development_records, calibration_records,
) -> dict[str, Any]:
    print(f"  [{arm_label}] 3A/3B cadence experiment (matched-physical-time + post-onset diagnostic)...", flush=True)
    cadence_report = _trim_heavy_rows(run_cadence_experiment(model, standard_library, target_timestamps, incidents), _HEAVY_ROW_KEYS["cadence"])

    print(f"  [{arm_label}] irregular-telemetry stress diagnostics...", flush=True)
    irregular_report = _trim_heavy_rows(run_irregular_telemetry_experiment(model, standard_library, target_timestamps, incidents), _HEAVY_ROW_KEYS["irregular"])

    print(f"  [{arm_label}] standard EARLY/MID/MATURE regime...", flush=True)
    standard_report = _standard_regime_evaluation(model, standard_library, development_records)

    print(f"  [{arm_label}] calibration (frozen B_DEPTH_AWARE scheme)...", flush=True)
    calibrator = _fit_frozen_calibrator(model, standard_library, calibration_records)
    calibration_examples = _collect_calibration_examples(model, standard_library, calibration_records)
    calibration_report = _calibration_summary(calibrator, calibration_examples)

    return {
        "arm": arm_label,
        "cadence": cadence_report,
        "irregular_telemetry": irregular_report,
        "standard_regime": standard_report,
        "calibration": calibration_report,
    }


# ---------------------------------------------------------------------------
# Section 6: promotion rule. Predeclared (written before any result this
# session was inspected) -- see PRIMARY_INVARIANCE_REDUCTION_BAR_PP/
# STANDARD_REGRESSION_GUARDRAIL_PP/CALIBRATION_MIN_ACCEPTABLE_COVERAGE above.
# ---------------------------------------------------------------------------


def _promotion_decision(control: dict[str, Any], diverse: dict[str, Any]) -> dict[str, Any]:
    control_n2 = control["cadence"]["post_onset_n2_identical_fraction"]
    control_n3 = control["cadence"]["post_onset_n3_identical_fraction"]
    diverse_n2 = diverse["cadence"]["post_onset_n2_identical_fraction"]
    diverse_n3 = diverse["cadence"]["post_onset_n3_identical_fraction"]

    def _reduction_pp(control_value, diverse_value):
        if control_value is None or diverse_value is None:
            return None
        return (control_value - diverse_value) * 100

    n2_reduction_pp = _reduction_pp(control_n2, diverse_n2)
    n3_reduction_pp = _reduction_pp(control_n3, diverse_n3)
    best_reduction_pp = max((v for v in (n2_reduction_pp, n3_reduction_pp) if v is not None), default=None)
    criterion_1_invariance_reduced = bool(best_reduction_pp is not None and best_reduction_pp >= PRIMARY_INVARIANCE_REDUCTION_BAR_PP)

    control_n2_dist = control["cadence"]["post_onset_paired_by_n"].get("2", {}).get("mean_l1_probability_distance")
    diverse_n2_dist = diverse["cadence"]["post_onset_paired_by_n"].get("2", {}).get("mean_l1_probability_distance")
    control_n3_dist = control["cadence"]["post_onset_paired_by_n"].get("3", {}).get("mean_l1_probability_distance")
    diverse_n3_dist = diverse["cadence"]["post_onset_paired_by_n"].get("3", {}).get("mean_l1_probability_distance")
    distance_increased = bool(
        (control_n2_dist is not None and diverse_n2_dist is not None and diverse_n2_dist > control_n2_dist)
        or (control_n3_dist is not None and diverse_n3_dist is not None and diverse_n3_dist > control_n3_dist)
    )
    criterion_1_meaningful_not_noise = criterion_1_invariance_reduced and distance_increased

    def _bucket_top1(report, bucket):
        entry = report["standard_regime"]["by_bucket"].get(bucket)
        return entry["top1"] if entry else None

    early_regression_pp = None
    mature_regression_pp = None
    control_early, diverse_early = _bucket_top1(control, "EARLY"), _bucket_top1(diverse, "EARLY")
    control_mature, diverse_mature = _bucket_top1(control, "MATURE"), _bucket_top1(diverse, "MATURE")
    if control_early is not None and diverse_early is not None:
        early_regression_pp = (control_early - diverse_early) * 100
    if control_mature is not None and diverse_mature is not None:
        mature_regression_pp = (control_mature - diverse_mature) * 100
    criterion_2_no_standard_regression = bool(
        (early_regression_pp is None or early_regression_pp <= STANDARD_REGRESSION_GUARDRAIL_PP)
        and (mature_regression_pp is None or mature_regression_pp <= STANDARD_REGRESSION_GUARDRAIL_PP)
    )

    control_spread = control["cadence"]["max_cadence_sensitivity_spread_pp"]
    diverse_spread = diverse["cadence"]["max_cadence_sensitivity_spread_pp"]
    criterion_3_matched_time_stable = bool(
        diverse_spread is not None and control_spread is not None
        and (diverse_spread <= control_spread + 1.0 or not diverse["cadence"]["strongly_cadence_sensitive"])
    )

    diverse_coverage = diverse["calibration"]["coverage"]
    criterion_4_calibration_acceptable = bool(
        diverse_coverage is not None and diverse_coverage >= CALIBRATION_MIN_ACCEPTABLE_COVERAGE
    )

    criterion_5_no_authority_weakened = True  # trivially true: K/alpha/production fusion/OOD code never touched.

    all_criteria_met = bool(
        criterion_1_meaningful_not_noise and criterion_2_no_standard_regression
        and criterion_3_matched_time_stable and criterion_4_calibration_acceptable and criterion_5_no_authority_weakened
    )
    decision = "CADENCE_DIVERSE_TRAINING_JUSTIFIED" if all_criteria_met else "KEEP_CURRENT_TEMPORAL_TRAINING"

    return {
        "control_n2_identical_fraction": control_n2, "diverse_n2_identical_fraction": diverse_n2,
        "control_n3_identical_fraction": control_n3, "diverse_n3_identical_fraction": diverse_n3,
        "n2_reduction_pp": n2_reduction_pp, "n3_reduction_pp": n3_reduction_pp,
        "best_reduction_pp": best_reduction_pp,
        "primary_invariance_reduction_bar_pp": PRIMARY_INVARIANCE_REDUCTION_BAR_PP,
        "criterion_1_invariance_reduced": criterion_1_invariance_reduced,
        "criterion_1_distance_increased_not_noise": distance_increased,
        "criterion_1_met": criterion_1_meaningful_not_noise,
        "early_top1_regression_pp": early_regression_pp, "mature_top1_regression_pp": mature_regression_pp,
        "standard_regression_guardrail_pp": STANDARD_REGRESSION_GUARDRAIL_PP,
        "criterion_2_met": criterion_2_no_standard_regression,
        "control_matched_time_spread_pp": control_spread, "diverse_matched_time_spread_pp": diverse_spread,
        "criterion_3_met": criterion_3_matched_time_stable,
        "diverse_calibration_coverage": diverse_coverage,
        "calibration_min_acceptable_coverage": CALIBRATION_MIN_ACCEPTABLE_COVERAGE,
        "criterion_4_met": criterion_4_calibration_acceptable,
        "criterion_5_met": criterion_5_no_authority_weakened,
        "all_criteria_met": all_criteria_met,
        "decision": decision,
    }


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for Milestone 6B"
    locked_before = locked_test_opened(ROOT)
    started = time.time()

    standard_train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    standard_library = fit_pool_signature_library(standard_train_records)
    development_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)
    junctions = tuple(sorted(standard_train_records[0].network.junction_name_list))
    # The STANDARD (native-hourly, 25-point) reference grid `standard_library`
    # was fit against -- matches run_m6_temporal.py's own `main()` exactly.
    # `model_input_classical_prior`/`aligned_observations_from_series` aligns
    # WHATEVER resolution a given incident's `series` happens to be onto this
    # FIXED grid by nearest-time matching, so it is correct for both the
    # native-hourly control incidents AND the 15-minute-resolution cadence
    # incident pool below -- confirmed directly this session (using the
    # incident pool's own 97-point grid here instead raises a shape
    # mismatch in SignatureLibrary.posterior_from_observations).
    target_timestamps = build_sensor_series(standard_train_records[0].scenario, standard_train_records[0].feature_context)[0].timestamps_seconds

    print("generating shared cadence/stress incident pool (identical for both arms)...", flush=True)
    incidents = _generate_cadence_incident_pool(junctions)
    print(f"  {len(incidents)} incidents generated", flush=True)

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)

        print(f"[seed {seed}] loading FIXED_CADENCE_CONTROL (frozen Milestone-1 arm-A checkpoint)...", flush=True)
        control_export_path = _current_arm_checkpoint(seed)
        control_model = _load_model(control_export_path)
        control_report = _evaluate_arm("FIXED_CADENCE_CONTROL", control_model, standard_library, target_timestamps, incidents, development_records, calibration_records)

        print(f"[seed {seed}] training CADENCE_DIVERSE arm...", flush=True)
        diverse_training = _train_cadence_diverse_arm(seed)
        diverse_model = _load_model(diverse_training["export_path"])
        diverse_report = _evaluate_arm("CADENCE_DIVERSE", diverse_model, standard_library, target_timestamps, incidents, development_records, calibration_records)

        promotion = _promotion_decision(control_report, diverse_report)
        print(f"[seed {seed}] decision: {promotion['decision']} "
              f"(N2 reduction={promotion['n2_reduction_pp']}, N3 reduction={promotion['n3_reduction_pp']})", flush=True)

        per_seed.append({
            "seed": seed,
            "control_export_path": control_export_path,
            "diverse_export_path": diverse_training["export_path"],
            "control": control_report,
            "diverse": diverse_report,
            "promotion": promotion,
        })

    overall_justified = all(entry["promotion"]["all_criteria_met"] for entry in per_seed)
    overall_decision = "CADENCE_DIVERSE_TRAINING_JUSTIFIED" if overall_justified else "KEEP_CURRENT_TEMPORAL_TRAINING"

    wall_seconds = time.time() - started
    locked_after = locked_test_opened(ROOT)

    results = {
        "schema_version": 1,
        "section": "milestone_6b_cadence_diverse_training",
        "branch": "exp/hydrocore-v5-causal",
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "seeds": list(SEEDS),
        "cadence_strides": list(CADENCE_STRIDES),
        "cadence_minutes": [stride * 15 for stride in CADENCE_STRIDES],
        "primary_promotion_bars": {
            "primary_invariance_reduction_bar_pp": PRIMARY_INVARIANCE_REDUCTION_BAR_PP,
            "standard_regression_guardrail_pp": STANDARD_REGRESSION_GUARDRAIL_PP,
            "calibration_min_acceptable_coverage": CALIBRATION_MIN_ACCEPTABLE_COVERAGE,
        },
        "arm_c_timestamp_ablation": {"run": False, "reason": SKIPPED_ARM_C_REASON},
        "wall_seconds": wall_seconds,
        "per_seed": per_seed,
        "overall_decision": overall_decision,
        "overall_justified_all_seeds": overall_justified,
    }
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    # -- Summary markdown --
    lines = ["# Milestone 6B summary: cadence-diverse training", ""]
    lines.append(f"Seeds: {list(SEEDS)}. Wall seconds: {wall_seconds:.1f}. Overall decision: **{overall_decision}**")
    lines.append("")
    lines.append("Arm C (timestamp-conditioning ablation): SKIPPED_WITH_REASON (see JSON `arm_c_timestamp_ablation.reason`).")
    lines.append("")
    lines.append("## Primary endpoint: post-onset fixed-report-count identical-prediction fraction")
    lines.append("")
    lines.append("| seed | N | FIXED_CADENCE_CONTROL | CADENCE_DIVERSE | reduction (pp) |")
    lines.append("|---|---|---|---|---|")
    for entry in per_seed:
        promotion = entry["promotion"]
        for n, control_key, diverse_key, reduction_key in (
            (2, "control_n2_identical_fraction", "diverse_n2_identical_fraction", "n2_reduction_pp"),
            (3, "control_n3_identical_fraction", "diverse_n3_identical_fraction", "n3_reduction_pp"),
        ):
            control_val = promotion[control_key]
            diverse_val = promotion[diverse_key]
            reduction_val = promotion[reduction_key]
            lines.append(
                f"| {entry['seed']} | {n} | "
                f"{'n/a' if control_val is None else f'{control_val:.3f}'} | "
                f"{'n/a' if diverse_val is None else f'{diverse_val:.3f}'} | "
                f"{'n/a' if reduction_val is None else f'{reduction_val:.1f}'} |"
            )
    lines.append("")
    lines.append("## Standard regime (EARLY/MATURE top1) and matched-time spread")
    lines.append("")
    lines.append("| seed | arm | EARLY top1 | MATURE top1 | matched-time spread (pp) | calibration coverage |")
    lines.append("|---|---|---|---|---|---|")
    for entry in per_seed:
        for arm_label, arm_key in (("FIXED_CADENCE_CONTROL", "control"), ("CADENCE_DIVERSE", "diverse")):
            report = entry[arm_key]
            early = report["standard_regime"]["by_bucket"].get("EARLY", {}).get("top1")
            mature = report["standard_regime"]["by_bucket"].get("MATURE", {}).get("top1")
            spread = report["cadence"]["max_cadence_sensitivity_spread_pp"]
            coverage = report["calibration"]["coverage"]
            lines.append(
                f"| {entry['seed']} | {arm_label} | "
                f"{'n/a' if early is None else f'{early:.3f}'} | {'n/a' if mature is None else f'{mature:.3f}'} | "
                f"{spread:.2f} | {'n/a' if coverage is None else f'{coverage:.3f}'} |"
            )
    lines.append("")
    lines.append("## Per-seed promotion criteria")
    lines.append("")
    lines.append("| seed | criterion 1 (invariance) | criterion 2 (no regression) | criterion 3 (matched-time stable) | criterion 4 (calibration) | decision |")
    lines.append("|---|---|---|---|---|---|")
    for entry in per_seed:
        promotion = entry["promotion"]
        lines.append(
            f"| {entry['seed']} | {promotion['criterion_1_met']} | {promotion['criterion_2_met']} | "
            f"{promotion['criterion_3_met']} | {promotion['criterion_4_met']} | {promotion['decision']} |"
        )
    lines.append("")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "wall_seconds": wall_seconds,
        "locked_test_opened_after": locked_after,
        "overall_decision": overall_decision,
        "seeds": list(SEEDS),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
