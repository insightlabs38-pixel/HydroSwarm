"""Milestone 1 (experiments.txt) causal-prefix training corpus.

Builds depth-truncated ``ScenarioExample`` instances directly against
``hydroswarm.training.corpus``'s existing scenario-to-tensor construction, so
causal-prefix examples and full-history examples share one code path for
everything except the evidence-depth truncation itself. Truncation uses the
same earliest-``k`` ("causal prefix": growing evidence from incident onset)
semantics as ``scripts/capability_diagnostic/temporal_evidence_ablation.py``'s
``_truncate_causal_prefix`` -- the exact construction that produced the
frozen v4 baseline curve carried forward into
``reports/evaluation/hydrocore-v5/m0-baseline.json``. Not imported from that
script because it is frozen historical-diagnostic evidence, not a library.

Scope (documented, not silently narrowed -- see
``reports/evaluation/hydrocore-v5/m1-prefix-dataset.json``): this corpus
supervises the Sentinel task family only (source_node, source_region,
start_time, duration, relative_strength, event_presence, event_cause,
evidence_sufficiency, sensor_fault) on the single golden-reference topology,
contamination events only. Scout/Strategist/OOD/auxiliary task heads remain
declared in configs/training-v5-causal.yaml's task_weights (satisfying
``require_complete_task_weights=True``) but receive no target in this
corpus, matching the existing corpus format's own per-example masking
pattern (e.g. event_presence-conditional Sentinel sub-target masking)
rather than inventing a new mechanism.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from hydroswarm.data.scenarios import (
    CurriculumStage as ScenarioCurriculumStage,
    DatasetSplit,
    EventType,
    GeneratedScenario,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.preprocessing import HydraulicFeatureBuilder, SensorSeries
from hydroswarm.preprocessing.schema import NormalizationStats
from hydroswarm.simulation.wrapper import HydraulicSimulator

from .corpus import (
    EVENT_CAUSE_INDEX,
    STAGE_MAP,
    FeatureContext,
    SignatureLibrary,
    _event_cause,
    _evidence_sufficiency,
    _hydraulic_state_hash,
    assign_source_regions,
    build_feature_context,
    build_sensor_series,
    fit_signature_library,
)
from .data import CurriculumStage, ScenarioExample, TopologyMetadata
from .targets_v2 import TARGETS_V2_SCHEMA_VERSION

#: Milestone 1.1's approved evidence depths (experiments.txt): "1 2 3 4 6 12
#: 25 available reports". Identical to temporal_evidence_ablation.py's
#: CAUSAL_PREFIXES.
CAUSAL_PREFIX_DEPTHS: tuple[int, ...] = (1, 2, 3, 4, 6, 12, 25)
FULL_HISTORY_DEPTH = 25

#: Sentinel-family target keys this corpus actually supervises (see module
#: docstring's scope note). Kept as an explicit, importable constant so
#: every script that reports on this corpus (M1 eval, M2 conflict analysis)
#: agrees on exactly which tasks have real (non-fully-masked) targets here.
SUPERVISED_TASKS: frozenset[str] = frozenset({
    "source_node",
    "source_region",
    "start_time",
    "duration",
    "relative_strength",
    "event_presence",
    "event_cause",
    "evidence_sufficiency",
    "sensor_fault",
})


def truncate_causal_prefix(series: SensorSeries, depth: int) -> SensorSeries:
    """First `depth` reports of `series` -- growing evidence from incident
    onset, never a sliding late window. Mirrors
    temporal_evidence_ablation.py's `_truncate_causal_prefix` exactly."""

    n = len(series.timestamps_seconds)
    k = min(max(depth, 1), n)
    sl = slice(0, k)
    return SensorSeries(
        node_id=series.node_id,
        timestamps_seconds=series.timestamps_seconds[sl],
        concentration_mg_l=series.concentration_mg_l[sl],
        pressure_m=series.pressure_m[sl],
        health=series.health[sl],
        missing=series.missing[sl],
        drift=series.drift[sl],
        delayed=series.delayed[sl],
        frozen=series.frozen[sl] if series.frozen else (),
    )


def _fault_any_within_depth(mask: Any, sensor_nodes: Sequence[str], node_id: str, depth: int) -> bool:
    """Whether a frozen/outage/drift/unit-mismatch mask is set for `node_id`
    within the first `depth` reports only -- computed the same way
    `_evidence_sufficiency` is here: from the depth-truncated evidence, not
    the scenario's full future trajectory, so sensor_fault reflects what is
    actually knowable at this evidence depth rather than an oracle peek at
    a fault that has not happened yet within the prefix window."""

    if node_id not in sensor_nodes:
        return False
    column = sensor_nodes.index(node_id)
    n = mask.shape[0]
    k = min(max(depth, 1), n)
    return bool(mask[:k, column].any())


def _prefix_classical_prior(
    signature_library: SignatureLibrary,
    node_ids: Sequence[str],
    full_series: Sequence[SensorSeries],
    depth: int,
) -> dict[str, float]:
    """classical_prior computed from only the first `depth` reports,
    aligned onto the signature library's full fixed-length (25-step)
    template grid so `posterior_from_observations` can compare against it
    directly.

    NOT built via `model_input_classical_prior` /
    `aligned_observations_from_series`: that function resamples onto its
    target grid by NEAREST-TIME MATCHING with no distance cutoff, so
    feeding it a truncated series would make every one of the 25 template
    slots "valid" by repeatedly matching the single nearest (early)
    reading -- silently turning genuinely-missing future evidence into 25
    duplicated pseudo-observations and making low-depth priors misleadingly
    confident. Here, reference slots at or after `depth` are left
    genuinely unobserved (NaN / valid=False) instead."""

    if not full_series:
        return {node_id: 0.0 for node_id in node_ids}
    n = len(full_series[0].timestamps_seconds)
    k = min(max(depth, 1), n)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    values = np.full((n, len(node_ids)), np.nan, dtype=np.float32)
    valid = np.zeros((n, len(node_ids)), dtype=bool)
    for item in full_series:
        column = positions.get(item.node_id)
        if column is None:
            continue
        for row in range(k):
            value = item.concentration_mg_l[row]
            if value is not None and not item.missing[row] and np.isfinite(value):
                values[row, column] = float(value)
                valid[row, column] = True
    vector = signature_library.posterior_from_observations(values, valid)
    return dict(zip(signature_library.node_ids, map(float, vector), strict=True))


def scenario_to_prefix_example(
    scenario: GeneratedScenario,
    network: Any,
    signature_library: SignatureLibrary,
    depth: int,
    *,
    feature_context: FeatureContext | None = None,
    node_normalization: NormalizationStats | None = None,
    edge_normalization: NormalizationStats | None = None,
    # Milestone 8.7: threaded straight through to HydraulicFeatureBuilder.
    # build (see that method's own docstring/module-level constants for
    # what these change). Both default to the ORIGINAL/current behavior,
    # so every existing caller/checkpoint (Arm A included) is unaffected.
    unobserved_age_sentinel: str = "incident_elapsed",
    include_relative_gap_feature: bool = False,
) -> ScenarioExample:
    """Depth-truncated counterpart of
    `hydroswarm.training.corpus.scenario_to_example`: identical target
    derivation for incident-identity labels (privileged ground truth the
    model is being trained to infer, not leaked input -- unaffected by
    evidence depth, exactly as `scenario_to_example` already draws them from
    `scenario.manifest` for the full-history case), but with
    - INPUT sensor evidence truncated to the first `depth` reports,
    - classical_prior recomputed from only that truncated evidence via
      `_prefix_classical_prior` (never `signature_library.posterior`,
      which reads the scenario's full observed array), and
    - the sensor_fault target recomputed from the truncated evidence too
      (see `_fault_any_within_depth`), since fault status is knowable
      state, not a ground-truth identity label.
    """

    junction_ids = tuple(sorted(network.junction_name_list))
    if junction_ids != signature_library.node_ids:
        raise ValueError("scenario network nodes do not match the signature library")
    context = feature_context or build_feature_context(network)
    full_series = build_sensor_series(scenario, context)
    series = [truncate_causal_prefix(item, depth) for item in full_series]
    target_timestamps = series[0].timestamps_seconds if series else ()
    prior = _prefix_classical_prior(signature_library, junction_ids, full_series, depth)
    built = HydraulicFeatureBuilder(
        node_normalization=node_normalization, edge_normalization=edge_normalization
    ).build(
        network,
        context.graph,
        context.state,
        series,
        classical_prior=prior,
        window_steps=len(target_timestamps),
        unobserved_age_sentinel=unobserved_age_sentinel,
        include_relative_gap_feature=include_relative_gap_feature,
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
                _fault_any_within_depth(scenario.frozen_mask, scenario.sensor_nodes, node_id, depth)
                or _fault_any_within_depth(scenario.communication_outage_mask, scenario.sensor_nodes, node_id, depth)
                or _fault_any_within_depth(scenario.drift_mask, scenario.sensor_nodes, node_id, depth)
                or _fault_any_within_depth(scenario.unit_mismatch_mask, scenario.sensor_nodes, node_id, depth)
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


DepthPolicy = Callable[[random.Random], int]


def full_history_policy(rng: random.Random) -> int:
    """Arm A: corrected full-history control."""

    return FULL_HISTORY_DEPTH


def uniform_prefix_policy(rng: random.Random) -> int:
    """Arm B: prefix depths sampled approximately uniformly."""

    return rng.choice(CAUSAL_PREFIX_DEPTHS)


#: Arm C weighting (experiments.txt Milestone 1.3): 1-3 high probability,
#: 4-6 medium, 12-25 lower but still substantial. Mature history (12, 25)
#: is never eliminated, matching "Do not eliminate mature-history training."
_EARLY_WEIGHTED_CHOICES: tuple[tuple[int, int], ...] = (
    (1, 3), (2, 3), (3, 3),
    (4, 2), (6, 2),
    (12, 1), (25, 1),
)


def early_weighted_prefix_policy(rng: random.Random) -> int:
    """Arm C: early-weighted causal prefix."""

    depths = [depth for depth, _weight in _EARLY_WEIGHTED_CHOICES]
    weights = [weight for _depth, weight in _EARLY_WEIGHTED_CHOICES]
    return rng.choices(depths, weights=weights, k=1)[0]


ARM_POLICIES: dict[str, DepthPolicy] = {
    "A": full_history_policy,
    "B": uniform_prefix_policy,
    "C": early_weighted_prefix_policy,
}


@dataclass(frozen=True, slots=True)
class ScenarioRecord:
    """One frozen, already-split-assigned scenario this view can build
    prefix examples from at any depth. Holding the `GeneratedScenario` (not
    just a manifest) means depth truncation never re-simulates anything --
    the lazy view slices the same in-memory trajectory every draw, matching
    "prefer a lazy prefix view/wrapper rather than physically copying
    massive tensors" (experiments.txt Milestone 1.1)."""

    scenario: GeneratedScenario
    network: Any
    feature_context: FeatureContext


class CausalPrefixDatasetView:
    """Lazy `ScenarioDatasetView` (see hydroswarm.training.data) that draws
    ONE depth per MICRO-BATCH (not per example) according to an arm's
    `DepthPolicy`, rather than materializing every depth as a separate
    stored example -- the latter would violate `GovernedScenarioDataset`'s
    one-entry-per-seed-family invariant and would defeat the point of a
    lazy view. Per-batch (not per-example) depth is a batching-mechanics
    constraint, not a scientific compromise: `collate_scenarios` requires
    every example in one micro-batch to share identical tensor shapes, and
    a batch mixing two evidence depths has two different time-axis
    lengths. Depth itself is still resampled every `batch_size` draws, so
    it varies batch-to-batch and epoch-to-epoch -- the intended reading of
    "prefix depths sampled approximately uniformly" (Milestone 1.3), just
    at batch instead of per-example granularity. Relies on
    torch.utils.data.DataLoader (num_workers=0, the only device Trainer's
    TrainingConfig ever configures) fetching all `batch_size` indices of
    one batch via consecutive `__getitem__` calls before the next batch --
    true for the standard single-process map-style fetch path."""

    def __init__(
        self,
        records: Sequence[ScenarioRecord],
        *,
        expected_split: str,
        signature_library: SignatureLibrary,
        depth_policy: DepthPolicy,
        base_seed: int,
        batch_size: int = 2,
        # Milestone 8.7: see scenario_to_prefix_example's own docstring.
        # Defaults preserve current/Arm-A behavior for every existing caller.
        unobserved_age_sentinel: str = "incident_elapsed",
        include_relative_gap_feature: bool = False,
    ) -> None:
        if not records:
            raise ValueError("causal-prefix dataset view cannot be empty")
        wrong = [r.scenario.manifest.scenario_id for r in records if r.scenario.manifest.split.value != expected_split]
        if wrong:
            raise ValueError(f"records belong to the wrong split: {wrong}")
        self.expected_split = expected_split
        self._records = tuple(records)
        self._signature_library = signature_library
        self._depth_policy = depth_policy
        self._base_seed = base_seed
        self._batch_size = max(1, batch_size)
        self._unobserved_age_sentinel = unobserved_age_sentinel
        self._include_relative_gap_feature = include_relative_gap_feature
        self._draw = 0
        self._current_batch_index: int | None = None
        self._current_depth: int | None = None

    def __len__(self) -> int:
        return len(self._records)

    def _depth_for_current_batch(self) -> int:
        batch_index = self._draw // self._batch_size
        if batch_index != self._current_batch_index:
            seed_material = f"{self._base_seed}:{self.expected_split}:{batch_index}"
            seed_int = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
            rng = random.Random(seed_int)
            self._current_depth = self._depth_policy(rng)
            self._current_batch_index = batch_index
        assert self._current_depth is not None
        return self._current_depth

    def __getitem__(self, index: int) -> ScenarioExample:
        record = self._records[index]
        depth = self._depth_for_current_batch()
        self._draw += 1
        return scenario_to_prefix_example(
            record.scenario,
            record.network,
            self._signature_library,
            depth,
            feature_context=record.feature_context,
            unobserved_age_sentinel=self._unobserved_age_sentinel,
            include_relative_gap_feature=self._include_relative_gap_feature,
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

    def stages_through(self, stage: CurriculumStage) -> "CausalPrefixDatasetView":
        selected = [r for r in self._records if STAGE_MAP[r.scenario.manifest.stage.value] <= stage]
        if not selected:
            selected = list(self._records[:1])
        return CausalPrefixDatasetView(
            selected,
            expected_split=self.expected_split,
            signature_library=self._signature_library,
            depth_policy=self._depth_policy,
            base_seed=self._base_seed,
            batch_size=self._batch_size,
            # Milestone 8.7: Trainer.fit() calls stages_through() every
            # epoch and trains on ITS return value, never on the original
            # view directly -- omitting these here silently reverted every
            # epoch's actual training data to Arm-A/default feature
            # semantics regardless of what the caller configured (caught
            # empirically: AGE_FIX_PLUS_RELATIVE_TIME's dimension mismatch
            # crashed loudly; AGE_FIX_ONLY's identical-shape silent
            # reversion would not have).
            unobserved_age_sentinel=self._unobserved_age_sentinel,
            include_relative_gap_feature=self._include_relative_gap_feature,
        )


#: Milestone 1.1 corpus scope (see module docstring): a single canonical
#: topology (golden-reference -- the same network m0-baseline.json and
#: temporal-capability.json's frozen curve were measured against), fresh
#: host-native scenarios rather than a replay of the historical committed
#: cycle-b2 corpus. Replaying cycle-b2 was attempted and rejected: this
#: session's WNTR/numpy stack does not reproduce cycle-b2's committed
#: scenario_ids, source nodes, or sensor layouts bit-for-bit even when
#: exactly replaying scripts/generate_cycle_b_corpus.py's own
#: `_generate_topology_scenarios` construction (0/3000 golden-reference
#: train scenarios matched on this host) -- a cross-environment RNG/library
#: divergence, matching the cross-architecture non-reproducibility already
#: documented in scripts/restore_cycle_b2_train_scenario_cache.py's module
#: docstring, just more severe on this host. Splitting is therefore done
#: here via disjoint seed ranges per split, assigned BEFORE any prefix
#: depth is generated (experiments.txt's "split scenario first"); depths
#: are always views over the resulting split-assigned scenario, so no
#: split-safety guarantee here actually depends on matching cycle-b2.
NETWORK_FAMILY = "golden-reference"
SPLIT_SEED_RANGES: dict[str, tuple[int, int]] = {
    "train": (900_000_000, 600),
    "validation": (901_000_000, 100),
    "development_holdout": (902_000_000, 120),
    "calibration": (903_000_000, 150),
}


def _degradation_probabilities(stage: ScenarioCurriculumStage) -> dict[str, float]:
    """Same construction as scripts/generate_cycle_b_corpus.py's own
    `_degradation_probabilities` (not imported -- that module is a script,
    not a library, and this is a two-line pure function): higher
    missingness/fault probability for later curriculum stages."""

    degraded = stage in {ScenarioCurriculumStage.DEGRADED, ScenarioCurriculumStage.SHIFT, ScenarioCurriculumStage.ADVERSARIAL}
    shifted = stage in {ScenarioCurriculumStage.SHIFT, ScenarioCurriculumStage.ADVERSARIAL}
    return {
        "missing_probability": 0.08 if stage != ScenarioCurriculumStage.CLEAN else 0.0,
        "frozen_probability": 0.06 if degraded else 0.01,
        "communication_outage_probability": 0.06 if degraded else 0.01,
        "unit_mismatch_probability": 0.02 if shifted else 0.0,
    }


def build_scenario_pool(split: str, *, network_loader: Callable[[], Any]) -> list[ScenarioRecord]:
    """Deterministically (re)generate the full scenario pool for one split.
    Cheap (~0.1s/scenario) and self-consistent on a single host, so every
    script that needs this split regenerates it fresh rather than
    depending on a persisted, potentially-stale cache."""

    if split not in SPLIT_SEED_RANGES:
        raise ValueError(f"unknown causal-prefix split: {split}")
    start_seed, count = SPLIT_SEED_RANGES[split]
    generator = WNTRScenarioGenerator()
    stages = tuple(ScenarioCurriculumStage)
    records: list[ScenarioRecord] = []
    for index in range(count):
        stage = stages[index % len(stages)]
        seed = start_seed + index * 100
        network = network_loader()
        config = ScenarioGenerationConfig(
            seed=seed,
            network_id=NETWORK_FAMILY,
            network_family=NETWORK_FAMILY,
            split=DatasetSplit(split),
            stage=stage,
            event_type=EventType.CONTAMINATION,
            sensor_count=3 + (index % 3),
            pipe_outage_probability=0.0,
            **_degradation_probabilities(stage),
        )
        scenario, randomized_network = generator.generate_with_network(network, config)
        context = build_feature_context(randomized_network)
        records.append(ScenarioRecord(scenario=scenario, network=randomized_network, feature_context=context))
    return records


def fit_pool_signature_library(train_records: Sequence[ScenarioRecord]) -> SignatureLibrary:
    """Fit the classical signature library from the train split's own pool
    -- never validation/development_holdout/calibration -- matching
    fit_signature_library's own "training scenarios only" contract."""

    if not train_records:
        raise ValueError("cannot fit a signature library from an empty train pool")
    node_ids = tuple(sorted(train_records[0].network.junction_name_list))
    scenarios = [record.scenario for record in train_records]
    return fit_signature_library(scenarios, node_ids)
