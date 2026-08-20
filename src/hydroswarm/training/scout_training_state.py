"""M10.2 Scout supervision/representation refit amendment, Part 3: real
multi-round Scout TRAINING-state corpus wiring.

Honors `hydroswarm.training.scout_state_contract`'s already-frozen,
versioned input-conditioning contract (Section 18.2): every one of the six
required Scout-state items is representable via an EXISTING `HydroBatch`
channel, with NO new `HydroCore` constructor parameter. This module is the
"corpus/pipeline wiring pass" that contract's own docstring said was still
missing -- it POPULATES those channels with real multi-round Scout state for
the first time, for a genuine training corpus (not merely the evaluation-time
adapter `hydroswarm.evaluation.scout_state` already built for M10.2's
preflight).

Versioned SEPARATELY from:
- `hydroswarm.evaluation.scout_state.SCOUT_EVAL_STATE_SCHEMA_VERSION`
  (`"scout-eval-state-v1"`) -- that module decodes an ALREADY-BUILT batch's
  raw Scout outputs at evaluation time; it does not build the batch itself.
- `hydroswarm.training.checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION`
  (still `"scout-state-v1-unbuilt"`, UNCHANGED by this module) -- that
  placeholder names a *checkpoint-identity-recorded* training-corpus schema
  claim, which must only ever be bumped once a real training RUN has
  actually used this schema and the resulting checkpoint records it (the
  Level-A/B refit checkpoint identity this task's Part 5/6 produces does
  that; the ORIGINAL M9.6 checkpoints never used this schema and must never
  be described as if they did).

## Channel wiring (frozen, versioned by `SCOUT_TRAINING_STATE_SCHEMA_VERSION`)

| Contract item | `HydroBatch` channel | This module's encoding |
|---|---|---|
| `already_sampled` | `sensor_mask` (via a synthetic `SensorSeries`) | a scout-sampled node becomes a real, additional sensor entry fed through `HydraulicFeatureBuilder` -- the SAME real feature-encoding path (structural sensor-distance features, `quality_features`, `sensor_mask`) every ordinary sensor already goes through, not a hand-patched tensor |
| `currently_revealed_evidence` | `quality_features` + `sensor_mask` (same synthetic series) | the revealed grab-sample concentration value, valid from the reveal timestep onward, missing before it -- exactly `hydroswarm.training.scout_trajectory.reveal_sample_measurement`'s own "read at the scenario's own final observation timestamp, broadcast forward" convention |
| `current_posterior_candidate_state` | `classical_prior` | recomputed via the existing `hydroswarm.training.causal_prefix._prefix_classical_prior`, given the AUGMENTED (original + synthetic scout) sensor series -- the same real prior machinery every other v5-causal example already uses, now correctly reflecting revealed scout evidence too |
| `current_sampling_round` | `role_features[0]` | `sampling_round / MAXIMUM_SAMPLES_BOUND`, one incident-level scalar |
| `sample_budget_remaining` | `role_features[1]` | `sample_budget_remaining / MAXIMUM_SAMPLES_BOUND`, same vector |
| `sampling_constraints_accessibility` | `residual_features[:, :, 0]` | `1.0` for junction (real sample-candidate) nodes, `0.0` for reservoir/tank nodes -- per-node |

`role_features[2:8]` and `residual_features[:, :, 1:4]` are left `0.0` (reserved, unused) -- confirmed by direct
`grep` (and this module's own tests) that NO other real caller anywhere in the codebase populates
`role_features`/`residual_features` today, so these index assignments cannot collide with any existing usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from hydroswarm.classical.signatures import SignatureLibrary
from hydroswarm.data.scenarios import GeneratedScenario
from hydroswarm.model.core import HydroBatch
from hydroswarm.preprocessing import HydraulicFeatureBuilder, SensorSeries
from hydroswarm.training.causal_prefix import (
    FeatureContext,
    _prefix_classical_prior,
    build_feature_context,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series
from hydroswarm.training.scout_trajectory import MAXIMUM_SAMPLES_BOUND

SCOUT_TRAINING_STATE_SCHEMA_VERSION = "scout-training-state-v1"

ROLE_FEATURE_ROUND_INDEX = 0
ROLE_FEATURE_BUDGET_INDEX = 1
RESIDUAL_FEATURE_ACCESSIBILITY_INDEX = 0


def _synthetic_scout_sensor_series(
    node_id: str,
    context: FeatureContext,
    timestamps_seconds: Sequence[float],
    *,
    reveal_at_index: int,
    revealed_concentration_mg_l: float,
) -> SensorSeries:
    """One additional, synthetic `SensorSeries` entry representing a Scout
    grab sample -- fed through the exact same real `HydraulicFeatureBuilder`
    path as every ordinary sensor, so `sensor_mask`/`quality_features` (and
    structural sensor-distance features) correctly reflect this node now
    being observed, with no hand-patched tensor arithmetic."""

    n = len(timestamps_seconds)
    if n == 0:
        raise ValueError("timestamps_seconds must be non-empty")
    reveal_at_index = max(0, min(reveal_at_index, n - 1))
    pressure_estimate = context.state.pressure_m[node_id].estimate
    revealed = [index >= reveal_at_index for index in range(n)]
    return SensorSeries(
        node_id=node_id,
        timestamps_seconds=tuple(float(value) for value in timestamps_seconds),
        concentration_mg_l=tuple(float(revealed_concentration_mg_l) if is_revealed else None for is_revealed in revealed),
        pressure_m=tuple(pressure_estimate if is_revealed else None for is_revealed in revealed),
        health=tuple(1.0 if is_revealed else 0.0 for is_revealed in revealed),
        missing=tuple(not is_revealed for is_revealed in revealed),
        drift=tuple(False for _ in range(n)),
        delayed=tuple(False for _ in range(n)),
    )


@dataclass(frozen=True, slots=True)
class ScoutTrainingStateBatch:
    """One real, leakage-safe, decision-time Scout training INPUT (unbatched
    -- caller collates via `torch.stack`/`unsqueeze(0)` the same way every
    other `ScenarioExample` in this codebase is collated). `node_ids` is the
    canonical full-topology ordering `batch` tensors are indexed against
    (`HydraulicFeatureBuilder`'s own `canonical_node_order`)."""

    node_ids: tuple[str, ...]
    junction_ids: tuple[str, ...]
    batch: HydroBatch
    schema_version: str = SCOUT_TRAINING_STATE_SCHEMA_VERSION


def build_scout_training_state_batch(
    scenario: GeneratedScenario,
    network: Any,
    signature_library: SignatureLibrary,
    depth: int,
    *,
    already_sampled: Sequence[str],
    revealed_values_mg_l: Mapping[str, float],
    sampling_round: int,
    sample_budget_total: int,
    feature_context: FeatureContext | None = None,
    unobserved_age_sentinel: str = "fixed",
    include_relative_gap_feature: bool = False,
) -> ScoutTrainingStateBatch:
    """Builds one real decision-time Scout training input. `already_sampled`
    is the set of nodes a Scout policy has grab-sampled STRICTLY BEFORE this
    decision (their revealed values must come from `revealed_values_mg_l`,
    computed offline -- see `hydroswarm.training.scout_targets` Part 4 -- and
    must never depend on anything at or after this decision step). Nodes
    already covered by the scenario's own ORIGINAL sensors are skipped (that
    evidence already flows through the ordinary `build_sensor_series` path;
    adding a duplicate synthetic entry for them would double-count, not add
    information)."""

    if depth < 1:
        raise ValueError("depth must be at least 1")
    junction_ids = tuple(sorted(network.junction_name_list))
    if junction_ids != signature_library.node_ids:
        raise ValueError("scenario network nodes do not match the signature library")
    context = feature_context or build_feature_context(network)
    base_series = build_sensor_series(scenario, context)
    existing_sensor_ids = {item.node_id for item in base_series}
    reveal_at_index = min(depth, len(scenario.timestamps_seconds)) - 1

    synthetic_series = [
        _synthetic_scout_sensor_series(
            node_id, context, scenario.timestamps_seconds,
            reveal_at_index=reveal_at_index, revealed_concentration_mg_l=revealed_values_mg_l[node_id],
        )
        for node_id in already_sampled
        if node_id not in existing_sensor_ids and node_id in revealed_values_mg_l
    ]
    full_series = base_series + synthetic_series
    series = [truncate_causal_prefix(item, depth) for item in full_series]
    target_timestamps = series[0].timestamps_seconds if series else ()
    prior = _prefix_classical_prior(signature_library, junction_ids, full_series, depth)

    built = HydraulicFeatureBuilder().build(
        network, context.graph, context.state, series,
        classical_prior=prior, window_steps=len(target_timestamps),
        unobserved_age_sentinel=unobserved_age_sentinel,
        include_relative_gap_feature=include_relative_gap_feature,
    )
    node_ids = built.node_ids
    nodes = len(node_ids)

    batch: dict[str, torch.Tensor] = dict(built.batch)

    role_features = torch.zeros(1, 8, dtype=torch.float32)
    role_features[0, ROLE_FEATURE_ROUND_INDEX] = float(sampling_round) / float(MAXIMUM_SAMPLES_BOUND)
    remaining = max(0, sample_budget_total - len(already_sampled))
    role_features[0, ROLE_FEATURE_BUDGET_INDEX] = float(remaining) / float(MAXIMUM_SAMPLES_BOUND)
    batch["role_features"] = role_features

    residual_features = torch.zeros(1, nodes, 4, dtype=torch.float32)
    junction_set = set(junction_ids)
    for position, node_id in enumerate(node_ids):
        residual_features[0, position, RESIDUAL_FEATURE_ACCESSIBILITY_INDEX] = 1.0 if node_id in junction_set else 0.0
    batch["residual_features"] = residual_features

    return ScoutTrainingStateBatch(node_ids=node_ids, junction_ids=junction_ids, batch=batch)  # type: ignore[arg-type]
