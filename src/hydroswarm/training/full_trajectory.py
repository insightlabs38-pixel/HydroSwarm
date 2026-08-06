"""Full incident trajectory assembly (core-issues2.txt Phase 7).

Combines every label generator Phase 1-6 built into one governed
per-incident bundle: the Sentinel-level snapshot (source localization,
event/control/auxiliary targets -- folded into one ScenarioExample.targets
dict, exactly like scenario_to_example already produces, now extended),
plus the incident's Scout sampling sub-trajectory and Strategist planning
sub-trajectory.

Deliberately NOT flattened into one single FullTrajectory hash chain:
Scout's and Strategist's own TrajectoryState sequences already correctly
capture their own internal sequential integrity (ScoutTrajectory/
StrategistTrajectory each wrap a real, hash-chain-validated FullTrajectory
-- see trajectory_v2.py). Inventing connective-tissue steps to splice all
three into one artificial mega-sequence would not improve on that without
a concrete consumer that needs it; IncidentTrajectory is the governed
container the plan's "every trajectory must record ... incident ID, step
index, ... controller decision, candidate plans, exact verification
results, ..." requirement maps onto, without discarding any of the
sub-structure each piece already validates independently.

No future-observation leakage: every piece here is built from the same
scenario's own already-simulated evidence (Scout/Strategist/auxiliary
targets all read from data the scenario generator produced once, at
generation time) -- nothing here peeks at a later trajectory step's
outcome to label an earlier one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hydroswarm.classical.metrics import entropy
from hydroswarm.classical.signatures import SignatureArtifact, SignatureLibrary, localize_with_signatures
from hydroswarm.data.scenarios import GeneratedScenario
from hydroswarm.inference.pipeline import HybridInferencePipeline
from hydroswarm.preprocessing.schema import NormalizationStats

from .auxiliary_labels import future_concentration_target, sensor_reconstruction_target, travel_time_target
from .control_labels import classify_evidence_sufficiency, classify_next_step, next_step_target
from .corpus import (
    FeatureContext,
    _event_cause,
    build_feature_context,
    build_sensor_series,
    scenario_to_example,
    sensor_health_summary,
)
from .data import ScenarioExample
from .ood_categories import OODCategory
from .ood_labels import classify_ood_category, ood_class_target
from .scout_trajectory import ScoutTrajectory, build_scout_trajectory
from .strategist_trajectory import StrategistTrajectory, build_strategist_trajectory


@dataclass(frozen=True, slots=True)
class IncidentTrajectory:
    example: ScenarioExample
    ood_category: OODCategory
    scout: ScoutTrajectory
    strategist: StrategistTrajectory


def build_incident_trajectory(
    scenario: GeneratedScenario,
    network: Any,
    signature_library: SignatureLibrary,
    artifact: SignatureArtifact,
    *,
    topology_hash: str,
    validated_topology_hashes: Any,
    broader_validated_artifact_exists: bool = False,
    feature_context: FeatureContext | None = None,
    node_normalization: NormalizationStats | None = None,
    edge_normalization: NormalizationStats | None = None,
    maximum_samples: int = 5,
    maximum_exact_simulations: int = 3,
    maximum_plans: int = 9,
) -> IncidentTrajectory:
    context = feature_context or build_feature_context(network)
    example = scenario_to_example(
        scenario,
        network,
        signature_library,
        feature_context=context,
        node_normalization=node_normalization,
        edge_normalization=edge_normalization,
    )
    # The canonical, full topology node space (junctions + reservoirs +
    # tanks) example.topology and its sensor_fault/source_node targets are
    # already indexed against -- deliberately NOT re-derived as
    # sorted(network.junction_name_list) here. That smaller, junction-only
    # ordering is what Scout/Strategist/auxiliary label generation was
    # originally tested against in isolation (Phase 2/3/6), and it happens
    # to agree with the canonical order only when a network has zero
    # reservoirs/tanks -- not the case for any real network. Reading
    # node_ids from example.topology (built once, by scenario_to_example,
    # from HydraulicFeatureBuilder's own canonical_node_order) is what
    # makes every graph-local index this function produces
    # (sample_node/target_pointer/travel_time's node axis) share the exact
    # same index space source_node_logits/sensor_fault_logits already use.
    node_ids = example.topology.node_ids if example.topology is not None else tuple(sorted(network.node_name_list))

    category = classify_ood_category(
        scenario,
        topology_hash=topology_hash,
        validated_topology_hashes=validated_topology_hashes,
        broader_validated_artifact_exists=broader_validated_artifact_exists,
    )

    series = build_sensor_series(scenario, context)
    healthy_fraction, sensors_ever_healthy = sensor_health_summary(series)

    observations, observation_mask = HybridInferencePipeline._signature_observations(series, artifact)
    localization = localize_with_signatures(observations, artifact, observation_mask=observation_mask)
    posterior_entropy_bits = entropy(np.asarray(list(localization.posterior.probabilities.values())))

    sufficient = classify_evidence_sufficiency(
        healthy_fraction=healthy_fraction,
        sensors_ever_healthy=sensors_ever_healthy,
        posterior_entropy_bits=posterior_entropy_bits,
        ood_category=category,
    )
    # OODCategory does not itself distinguish CAUTION vs. OUTSIDE_VALIDATED_RANGE
    # severity (that is a separate, runtime-only concept -- inference.ood.
    # OODDetector's combined score -- not something generated as ground
    # truth here). Any non-NONE category means calibration_valid is False
    # per OOD_CATEGORY_BEHAVIOR's fail-closed table, which next_step's
    # ABSTAIN branch treats as "outside validated range" for control
    # purposes.
    step = classify_next_step(
        ood_level_outside_validated_range=category != OODCategory.NONE,
        evidence_sufficient=sufficient,
        sample_count=0,
        event_cause=_event_cause(scenario),
    )

    edge_ids = example.topology.edge_ids if example.topology is not None else ()
    scout = build_scout_trajectory(scenario, artifact, node_ids, maximum_samples=maximum_samples)
    strategist = build_strategist_trajectory(
        scenario,
        network,
        context,
        artifact,
        node_ids,
        edge_ids,
        maximum_exact_simulations=maximum_exact_simulations,
        maximum_plans=maximum_plans,
    )

    extra_targets: dict[str, torch.Tensor] = {
        **ood_class_target(category),
        **next_step_target(step),
        "evidence_sufficiency": torch.tensor(sufficient),
        **sensor_reconstruction_target(scenario, node_ids),
        **future_concentration_target(scenario, node_ids),
        **travel_time_target(scenario, context.graph, node_ids),
    }
    merged_example = dataclasses.replace(example, targets={**example.targets, **extra_targets})

    return IncidentTrajectory(example=merged_example, ood_category=category, scout=scout, strategist=strategist)
