"""Strategist trajectory-state generation (core-issues2.txt Phase 3).

Wires the already-existing strategist_labels.generate_strategist_labels()
into TrajectoryState/FullTrajectory, mirroring scout_trajectory.py's
pattern for Scout.

Deliberately reuses HybridInferencePipeline's own classical-localization
and plan-context-construction logic (_signature_observations/
_credible_nodes/_planning_context) rather than re-deriving "which nodes are
probable enough to plan around" a second way. This is not a layering
accident: core-issues.txt's repair pass (commit a99cdbc) fixed a real,
previously-undiscovered train/serve skew defect that arose from exactly
this class of duplicated logic (corpus generation and live inference each
independently computing the same conceptual value, and drifting). Reusing
the production path here is a deliberate defense against that same failure
mode recurring for plan-context construction.

Strategist supervision is a single decision point per incident state (Phase
3's own spec: "For each eligible incident state: 1. Generate bounded
candidate plans. 2. Run every training-label plan through exact WNTR
verification. 3. Record: ..."), not an iterative loop like Scout's sampling
sequence -- generate_strategist_labels does not call
hydroswarm.planning.response.revise_rejected_plan (multi-round plan
revision), so neither does this module; that is a distinct, larger future
extension (trajectory_v2.py's own docstring names it: "supervised
Strategist training can see the actual verifier feedback that preceded a
plan revision" -- not built yet, tracked as a known gap, not silently
dropped).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

import torch

from hydroswarm.classical.signatures import SignatureArtifact, localize_with_signatures
from hydroswarm.data.scenarios import EventType, GeneratedScenario
from hydroswarm.domain import IncidentState
from hydroswarm.inference.pipeline import HybridInferencePipeline
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_INDEX, ACTION_TEMPLATES

from .corpus import FeatureContext, build_sensor_series
from .strategist_labels import StrategistLabel, generate_strategist_labels
from .trajectory_v2 import FullTrajectory, RemainingBudgets, TrajectoryState

#: Re-exported for existing importers; the canonical definition now lives in
#: hydroswarm.planning.action_templates (core-issues3.txt Phase 3.5) so
#: model construction, runtime mapping, and checkpoint metadata all share
#: one source instead of each keeping their own mirror of this tuple.
__all__ = ["ACTION_TEMPLATES", "StrategistTrajectory", "StrategistTrajectoryStep", "build_strategist_trajectory"]


@dataclass(frozen=True, slots=True)
class StrategistTrajectoryStep:
    """One TrajectoryState paired with every plan label generated for it.

    `targets` is one targets_v2-governed strategist-category dict per label
    (aligned 1:1 with `labels`) -- each individually valid input to
    validate_targets_v2(); together they satisfy validate_targets_v2's
    PLAN_DIMENSION_TARGETS agreement check if stacked into batched tensors
    (all entries share the same "number of plans" leading dimension by
    construction, since they come from the same `labels` tuple).
    """

    state: TrajectoryState
    labels: tuple[StrategistLabel, ...]
    targets: tuple[dict[str, torch.Tensor], ...]


@dataclass(frozen=True, slots=True)
class StrategistTrajectory:
    trajectory: FullTrajectory
    steps: tuple[StrategistTrajectoryStep, ...]


def _resolve_target_pointer(
    label: StrategistLabel, node_ids: Sequence[str], edge_ids: Sequence[tuple[str, str]], network: Any
) -> tuple[int, bool]:
    """Resolve target_pointer against the CORRECT canonical index space for
    this label's target type (Phase 3.4 repair): node_ids for a NODE
    target, edge_ids for a LINK target -- never sorted(network.
    junction_name_list), which silently disagreed with the canonical node
    space (junctions + reservoirs + tanks) every other node-indexed target
    uses, and had no representation for link targets at all. targets_v2's
    DUAL_INDEX_TARGETS already documents that target_pointer's space is
    action_template-dependent; primary_target_type is exactly that
    disambiguator."""

    if label.primary_target_type == "NONE" or label.primary_target_id is None:
        return -1, False
    if label.primary_target_type == "NODE":
        if label.primary_target_id not in node_ids:
            return -1, False
        return node_ids.index(label.primary_target_id), True
    # LINK: edge_ids stores (start_node, end_node) pairs in
    # sorted(network.link_name_list) order (see hydroswarm.training.corpus.
    # scenario_to_example) -- reconstruct that same name->index mapping
    # rather than matching on endpoints, which is not unique for parallel
    # links.
    link_names = sorted(network.link_name_list)
    if label.primary_target_id not in link_names:
        return -1, False
    link_index = link_names.index(label.primary_target_id)
    if link_index >= len(edge_ids):
        return -1, False
    return link_index, True


def _strategist_label_targets(
    label: StrategistLabel, node_ids: Sequence[str], edge_ids: Sequence[tuple[str, str]], network: Any
) -> dict[str, torch.Tensor]:
    target_pointer, has_target = _resolve_target_pointer(label, node_ids, edge_ids, network)
    has_consequences = label.consequence_vector is not None
    has_plan_value = label.plan_value is not None

    def _proxy(value: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(value if value is not None else 0.0), torch.tensor(value is not None)

    exposure_proxy, exposure_proxy_mask = _proxy(label.exposure_proxy)
    pressure_risk_proxy, pressure_risk_proxy_mask = _proxy(label.pressure_risk_proxy)
    service_loss_proxy, service_loss_proxy_mask = _proxy(label.service_loss_proxy)
    containment_time_proxy, containment_time_proxy_mask = _proxy(label.containment_time_proxy)
    plan_regret_proxy, plan_regret_proxy_mask = _proxy(label.plan_regret_proxy)

    return {
        "action_template": torch.tensor(ACTION_TEMPLATE_INDEX[label.action_template]),
        "action_template_mask": torch.tensor(True),
        "target_pointer": torch.tensor(target_pointer),
        "target_pointer_mask": torch.tensor(has_target),
        # plan_validity is read only from PlanVerifier's own decision
        # (generate_strategist_labels never assigns it from a template's
        # predicted score) -- WNTR remains authoritative, per targets_v2's
        # plan_validity source_of_truth.
        "plan_validity": torch.tensor(int(label.plan_validity)),
        "plan_validity_mask": torch.tensor(True),
        # targets_v2's plan_value masking_rule: "undefined for invalid plans
        # without a computed consequence vector" -- label.plan_value is
        # None whenever generate_strategist_labels could not compute it
        # (invalid plan, or no comparator available), never a heuristic
        # fallback value (Phase 3.2 repair).
        "plan_value": torch.tensor(label.plan_value if has_plan_value else 0.0),
        "plan_value_mask": torch.tensor(has_plan_value),
        "consequence_vector": torch.tensor(label.consequence_vector if has_consequences else (0.0, 0.0, 0.0, 0.0)),
        "consequence_vector_mask": torch.tensor(has_consequences),
        # Phase 3.3 repair: these five proxy targets were defined in
        # targets_v2 but never populated by this module.
        "exposure_proxy": exposure_proxy,
        "exposure_proxy_mask": exposure_proxy_mask,
        "pressure_risk_proxy": pressure_risk_proxy,
        "pressure_risk_proxy_mask": pressure_risk_proxy_mask,
        "service_loss_proxy": service_loss_proxy,
        "service_loss_proxy_mask": service_loss_proxy_mask,
        "containment_time_proxy": containment_time_proxy,
        "containment_time_proxy_mask": containment_time_proxy_mask,
        "plan_regret_proxy": plan_regret_proxy,
        "plan_regret_proxy_mask": plan_regret_proxy_mask,
    }


def build_strategist_trajectory(
    scenario: GeneratedScenario,
    network: Any,
    feature_context: FeatureContext,
    artifact: SignatureArtifact,
    node_ids: Sequence[str],
    edge_ids: Sequence[tuple[str, str]] = (),
    *,
    trajectory_id: str | None = None,
    maximum_exact_simulations: int = 3,
    maximum_plans: int = 9,
) -> StrategistTrajectory:
    """Build a single-step trajectory: classify the incident's probable
    source nodes from its own observations (the same classical localizer
    the live pipeline uses), generate and exactly verify a bounded plan
    set against them, and package the result as a governed
    StrategistTrajectory.

    Deterministic for a fixed scenario/artifact (Phase 3's "Exact repeated
    fitting is deterministic" carries over from generate_strategist_labels'
    own determinism) -- trajectory_id/incident_id are uuid5-derived from
    the scenario id, not randomly generated.

    `node_ids` MUST be the canonical full topology node space (junctions +
    reservoirs + tanks), matching source_node_logits/sensor_fault_logits'
    own index space -- not sorted(network.junction_name_list). See
    scout_trajectory.build_scout_trajectory's docstring for the full
    rationale; full_trajectory.py's build_incident_trajectory derives
    node_ids from example.topology for exactly this reason. `edge_ids`
    (defaulting to empty for existing callers that don't yet thread it
    through) SHOULD be example.topology.edge_ids -- link-target plans
    (ISOLATE_SOURCE/ISOLATE_AND_FLUSH/ALTERNATE_VALVE_CUT) cannot resolve a
    target_pointer without it and are masked out (target_pointer_mask=False)
    when omitted, never silently misresolved against the node space.
    """

    scenario_id = str(scenario.manifest.scenario_id)
    trajectory_id = trajectory_id or str(
        uuid5(NAMESPACE_URL, f"https://hydroswarm.local/trajectories/strategist/{scenario_id}")
    )
    incident_id = uuid5(NAMESPACE_URL, f"https://hydroswarm.local/incidents/strategist/{scenario_id}")

    series = build_sensor_series(scenario, feature_context)
    observations, mask = HybridInferencePipeline._signature_observations(series, artifact)
    localization = localize_with_signatures(observations, artifact, observation_mask=mask)
    probable_nodes = HybridInferencePipeline._credible_nodes(localization.source_probabilities)
    context = HybridInferencePipeline._planning_context(
        incident_id, network, feature_context.graph, probable_nodes, frozenset()
    )
    #: important-issues.txt requirement 6: training-label exposure
    #: evaluation must come from the scenario's own exact ground-truth
    #: IncidentTruth, never from the classical localizer's inferred
    #: `probable_nodes` above (those exist only to build the deterministic
    #: plan-generation context -- which links/nodes are eligible action
    #: targets -- not to supply the source used for exposure consequences).
    labels = generate_strategist_labels(
        network,
        context,
        maximum_exact_simulations=maximum_exact_simulations,
        maximum_plans=maximum_plans,
        incident_truth=scenario.manifest.incident,
        is_contamination=scenario.manifest.event_type == EventType.CONTAMINATION.value,
    )
    targets = tuple(_strategist_label_targets(label, node_ids, edge_ids, network) for label in labels)

    incident_state = IncidentState(
        incident_id=incident_id,
        network_id=scenario.manifest.network_id,
        status="PLANNING",
        observations=(),
    )
    remaining = RemainingBudgets(
        samples=0, exact_simulations=max(0, maximum_exact_simulations - len(labels)), actions=0
    )
    state = TrajectoryState(step_index=0, incident_state=incident_state, remaining_budgets=remaining)
    trajectory = FullTrajectory(trajectory_id=trajectory_id, scenario_id=scenario_id, steps=(state,))
    step = StrategistTrajectoryStep(state=state, labels=labels, targets=targets)
    return StrategistTrajectory(trajectory=trajectory, steps=(step,))
