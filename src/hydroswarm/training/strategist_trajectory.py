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
from hydroswarm.data.scenarios import GeneratedScenario
from hydroswarm.domain import IncidentState
from hydroswarm.inference.pipeline import HybridInferencePipeline

from .corpus import FeatureContext, build_sensor_series
from .strategist_labels import StrategistLabel, generate_strategist_labels
from .trajectory_v2 import FullTrajectory, RemainingBudgets, TrajectoryState

#: The bounded deterministic template set hydroswarm.planning.response.
#: generate_response_plans() actually produces, in its own append order --
#: the canonical action_template categorical index. NOTE: this is 9
#: templates, but hydroswarm.model.core.HydroCore's action_head defaults to
#: action_vocabulary_size=8 -- a real, previously-undiscovered class-count
#: mismatch this module's own construction surfaced (targets_v2.py
#: deliberately excludes action_template from its fixed TARGET_CLASS_COUNTS
#: for exactly this reason: the head's width is a configurable constructor
#: argument, not a fixed constant). Any future Strategist-enabled training
#: run MUST pass action_vocabulary_size=len(ACTION_TEMPLATES) explicitly --
#: the current bare default is insufficient and would misclassify or error
#: on the last template. Flagged here rather than changed unilaterally: a
#: model-architecture default change affects checkpoint-loading
#: compatibility (core-issues.txt Task 4.0) and is out of this label-
#: generation module's scope.
ACTION_TEMPLATES = (
    "NO_ACTION",
    "ISOLATE_SOURCE",
    "FLUSH_DOWNSTREAM",
    "ISOLATE_AND_FLUSH",
    "PROTECT_CRITICAL",
    "INCREASE_MONITORING",
    "REQUEST_SAMPLE",
    "WAIT_OBSERVE",
    "ALTERNATE_VALVE_CUT",
)
_ACTION_TEMPLATE_INDEX = {name: index for index, name in enumerate(ACTION_TEMPLATES)}


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


def _strategist_label_targets(label: StrategistLabel, node_ids: Sequence[str]) -> dict[str, torch.Tensor]:
    has_target = label.target_node_index is not None
    has_consequences = label.consequence_vector is not None
    return {
        "action_template": torch.tensor(_ACTION_TEMPLATE_INDEX[label.action_template]),
        "action_template_mask": torch.tensor(True),
        "target_pointer": torch.tensor(label.target_node_index if has_target else -1),
        "target_pointer_mask": torch.tensor(has_target),
        # plan_validity is read only from PlanVerifier's own decision
        # (generate_strategist_labels never assigns it from a template's
        # predicted score) -- WNTR remains authoritative, per targets_v2's
        # plan_validity source_of_truth.
        "plan_validity": torch.tensor(int(label.plan_validity)),
        "plan_validity_mask": torch.tensor(True),
        # targets_v2's plan_value masking_rule: "undefined for invalid plans
        # without a computed consequence vector" -- label.plan_value is
        # always numerically present (it's the template's own predicted
        # score), but the governed target is masked out wherever the
        # schema declares it undefined.
        "plan_value": torch.tensor(label.plan_value),
        "plan_value_mask": torch.tensor(has_consequences),
        "consequence_vector": torch.tensor(label.consequence_vector if has_consequences else (0.0, 0.0, 0.0, 0.0)),
        "consequence_vector_mask": torch.tensor(has_consequences),
    }


def build_strategist_trajectory(
    scenario: GeneratedScenario,
    network: Any,
    feature_context: FeatureContext,
    artifact: SignatureArtifact,
    node_ids: Sequence[str],
    *,
    trajectory_id: str | None = None,
    maximum_exact_simulations: int = 3,
    maximum_plans: int = 8,
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
    node_ids from example.topology for exactly this reason.
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
    labels = generate_strategist_labels(
        network, context, maximum_exact_simulations=maximum_exact_simulations, maximum_plans=maximum_plans
    )
    targets = tuple(_strategist_label_targets(label, node_ids) for label in labels)

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
