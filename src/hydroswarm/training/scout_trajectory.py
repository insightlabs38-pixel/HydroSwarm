"""Scout trajectory-state generation (core-issues2.txt Phase 2).

Wires the already-existing scout_labels.generate_scout_label() into the
already-existing TrajectoryState/FullTrajectory container (trajectory_v2.py)
-- exactly the missing piece both modules' own docstrings describe: "a
future corpus builder wires generate_scout_label() into that trajectory
structure, called once per sampling step with an updated already_sampled
set" / "supervised Scout training can see the actual candidate set and
observation history a sample recommendation was conditioned on."

Each step's governed Scout targets (targets_v2's scout category:
sample_node, information_gain, candidate_reduction, should_continue_sampling)
are recorded alongside its TrajectoryState in a companion dict rather than
folded into IncidentState, whose schema is deliberately runtime/API-shaped
and has no Scout-specific fields (information_gain, candidate_reduction,
per-candidate accessibility).

Known simplification, documented rather than hidden: generate_scout_label
always evaluates the *same* base observations (the scenario's own,
already-simulated evidence) at every step -- "already_sampled" changes
which candidates are excluded from consideration, not what the underlying
evidence looks like. This module does not simulate a genuinely new
concentration reading being revealed after a sample is taken; that fuller
closed-loop notion belongs to Phase 7's full trajectory corpus (Sentinel ->
... -> Scout -> simulated next observation -> posterior update -> ...),
which does not exist yet either. Phase 2's own spec is satisfied as
written: "For every Scout state: enumerate accessible, unsampled candidate
nodes... record sample_node/information_gain/candidate_reduction/
should_continue_sampling/accessibility and already-sampled masks."
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

import torch

from hydroswarm.classical.signatures import SignatureArtifact
from hydroswarm.data.scenarios import GeneratedScenario
from hydroswarm.domain import ActionType, IncidentState, OperationalAction
from hydroswarm.sampling.active import SamplingConstraints

from .scout_labels import ScoutLabel, generate_scout_label
from .trajectory_v2 import FullTrajectory, RemainingBudgets, TrajectoryState

#: Fixed reference epoch trajectory timestamps would be computed against if
#: needed -- matches hydroswarm.evaluation.golden's TIMESTAMP convention.
#: Scenario timestamps are relative seconds-from-simulation-start; absolute
#: wall time carries no scientific meaning here.
TRAJECTORY_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

#: hydroswarm.domain.schemas.IncidentState.sample_count is bounded (le=5),
#: matching IncidentCreate.maximum_samples' product-level sample cap.
MAXIMUM_SAMPLES_BOUND = 5


@dataclass(frozen=True, slots=True)
class ScoutTrajectoryStep:
    """One TrajectoryState paired with the ScoutLabel it was built from.

    `targets` is strictly targets_v2's governed scout-category subset
    (sample_node, information_gain, candidate_reduction,
    should_continue_sampling, plus each maskable one's `_mask` companion) --
    it is a valid input to validate_targets_v2() as-is, with no extra keys.
    `diagnostics` carries the rest of Phase 2 item 3's requirement list that
    targets_v2 does not itself define a governed field for ("per-node
    expected information gain where supported", per-candidate
    accessibility) -- useful for analysis/debugging, never trained against.
    """

    state: TrajectoryState
    label: ScoutLabel
    targets: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScoutTrajectory:
    trajectory: FullTrajectory
    steps: tuple[ScoutTrajectoryStep, ...]


def _scout_step_targets(label: ScoutLabel, node_ids: Sequence[str]) -> dict[str, torch.Tensor]:
    # torch.Tensor-valued, matching hydroswarm.training.corpus.scenario_to_example's
    # convention and validate_targets_v2's Mapping[str, Tensor] contract.
    has_recommendation = label.sample_node_id is not None
    return {
        "sample_node": torch.tensor(node_ids.index(label.sample_node_id) if has_recommendation else -1),
        "sample_node_mask": torch.tensor(has_recommendation),
        "information_gain": torch.tensor(label.information_gain_bits),
        "information_gain_mask": torch.tensor(has_recommendation),
        "candidate_reduction": torch.tensor(label.candidate_reduction_fraction),
        "candidate_reduction_mask": torch.tensor(has_recommendation),
        "should_continue_sampling": torch.tensor(label.should_continue_sampling),
    }


def _scout_step_diagnostics(label: ScoutLabel, already_sampled: Sequence[str]) -> dict[str, Any]:
    return {
        "per_node_information_gain": {
            candidate.node_id: candidate.expected_information_gain_bits for candidate in label.candidates
        },
        "accessible": {candidate.node_id: candidate.accessible for candidate in label.candidates},
        "already_sampled": tuple(already_sampled),
    }


def _chain_states(states: Sequence[TrajectoryState]) -> tuple[TrajectoryState, ...]:
    """Set each step's resulting_next_state_hash to the following step's
    actual state_hash, satisfying FullTrajectory's hash-chain integrity
    check. Safe to compute the next state's hash before chaining it in turn
    -- state_hash excludes resulting_next_state_hash from its own digest, so
    it never depends on what a predecessor is about to point at."""

    if len(states) <= 1:
        return tuple(states)
    chained: list[TrajectoryState] = []
    for index, state in enumerate(states):
        if index == len(states) - 1:
            chained.append(state)
        else:
            chained.append(dataclasses.replace(state, resulting_next_state_hash=states[index + 1].state_hash))
    return tuple(chained)


def build_scout_trajectory(
    scenario: GeneratedScenario,
    artifact: SignatureArtifact,
    node_ids: Sequence[str],
    *,
    trajectory_id: str | None = None,
    maximum_samples: int = MAXIMUM_SAMPLES_BOUND,
    constraints: SamplingConstraints | None = None,
    noise_scale_mg_l: float = 0.5,
) -> ScoutTrajectory:
    """Repeatedly call generate_scout_label with a growing already_sampled
    set, terminating when should_continue_sampling is false, no candidate is
    recommended, or maximum_samples is reached (Phase 2's "mask examples
    where ... the sampling budget is exhausted"). Packages each step into a
    hash-chained FullTrajectory plus its governed Scout targets.

    Deterministic for a fixed scenario/artifact/constraints (Phase 2's
    "Sample labels are deterministic under a fixed seed") -- every source of
    randomness here is scenario.manifest.seed-derived upstream, and
    trajectory_id/incident_id are uuid5-derived from the scenario id rather
    than randomly generated.
    """

    if maximum_samples < 1:
        raise ValueError("maximum_samples must be at least 1")
    if maximum_samples > MAXIMUM_SAMPLES_BOUND:
        raise ValueError(
            f"maximum_samples ({maximum_samples}) exceeds IncidentState.sample_count's "
            f"product-level bound ({MAXIMUM_SAMPLES_BOUND})"
        )

    scenario_id = str(scenario.manifest.scenario_id)
    trajectory_id = trajectory_id or str(
        uuid5(NAMESPACE_URL, f"https://hydroswarm.local/trajectories/scout/{scenario_id}")
    )
    incident_id = uuid5(NAMESPACE_URL, f"https://hydroswarm.local/incidents/scout/{scenario_id}")

    already_sampled: list[str] = []
    labels: list[ScoutLabel] = []
    states: list[TrajectoryState] = []
    diagnostics: list[dict[str, Any]] = []
    step_index = 0
    while True:
        label = generate_scout_label(
            scenario,
            artifact,
            already_sampled=already_sampled,
            constraints=constraints,
            noise_scale_mg_l=noise_scale_mg_l,
        )
        incident_state = IncidentState(
            incident_id=incident_id,
            network_id=scenario.manifest.network_id,
            status="SAMPLING",
            observations=(),
            sample_count=len(already_sampled),
        )
        remaining = RemainingBudgets(
            samples=max(0, maximum_samples - len(already_sampled)), exact_simulations=0, actions=0
        )
        state = TrajectoryState(
            step_index=step_index,
            incident_state=incident_state,
            remaining_budgets=remaining,
            previous_action=(
                OperationalAction(action_type=ActionType.COLLECT_SAMPLE, target_id=already_sampled[-1])
                if already_sampled
                else None
            ),
            selected_next_action=(
                OperationalAction(action_type=ActionType.COLLECT_SAMPLE, target_id=label.sample_node_id)
                if label.sample_node_id is not None
                else None
            ),
        )
        labels.append(label)
        states.append(state)
        diagnostics.append(_scout_step_diagnostics(label, already_sampled))

        if label.sample_node_id is None or not label.should_continue_sampling:
            break
        # Budget check uses the count *after* this step's own recommendation
        # is accounted for -- maximum_samples=1 must stop after the single
        # step that requests a sample, not run one further step past it.
        would_have_sampled = len(already_sampled) + 1
        already_sampled.append(label.sample_node_id)
        if would_have_sampled >= maximum_samples:
            break
        step_index += 1

    chained_states = _chain_states(states)
    trajectory = FullTrajectory(trajectory_id=trajectory_id, scenario_id=scenario_id, steps=chained_states)
    steps = tuple(
        ScoutTrajectoryStep(
            state=state, label=label, targets=_scout_step_targets(label, node_ids), diagnostics=step_diagnostics
        )
        for state, label, step_diagnostics in zip(chained_states, labels, diagnostics)
    )
    return ScoutTrajectory(trajectory=trajectory, steps=steps)
