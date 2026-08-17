"""M10.2 Scout supervision/representation refit amendment, Part 4: real,
leakage-safe (INPUT, TARGET) pairing for genuine supervised Scout training.

Reuses, unmodified:
- `hydroswarm.training.scout_labels.generate_scout_label` -- the governed
  offline-truth Scout label generator (classical EIG over a real,
  exact-simulation-derived `SignatureArtifact`), already leakage-audited
  (module docstring: "computing a Scout label ... uses only that scenario's
  own observations plus a signature artifact built from exact network
  simulation").
- `hydroswarm.training.scout_trajectory.reveal_sample_measurement` -- the
  governed "genuinely new measurement, revealed only after this step's own
  decision" mechanism (Phase 5's leakage-cutoff design).
- `hydroswarm.training.scout_trajectory._scout_step_targets` -- the exact
  governed target-tensor construction (`sample_node`/`information_gain`/
  `candidate_reduction`/`should_continue_sampling`, correctly masked) an
  already-existing, already-tested function already produces; not
  reimplemented here.
- `hydroswarm.training.scout_training_state.build_scout_training_state_batch`
  (this amendment's own Part 3) for the paired model INPUT.

NOT reused: `scripts/train_scout_heads.py`'s corpus
(`data/learning-v2/cycle-b2-trajectories-v3/scout-tensors-normalized`) or
`scripts/run_stage_d_scout_policy_comparison.py`'s own ad hoc tensor
handling -- both predate the v5-causal causal-prefix architecture and the
frozen `hydroswarm.training.scout_state_contract` input-conditioning
contract this amendment implements; their SEMANTICS (what a Scout label
means, how EIG is computed) are still valid and reused via
`generate_scout_label` above, but their DATA/TENSOR representations are not.

## INPUT vs TARGET (explicit, per this amendment's Part 4 requirement)

INPUT (what `ScoutTrainingExample.batch` may contain): the current,
already-revealed evidence only -- the scenario's original sensors truncated
to the causal-prefix `depth`, plus any node this SAME trajectory has
already sampled in a STRICTLY earlier step (via
`hydroswarm.training.scout_training_state`'s synthetic-sensor overlay), plus
the current round/budget/accessibility/posterior state. Built exclusively
from information available at or before this decision.

TARGET (what `ScoutTrainingExample.targets` contains): `generate_scout_label`'s
offline classical-EIG recommendation, computed against the SAME
already-revealed evidence PLUS the full hypothesis-space `SignatureArtifact`
(a training-only asset, never a model input) -- i.e. "what a policy with
perfect access to the classical model's own full posterior would recommend
right now," not any FUTURE outcome. `ScoutTrainingExample.label` additionally
carries the offline diagnostics (per-candidate expected gain, ranked
alternatives) for reporting/inspection only -- never fed into
`ScoutTrainingExample.batch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from hydroswarm.classical.signatures import SignatureArtifact
from hydroswarm.sampling.active import SamplingConstraints
from hydroswarm.training.corpus import SignatureLibrary
from hydroswarm.training.scenario_reconstruction import ReconstructedScenarioContext, simulate_all_node_truth
from hydroswarm.training.scout_labels import ScoutLabel, generate_scout_label
from hydroswarm.training.scout_training_state import ScoutTrainingStateBatch, build_scout_training_state_batch
from hydroswarm.training.scout_trajectory import MAXIMUM_SAMPLES_BOUND, _scout_step_targets, reveal_sample_measurement

SCOUT_TARGET_SCHEMA_VERSION = "scout-target-v1"


@dataclass(frozen=True, slots=True)
class ScoutTrainingExample:
    step_index: int
    state: ScoutTrainingStateBatch
    targets: dict[str, torch.Tensor]
    #: Offline diagnostics ONLY (realized/offline classical-EIG label,
    #: per-candidate gain, stop reason) -- never present in `state.batch`.
    label: ScoutLabel
    already_sampled: tuple[str, ...]


def build_scout_training_examples(
    scenario: Any,
    network: Any,
    input_signature_library: SignatureLibrary,
    target_artifact: SignatureArtifact,
    node_ids: Sequence[str],
    depth: int,
    *,
    reconstruction: ReconstructedScenarioContext,
    maximum_samples: int = MAXIMUM_SAMPLES_BOUND,
    constraints: SamplingConstraints | None = None,
    noise_scale_mg_l: float = 0.5,
    feature_context: Any = None,
) -> list[ScoutTrainingExample]:
    """One scenario's full sequential Scout trajectory, as a list of
    (INPUT, TARGET) training examples -- one per decision step, terminating
    exactly like `hydroswarm.training.scout_trajectory.build_scout_trajectory`
    (no further recommendation, `should_continue_sampling=False`, or
    `maximum_samples` reached)."""

    if maximum_samples < 1:
        raise ValueError("maximum_samples must be at least 1")
    if maximum_samples > MAXIMUM_SAMPLES_BOUND:
        raise ValueError(f"maximum_samples ({maximum_samples}) exceeds the {MAXIMUM_SAMPLES_BOUND} product bound")

    all_node_truth = simulate_all_node_truth(reconstruction)
    revealed_samples: dict[str, float] = {}
    already_sampled: list[str] = []
    examples: list[ScoutTrainingExample] = []
    step_index = 0
    while True:
        # Same leakage-cutoff assertion build_scout_trajectory itself makes:
        # this step's decision may only use evidence revealed by STRICTLY
        # earlier steps.
        assert not (already_sampled and already_sampled[-1] not in revealed_samples), (
            "the immediately preceding step's sample must be revealed before this step's decision"
        )
        label = generate_scout_label(
            scenario, target_artifact, already_sampled=already_sampled, constraints=constraints,
            noise_scale_mg_l=noise_scale_mg_l, revealed_samples=revealed_samples,
        )
        state = build_scout_training_state_batch(
            scenario, network, input_signature_library, depth,
            already_sampled=already_sampled, revealed_values_mg_l=dict(revealed_samples),
            sampling_round=step_index, sample_budget_total=maximum_samples,
            feature_context=feature_context,
        )
        targets = _scout_step_targets(label, list(node_ids))
        examples.append(
            ScoutTrainingExample(
                step_index=step_index, state=state, targets=targets, label=label,
                already_sampled=tuple(already_sampled),
            )
        )
        if label.sample_node_id is None or not label.should_continue_sampling:
            break
        would_have_sampled = len(already_sampled) + 1
        already_sampled.append(label.sample_node_id)
        revealed_samples[label.sample_node_id] = reveal_sample_measurement(
            reconstruction, all_node_truth, label.sample_node_id, step_index, noise_scale_mg_l=noise_scale_mg_l,
        )
        if would_have_sampled >= maximum_samples:
            break
        step_index += 1
    return examples
