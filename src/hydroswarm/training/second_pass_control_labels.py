"""Second-pass calibrated control-label generation (core-issues3.txt Phase 8).

corpus.py's `_evidence_sufficiency` (used to train the Stage-A checkpoint
itself, via `control_labels.classify_evidence_sufficiency`) implements only
the sensor-health/posterior-entropy/OOD-category subset of targets_v2's
governed `evidence_sufficiency` definition, documented there as an accepted
interim state: calibrated candidate-set size, candidate coverage, and
classical-neural disagreement all require a CalibrationArtifact fit against
a *trained* checkpoint, which does not exist until after Stage-A training
completes. This module is that second pass, run once a Stage-A checkpoint
and its calibration artifact exist (core-issues3.txt Phase 8 steps 1-3).

Circular self-label leakage (item 7): every label this module produces
carries `teacher_checkpoint_hash` (the frozen model's own fingerprint, via
the same `HybridInferencePipeline._fingerprint_model` convention Stage 3's
calibration fitting already uses). A caller training a NEW checkpoint from
these labels must record that teacher hash and must never silently
regenerate labels from the checkpoint currently being optimized -- this
module itself has no opinion on that; it is enforced by generate_
second_pass_control_labels raising if `model.training` is True (a frozen,
`.eval()`-mode model is the only thing this module will accept), which
catches the most common accidental version of the mistake (forgetting to
freeze/eval the model before using it as a label-generation teacher) even
though it cannot detect "same weights, different Python object" by itself.

Deliberately does NOT fold the model's own evidence_sufficiency head
output into this label (unlike hydroswarm.inference.pipeline's live
`evidence_sufficient` runtime decision, which does: `calibrated and
0 < len(conformal_nodes) <= maximum_planning_candidates and model_evidence`
-- see pipeline.py). Using a checkpoint's own current prediction as part of
the training label for that SAME target is the specific circularity item 7
warns against; the live pipeline can safely use it because that's an
operational decision, not something being fed back into further training of
the same head. The signals used here (calibrated candidate-set size,
candidate coverage, posterior entropy of the fused hybrid probability
vector, classical-neural disagreement, sensor health, OOD category,
calibration validity) are otherwise the same governed inputs, computed from
the frozen model's raw output plus the classical prior -- not from the
model's own evidence_sufficiency prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from hydroswarm.calibration.conformal import SplitConformalCalibrator
from hydroswarm.classical.metrics import entropy
from hydroswarm.inference.fusion import fixed_weight_fusion, jensen_shannon_divergence
from hydroswarm.model import HydroCore

from .control_labels import (
    DEFAULT_ENTROPY_THRESHOLD_BITS,
    MAXIMUM_SAMPLING_ROUNDS,
    classify_next_step,
)
from .data import ScenarioDatasetView
from .ood_categories import topology_calibration_is_valid
from .targets_v2 import EventCause, NextStep
from .variable_collate import collate_variable_topology

#: Same fusion weighting Stage 3's calibration fitting used (core-issues.txt
#: repair item 10) -- the second pass must evaluate against the identical
#: fused hybrid probability vector the calibrator was actually fit on, not
#: raw neural output, or its own candidate-set/coverage numbers would
#: silently disagree with the calibration artifact's own measured report.
SECOND_PASS_FUSION_NEURAL_WEIGHT = 0.6

#: Mirrors hydroswarm.inference.pipeline's maximum_planning_candidates
#: default -- a calibrated candidate set wider than this is "too broad to
#: plan against" there; reused here as the same governed bound for what
#: counts as a sufficiently narrow calibrated candidate set.
DEFAULT_MAXIMUM_CANDIDATE_SET_SIZE = 3

#: Mirrors hydroswarm.inference.fusion.uncertainty_control's
#: disagreement_js >= 0.5 threshold for flagging high classical/neural
#: disagreement (there, triggering ControlAction.INSPECT_SENSORS).
DEFAULT_DISAGREEMENT_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class SecondPassControlLabel:
    scenario_id: str
    calibrated_candidate_set_size: int
    #: None when no source ground truth is available for this example
    #: (event_presence is false) -- "coverage" is undefined, not False.
    candidate_covered: bool | None
    posterior_entropy_bits: float
    classical_neural_disagreement_js: float
    calibration_valid: bool
    evidence_sufficiency: bool
    next_step: NextStep
    teacher_checkpoint_hash: str


def classify_evidence_sufficiency_second_pass(
    *,
    calibrated_candidate_set_size: int,
    calibration_valid: bool,
    posterior_entropy_bits: float,
    disagreement_js: float,
    healthy_fraction: float,
    sensors_ever_healthy: int,
    health_threshold: float = 0.5,
    minimum_healthy_sensors: int = 2,
    entropy_threshold_bits: float = DEFAULT_ENTROPY_THRESHOLD_BITS,
    maximum_candidate_set_size: int = DEFAULT_MAXIMUM_CANDIDATE_SET_SIZE,
    disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> bool:
    """The full governed evidence_sufficiency rule (core-issues3.txt Phase 8
    step 4), extending control_labels.classify_evidence_sufficiency's
    sensor-health/entropy/OOD-validity rule with the two signals that
    require a trained checkpoint + calibration artifact: a narrow,
    non-empty calibrated candidate set, and low classical-neural
    disagreement. All conditions must hold -- any one failing means the
    evidence is not trustworthy enough to plan against, matching the
    fail-closed framing the first-pass rule already established."""

    if calibrated_candidate_set_size <= 0 or calibrated_candidate_set_size > maximum_candidate_set_size:
        return False
    if not calibration_valid:
        return False
    if disagreement_js >= disagreement_threshold:
        return False
    if healthy_fraction < health_threshold or sensors_ever_healthy < minimum_healthy_sensors:
        return False
    return posterior_entropy_bits <= entropy_threshold_bits


def generate_second_pass_control_labels(
    model: HydroCore,
    dataset: ScenarioDatasetView,
    calibrator: SplitConformalCalibrator,
    *,
    teacher_checkpoint_hash: str,
    validated_topology_hashes: object,
    batch_size: int = 16,
    fusion_neural_weight: float = SECOND_PASS_FUSION_NEURAL_WEIGHT,
    maximum_candidate_set_size: int = DEFAULT_MAXIMUM_CANDIDATE_SET_SIZE,
    disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> Iterator[SecondPassControlLabel]:
    """Run `model` (frozen) over `dataset` in batches and yield one governed
    SecondPassControlLabel per example.

    Raises ValueError if `model` is not in eval mode (`model.training` is
    True) -- the caller must freeze/`.eval()` the teacher checkpoint before
    it can be used to generate labels for anything, including a fresh
    training run of the same architecture (see module docstring, item 7).

    Only ever materializes one batch's worth of examples at a time (the
    same lazy-dataset discipline `_predict_rows` in
    scripts/run_stage3_finalist_training.py already establishes), so this
    is safe to run over a full train/validation split without holding the
    whole corpus resident.
    """

    if model.training:
        raise ValueError(
            "model must be frozen (call model.eval()) before generating second-pass "
            "control labels -- passing a model still in training mode risks "
            "circular self-label leakage (core-issues3.txt Phase 8 item 7)"
        )

    total = len(dataset)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_examples = [dataset[index] for index in range(start, min(start + batch_size, total))]
            inputs, targets = collate_variable_topology(batch_examples)
            output = model(inputs)
            neural_probabilities = torch.softmax(output["source_node_logits"], dim=-1).numpy()
            classical_prior = inputs["classical_prior"].numpy()
            source_mask = targets.get("source_node_mask")
            for row, example in enumerate(batch_examples):
                fused = fixed_weight_fusion(
                    classical_prior[row], neural_probabilities[row], neural_weight=fusion_neural_weight
                )
                topology = example.topology
                topology_hash = topology.topology_hash if topology is not None else ""
                calibration_valid = topology_calibration_is_valid(
                    topology_hash, validated_topology_hashes
                )
                candidate_indices = calibrator.candidate_set(
                    fused, condition=example.stage.name, network_id=example.network_id
                )
                disagreement_js = jensen_shannon_divergence(neural_probabilities[row], classical_prior[row])
                posterior_entropy_bits = entropy(fused)

                has_source = source_mask is None or bool(source_mask[row])
                candidate_covered = (
                    bool(int(targets["source_node"][row].item()) in candidate_indices) if has_source else None
                )

                effective_candidate_set_size = len(candidate_indices) if calibration_valid else 0
                sufficiency = classify_evidence_sufficiency_second_pass(
                    calibrated_candidate_set_size=effective_candidate_set_size,
                    calibration_valid=calibration_valid,
                    posterior_entropy_bits=posterior_entropy_bits,
                    disagreement_js=disagreement_js,
                    healthy_fraction=_healthy_fraction(inputs, row),
                    sensors_ever_healthy=_sensors_ever_healthy(inputs, row),
                    maximum_candidate_set_size=maximum_candidate_set_size,
                    disagreement_threshold=disagreement_threshold,
                )
                event_cause = _event_cause_from_targets(targets, row)
                step = classify_next_step(
                    ood_level_outside_validated_range=not calibration_valid,
                    evidence_sufficient=sufficiency,
                    sample_count=0,
                    event_cause=event_cause,
                    maximum_sampling_rounds=MAXIMUM_SAMPLING_ROUNDS,
                )

                yield SecondPassControlLabel(
                    scenario_id=example.scenario_id,
                    calibrated_candidate_set_size=effective_candidate_set_size,
                    candidate_covered=candidate_covered,
                    posterior_entropy_bits=posterior_entropy_bits,
                    classical_neural_disagreement_js=disagreement_js,
                    calibration_valid=calibration_valid,
                    evidence_sufficiency=sufficiency,
                    next_step=step,
                    teacher_checkpoint_hash=teacher_checkpoint_hash,
                )


#: quality_features' channel 0 is "health" (preprocessing/builder.py's
#: `quality[time_index, node_index] = [series.health[...], missing, drift,
#: age]`) -- both helpers below read this exact channel.
_HEALTH_CHANNEL = 0
_HEALTH_THRESHOLD = 0.75


def _healthy_fraction(inputs: dict[str, torch.Tensor], row: int) -> float:
    """Reconstructs corpus.sensor_health_summary's healthy_fraction signal
    (fraction of valid (sensor, time) observations at or above the health
    threshold) from the already-collated batch's quality_features
    [B, T, N, 4] / sensor_mask [B, T, N] -- the second pass reads it here
    rather than re-deriving it from a GeneratedScenario, since only the
    collated tensors are available at this point in the pipeline."""

    quality = inputs.get("quality_features")
    sensor_mask = inputs.get("sensor_mask")
    if quality is None or sensor_mask is None:
        return 0.0
    mask = sensor_mask[row].bool()  # [T, N]
    if not bool(mask.any()):
        return 0.0
    health = quality[row, ..., _HEALTH_CHANNEL]  # [T, N]
    valid_health = health[mask]
    if valid_health.numel() == 0:
        return 0.0
    return float((valid_health >= _HEALTH_THRESHOLD).float().mean())


def _sensors_ever_healthy(inputs: dict[str, torch.Tensor], row: int) -> int:
    """Count of distinct sensor nodes healthy at or above the threshold at
    at least one valid timestep."""

    quality = inputs.get("quality_features")
    sensor_mask = inputs.get("sensor_mask")
    if quality is None or sensor_mask is None:
        return 0
    mask = sensor_mask[row].bool()  # [T, N]
    health = quality[row, ..., _HEALTH_CHANNEL]  # [T, N]
    ever_healthy_per_node = ((health >= _HEALTH_THRESHOLD) & mask).any(dim=0)  # [N]
    return int(ever_healthy_per_node.sum())


def _event_cause_from_targets(targets: dict[str, torch.Tensor], row: int) -> EventCause:
    cause_index = targets.get("event_cause")
    if cause_index is None:
        return EventCause.NORMAL
    return list(EventCause)[int(cause_index[row].item())]
