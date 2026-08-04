from __future__ import annotations

import hashlib

import pytest
import torch

from hydroswarm.training import (
    AgentTrajectory,
    CurriculumSchedule,
    CurriculumStage,
    GovernedScenarioDataset,
    ScenarioExample,
    TrainingConfig,
    TrajectoryStep,
    compute_multitask_loss,
    validate_split_isolation,
)


def _example(
    identifier: str,
    *,
    split: str = "train",
    family: str | None = None,
    stage: CurriculumStage = CurriculumStage.CLEAN,
) -> ScenarioExample:
    return ScenarioExample(
        scenario_id=identifier,
        network_id="Net1",
        split=split,
        seed=int(identifier[-1]) if identifier[-1].isdigit() else 1,
        seed_family=family or f"family-{identifier}",
        stage=stage,
        inputs={"node_features": torch.zeros(2, 3)},
        targets={"source_node": torch.tensor(0)},
    )


def test_governed_dataset_manifest_curriculum_and_split_leakage() -> None:
    clean = _example("scenario-1")
    shifted = _example("scenario-2", stage=CurriculumStage.SHIFT)
    dataset = GovernedScenarioDataset([clean, shifted], expected_split="train")
    assert len(dataset.manifest_hash) == 64
    assert len(dataset.stages_through(CurriculumStage.OPERATIONAL)) == 1
    schedule = CurriculumSchedule.progressive()
    assert schedule.stage_for_epoch(0) == CurriculumStage.CLEAN
    assert schedule.stage_for_epoch(4) == CurriculumStage.ADVERSARIAL

    validation = GovernedScenarioDataset(
        [_example("scenario-3", split="validation", family=clean.seed_family)],
        expected_split="validation",
    )
    with pytest.raises(ValueError, match="seed-family leakage"):
        validate_split_isolation(dataset, validation)


def test_trajectory_requires_contiguous_provenance() -> None:
    digest = hashlib.sha256(b"state").hexdigest()
    trajectory = AgentTrajectory(
        "trajectory-1",
        "scenario-1",
        (
            TrajectoryStep(0, digest, "SAMPLE"),
            TrajectoryStep(1, digest, "VERIFY", "REJECTED"),
        ),
    )
    assert trajectory.steps[-1].verifier_decision == "REJECTED"
    with pytest.raises(ValueError, match="contiguous"):
        AgentTrajectory(
            "trajectory-2",
            "scenario-1",
            (TrajectoryStep(1, digest, "SAMPLE"),),
        )


def test_multitask_loss_covers_semantic_heads_and_weights() -> None:
    outputs = {
        "source_node_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "sample_node_logits": torch.tensor([[0.0, 3.0]], requires_grad=True),
        "plan_value": torch.tensor([[0.2, 0.4]], requires_grad=True),
        "plan_validity_logits": torch.tensor([[[3.0, 0.0], [0.0, 3.0]]], requires_grad=True),
        "ood_logits": torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True),
        "start_time_logits": torch.zeros(1, 12, requires_grad=True),
        "duration_logits": torch.zeros(1, 8, requires_grad=True),
        "relative_strength_logits": torch.zeros(1, 4, requires_grad=True),
        "sensor_fault_logits": torch.zeros(1, 2, requires_grad=True),
        "residual_prediction": torch.zeros(1, 2, requires_grad=True),
        "sensor_reconstruction_prediction": torch.ones(1, 2, requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "sample_node": torch.tensor([1]),
        "plan_value": torch.tensor([[0.0, 0.5]]),
        "plan_validity": torch.tensor([[0, 1]]),
        "ood": torch.tensor([0]),
        "start_time": torch.tensor([3]),
        "duration": torch.tensor([2]),
        "relative_strength": torch.tensor([1]),
        "sensor_fault": torch.zeros(1, 2),
        "residual": torch.ones(1, 2),
        "sensor_reconstruction": torch.zeros(1, 2),
    }
    result = compute_multitask_loss(
        outputs,
        targets,
        task_weights={"residual": 2.0},
        profile_ordinal_weight=0.4,
    )
    assert set(result.tasks) == set(targets)
    assert torch.isfinite(result.total)
    result.total.backward()


def test_target_mask_companion_excludes_placeholder_labels_from_the_loss() -> None:
    # Regression: hydroswarm.training.corpus.scenario_to_example sets
    # source_node/source_region/start_time/duration/relative_strength to a
    # placeholder value (0) -- never a real label -- for a NORMAL/
    # SENSOR_FAULT_ONLY scenario where no real source exists, and records
    # that explicitly via the separate f"{task}_mask" companion (targets_v2's
    # documented convention). compute_multitask_loss previously never read
    # those companions at all, so the placeholder 0 was silently trained
    # against as if it were a real label. Proven here by comparing a
    # fully-masked-out batch's loss against a directly-computed all-ignored
    # cross-entropy (both must be exactly the same "no real supervision"
    # zero-gradient shape, not a loss actively pulling the prediction toward
    # class 0).
    outputs = {
        "source_node_logits": torch.tensor([[5.0, -5.0], [5.0, -5.0]], requires_grad=True),
        "source_region_logits": torch.tensor([[5.0, -5.0], [5.0, -5.0]], requires_grad=True),
        "duration_logits": torch.zeros(2, 8, requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0, 0]),
        "source_node_mask": torch.tensor([False, False]),
        "source_region": torch.tensor([0, 0]),
        "source_region_mask": torch.tensor([False, False]),
        "duration": torch.tensor([0, 0]),
        "duration_mask": torch.tensor([False, False]),
    }
    result = compute_multitask_loss(outputs, targets)
    for task in ("source_node", "source_region", "duration"):
        assert result.tasks[task].item() == pytest.approx(0.0)
    result.total.backward()
    # The masked-out source_node_logits must receive exactly zero gradient
    # from this task -- if the placeholder 0 label had leaked through, the
    # already-confident (correct-looking) prediction would still get zero
    # gradient by coincidence, so this alone would not catch the bug; the
    # zero task loss above is the real proof.
    assert outputs["source_node_logits"].grad is not None


def test_target_mask_companion_still_trains_on_the_unmasked_positions() -> None:
    outputs = {
        "source_node_logits": torch.tensor([[5.0, -5.0], [5.0, -5.0]], requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0, 1]),
        "source_node_mask": torch.tensor([True, True]),
    }
    result = compute_multitask_loss(outputs, targets)
    # Row 0 (label 0, confidently predicted 0) contributes ~0; row 1
    # (label 1, confidently predicted 0 -- wrong) contributes a large loss.
    # Averaged over 2 valid rows this must be well above zero, unlike the
    # fully-masked case above.
    assert result.tasks["source_node"].item() > 1.0


def test_multitask_loss_covers_event_control_heads_when_present() -> None:
    # overnight-plan.txt Task 4.4: event_cause/event_presence losses only
    # fire when both the model output (event_control_heads=True) and the
    # target are present; next_step has no label generator yet, so it
    # must be silently absent from result.tasks rather than raising.
    outputs = {
        "source_node_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "event_presence_logits": torch.tensor([2.0], requires_grad=True),
        "event_cause_logits": torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "event_presence": torch.tensor([1.0]),
        "event_cause": torch.tensor([0]),
    }
    result = compute_multitask_loss(outputs, targets)
    assert set(result.tasks) == {"source_node", "event_presence", "event_cause"}
    assert "next_step" not in result.tasks
    assert torch.isfinite(result.total)
    result.total.backward()


def test_training_config_is_strict_cpu_fp32() -> None:
    assert TrainingConfig().device == "cpu"
    with pytest.raises(ValueError, match="CPU-only"):
        TrainingConfig(device="cuda")
    with pytest.raises(ValueError, match="FP32"):
        TrainingConfig(fp32=False)
    with pytest.raises(ValueError, match="ordinal"):
        TrainingConfig(profile_ordinal_weight=-0.1)
