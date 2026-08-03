from __future__ import annotations

import hashlib

import pytest
import torch

from hydroswarm.model import HydroCore
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
        "sensor_fault_logits": torch.zeros(1, 2, requires_grad=True),
        "residual_prediction": torch.zeros(1, 2, requires_grad=True),
        "reconstruction_prediction": torch.ones(1, 2, requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "sample_node": torch.tensor([1]),
        "plan_value": torch.tensor([[0.0, 0.5]]),
        "plan_validity": torch.tensor([[0, 1]]),
        "ood": torch.tensor([0]),
        "sensor_fault": torch.zeros(1, 2),
        "residual": torch.ones(1, 2),
        "reconstruction": torch.zeros(1, 2),
    }
    result = compute_multitask_loss(outputs, targets, task_weights={"residual": 2.0})
    assert set(result.tasks) == set(targets)
    assert torch.isfinite(result.total)
    result.total.backward()


def test_training_config_is_strict_cpu_fp32() -> None:
    assert TrainingConfig().device == "cpu"
    with pytest.raises(ValueError, match="CPU-only"):
        TrainingConfig(device="cuda")
    with pytest.raises(ValueError, match="FP32"):
        TrainingConfig(fp32=False)

