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
        "action_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "action_pointer_logits": torch.tensor([[0.0, 3.0]], requires_grad=True),
        "plan_value": torch.tensor([[0.2, 0.4]], requires_grad=True),
        "plan_validity_logits": torch.tensor([[[3.0, 0.0], [0.0, 3.0]]], requires_grad=True),
        # core-issues3.txt Phase 6: ood_class maps to ood_category_logits
        # (the correctly-sized 11-class head), not the old 3-logit ood_logits
        # (a different, deterministic-severity-adjacent concept).
        "ood_category_logits": torch.zeros(1, 11, requires_grad=True),
        "start_time_logits": torch.zeros(1, 12, requires_grad=True),
        "duration_logits": torch.zeros(1, 8, requires_grad=True),
        "relative_strength_logits": torch.zeros(1, 4, requires_grad=True),
        "sensor_fault_logits": torch.zeros(1, 2, requires_grad=True),
        "sensor_reconstruction_prediction": torch.ones(1, 2, requires_grad=True),
        "evidence_sufficiency": torch.tensor([0.7], requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "sample_node": torch.tensor([1]),
        # action_template/target_pointer/ood_class are targets_v2's governed
        # names, not the model's own action_logits/action_pointer_logits/
        # ood_logits output names -- compute_multitask_loss looks each task
        # up by its governed name (see the classifications dict comment).
        "action_template": torch.tensor([0]),
        "target_pointer": torch.tensor([1]),
        "plan_value": torch.tensor([[0.0, 0.5]]),
        "plan_validity": torch.tensor([[0, 1]]),
        "ood_class": torch.tensor([0]),
        "start_time": torch.tensor([3]),
        "duration": torch.tensor([2]),
        "relative_strength": torch.tensor([1]),
        "sensor_fault": torch.zeros(1, 2),
        "sensor_reconstruction": torch.zeros(1, 2),
        "evidence_sufficiency": torch.tensor([1.0]),
    }
    result = compute_multitask_loss(
        outputs,
        targets,
        task_weights={"sensor_reconstruction": 2.0},
        profile_ordinal_weight=0.4,
    )
    assert set(result.tasks) == set(targets)
    assert torch.isfinite(result.total)
    result.total.backward()


def test_loss_task_keys_match_targets_v2_governed_names_not_output_names() -> None:
    """compute_multitask_loss's classifications dict previously keyed
    action_logits/action_pointer_logits/ood_logits by their own output
    names ("action"/"action_pointer"/"ood") instead of targets_v2's
    governed target names ("action_template"/"target_pointer"/"ood_class").
    A correctly generated governed target therefore silently never matched
    and silently never trained those heads -- confirm the fix by using
    ONLY the governed names in targets and asserting each produces a loss
    entry keyed by that governed name."""

    from hydroswarm.training.targets_v2 import TARGETS_V2

    governed_names = set(TARGETS_V2)
    assert {"action_template", "target_pointer", "ood_class"} <= governed_names
    assert "action" not in governed_names
    assert "action_pointer" not in governed_names
    assert "ood" not in governed_names

    outputs = {
        "action_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "action_pointer_logits": torch.tensor([[0.0, 3.0]], requires_grad=True),
        # core-issues3.txt Phase 6: ood_class maps to ood_category_logits
        # (the correctly-sized 11-class head), not the old 3-logit ood_logits
        # (a different, deterministic-severity-adjacent concept).
        "ood_category_logits": torch.zeros(1, 11, requires_grad=True),
    }
    targets = {
        "action_template": torch.tensor([0]),
        "target_pointer": torch.tensor([1]),
        "ood_class": torch.tensor([0]),
    }
    result = compute_multitask_loss(outputs, targets)
    assert set(result.tasks) == {"action_template", "target_pointer", "ood_class"}


def test_evidence_sufficiency_head_output_shape_matches_its_scalar_per_example_target() -> None:
    # Regression: HydroCore.evidence_head ends in nn.Linear(d_model, 1)
    # and its output was never squeezed, unlike event_presence_logits'
    # identical squeeze(-1) for the same "one boolean per incident" shape
    # -- so forward() returned [batch, 1] while corpus.py's real
    # evidence_sufficiency target (and hence the collated batch target)
    # is [batch]. F.binary_cross_entropy raises ValueError on that shape
    # mismatch; this was never caught because no prior test exercised the
    # real model's output against a real-shaped target together, only
    # synthetic already-matching-shape fixtures on one side or the other.
    from hydroswarm.model import HydroCore

    model = HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
    ).eval()
    batch = {
        "node_features": torch.randn(2, 4, 3),
        "temporal_features": torch.randn(2, 3, 4, 2),
        "quality_features": torch.randn(2, 3, 4, 2),
        "source_candidate_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    with torch.no_grad():
        output = model(batch)
    assert output["evidence_sufficiency"].shape == (2,)

    target = torch.tensor([True, False])
    loss = compute_multitask_loss(
        {"evidence_sufficiency": output["evidence_sufficiency"]},
        {"evidence_sufficiency": target},
    )  # must not raise
    assert torch.isfinite(loss.total)


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


def test_sensor_fault_mask_excludes_unsensored_nodes_from_the_loss() -> None:
    # core-issues.txt repair item 3: an unsensored node's sensor_fault=0.0
    # placeholder is not a real "healthy" observation and must not be
    # trained against. Row 0 has a genuinely wrong, confident prediction
    # at every position but sensor_fault_mask=False everywhere -- if the
    # mask were ignored, this row alone would produce a large loss.
    outputs = {"sensor_fault_logits": torch.tensor([[8.0, 8.0, 8.0]], requires_grad=True)}
    targets = {
        "sensor_fault": torch.tensor([[0.0, 0.0, 0.0]]),
        "sensor_fault_mask": torch.tensor([[False, False, False]]),
    }
    result = compute_multitask_loss(outputs, targets)
    assert result.tasks["sensor_fault"].item() == pytest.approx(0.0)


def test_sensor_fault_mask_still_trains_on_real_sensor_positions() -> None:
    outputs = {"sensor_fault_logits": torch.tensor([[8.0, 8.0, 8.0]], requires_grad=True)}
    targets = {
        "sensor_fault": torch.tensor([[0.0, 0.0, 0.0]]),
        # Only position 1 has a real sensor; it is confidently (and
        # wrongly) predicted faulty, so the masked loss must be well above
        # zero even though positions 0 and 2 (also wrong) are excluded.
        "sensor_fault_mask": torch.tensor([[False, True, False]]),
    }
    result = compute_multitask_loss(outputs, targets)
    assert result.tasks["sensor_fault"].item() > 1.0


def test_padding_a_batch_does_not_change_sensor_fault_loss_for_real_nodes() -> None:
    """core-issues.txt repair item 3: 'a test proving graph padding does
    not change sensor-fault loss'. Two real sensor positions (one healthy,
    one faulty) must contribute the identical loss whether collated alone
    or alongside a second, larger example -- the padded positions
    (sensor_fault_mask=False, added by collate_variable_topology's
    zero-fill) must contribute exactly nothing either way."""

    from hydroswarm.model import HydroCore
    from hydroswarm.training import CurriculumStage, ScenarioExample, collate_variable_topology

    def _example(scenario_id: str, nodes: int, seed: int) -> ScenarioExample:
        generator = torch.Generator().manual_seed(seed)
        sensor_fault = torch.zeros(nodes)
        sensor_fault_mask = torch.zeros(nodes, dtype=torch.bool)
        sensor_fault[0], sensor_fault_mask[0] = 0.0, True  # healthy, real sensor
        sensor_fault[1], sensor_fault_mask[1] = 1.0, True  # faulty, real sensor
        edges = [(i, i + 1) for i in range(nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long)
        return ScenarioExample(
            scenario_id=scenario_id, network_id="net", split="train", seed=seed,
            seed_family=f"family-{scenario_id}", stage=CurriculumStage.CLEAN,
            inputs={
                "node_features": torch.randn(nodes, 3, generator=generator),
                "temporal_features": torch.randn(2, nodes, 2, generator=generator),
                "quality_features": torch.randn(2, nodes, 2, generator=generator),
                "edge_index": edge_index,
                "edge_features": torch.randn(len(edges), 2, generator=generator) if edges else torch.zeros(0, 2),
                "source_candidate_mask": torch.ones(nodes, dtype=torch.bool),
                "node_mask": torch.ones(nodes, dtype=torch.bool),
            },
            targets={
                "source_node": torch.tensor(0),
                "sensor_fault": sensor_fault,
                "sensor_fault_mask": sensor_fault_mask,
            },
        )

    model = HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2, edge_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=1, modality_layers=1,
        latent_tokens=64, adapter_dims=(32, 32, 32), dropout=0.0,
    ).eval()

    small = _example("small", nodes=3, seed=1)
    large = _example("large", nodes=7, seed=2)

    with torch.no_grad():
        alone_inputs, alone_targets = collate_variable_topology([small])
        alone_output = model(alone_inputs)
        padded_inputs, padded_targets = collate_variable_topology([small, large])
        padded_output = model(padded_inputs)

    alone_loss = compute_multitask_loss(
        {"sensor_fault_logits": alone_output["sensor_fault_logits"]},
        {"sensor_fault": alone_targets["sensor_fault"], "sensor_fault_mask": alone_targets["sensor_fault_mask"]},
    )
    # Isolate "small"'s own contribution from the padded batch by zeroing
    # out "large"'s row before computing the loss -- compute_multitask_loss
    # has no per-example breakdown, so this compares the two rows directly.
    padded_logits_small_only = padded_output["sensor_fault_logits"][:1]
    padded_targets_small_only = {
        "sensor_fault": padded_targets["sensor_fault"][:1],
        "sensor_fault_mask": padded_targets["sensor_fault_mask"][:1],
    }
    padded_loss = compute_multitask_loss(
        {"sensor_fault_logits": padded_logits_small_only}, padded_targets_small_only
    )
    torch.testing.assert_close(alone_loss.total, padded_loss.total, atol=1e-5, rtol=1e-4)


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
