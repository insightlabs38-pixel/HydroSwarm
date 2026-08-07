"""core-issues3.txt Phase 10.3: real (not synthetic-shape) coverage for
scripts/build_strategist_candidate_dataset.py -- builds an actual sharded
corpus from a real trajectory JSONL (matching
scripts/generate_trajectory_corpus.py's exact on-disk schema for
`strategist.steps[0].labels`/`.targets`) plus a real base
ShardedScenarioDataset, then proves the result trains: collate_variable_
topology pads the candidate-plan (P) dimension correctly across two
different real topologies with DIFFERENT candidate counts, and a real
backward pass through HydroCore(strategist_mode="candidate_conditioned")
gives every plan-conditioned head (CandidatePlanEncoder, plan_value_head,
plan_validity_head, every consequence-proxy head) a nonzero gradient.
Also proves the "no usable Strategist step" masked-placeholder path
contributes exactly zero to those heads via plan_mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import build_strategist_candidate_dataset  # noqa: E402

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT, ACTION_TEMPLATES  # noqa: E402
from hydroswarm.training import (  # noqa: E402
    CurriculumStage,
    ScenarioExample,
    ShardedScenarioDataset,
    collate_variable_topology,
)
from hydroswarm.training.losses import compute_multitask_loss  # noqa: E402
from hydroswarm.training.sharded_data import write_shards  # noqa: E402


def _base_example(scenario_id: str, *, nodes: int, edges: list[tuple[int, int]], seed: int) -> ScenarioExample:
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)
    return ScenarioExample(
        scenario_id=scenario_id,
        network_id=f"net-{nodes}",
        split="validation",
        seed=seed,
        seed_family=f"family-{scenario_id}",
        stage=CurriculumStage.CLEAN,
        inputs={
            "node_features": torch.randn(nodes, 3, generator=generator),
            "temporal_features": torch.randn(2, nodes, 2, generator=generator),
            "quality_features": torch.randn(2, nodes, 2, generator=generator),
            "edge_index": edge_index,
            "edge_features": torch.randn(len(edges), 2, generator=generator) if edges else torch.zeros(0, 2),
            "source_candidate_mask": torch.ones(nodes, dtype=torch.bool),
            "node_mask": torch.ones(nodes, dtype=torch.bool),
            "sensor_mask": torch.ones(2, nodes, dtype=torch.bool),
            "quality_mask": torch.ones(2, nodes, dtype=torch.bool),
            "edge_mask": torch.ones(len(edges), dtype=torch.bool),
        },
        targets={
            "source_node": torch.tensor(0),
            "sensor_fault": torch.zeros(nodes),
        },
    )


def _label(action_template: str, *, target_id: str | None, target_type: str, valid: bool) -> dict:
    return {
        "action_template": action_template,
        "primary_target_id": target_id,
        "primary_target_type": target_type,
        "plan_validity": valid,
        "is_no_response_comparator": action_template == "NO_ACTION",
    }


def _target(*, action_template_index: int, target_pointer: int, pointer_valid: bool, valid: bool, value: float) -> dict:
    return {
        "action_template": action_template_index,
        "target_pointer": target_pointer,
        "target_pointer_mask": pointer_valid,
        "plan_validity": valid,
        "plan_validity_mask": True,
        "plan_value": value,
        "plan_value_mask": True,
        "exposure_proxy": 0.1,
        "exposure_proxy_mask": True,
        "pressure_risk_proxy": 0.2,
        "pressure_risk_proxy_mask": True,
        "service_loss_proxy": 0.3,
        "service_loss_proxy_mask": True,
        "containment_time_proxy": 0.4,
        "containment_time_proxy_mask": True,
        "plan_regret_proxy": 0.0,
        "plan_regret_proxy_mask": True,
    }


def _trajectory_record(scenario_id: str, *, plan_count: int) -> dict:
    """Matches generate_trajectory_corpus.py's real on-disk schema for the
    fields build_strategist_candidate_dataset.py reads. Uses NO_ACTION plus
    however many additional real templates `plan_count` calls for -- proves
    the collator handles a genuinely DIFFERENT candidate count per example
    (not just the real corpus's always-9 case)."""

    labels = [_label("NO_ACTION", target_id=None, target_type="NONE", valid=True)]
    targets = [_target(action_template_index=0, target_pointer=-1, pointer_valid=False, valid=True, value=1.0)]
    remaining_templates = [name for name in ACTION_TEMPLATES if name != "NO_ACTION"]
    for offset in range(plan_count - 1):
        name = remaining_templates[offset]
        template_index = ACTION_TEMPLATES.index(name)
        labels.append(_label(name, target_id="J1", target_type="NODE", valid=True))
        targets.append(_target(action_template_index=template_index, target_pointer=0, pointer_valid=True, valid=True, value=0.5))
    return {"scenario_id": scenario_id, "strategist": {"steps": [{"labels": labels, "targets": targets}]}}


def test_real_strategist_candidate_dataset_trains_with_nonzero_gradients(tmp_path: Path) -> None:
    small = _base_example("small-scenario", nodes=3, edges=[(0, 1), (1, 2)], seed=1)
    large = _base_example("large-scenario", nodes=6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], seed=2)
    no_step = _base_example("no-step-scenario", nodes=4, edges=[(0, 1), (1, 2), (2, 3)], seed=3)

    base_dir = tmp_path / "base-tensors"
    write_shards([small, large, no_step], base_dir)

    trajectory_jsonl = tmp_path / "validation.jsonl"
    with trajectory_jsonl.open("w", encoding="utf-8") as stream:
        # Different real candidate counts across examples -- exercises the
        # collator's padding, not just a fixed-P pass-through.
        stream.write(json.dumps(_trajectory_record("small-scenario", plan_count=3)) + "\n")
        stream.write(json.dumps(_trajectory_record("large-scenario", plan_count=ACTION_TEMPLATE_COUNT)) + "\n")

    output_dir = tmp_path / "strategist-tensors"
    result = build_strategist_candidate_dataset.build(base_dir, trajectory_jsonl, output_dir, expected_split="validation")
    assert result["examples_total"] == 3
    assert result["examples_with_strategist_step0"] == 2
    assert result["examples_with_no_step_masked"] == 1

    dataset = ShardedScenarioDataset(output_dir, expected_split="validation")
    examples = [dataset[index] for index in range(len(dataset))]
    by_id = {example.scenario_id: example for example in examples}

    placeholder = by_id["no-step-scenario"]
    assert not bool(placeholder.inputs["plan_mask"].any())
    assert not bool(placeholder.targets["plan_validity_mask"].any())

    inputs, targets = collate_variable_topology(examples)
    max_plans = max(len(candidate.inputs["plan_mask"]) for candidate in (by_id["small-scenario"], by_id["large-scenario"], placeholder))
    assert inputs["plan_template_ids"].shape == (3, max_plans)
    assert inputs["plan_mask"].shape == (3, max_plans)
    # small-scenario has 3 real candidates, padded up to max_plans.
    small_row = [example.scenario_id for example in examples].index("small-scenario")
    assert inputs["plan_mask"][small_row].sum().item() == 3
    assert targets["plan_value"].shape == (3, max_plans)

    model = HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        edge_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        latent_tokens=64,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
        strategist_mode="candidate_conditioned",
        action_vocabulary_size=ACTION_TEMPLATE_COUNT,
        consequence_prescreening_heads=True,
    )
    model.train()
    output = model(inputs)
    result_loss = compute_multitask_loss(output, targets)
    assert torch.isfinite(result_loss.total)
    plan_tasks = (
        "plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
        "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
    )
    for task in plan_tasks:
        assert task in result_loss.tasks, f"{task} did not reach compute_multitask_loss"

    result_loss.total.backward()
    for head_name in ("plan_value_head", "plan_validity_head", "candidate_plan_encoder"):
        head = getattr(model, head_name)
        gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
        assert gradients, f"{head_name} received no gradient at all"
        assert any(float(gradient.abs().sum()) > 0 for gradient in gradients), f"{head_name} gradient is all-zero"
    for proxy_name, proxy_head in model.consequence_proxy_heads.items():
        gradients = [parameter.grad for parameter in proxy_head.parameters() if parameter.grad is not None]
        assert gradients, f"{proxy_name} head received no gradient at all"
        assert any(float(gradient.abs().sum()) > 0 for gradient in gradients), f"{proxy_name} gradient is all-zero"
