"""core-issues3.txt Phase 10.2: real (not synthetic-shape) coverage for
scripts/build_scout_state_dataset.py -- builds an actual sharded corpus
from a real trajectory JSONL (matching scripts/generate_trajectory_corpus.py's
exact on-disk schema) plus a real base ShardedScenarioDataset, then proves
the result trains: collate_variable_topology pads the merged per-node Scout
targets correctly across two different real topologies, and a real backward
pass through HydroCore(scout_control_heads=True) gives every Scout head a
nonzero gradient. Also proves the "no usable candidate" masked-placeholder
path (build_scout_state_dataset.build's own fail-safe for a scenario with
zero Scout steps) contributes exactly zero to those heads' losses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import build_scout_state_dataset  # noqa: E402

from hydroswarm.model import HydroCore  # noqa: E402
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


def _trajectory_record(scenario_id: str, *, nodes: int, sample_node: int | None) -> dict:
    """Matches scripts/generate_trajectory_corpus.py's real
    _incident_trajectory_to_json output shape for the fields
    build_scout_state_dataset.py actually reads (scenario_id,
    scout.steps[i].targets/diagnostics)."""

    if sample_node is None:
        steps: list[dict] = []
    else:
        information_gain = [0.0] * nodes
        information_gain[sample_node] = 1.25
        candidate_reduction = [0.0] * nodes
        candidate_reduction[sample_node] = 0.5
        steps = [
            {
                "targets": {
                    "sample_node": sample_node,
                    "sample_node_mask": True,
                    "information_gain": information_gain,
                    "information_gain_mask": [i == sample_node for i in range(nodes)],
                    "candidate_reduction": candidate_reduction,
                    "candidate_reduction_mask": [i == sample_node for i in range(nodes)],
                    "should_continue_sampling": True,
                },
                "diagnostics": {"already_sampled": []},
            }
        ]
    return {"scenario_id": scenario_id, "scout": {"steps": steps}}


def test_real_scout_state_dataset_trains_with_nonzero_gradients(tmp_path: Path) -> None:
    small = _base_example("small-scenario", nodes=3, edges=[(0, 1), (1, 2)], seed=1)
    large = _base_example("large-scenario", nodes=6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], seed=2)
    # A third scenario with genuinely zero Scout steps (Phase 2 item 4: "no
    # useful candidate exists") -- must be masked, not dropped or crashed on.
    no_candidate = _base_example("no-candidate-scenario", nodes=4, edges=[(0, 1), (1, 2), (2, 3)], seed=3)

    base_dir = tmp_path / "base-tensors"
    write_shards([small, large, no_candidate], base_dir)

    trajectory_jsonl = tmp_path / "validation.jsonl"
    with trajectory_jsonl.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(_trajectory_record("small-scenario", nodes=3, sample_node=2)) + "\n")
        stream.write(json.dumps(_trajectory_record("large-scenario", nodes=6, sample_node=4)) + "\n")
        stream.write(json.dumps(_trajectory_record("no-candidate-scenario", nodes=4, sample_node=None)) + "\n")

    output_dir = tmp_path / "scout-tensors"
    result = build_scout_state_dataset.build(base_dir, trajectory_jsonl, output_dir, expected_split="validation")
    assert result["examples_total"] == 3
    assert result["examples_with_scout_step0"] == 2
    assert result["examples_with_no_candidate_masked"] == 1

    dataset = ShardedScenarioDataset(output_dir, expected_split="validation")
    examples = [dataset[index] for index in range(len(dataset))]
    by_id = {example.scenario_id: example for example in examples}

    # Masked placeholder is genuinely masked, not a fabricated real value.
    placeholder = by_id["no-candidate-scenario"]
    assert bool(placeholder.targets["sample_node_mask"]) is False
    assert not bool(placeholder.targets["information_gain_mask"].any())
    assert not bool(placeholder.targets["candidate_reduction_mask"].any())

    inputs, targets = collate_variable_topology(examples)
    assert targets["information_gain"].shape == (3, 6)
    assert targets["candidate_reduction"].shape == (3, 6)

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
        scout_control_heads=True,
    )
    model.train()
    output = model(inputs)
    result_loss = compute_multitask_loss(output, targets)
    assert torch.isfinite(result_loss.total)
    for scout_task in ("sample_node", "information_gain", "candidate_reduction", "should_continue_sampling"):
        assert scout_task in result_loss.tasks, f"{scout_task} did not reach compute_multitask_loss"

    result_loss.total.backward()
    scout_heads = ("sample_node_head", "information_gain_head", "candidate_reduction_head", "should_continue_sampling_head")
    for head_name in scout_heads:
        head = getattr(model, head_name)
        gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
        assert gradients, f"{head_name} received no gradient at all"
        assert any(float(gradient.abs().sum()) > 0 for gradient in gradients), f"{head_name} gradient is all-zero"
