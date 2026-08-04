"""Variable-size topology batch collation (overnight-plan.txt Task 1.3).

Bridges ScenarioExample (per-example, potentially different node/edge
counts once Task 1.1 topology metadata is populated by real multi-topology
data) to HydroCore's already-existing padded/masked batch format
(hydroswarm.preprocessing.batching.pad_graph_batch), which the model
already consumes via node_mask/edge_mask/source_candidate_mask -- this
module does not change model behavior, only how a batch of differently-
shaped examples is assembled into it.

Unlike hydroswarm.training.data.collate_scenarios (which requires every
example in a batch to already share identical tensor shapes -- true only
for the single-topology learning-v1 corpus), collate_variable_topology()
pads to the batch's max node/edge count. Topology boundaries are preserved:
edge_index stays local to each example's own node ordering
([batch, 2, edges]), never offset into a shared/concatenated graph, so no
message can cross between separate graphs in a batch.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from hydroswarm.preprocessing.batching import NODE_SCALAR_FEATURE_KEYS, GraphSample, pad_graph_batch

from .data import ScenarioExample

#: Target keys that are node-indexed (one value per node) rather than a
#: single scalar per example, and therefore need the same padding as node
#: inputs rather than a plain stack.
NODE_INDEXED_TARGET_KEYS = ("sensor_fault",)


def _example_to_graph_sample(example: ScenarioExample) -> GraphSample:
    inputs = example.inputs
    node_scalars = {key: inputs[key] for key in NODE_SCALAR_FEATURE_KEYS if key in inputs}
    return GraphSample(
        node_features=inputs["node_features"],
        temporal_features=inputs["temporal_features"],
        quality_features=inputs["quality_features"],
        edge_index=inputs.get("edge_index", torch.zeros(2, 0, dtype=torch.long)),
        edge_features=inputs.get("edge_features", torch.zeros(0, 0)),
        timestamps=inputs.get("timestamps"),
        node_scalar_features=node_scalars or None,
    )


def collate_variable_topology(
    examples: Sequence[ScenarioExample],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    samples = [_example_to_graph_sample(example) for example in examples]
    inputs = pad_graph_batch(samples)
    max_nodes = inputs["node_features"].shape[1]

    target_keys = set(examples[0].targets)
    if any(set(example.targets) != target_keys for example in examples):
        raise ValueError("all examples in a batch require identical target keys")

    targets: dict[str, Tensor] = {}
    for key in sorted(target_keys):
        if key in NODE_INDEXED_TARGET_KEYS:
            padded = torch.zeros(len(examples), max_nodes, device=examples[0].targets[key].device)
            for index, example in enumerate(examples):
                value = example.targets[key]
                padded[index, : value.shape[0]] = value
            targets[key] = padded
        else:
            try:
                targets[key] = torch.stack([example.targets[key] for example in examples])
            except RuntimeError as error:
                raise ValueError(
                    f"target {key!r} is not node-indexed but has inconsistent shape across the "
                    "batch; add it to NODE_INDEXED_TARGET_KEYS if it is actually per-node"
                ) from error
    return inputs, targets
