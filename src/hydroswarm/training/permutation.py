"""Graph permutation augmentation and equivariance measurement (Task 1.5).

Permuting an example's node ordering must never change what the example
*means* -- the same physical incident, same true source, same masks -- only
which array index each node happens to sit at. permute_example() applies a
node permutation and remaps every node-indexed input, target, and piece of
topology metadata so that semantic identity survives; measure_equivariance()
then runs the model on both versions, maps the permuted output back to the
original ordering, and reports how close the two are, so any accidentally
non-equivariant feature or model path can be identified rather than assumed
away.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Module

from .data import ScenarioExample, TopologyMetadata
from .targets_v2 import NODE_ARRAY_TARGETS, NODE_INDEX_TARGETS

CollateFn = Callable[[Sequence[ScenarioExample]], tuple[dict[str, Tensor], dict[str, Tensor]]]

#: Inputs indexed [nodes, ...] -- permuted along dim 0.
NODE_INDEXED_INPUT_KEYS = (
    "node_features",
    "travel_time",
    "reservoir_reachability",
    "demand_centrality",
    "classical_prior",
    "source_candidate_mask",
    "node_mask",
)
#: Inputs indexed [time, nodes, ...] -- permuted along dim 1.
TIME_NODE_INDEXED_INPUT_KEYS = ("temporal_features", "quality_features")
#: Targets indexed [nodes] -- permuted along dim 0. Derived from
#: targets_v2.NODE_ARRAY_TARGETS (never listed separately by hand) --
#: core-issues3.txt Phase 10.2: this module's own hand-maintained tuple
#: (previously only `("sensor_fault",)`) had silently fallen out of sync
#: with NODE_ARRAY_TARGETS, missing sensor_reconstruction/
#: future_concentration/travel_time/information_gain/candidate_reduction
#: entirely -- an unpermuted node-array target compared against a permuted
#: model output would report a FALSE equivariance violation (or mask a
#: real one), not merely a missing feature. Found while fixing the exact
#: same drift pattern in variable_collate.py's own NODE_INDEXED_TARGET_KEYS
#: (also derived from NODE_ARRAY_TARGETS now, for the same reason).
NODE_INDEXED_TARGET_KEYS = tuple(NODE_ARRAY_TARGETS)
#: Targets that are themselves a single local node index needing remapping.
#: Derived from targets_v2.NODE_INDEX_TARGETS -- was missing `sample_node`
#: (only had `source_node`) for the same reason as above.
NODE_INDEX_TARGET_KEYS = tuple(NODE_INDEX_TARGETS)


def _validate_permutation(permutation: Sequence[int], node_count: int) -> list[int]:
    values = list(permutation)
    if sorted(values) != list(range(node_count)):
        raise ValueError(f"permutation must be a rearrangement of range({node_count})")
    return values


def _permute_topology(topology: TopologyMetadata, permutation: Sequence[int]) -> TopologyMetadata:
    new_node_ids = tuple(topology.node_ids[old_index] for old_index in permutation)
    return TopologyMetadata(
        topology_hash=topology.topology_hash,
        network_hash=topology.network_hash,
        node_ids=new_node_ids,
        edge_ids=topology.edge_ids,  # real-world IDs are permutation-independent
        source_candidate_ids=topology.source_candidate_ids,
        hydraulic_state_hash=topology.hydraulic_state_hash,
        signature_library_hash=topology.signature_library_hash,
        target_schema_version=topology.target_schema_version,
        feature_schema_version=topology.feature_schema_version,
    )


def permute_example(example: ScenarioExample, permutation: Sequence[int]) -> ScenarioExample:
    """Return a new example with node ordering rearranged by ``permutation``.

    ``permutation[new_index] == old_index``, i.e. the node that used to sit
    at ``old_index`` now sits at ``new_index`` (the same convention NumPy's
    fancy indexing uses: ``permuted = original[permutation]``).
    """

    node_count = example.inputs["node_features"].shape[0]
    permutation = _validate_permutation(permutation, node_count)
    perm_index = torch.tensor(permutation, dtype=torch.long)
    old_index_to_new_index = [0] * node_count
    for new_index, old_index in enumerate(permutation):
        old_index_to_new_index[old_index] = new_index
    remap = torch.tensor(old_index_to_new_index, dtype=torch.long)

    new_inputs = dict(example.inputs)
    for key in NODE_INDEXED_INPUT_KEYS:
        if key in new_inputs:
            new_inputs[key] = new_inputs[key][perm_index]
    for key in TIME_NODE_INDEXED_INPUT_KEYS:
        if key in new_inputs:
            new_inputs[key] = new_inputs[key][:, perm_index]
    if "edge_index" in new_inputs and new_inputs["edge_index"].numel():
        new_inputs["edge_index"] = remap[new_inputs["edge_index"]]

    new_targets = dict(example.targets)
    for key in NODE_INDEXED_TARGET_KEYS:
        if key in new_targets:
            new_targets[key] = new_targets[key][perm_index]
    for key in NODE_INDEX_TARGET_KEYS:
        if key in new_targets:
            old_local_index = int(new_targets[key])
            new_targets[key] = torch.tensor(old_index_to_new_index[old_local_index])

    new_topology = _permute_topology(example.topology, permutation) if example.topology is not None else None

    return ScenarioExample(
        scenario_id=example.scenario_id,
        network_id=example.network_id,
        split=example.split,
        seed=example.seed,
        seed_family=example.seed_family,
        stage=example.stage,
        inputs=new_inputs,
        targets=new_targets,
        trajectory=example.trajectory,
        topology=new_topology,
    )


@dataclass(frozen=True, slots=True)
class EquivarianceReport:
    max_absolute_source_logit_difference: float
    predicted_source_agrees: bool
    non_equivariant_keys: tuple[str, ...]


def measure_equivariance(
    model: Module,
    example: ScenarioExample,
    permutation: Sequence[int],
    *,
    collate_fn: CollateFn,
    atol: float = 1e-4,
) -> EquivarianceReport:
    """Run ``model`` on ``example`` and on its permuted version, map the
    permuted output back to the original node ordering, and report how
    close they are. Any input key whose padded/collated value differs after
    accounting for the permutation (beyond floating-point noise) is
    surfaced in ``non_equivariant_keys`` rather than silently ignored.
    """

    node_count = example.inputs["node_features"].shape[0]
    permutation = _validate_permutation(permutation, node_count)
    permuted_example = permute_example(example, permutation)

    model.eval()
    with torch.no_grad():
        original_inputs, _ = collate_fn([example])
        original_output = model(original_inputs)
        permuted_inputs, _ = collate_fn([permuted_example])
        permuted_output = model(permuted_inputs)

    perm_index = torch.tensor(permutation, dtype=torch.long)
    original_logits = original_output["source_node_logits"][0, :node_count]
    permuted_logits = permuted_output["source_node_logits"][0, :node_count]
    # Map the permuted output back: position `new_index` in permuted_logits
    # corresponds to the node that was originally at `permutation[new_index]`.
    remapped_logits = torch.empty_like(original_logits)
    remapped_logits[perm_index] = permuted_logits

    difference = (original_logits - remapped_logits).abs()
    max_difference = float(difference.max())

    original_prediction = int(original_logits.argmax())
    remapped_prediction = int(remapped_logits.argmax())

    non_equivariant: list[str] = []
    for key in NODE_INDEXED_INPUT_KEYS:
        if key not in example.inputs:
            continue
        original_value = example.inputs[key]
        permuted_value = permuted_example.inputs[key]
        remapped_value = torch.empty_like(original_value)
        remapped_value[perm_index] = permuted_value
        if not torch.allclose(original_value.float(), remapped_value.float(), atol=atol):
            non_equivariant.append(key)

    return EquivarianceReport(
        max_absolute_source_logit_difference=max_difference,
        predicted_source_agrees=original_prediction == remapped_prediction,
        non_equivariant_keys=tuple(non_equivariant),
    )
