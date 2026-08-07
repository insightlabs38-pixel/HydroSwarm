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
from .targets_v2 import NODE_ARRAY_TARGETS, PLAN_DIMENSION_TARGETS

#: Target keys that are node-indexed (one value per node) rather than a
#: single scalar per example, and therefore need the same padding as node
#: inputs rather than a plain stack. Every *_mask here pads with False,
#: which is correct for a padded position either way: it is not a real
#: node (sensored or otherwise) in that example's own topology.
#:
#: Derived from targets_v2.NODE_ARRAY_TARGETS (never listed separately by
#: hand) -- core-issues3.txt Phase 10.2: found this module's own
#: previously hand-maintained tuple had silently fallen out of sync with
#: NODE_ARRAY_TARGETS (missing information_gain/candidate_reduction,
#: Phase 7.2's per-node conversion), the third instance of this exact
#: "two independently-maintained lists drift apart" defect class in this
#: project (after the 8-vs-9 action-template count and the ood_head-vs-
#: ood_category_head loss mapping) -- caught only once a real multi-
#: topology batch containing a real information_gain target was actually
#: collated, not by inspection. Deriving this set programmatically, as
#: this fix does, makes that specific drift structurally impossible for
#: every current and future NODE_ARRAY_TARGETS entry, not just the two
#: this pass happened to find.
NODE_INDEXED_TARGET_KEYS = frozenset(NODE_ARRAY_TARGETS) | frozenset(f"{name}_mask" for name in NODE_ARRAY_TARGETS)

#: core-issues3.txt Phase 10.3: candidate-conditioned Strategist plan
#: INPUT keys (HydroBatch's own plan_* fields, model/core.py Section E) --
#: one row per candidate plan, padded along the "number of plans" (P)
#: dimension exactly like NODE_INDEXED_* keys pad along the node
#: dimension. Not derived from a targets_v2 registry (these are model-
#: input concepts, not governed targets) -- mirrors HydroBatch's own
#: plan_* field declarations in model/core.py directly; keep both in sync
#: by hand if HydroBatch's candidate-plan fields ever change, the same way
#: every other input-side HydroBatch key in this module already is.
PLAN_INDEXED_INPUT_KEYS = (
    "plan_template_ids",
    "plan_target_type",
    "plan_target_node_index",
    "plan_target_link_index",
    "plan_features",
    "plan_mask",
)
#: Padded (plan_mask=False) sentinel for each PLAN_INDEXED_INPUT_KEYS entry
#: that is not itself a boolean mask or a float feature vector -- 0 is
#: always in-vocabulary for template_ids/target_type (masked out downstream
#: regardless; see model/core.py's own "padded plans may legitimately carry
#: a sentinel template id" comment), -1 matches the existing masked-index
#: sentinel convention this same module already uses for node targets.
_PLAN_INPUT_PAD_VALUE: dict[str, int] = {
    "plan_template_ids": 0,
    "plan_target_type": 0,
    "plan_target_node_index": -1,
    "plan_target_link_index": -1,
}

#: Governed Strategist per-plan TARGETS (plan_validity, plan_value, the
#: five consequence proxies) -- reuses targets_v2.PLAN_DIMENSION_TARGETS
#: directly (the existing schema-validation registry for exactly this
#: "leading dimension is the candidate count" property) rather than a
#: fourth independently-maintained list. PLAN_DIMENSION_TARGETS also
#: includes the legacy anonymous-query action_template/target_pointer
#: targets -- harmless here since padding is a no-op for any key simply
#: absent from a given example's targets.
PLAN_INDEXED_TARGET_KEYS = frozenset(PLAN_DIMENSION_TARGETS) | frozenset(
    f"{name}_mask" for name in PLAN_DIMENSION_TARGETS
)


def _pad_plan_indexed_inputs(inputs: dict[str, Tensor], examples: Sequence[ScenarioExample]) -> None:
    """Mutates `inputs` in place, adding padded plan_* tensors for any
    PLAN_INDEXED_INPUT_KEYS present in any example -- mirrors
    collate_variable_topology's own node-padding loop, applied to the
    plan (candidate) dimension instead. Every example in the batch must
    either carry the full plan_* input set or none of it (a partial set
    would silently desync plan_mask from the other plan_* tensors)."""

    present_keys = {key for key in PLAN_INDEXED_INPUT_KEYS if any(key in example.inputs for example in examples)}
    if not present_keys:
        return
    for example in examples:
        missing = present_keys - set(example.inputs)
        if missing:
            raise ValueError(
                f"example {example.scenario_id!r} carries some but not all candidate-plan input "
                f"fields (missing {sorted(missing)}); a partial set would desync plan_mask from "
                "the other plan_* tensors"
            )

    max_plans = max(example.inputs["plan_mask"].shape[0] for example in examples)
    batch = len(examples)
    for key in present_keys:
        reference = examples[0].inputs[key]
        if key == "plan_features":
            feature_dim = reference.shape[-1]
            padded = torch.zeros(batch, max_plans, feature_dim, dtype=reference.dtype, device=reference.device)
        elif key == "plan_mask":
            padded = torch.zeros(batch, max_plans, dtype=torch.bool, device=reference.device)
        else:
            padded = torch.full(
                (batch, max_plans), _PLAN_INPUT_PAD_VALUE[key], dtype=reference.dtype, device=reference.device
            )
        for index, example in enumerate(examples):
            value = example.inputs[key]
            padded[index, : value.shape[0]] = value
        inputs[key] = padded


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
        # The real per-example masks were already computed once, correctly,
        # against genuine NaNs at corpus-generation time (see
        # HydraulicFeatureBuilder.build's own pad_graph_batch([sample])
        # call) and are stored verbatim in the shard. inputs["temporal_features"]
        # / inputs["quality_features"] above are that same generation
        # pass's NaN-replaced-with-zero output, so re-deriving masks from
        # them here via isfinite() would find nothing missing and produce
        # an all-True mask -- pass the stored masks through instead of
        # letting pad_graph_batch recompute them from already-sanitized data.
        node_mask=inputs.get("node_mask"),
        sensor_mask=inputs.get("sensor_mask"),
        quality_mask=inputs.get("quality_mask"),
        edge_mask=inputs.get("edge_mask"),
    )


def collate_variable_topology(
    examples: Sequence[ScenarioExample],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    samples = [_example_to_graph_sample(example) for example in examples]
    inputs = pad_graph_batch(samples)
    max_nodes = inputs["node_features"].shape[1]
    _pad_plan_indexed_inputs(inputs, examples)

    target_keys = set(examples[0].targets)
    if any(set(example.targets) != target_keys for example in examples):
        raise ValueError("all examples in a batch require identical target keys")

    max_plans = max((example.targets[key].shape[0] for example in examples for key in target_keys & PLAN_INDEXED_TARGET_KEYS), default=0)

    targets: dict[str, Tensor] = {}
    for key in sorted(target_keys):
        if key in NODE_INDEXED_TARGET_KEYS:
            reference = examples[0].targets[key]
            padded = torch.zeros(
                len(examples), max_nodes, dtype=reference.dtype, device=reference.device
            )
            for index, example in enumerate(examples):
                value = example.targets[key]
                padded[index, : value.shape[0]] = value
            targets[key] = padded
        elif key in PLAN_INDEXED_TARGET_KEYS:
            reference = examples[0].targets[key]
            padded = torch.zeros(
                len(examples), max_plans, dtype=reference.dtype, device=reference.device
            )
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
