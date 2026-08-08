"""Canonical deterministic-candidate -> HydroCore plan-tensor conversion
(core-issues5.txt Section 6, P0 blocker).

This is the ONE place a bounded deterministic candidate set (a tuple of
`PlanProposal`, from `hydroswarm.planning.response.generate_response_plans`)
is converted into `HydroBatch`'s candidate-conditioned INPUT fields
(`plan_template_ids`/`plan_target_type`/`plan_target_node_index`/
`plan_target_link_index`/`plan_features`/`plan_mask`). Both the live
runtime planning path (`hydroswarm.inference.pipeline.
HybridInferencePipeline._score_candidate_plans`) and any future training-
side consumer must import this rather than reimplementing the mapping --
`scripts/build_strategist_candidate_dataset.py`'s own
`_candidate_plan_tensors` builds the same INPUT half from a different
input shape (already-serialized trajectory-corpus label dicts, not live
`PlanProposal` objects) and imports `PLAN_FEATURE_NAMES` from here so the
6-dimensional `plan_features` schema itself has exactly one definition.

Every candidate's semantic identity (`action_template`, target type,
target node/link id) comes directly from `hydroswarm.planning.
action_templates.ACTION_TEMPLATES`/`LINK_TARGET_TEMPLATES` -- the same
canonical vocabulary the model's checkpoint identity pins
(`hydroswarm.training.checkpoint_identity.ACTION_TEMPLATE_SCHEMA_HASH`).
Candidate ORDER never carries semantic meaning: every tensor row is keyed
back to its owning `PlanProposal` purely by `action_template` name (a
stable identity), so reordering `proposals` before calling this function
changes nothing about what a resulting learned score is attributed to.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from hydroswarm.model.candidate_plan_encoder import TARGET_TYPE_INDEX
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_INDEX, LINK_TARGET_TEMPLATES
from hydroswarm.planning.response import PlanProposal

__all__ = ["PLAN_FEATURE_NAMES", "plan_proposals_to_candidate_tensors"]

#: Matches HydroCore.PLAN_FEATURE_DIM's default width (6). The 6th
#: dimension, `plan_validity_target_present`, is a training-time concept
#: (whether a governed WNTR-verified validity label exists for this
#: candidate) with no live-inference equivalent at PASS-2 scoring time --
#: every candidate scored here WILL receive a WNTR verification outcome
#: downstream, so this position is fixed at 1.0 (the overwhelming common
#: case in training data) rather than fabricating a target-derived value
#: the live path cannot actually know yet.
PLAN_FEATURE_NAMES = (
    "is_no_target_template",
    "is_node_target_template",
    "is_link_target_template",
    "has_target",
    "is_no_response_comparator",
    "plan_validity_target_present",
)


def _primary_target(proposal: PlanProposal) -> tuple[str, str | None]:
    """Which single target identifies this candidate's primary effect --
    the LINK a template in `LINK_TARGET_TEMPLATES` isolates/cuts, else the
    first NODE any of its actions targets, else NONE (NO_ACTION/
    WAIT_OBSERVE)."""

    if proposal.template in LINK_TARGET_TEMPLATES:
        for action in proposal.plan.actions:
            if action.target_id is not None:
                return "LINK", action.target_id
        return "LINK", None
    for action in proposal.plan.actions:
        if action.target_id is not None:
            return "NODE", action.target_id
    return "NONE", None


def _link_positions(graph: Any, node_ids: Sequence[str]) -> dict[str, int]:
    """The link_id -> graph-local edge-index-column mapping for THIS
    incident's batch -- replicates
    `hydroswarm.preprocessing.alignment.align_edges`'s own deterministic
    sort (by (source_index, target_index, original_row)) exactly, since
    that is the real function `HydraulicFeatureBuilder.build` uses to
    produce `batch["edge_index"]`'s column order. Must be computed from
    the SAME `graph` object (or an equivalent one with identical edge
    iteration order) the incident batch was actually built from."""

    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    indexed: list[tuple[int, int, int, str]] = []
    for row, (start, end, key, attributes) in enumerate(graph.edges(keys=True, data=True)):
        link_id = str(attributes.get("link_id", key))
        start_id, end_id = str(start), str(end)
        if start_id not in positions or end_id not in positions:
            continue
        indexed.append((positions[start_id], positions[end_id], row, link_id))
    indexed.sort(key=lambda item: (item[0], item[1], item[2]))
    return {link_id: position for position, (_source, _target, _row, link_id) in enumerate(indexed)}


def plan_proposals_to_candidate_tensors(
    proposals: Sequence[PlanProposal], *, node_ids: Sequence[str], graph: Any
) -> dict[str, torch.Tensor]:
    """Build a batch-size-1 `HydroBatch` plan-tensor dict from a real,
    bounded deterministic candidate set. `node_ids` and `graph` must be
    the exact same values (or byte-for-byte equivalent) the incident's own
    `HydraulicFeatureBuilder.build` call used, so target indices resolve
    into the correct graph-local positions."""

    plan_count = len(proposals)
    node_position = {node_id: index for index, node_id in enumerate(node_ids)}
    link_position = _link_positions(graph, node_ids)

    template_ids = torch.zeros(1, plan_count, dtype=torch.long)
    target_type = torch.zeros(1, plan_count, dtype=torch.long)
    node_index = torch.full((1, plan_count), -1, dtype=torch.long)
    link_index = torch.full((1, plan_count), -1, dtype=torch.long)
    features = torch.zeros(1, plan_count, len(PLAN_FEATURE_NAMES), dtype=torch.float32)
    plan_mask = torch.ones(1, plan_count, dtype=torch.bool)

    for position, proposal in enumerate(proposals):
        template_ids[0, position] = ACTION_TEMPLATE_INDEX[proposal.template]
        target_kind, target_id = _primary_target(proposal)
        target_type[0, position] = TARGET_TYPE_INDEX[target_kind]
        if target_kind == "NODE" and target_id in node_position:
            node_index[0, position] = node_position[target_id]
        elif target_kind == "LINK" and target_id in link_position:
            link_index[0, position] = link_position[target_id]
        features[0, position] = torch.tensor([
            float(target_kind == "NONE"),
            float(target_kind == "NODE"),
            float(target_kind == "LINK"),
            float(target_id is not None),
            float(proposal.template == "NO_ACTION"),
            1.0,
        ])

    return {
        "plan_template_ids": template_ids,
        "plan_target_type": target_type,
        "plan_target_node_index": node_index,
        "plan_target_link_index": link_index,
        "plan_features": features,
        "plan_mask": plan_mask,
    }
