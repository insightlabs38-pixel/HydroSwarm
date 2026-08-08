"""Frozen Scout state-conditioning input contract (core-issues5.txt
Section 18.2, future-proofing before architecture freeze).

Learned Scout (`sample_node`/`information_gain`/`candidate_reduction`/
`should_continue_sampling`) is disabled at runtime today -- the
deterministic classical expected-information-gain recommendation remains
authoritative (`hydroswarm.inference.authority.scout_certificate`,
suppression reason `LEARNED_SCOUT_SUPPRESSED:FAILED_PROMOTION_GATE`).

Section 18.2 requires that, BEFORE freeze, the codebase pin down whether a
future one-retrain Scout improvement can condition on genuine multi-round
sampling state -- already_sampled nodes, current round, remaining budget,
newly revealed evidence, sampling constraints, and current posterior --
using `HydroCore`'s EXISTING input channels, with NO new parameters added
to the frozen architecture. If that is true, the mapping must be defined
and versioned now, not left implicit; if any item genuinely has no
representation, that must be stated honestly rather than assumed away.

Audit finding (this module's content): all six items ARE representable
today via already-existing `HydroBatch` channels
(`hydroswarm.model.core.HydroBatch`) -- no new HydroCore parameters are
required. What is NOT true yet is that any current corpus-generation or
runtime pipeline code actually POPULATES these channels with real
multi-round Scout state; today they are architecturally present but
unwired for this purpose (several, like `residual_features`, are unused
by any real caller at all; `role_features` currently carries only
sampling-history/verifier-feedback style single-shot context per
`HybridInferencePipeline`, not scout-round bookkeeping). A future
retrain-only Scout improvement is therefore honestly describable as "one
retrain, PLUS a corpus/pipeline wiring pass to populate the channels named
here" -- not "one retrain" alone. That distinction is the whole point of
recording this contract explicitly rather than letting a future effort
either invent new parameters unnecessarily or discover the wiring gap
partway through a retrain attempt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

#: The six items Section 18.2 requires the data/runtime contract to be
#: able to represent.
ScoutStateItem = Literal[
    "already_sampled",
    "current_sampling_round",
    "sample_budget_remaining",
    "currently_revealed_evidence",
    "sampling_constraints_accessibility",
    "current_posterior_candidate_state",
]

SCOUT_STATE_ITEMS: tuple[ScoutStateItem, ...] = (
    "already_sampled",
    "current_sampling_round",
    "sample_budget_remaining",
    "currently_revealed_evidence",
    "sampling_constraints_accessibility",
    "current_posterior_candidate_state",
)


@dataclass(frozen=True, slots=True)
class ScoutStateChannelBinding:
    """One item's binding to an existing `HydroBatch` channel -- no new
    HydroCore parameters implied by any binding in this contract."""

    item: ScoutStateItem
    #: The `HydroBatch` (hydroswarm.model.core) field this item would be
    #: encoded into.
    channel: str
    #: How the item maps onto that channel's existing shape/semantics.
    encoding: str
    #: Whether any real code today actually populates the channel with
    #: this item's data -- NOT whether the channel itself exists.
    wired: bool


#: Versioned, frozen mapping. A future retrain effort must add real
#: corpus/pipeline code that populates these channels per `encoding`
#: below; it must NOT need to add a new field to `HydroBatch` or a new
#: input projection to `HydroCore` to represent any of these six items.
SCOUT_STATE_CONTRACT: tuple[ScoutStateChannelBinding, ...] = (
    ScoutStateChannelBinding(
        item="already_sampled",
        channel="sensor_mask",
        encoding=(
            "A node that has been sampled this incident becomes an "
            "observed sensor -- the same per-node boolean channel "
            "Sentinel's multi-round posterior update already reads "
            "(HybridInferencePipeline's own round-over-round observation "
            "accumulation). No dedicated 'was this node ever sampled' "
            "flag is needed; sensor_mask already IS that flag once a "
            "sample is folded into the observation set."
        ),
        wired=True,
    ),
    ScoutStateChannelBinding(
        item="current_sampling_round",
        channel="role_features",
        encoding=(
            "One scalar packed into the existing 8-dimensional global "
            "role_features vector (HydroCore.role_feature_dim=8), "
            "projected via the existing role_projection linear layer. "
            "role_features is [batch, 8] or [batch, sequence, 8] "
            "(mean-pooled over sequence) -- a single round-index scalar "
            "fits without touching the dimension."
        ),
        wired=False,
    ),
    ScoutStateChannelBinding(
        item="sample_budget_remaining",
        channel="role_features",
        encoding=(
            "Another scalar in the same 8-dim role_features vector as "
            "current_sampling_round. hydroswarm.training.trajectory_v2."
            "RemainingBudgets already computes samples/exact_simulations/"
            "actions remaining as a training-time struct; it is not yet "
            "projected into any tensor a model can read."
        ),
        wired=False,
    ),
    ScoutStateChannelBinding(
        item="currently_revealed_evidence",
        channel="quality_features+sensor_mask",
        encoding=(
            "The same per-node, per-timestep quality_features tensor "
            "(gated by sensor_mask) Sentinel's multi-round posterior "
            "update already consumes for newly observed concentration "
            "readings -- no separate channel needed. The real gap is "
            "corpus-side, not schema-side: hydroswarm.training."
            "scout_trajectory's generator does not yet simulate a "
            "genuinely new concentration reading being revealed after a "
            "sample is taken, so this channel has nowhere real to read "
            "revealed-evidence data FROM yet, even though the channel "
            "itself is unchanged."
        ),
        wired=False,
    ),
    ScoutStateChannelBinding(
        item="sampling_constraints_accessibility",
        channel="residual_features",
        encoding=(
            "The existing per-node, 4-dimensional residual_features "
            "channel (HydroCore.residual_feature_dim=4), currently "
            "unpopulated by any real caller. A per-node "
            "accessible/inaccessible (and optionally a constraint-type "
            "ordinal) encoding fits within the existing 4 dimensions."
        ),
        wired=False,
    ),
    ScoutStateChannelBinding(
        item="current_posterior_candidate_state",
        channel="classical_prior",
        encoding=(
            "The existing per-node classical_prior scalar channel, "
            "purpose-built for exactly this and already wired into "
            "prior_projection -- the same posterior HydroCore's fusion "
            "and Strategist paths already condition on."
        ),
        wired=True,
    ),
)


def _payload() -> list[dict[str, object]]:
    return [
        {
            "item": binding.item,
            "channel": binding.channel,
            "encoding": binding.encoding,
            "wired": binding.wired,
        }
        for binding in SCOUT_STATE_CONTRACT
    ]


#: A content hash over the full contract -- a future edit that changes
#: which channel an item is bound to (not just prose) changes this hash,
#: giving any consumer (e.g. a promotion-readiness report) something
#: concrete to cite rather than "the contract is defined somewhere".
CONTRACT_VERSION = "scout-state-contract-v1"


def contract_hash() -> str:
    payload = {"version": CONTRACT_VERSION, "bindings": _payload()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def unwired_items() -> tuple[ScoutStateItem, ...]:
    """Items with a defined, parameter-shape-safe channel binding but no
    real code populating it yet -- the honest "one retrain PLUS wiring"
    gap this module's docstring describes."""

    return tuple(binding.item for binding in SCOUT_STATE_CONTRACT if not binding.wired)
