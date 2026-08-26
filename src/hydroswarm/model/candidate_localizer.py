"""Candidate-conditioned source-localization scorer (EXPERIMENTAL).

Branch: `exp/candidate-conditioned-localizer-v1`. Motivated by
`docs/evaluation/experimental/CANDIDATE_CONDITIONED_LOCALIZER_V1_PLAN.md`
and the source-identifiability finding that a physics oracle recovers the
true source in the large majority of HydroCore-v5's own Top-1 failures
using the SAME real sensor evidence -- suggesting the default localizer's
bottleneck is how it uses evidence, not whether the evidence exists.

The default `HydroCore.source_node_head` scores each candidate node from
its OWN post-backbone hidden state (`self.source_node_head(sentinel_nodes)`,
a shared-weight per-node linear head -- already topology-count-agnostic,
see the plan doc Section 1). This module keeps that shared-weight,
no-fixed-node-ID-parameter property but makes the candidate-vs-evidence
comparison EXPLICIT: every candidate node forms a query that cross-attends
directly over sensor-node evidence (bypassing however many backbone hops
separate a peripheral candidate from the nearest sensor), biased by a
label-free candidate-to-sensor hop-distance feature, then a shared MLP
scores the query/attended-evidence pair. Every candidate in a batch is
scored by the exact same parameters -- no per-node or per-topology
embedding table, so it is invariant to node relabeling, sensor-list
ordering, and candidate-list ordering by construction, and supports any
candidate/sensor count (see `tests/unit/test_candidate_localizer.py`).

Never imported by default: `HydroCore` only constructs/uses this module
when `localizer_mode="candidate_conditioned"` is explicitly passed to its
constructor (default remains `"default"`, i.e. `source_node_head` alone --
byte-identical to pre-existing behavior). See core.py's `localizer_mode`
wiring.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoders import make_activation, make_norm

#: Candidate<->sensor hop-distance buckets fed to `self.hop_bias`
#: (`nn.Embedding`) -- 0, 1, 2, 3, 4-6, 7+ (index 5), plus one dedicated
#: "unreachable/no path" bucket (index 6). Coarse and small by design: this
#: is a bias on an attention score, not a distance regression target, and
#: the pilot networks here are 4-12 nodes wide (see plan doc Section 6) so
#: most real hop distances fall in the first few buckets already.
HOP_BUCKET_EDGES: tuple[int, ...] = (0, 1, 2, 3, 4, 7)
NUM_HOP_BUCKETS = len(HOP_BUCKET_EDGES) + 1  # + one "unreachable" bucket
UNREACHABLE_HOP_SENTINEL = -1  # caller convention: negative = no path


def bucket_hop_distance(hop_distance: Tensor) -> Tensor:
    """`[*, ...]` int hop distances -> bucket indices in `[0, NUM_HOP_BUCKETS)`.

    Negative values (this module's `UNREACHABLE_HOP_SENTINEL` convention for
    "no path between this candidate and this sensor") map to the dedicated
    last bucket, never negative-indexed into the embedding table.
    """

    edges = torch.as_tensor(HOP_BUCKET_EDGES, device=hop_distance.device, dtype=hop_distance.dtype)
    bucket = torch.bucketize(hop_distance.clamp_min(0), edges, right=True) - 1
    bucket = bucket.clamp(min=0, max=NUM_HOP_BUCKETS - 2)
    return torch.where(hop_distance < 0, torch.full_like(bucket, NUM_HOP_BUCKETS - 1), bucket)


class CandidateConditionedLocalizer(nn.Module):
    """Shared candidate<->sensor compatibility scorer.

    Forward contract (all tensors batch-first, `nodes` = padded node count):

    - `sentinel_nodes`: `[B, N, D]` -- post-backbone per-node hidden state
      (same tensor the default `source_node_head` already consumes).
    - `candidate_mask`: `[B, N]` bool -- True at real candidate positions
      (`HydroBatch["source_candidate_mask"]`, masked_fill'd to `node_mask`
      the same way `HydroCore.forward` already does before calling this).
    - `sensor_mask_nodes`: `[B, N]` bool -- True at active-sensor node
      positions (reduced from the per-timestep `sensor_mask` the temporal
      encoder consumes -- see `candidate_sensor_features.active_sensor_mask`).
    - `hop_distance`: `[B, N, N]` int, `hop_distance[b, i, j]` = unweighted
      graph hop count from node i to node j (`UNREACHABLE_HOP_SENTINEL` if
      no path). Label-free, computed once per topology
      (`scripts/hydrocore_v5_experimental/candidate_conditioned_localizer_v1/
      candidate_sensor_features.py`).
    - `structural_features`: optional `[B, N, F_struct]` -- label-free
      per-candidate structural features (degree/centrality/hop-to-sensor
      summary), added into the candidate query. `None` iff this instance
      was built with `structural_feature_dim=0`.
    - `physics_features`: optional `[B, N, F_phys]` -- label-free per-
      candidate physics-compatibility features (Arm C only; see plan doc
      Section 4/5). `None` iff this instance was built with
      `physics_feature_dim=0`.

    Returns `[B, N]` float logits, `-inf`-filled (via `torch.finfo(...).min`)
    at non-candidate positions, exactly matching `source_node_head`'s own
    masking convention in `HydroCore.forward` so downstream softmax/loss
    code is unchanged either way.
    """

    def __init__(
        self,
        d_model: int,
        *,
        structural_feature_dim: int = 0,
        physics_feature_dim: int = 0,
        dropout: float = 0.1,
        normalization: str = "rmsnorm",
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if structural_feature_dim < 0 or physics_feature_dim < 0:
            raise ValueError("feature dims must be non-negative")
        self.d_model = d_model
        self.structural_feature_dim = structural_feature_dim
        self.physics_feature_dim = physics_feature_dim

        self.structural_projection = (
            nn.Linear(structural_feature_dim, d_model) if structural_feature_dim > 0 else None
        )
        self.physics_projection = nn.Linear(physics_feature_dim, d_model) if physics_feature_dim > 0 else None

        self.query_norm = make_norm(normalization, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.hop_bias = nn.Embedding(NUM_HOP_BUCKETS, 1)
        nn.init.zeros_(self.hop_bias.weight)
        self.attn_dropout = nn.Dropout(dropout)

        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            make_activation(activation),
            make_norm(normalization, d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        sentinel_nodes: Tensor,
        *,
        candidate_mask: Tensor,
        sensor_mask_nodes: Tensor,
        hop_distance: Tensor,
        structural_features: Tensor | None = None,
        physics_features: Tensor | None = None,
    ) -> Tensor:
        batch, nodes, d_model = sentinel_nodes.shape
        if d_model != self.d_model:
            raise ValueError(f"sentinel_nodes last dim {d_model} != configured d_model {self.d_model}")
        for name, tensor in (("candidate_mask", candidate_mask), ("sensor_mask_nodes", sensor_mask_nodes)):
            if tensor.shape != (batch, nodes):
                raise ValueError(f"{name} must have shape [batch, nodes], got {tuple(tensor.shape)}")
        if hop_distance.shape != (batch, nodes, nodes):
            raise ValueError(f"hop_distance must have shape [batch, nodes, nodes], got {tuple(hop_distance.shape)}")
        candidate_mask = candidate_mask.bool()
        sensor_mask_nodes = sensor_mask_nodes.bool()

        query_input = self.query_norm(sentinel_nodes)
        if self.structural_projection is not None:
            if structural_features is None:
                raise ValueError("structural_features required: module built with structural_feature_dim > 0")
            if structural_features.shape != (batch, nodes, self.structural_feature_dim):
                raise ValueError("structural_features must have shape [batch, nodes, structural_feature_dim]")
            query_input = query_input + self.structural_projection(torch.nan_to_num(structural_features.float()))
        if self.physics_projection is not None:
            if physics_features is None:
                raise ValueError("physics_features required: module built with physics_feature_dim > 0")
            if physics_features.shape != (batch, nodes, self.physics_feature_dim):
                raise ValueError("physics_features must have shape [batch, nodes, physics_feature_dim]")
            query_input = query_input + self.physics_projection(torch.nan_to_num(physics_features.float()))

        q = self.query_proj(query_input)
        k = self.key_proj(sentinel_nodes)
        v = self.value_proj(sentinel_nodes)

        scores = torch.einsum("bqd,bkd->bqk", q, k) / (d_model**0.5)
        bucket = bucket_hop_distance(hop_distance)
        scores = scores + self.hop_bias(bucket).squeeze(-1)

        neg_inf = torch.finfo(scores.dtype).min
        sensor_key_mask = sensor_mask_nodes[:, None, :].expand(batch, nodes, nodes)
        scores = scores.masked_fill(~sensor_key_mask, neg_inf)
        # A candidate row with zero real sensors in this example (degenerate,
        # only possible if sensor_mask_nodes is all-False) would softmax an
        # all-neg_inf row to NaN -- give it one safe self-attending position
        # so softmax stays finite; that row's output is discarded anyway
        # since it can only occur when there is no real evidence to attend
        # to, and the caller still masks non-candidate positions downstream.
        no_valid_sensor = ~sensor_key_mask.any(dim=-1)
        if bool(no_valid_sensor.any()):
            scores = scores.masked_fill(no_valid_sensor.unsqueeze(-1) & torch.eye(nodes, device=scores.device, dtype=torch.bool)[None], 0.0)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = torch.einsum("bqk,bkd->bqd", weights, v)

        combined = torch.cat((query_input, attended), dim=-1)
        logits = self.scorer(combined).squeeze(-1)
        return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)


__all__ = [
    "HOP_BUCKET_EDGES",
    "NUM_HOP_BUCKETS",
    "UNREACHABLE_HOP_SENTINEL",
    "bucket_hop_distance",
    "CandidateConditionedLocalizer",
]
