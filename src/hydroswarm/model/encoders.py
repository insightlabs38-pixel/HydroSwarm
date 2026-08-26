"""Native static, structural and timestamp-aware temporal encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def make_norm(kind: str, width: int) -> nn.Module:
    normalized = kind.lower()
    if normalized == "rmsnorm":
        return nn.RMSNorm(width)
    if normalized == "layernorm":
        return nn.LayerNorm(width)
    raise ValueError("normalization must be 'rmsnorm' or 'layernorm'")


def make_activation(kind: str) -> nn.Module:
    normalized = kind.lower()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError("activation must be 'silu' or 'gelu'")


def _masked_temporal_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


#: Milestone 8.7 Arm C (AGE_FIX_PLUS_RELATIVE_TIME): the "window_relative"
#: elapsed-time normalization TemporalEncoder always used before this
#: (`elapsed / elapsed.abs().amax(...)`) rescales every window to span
#: [-1, 1] regardless of its actual physical duration -- two reports 10
#: minutes apart and two reports 10 hours apart produce IDENTICAL
#: normalized phase values, which Milestone 8.6's REPRESENTATION_
#: SENSITIVITY_COUNTERFACTUAL empirically confirmed (near-zero posterior
#: sensitivity to real elapsed-spacing changes on the golden-reference
#: network). "fixed_scale" divides by this fixed constant instead,
#: preserving actual elapsed MAGNITUDE (two windows of different physical
#: duration produce different phase values) while staying origin-
#: invariant (the subtraction against `elapsed[:, :1]` still happens
#: first) and consistent with the SAME normalization scale
#: `hydroswarm.preprocessing.builder.HydraulicFeatureBuilder` already uses
#: for its own age-derived features (divided by 86,400s), rather than an
#: arbitrarily chosen new constant.
FIXED_ELAPSED_TIME_SCALE_SECONDS = 86_400.0
ELAPSED_TIME_NORMALIZATION_MODES = ("window_relative", "fixed_scale")


class GraphStructuralEncoder(nn.Module):
    feature_names = ("travel_time", "reservoir_reachability", "demand_centrality")

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        *,
        normalization: str = "rmsnorm",
        activation: str = "silu",
        # exp/graph-structural-encoder-v2 (EXPERIMENTAL, NON-RELEASE): both
        # parameters default to the module's original behavior exactly --
        # structural_feature_dim=0 keeps the input width at 4 (identical
        # Linear(4, hidden_dim) as before) and use_edge_aggregation=False
        # skips building/calling the aggregation block entirely, so every
        # existing caller/checkpoint is unaffected unless it explicitly
        # opts in. See docs/evaluation/experimental/
        # GRAPH_STRUCTURAL_ENCODER_V2_PLAN.md Section 2 for the experiment
        # this exists to run (Arms B/C/D/E).
        structural_feature_dim: int = 0,
        use_edge_aggregation: bool = False,
        # exp/graph-structural-encoder-v2 Section 8 (capacity control):
        # "graph" (default) is Arm D/E's real edge_index aggregation.
        # "self_only" builds the IDENTICAL submodules (so an exactly
        # parameter-matched model) but ignores edge_index/edge_mask
        # entirely at forward time -- each node aggregates only from
        # itself (degree 1, no cross-node message) -- isolating "does
        # extra capacity in this encoder help" from "does actual graph
        # connectivity help." Only meaningful when use_edge_aggregation
        # is True; ignored otherwise.
        edge_aggregation_source: str = "graph",
    ) -> None:
        super().__init__()
        if edge_aggregation_source not in ("graph", "self_only"):
            raise ValueError("edge_aggregation_source must be 'graph' or 'self_only'")
        hidden_dim = hidden_dim or d_model
        self.structural_feature_dim = structural_feature_dim
        self.use_edge_aggregation = use_edge_aggregation
        self.edge_aggregation_source = edge_aggregation_source
        self.network = nn.Sequential(
            nn.Linear(4 + structural_feature_dim, hidden_dim),
            make_activation(activation),
            make_norm(normalization, hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )
        if use_edge_aggregation:
            # Arm D/E: one lightweight, controlled-capacity mean-neighbor
            # aggregation pass over this encoder's own per-node hidden
            # output -- deliberately smaller than the backbone's own
            # EdgeAwareGraphConv (layers.py: no learned edge-feature
            # projection, since this encoder has no edge features of its
            # own) and applied once, not per backbone layer. Gives
            # GraphStructuralEncoder direct edge_index access for the
            # first time without duplicating or replacing the backbone's
            # own message passing.
            self.neighbor_projection = nn.Linear(d_model, d_model, bias=False)
            self.aggregate_norm = make_norm(normalization, d_model)
            self.aggregate_output = nn.Linear(d_model, d_model)
            self.aggregate_activation = make_activation(activation)

    def _aggregate_neighbors(
        self,
        hidden: Tensor,
        edge_index: Tensor | None,
        edge_mask: Tensor | None,
        node_mask: Tensor | None,
    ) -> Tensor:
        batch, nodes, width = hidden.shape
        if self.edge_aggregation_source == "self_only":
            # Capacity-matched control: same projection/norm/output layers,
            # same parameter count, but each node's "neighbor" is only
            # itself -- no edge_index is read at all, so any measured
            # effect cannot come from graph connectivity.
            projected = self.neighbor_projection(self.aggregate_norm(hidden))
            update = self.aggregate_output(self.aggregate_activation(projected))
            result = hidden + update
            if node_mask is not None:
                result = result.masked_fill(~node_mask.bool().unsqueeze(-1), 0.0)
            return result
        if edge_index is None or edge_index.numel() == 0:
            return hidden
        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0).expand(batch, -1, -1)
        valid = (
            torch.ones(batch, edge_index.shape[-1], dtype=torch.bool, device=hidden.device)
            if edge_mask is None
            else edge_mask.bool()
        )
        projected = self.neighbor_projection(self.aggregate_norm(hidden))
        aggregate = hidden.new_zeros(batch, nodes, width)
        degree = hidden.new_zeros(batch, nodes, 1)
        for batch_index in range(batch):
            selected = valid[batch_index]
            source = edge_index[batch_index, 0, selected].long()
            target = edge_index[batch_index, 1, selected].long()
            if source.numel() == 0:
                continue
            # Undirected mean aggregation: each edge contributes to both
            # endpoints. Unlike the backbone's directed hydraulic-transport
            # convolution, this encoder's per-node position features (degree,
            # centrality, hop-distance) have no inherent up/downstream
            # direction, so there is no "reverse channel" to mirror here.
            aggregate[batch_index].index_add_(0, target, projected[batch_index, source])
            degree[batch_index].index_add_(0, target, torch.ones(target.numel(), 1, device=hidden.device))
            aggregate[batch_index].index_add_(0, source, projected[batch_index, target])
            degree[batch_index].index_add_(0, source, torch.ones(source.numel(), 1, device=hidden.device))
        update = self.aggregate_output(self.aggregate_activation(aggregate / degree.clamp_min(1.0)))
        result = hidden + update
        if node_mask is not None:
            result = result.masked_fill(~node_mask.bool().unsqueeze(-1), 0.0)
        return result

    def forward(
        self,
        travel_time: Tensor,
        reservoir_reachability: Tensor,
        demand_centrality: Tensor,
        structural_features: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_mask: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        travel = torch.nan_to_num(travel_time.float(), nan=0.0, posinf=0.0, neginf=0.0)
        reachability = torch.nan_to_num(
            reservoir_reachability.float(), nan=0.0, posinf=1.0, neginf=0.0
        )
        centrality = torch.nan_to_num(
            demand_centrality.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        columns = [travel, torch.log1p(travel.clamp_min(0.0)), reachability, centrality]
        if self.structural_feature_dim:
            if structural_features is None:
                raise ValueError(
                    "GraphStructuralEncoder was built with structural_feature_dim="
                    f"{self.structural_feature_dim} but forward() received no structural_features"
                )
            extra = torch.nan_to_num(structural_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if extra.shape[-1] != self.structural_feature_dim:
                raise ValueError("structural_features width does not match structural_feature_dim")
            columns.extend(extra.unbind(dim=-1))
        features = torch.stack(columns, dim=-1)
        scale = features.abs().amax(dim=-2, keepdim=True).clamp_min(1.0)
        hidden = self.network(features / scale)
        if self.use_edge_aggregation:
            hidden = self._aggregate_neighbors(hidden, edge_index, edge_mask, node_mask)
        return hidden


class TemporalEncoder(nn.Module):
    """Encode masked histories using elapsed timestamps rather than array position."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        *,
        normalization: str = "rmsnorm",
        activation: str = "silu",
        elapsed_time_normalization: str = "window_relative",
    ) -> None:
        super().__init__()
        if elapsed_time_normalization not in ELAPSED_TIME_NORMALIZATION_MODES:
            raise ValueError(f"elapsed_time_normalization must be one of {ELAPSED_TIME_NORMALIZATION_MODES}")
        self.elapsed_time_normalization = elapsed_time_normalization
        self.input_projection = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu" if activation.lower() == "gelu" else torch.nn.functional.silu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=make_norm(normalization, d_model),
            enable_nested_tensor=False,
        )
        self.frequency = nn.Parameter(torch.empty(d_model // 2))
        nn.init.normal_(self.frequency, mean=0.0, std=0.02)

    def forward(
        self,
        values: Tensor,
        sensor_mask: Tensor | None = None,
        timestamps: Tensor | None = None,
    ) -> Tensor:
        if values.ndim != 4:
            raise ValueError("temporal features must have shape [batch, time, nodes, features]")
        batch, steps, nodes, _ = values.shape
        finite = torch.isfinite(values).all(dim=-1)
        if sensor_mask is None:
            valid = finite
        else:
            supplied = sensor_mask.bool()
            if supplied.ndim == values.ndim:
                supplied = supplied.all(dim=-1)
            if supplied.shape != finite.shape:
                raise ValueError("sensor mask must have shape [batch, time, nodes] or match features")
            valid = finite & supplied
        clean = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        sequence = self.input_projection(clean).permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        if timestamps is None:
            elapsed = torch.arange(steps, device=values.device, dtype=sequence.dtype)[None].expand(batch, -1)
        else:
            elapsed = timestamps.to(device=values.device, dtype=sequence.dtype)
            if elapsed.shape != (batch, steps):
                raise ValueError("timestamps must have shape [batch, time]")
            elapsed = elapsed - elapsed[:, :1]
            if self.elapsed_time_normalization == "fixed_scale":
                elapsed = elapsed / FIXED_ELAPSED_TIME_SCALE_SECONDS
            else:
                elapsed = elapsed / elapsed.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        elapsed = elapsed[:, None, :].expand(batch, nodes, steps).reshape(batch * nodes, steps)
        phase = elapsed[..., None] * self.frequency.to(sequence.dtype)
        position = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if position.shape[-1] < sequence.shape[-1]:
            position = torch.nn.functional.pad(position, (0, sequence.shape[-1] - position.shape[-1]))
        sequence = sequence + position

        valid = valid.permute(0, 2, 1).reshape(batch * nodes, steps)
        safe_valid = valid.clone()
        safe_valid[~safe_valid.any(dim=1), 0] = True
        encoded = self.encoder(sequence, src_key_padding_mask=~safe_valid)
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), 0.0)
        return _masked_temporal_mean(encoded, valid).reshape(batch, nodes, -1)


class QualityEncoder(TemporalEncoder):
    """Independent temporal encoder for quality and sensor-health channels."""


class StaticFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        normalization: str = "rmsnorm",
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            make_activation(activation),
            make_norm(normalization, d_model),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.projection(
            torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        )
