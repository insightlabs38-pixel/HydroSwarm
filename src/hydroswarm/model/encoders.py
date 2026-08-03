"""Feature encoders for the HydroSwarm foundation model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _masked_temporal_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GraphStructuralEncoder(nn.Module):
    """Encode hydraulic structure: travel time, reservoir reachability and demand centrality."""

    feature_names = ("travel_time", "reservoir_reachability", "demand_centrality")

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        # Log travel time is retained alongside the three raw, normalized features.
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        travel_time: Tensor,
        reservoir_reachability: Tensor,
        demand_centrality: Tensor,
    ) -> Tensor:
        travel_time = torch.nan_to_num(travel_time.float(), nan=0.0, posinf=0.0, neginf=0.0)
        reachability = torch.nan_to_num(
            reservoir_reachability.float(), nan=0.0, posinf=1.0, neginf=0.0
        )
        centrality = torch.nan_to_num(demand_centrality.float(), nan=0.0, posinf=0.0, neginf=0.0)
        features = torch.stack(
            (travel_time, torch.log1p(travel_time.clamp_min(0.0)), reachability, centrality), dim=-1
        )
        scale = features.abs().amax(dim=-2, keepdim=True).clamp_min(1.0)
        return self.network(features / scale)


class TemporalEncoder(nn.Module):
    """Encode masked multivariate sensor histories into one state per node."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.frequency = nn.Parameter(torch.empty(d_model // 2))
        nn.init.normal_(self.frequency, mean=0.0, std=0.02)

    def forward(self, values: Tensor, sensor_mask: Tensor | None = None) -> Tensor:
        if values.ndim != 4:
            raise ValueError("temporal features must have shape [batch, time, nodes, features]")
        batch, steps, nodes, _ = values.shape
        finite = torch.isfinite(values).all(dim=-1)
        if sensor_mask is None:
            valid = finite
        else:
            supplied_mask = sensor_mask.bool()
            if supplied_mask.ndim == values.ndim:
                supplied_mask = supplied_mask.all(dim=-1)
            if supplied_mask.shape != finite.shape:
                raise ValueError("sensor mask must have shape [batch, time, nodes] or match features")
            valid = finite & supplied_mask
        clean = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        sequence = self.input_projection(clean).permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        position = torch.arange(steps, device=values.device, dtype=sequence.dtype)
        phase = position[:, None] * self.frequency[None, :].to(sequence.dtype)
        position_encoding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if position_encoding.shape[-1] < sequence.shape[-1]:
            position_encoding = torch.nn.functional.pad(
                position_encoding, (0, sequence.shape[-1] - position_encoding.shape[-1])
            )
        sequence = sequence + position_encoding[None, :, :]

        valid = valid.permute(0, 2, 1).reshape(batch * nodes, steps)
        safe_valid = valid.clone()
        all_missing = ~safe_valid.any(dim=1)
        safe_valid[all_missing, 0] = True
        encoded = self.encoder(sequence, src_key_padding_mask=~safe_valid)
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), 0.0)
        return _masked_temporal_mean(encoded, valid).reshape(batch, nodes, -1)


class QualityEncoder(TemporalEncoder):
    """Temporal encoder dedicated to water-quality observations."""


class StaticFeatureEncoder(nn.Module):
    """Encode optional node metadata without allowing NaNs into attention."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, values: Tensor) -> Tensor:
        return self.projection(torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0))
