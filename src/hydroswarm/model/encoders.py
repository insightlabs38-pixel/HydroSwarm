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


class GraphStructuralEncoder(nn.Module):
    feature_names = ("travel_time", "reservoir_reachability", "demand_centrality")

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        *,
        normalization: str = "rmsnorm",
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            make_activation(activation),
            make_norm(normalization, hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        travel_time: Tensor,
        reservoir_reachability: Tensor,
        demand_centrality: Tensor,
    ) -> Tensor:
        travel = torch.nan_to_num(travel_time.float(), nan=0.0, posinf=0.0, neginf=0.0)
        reachability = torch.nan_to_num(
            reservoir_reachability.float(), nan=0.0, posinf=1.0, neginf=0.0
        )
        centrality = torch.nan_to_num(
            demand_centrality.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        features = torch.stack(
            (travel, torch.log1p(travel.clamp_min(0.0)), reachability, centrality), dim=-1
        )
        scale = features.abs().amax(dim=-2, keepdim=True).clamp_min(1.0)
        return self.network(features / scale)


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
    ) -> None:
        super().__init__()
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
