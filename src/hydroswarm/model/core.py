"""HydroCore: a shared hydraulic representation model with three role outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict

import torch
from torch import Tensor, nn

from .adapters import BottleneckAdapter, RoleHead
from .encoders import GraphStructuralEncoder, QualityEncoder, StaticFeatureEncoder, TemporalEncoder


class HydroBatch(TypedDict, total=False):
    node_features: Tensor
    temporal_features: Tensor
    quality_features: Tensor
    travel_time: Tensor
    reservoir_reachability: Tensor
    demand_centrality: Tensor
    node_mask: Tensor
    sensor_mask: Tensor
    quality_mask: Tensor


class HydroOutput(TypedDict):
    hidden_state: Tensor
    sentinel: Tensor
    scout: Tensor
    strategist: Tensor
    node_mask: Tensor


@dataclass(frozen=True)
class ParameterReport:
    total: int
    trainable: int
    backbone: int
    encoders: int
    adapters: int
    heads: int


class HydroCore(nn.Module):
    """Node-centric transformer shared by Sentinel, Scout and Strategist roles."""

    def __init__(
        self,
        *,
        node_feature_dim: int = 8,
        temporal_feature_dim: int = 6,
        quality_feature_dim: int = 4,
        d_model: int = 384,
        nhead: int = 8,
        dim_feedforward: int = 1152,
        num_layers: int = 10,
        modality_layers: int = 3,
        dropout: float = 0.0,
        sentinel_output_dim: int = 2,
        scout_output_dim: int = 2,
        strategist_output_dim: int = 3,
        adapter_dims: tuple[int, int, int] = (32, 48, 64),
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.num_layers = num_layers
        self.node_encoder = StaticFeatureEncoder(node_feature_dim, d_model)
        self.graph_encoder = GraphStructuralEncoder(d_model)
        self.temporal_encoder = TemporalEncoder(
            temporal_feature_dim, d_model, nhead, dim_feedforward, modality_layers, dropout
        )
        self.quality_encoder = QualityEncoder(
            quality_feature_dim, d_model, nhead, dim_feedforward, modality_layers, dropout
        )
        self.modality_fusion = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )

        roles = ("sentinel", "scout", "strategist")
        outputs = (sentinel_output_dim, scout_output_dim, strategist_output_dim)
        self.adapters = nn.ModuleDict(
            {role: BottleneckAdapter(d_model, width) for role, width in zip(roles, adapter_dims, strict=True)}
        )
        self.heads = nn.ModuleDict(
            {role: RoleHead(d_model, size) for role, size in zip(roles, outputs, strict=True)}
        )

    def forward(self, batch: HydroBatch) -> HydroOutput:
        required = (
            "node_features",
            "temporal_features",
            "quality_features",
            "travel_time",
            "reservoir_reachability",
            "demand_centrality",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"missing HydroBatch fields: {', '.join(missing)}")

        node_features = batch["node_features"]
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [batch, nodes, features]")
        batch_size, nodes, _ = node_features.shape
        node_mask = batch.get(
            "node_mask", torch.ones(batch_size, nodes, dtype=torch.bool, device=node_features.device)
        ).bool()
        if node_mask.shape != (batch_size, nodes):
            raise ValueError("node_mask must have shape [batch, nodes]")

        static = self.node_encoder(node_features)
        graph = self.graph_encoder(
            batch["travel_time"], batch["reservoir_reachability"], batch["demand_centrality"]
        )
        temporal = self.temporal_encoder(batch["temporal_features"], batch.get("sensor_mask"))
        quality = self.quality_encoder(batch["quality_features"], batch.get("quality_mask"))
        hidden = self.modality_fusion(torch.cat((static, graph, temporal, quality), dim=-1))

        safe_mask = node_mask.clone()
        all_missing = ~safe_mask.any(dim=1)
        safe_mask[all_missing, 0] = True
        hidden = self.backbone(hidden, src_key_padding_mask=~safe_mask)
        hidden = hidden.masked_fill(~node_mask.unsqueeze(-1), 0.0)

        role_outputs = {
            role: self.heads[role](self.adapters[role](hidden)).masked_fill(
                ~node_mask.unsqueeze(-1), 0.0
            )
            for role in self.adapters
        }
        return HydroOutput(
            hidden_state=hidden,
            sentinel=role_outputs["sentinel"],
            scout=role_outputs["scout"],
            strategist=role_outputs["strategist"],
            node_mask=node_mask,
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def parameter_report(self) -> ParameterReport:
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        encoders = count(self.node_encoder) + count(self.graph_encoder)
        encoders += count(self.temporal_encoder) + count(self.quality_encoder) + count(self.modality_fusion)
        return ParameterReport(
            total=self.parameter_count(),
            trainable=self.parameter_count(trainable_only=True),
            backbone=count(self.backbone),
            encoders=encoders,
            adapters=count(self.adapters),
            heads=count(self.heads),
        )

    def parameter_report_dict(self) -> dict[str, int]:
        return asdict(self.parameter_report())
