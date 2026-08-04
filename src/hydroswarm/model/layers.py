"""Edge-aware local transport and latent-token global reasoning blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoders import make_activation, make_norm


class EdgeAwareGraphConv(nn.Module):
    def __init__(
        self,
        d_model: int,
        edge_feature_dim: int,
        *,
        dropout: float,
        normalization: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.source = nn.Linear(d_model, d_model, bias=False)
        self.edge = nn.Linear(edge_feature_dim, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        self.norm = make_norm(normalization, d_model)
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def _aggregate(
        self,
        hidden: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
        *,
        reverse: bool = False,
    ) -> Tensor:
        """Compute the message-passing update (pre-residual). `reverse`
        swaps which endpoint is treated as source vs. target, used by
        DualChannelGraphConv's upstream-diagnostic channel (overnight-
        plan.txt Task 4.3) to reuse this same convolution logic against
        the reversed edge direction without a separate reversed-edge-
        feature set."""

        batch, nodes, width = hidden.shape
        normalized_hidden = self.norm(hidden)
        aggregate = hidden.new_zeros(batch, nodes, width)
        degree = hidden.new_zeros(batch, nodes, 1)
        if edge_index is not None and edge_features is not None:
            if edge_index.ndim == 2:
                edge_index = edge_index.unsqueeze(0).expand(batch, -1, -1)
            if edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0).expand(batch, -1, -1)
            if edge_index.shape[:2] != (batch, 2) or edge_features.shape[:2] != (
                batch,
                edge_index.shape[-1],
            ):
                raise ValueError("edge tensors must be [batch,2,edges] and [batch,edges,features]")
            valid = (
                torch.ones(batch, edge_index.shape[-1], dtype=torch.bool, device=hidden.device)
                if edge_mask is None
                else edge_mask.bool()
            )
            if valid.shape != (batch, edge_index.shape[-1]):
                raise ValueError("edge_mask must have shape [batch, edges]")
            source_row, target_row = (1, 0) if reverse else (0, 1)
            for batch_index in range(batch):
                selected = valid[batch_index]
                source = edge_index[batch_index, source_row, selected].long()
                target = edge_index[batch_index, target_row, selected].long()
                if source.numel() == 0:
                    continue
                if source.min() < 0 or source.max() >= nodes or target.min() < 0 or target.max() >= nodes:
                    raise ValueError("edge_index references a padded or unknown node")
                message = self.source(normalized_hidden[batch_index, source]) + self.edge(
                    edge_features[batch_index, selected].float()
                )
                aggregate[batch_index].index_add_(0, target, message)
                degree[batch_index].index_add_(
                    0, target, torch.ones(target.numel(), 1, device=hidden.device)
                )
        return self.output(self.activation(aggregate / degree.clamp_min(1.0)))

    def forward(
        self,
        hidden: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> Tensor:
        update = self._aggregate(hidden, edge_index, edge_features, edge_mask, reverse=False)
        return hidden + self.dropout(update)


class DualChannelGraphConv(nn.Module):
    """Separately parameterized downstream-transport and upstream-
    diagnostic convolutions, fused with a learned gate (overnight-plan.txt
    Task 4.3). The upstream channel reuses the same edge features with the
    edge direction reversed (source/target swapped) rather than a separate
    reversed-edge-feature set -- documented as the bounded design choice
    this task made, not an oversight."""

    def __init__(
        self,
        d_model: int,
        edge_feature_dim: int,
        *,
        dropout: float,
        normalization: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.downstream = EdgeAwareGraphConv(
            d_model, edge_feature_dim, dropout=dropout, normalization=normalization, activation=activation
        )
        self.upstream = EdgeAwareGraphConv(
            d_model, edge_feature_dim, dropout=dropout, normalization=normalization, activation=activation
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> Tensor:
        downstream_update = self.downstream._aggregate(hidden, edge_index, edge_features, edge_mask, reverse=False)
        upstream_update = self.upstream._aggregate(hidden, edge_index, edge_features, edge_mask, reverse=True)
        gate = torch.sigmoid(self.gate(torch.cat((downstream_update, upstream_update), dim=-1)))
        combined = gate * downstream_update + (1.0 - gate) * upstream_update
        return hidden + self.dropout(combined)


class LatentHydraulicBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        edge_feature_dim: int,
        *,
        dropout: float,
        normalization: str,
        activation: str,
        message_direction: str = "forward_only",
    ) -> None:
        super().__init__()
        if message_direction not in ("forward_only", "dual_gated"):
            raise ValueError(
                f"message_direction must be 'forward_only' or 'dual_gated', got {message_direction!r}"
            )
        conv_cls = DualChannelGraphConv if message_direction == "dual_gated" else EdgeAwareGraphConv
        self.local = conv_cls(
            d_model,
            edge_feature_dim,
            dropout=dropout,
            normalization=normalization,
            activation=activation,
        )
        self.node_norm = make_norm(normalization, d_model)
        self.latent_norm = make_norm(normalization, d_model)
        self.to_latent = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.latent_self = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.to_node = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            make_norm(normalization, d_model),
            nn.Linear(d_model, dim_feedforward),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: Tensor,
        latents: Tensor,
        node_mask: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.local(hidden, edge_index, edge_features, edge_mask)
        latent_update, _ = self.to_latent(
            self.latent_norm(latents),
            self.node_norm(hidden),
            self.node_norm(hidden),
            key_padding_mask=~node_mask,
            need_weights=False,
        )
        latents = latents + latent_update
        latent_update, _ = self.latent_self(
            self.latent_norm(latents), self.latent_norm(latents), self.latent_norm(latents), need_weights=False
        )
        latents = latents + latent_update
        node_update, _ = self.to_node(
            self.node_norm(hidden), self.latent_norm(latents), self.latent_norm(latents), need_weights=False
        )
        hidden = hidden + node_update
        hidden = hidden + self.feed_forward(hidden)
        return hidden.masked_fill(~node_mask.unsqueeze(-1), 0.0), latents
