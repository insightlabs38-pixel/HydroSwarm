"""Parameter-efficient role adapters and prediction heads."""

from __future__ import annotations

from torch import Tensor, nn


class BottleneckAdapter(nn.Module):
    """Residual role specialization through a 32--64 dimensional bottleneck."""

    def __init__(self, d_model: int, bottleneck_dim: int) -> None:
        super().__init__()
        if not 32 <= bottleneck_dim <= 64:
            raise ValueError("role adapter bottleneck_dim must be between 32 and 64")
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.up(self.activation(self.down(self.norm(hidden))))


class RoleHead(nn.Module):
    """A normalized role-specific output projection."""

    def __init__(self, d_model: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))

    def forward(self, hidden: Tensor) -> Tensor:
        return self.network(hidden)
