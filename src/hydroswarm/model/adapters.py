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


class CapacityMatchedProjection(nn.Module):
    """Generic, non-structural residual MLP widening for HydroCore's
    experimental `localizer_capacity_hidden_dim` knob (see core.py's
    `HydroCore.__init__`).

    `exp/physics-informed-localizer-validation`'s Phase 2 capacity-matched
    control (`A_CAPACITY_MATCHED`) needs a way to add roughly the same
    parameter count as `CandidateConditionedLocalizer` (candidate_localizer.py)
    contributes to Arms B/C, without adding candidate conditioning,
    candidate-to-sensor structure, physics features, graph information, or
    source-specific topology information -- i.e. capacity alone. Same
    residual-bottleneck shape as `BottleneckAdapter` above (norm -> down ->
    activation -> up, added back to the input), but with an unconstrained
    `hidden_dim` (`BottleneckAdapter` is deliberately capped to
    32-64 for role specialization; here the whole point is to size the
    block to hit a specific target parameter count) and GELU zero-inited
    only on `down` rather than `up`, since this block's only job is to
    be a real extra capacity path the default per-node head can use
    (`source_node_head` on top of `sentinel_nodes + this(sentinel_nodes)`),
    not a residual that starts as a no-op. Reads only its own input
    tensor -- no mask, edge, or feature-schema argument -- so it cannot
    smuggle in structural or physics information by construction.
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.up = nn.Linear(hidden_dim, d_model)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.up(self.hidden_norm(self.activation(self.down(self.norm(hidden)))))
