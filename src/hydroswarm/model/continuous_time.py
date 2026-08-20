"""HydroCore-v5 Milestone 9.1 preflight: continuous-time TemporalDynamicsBase
interface and the GRAPH_ODE / GRAPH_CDE / GRAPH_SDE candidate architectures.

Experiment-scoped (docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md).
Not wired into any production runtime path: HydroCore's `temporal_dynamics`
constructor argument defaults to None (original TemporalEncoder/
QualityEncoder behavior) for every existing caller; only experiment scripts
under scripts/hydrocore_v5/ construct one of the classes below explicitly.

Time semantics (frozen, preflight protocol Section 3):

    t = (timestamps_seconds - timestamps_seconds[:, :1]) / 86_400.0

translation-invariant (subtracts each sequence's own first timestamp before
any scaling) and physically scaled (a fixed divisor, never a per-window
`.amax()` renormalization) -- reusing the exact constant M8.7's
AGE_FIX_PLUS_RELATIVE_TIME arm already validated as origin-invariant AND
duration-preserving, rather than inventing a new magnitude.

torchdiffeq/torchcde/torchsde are optional research dependencies (the
`continuous-time` extra in pyproject.toml) and are imported lazily below so
`import hydroswarm` never requires them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

from .encoders import FIXED_ELAPSED_TIME_SCALE_SECONDS, QualityEncoder, TemporalEncoder

try:
    import torchdiffeq
except ImportError as exc:  # pragma: no cover - optional dependency path
    torchdiffeq = None
    _TORCHDIFFEQ_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCHDIFFEQ_IMPORT_ERROR = None

try:
    import torchcde
except ImportError as exc:  # pragma: no cover - optional dependency path
    torchcde = None
    _TORCHCDE_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCHCDE_IMPORT_ERROR = None

try:
    import torchsde
except ImportError as exc:  # pragma: no cover - optional dependency path
    torchsde = None
    _TORCHSDE_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCHSDE_IMPORT_ERROR = None


def _require(module: Any, error: ImportError | None, package: str) -> None:
    if module is None:
        raise ImportError(
            f"{package} is required for this M9.1 preflight continuous-time variant but is "
            f"not installed. Install the optional research dependency group: "
            f"pip install -e '.[continuous-time]'."
        ) from error


def compute_relative_physical_time(timestamps: Tensor) -> Tensor:
    """Origin-independent, duration-preserving physical time (frozen
    preflight protocol Section 3; reuses M8.7's fixed_scale convention
    exactly -- same constant, same subtract-then-divide order).

    `timestamps`: [batch, steps] elapsed seconds (already incident-relative
    per HydraulicFeatureBuilder's own contract, never raw epoch). Returns
    [batch, steps] in fixed units of FIXED_ELAPSED_TIME_SCALE_SECONDS
    (86,400s = 1.0)."""

    if timestamps.ndim != 2:
        raise ValueError("timestamps must have shape [batch, steps]")
    elapsed = timestamps - timestamps[:, :1]
    return elapsed / FIXED_ELAPSED_TIME_SCALE_SECONDS


def _valid_mask(values: Tensor, supplied_mask: Tensor | None) -> Tensor:
    """[batch, steps, nodes, features] (+ optional [batch, steps, nodes] or
    matching-shape mask) -> [batch, steps, nodes] bool. Matches
    encoders.TemporalEncoder.forward's own validity convention exactly."""

    finite = torch.isfinite(values).all(dim=-1)
    if supplied_mask is None:
        return finite
    supplied = supplied_mask.bool()
    if supplied.ndim == values.ndim:
        supplied = supplied.all(dim=-1)
    return finite & supplied


def _aggregate_edge_mean(
    h: Tensor,
    edge_index: Tensor | None,
    edge_features: Tensor | None,
    edge_feature_dim: int,
    edge_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Mean-aggregate neighbor states and edge features into every target
    node, matching layers.EdgeAwareGraphConv._aggregate's own scatter-mean
    convention exactly (same edge_index=[batch,2,edges] source/target-row
    layout, same per-batch-item Python loop, same index_add_ + degree-clamp
    pattern) -- reused rather than reinvented so every M9.1 continuous-time
    vector field aggregates neighbors identically to the existing graph
    backbone. Returns (h_aggregate, edge_aggregate), each mean-pooled over
    the (possibly zero) incoming edges of every node; both are exact zeros
    for a node with no valid incoming edge (degree-clamped to 1 for the
    division, matching the existing convention's fail-safe)."""

    batch, nodes, width = h.shape
    h_aggregate = h.new_zeros(batch, nodes, width)
    edge_aggregate = h.new_zeros(batch, nodes, edge_feature_dim)
    if edge_index is None or edge_features is None:
        return h_aggregate, edge_aggregate
    degree = h.new_zeros(batch, nodes, 1)
    if edge_index.ndim == 2:
        edge_index = edge_index.unsqueeze(0).expand(batch, -1, -1)
    if edge_features.ndim == 2:
        edge_features = edge_features.unsqueeze(0).expand(batch, -1, -1)
    valid = (
        torch.ones(batch, edge_index.shape[-1], dtype=torch.bool, device=h.device)
        if edge_mask is None
        else edge_mask.bool()
    )
    for batch_index in range(batch):
        selected = valid[batch_index]
        source = edge_index[batch_index, 0, selected].long()
        target = edge_index[batch_index, 1, selected].long()
        if source.numel() == 0:
            continue
        h_aggregate[batch_index].index_add_(0, target, h[batch_index, source])
        edge_aggregate[batch_index].index_add_(
            0, target, edge_features[batch_index, selected].float()
        )
        degree[batch_index].index_add_(
            0, target, torch.ones(target.numel(), 1, device=h.device)
        )
    safe_degree = degree.clamp_min(1.0)
    return h_aggregate / safe_degree, edge_aggregate / safe_degree


class TemporalDynamicsBase(nn.Module, ABC):
    """Common interface every M9.1 candidate temporal-latent-production
    module implements, replacing HydroCore's temporal_encoder+quality_encoder
    pair at a single seam (preflight protocol Section 2). Every input/output
    shape below matches exactly what HydroCore.forward already has available
    at that call site."""

    @abstractmethod
    def forward(
        self,
        temporal_features: Tensor,
        quality_features: Tensor,
        sensor_mask: Tensor | None,
        quality_mask: Tensor | None,
        timestamps: Tensor | None,
        node_mask: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (temporal_latent, quality_latent), each [batch, nodes, d_model]."""


class CurrentTemporalDynamics(TemporalDynamicsBase):
    """Wraps the exact original TemporalEncoder+QualityEncoder pair for API
    uniformity in experiment scripts only. NOT wired into HydroCore's
    default construction path -- HydroCore builds its own temporal_encoder/
    quality_encoder directly whenever temporal_dynamics=None (the default
    for every existing caller). This wrapper exists solely so a preflight
    script can iterate CURRENT/ODE/CDE/SDE through one shared interface; its
    numerical output is required to match vanilla HydroCore's own two-encoder
    computation exactly (tested directly, preflight protocol Section 9.12).
    Ignores node_mask/edge_index/edge_features/edge_mask exactly as the
    original two encoders do -- CURRENT is not graph-aware in its temporal
    pathway, matching the architecture-seam report's own finding."""

    def __init__(
        self,
        temporal_feature_dim: int,
        quality_feature_dim: int,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        normalization: str = "rmsnorm",
        activation: str = "silu",
        elapsed_time_normalization: str = "window_relative",
    ) -> None:
        super().__init__()
        shared_args = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
            normalization=normalization,
            activation=activation,
            elapsed_time_normalization=elapsed_time_normalization,
        )
        self.temporal_encoder = TemporalEncoder(temporal_feature_dim, **shared_args)
        self.quality_encoder = QualityEncoder(quality_feature_dim, **shared_args)

    def forward(
        self,
        temporal_features: Tensor,
        quality_features: Tensor,
        sensor_mask: Tensor | None,
        quality_mask: Tensor | None,
        timestamps: Tensor | None,
        node_mask: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        temporal = self.temporal_encoder(temporal_features, sensor_mask, timestamps)
        quality = self.quality_encoder(quality_features, quality_mask, timestamps)
        return temporal, quality


class GraphVectorField(nn.Module):
    """Shared graph-aware drift module: dh/dt = f_theta(h,
    aggregate_neighbors(h, edge_ij), edge_context) -- reused, structurally
    identical, by both GRAPH_ODE and GRAPH_SDE (preflight protocol Sections
    4/6: "the drift should be as directly comparable to the Graph ODE drift
    as practical"; SDE's only qualitatively new mechanism is its diffusion
    term). `mlp_width` is the sole parameter-matching knob searched in
    Section 8 of the protocol -- never tuned against predictive quality."""

    def __init__(self, hidden_dim: int, edge_feature_dim: int, *, mlp_width: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim
        self.self_projection = nn.Linear(hidden_dim, mlp_width)
        self.neighbor_projection = nn.Linear(hidden_dim, mlp_width)
        self.edge_projection = nn.Linear(edge_feature_dim, mlp_width)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.SiLU(),
            nn.Linear(mlp_width, hidden_dim),
        )

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> Tensor:
        h_aggregate, edge_aggregate = _aggregate_edge_mean(
            h, edge_index, edge_features, self.edge_feature_dim, edge_mask
        )
        combined = (
            self.self_projection(h)
            + self.neighbor_projection(h_aggregate)
            + self.edge_projection(edge_aggregate)
        )
        return self.mlp(combined)


class _FirstValidEvidenceInitialState(nn.Module):
    """Shared initial-condition construction for GRAPH_ODE/GRAPH_SDE
    (preflight correction Issue 1, docs/evaluation/
    HYDROCORE_V5_M9_1_PREFLIGHT_CORRECTION.md): for each batch item / node /
    modality independently, uses ONLY that modality's own first causally
    valid observation (earliest step index where sensor_mask/quality_mask is
    True) to build h0 -- never a pooled/averaged combination of the whole
    prefix (that was the corrected Issue-1 defect, `_PooledEvidenceInitialState`,
    removed), never array position 0 blindly (a node's earliest ARRAY
    position may itself be invalid/missing). The two modalities' first-valid
    indices may differ and are resolved independently. A node with no valid
    observation in a modality gets a deterministic zero vector for that
    modality's component (no learned missing-state token -- kept minimal per
    the correction's own scope). This is what makes GRAPH_ODE/GRAPH_SDE's
    entire dependence on raw evidence values happen at a single point t_0 --
    after that they evolve purely under the graph-aware drift (+diffusion
    for SDE) through the actual physical elapsed duration, never re-reading
    evidence mid-trajectory. This is the deliberate ODE/SDE-vs-CDE
    distinction the preflight protocol Section 6 of the milestone
    instructions draws ("ODE: latent dynamics evolve continuously" from a
    point condition vs. "CDE: latent dynamics are continuously controlled by
    the observed evidence path") -- GRAPH_CDE (below) instead stays
    continuously driven by evidence throughout via its control path."""

    def __init__(self, temporal_feature_dim: int, quality_feature_dim: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(temporal_feature_dim + quality_feature_dim, d_model)

    @staticmethod
    def _first_valid(values: Tensor, valid: Tensor) -> Tensor:
        """values: [batch, steps, nodes, features], valid: [batch, steps,
        nodes] bool -> [batch, nodes, features]: each node's feature vector
        at its OWN earliest valid step (never array position 0 unless that
        position is itself the earliest valid one), or an exact zero vector
        for a node with no valid step at all."""

        batch, _steps, nodes, features = values.shape
        any_valid = valid.any(dim=1)  # [batch, nodes]
        # torch.argmax returns the index of the FIRST occurrence of the max
        # value along the given dim (documented tie-breaking behavior) --
        # over a 0.0/1.0-valued mask this is exactly the first True index;
        # when no step is valid it returns 0, which `any_valid` below then
        # zeroes out rather than treating as real evidence.
        first_index = valid.to(values.dtype).argmax(dim=1)  # [batch, nodes]
        index = first_index[:, None, :, None].expand(batch, 1, nodes, features)
        gathered = torch.gather(values, 1, index).squeeze(1)  # [batch, nodes, features]
        return gathered * any_valid.unsqueeze(-1).to(values.dtype)

    def forward(
        self,
        temporal_features: Tensor,
        quality_features: Tensor,
        sensor_mask: Tensor | None,
        quality_mask: Tensor | None,
    ) -> Tensor:
        temporal_valid = _valid_mask(temporal_features, sensor_mask)
        quality_valid = _valid_mask(quality_features, quality_mask)
        temporal_clean = torch.nan_to_num(temporal_features.float())
        quality_clean = torch.nan_to_num(quality_features.float())
        temporal_first = self._first_valid(temporal_clean, temporal_valid)
        quality_first = self._first_valid(quality_clean, quality_valid)
        return self.projection(torch.cat((temporal_first, quality_first), dim=-1))


def _physical_delta(timestamps: Tensor | None, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """[batch] physical elapsed duration (frozen time units, Section 3) of
    the evidence window actually available -- 0.0 (a well-defined, tested
    degenerate case; see depth-1 handling) when there is only one or zero
    distinguishable timestamp. Never per-window renormalized: two examples
    with different real elapsed durations get different `delta`, which is
    exactly what makes the physical-gap-plumbing test meaningful."""

    if timestamps is None:
        return torch.ones(batch, device=device, dtype=dtype)
    relative = compute_relative_physical_time(timestamps)
    return (relative[:, -1] - relative[:, 0]).clamp_min(0.0).to(dtype)


class GraphODEDynamics(TemporalDynamicsBase):
    """GRAPH_ODE (preflight protocol Section 4): dh_i/dt = f_theta(h_i,
    aggregate_neighbors(h_j, edge_ij), static context), integrated from a
    pooled-evidence initial state through the evidence window's actual
    physical elapsed duration. Solver: torchdiffeq.odeint (plain backprop,
    not adjoint), method="dopri5", rtol=1e-3, atol=1e-4 (frozen, Section 4).

    Batched varying-duration integration: rather than calling odeint with a
    different final time per batch example (unsupported -- odeint integrates
    the whole batch to one shared set of times), the drift is rescaled by
    each example's own physical `delta` and integrated over a SHARED nominal
    interval tau in [0, 1]: dh/dtau = delta_b * f_theta(h, ...) is exactly
    equivalent (by the standard time-rescaling substitution t = delta_b *
    tau) to dh/dt = f_theta(h, ...) for t in [0, delta_b]. This is exact, not
    an approximation, and is what makes different physical durations produce
    different integrated results (delta_b=0, e.g. depth-1 with only one
    timestamp, collapses the ODE to the identity map on h0 -- deterministic,
    finite, tested explicitly)."""

    def __init__(
        self,
        temporal_feature_dim: int,
        quality_feature_dim: int,
        *,
        d_model: int,
        edge_feature_dim: int,
        mlp_width: int,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        max_num_steps: int = 2000,
    ) -> None:
        super().__init__()
        _require(torchdiffeq, _TORCHDIFFEQ_IMPORT_ERROR, "torchdiffeq")
        self.d_model = d_model
        self.initial_state = _FirstValidEvidenceInitialState(
            temporal_feature_dim, quality_feature_dim, d_model
        )
        self.field = GraphVectorField(d_model, edge_feature_dim, mlp_width=mlp_width)
        self.temporal_head = nn.Linear(d_model, d_model)
        self.quality_head = nn.Linear(d_model, d_model)
        self.rtol = rtol
        self.atol = atol
        # M9.1 scientific protocol Section 7 (docs/evaluation/
        # HYDROCORE_V5_M9_1_PROTOCOL.md): frozen engineering safety bound on
        # worst-case dopri5 adaptive-solver step count during real training/
        # evaluation -- the preflight's own smoke tests never ran enough
        # steps to need this. Exceeding it raises inside torchdiffeq itself
        # (the runner is responsible for catching that and recording
        # SOLVER_STEP_LIMIT_EXCEEDED, per the protocol); this bound is never
        # adjusted to change predictive output, only to correct a
        # demonstrated non-convergence/pathological-runtime failure, and any
        # such change must be logged as a dated protocol addendum first.
        self.max_num_steps = max_num_steps

    def forward(
        self,
        temporal_features: Tensor,
        quality_features: Tensor,
        sensor_mask: Tensor | None,
        quality_mask: Tensor | None,
        timestamps: Tensor | None,
        node_mask: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        h0 = self.initial_state(temporal_features, quality_features, sensor_mask, quality_mask)
        delta = _physical_delta(timestamps, h0.shape[0], h0.device, h0.dtype)

        def field_fn(tau: Tensor, h: Tensor) -> Tensor:
            raw = self.field(h, edge_index, edge_features, edge_mask)
            return delta.view(-1, 1, 1) * raw

        eval_times = h0.new_tensor([0.0, 1.0])
        assert torchdiffeq is not None
        trajectory = torchdiffeq.odeint(
            field_fn, h0, eval_times, method="dopri5", rtol=self.rtol, atol=self.atol,
            options={"max_num_steps": self.max_num_steps},
        )
        h_final = trajectory[-1].masked_fill(~node_mask.unsqueeze(-1), 0.0)
        return self.temporal_head(h_final), self.quality_head(h_final)


class GraphCDEField(nn.Module):
    """torchcde's required field contract: f_theta(t, z) -> [..., hidden_dim,
    input_channels] (a MATRIX, since dh = f_theta(h,G) dX_t is a
    matrix-vector product against dX/dt internally). Graph-aware via the
    same _aggregate_edge_mean primitive GraphVectorField uses, applied after
    reshaping the flat (batch*nodes, hidden_dim) state torchcde operates on
    back to (batch, nodes, hidden_dim). Output is bounded via
    `tanh(...) * output_scale` -- a plain engineering stability measure (an
    unbounded matrix field under a fixed-step rk4 solver is a common source
    of blow-up with linear-interpolation control paths that have derivative
    discontinuities at every knot), not a predictive-accuracy choice."""

    def __init__(
        self,
        hidden_dim: int,
        input_channels: int,
        edge_feature_dim: int,
        *,
        mlp_width: int,
        output_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        self.edge_feature_dim = edge_feature_dim
        self.output_scale = output_scale
        self.self_projection = nn.Linear(hidden_dim, mlp_width)
        self.neighbor_projection = nn.Linear(hidden_dim, mlp_width)
        self.edge_projection = nn.Linear(edge_feature_dim, mlp_width)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.SiLU(),
            nn.Linear(mlp_width, hidden_dim * input_channels),
        )
        self._graph_context: dict[str, Any] = {}

    def set_graph_context(
        self,
        batch: int,
        nodes: int,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> None:
        self._graph_context = {
            "batch": batch,
            "nodes": nodes,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "edge_mask": edge_mask,
        }

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        ctx = self._graph_context
        batch, nodes = ctx["batch"], ctx["nodes"]
        h = z.view(batch, nodes, self.hidden_dim)
        h_aggregate, edge_aggregate = _aggregate_edge_mean(
            h, ctx["edge_index"], ctx["edge_features"], self.edge_feature_dim, ctx["edge_mask"]
        )
        combined = (
            self.self_projection(h)
            + self.neighbor_projection(h_aggregate)
            + self.edge_projection(edge_aggregate)
        )
        out = self.mlp(combined).view(batch, nodes, self.hidden_dim, self.input_channels)
        out = torch.tanh(out) * self.output_scale
        return out.reshape(batch * nodes, self.hidden_dim, self.input_channels)


class GraphCDEDynamics(TemporalDynamicsBase):
    """GRAPH_CDE (preflight protocol Section 5): dh_t = f_theta(h_t, G) dX_t,
    X_t built by concatenating relative physical time with
    temporal_features/quality_features per node, per causally-available step
    -- so X_t itself carries evidence continuously, unlike GRAPH_ODE/SDE's
    first-valid-then-evolved initial condition. Interpolation: LINEAR
    (torchcde.linear_interpolation_coeffs), chosen for strictly local support
    (a future knot cannot alter the interpolant's value or derivative at an
    earlier query point), which is what makes the mandatory causality gate
    (preflight protocol Section 9.4) provable rather than merely
    approximately true. Solved via torchcde.cdeint(method="rk4"), a
    fixed-step solver (the standard pairing for a kinked linear-interpolation
    path), with `step_size` frozen below.

    Mask-aware control path (preflight correction Issue 2, docs/evaluation/
    HYDROCORE_V5_M9_1_PREFLIGHT_CORRECTION.md): HydraulicFeatureBuilder/
    pad_graph_batch NaN-replace-with-zero every invalid temporal/quality
    position at corpus-generation time, so a bare feature value of 0.0 is
    structurally ambiguous between "genuinely observed zero" and "missing,
    zero-filled" -- sensor_mask/quality_mask are the only remaining signal
    distinguishing them. X_t therefore includes two explicit validity
    channels, `sensor_valid`/`quality_valid` (0.0/1.0 per [batch,time,node],
    via the same `_valid_mask` convention used elsewhere in this module),
    alongside relative_time/temporal_features/quality_features --
    `input_channels = 1 + temporal_feature_dim + quality_feature_dim + 2`.

    Causal cutoff: `cutoff_index` (0-indexed, inclusive) selects which
    leading steps of the ALREADY globally-ordered step axis are used to
    build the interpolation -- filtering happens BEFORE
    `linear_interpolation_coeffs` is called, never by building one
    full-sequence interpolant and integrating only partway. Defaults to
    `steps - 1` (use every available step, matching HydroCore's existing
    "evaluate now" semantics) when not explicitly supplied; the mandatory
    causality test (preflight protocol Section 9.4) supplies it explicitly
    so a prefix P and P-plus-future-observation can both be evaluated at the
    identical cutoff and compared. The depth-1 degenerate-control expansion
    (below) duplicates the ENTIRE per-step vector, including the two
    validity channels, across the synthetic two-point path -- an invalid
    single observation's validity channel stays 0.0 in both duplicated
    points, never silently promoted to valid."""

    #: Fixed rk4 step count per unit of the interpolation's own index axis
    #: (which advances by exactly 1 per real observation step) -- frozen at
    #: preflight time for numerical correctness (resolving each individual
    #: knot-to-knot interval, never leaving one interval unresolved by the
    #: fixed-step solver), not tuned for predictive accuracy.
    STEP_SIZE = 0.25

    def __init__(
        self,
        temporal_feature_dim: int,
        quality_feature_dim: int,
        *,
        d_model: int,
        edge_feature_dim: int,
        mlp_width: int,
        step_size: float = STEP_SIZE,
    ) -> None:
        super().__init__()
        _require(torchcde, _TORCHCDE_IMPORT_ERROR, "torchcde")
        self.d_model = d_model
        # +2: sensor_valid, quality_valid (preflight correction Issue 2).
        self.input_channels = 1 + temporal_feature_dim + quality_feature_dim + 2
        self.initial_projection = nn.Linear(self.input_channels, d_model)
        self.field = GraphCDEField(
            d_model, self.input_channels, edge_feature_dim, mlp_width=mlp_width
        )
        self.temporal_head = nn.Linear(d_model, d_model)
        self.quality_head = nn.Linear(d_model, d_model)
        self.step_size = step_size

    def _build_path_values(
        self,
        temporal_features: Tensor,
        quality_features: Tensor,
        sensor_mask: Tensor | None,
        quality_mask: Tensor | None,
        timestamps: Tensor | None,
    ) -> Tensor:
        """[batch, steps, nodes, 1 + temporal + quality + 2] -- NaN-safe
        (missing/invalid feature entries filled 0.0, matching every other
        feature builder in this codebase's own nan_to_num convention) PLUS
        two explicit 0.0/1.0 validity channels (preflight correction Issue
        2) so a genuinely observed zero reading (feature=0.0, valid=1.0)
        stays distinguishable from a missing, zero-filled position
        (feature=0.0, valid=0.0) -- linear interpolation still requires real
        numeric knot values for every channel, including the validity
        channels themselves, which is exactly why they are represented as
        an explicit 0.0/1.0 channel rather than left implicit."""

        batch, steps, nodes, _ = temporal_features.shape
        if timestamps is None:
            relative_time = torch.arange(steps, device=temporal_features.device, dtype=torch.float32)
            relative_time = relative_time[None, :].expand(batch, -1)
        else:
            relative_time = compute_relative_physical_time(timestamps)
        time_channel = relative_time[:, :, None, None].expand(batch, steps, nodes, 1)
        temporal_clean = torch.nan_to_num(temporal_features.float())
        quality_clean = torch.nan_to_num(quality_features.float())
        sensor_valid = _valid_mask(temporal_features, sensor_mask).to(temporal_clean.dtype).unsqueeze(-1)
        quality_valid = _valid_mask(quality_features, quality_mask).to(quality_clean.dtype).unsqueeze(-1)
        return torch.cat(
            (time_channel, temporal_clean, quality_clean, sensor_valid, quality_valid), dim=-1
        )

    def forward(
        self,
        temporal_features: Tensor,
        quality_features: Tensor,
        sensor_mask: Tensor | None,
        quality_mask: Tensor | None,
        timestamps: Tensor | None,
        node_mask: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
        *,
        cutoff_index: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, steps, nodes, _ = temporal_features.shape
        assert torchcde is not None
        if cutoff_index is None:
            cutoff_index = steps - 1
        cutoff_index = max(0, min(cutoff_index, steps - 1))
        kept = cutoff_index + 1

        path_values = self._build_path_values(
            temporal_features, quality_features, sensor_mask, quality_mask, timestamps
        )
        # Causal filter BEFORE interpolation coefficients are ever computed --
        # this is the entire causality guarantee (Section 9.4).
        path_values = path_values[:, :kept]
        if kept == 1:
            # Depth-1 degenerate control: torchcde.linear_interpolation_coeffs
            # requires length >= 2, so a single real observation is held
            # constant across a synthetic two-point path spanning zero
            # index-duration -- finite, deterministic, matches the frozen
            # "evaluate at the single available point" semantics rather than
            # crashing.
            path_values = path_values.expand(batch, 2, nodes, path_values.shape[-1])

        flat_path = path_values.permute(0, 2, 1, 3).reshape(batch * nodes, path_values.shape[1], -1)
        coeffs = torchcde.linear_interpolation_coeffs(flat_path)
        control_path = torchcde.LinearInterpolation(coeffs)

        h0 = self.initial_projection(control_path.evaluate(control_path.interval[0]))
        self.field.set_graph_context(batch, nodes, edge_index, edge_features, edge_mask)
        trajectory = torchcde.cdeint(
            X=control_path,
            func=self.field,
            z0=h0,
            t=control_path.interval,
            method="rk4",
            options={"step_size": self.step_size},
        )
        # torchcde.cdeint returns shape (..., len(t), hidden_channels) -- the
        # time axis is second-to-last (unlike torchdiffeq.odeint, whose time
        # axis is axis 0), since `...` here mirrors z0's own leading
        # (batch*nodes,) shape.
        h_final = trajectory[..., -1, :].view(batch, nodes, self.d_model)
        h_final = h_final.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        return self.temporal_head(h_final), self.quality_head(h_final)


class _SDEDrift(nn.Module):
    """torchsde contract: an object exposing .f(t, y) and .g(t, y), y always
    2-D [torchsde_batch, state_size]. State here folds (nodes, hidden_dim)
    into one flat `state_size = nodes * hidden_dim` axis, torchsde's own
    batch axis stays the real example batch -- f/g reshape to
    (batch, nodes, hidden_dim) internally to run the graph-aware drift, then
    flatten back. Drift reuses GraphVectorField (structurally identical to
    GRAPH_ODE's own drift, per Section 6's "as directly comparable ... as
    practical"); diffusion is diagonal/node-local, non-negative, and bounded
    (`diffusion_scale * sigmoid(...)`) -- structured per Section 6's
    diffusion-structure requirement, never a dense/coupled covariance.
    `delta`/`sqrt(delta)` implement the same batched varying-duration
    time-rescaling trick as GRAPH_ODE (`dh = delta*f dtau +
    sqrt(delta)*g dW_tau` is the exact Brownian-motion time-change
    equivalent of integrating the un-rescaled SDE over [0, delta])."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        field: GraphVectorField,
        diffusion_net: nn.Sequential,
        *,
        batch: int,
        nodes: int,
        hidden_dim: int,
        delta: Tensor,
        diffusion_scale: float,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
    ) -> None:
        super().__init__()
        self.field = field
        self.diffusion_net = diffusion_net
        self.batch = batch
        self.nodes = nodes
        self.hidden_dim = hidden_dim
        self.delta = delta
        self.diffusion_scale = diffusion_scale
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.edge_mask = edge_mask

    def f(self, t: Tensor, y: Tensor) -> Tensor:
        h = y.view(self.batch, self.nodes, self.hidden_dim)
        raw = self.field(h, self.edge_index, self.edge_features, self.edge_mask)
        raw = self.delta.view(-1, 1, 1) * raw
        return raw.reshape(self.batch, self.nodes * self.hidden_dim)

    def g(self, t: Tensor, y: Tensor) -> Tensor:
        h = y.view(self.batch, self.nodes, self.hidden_dim)
        bounded = self.diffusion_scale * torch.sigmoid(self.diffusion_net(h))
        scaled = bounded * self.delta.clamp_min(0.0).sqrt().view(-1, 1, 1)
        return scaled.reshape(self.batch, self.nodes * self.hidden_dim)


class GraphSDEDynamics(TemporalDynamicsBase):
    """STABLE_GRAPH_NEURAL_SDE (preflight protocol Section 6): dh_t =
    f_theta(h_t, G, context) dt + g_phi(h_t, quality/context) dW_t. The
    diffusion term represents LATENT/PROCESS uncertainty in the continuous-
    time dynamics -- it is not a model of per-sensor measurement noise,
    dropout, or bias, which remain the separate, unmodified
    quality_features/sensor-fault machinery (preflight protocol Section 6,
    "process uncertainty vs. observation noise").

    Solver: torchsde.sdeint, method="euler", sde_type="ito",
    noise_type="diagonal" (structured, node-local, bounded diffusion; never
    dense/coupled). `diffusion_scale=0.1` bounds g_phi to [0, 0.1] --
    small relative to typical post-projection activation magnitude, chosen
    only for finite dynamics/gradients and a reasonable stochastic
    magnitude, never tuned against predictive quality (frozen preflight
    protocol Section 6). The Brownian path is explicitly seeded via
    `torchsde.BrownianInterval(..., entropy=seed)`, so fixed-seed
    reproducibility (Test A) and different-seed stochasticity (Test B) are
    both mechanically controllable (preflight protocol Section 9.6).

    Permutation-invariance caveat (an engineering finding, not a defect):
    `BrownianInterval`'s noise is keyed by flat state-vector POSITION, not
    node identity, so a bare `entropy=` seed does not, by itself, make the
    full stochastic trajectory numerically invariant under a node
    permutation (permuting nodes reassigns which node receives which noise
    realization). The drift (`field`) and diffusion (`diffusion_net`)
    FUNCTIONS are exactly permutation-equivariant on their own (verified
    directly); `forward(..., explicit_noise=...)` provides a from-scratch
    fixed-step Euler-Maruyama integration path, independent of torchsde,
    that accepts a caller-supplied noise tensor -- permuting that tensor
    consistently alongside the node axis is what actually proves the full
    integration scheme (not just the two functions in isolation) is
    permutation-equivariant, and is the methodology the M9.1 preflight test
    suite uses for this arm's structural-invariance gate."""

    DIFFUSION_SCALE = 0.1
    DT = 0.05

    def __init__(
        self,
        temporal_feature_dim: int,
        quality_feature_dim: int,
        *,
        d_model: int,
        edge_feature_dim: int,
        mlp_width: int,
        diffusion_scale: float = DIFFUSION_SCALE,
        dt: float = DT,
    ) -> None:
        super().__init__()
        _require(torchsde, _TORCHSDE_IMPORT_ERROR, "torchsde")
        self.d_model = d_model
        self.initial_state = _FirstValidEvidenceInitialState(
            temporal_feature_dim, quality_feature_dim, d_model
        )
        self.field = GraphVectorField(d_model, edge_feature_dim, mlp_width=mlp_width)
        self.diffusion_net = nn.Sequential(
            nn.Linear(d_model, mlp_width), nn.SiLU(), nn.Linear(mlp_width, d_model)
        )
        self.temporal_head = nn.Linear(d_model, d_model)
        self.quality_head = nn.Linear(d_model, d_model)
        self.diffusion_scale = diffusion_scale
        self.dt = dt

    def forward(
        self,
        temporal_features: Tensor,
        quality_features: Tensor,
        sensor_mask: Tensor | None,
        quality_mask: Tensor | None,
        timestamps: Tensor | None,
        node_mask: Tensor,
        edge_index: Tensor | None,
        edge_features: Tensor | None,
        edge_mask: Tensor | None,
        *,
        seed: int = 20260814,
        force_zero_diffusion: bool = False,
        explicit_noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert torchsde is not None
        batch = temporal_features.shape[0]
        nodes = temporal_features.shape[2]
        hidden_dim = self.d_model
        h0 = self.initial_state(temporal_features, quality_features, sensor_mask, quality_mask)
        delta = _physical_delta(timestamps, batch, h0.device, h0.dtype)
        diffusion_scale = 0.0 if force_zero_diffusion else self.diffusion_scale

        if explicit_noise is not None:
            # Verification path (preflight protocol Section 9.8): torchsde's
            # BrownianInterval keys noise by FLAT STATE-VECTOR POSITION, not
            # node identity -- permuting which node occupies which position
            # (as a graph structural-invariance test does) therefore assigns
            # each node a genuinely different noise realization even under
            # an identical `entropy=` seed, which is an expected property of
            # position-indexed diagonal noise, not a defect. This explicit,
            # from-scratch fixed-step Euler-Maruyama integrator (independent
            # of torchsde) instead consumes a noise tensor the CALLER
            # constructs and can permute consistently alongside the node
            # axis, which is what actually proves the drift/diffusion
            # FUNCTIONS (and this integration scheme) are permutation-
            # equivariant, and is also reused directly by the zero-diffusion
            # collapse test (Section 9.6 Test C) as an independent reference
            # integrator (a useful cross-check against torchsde's own
            # "euler" method, not merely test-only scaffolding).
            # `explicit_noise`: [steps, batch, nodes, hidden_dim], each entry
            # already scaled to a dt=self.dt Euler-Maruyama increment (i.e.
            # already multiplied by sqrt(self.dt) by the caller).
            steps_n = explicit_noise.shape[0]
            h = h0.clone()
            for step_index in range(steps_n):
                drift = delta.view(-1, 1, 1) * self.field(h, edge_index, edge_features, edge_mask)
                diffusion = diffusion_scale * torch.sigmoid(self.diffusion_net(h))
                diffusion = diffusion * delta.clamp_min(0.0).sqrt().view(-1, 1, 1)
                h = h + self.dt * drift + diffusion * explicit_noise[step_index]
            h_final = h.masked_fill(~node_mask.unsqueeze(-1), 0.0)
            return self.temporal_head(h_final), self.quality_head(h_final)

        sde = _SDEDrift(
            self.field,
            self.diffusion_net,
            batch=batch,
            nodes=nodes,
            hidden_dim=hidden_dim,
            delta=delta,
            diffusion_scale=diffusion_scale,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_mask=edge_mask,
        )
        y0 = h0.reshape(batch, nodes * hidden_dim)
        ts = y0.new_tensor([0.0, 1.0])
        brownian = torchsde.BrownianInterval(
            t0=0.0,
            t1=1.0,
            size=(batch, nodes * hidden_dim),
            dtype=y0.dtype,
            device=y0.device,
            entropy=seed,
        )
        trajectory = torchsde.sdeint(sde, y0, ts, method="euler", dt=self.dt, bm=brownian)
        h_final = trajectory[-1].view(batch, nodes, hidden_dim)
        h_final = h_final.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        return self.temporal_head(h_final), self.quality_head(h_final)


def swap_temporal_dynamics_parameter_count(
    baseline_total_params: int,
    baseline_temporal_and_quality_params: int,
    candidate_temporal_dynamics_params: int,
) -> int:
    """Full-model parameter count if `temporal_encoder`+`quality_encoder`
    (whose combined param count is `baseline_temporal_and_quality_params`)
    were replaced by a continuous-time module with
    `candidate_temporal_dynamics_params` parameters, without constructing
    the candidate full model -- used by the parameter-matching search
    (preflight protocol Section 8) to evaluate many candidate widths cheaply
    before building the (expensive) matched-width final model."""

    return (
        baseline_total_params
        - baseline_temporal_and_quality_params
        + candidate_temporal_dynamics_params
    )
