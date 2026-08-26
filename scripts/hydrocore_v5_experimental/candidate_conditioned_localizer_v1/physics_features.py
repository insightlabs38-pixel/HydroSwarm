"""Cheap, label-free, physically-motivated candidate-compatibility features
(Arm C -- PHYSICS_INFORMED_CANDIDATE) for the candidate-conditioned-
localizer-v1 experiment.

Scope note (plan doc Section 4/9): a full simulator-grounded compatibility
feature (candidate-vs-observed EPANET signature residual, as in
`exp/source-identifiability-analysis`'s `fair_oracle.py`) would require a
per-topology nuisance-grid precompute this pilot's compute budget cannot
absorb inside the training loop for every one of thousands of examples with
their own randomized demand/hydraulics. This module instead computes an
ARRIVAL-PATTERN COMPATIBILITY proxy directly from each example's own
observed sensor readings (`temporal_features[..., 0] ==
log1p(concentration_mg_l)`, `hydroswarm.preprocessing.builder`'s own
channel-0 convention) and the label-free candidate-to-sensor hop distance
(`candidate_sensor_features.compute_hop_distance`): physically, a true
contamination source should (a) sit structurally close to sensors reading
HIGH concentration, and (b) show a hop-distance-vs-arrival-time relationship
consistent with transport away from that candidate (farther sensors detect
later). Both directions are real physical priors, computed with zero
additional EPANET calls -- never a simulator replay, and clearly weaker
than the full nuisance-searched residual used for the Task-1 oracle audit.
This is an intentional, documented scope reduction (see the plan doc's
Section 9 "what Arm C does NOT do"), not a hidden substitution.

Never reads `source_node`/`source_node_mask`/any evaluation-outcome tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hydroswarm.model.candidate_localizer import UNREACHABLE_HOP_SENTINEL

PHYSICS_FEATURE_COLUMNS: tuple[str, ...] = (
    "nearest_sensor_log_concentration",
    "hop_magnitude_compatibility",
    "hop_arrival_time_compatibility",
)


def _peak_concentration_and_arrival(
    temporal_features: Tensor, timestamps: Tensor | None
) -> tuple[Tensor, Tensor]:
    """`temporal_features`: `[batch, time, nodes, features]`, channel 0 =
    `log1p(concentration_mg_l)` (NaN where unobserved). Returns
    `(peak_log_concentration, arrival_time)`, both `[batch, nodes]`;
    `arrival_time` is the elapsed time (seconds, relative to the window's
    first timestamp) of that node's own peak reading, `+inf` if the node
    has no finite reading anywhere in the window (so it never falsely looks
    "early")."""

    batch, steps, nodes, _ = temporal_features.shape
    concentration = temporal_features[..., 0]
    finite = torch.isfinite(concentration)
    safe = torch.nan_to_num(concentration, nan=float("-inf"))
    peak, peak_index = safe.max(dim=1)
    peak = torch.where(finite.any(dim=1), peak, torch.zeros_like(peak))

    if timestamps is None:
        elapsed = torch.arange(steps, device=temporal_features.device, dtype=temporal_features.dtype)
        elapsed = elapsed[None, :].expand(batch, steps)
    else:
        elapsed = timestamps.to(device=temporal_features.device, dtype=temporal_features.dtype)
        elapsed = elapsed - elapsed[:, :1]
    arrival = torch.gather(elapsed, 1, peak_index).squeeze(1) if elapsed.ndim == 2 else elapsed[peak_index]
    arrival = torch.where(finite.any(dim=1), arrival, torch.full_like(arrival, float("inf")))
    return peak, arrival


def _masked_correlation(x: Tensor, y: Tensor, valid: Tensor) -> float:
    """Pearson correlation over `valid` positions only; 0.0 (neutral, not
    NaN) if fewer than 2 valid points or either series is constant."""

    if int(valid.sum()) < 2:
        return 0.0
    xs, ys = x[valid], y[valid]
    if torch.std(xs) < 1e-9 or torch.std(ys) < 1e-9:
        return 0.0
    corr = torch.corrcoef(torch.stack((xs, ys)))[0, 1]
    return 0.0 if not torch.isfinite(corr) else float(corr)


def compute_physics_features(
    temporal_features: Tensor,
    hop_distance: Tensor,
    active_sensor_mask_nodes: Tensor,
    node_mask: Tensor,
    timestamps: Tensor | None = None,
) -> Tensor:
    """Returns `[batch, nodes, len(PHYSICS_FEATURE_COLUMNS)]`, zero at
    padded node positions or where fewer than 2 active sensors exist (the
    correlation features are then neutral/uninformative by construction,
    not silently wrong)."""

    batch, nodes = node_mask.shape
    out = torch.zeros(batch, nodes, len(PHYSICS_FEATURE_COLUMNS), dtype=torch.float32)
    peak_log_conc, arrival = _peak_concentration_and_arrival(temporal_features, timestamps)

    for index in range(batch):
        sensors = torch.nonzero(active_sensor_mask_nodes[index], as_tuple=True)[0]
        if sensors.numel() == 0:
            continue
        sensor_peak = peak_log_conc[index, sensors]
        sensor_arrival = arrival[index, sensors]
        finite_arrival = torch.isfinite(sensor_arrival)
        for node in torch.nonzero(node_mask[index], as_tuple=True)[0].tolist():
            hop_to_sensors = hop_distance[index, node, sensors].float()
            reachable = hop_to_sensors > UNREACHABLE_HOP_SENTINEL
            if not bool(reachable.any()):
                continue
            reachable_hops = hop_to_sensors[reachable]
            reachable_peak = sensor_peak[reachable]
            nearest_index = int(torch.argmin(reachable_hops))
            out[index, node, 0] = reachable_peak[nearest_index]

            # Physically: shorter hop distance should coincide with HIGHER
            # peak concentration if this candidate were the true source --
            # negate the raw (expected-negative) correlation so higher
            # output = more physically consistent, matching the sign
            # convention of column 0 above.
            out[index, node, 1] = -_masked_correlation(reachable_hops, reachable_peak, torch.ones_like(reachable_hops, dtype=torch.bool))

            reachable_and_finite = reachable & finite_arrival
            if int(reachable_and_finite.sum()) >= 2:
                hops_f = hop_distance[index, node, sensors][reachable_and_finite].float()
                arrival_f = sensor_arrival[reachable_and_finite]
                # Physically: shorter hop distance should coincide with
                # EARLIER arrival -- positive raw correlation is expected,
                # kept as-is (higher = more consistent).
                out[index, node, 2] = _masked_correlation(hops_f, arrival_f, torch.ones_like(hops_f, dtype=torch.bool))
    return out


__all__ = ["PHYSICS_FEATURE_COLUMNS", "compute_physics_features"]
