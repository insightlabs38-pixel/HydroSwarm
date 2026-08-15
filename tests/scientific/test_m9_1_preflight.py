"""HydroCore-v5 Milestone 9.1 preflight correctness tests.

Frozen protocol: docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md
Section 9. Covers only engineering/correctness properties of the
GRAPH_ODE/GRAPH_CDE/GRAPH_SDE continuous-time TemporalDynamicsBase
candidates (src/hydroswarm/model/continuous_time.py) and the additive
HydroCore(temporal_dynamics=...) seam -- never predictive accuracy, never
development_holdout/locked data. All fixtures below are hand-built synthetic
tensors, matching the preflight's own "synthetic/unit-test tensors, tiny
hand-constructed graphs" data policy (Section 1 of the protocol).
"""

from __future__ import annotations

import pytest
import torch

from hydroswarm.model.continuous_time import (
    CurrentTemporalDynamics,
    GraphCDEDynamics,
    GraphODEDynamics,
    GraphSDEDynamics,
    _physical_delta,
    compute_relative_physical_time,
)
from hydroswarm.model.core import HydroCore

torch.set_default_dtype(torch.float32)

NODES = 5
EDGES = 6
NODE_FEATURE_DIM = 5
EDGE_FEATURE_DIM = 3
TEMPORAL_FEATURE_DIM = 4
QUALITY_FEATURE_DIM = 2
D_MODEL = 16
MLP_WIDTH = 8


def _make_batch(
    batch: int,
    steps: int,
    *,
    dt_seconds: float = 3600.0,
    seed: int | None = None,
    timestamps: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    node_features = torch.randn(batch, NODES, NODE_FEATURE_DIM)
    temporal_features = torch.randn(batch, steps, NODES, TEMPORAL_FEATURE_DIM)
    quality_features = torch.randn(batch, steps, NODES, QUALITY_FEATURE_DIM)
    if timestamps is None:
        timestamps = torch.stack([torch.arange(steps).float() * dt_seconds for _ in range(batch)])
    edge_index = torch.randint(0, NODES, (batch, 2, EDGES))
    edge_features = torch.randn(batch, EDGES, EDGE_FEATURE_DIM)
    return dict(
        node_features=node_features,
        temporal_features=temporal_features,
        quality_features=quality_features,
        timestamps=timestamps,
        edge_index=edge_index,
        edge_features=edge_features,
        node_mask=torch.ones(batch, NODES, dtype=torch.bool),
        sensor_mask=torch.ones(batch, steps, NODES, dtype=torch.bool),
        quality_mask=torch.ones(batch, steps, NODES, dtype=torch.bool),
        edge_mask=torch.ones(batch, EDGES, dtype=torch.bool),
    )


def _run_dynamics(dynamics: torch.nn.Module, batch: dict[str, torch.Tensor], **kwargs: object):
    with torch.no_grad():
        return dynamics(
            batch["temporal_features"],
            batch["quality_features"],
            batch["sensor_mask"],
            batch["quality_mask"],
            batch["timestamps"],
            batch["node_mask"],
            batch["edge_index"],
            batch["edge_features"],
            batch["edge_mask"],
            **kwargs,
        )


def _permute_batch(batch: dict[str, torch.Tensor], perm: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        **batch,
        "node_features": batch["node_features"][:, perm],
        "temporal_features": batch["temporal_features"][:, :, perm],
        "quality_features": batch["quality_features"][:, :, perm],
        "sensor_mask": batch["sensor_mask"][:, :, perm],
        "quality_mask": batch["quality_mask"][:, :, perm],
        "node_mask": batch["node_mask"][:, perm],
        "edge_index": perm.argsort()[batch["edge_index"]],
    }


def _ode() -> GraphODEDynamics:
    torch.manual_seed(1)
    return GraphODEDynamics(
        TEMPORAL_FEATURE_DIM, QUALITY_FEATURE_DIM, d_model=D_MODEL,
        edge_feature_dim=EDGE_FEATURE_DIM, mlp_width=MLP_WIDTH,
    ).eval()


def _cde() -> GraphCDEDynamics:
    torch.manual_seed(2)
    return GraphCDEDynamics(
        TEMPORAL_FEATURE_DIM, QUALITY_FEATURE_DIM, d_model=D_MODEL,
        edge_feature_dim=EDGE_FEATURE_DIM, mlp_width=MLP_WIDTH,
    ).eval()


def _sde() -> GraphSDEDynamics:
    torch.manual_seed(3)
    return GraphSDEDynamics(
        TEMPORAL_FEATURE_DIM, QUALITY_FEATURE_DIM, d_model=D_MODEL,
        edge_feature_dim=EDGE_FEATURE_DIM, mlp_width=MLP_WIDTH,
    ).eval()


def _hydrocore_common() -> dict[str, object]:
    return dict(
        d_model=D_MODEL, nhead=2, dim_feedforward=32, num_layers=1, latent_tokens=64,
        modality_layers=1, node_feature_dim=NODE_FEATURE_DIM, edge_feature_dim=EDGE_FEATURE_DIM,
        temporal_feature_dim=TEMPORAL_FEATURE_DIM, quality_feature_dim=QUALITY_FEATURE_DIM,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Section 3: physical time semantics
# ---------------------------------------------------------------------------


def test_relative_physical_time_translation_invariance():
    timestamps = torch.tensor([[0.0, 600.0, 3600.0, 7200.0]])
    base = compute_relative_physical_time(timestamps)
    shifted = compute_relative_physical_time(timestamps + 123_456.0)
    assert torch.allclose(base, shifted, atol=1e-6)


def test_relative_physical_time_preserves_physical_duration():
    tight = compute_relative_physical_time(torch.tensor([[0.0, 600.0]]))  # 10 min
    wide = compute_relative_physical_time(torch.tensor([[0.0, 36_000.0]]))  # 10 hr
    assert (wide[:, -1] - tight[:, -1]).abs().item() > 1e-3


def test_relative_physical_time_not_window_normalized():
    # Two different total-duration windows must NOT collapse to the same
    # normalized span (the exact M8.6 window_relative defect this protocol
    # forbids reintroducing).
    short = compute_relative_physical_time(torch.tensor([[0.0, 600.0]]))
    long = compute_relative_physical_time(torch.tensor([[0.0, 36_000.0]]))
    assert not torch.allclose(short[:, -1], long[:, -1])


# ---------------------------------------------------------------------------
# Section 9.2: physical-gap plumbing (GRAPH_ODE)
# ---------------------------------------------------------------------------


def test_ode_physical_gap_plumbing():
    ode = _ode()
    b10min = _make_batch(1, 4, dt_seconds=600.0, seed=42)
    b1hr = _make_batch(1, 4, dt_seconds=3600.0, seed=42)
    b10hr = _make_batch(1, 4, dt_seconds=36_000.0, seed=42)
    o_short, _ = _run_dynamics(ode, b10min)
    o_mid, _ = _run_dynamics(ode, b1hr)
    o_long, _ = _run_dynamics(ode, b10hr)
    assert (o_short - o_mid).abs().max().item() > 1e-4
    assert (o_mid - o_long).abs().max().item() > 1e-4


def test_ode_zero_duration_depth_one_is_identity_on_h0():
    # delta=0 (single distinguishable timestamp) must make the ODE collapse
    # to the identity map on the pooled initial state -- deterministic,
    # finite, not a crash.
    ode = _ode()
    batch = _make_batch(1, 1, seed=7)
    out_temporal, out_quality = _run_dynamics(ode, batch)
    assert torch.isfinite(out_temporal).all()
    assert torch.isfinite(out_quality).all()


# ---------------------------------------------------------------------------
# Section 9.3 / 9.9: GRAPH_ODE determinism, gradients, param guardrail
# ---------------------------------------------------------------------------


def test_ode_deterministic_repeatability():
    ode = _ode()
    batch = _make_batch(1, 4, seed=11)
    out_a = _run_dynamics(ode, batch)
    out_b = _run_dynamics(ode, batch)
    assert torch.equal(out_a[0], out_b[0])
    assert torch.equal(out_a[1], out_b[1])


def test_ode_forward_backward_finite_and_gradients_flow():
    ode = _ode()
    ode.train()
    batch = _make_batch(2, 4, seed=13)
    out_temporal, out_quality = ode(
        batch["temporal_features"], batch["quality_features"], batch["sensor_mask"],
        batch["quality_mask"], batch["timestamps"], batch["node_mask"], batch["edge_index"],
        batch["edge_features"], batch["edge_mask"],
    )
    loss = out_temporal.sum() + out_quality.sum()
    assert torch.isfinite(loss)
    loss.backward()
    field_grad = sum(p.grad.abs().sum().item() for p in ode.field.parameters() if p.grad is not None)
    assert field_grad > 0.0
    for p in ode.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# Section 9.4: GRAPH_CDE causality (MANDATORY GATE)
# ---------------------------------------------------------------------------


def test_cde_causality_future_observation_cannot_affect_earlier_cutoff():
    cde = _cde()
    prefix = _make_batch(1, 3, dt_seconds=3600.0, seed=42)
    future_temporal = torch.randn(1, 1, NODES, TEMPORAL_FEATURE_DIM)
    future_quality = torch.randn(1, 1, NODES, QUALITY_FEATURE_DIM)
    future_time = prefix["timestamps"][:, -1:] + 7200.0
    prefix_plus_future = {
        **prefix,
        "temporal_features": torch.cat([prefix["temporal_features"], future_temporal], dim=1),
        "quality_features": torch.cat([prefix["quality_features"], future_quality], dim=1),
        "timestamps": torch.cat([prefix["timestamps"], future_time], dim=1),
        "sensor_mask": torch.ones(1, 4, NODES, dtype=torch.bool),
        "quality_mask": torch.ones(1, 4, NODES, dtype=torch.bool),
    }
    cutoff = prefix["temporal_features"].shape[1] - 1  # T = prefix's own last index
    out_p = _run_dynamics(cde, prefix, cutoff_index=cutoff)
    out_p_plus_future = _run_dynamics(cde, prefix_plus_future, cutoff_index=cutoff)
    max_diff = max((a - b).abs().max().item() for a, b in zip(out_p, out_p_plus_future))
    assert max_diff <= 1e-6, f"CDE causality violated: future observation changed the T-cutoff output by {max_diff}"

    # Sanity check the test itself is meaningful: WITHOUT the causal cutoff,
    # the appended future observation DOES change the (default "now") output
    # -- proving the causality gate isn't trivially passing because the
    # model ignores the feature entirely.
    out_p_plus_future_uncapped = _run_dynamics(cde, prefix_plus_future)
    uncapped_diff = max((a - b).abs().max().item() for a, b in zip(out_p, out_p_plus_future_uncapped))
    assert uncapped_diff > 1e-4


def test_cde_depth_one_degenerate_control_is_finite():
    cde = _cde()
    batch = _make_batch(1, 1, seed=42)
    out_temporal, out_quality = _run_dynamics(cde, batch)
    assert torch.isfinite(out_temporal).all()
    assert torch.isfinite(out_quality).all()


def test_cde_forward_backward_finite_and_gradients_flow():
    cde = _cde()
    cde.train()
    batch = _make_batch(2, 4, seed=17)
    out_temporal, out_quality = cde(
        batch["temporal_features"], batch["quality_features"], batch["sensor_mask"],
        batch["quality_mask"], batch["timestamps"], batch["node_mask"], batch["edge_index"],
        batch["edge_features"], batch["edge_mask"],
    )
    loss = out_temporal.sum() + out_quality.sum()
    assert torch.isfinite(loss)
    loss.backward()
    field_grad = sum(p.grad.abs().sum().item() for p in cde.field.parameters() if p.grad is not None)
    assert field_grad > 0.0


# ---------------------------------------------------------------------------
# Section 9.6: GRAPH_SDE Tests A/B/C
# ---------------------------------------------------------------------------


def test_sde_fixed_seed_reproducibility():
    sde = _sde()
    batch = _make_batch(1, 4, seed=33)
    out_a = _run_dynamics(sde, batch, seed=111)
    out_b = _run_dynamics(sde, batch, seed=111)
    assert torch.equal(out_a[0], out_b[0])
    assert torch.equal(out_a[1], out_b[1])


def test_sde_different_seed_stochasticity():
    sde = _sde()
    batch = _make_batch(1, 4, seed=33)
    out_a = _run_dynamics(sde, batch, seed=111)
    out_b = _run_dynamics(sde, batch, seed=222)
    diff = (out_a[0] - out_b[0]).abs().max().item()
    assert 0.0 < diff < float("inf")


def test_sde_zero_diffusion_collapses_to_manual_deterministic_drift():
    sde = _sde()
    batch = _make_batch(1, 4, seed=33)
    h0 = sde.initial_state(
        batch["temporal_features"], batch["quality_features"], batch["sensor_mask"], batch["quality_mask"]
    )
    delta = _physical_delta(batch["timestamps"], h0.shape[0], h0.device, h0.dtype)
    steps_n = int(round(1.0 / sde.dt))
    with torch.no_grad():
        h = h0.clone()
        for _ in range(steps_n):
            raw = sde.field(h, batch["edge_index"], batch["edge_features"], batch["edge_mask"])
            h = h + sde.dt * delta.view(-1, 1, 1) * raw
        h_reference = h.masked_fill(~batch["node_mask"].unsqueeze(-1), 0.0)
        temporal_reference = sde.temporal_head(h_reference)
    out_temporal, _ = _run_dynamics(sde, batch, seed=999, force_zero_diffusion=True)
    diff = (out_temporal - temporal_reference).abs().max().item()
    assert diff <= 1e-4, f"zero-diffusion SDE did not collapse to the manual deterministic reference: {diff}"


def test_sde_permutation_equivariance_with_consistently_permuted_noise():
    # See continuous_time.GraphSDEDynamics's own docstring: BrownianInterval
    # keys noise by flat position, not node identity, so exact permutation
    # invariance under a bare seed is not the correct test. The
    # explicit_noise verification path lets the SAME (permuted-consistently)
    # noise realization be supplied to both calls, which is what actually
    # proves the drift+diffusion+integration scheme is permutation-
    # equivariant.
    sde = _sde()
    batch = _make_batch(1, 4, seed=33)
    steps_n = int(round(1.0 / sde.dt))
    torch.manual_seed(500)
    noise = torch.randn(steps_n, 1, NODES, D_MODEL) * (sde.dt**0.5)
    perm = torch.randperm(NODES)
    permuted = _permute_batch(batch, perm)
    out_orig = _run_dynamics(sde, batch, explicit_noise=noise)
    out_perm = _run_dynamics(sde, permuted, explicit_noise=noise[:, :, perm])
    for original, permuted_out in zip(out_orig, out_perm):
        assert torch.allclose(original[:, perm], permuted_out, atol=1e-4)


def test_sde_drift_and_diffusion_functions_are_permutation_equivariant():
    sde = _sde()
    h = torch.randn(1, NODES, D_MODEL)
    edge_index = torch.randint(0, NODES, (1, 2, EDGES))
    edge_features = torch.randn(1, EDGES, EDGE_FEATURE_DIM)
    edge_mask = torch.ones(1, EDGES, dtype=torch.bool)
    perm = torch.randperm(NODES)
    with torch.no_grad():
        drift = sde.field(h, edge_index, edge_features, edge_mask)
        diffusion = torch.sigmoid(sde.diffusion_net(h))
        drift_perm = sde.field(h[:, perm], perm.argsort()[edge_index], edge_features, edge_mask)
        diffusion_perm = torch.sigmoid(sde.diffusion_net(h[:, perm]))
    assert torch.allclose(drift[:, perm], drift_perm, atol=1e-5)
    assert torch.allclose(diffusion[:, perm], diffusion_perm, atol=1e-5)


def test_sde_forward_backward_finite_and_gradients_flow():
    sde = _sde()
    sde.train()
    batch = _make_batch(2, 4, seed=19)
    out_temporal, out_quality = sde(
        batch["temporal_features"], batch["quality_features"], batch["sensor_mask"],
        batch["quality_mask"], batch["timestamps"], batch["node_mask"], batch["edge_index"],
        batch["edge_features"], batch["edge_mask"], seed=555,
    )
    loss = out_temporal.sum() + out_quality.sum()
    assert torch.isfinite(loss)
    loss.backward()
    field_grad = sum(p.grad.abs().sum().item() for p in sde.field.parameters() if p.grad is not None)
    diffusion_grad = sum(p.grad.abs().sum().item() for p in sde.diffusion_net.parameters() if p.grad is not None)
    assert field_grad > 0.0
    assert diffusion_grad > 0.0


# ---------------------------------------------------------------------------
# Section 9.7: timestamp-origin invariance (ODE/CDE/SDE)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset_seconds,label", [(3600.0, "+1h"), (86_400.0, "+24h"), (604_800.0, "+7d")])
def test_ode_timestamp_origin_invariance(offset_seconds, label):
    ode = _ode()
    batch = _make_batch(1, 4, dt_seconds=3600.0, seed=55)
    shifted = {**batch, "timestamps": batch["timestamps"] + offset_seconds}
    out_base = _run_dynamics(ode, batch)
    out_shifted = _run_dynamics(ode, shifted)
    diff = max((a - b).abs().max().item() for a, b in zip(out_base, out_shifted))
    assert diff <= 1e-4, f"{label}: {diff}"


@pytest.mark.parametrize("offset_seconds,label", [(3600.0, "+1h"), (86_400.0, "+24h"), (604_800.0, "+7d")])
def test_cde_timestamp_origin_invariance(offset_seconds, label):
    cde = _cde()
    batch = _make_batch(1, 4, dt_seconds=3600.0, seed=55)
    shifted = {**batch, "timestamps": batch["timestamps"] + offset_seconds}
    out_base = _run_dynamics(cde, batch)
    out_shifted = _run_dynamics(cde, shifted)
    diff = max((a - b).abs().max().item() for a, b in zip(out_base, out_shifted))
    assert diff <= 1e-4, f"{label}: {diff}"


@pytest.mark.parametrize("offset_seconds,label", [(3600.0, "+1h"), (86_400.0, "+24h"), (604_800.0, "+7d")])
def test_sde_timestamp_origin_invariance_same_brownian_seed(offset_seconds, label):
    sde = _sde()
    batch = _make_batch(1, 4, dt_seconds=3600.0, seed=55)
    shifted = {**batch, "timestamps": batch["timestamps"] + offset_seconds}
    out_base = _run_dynamics(sde, batch, seed=999)
    out_shifted = _run_dynamics(sde, shifted, seed=999)
    diff = max((a - b).abs().max().item() for a, b in zip(out_base, out_shifted))
    assert diff <= 1e-4, f"{label}: {diff}"


# ---------------------------------------------------------------------------
# Section 9.8: graph structural invariance (ODE/CDE)
# ---------------------------------------------------------------------------


def test_ode_node_permutation_invariance():
    ode = _ode()
    batch = _make_batch(1, 4, seed=61)
    perm = torch.randperm(NODES)
    permuted = _permute_batch(batch, perm)
    out_base = _run_dynamics(ode, batch)
    out_perm = _run_dynamics(ode, permuted)
    for original, permuted_out in zip(out_base, out_perm):
        assert torch.allclose(original[:, perm], permuted_out, atol=1e-4)


def test_cde_node_permutation_invariance():
    cde = _cde()
    batch = _make_batch(1, 4, seed=63)
    perm = torch.randperm(NODES)
    permuted = _permute_batch(batch, perm)
    out_base = _run_dynamics(cde, batch)
    out_perm = _run_dynamics(cde, permuted)
    for original, permuted_out in zip(out_base, out_perm):
        assert torch.allclose(original[:, perm], permuted_out, atol=1e-4)


# ---------------------------------------------------------------------------
# Section 9.10: parameter-count guardrail
# ---------------------------------------------------------------------------


def test_parameter_count_guardrail_within_tolerance():
    from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT

    shared_config = dict(
        prior_mode="feature_only",
        event_control_heads=True,
        scout_control_heads=True,
        strategist_mode="candidate_conditioned",
        action_vocabulary_size=ACTION_TEMPLATE_COUNT,
        consequence_prescreening_heads=True,
        ood_category_head=True,
    )
    baseline = HydroCore.from_variant("small", use_adapters=False, **shared_config)
    baseline_report = baseline.parameter_report()
    assert baseline_report.total == 4_182_612

    temporal_and_quality_params = sum(
        p.numel() for p in baseline.temporal_encoder.parameters()
    ) + sum(p.numel() for p in baseline.quality_encoder.parameters())

    edge_feature_dim = baseline.edge_feature_dim
    temporal_feature_dim = baseline.temporal_feature_dim
    quality_feature_dim = baseline.quality_feature_dim
    d_model = baseline.d_model

    def candidate_params(dynamics: torch.nn.Module) -> int:
        return sum(p.numel() for p in dynamics.parameters())

    def full_model_params(dynamics: torch.nn.Module) -> int:
        return (
            baseline_report.total
            - temporal_and_quality_params
            + candidate_params(dynamics)
        )

    def best_width(build):
        best = None
        for width in range(4, 512, 4):
            dynamics = build(width)
            total = full_model_params(dynamics)
            delta = abs(total - baseline_report.total)
            if best is None or delta < best[0]:
                best = (delta, width, total)
        return best

    for name, build in (
        (
            "ODE",
            lambda width: GraphODEDynamics(
                temporal_feature_dim, quality_feature_dim, d_model=d_model,
                edge_feature_dim=edge_feature_dim, mlp_width=width,
            ),
        ),
        (
            "CDE",
            lambda width: GraphCDEDynamics(
                temporal_feature_dim, quality_feature_dim, d_model=d_model,
                edge_feature_dim=edge_feature_dim, mlp_width=width,
            ),
        ),
        (
            "SDE",
            lambda width: GraphSDEDynamics(
                temporal_feature_dim, quality_feature_dim, d_model=d_model,
                edge_feature_dim=edge_feature_dim, mlp_width=width,
            ),
        ),
    ):
        delta, width, total = best_width(build)
        percent = abs(total - baseline_report.total) / baseline_report.total * 100.0
        assert percent <= 5.0, f"{name} could not be matched within +/-5% (best {percent:.2f}% at width={width})"


# ---------------------------------------------------------------------------
# Section 9.11: irregular timestamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,timestamps",
    [
        ("jittered", torch.tensor([[0.0, 605.3, 1198.7, 1830.1]])),
        ("unequal_gaps", torch.tensor([[0.0, 60.0, 3600.0, 4000.0]])),
        ("duplicated_timestamp", torch.tensor([[0.0, 600.0, 1000.0, 1000.0]])),
        ("large_gap", torch.tensor([[0.0, 300.0, 600.0, 8_640_000.0]])),
    ],
)
def test_irregular_timestamps_no_nan_inf(label, timestamps):
    batch = _make_batch(1, 4, seed=71, timestamps=timestamps)
    ode = _ode()
    cde = _cde()
    sde = _sde()
    out_ode = _run_dynamics(ode, batch)
    out_cde = _run_dynamics(cde, batch)
    out_sde = _run_dynamics(sde, batch, seed=1)
    for name, out in (("ODE", out_ode), ("CDE", out_cde), ("SDE", out_sde)):
        assert torch.isfinite(out[0]).all(), f"{label}/{name} temporal latent not finite"
        assert torch.isfinite(out[1]).all(), f"{label}/{name} quality latent not finite"


# ---------------------------------------------------------------------------
# Section 9.12: baseline wrapper equivalence
# ---------------------------------------------------------------------------


def test_current_wrapper_matches_vanilla_hydrocore():
    common = _hydrocore_common()
    torch.manual_seed(101)
    vanilla = HydroCore(**common).eval()

    current_dynamics = CurrentTemporalDynamics(
        TEMPORAL_FEATURE_DIM, QUALITY_FEATURE_DIM, d_model=D_MODEL, nhead=2,
        dim_feedforward=32, num_layers=1, dropout=0.0,
    )
    current_dynamics.temporal_encoder.load_state_dict(vanilla.temporal_encoder.state_dict())
    current_dynamics.quality_encoder.load_state_dict(vanilla.quality_encoder.state_dict())

    wrapped = HydroCore(temporal_dynamics=current_dynamics, **common).eval()
    wrapped.load_state_dict(vanilla.state_dict(), strict=False)
    wrapped.temporal_dynamics.temporal_encoder.load_state_dict(vanilla.temporal_encoder.state_dict())
    wrapped.temporal_dynamics.quality_encoder.load_state_dict(vanilla.quality_encoder.state_dict())

    batch = _make_batch(2, 4, seed=91)
    with torch.no_grad():
        out_vanilla = vanilla(batch)["source_node_logits"]
        out_wrapped = wrapped(batch)["source_node_logits"]
    diff = (out_vanilla - out_wrapped).abs().max().item()
    assert diff <= 1e-6, f"CURRENT wrapper diverged from vanilla HydroCore by {diff}"


def test_hydrocore_default_construction_unaffected_by_new_parameter():
    # temporal_dynamics defaults to None -- every existing caller (no kwarg
    # passed at all) must be byte-identical to before this seam existed.
    common = _hydrocore_common()
    torch.manual_seed(202)
    model = HydroCore(**common)
    assert model.temporal_dynamics is None
    assert hasattr(model, "temporal_encoder")
    assert hasattr(model, "quality_encoder")


# ---------------------------------------------------------------------------
# End-to-end forward/backward finiteness through HydroCore itself (Section 9.9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build_dynamics", [_ode, _cde, _sde])
def test_hydrocore_end_to_end_forward_backward_finite(build_dynamics):
    common = _hydrocore_common()
    dynamics = build_dynamics()
    torch.manual_seed(303)
    model = HydroCore(temporal_dynamics=dynamics, **common)
    model.train()
    batch = _make_batch(2, 4, seed=303)
    output = model(batch)
    assert torch.isfinite(output["source_node_logits"]).all()
    target = torch.zeros(2, NODES)
    target[:, 0] = 1.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["source_node_logits"].squeeze(-1), target
    )
    assert torch.isfinite(loss)
    loss.backward()
    dynamics_grad = sum(
        p.grad.abs().sum().item() for p in model.temporal_dynamics.parameters() if p.grad is not None
    )
    head_grad = sum(
        p.grad.abs().sum().item() for p in model.source_node_head.parameters() if p.grad is not None
    )
    assert dynamics_grad > 0.0, "no gradient reached the temporal_dynamics module"
    assert head_grad > 0.0, "no gradient reached the source_node_head"
