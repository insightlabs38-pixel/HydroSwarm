"""Milestone 8.7 Section 4: temporal-feature correctness tests.

Proves, BEFORE any arm is trained, that:
1. `unobserved_age_sentinel="fixed"` actually removes the Milestone 8.6
   ABSOLUTE_TIME_ORIGIN_LEAKAGE defect (and that the default/Arm-A
   behavior still reproduces it, so this test would have caught the
   original bug).
2. `include_relative_gap_feature=True` + `elapsed_time_normalization=
   "fixed_scale"` (Arm C) is genuinely sensitive to physical inter-report
   spacing changes that the OLD window-relative encoding collapsed away.
3. That new sensitivity does not come at the cost of origin invariance --
   Arm C must still be invariant to a pure additive shift of every
   timestamp.
4. The new `time_since_previous_report` channel is causal: it never
   depends on reports beyond the ones actually supplied.
5. M8.6's node/edge-order equivariance still holds under the new
   feature-builder/model options.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from hydroswarm.classical.state_estimation import HydraulicStateEstimator, OperationalTelemetry
from hydroswarm.model import HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder, SensorSeries
from hydroswarm.preprocessing.schema import NODE_FEATURE_NAMES
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.simulation.wrapper import HydraulicSimulator
from hydroswarm.training.causal_prefix import (
    ARM_POLICIES,
    CausalPrefixDatasetView,
    build_scenario_pool,
    fit_pool_signature_library,
)
from hydroswarm.training.data import CurriculumStage

pytestmark = pytest.mark.real_simulation

SHARED_MODEL_CONFIG = dict(
    prior_mode="feature_only", event_control_heads=True, scout_control_heads=True,
    strategist_mode="candidate_conditioned", action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True, ood_category_head=True,
)
MEASUREMENT_AGE_INDEX = NODE_FEATURE_NAMES.index("measurement_age")
STRUCTURAL_TOLERANCE = 1e-4


def _context():
    network = build_wntr_network()
    simulator = HydraulicSimulator(network)
    raw = simulator.calculate_state(3600)
    estimated = HydraulicStateEstimator().estimate(raw, OperationalTelemetry())
    graph = simulator.build_dynamic_graph(estimated.as_hydraulic_state())
    return network, graph, estimated


def _series(node_id: str, timestamps: tuple[float, ...], concentrations: tuple[float, ...]) -> SensorSeries:
    n = len(timestamps)
    return SensorSeries(
        node_id=node_id, timestamps_seconds=timestamps, concentration_mg_l=concentrations,
        pressure_m=tuple(25.0 for _ in range(n)), health=tuple(1.0 for _ in range(n)),
        missing=tuple(False for _ in range(n)), drift=tuple(False for _ in range(n)), delayed=tuple(False for _ in range(n)),
    )


def _truncate(series: SensorSeries, n: int) -> SensorSeries:
    return SensorSeries(
        node_id=series.node_id, timestamps_seconds=series.timestamps_seconds[:n],
        concentration_mg_l=series.concentration_mg_l[:n], pressure_m=series.pressure_m[:n],
        health=series.health[:n], missing=series.missing[:n], drift=series.drift[:n],
        delayed=series.delayed[:n], frozen=series.frozen[:n],
    )


def _build(network, graph, state, series, node_ids, **kwargs):
    prior = {n: 1.0 / len(node_ids) for n in node_ids}
    return HydraulicFeatureBuilder().build(network, graph, state, series, classical_prior=prior, **kwargs)


def _model_ab() -> HydroCore:
    return HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG).eval()


def _model_c() -> HydroCore:
    return HydroCore.from_variant(
        "small", use_adapters=False, temporal_feature_dim=7, quality_feature_dim=5,
        elapsed_time_normalization="fixed_scale", **SHARED_MODEL_CONFIG,
    ).eval()


def _posterior(model: HydroCore, batch: dict[str, torch.Tensor]) -> np.ndarray:
    with torch.no_grad():
        output = model(batch)
    return torch.softmax(output["source_node_logits"][0], dim=-1).numpy()


def test_default_unobserved_age_sentinel_still_reproduces_m8_6_bug() -> None:
    """Regression guard: proves this test file WOULD have caught the
    original defect -- the default ("incident_elapsed") fallback is still
    origin-dependent for a never-observed node."""

    network, graph, state = _context()
    node_ids = tuple(sorted(network.node_name_list))
    series = [_series("J1", (0.0, 3600.0), (0.0, 0.2))]
    shifted = [replace(s, timestamps_seconds=tuple(t + 604_800.0 for t in s.timestamps_seconds)) for s in series]

    original = _build(network, graph, state, series, node_ids)
    shifted_built = _build(network, graph, state, shifted, node_ids)

    j2_index = node_ids.index("J2")  # J2 has no sensor in this test's series.
    original_age = original.batch["node_features"][0, j2_index, MEASUREMENT_AGE_INDEX].item()
    shifted_age = shifted_built.batch["node_features"][0, j2_index, MEASUREMENT_AGE_INDEX].item()
    assert abs(original_age - shifted_age) > 1e-6, (
        "expected the DEFAULT (unfixed) sentinel to still be origin-dependent; if this now fails, "
        "the M8.6 bug's default reproduction changed and the 'fixed' comparison below is no longer meaningful"
    )


def test_fixed_sentinel_is_origin_invariant_for_never_observed_nodes() -> None:
    network, graph, state = _context()
    node_ids = tuple(sorted(network.node_name_list))
    series = [_series("J1", (0.0, 3600.0), (0.0, 0.2))]
    shifted = [replace(s, timestamps_seconds=tuple(t + 604_800.0 for t in s.timestamps_seconds)) for s in series]

    original = _build(network, graph, state, series, node_ids, unobserved_age_sentinel="fixed")
    shifted_built = _build(network, graph, state, shifted, node_ids, unobserved_age_sentinel="fixed")

    for node_id in node_ids:
        if node_id == "J1":
            continue  # the one observed node; its age is expected to be identical too, checked below.
        index = node_ids.index(node_id)
        original_age = original.batch["node_features"][0, index, MEASUREMENT_AGE_INDEX].item()
        shifted_age = shifted_built.batch["node_features"][0, index, MEASUREMENT_AGE_INDEX].item()
        assert original_age == pytest.approx(shifted_age, abs=1e-9), f"node {node_id} age not origin-invariant"

    # The OBSERVED node's age is a genuine elapsed difference either way -- also invariant.
    j1_index = node_ids.index("J1")
    assert original.batch["node_features"][0, j1_index, MEASUREMENT_AGE_INDEX].item() == pytest.approx(
        shifted_built.batch["node_features"][0, j1_index, MEASUREMENT_AGE_INDEX].item(), abs=1e-9
    )


@pytest.mark.parametrize("scenario", ["sparse", "no_event"])
def test_fixed_sentinel_full_posterior_origin_invariant(scenario: str) -> None:
    network, graph, state = _context()
    node_ids = tuple(sorted(network.node_name_list))
    if scenario == "sparse":
        series = [_series("J1", (0.0, 3600.0, 7200.0), (0.0, 0.3, 0.6))]
    else:
        series = [_series(n, (0.0, 3600.0, 7200.0), (0.0, 0.0, 0.0)) for n in ("J1", "J2", "J3", "J4")]

    shifted = [replace(s, timestamps_seconds=tuple(t + 604_800.0 for t in s.timestamps_seconds)) for s in series]

    model = _model_ab()
    original = _build(network, graph, state, series, node_ids, unobserved_age_sentinel="fixed")
    shifted_built = _build(network, graph, state, shifted, node_ids, unobserved_age_sentinel="fixed")

    reference = _posterior(model, original.batch)
    candidate = _posterior(model, shifted_built.batch)
    max_abs_diff = float(np.abs(reference - candidate).max())
    assert max_abs_diff <= STRUCTURAL_TOLERANCE, (
        f"scenario={scenario}: fixed sentinel should restore origin invariance, got max_abs_diff={max_abs_diff}"
    )


def test_relative_time_representation_is_sensitive_to_physical_spacing() -> None:
    """Arm C must NOT collapse [0, 10min] and [0, 10hr] to the same
    representation the way the old window-relative encoding did (M8.6
    Section 9's REPRESENTATION_SENSITIVITY_COUNTERFACTUAL)."""

    network, graph, state = _context()
    node_ids = tuple(sorted(network.node_name_list))
    model = _model_c()

    tight = [_series("J1", (0.0, 600.0), (0.0, 0.4))]
    wide = [_series("J1", (0.0, 36_000.0), (0.0, 0.4))]

    tight_built = _build(
        network, graph, state, tight, node_ids, unobserved_age_sentinel="fixed", include_relative_gap_feature=True,
    )
    wide_built = _build(
        network, graph, state, wide, node_ids, unobserved_age_sentinel="fixed", include_relative_gap_feature=True,
    )
    tight_posterior = _posterior(model, tight_built.batch)
    wide_posterior = _posterior(model, wide_built.batch)
    max_abs_diff = float(np.abs(tight_posterior - wide_posterior).max())
    # An UNTRAINED (random-weight) model's output sensitivity magnitude is
    # not itself a meaningful scientific quantity (that question is what
    # M8.7's real training run answers) -- this only checks it clears the
    # structural-equivariance noise floor, i.e. the two inputs are not
    # being silently treated as identical.
    assert max_abs_diff > STRUCTURAL_TOLERANCE, (
        f"expected Arm C's representation to change at all with physical spacing, got max_abs_diff={max_abs_diff}"
    )

    # The raw relative-gap feature itself must differ numerically (the
    # actual, most direct proof -- independent of the untrained model's
    # own sensitivity).
    tight_gap = tight_built.batch["temporal_features"][0, 1, node_ids.index("J1"), 6].item()
    wide_gap = wide_built.batch["temporal_features"][0, 1, node_ids.index("J1"), 6].item()
    assert tight_gap == pytest.approx(600.0 / 86_400.0, abs=1e-6)
    assert wide_gap == pytest.approx(36_000.0 / 86_400.0, abs=1e-6)
    assert tight_gap != wide_gap


def test_relative_time_representation_remains_origin_invariant() -> None:
    """Arm C's new sensitivity to MAGNITUDE must not reintroduce
    sensitivity to arbitrary ORIGIN -- these are independent properties."""

    network, graph, state = _context()
    node_ids = tuple(sorted(network.node_name_list))
    model = _model_c()

    series = [_series("J1", (0.0, 600.0), (0.0, 0.4))]
    shifted = [_series("J1", (604_800.0, 604_800.0 + 600.0), (0.0, 0.4))]

    built = _build(network, graph, state, series, node_ids, unobserved_age_sentinel="fixed", include_relative_gap_feature=True)
    shifted_built = _build(network, graph, state, shifted, node_ids, unobserved_age_sentinel="fixed", include_relative_gap_feature=True)

    reference = _posterior(model, built.batch)
    candidate = _posterior(model, shifted_built.batch)
    max_abs_diff = float(np.abs(reference - candidate).max())
    assert max_abs_diff <= STRUCTURAL_TOLERANCE, f"Arm C should remain origin-invariant, got max_abs_diff={max_abs_diff}"


def test_relative_gap_feature_is_causal_under_truncation() -> None:
    """The gap value for report i must depend only on reports <= i --
    truncating a series to its first k points must not change the gap
    value recorded for any of those retained points (no leakage from
    reports that would come after the truncation boundary)."""

    network, graph, state = _context()
    node_ids = tuple(sorted(network.node_name_list))
    full = [_series("J1", (0.0, 600.0, 1_800.0, 5_400.0), (0.0, 0.2, 0.4, 0.6))]
    truncated = [_truncate(full[0], 2)]

    full_built = _build(network, graph, state, full, node_ids, unobserved_age_sentinel="fixed", include_relative_gap_feature=True, window_steps=4)
    truncated_built = _build(network, graph, state, truncated, node_ids, unobserved_age_sentinel="fixed", include_relative_gap_feature=True, window_steps=2)

    j1 = node_ids.index("J1")
    full_gap_at_1 = full_built.batch["temporal_features"][0, 1, j1, 6].item()
    truncated_gap_at_1 = truncated_built.batch["temporal_features"][0, 1, j1, 6].item()
    assert full_gap_at_1 == pytest.approx(truncated_gap_at_1, abs=1e-9)
    assert full_gap_at_1 == pytest.approx(600.0 / 86_400.0, abs=1e-6)

    # First report of any series always has gap 0.0 (no prior report), never a fabricated future/negative value.
    assert full_built.batch["temporal_features"][0, 0, j1, 6].item() == pytest.approx(0.0, abs=1e-9)


def test_node_order_permutation_equivariance_holds_under_arm_c_options() -> None:
    """Regression guard: M8.6's node-order equivariance property must
    still hold with the new temporal_feature_dim=7/quality_feature_dim=5/
    include_relative_gap_feature=True configuration."""

    network, graph, state = _context()
    node_ids = tuple(sorted(network.node_name_list))
    series = [_series(n, (0.0, 3600.0), (0.0, 0.3)) for n in ("J1", "J2")]
    built = _build(network, graph, state, series, node_ids, unobserved_age_sentinel="fixed", include_relative_gap_feature=True)
    model = _model_c()

    n_nodes = len(node_ids)
    perm = np.random.default_rng(777).permutation(n_nodes)
    inverse = np.argsort(perm)
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    inverse_t = torch.as_tensor(inverse, dtype=torch.long)

    batch = built.batch
    permuted = dict(batch)
    for key in ("node_features", "node_mask", "classical_prior", "source_candidate_mask", "travel_time", "reservoir_reachability", "demand_centrality"):
        if key in permuted:
            permuted[key] = permuted[key].index_select(1, perm_t)
    for key in ("temporal_features", "quality_features", "sensor_mask", "quality_mask"):
        if key in permuted:
            permuted[key] = permuted[key].index_select(2, perm_t)
    permuted["edge_index"] = inverse_t[permuted["edge_index"]]

    reference = _posterior(model, batch)
    permuted_posterior = _posterior(model, permuted)
    mapped_back = permuted_posterior[inverse]
    max_abs_diff = float(np.abs(reference - mapped_back).max())
    assert max_abs_diff <= STRUCTURAL_TOLERANCE


def test_stages_through_forwards_feature_kwargs() -> None:
    """Regression test for a real bug caught while training Milestone 8.7's
    arms: `Trainer.fit()` calls `train_dataset.stages_through(stage)` every
    epoch and trains on ITS return value, never on the view passed to
    `Trainer.__init__` directly. `stages_through` used to construct its
    returned `CausalPrefixDatasetView` without forwarding
    `unobserved_age_sentinel`/`include_relative_gap_feature`, so every
    epoch's actual training data silently reverted to Arm-A/default
    feature semantics regardless of what the caller configured --
    AGE_FIX_PLUS_RELATIVE_TIME's dimension mismatch crashed loudly (caught
    immediately); AGE_FIX_ONLY's identical-shape reversion would have
    trained silently on the WRONG (unfixed) data with no error at all."""

    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)
    view = CausalPrefixDatasetView(
        train_records[:4], expected_split="train", signature_library=library, depth_policy=ARM_POLICIES["A"],
        base_seed=31874, batch_size=2, unobserved_age_sentinel="fixed", include_relative_gap_feature=True,
    )
    staged = view.stages_through(CurriculumStage.ADVERSARIAL)  # the highest stage -- includes every record.
    assert staged._unobserved_age_sentinel == "fixed"
    assert staged._include_relative_gap_feature is True

    example = staged[0]
    assert example.inputs["temporal_features"].shape[-1] == 7
    assert example.inputs["quality_features"].shape[-1] == 5
