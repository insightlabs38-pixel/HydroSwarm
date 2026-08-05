import pytest

from hydroswarm.inference.fusion import (
    DYNAMIC_TRUST_FUSION_CONFIG,
    ControlAction,
    TrustFeatures,
    conformal_candidate_set,
    fixed_weight_fusion,
    fixed_weight_fusion_config,
    fuse_source_probabilities,
    jensen_shannon_divergence,
    uncertainty_control,
)


def features(**overrides: float) -> TrustFeatures:
    values = {
        "healthy_sensor_fraction": 0.9,
        "missing_rate": 0.1,
        "normalized_residual": 0.1,
        "hydraulic_uncertainty": 0.1,
        "neural_entropy": 0.4,
        "classical_entropy": 0.3,
        "ood_score": 0.1,
    }
    values.update(overrides)
    return TrustFeatures(**values)


def test_js_divergence_is_bounded_and_symmetric() -> None:
    p = [0.99, 0.01]
    q = [0.01, 0.99]
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0)
    assert jensen_shannon_divergence(p, q) == pytest.approx(
        jensen_shannon_divergence(q, p)
    )
    assert 0.0 < jensen_shannon_divergence(p, q) <= 1.0


def test_fusion_applies_physical_mask_and_dynamic_trust() -> None:
    fused, diagnostics = fuse_source_probabilities(
        neural_logits=[0.0, 2.0, 1.0],
        classical_probabilities=[0.8, 0.1, 0.1],
        physical_mask=[True, False, True],
        features=features(),
    )
    assert fused.sum() == pytest.approx(1.0)
    assert fused[1] == 0.0
    assert 0.05 <= diagnostics.classical_trust <= 0.95
    assert diagnostics.masked_nodes == 1

    unhealthy = fuse_source_probabilities(
        [0.0, 2.0, 1.0],
        [0.8, 0.1, 0.1],
        [True, False, True],
        features=features(healthy_sensor_fraction=0.2, missing_rate=0.8),
    )[1]
    assert unhealthy.classical_trust < diagnostics.classical_trust


def test_fixed_weight_fusion_blends_classical_and_neural_and_normalizes() -> None:
    fused = fixed_weight_fusion([0.7, 0.2, 0.1], [0.1, 0.2, 0.7], neural_weight=0.6)
    assert fused.sum() == pytest.approx(1.0)
    # Purely classical (neural_weight=0) reproduces the classical vector.
    assert fixed_weight_fusion([0.7, 0.2, 0.1], [0.1, 0.2, 0.7], neural_weight=0.0) == pytest.approx(
        [0.7, 0.2, 0.1]
    )
    # Purely neural (neural_weight=1) reproduces the neural vector.
    assert fixed_weight_fusion([0.7, 0.2, 0.1], [0.1, 0.2, 0.7], neural_weight=1.0) == pytest.approx(
        [0.1, 0.2, 0.7]
    )


def test_fixed_weight_fusion_works_batched_and_unbatched_identically() -> None:
    classical = [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]]
    neural = [[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]]
    batched = fixed_weight_fusion(classical, neural)
    for row_index in range(2):
        single = fixed_weight_fusion(classical[row_index], neural[row_index])
        assert single == pytest.approx(batched[row_index])


def test_fixed_weight_fusion_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValueError, match="neural_weight"):
        fixed_weight_fusion([0.5, 0.5], [0.5, 0.5], neural_weight=1.5)


def test_fixed_weight_fusion_config_is_distinct_per_weight_and_from_dynamic_trust() -> None:
    assert fixed_weight_fusion_config(0.6) != fixed_weight_fusion_config(0.5)
    assert fixed_weight_fusion_config(0.6) != DYNAMIC_TRUST_FUSION_CONFIG


def test_split_conformal_candidate_set_uses_finite_sample_quantile() -> None:
    selected = conformal_candidate_set(
        [0.75, 0.15, 0.10],
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55],
        alpha=0.2,
    )
    assert selected.tolist() == [0]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"candidate_count": 1}, ControlAction.GENERATE_PLANS),
        ({"candidate_count": 6}, ControlAction.REQUEST_SAMPLE),
        ({"candidate_count": 3, "disagreement_js": 0.7}, ControlAction.INSPECT_SENSORS),
        ({"candidate_count": 1, "ood_score": 0.9}, ControlAction.ABSTAIN),
        (
            {"candidate_count": 3, "sample_budget_remaining": 0},
            ControlAction.ABSTAIN,
        ),
    ],
)
def test_uncertainty_changes_operational_behavior(kwargs, expected) -> None:
    inputs = {
        "candidate_count": 1,
        "disagreement_js": 0.1,
        "ood_score": 0.1,
        "healthy_sensor_fraction": 0.9,
        "sample_budget_remaining": 3,
    }
    inputs.update(kwargs)
    assert uncertainty_control(**inputs) == expected
