import numpy as np
import pytest

from hydroswarm.inference.fusion import (
    ControlAction,
    TrustFeatures,
    conformal_candidate_set,
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
