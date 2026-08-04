"""Task 4.1: configurable classical-prior injection modes."""

from __future__ import annotations

import pytest
import torch

from hydroswarm.model import (
    ARCHITECTURE_VERSION,
    ArchitectureCompatibilityError,
    HydroCore,
    verify_architecture_compatibility,
)


def _tiny_model(**overrides) -> HydroCore:
    base = dict(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
    )
    base.update(overrides)
    return HydroCore(**base)


def _batch(prior: torch.Tensor | None) -> dict:
    generator = torch.Generator().manual_seed(7)
    batch = {
        "node_features": torch.randn(2, 4, 3, generator=generator),
        "temporal_features": torch.randn(2, 3, 4, 2, generator=generator),
        "quality_features": torch.randn(2, 3, 4, 2, generator=generator),
        "source_candidate_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    if prior is not None:
        batch["classical_prior"] = prior
    return batch


def test_invalid_prior_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="prior_mode"):
        _tiny_model(prior_mode="not_a_real_mode")


def test_default_prior_mode_matches_original_unconditional_behavior() -> None:
    model = _tiny_model()
    assert model.prior_mode == "feature_and_logit"


def test_prior_mode_none_produces_identical_output_with_or_without_a_prior() -> None:
    torch.manual_seed(0)
    model = _tiny_model(prior_mode="none").eval()
    prior = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    with torch.no_grad():
        with_prior = model(_batch(prior))
        without_prior = model(_batch(None))
    torch.testing.assert_close(with_prior["hidden_state"], without_prior["hidden_state"])
    torch.testing.assert_close(with_prior["source_node_logits"], without_prior["source_node_logits"])


def test_feature_only_vs_feature_and_logit_differ_by_exactly_the_log_prior_term() -> None:
    torch.manual_seed(1)
    feature_only = _tiny_model(prior_mode="feature_only").eval()
    feature_and_logit = _tiny_model(prior_mode="feature_and_logit").eval()
    feature_and_logit.load_state_dict(feature_only.state_dict())  # identical weights

    prior = torch.tensor([[0.6, 0.2, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1]])
    batch = _batch(prior)
    with torch.no_grad():
        only_output = feature_only(batch)
        both_output = feature_and_logit(batch)

    # Hidden state (and therefore the raw source-head output before the
    # direct log-prior term) must be identical: feature injection is on in
    # both modes, and the only difference is the extra logit term.
    torch.testing.assert_close(only_output["hidden_state"], both_output["hidden_state"])

    expected_term = torch.nn.functional.softplus(feature_and_logit.prior_logit_scale) * torch.log(
        prior.clamp_min(1e-8)
    )
    difference = both_output["source_node_logits"] - only_output["source_node_logits"]
    torch.testing.assert_close(difference, expected_term, atol=1e-5, rtol=1e-4)


def test_logit_only_vs_none_hidden_state_identical_but_logits_differ() -> None:
    torch.manual_seed(2)
    none_model = _tiny_model(prior_mode="none").eval()
    logit_only_model = _tiny_model(prior_mode="logit_only").eval()
    logit_only_model.load_state_dict(none_model.state_dict())

    prior = torch.tensor([[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]])
    batch = _batch(prior)
    with torch.no_grad():
        none_output = none_model(batch)
        logit_output = logit_only_model(batch)

    torch.testing.assert_close(none_output["hidden_state"], logit_output["hidden_state"])
    assert not torch.allclose(none_output["source_node_logits"], logit_output["source_node_logits"])


def test_architecture_config_records_version_and_prior_mode() -> None:
    model = _tiny_model(prior_mode="logit_only")
    config = model.architecture_config()
    assert config["architecture_version"] == ARCHITECTURE_VERSION
    assert config["prior_mode"] == "logit_only"


def test_state_dict_shape_is_identical_across_prior_modes() -> None:
    # The core Task 4.0 compatibility guarantee: prior_mode only gates
    # forward-pass behavior, never parameter shapes, so any checkpoint
    # remains structurally loadable regardless of which mode it was
    # exported under.
    reference = _tiny_model(prior_mode="feature_and_logit").state_dict()
    for mode in ("none", "feature_only", "logit_only"):
        other = _tiny_model(prior_mode=mode).state_dict()
        assert set(other.keys()) == set(reference.keys())
        for key in reference:
            assert other[key].shape == reference[key].shape


def test_verify_architecture_compatibility_accepts_matching_metadata() -> None:
    model = _tiny_model(prior_mode="feature_only")
    verify_architecture_compatibility(model, model.architecture_config())  # must not raise


def test_verify_architecture_compatibility_accepts_missing_fields_for_old_checkpoints() -> None:
    model = _tiny_model()
    verify_architecture_compatibility(model, {})  # must not raise


def test_verify_architecture_compatibility_rejects_prior_mode_mismatch() -> None:
    model = _tiny_model(prior_mode="feature_only")
    with pytest.raises(ArchitectureCompatibilityError, match="prior_mode"):
        verify_architecture_compatibility(model, {"prior_mode": "logit_only"})


def test_verify_architecture_compatibility_rejects_version_mismatch() -> None:
    model = _tiny_model()
    with pytest.raises(ArchitectureCompatibilityError, match="architecture_version"):
        verify_architecture_compatibility(model, {"architecture_version": "hydrocore-v1"})
