"""core-issues4.txt Section A/B: HydroCore-v4 checkpoint identity."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from hydroswarm.model.core import ArchitectureCompatibilityError, HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.training import checkpoint_identity as ci
from hydroswarm.training.checkpoint import (
    LegacyLoaderRejectedV4CheckpointError,
    load_checkpoint,
    save_checkpoint,
)

_TINY_KWARGS = dict(
    d_model=32,
    nhead=2,
    dim_feedforward=64,
    num_layers=1,
    modality_layers=1,
    latent_tokens=64,
    plan_queries=1,
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
)


def _tiny_model(**overrides: object) -> HydroCore:
    kwargs = {**_TINY_KWARGS, **overrides}
    model = HydroCore(**kwargs)  # type: ignore[arg-type]
    model.variant_name = "test-tiny"
    return model


def _identity_for(model: HydroCore, **overrides: object) -> ci.CheckpointIdentity:
    kwargs = dict(
        normalization_hash="no-normalization",
        fusion_policy_hash="fixed-weight-v1:neural=0.5",
        source_corpus_manifest_hashes=("abc123",),
        trained_outputs=frozenset({"source_node"}),
        validated_outputs=frozenset({"source_node"}),
        runtime_enabled_outputs=frozenset({"source_node"}),
    )
    kwargs.update(overrides)
    return ci.build_checkpoint_identity(model, **kwargs)  # type: ignore[arg-type]


def _optimizer_scheduler(model: HydroCore):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    return optimizer, scheduler


# --- build/validate ------------------------------------------------------


def test_build_checkpoint_identity_requires_a_named_variant() -> None:
    model = _tiny_model()
    model.variant_name = None
    with pytest.raises(ci.CheckpointIdentityError, match="named model.variant_name"):
        _identity_for(model)


def test_build_checkpoint_identity_requires_canonical_action_vocabulary() -> None:
    model = _tiny_model(action_vocabulary_size=ACTION_TEMPLATE_COUNT - 1)
    with pytest.raises(ci.CheckpointIdentityError, match="canonical .* action vocabulary"):
        _identity_for(model)


def test_build_checkpoint_identity_rejects_future_concentration_as_trained() -> None:
    """core-issues4.txt Section D item 11: future_concentration_target
    always returns an all-masked placeholder (core-issues3.txt Phase 7.4's
    leakage fix), so no checkpoint has ever actually trained it -- claiming
    otherwise in a v4 identity would be a defect, not a fact."""

    model = _tiny_model()
    with pytest.raises(ci.CheckpointIdentityError, match="future_concentration"):
        _identity_for(
            model,
            trained_outputs=frozenset({"future_concentration"}),
            validated_outputs=frozenset(),
            runtime_enabled_outputs=frozenset(),
        )


def test_trained_ood_categories_derived_from_ood_category_output_membership() -> None:
    model = _tiny_model()
    without = _identity_for(model, trained_outputs=frozenset(), validated_outputs=frozenset(), runtime_enabled_outputs=frozenset())
    assert without.trained_ood_categories == ()
    with_ood = _identity_for(
        model, trained_outputs=frozenset({"ood_category"}), validated_outputs=frozenset(), runtime_enabled_outputs=frozenset()
    )
    assert len(with_ood.trained_ood_categories) > 0
    assert set(with_ood.trained_ood_categories) <= set(with_ood.supported_ood_categories)


def test_build_checkpoint_identity_rejects_invalid_output_governance() -> None:
    model = _tiny_model()
    with pytest.raises(ci.CheckpointIdentityError, match="output governance invalid"):
        _identity_for(model, runtime_enabled_outputs=frozenset({"source_region"}))


def test_valid_identity_round_trips_through_validate() -> None:
    identity = _identity_for(_tiny_model())
    ci.validate_checkpoint_identity(identity)  # must not raise


def test_architecture_config_records_every_construction_field() -> None:
    model = _tiny_model(prior_mode="logit_only", incident_pooling="attention", activation="gelu")
    identity = _identity_for(model)
    assert identity.prior_mode == "logit_only"
    assert identity.incident_pooling == "attention"
    assert identity.activation == "gelu"
    assert identity.d_model == 32
    assert identity.action_template_names[0] == "NO_ACTION"


# --- verify_model_matches_identity: behavior-critical mismatches ---------


def test_changed_activation_fails() -> None:
    identity = _identity_for(_tiny_model(activation="silu"))
    other = _tiny_model(activation="gelu")
    with pytest.raises(ArchitectureCompatibilityError, match="activation"):
        ci.verify_model_matches_identity(other, identity)


def test_changed_normalization_fails() -> None:
    identity = _identity_for(_tiny_model(normalization="rmsnorm"))
    other = _tiny_model(normalization="layernorm")
    with pytest.raises(ArchitectureCompatibilityError, match="normalization"):
        ci.verify_model_matches_identity(other, identity)


def test_verify_model_matches_identity_round_trip_succeeds() -> None:
    model = _tiny_model()
    identity = _identity_for(model)
    rebuilt = ci.build_model_from_identity(identity)
    ci.verify_model_matches_identity(rebuilt, identity)  # must not raise


# --- validate_checkpoint_identity: stale/tampered identity fields --------


def test_changed_action_template_ordering_fails() -> None:
    identity = _identity_for(_tiny_model())
    reordered = replace(identity, action_template_names=tuple(reversed(identity.action_template_names)))
    with pytest.raises(ci.CheckpointIdentityError, match="action_template_names"):
        ci.validate_checkpoint_identity(reordered)


def test_changed_action_template_schema_hash_fails() -> None:
    identity = _identity_for(_tiny_model())
    tampered = replace(identity, action_template_schema_hash="stale-hash")
    with pytest.raises(ci.CheckpointIdentityError, match="action_template_schema_hash"):
        ci.validate_checkpoint_identity(tampered)


def test_changed_ood_category_ordering_fails() -> None:
    identity = _identity_for(_tiny_model())
    reordered = replace(identity, ood_category_names=tuple(reversed(identity.ood_category_names)))
    with pytest.raises(ci.CheckpointIdentityError, match="ood_category_names"):
        ci.validate_checkpoint_identity(reordered)


def test_changed_plan_value_policy_fails() -> None:
    identity = _identity_for(_tiny_model())
    stale = replace(identity, plan_value_policy_version="plan-value-policy-v0-stale")
    with pytest.raises(ci.CheckpointIdentityError, match="plan_value_policy_version"):
        ci.validate_checkpoint_identity(stale)


def test_changed_signature_policy_fails() -> None:
    identity = _identity_for(_tiny_model())
    stale = replace(identity, signature_artifact_policy_version="per-scenario-v0-stale")
    with pytest.raises(ci.CheckpointIdentityError, match="signature_artifact_policy_version"):
        ci.validate_checkpoint_identity(stale)


def test_unsupported_ood_category_recorded_as_supported_fails() -> None:
    identity = _identity_for(_tiny_model())
    tampered = replace(identity, supported_ood_categories=("NOT_A_REAL_CATEGORY",))
    with pytest.raises(ci.CheckpointIdentityError, match="unknown names"):
        ci.validate_checkpoint_identity(tampered)


# --- save/load round trip -------------------------------------------------


def test_v4_save_load_round_trip(tmp_path) -> None:
    model = _tiny_model()
    identity = _identity_for(model)
    optimizer, scheduler = _optimizer_scheduler(model)
    directory = tmp_path / "checkpoint"
    ci.save_v4_checkpoint(
        directory,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        global_step=42,
        best_validation_loss=0.75,
        identity=identity,
        resolved_training_config={"lr": 1e-3},
        dataset_manifest_hashes={"train": "abc123"},
    )
    for name in (
        ci.MODEL_WEIGHTS_FILENAME,
        ci.OPTIMIZER_STATE_FILENAME,
        ci.TRAINER_STATE_FILENAME,
        ci.IDENTITY_FILENAME,
        ci.RESOLVED_CONFIG_FILENAME,
        ci.ARTIFACT_MANIFEST_FILENAME,
    ):
        assert (directory / name).exists(), name

    loaded_model, loaded_identity, trainer_state = ci.load_v4_checkpoint(directory)
    assert loaded_identity.fingerprint() == identity.fingerprint()
    assert trainer_state == {"epoch": 3, "global_step": 42, "best_validation_loss": 0.75}
    # weights actually match (not just the identity)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded_model.state_dict()[key])


def test_missing_identity_file_fails(tmp_path) -> None:
    model = _tiny_model()
    optimizer, scheduler = _optimizer_scheduler(model)
    directory = tmp_path / "no-identity"
    save_checkpoint(
        directory, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=1, global_step=1, best_validation_loss=1.0,
    )
    with pytest.raises(ci.NotAV4CheckpointError):
        ci.load_v4_checkpoint(directory)


def test_altered_identity_fingerprint_fails(tmp_path) -> None:
    model = _tiny_model()
    identity = _identity_for(model)
    optimizer, scheduler = _optimizer_scheduler(model)
    directory = tmp_path / "tampered"
    ci.save_v4_checkpoint(
        directory, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=1, global_step=1, best_validation_loss=1.0, identity=identity,
        resolved_training_config={}, dataset_manifest_hashes={"train": "abc123"},
    )
    identity_path = directory / ci.IDENTITY_FILENAME
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    payload["d_model"] = payload["d_model"] + 1
    identity_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ci.CheckpointIdentityError, match="fingerprint"):
        ci.load_v4_checkpoint(directory)


def test_v3_checkpoint_still_loads_through_the_legacy_loader(tmp_path) -> None:
    model = _tiny_model()
    optimizer, scheduler = _optimizer_scheduler(model)
    directory = tmp_path / "v3"
    save_checkpoint(
        directory, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=2, global_step=5, best_validation_loss=0.9,
    )
    fresh = _tiny_model()
    state = load_checkpoint(directory, model=fresh)
    assert state["epoch"] == 2


def test_v3_checkpoint_cannot_load_through_the_v4_loader(tmp_path) -> None:
    model = _tiny_model()
    optimizer, scheduler = _optimizer_scheduler(model)
    directory = tmp_path / "v3-only"
    save_checkpoint(
        directory, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=1, global_step=1, best_validation_loss=1.0,
    )
    with pytest.raises(ci.NotAV4CheckpointError):
        ci.load_v4_checkpoint(directory)


def test_v4_checkpoint_cannot_load_through_the_legacy_path(tmp_path) -> None:
    model = _tiny_model()
    identity = _identity_for(model)
    optimizer, scheduler = _optimizer_scheduler(model)
    directory = tmp_path / "v4-only"
    ci.save_v4_checkpoint(
        directory, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=1, global_step=1, best_validation_loss=1.0, identity=identity,
        resolved_training_config={}, dataset_manifest_hashes={"train": "abc123"},
    )
    with pytest.raises(LegacyLoaderRejectedV4CheckpointError):
        load_checkpoint(directory, model=_tiny_model())


def test_from_checkpoint_identity_is_attached_to_hydrocore() -> None:
    identity = _identity_for(_tiny_model())
    model = HydroCore.from_checkpoint_identity(identity)
    assert isinstance(model, HydroCore)
    assert model.variant_name == identity.variant
    ci.verify_model_matches_identity(model, identity)
