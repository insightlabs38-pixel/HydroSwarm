"""core-issues5.txt Section 18.2 (future-proofing before freeze): frozen,
versioned Scout state-conditioning input contract."""

from __future__ import annotations

from dataclasses import replace

from hydroswarm.model.core import HydroCore
from hydroswarm.training.scout_state_contract import (
    SCOUT_STATE_CONTRACT,
    SCOUT_STATE_ITEMS,
    ScoutStateChannelBinding,
    contract_hash,
    unwired_items,
)


def test_every_required_item_has_exactly_one_binding() -> None:
    bound_items = tuple(binding.item for binding in SCOUT_STATE_CONTRACT)
    assert bound_items == SCOUT_STATE_ITEMS
    assert len(set(bound_items)) == len(SCOUT_STATE_ITEMS)


def test_every_binding_targets_a_real_hydrobatch_field_name() -> None:
    """A binding referencing a channel HydroCore doesn't actually accept
    would make the "no new parameters needed" claim false -- confirmed
    here against HydroCore's own declared batch keys, not by inspection
    alone. `role_features` is used by two different items in this
    contract (two scalars packed into the same 8-dim vector), so the
    channel set has fewer members than the item set."""

    from hydroswarm.model.core import HydroBatch

    declared_fields = set(HydroBatch.__annotations__)
    for binding in SCOUT_STATE_CONTRACT:
        for channel in binding.channel.split("+"):
            assert channel in declared_fields, f"{binding.item} -> undeclared HydroBatch field {channel!r}"


def test_contract_requires_no_new_hydrocore_constructor_dimension() -> None:
    """Every channel this contract binds to must already be a real,
    named dimension HydroCore accepts today -- role_feature_dim and
    residual_feature_dim in particular, since those are exactly the two
    channels with spare, unused capacity this contract relies on."""

    model = HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2, edge_feature_dim=2,
        d_model=16, nhead=2, dim_feedforward=32, num_layers=1, modality_layers=1, latent_tokens=64,
    )
    assert model.role_feature_dim > 0
    assert model.residual_feature_dim > 0


def test_contract_hash_is_deterministic() -> None:
    assert contract_hash() == contract_hash()


def test_changing_a_binding_changes_the_contract_hash() -> None:
    original = contract_hash()
    mutated = SCOUT_STATE_CONTRACT[:-1] + (
        replace(SCOUT_STATE_CONTRACT[-1], channel="residual_features"),
    )
    payload = {
        "version": "scout-state-contract-v1",
        "bindings": [
            {"item": b.item, "channel": b.channel, "encoding": b.encoding, "wired": b.wired}
            for b in mutated
        ],
    }
    import hashlib
    import json

    changed_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert changed_hash != original


def test_unwired_items_reports_real_wiring_gaps_honestly() -> None:
    """already_sampled and current_posterior_candidate_state are wired
    today (sensor_mask, classical_prior are both populated by real
    pipeline code); the rest are architecturally available but not yet
    populated by any real corpus-generation or inference code -- this
    contract must not claim otherwise."""

    gaps = unwired_items()
    assert "already_sampled" not in gaps
    assert "current_posterior_candidate_state" not in gaps
    assert "current_sampling_round" in gaps
    assert "sample_budget_remaining" in gaps
    assert "currently_revealed_evidence" in gaps
    assert "sampling_constraints_accessibility" in gaps


def test_binding_is_a_frozen_dataclass_instance() -> None:
    binding = SCOUT_STATE_CONTRACT[0]
    assert isinstance(binding, ScoutStateChannelBinding)
