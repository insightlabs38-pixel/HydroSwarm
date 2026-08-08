"""core-issues5.txt Section 4 (P0 blocker): versioned classical-signature
generation policy."""

from __future__ import annotations

from dataclasses import replace

from hydroswarm.classical.signature_policy import (
    GOVERNED_TRAINING_SIGNATURE_POLICY,
    KNOWN_TRAINING_TOPOLOGY_HASHES,
    SignaturePolicy,
    resolve_signature_mode,
)


def test_policy_hash_is_deterministic() -> None:
    assert GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash == GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash
    rebuilt = SignaturePolicy(**{
        field: getattr(GOVERNED_TRAINING_SIGNATURE_POLICY, field)
        for field in GOVERNED_TRAINING_SIGNATURE_POLICY.__dataclass_fields__
    })
    assert rebuilt.policy_hash == GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash


def test_different_bins_change_the_policy_hash() -> None:
    changed = replace(GOVERNED_TRAINING_SIGNATURE_POLICY, start_time_bins=(0, 60))
    assert changed.policy_hash != GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash


def test_different_sensor_layout_policy_changes_the_policy_hash() -> None:
    changed = replace(GOVERNED_TRAINING_SIGNATURE_POLICY, sensor_layout_policy="deployed_sensors_only")
    assert changed.policy_hash != GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash


def test_different_policy_version_changes_the_policy_hash() -> None:
    changed = replace(GOVERNED_TRAINING_SIGNATURE_POLICY, policy_version="hydroswarm-signature-policy-v2")
    assert changed.policy_hash != GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash


def test_known_training_topology_resolves_to_governed_mode() -> None:
    for topology_hash in KNOWN_TRAINING_TOPOLOGY_HASHES:
        assert resolve_signature_mode(topology_hash) == "GOVERNED_KNOWN_NETWORK"


def test_unknown_topology_resolves_to_runtime_generated_mode() -> None:
    assert resolve_signature_mode("0" * 64) == "RUNTIME_GENERATED_IMPORTED_NETWORK"


def test_known_training_topology_hashes_match_the_real_training_networks() -> None:
    """Regression: these constants must stay in sync with the actual
    networks scripts/generate_cycle_b_corpus.py builds cycle-b2's train
    split from, not silently drift from a stale hand-copied value."""

    import wntr

    from hydroswarm.data.scenarios import network_sha256
    from hydroswarm.simulation import build_wntr_network

    repo_root_data = __import__("pathlib").Path(__file__).resolve().parents[2] / "data"

    golden = build_wntr_network()
    branched = wntr.network.WaterNetworkModel(str(repo_root_data / "topology-transfer" / "branched-loop.inp"))
    loop_grid = wntr.network.WaterNetworkModel(str(repo_root_data / "topologies" / "loop-grid.inp"))

    assert {network_sha256(golden), network_sha256(branched), network_sha256(loop_grid)} == (
        KNOWN_TRAINING_TOPOLOGY_HASHES
    )
