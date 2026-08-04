from dataclasses import dataclass

import pandas as pd
import pytest

from hydroswarm.classical.signature_registry import SignatureRegistry, SignatureRegistryError
from hydroswarm.classical.signatures import SignatureBuilder, SignatureCache, SignatureCacheKey


@dataclass
class Result:
    concentration_mg_l: pd.DataFrame


class FakeSimulator:
    simulator_version = "test-1"

    def __init__(self) -> None:
        self.calls = 0

    def simulate_incident(self, source_node_id, *, strength_mg_min, start_minute, duration_minutes):
        self.calls += 1
        base = 1.0 if source_node_id == "J1" else 0.1
        return Result(
            pd.DataFrame(
                {"S1": [0.0, base * strength_mg_min / 10], "S2": [0.0, base / 2]},
                index=[0, 3600],
            )
        )


def _key(network_hash: str = "a" * 64, hydraulic_state_hash: str = "b" * 64) -> SignatureCacheKey:
    return SignatureCacheKey(network_hash, hydraulic_state_hash, "test-1", "c" * 64, "d" * 64)


def _build(cache: SignatureCache, key: SignatureCacheKey):
    simulator = FakeSimulator()
    builder = SignatureBuilder(simulator, cache)
    return builder.build_or_load(
        key=key,
        source_nodes=["J1", "J2"],
        start_time_bins=[0],
        duration_bins=[60],
        strength_bins=[1.0],
        demand_regimes=["nominal"],
        sensor_nodes=["S1", "S2"],
        sample_times_seconds=[0, 3600],
    )


def test_register_and_lookup_round_trip(tmp_path) -> None:
    cache = SignatureCache(tmp_path / "signatures")
    registry = SignatureRegistry(cache)
    key = _key()
    artifact = _build(cache, key)

    registry.register(
        topology_hash="topo-1", hydraulic_regime_hash="regime-1", key=key, artifact=artifact, fit_split="train"
    )
    found = registry.lookup(topology_hash="topo-1", hydraulic_regime_hash="regime-1", key=key)
    assert found is not None
    assert found.artifact_hash == artifact.artifact_hash


def test_test_only_topology_is_rejected_during_fitting(tmp_path) -> None:
    cache = SignatureCache(tmp_path / "signatures")
    registry = SignatureRegistry(cache)
    key = _key()
    artifact = _build(cache, key)

    for bad_split in ("test", "validation", "calibration"):
        with pytest.raises(SignatureRegistryError, match="training data only"):
            registry.register(
                topology_hash="topo-1",
                hydraulic_regime_hash="regime-1",
                key=key,
                artifact=artifact,
                fit_split=bad_split,
            )


def test_missing_signatures_are_reported_clearly(tmp_path) -> None:
    cache = SignatureCache(tmp_path / "signatures")
    registry = SignatureRegistry(cache)
    key = _key()

    assert registry.lookup(topology_hash="unknown", hydraulic_regime_hash="unknown", key=key) is None
    with pytest.raises(SignatureRegistryError, match="no registered signatures"):
        registry.require(topology_hash="unknown", hydraulic_regime_hash="unknown", key=key)


def test_separate_topologies_produce_separate_signature_hashes(tmp_path) -> None:
    cache = SignatureCache(tmp_path / "signatures")
    registry = SignatureRegistry(cache)
    key_a = _key(network_hash="a" * 64)
    key_b = _key(network_hash="b" * 64)
    assert key_a.digest != key_b.digest

    artifact_a = _build(cache, key_a)
    artifact_b = _build(cache, key_b)
    registry.register(topology_hash="topo-a", hydraulic_regime_hash="r", key=key_a, artifact=artifact_a, fit_split="train")
    registry.register(topology_hash="topo-b", hydraulic_regime_hash="r", key=key_b, artifact=artifact_b, fit_split="train")

    entry_a = registry.entry(topology_hash="topo-a", hydraulic_regime_hash="r")
    entry_b = registry.entry(topology_hash="topo-b", hydraulic_regime_hash="r")
    assert entry_a.cache_key_digest != entry_b.cache_key_digest


def test_signature_lookup_cannot_cross_topology_hashes(tmp_path) -> None:
    cache = SignatureCache(tmp_path / "signatures")
    registry = SignatureRegistry(cache)
    key_a = _key(network_hash="a" * 64)
    key_b = _key(network_hash="b" * 64)
    artifact_a = _build(cache, key_a)
    registry.register(topology_hash="topo-a", hydraulic_regime_hash="r", key=key_a, artifact=artifact_a, fit_split="train")

    # Even though "topo-a"/"r" is registered, asking with topo-b's key must not
    # return topo-a's artifact -- the digest mismatch must be caught.
    crossed = registry.lookup(topology_hash="topo-a", hydraulic_regime_hash="r", key=key_b)
    assert crossed is None


def test_exact_repeated_fitting_is_deterministic(tmp_path) -> None:
    cache = SignatureCache(tmp_path / "signatures")
    registry = SignatureRegistry(cache)
    key = _key()
    artifact = _build(cache, key)

    first = registry.register(
        topology_hash="topo-1", hydraulic_regime_hash="regime-1", key=key, artifact=artifact, fit_split="train"
    )
    # Re-fitting (build_or_load hits the cache) and re-registering must be
    # byte-for-byte reproducible.
    rebuilt = _build(cache, key)
    second = registry.register(
        topology_hash="topo-1", hydraulic_regime_hash="regime-1", key=key, artifact=rebuilt, fit_split="train"
    )
    assert first.artifact_hash == second.artifact_hash == artifact.artifact_hash


def test_topology_hashes_lists_registered_topologies(tmp_path) -> None:
    cache = SignatureCache(tmp_path / "signatures")
    registry = SignatureRegistry(cache)
    key = _key()
    artifact = _build(cache, key)
    registry.register(topology_hash="topo-z", hydraulic_regime_hash="r", key=key, artifact=artifact, fit_split="train")
    registry.register(topology_hash="topo-a", hydraulic_regime_hash="r", key=key, artifact=artifact, fit_split="train")
    assert registry.topology_hashes() == ("topo-a", "topo-z")


def test_registry_index_persists_across_instances(tmp_path) -> None:
    cache = SignatureCache(tmp_path / "signatures")
    key = _key()
    artifact = _build(cache, key)
    SignatureRegistry(cache).register(
        topology_hash="topo-1", hydraulic_regime_hash="regime-1", key=key, artifact=artifact, fit_split="train"
    )

    reloaded_registry = SignatureRegistry(cache)
    found = reloaded_registry.lookup(topology_hash="topo-1", hydraulic_regime_hash="regime-1", key=key)
    assert found is not None
    assert found.artifact_hash == artifact.artifact_hash
