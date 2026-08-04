from __future__ import annotations

import pytest
import wntr

from hydroswarm.simulation import (
    HydraulicContextCache,
    HydraulicSimulator,
    ScenarioHydraulicContextKey,
    build_wntr_network,
)


def _simulator(*, close_feeder: bool = False) -> HydraulicSimulator:
    network = build_wntr_network()
    network.options.time.duration = 4 * 3600
    if close_feeder:
        network.get_link("P_R1_J1").initial_status = wntr.network.LinkStatus.Closed
    return HydraulicSimulator(network)


def _key(simulator: HydraulicSimulator, *, timestamp: int = 3600, config_hash: str = "estimator-v1") -> ScenarioHydraulicContextKey:
    return ScenarioHydraulicContextKey.from_simulator(
        simulator, simulation_timestamp_seconds=timestamp, state_estimator_config_hash=config_hash
    )


def test_changing_valve_state_changes_the_context_key_digest() -> None:
    open_key = _key(_simulator(close_feeder=False))
    closed_key = _key(_simulator(close_feeder=True))
    assert open_key.network_state_hash != closed_key.network_state_hash
    assert open_key.digest != closed_key.digest


def test_changing_simulation_timestamp_changes_the_digest_even_with_identical_network_state() -> None:
    simulator = _simulator()
    early = _key(simulator, timestamp=0)
    later = _key(simulator, timestamp=3600)
    assert early.network_state_hash == later.network_state_hash  # same network config
    assert early.digest != later.digest  # but distinct scenario contexts


def test_changing_state_estimator_config_changes_the_digest() -> None:
    simulator = _simulator()
    key_a = _key(simulator, config_hash="estimator-v1")
    key_b = _key(simulator, config_hash="estimator-v2")
    assert key_a.digest != key_b.digest


def test_key_construction_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError, match="negative"):
        _key(_simulator(), timestamp=-1)


def test_identical_scenarios_reuse_the_cached_context(tmp_path) -> None:
    cache = HydraulicContextCache(tmp_path / "contexts")
    key = _key(_simulator())
    calls = 0

    def builder() -> dict:
        nonlocal calls
        calls += 1
        return {"edges": [["R1", "J1"], ["J1", "J2"]]}

    first = cache.get_or_build(key, builder)
    second = cache.get_or_build(key, builder)
    assert first == second == {"edges": [["R1", "J1"], ["J1", "J2"]]}
    assert calls == 1  # builder only ran once; second call was a cache hit
    assert cache.hits == 1
    assert cache.misses == 1


def test_different_hydraulic_states_never_collide(tmp_path) -> None:
    cache = HydraulicContextCache(tmp_path / "contexts")
    open_key = _key(_simulator(close_feeder=False))
    closed_key = _key(_simulator(close_feeder=True))

    cache.get_or_build(open_key, lambda: {"state": "open"})
    cache.get_or_build(closed_key, lambda: {"state": "closed"})

    assert cache.get(open_key) == {"state": "open"}
    assert cache.get(closed_key) == {"state": "closed"}


def test_corrupted_cache_entry_is_treated_as_a_miss_not_a_false_hit(tmp_path) -> None:
    cache = HydraulicContextCache(tmp_path / "contexts")
    key = _key(_simulator())
    cache.get_or_build(key, lambda: {"state": "original"})

    path = cache._path(key)
    path.write_text("not valid json", encoding="utf-8")

    rebuilt = cache.get_or_build(key, lambda: {"state": "rebuilt"})
    assert rebuilt == {"state": "rebuilt"}


def test_key_requires_all_provenance_fields() -> None:
    with pytest.raises(ValueError, match="required"):
        ScenarioHydraulicContextKey(
            network_state_hash="",
            simulator_version="v1",
            simulation_timestamp_seconds=0,
            state_estimator_config_hash="cfg",
        )
