from dataclasses import dataclass

import pandas as pd

from hydroswarm.classical.signatures import (
    SignatureBuilder,
    SignatureCache,
    SignatureCacheKey,
    localize_with_signatures,
)


@dataclass
class Result:
    concentration_mg_l: pd.DataFrame


class FakeSimulator:
    simulator_version = "test-1"
    calls = 0

    def simulate_incident(self, source_node_id, *, strength_mg_min, start_minute, duration_minutes):
        self.calls += 1
        base = 1.0 if source_node_id == "J1" else 0.1
        return Result(pd.DataFrame(
            {"S1": [0.0, base * strength_mg_min / 10], "S2": [0.0, base / 2]},
            index=[0, 3600],
        ))


def cache_key() -> SignatureCacheKey:
    return SignatureCacheKey("a" * 64, "b" * 64, "test-1", "c" * 64, "d" * 64)


def test_signature_cache_build_load_and_localization(tmp_path) -> None:
    simulator = FakeSimulator()
    builder = SignatureBuilder(simulator, SignatureCache(tmp_path / "signatures"))
    kwargs = dict(
        key=cache_key(), source_nodes=["J1", "J2"], start_time_bins=[0], duration_bins=[60],
        strength_bins=[1.0], demand_regimes=["nominal"], sensor_nodes=["S1", "S2"],
        sample_times_seconds=[0, 3600],
    )
    built = builder.build_or_load(**kwargs)
    assert not built.cache_hit
    assert simulator.calls == 2
    loaded = builder.build_or_load(**kwargs)
    assert loaded.cache_hit
    assert simulator.calls == 2

    result = localize_with_signatures(
        [[0.0, 0.0], [1.0, 0.5]], loaded, noise_scale=0.05,
        feasible_sources={"J1": True, "J2": True},
    )
    assert result.ranked_hypotheses[0].hypothesis.source_node == "J1"
    assert result.source_probabilities["J1"] > 0.99
    assert result.signature_artifact_hash == loaded.artifact_hash


def test_corrupt_cache_forces_exact_rebuild(tmp_path) -> None:
    simulator = FakeSimulator()
    cache = SignatureCache(tmp_path / "signatures")
    builder = SignatureBuilder(simulator, cache)
    kwargs = dict(
        key=cache_key(), source_nodes=["J1"], start_time_bins=[0], duration_bins=[60],
        strength_bins=[1.0], demand_regimes=["nominal"], sensor_nodes=["S1"],
        sample_times_seconds=[0, 3600],
    )
    builder.build_or_load(**kwargs)
    arrays_path = next((tmp_path / "signatures").glob("*.npz"))
    arrays_path.write_bytes(b"corrupt")
    rebuilt = builder.build_or_load(**kwargs)
    assert not rebuilt.cache_hit
    assert simulator.calls == 2
