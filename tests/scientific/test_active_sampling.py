import numpy as np

from hydroswarm.classical.prior import SignatureLibrary
from hydroswarm.classical.signatures import SignatureArtifact, SignatureCacheKey, SourceHypothesis
from hydroswarm.sampling.active import SamplingConstraints, rank_sample_locations


def artifact() -> SignatureArtifact:
    hypotheses = (
        SourceHypothesis("A", 0, 60, 1.0, "nominal"),
        SourceHypothesis("B", 0, 60, 1.0, "nominal"),
    )
    library = SignatureLibrary()
    library.add(hypotheses[0].identifier, np.array([[0.0, 0.0], [0.2, 1.0]]))
    library.add(hypotheses[1].identifier, np.array([[0.0, 0.0], [0.2, 0.0]]))
    return SignatureArtifact(
        key=SignatureCacheKey("a", "b", "v", "c", "d"), library=library,
        hypotheses=hypotheses, sensor_nodes=("S1", "S2"), sample_times_seconds=(0, 60),
        cache_hit=True, artifact_hash="f" * 64,
    )


def test_information_gain_selects_discriminating_sensor() -> None:
    data = artifact()
    posterior = {item.identifier: 0.5 for item in data.hypotheses}
    result = rank_sample_locations(data, posterior, noise_scale_mg_l=0.05)
    assert result.recommended_node == "S2"
    assert result.ranked[0].expected_information_gain_bits > 0.9
    assert result.ranked[0].leading_hypothesis_separation == 1.0


def test_duplicate_and_inaccessible_samples_are_not_recommended() -> None:
    data = artifact()
    posterior = {item.identifier: 0.5 for item in data.hypotheses}
    result = rank_sample_locations(
        data, posterior,
        constraints=SamplingConstraints(already_sampled=frozenset({"S2"}), accessible={"S1": False}),
    )
    assert result.stop
    assert result.stop_reason == "no_accessible_sample"
    assert result.recommended_node is None


def test_explicit_acquisition_time_does_not_peek_at_future_signature_peaks() -> None:
    data = artifact()
    posterior = {item.identifier: 0.5 for item in data.hypotheses}
    # S2 distinguishes hypotheses only at 60 seconds. At time zero, with no
    # collection delay, causal ranking must not use that future observation.
    result = rank_sample_locations(
        data,
        posterior,
        constraints=SamplingConstraints(collection_time_minutes={"S1": 0.0, "S2": 0.0}),
        target_sample_time_seconds=0.0,
    )
    assert result.stop
    assert result.stop_reason == "marginal_value_below_threshold"


def test_collection_delay_selects_the_measurement_time_used_for_ranking() -> None:
    data = artifact()
    posterior = {item.identifier: 0.5 for item in data.hypotheses}
    result = rank_sample_locations(
        data,
        posterior,
        constraints=SamplingConstraints(collection_time_minutes={"S1": 0.0, "S2": 1.0}),
        target_sample_time_seconds=0.0,
    )
    assert result.recommended_node == "S2"
    assert result.ranked[0].collection_time_minutes == 1.0
