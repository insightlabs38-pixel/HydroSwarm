from __future__ import annotations

import math

import numpy as np
import pytest

from hydroswarm.classical import (
    HydraulicLink,
    SensorObservation,
    SignatureLibrary,
    abstention_quality,
    bayesian_source_posterior,
    build_dynamic_graph,
    candidate_set_metrics,
    information_gain_per_sample,
    localization_top_k,
    mean_reciprocal_rank,
    pressure_violations,
    screen_candidates,
)


def test_dynamic_graph_tracks_transport_diagnosis_and_flow_reversal() -> None:
    graph = build_dynamic_graph(
        [
            HydraulicLink("p1", "A", "B", 1.0, 10.0, 1.0),
            HydraulicLink("p2", "B", "C", -2.0, 10.0, 1.0),
            HydraulicLink("p3", "C", "D", 0.0, 10.0, 1.0),
        ]
    )

    assert graph.transport.has_edge("A", "B", "p1")
    assert graph.transport.has_edge("C", "B", "p2")
    assert graph.transport["C"]["B"]["p2"]["flow_reversed"] is True
    assert graph.diagnostic.has_edge("B", "A", "p1")
    assert graph.diagnostic.has_edge("B", "C", "p2")
    assert not graph.transport.has_edge("C", "D")
    assert graph.possible_sources("B") == {"A", "B", "C"}


def test_screening_hard_masks_positive_impossibility_and_ignores_missing() -> None:
    graph = build_dynamic_graph(
        [
            HydraulicLink("p1", "A", "B", 1.0, 10.0, 1.0),
            HydraulicLink("p2", "B", "C", 1.0, 10.0, 1.0),
        ]
    )
    travel = graph.shortest_travel_time("A", "C")
    result = screen_candidates(
        graph,
        [
            SensorObservation("C", travel + 1.0, True, value=0.8),
            SensorObservation("B", travel + 1.0, False, value=0.0),
            SensorObservation("A", 1.0, True, value=None),
        ],
        candidates=["A", "C"],
    )

    assert result.candidate_mask == {"A": True, "C": True}
    assert result.features["A"].positive_sensor_consistency == 1.0
    assert result.features["C"].negative_sensor_consistency == 1.0
    assert result.ignored_sensor_count == 1

    impossible = screen_candidates(
        graph,
        [SensorObservation("A", 100.0, True, value=1.0)],
        candidates=["C"],
    )
    assert impossible.candidate_mask["C"] is False
    assert math.isinf(impossible.features["C"].min_travel_time)


def test_signature_library_and_bayesian_posterior_use_residuals_and_masks() -> None:
    library = SignatureLibrary()
    library.add("A", [0.0, 1.0, 2.0])
    library.add("B", [2.0, 1.0, 0.0])

    assert library.compressed_bytes == 12
    assert library.get("A").dtype == np.float32
    posterior = bayesian_source_posterior(
        [np.nan, 1.0, 2.1], library, noise_scale=0.2
    )
    assert posterior.used_observations == 2
    assert posterior.probabilities["A"] > 0.99
    assert sum(posterior.probabilities.values()) == pytest.approx(1.0)

    missing = bayesian_source_posterior(
        [np.nan, np.nan, np.nan], library, prior={"A": 1.0, "B": 3.0}
    )
    assert missing.probabilities == pytest.approx({"A": 0.25, "B": 0.75})

    masked = bayesian_source_posterior(
        [0.0, 1.0, 2.0], library, feasible_mask={"A": False, "B": True}
    )
    assert masked.probabilities == pytest.approx({"A": 0.0, "B": 1.0})


def test_operational_metrics() -> None:
    prediction = {"A": 0.6, "B": 0.3, "C": 0.1}
    assert localization_top_k(prediction, "B", k=1) == 0.0
    assert localization_top_k(prediction, "B", k=2) == 1.0
    assert mean_reciprocal_rank([prediction, prediction], ["A", "C"]) == pytest.approx(
        (1.0 + 1.0 / 3.0) / 2.0
    )

    sets = candidate_set_metrics([{"A", "B"}, {"A"}], ["B", "C"])
    assert sets.coverage == 0.5
    assert sets.average_size == 1.5

    gain = information_gain_per_sample([[0.5, 0.5], [1.0, 0.0]])
    assert gain == pytest.approx(1.0)

    pressure = pressure_violations(
        [[20.0, 14.0], [13.0, 17.0]], minimum_allowed=15.0, time_step=5.0
    )
    assert pressure.count == 2
    assert pressure.duration == 10.0
    assert pressure.minimum_pressure == 13.0

    abstention = abstention_quality(
        [False, True, False, True], safe_if_answered=[True, True, False, False]
    )
    assert abstention.false_abstention_rate == 0.5
    assert abstention.unsafe_non_abstention_rate == 0.5
    assert abstention.abstention_rate == 0.5
    assert abstention.selective_accuracy == 0.5


def test_validation_is_deterministic_and_strict() -> None:
    with pytest.raises(ValueError, match="duplicate link_id"):
        build_dynamic_graph(
            [
                HydraulicLink("p", "A", "B", 1.0, 1.0, 1.0),
                HydraulicLink("p", "B", "C", 1.0, 1.0, 1.0),
            ]
        )
    with pytest.raises(ValueError, match="shape"):
        library = SignatureLibrary()
        library.add("A", [1.0, 2.0])
        library.add("B", [1.0])

