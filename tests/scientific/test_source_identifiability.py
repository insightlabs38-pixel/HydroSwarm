"""Contract tests for the source-identifiability analysis (`scripts/
hydrocore_v5/source_identifiability/`).

This analysis is DIAGNOSTIC / ANALYSIS-ONLY (see
docs/evaluation/SOURCE_IDENTIFIABILITY_ANALYSIS_PROTOCOL.md) -- these tests
cover the pure-computation machinery's own correctness (distance-metric
arithmetic, identifiability-score edge cases, bootstrap determinism,
structural-feature correctness against a hand-built graph) plus one
`real_simulation` end-to-end reproducibility check against a real locked
scenario spec. Never asserts a promotion decision, never mutates locked
data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT / "src"))

from source_identifiability import centrality, common, signatures, stats_utils  # noqa: E402


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


def test_rmse_distance_zero_for_identical_arrays():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert signatures.rmse_distance(a, a) == 0.0


def test_rmse_distance_matches_hand_computation():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert signatures.rmse_distance(a, b) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_cosine_distance_orthogonal_vectors_is_one():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert signatures.cosine_distance(a, b) == pytest.approx(1.0)


def test_cosine_distance_identical_direction_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    b = a * 5.0
    assert signatures.cosine_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_zero_vector_is_maximal_not_nan():
    a = np.zeros(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert signatures.cosine_distance(a, b) == 1.0


def test_correlation_distance_perfect_positive_correlation_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = 2.0 * a + 1.0
    assert signatures.correlation_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_correlation_distance_both_constant_and_equal_is_zero():
    a = np.zeros(5)
    b = np.zeros(5)
    assert signatures.correlation_distance(a, b) == 0.0


def test_correlation_distance_one_constant_one_not_is_maximal():
    a = np.zeros(5)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert signatures.correlation_distance(a, b) == 1.0


def test_arrival_order_distance_is_mean_absolute_difference():
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([10.0, 25.0, 20.0])
    assert signatures.arrival_order_distance(a, b) == pytest.approx((0 + 5 + 10) / 3)


# ---------------------------------------------------------------------------
# Signature-set construction
# ---------------------------------------------------------------------------


def test_build_signature_set_normalizes_and_detects_arrival():
    timestamps = [0, 60, 120, 180]
    concentrations = {
        "A": np.array([[0.0], [0.0], [5.0], [5.0]]),
        "B": np.array([[0.0], [5.0], [5.0], [0.0]]),
    }
    sig_set = signatures.build_signature_set(concentrations, sensor_nodes=("S1",), timestamps_seconds=timestamps)
    assert sig_set.candidates == ("A", "B")
    # A first crosses the detection threshold at t=120, B at t=60.
    assert sig_set.arrival_order["A"][0] == 120
    assert sig_set.arrival_order["B"][0] == 60
    # normalized signatures have unit L2 norm.
    assert np.linalg.norm(sig_set.normalized["A"]) == pytest.approx(1.0)
    assert np.linalg.norm(sig_set.normalized["B"]) == pytest.approx(1.0)


def test_build_signature_set_rejects_negative_concentration():
    with pytest.raises(ValueError):
        signatures.build_signature_set(
            {"A": np.array([[-1.0]])}, sensor_nodes=("S1",), timestamps_seconds=[0]
        )


# ---------------------------------------------------------------------------
# Identifiability metrics
# ---------------------------------------------------------------------------


def test_identifiability_metrics_hand_computed():
    candidates = ("A", "B", "C")
    # distance(A,B)=1, distance(A,C)=2, distance(B,C)=3
    matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    result = signatures.identifiability_metrics(candidates, matrix, true_source="A", noise_floor_distance=1.5)
    assert result.nearest_competitor == "B"
    assert result.nearest_competitor_distance == 1.0
    assert result.second_nearest_distance == 2.0
    assert result.margin == pytest.approx(1.0)
    # incident_mean_pairwise_distance is mean of the upper triangle: (1+2+3)/3 = 2.
    assert result.incident_mean_pairwise_distance == pytest.approx(2.0)
    assert result.identifiability_score == pytest.approx(0.5)
    assert result.ambiguity_count_noise_floor == 1  # only distance(A,B)=1.0 <= 1.5


def test_identifiability_metrics_degenerate_zero_distance_scores_zero_not_nan():
    candidates = ("A", "B", "C")
    matrix = np.zeros((3, 3))
    result = signatures.identifiability_metrics(candidates, matrix, true_source="A", noise_floor_distance=0.1)
    assert result.identifiability_score == 0.0
    assert not np.isnan(result.identifiability_score)


def test_identifiability_metrics_requires_true_source_in_candidates():
    with pytest.raises(ValueError):
        signatures.identifiability_metrics(("A", "B"), np.zeros((2, 2)), true_source="Z", noise_floor_distance=0.1)


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------


def test_structural_features_on_hand_built_star_graph():
    # Star graph: hub "H" connected to leaves "L1".."L4". Hub has max
    # betweenness/closeness/degree; leaves are leaves.
    graph = nx.star_graph(4)  # node 0 is the hub, nodes 1-4 are leaves
    graph = nx.relabel_nodes(graph, {0: "H", 1: "L1", 2: "L2", 3: "L3", 4: "L4"})
    features = centrality.compute_structural_features(graph, reservoir_nodes=(), sensor_nodes=("L1",))
    assert features["H"].degree == 4
    assert features["L1"].degree == 1
    assert features["L1"].is_leaf is True
    assert features["H"].is_leaf is False
    assert features["H"].betweenness_centrality > features["L2"].betweenness_centrality
    assert features["L1"].sensor_distance_hops == 0
    assert features["L2"].sensor_distance_hops == 2  # L2 -> H -> L1


# ---------------------------------------------------------------------------
# Statistics utilities
# ---------------------------------------------------------------------------


def test_unpaired_bootstrap_diff_sign_and_determinism():
    group_a = [1.0] * 20
    group_b = [0.0] * 20
    result = stats_utils.unpaired_bootstrap_diff(group_a, group_b)
    assert result["observed_mean_diff"] == pytest.approx(1.0)
    assert result["ci_entirely_positive"] is True
    result_again = stats_utils.unpaired_bootstrap_diff(group_a, group_b)
    assert result == result_again  # deterministic given the fixed default seed


def test_unpaired_bootstrap_diff_empty_group_returns_none_not_crash():
    result = stats_utils.unpaired_bootstrap_diff([], [1.0, 2.0])
    assert result["observed_mean_diff"] is None


def test_tercile_labels_balanced():
    values = list(range(9))
    labels = stats_utils.tercile_labels(values)
    assert labels.count("T1") == 3
    assert labels.count("T2") == 3
    assert labels.count("T3") == 3
    assert labels[0] == "T1"
    assert labels[-1] == "T3"


# ---------------------------------------------------------------------------
# End-to-end reproducibility against a real locked scenario (real simulator)
# ---------------------------------------------------------------------------


@pytest.mark.real_simulation
def test_reconstruct_incident_is_deterministic_and_matches_recorded_network_hash():
    row = json.loads(common.LOCKED_FINAL_TEST.read_text().splitlines()[0])
    incident_a = common.reconstruct_incident(row)
    incident_b = common.reconstruct_incident(row)
    assert incident_a.sensor_nodes == incident_b.sensor_nodes
    assert incident_a.start_minute == incident_b.start_minute
    assert incident_a.duration_minutes == incident_b.duration_minutes
    assert incident_a.relative_strength == incident_b.relative_strength
    result_a = common.simulate_candidate(incident_a, row["source_node"])
    result_b = common.simulate_candidate(incident_b, row["source_node"])
    frame_a = result_a.concentration_mg_l.loc[:, list(incident_a.sensor_nodes)].to_numpy()
    frame_b = result_b.concentration_mg_l.loc[:, list(incident_b.sensor_nodes)].to_numpy()
    np.testing.assert_array_equal(frame_a, frame_b)


@pytest.mark.real_simulation
def test_reconstruct_incident_rejects_network_hash_mismatch():
    row = json.loads(common.LOCKED_FINAL_TEST.read_text().splitlines()[0])
    tampered = dict(row, network_sha256="0" * 64)
    with pytest.raises(ValueError):
        common.load_base_network(tampered)
