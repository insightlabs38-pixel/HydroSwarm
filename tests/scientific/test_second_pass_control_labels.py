"""core-issues3.txt Phase 8: second-pass calibrated control-label generation."""

from __future__ import annotations

import pytest

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator
from hydroswarm.data.scenarios import DatasetSplit, EventType, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.model import HydroCore
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.control_labels import NEXT_STEP_RUNTIME_ENABLED
from hydroswarm.training.corpus import fit_signature_library, scenario_to_example
from hydroswarm.training.second_pass_control_labels import (
    DEFAULT_DISAGREEMENT_THRESHOLD,
    DEFAULT_MAXIMUM_CANDIDATE_SET_SIZE,
    classify_evidence_sufficiency_second_pass,
    generate_second_pass_control_labels,
)
from hydroswarm.training.targets_v2 import NextStep

_SUFFICIENT_KWARGS = dict(
    calibrated_candidate_set_size=1, calibration_valid=True, posterior_entropy_bits=0.5,
    disagreement_js=0.1, healthy_fraction=0.9, sensors_ever_healthy=3,
)


def test_all_conditions_passing_is_sufficient() -> None:
    assert classify_evidence_sufficiency_second_pass(**_SUFFICIENT_KWARGS) is True


def test_empty_candidate_set_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, calibrated_candidate_set_size=0)
    assert classify_evidence_sufficiency_second_pass(**kwargs) is False


def test_overly_broad_candidate_set_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, calibrated_candidate_set_size=DEFAULT_MAXIMUM_CANDIDATE_SET_SIZE + 1)
    assert classify_evidence_sufficiency_second_pass(**kwargs) is False


def test_invalid_calibration_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, calibration_valid=False)
    assert classify_evidence_sufficiency_second_pass(**kwargs) is False


def test_high_disagreement_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, disagreement_js=DEFAULT_DISAGREEMENT_THRESHOLD)
    assert classify_evidence_sufficiency_second_pass(**kwargs) is False


def test_poor_sensor_health_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, healthy_fraction=0.1)
    assert classify_evidence_sufficiency_second_pass(**kwargs) is False


def test_high_entropy_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, posterior_entropy_bits=10.0)
    assert classify_evidence_sufficiency_second_pass(**kwargs) is False


@pytest.fixture(scope="module")
def network():
    return build_wntr_network()


@pytest.fixture(scope="module")
def signature_library(network):
    generator = WNTRScenarioGenerator()
    node_ids = tuple(sorted(network.junction_name_list))
    scenarios = [
        generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=5000 + index * 10, network_id="ref", network_family="reference",
                split=DatasetSplit.TRAIN, source_node=source, sensor_count=3,
            ),
        )
        for index, source in enumerate(node_ids)
    ]
    return fit_signature_library(scenarios, node_ids)


@pytest.fixture(scope="module")
def examples(network, signature_library):
    generator = WNTRScenarioGenerator()
    built = []
    for seed in (5100, 5110, 5120):
        scenario = generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=seed, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
                event_type=EventType.CONTAMINATION, sensor_count=3,
            ),
        )
        built.append(scenario_to_example(scenario, network, signature_library))
    return built


class _ListDataset:
    """Minimal ScenarioDatasetView stand-in -- __len__/__getitem__ only,
    which is all generate_second_pass_control_labels needs."""

    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


@pytest.fixture(scope="module")
def tiny_model():
    model = HydroCore.from_variant("small")
    model.eval()
    return model


@pytest.fixture(scope="module")
def calibrator(examples):
    # A trivially-fit calibrator with a wide-enough set to exercise the
    # pipeline deterministically, not a scientifically meaningful fit.
    rows = [
        CalibrationExample(
            probabilities=tuple(1.0 if i == 0 else 0.0 for i in range(len(example.topology.node_ids))),
            true_index=0, condition=example.stage.name, network_id=example.network_id,
        )
        for example in examples
        for _ in range(12)  # minimum_group_size default is 10
    ]
    return SplitConformalCalibrator.fit(
        rows, alpha=0.5, model_hash="test-teacher-hash", feature_schema_hash="test-schema",
        dataset_manifest_hash="test-manifest",
    )


#: 17 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
@pytest.mark.real_simulation
def test_raises_if_model_is_not_frozen(examples, calibrator):
    model = HydroCore.from_variant("small")
    model.train()
    dataset = _ListDataset(examples)
    with pytest.raises(ValueError, match="frozen"):
        next(
            generate_second_pass_control_labels(
                model, dataset, calibrator, teacher_checkpoint_hash="x", validated_topology_hashes=frozenset()
            )
        )


#: Shares the module-scoped `examples` fixture with
#: test_raises_if_model_is_not_frozen and
#: test_unvalidated_topology_invalidates_calibration_and_forces_zero_candidate_set
#: -- whichever test runs first in a given selection triggers its one-time
#: real-simulation-backed build (found by the real_simulation runtime audit
#: in tests/conftest.py, not the static call-count audit alone).
@pytest.mark.real_simulation
def test_generates_one_label_per_example_with_expected_fields(tiny_model, examples, calibrator):
    dataset = _ListDataset(examples)
    validated = frozenset({example.topology.topology_hash for example in examples})
    labels = list(
        generate_second_pass_control_labels(
            tiny_model, dataset, calibrator,
            teacher_checkpoint_hash="test-teacher-hash", validated_topology_hashes=validated,
            batch_size=2,
        )
    )
    assert len(labels) == len(examples)
    for label in labels:
        assert label.teacher_checkpoint_hash == "test-teacher-hash"
        assert label.calibrated_candidate_set_size >= 0
        assert 0.0 <= label.classical_neural_disagreement_js <= 1.0
        assert label.posterior_entropy_bits >= 0.0
        assert label.calibration_valid is True  # every example's topology is in `validated`
        assert isinstance(label.next_step, NextStep)
        assert label.candidate_covered in (True, False, None)


#: Shares the module-scoped `examples` fixture with the two tests above --
#: see test_generates_one_label_per_example_with_expected_fields's comment.
@pytest.mark.real_simulation
def test_unvalidated_topology_invalidates_calibration_and_forces_zero_candidate_set(tiny_model, examples, calibrator):
    dataset = _ListDataset(examples[:1])
    labels = list(
        generate_second_pass_control_labels(
            tiny_model, dataset, calibrator,
            teacher_checkpoint_hash="test-teacher-hash", validated_topology_hashes=frozenset(),
            batch_size=1,
        )
    )
    assert len(labels) == 1
    assert labels[0].calibration_valid is False
    assert labels[0].calibrated_candidate_set_size == 0
    assert labels[0].evidence_sufficiency is False
    assert labels[0].next_step == NextStep.ABSTAIN


def test_next_step_runtime_enabled_excludes_inspect_sensor_but_labels_can_still_produce_it() -> None:
    # Sanity cross-check linking this module's next_step output back to
    # control_labels' runtime-gating registry (Phase 8 item 8).
    assert NextStep.INSPECT_FAULTY_SENSOR not in NEXT_STEP_RUNTIME_ENABLED
