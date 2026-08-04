from __future__ import annotations

import torch

from hydroswarm.training import audit_corpus, audit_split, cross_split_leakage
from hydroswarm.training.data import CurriculumStage, ScenarioExample


def _example(
    scenario_id: str,
    *,
    network_id: str = "reference-a",
    split: str = "train",
    seed_family: str | None = None,
    source_node: int = 0,
    candidate_count: int = 4,
    classical_prior: list[float] | None = None,
    sensor_fault: list[float] | None = None,
    duration: int = 1,
) -> ScenarioExample:
    mask = torch.ones(candidate_count)
    prior = torch.tensor(classical_prior) if classical_prior is not None else torch.full((candidate_count,), 1 / candidate_count)
    fault = torch.tensor(sensor_fault) if sensor_fault is not None else torch.zeros(5)
    return ScenarioExample(
        scenario_id=scenario_id,
        network_id=network_id,
        split=split,
        seed=1,
        seed_family=seed_family or f"family-{scenario_id}",
        stage=CurriculumStage.CLEAN,
        inputs={
            "node_features": torch.arange(candidate_count * 2, dtype=torch.float32).reshape(candidate_count, 2),
            "source_candidate_mask": mask,
            "classical_prior": prior,
        },
        targets={
            "source_node": torch.tensor(source_node),
            "duration": torch.tensor(duration),
            "relative_strength": torch.tensor(0),
            "start_time": torch.tensor(0),
            "sensor_fault": fault,
        },
    )


def test_audit_split_reports_counts_and_balance() -> None:
    examples = [
        _example("a", network_id="net1", source_node=0),
        _example("b", network_id="net1", source_node=1),
        _example("c", network_id="net2", source_node=0),
    ]
    report = audit_split("train", examples, compute_baselines=True)
    assert report["count"] == 3
    assert report["network_balance"] == {"net1": 2, "net2": 1}
    assert report["source_balance_by_network"]["net1"] == {"0": 1, "1": 1}
    assert report["target_class_histograms"]["source_node"] == {"0": 2, "1": 1}


def test_duplicate_scenario_ids_detected() -> None:
    examples = [_example("dup", seed_family="f1"), _example("dup", seed_family="f2")]
    report = audit_split("train", examples, compute_baselines=False)
    assert report["duplicate_scenario_ids"] == ["dup"]


def test_near_duplicate_scenarios_flagged_when_state_and_targets_match_exactly() -> None:
    examples = [
        _example("x", seed_family="fx", source_node=0),
        _example("y", seed_family="fy", source_node=0),  # identical node_features + targets
        _example("z", seed_family="fz", source_node=1),  # different target -> different hash
    ]
    report = audit_split("train", examples, compute_baselines=False)
    assert report["near_duplicate_groups"] == [["x", "y"]]


def test_impossible_labels_flags_out_of_range_and_masked_source_and_bad_sensor_fault() -> None:
    out_of_range = _example("oor", source_node=99, candidate_count=4)
    bad_fault = _example("badfault", sensor_fault=[0.0, 1.5, 0.0, 0.0, 0.0])
    ok = _example("ok")
    report = audit_split("train", [out_of_range, bad_fault, ok], compute_baselines=False)
    reasons = {violation["scenario_id"]: violation["reason"] for violation in report["impossible_labels"]}
    assert reasons["oor"] == "source_node index out of range"
    assert "sensor_fault" in reasons["badfault"]
    assert "ok" not in reasons


def test_impossible_labels_flags_masked_out_source_candidate() -> None:
    example = _example("masked", source_node=1, candidate_count=3)
    example.inputs["source_candidate_mask"][1] = 0.0
    report = audit_split("train", [example], compute_baselines=False)
    assert any(v["scenario_id"] == "masked" for v in report["impossible_labels"])


def test_finite_value_violations_detect_nan() -> None:
    example = _example("hasnan")
    example.inputs["classical_prior"] = torch.tensor([float("nan"), 0.5, 0.25, 0.25])
    report = audit_split("train", [example], compute_baselines=False)
    assert any(
        v["scenario_id"] == "hasnan" and v["tensor"] == "classical_prior"
        for v in report["finite_value_violations"]
    )


def test_sanity_baselines_random_majority_and_classical_prior() -> None:
    # 4 candidates each -> random baseline should be 0.25 for every example.
    examples = [
        _example("a", source_node=0, classical_prior=[0.7, 0.1, 0.1, 0.1]),  # prior correct
        _example("b", source_node=0, classical_prior=[0.1, 0.7, 0.1, 0.1]),  # prior wrong
        _example("c", source_node=0, classical_prior=[0.1, 0.1, 0.1, 0.7]),  # prior wrong
    ]
    report = audit_split("train", examples, compute_baselines=True)
    baselines = report["sanity_baselines"]
    assert baselines["random_source"] == 0.25
    assert baselines["majority_source"]["accuracy"] == 1.0  # source_node is always 0
    assert baselines["majority_source"]["majority_class"] == 0
    assert abs(baselines["classical_signature_prior"] - (1 / 3)) < 1e-9
    assert baselines["nearest_positive_sensor"]["value"] is None
    assert baselines["classical_eig_sample"]["value"] is None
    assert baselines["deterministic_plan_template_ranking"]["value"] is None


def test_baselines_are_omitted_outside_decision_splits() -> None:
    examples = [_example("a"), _example("b")]
    corpus = audit_corpus({"train": examples, "test": examples})
    assert "sanity_baselines" not in corpus["splits"]["test"] or "accuracy" not in str(
        corpus["splits"]["test"]["sanity_baselines"]
    )
    assert "note" in corpus["splits"]["test"]["sanity_baselines"]
    assert "random_source" in corpus["splits"]["train"]["sanity_baselines"]


def test_cross_split_leakage_detects_shared_scenario_id_across_splits() -> None:
    shared = _example("shared-id", split="train")
    shared_in_test = _example("shared-id", split="test")
    report = cross_split_leakage({"train": [shared], "test": [shared_in_test]})
    assert len(report["scenario_id_leaks"]) == 1
    assert report["scenario_id_leaks"][0]["scenario_id"] == "shared-id"
    assert sorted(report["scenario_id_leaks"][0]["splits"]) == ["test", "train"]


def test_cross_split_leakage_detects_shared_seed_family() -> None:
    train_example = _example("train-1", network_id="net1", seed_family="shared-family")
    test_example = _example("test-1", network_id="net1", seed_family="shared-family")
    report = cross_split_leakage({"train": [train_example], "test": [test_example]})
    assert len(report["seed_family_leaks"]) == 1


def test_cross_split_leakage_clean_when_no_overlap() -> None:
    report = cross_split_leakage(
        {
            "train": [_example("t1", seed_family="fam-t1")],
            "test": [_example("v1", seed_family="fam-v1")],
        }
    )
    assert report["scenario_id_leaks"] == []
    assert report["seed_family_leaks"] == []


def test_audit_corpus_assembles_all_splits_and_leakage() -> None:
    corpus = audit_corpus(
        {
            "train": [_example("a", seed_family="fa")],
            "validation": [_example("b", seed_family="fb")],
            "calibration": [_example("c", seed_family="fc")],
            "test": [_example("d", seed_family="fd")],
        }
    )
    assert set(corpus["splits"]) == {"train", "validation", "calibration", "test"}
    assert corpus["decision_splits"] == ["train", "validation", "calibration"]
    assert "cross_split_leakage" in corpus
