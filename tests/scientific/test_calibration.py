import pytest

from hydroswarm.calibration import CalibrationExample, SplitConformalCalibrator


def examples():
    return [
        CalibrationExample((0.9, 0.1), 0, "clean", "Net1"),
        CalibrationExample((0.8, 0.2), 0, "clean", "Net1"),
        CalibrationExample((0.4, 0.6), 1, "noise", "Net2"),
        CalibrationExample((0.3, 0.7), 1, "noise", "Net2"),
        CalibrationExample((0.55, 0.45), 1, "shift", "C-Town"),
    ]


def test_fit_measure_save_load_and_ood_invalidation(tmp_path) -> None:
    calibrator = SplitConformalCalibrator.fit(
        examples(), alpha=0.1, model_hash="m", feature_schema_hash="f",
        dataset_manifest_hash="d", minimum_group_size=2,
    )
    assert calibrator.artifact.report.examples == 5
    assert 0 <= calibrator.artifact.report.coverage <= 1
    assert calibrator.candidate_set((0.8, 0.2), condition="clean", network_id="Net1")
    assert calibrator.candidate_set(
        (0.8, 0.2), ood_level="OUTSIDE_VALIDATED_RANGE"
    ) == ()
    path = tmp_path / "calibration.json"
    calibrator.save(path)
    loaded = SplitConformalCalibrator.load(path)
    loaded.artifact.validate_runtime(model_hash="m", feature_schema_hash="f", dataset_manifest_hash="d")
    assert loaded.artifact.artifact_hash == calibrator.artifact.artifact_hash

    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        SplitConformalCalibrator.load(path)


def test_normalization_hash_defaults_to_the_no_normalization_sentinel() -> None:
    from hydroswarm.preprocessing import NO_NORMALIZATION_SENTINEL

    calibrator = SplitConformalCalibrator.fit(
        examples(), alpha=0.1, model_hash="m", feature_schema_hash="f",
        dataset_manifest_hash="d", minimum_group_size=2,
    )
    assert calibrator.artifact.normalization_hash == NO_NORMALIZATION_SENTINEL
    # Compatible with a runtime reporting the same "no normalization" state.
    calibrator.artifact.validate_runtime(
        model_hash="m", feature_schema_hash="f", normalization_hash=NO_NORMALIZATION_SENTINEL,
    )


def test_validate_runtime_fails_closed_on_a_real_normalization_mismatch() -> None:
    calibrator = SplitConformalCalibrator.fit(
        examples(), alpha=0.1, model_hash="m", feature_schema_hash="f",
        dataset_manifest_hash="d", minimum_group_size=2, normalization_hash="fitted-against-artifact-A",
    )
    with pytest.raises(ValueError, match="normalization"):
        calibrator.artifact.validate_runtime(
            model_hash="m", feature_schema_hash="f", normalization_hash="fitted-against-artifact-B",
        )
    # A caller that does not pass normalization_hash at all is unaffected
    # (backward compatible with every existing call site).
    calibrator.artifact.validate_runtime(model_hash="m", feature_schema_hash="f")


def test_fusion_config_hash_defaults_to_none_and_skips_the_check() -> None:
    calibrator = SplitConformalCalibrator.fit(
        examples(), alpha=0.1, model_hash="m", feature_schema_hash="f",
        dataset_manifest_hash="d", minimum_group_size=2,
    )
    assert calibrator.artifact.fusion_config_hash is None
    # A caller not passing fusion_config_hash at all is unaffected -- this is
    # what every existing pipeline/test fixture predating repair item 10 does.
    calibrator.artifact.validate_runtime(model_hash="m", feature_schema_hash="f")


def test_validate_runtime_fails_closed_on_a_fusion_config_mismatch() -> None:
    calibrator = SplitConformalCalibrator.fit(
        examples(), alpha=0.1, model_hash="m", feature_schema_hash="f",
        dataset_manifest_hash="d", minimum_group_size=2,
        fusion_config_hash="fixed_weight_fusion-v1:neural_weight=0.6",
    )
    calibrator.artifact.validate_runtime(
        model_hash="m", feature_schema_hash="f",
        fusion_config_hash="fixed_weight_fusion-v1:neural_weight=0.6",
    )
    with pytest.raises(ValueError, match="fusion"):
        calibrator.artifact.validate_runtime(
            model_hash="m", feature_schema_hash="f",
            fusion_config_hash="fuse_source_probabilities-v1",
        )


def test_validated_topology_hashes_defaults_to_empty_and_skips_the_check() -> None:
    calibrator = SplitConformalCalibrator.fit(
        examples(), alpha=0.1, model_hash="m", feature_schema_hash="f",
        dataset_manifest_hash="d", minimum_group_size=2,
    )
    assert calibrator.artifact.validated_topology_hashes == ()
    # An empty set means "not declared" -- every topology is accepted,
    # never treated as if every topology were unknown.
    calibrator.artifact.validate_runtime(
        model_hash="m", feature_schema_hash="f", topology_hash="some-unseen-topology"
    )


def test_validate_runtime_fails_closed_on_an_unknown_topology() -> None:
    calibrator = SplitConformalCalibrator.fit(
        examples(), alpha=0.1, model_hash="m", feature_schema_hash="f",
        dataset_manifest_hash="d", minimum_group_size=2,
        topology_hashes=["net-1", "net-2", "net-1"],
    )
    assert calibrator.artifact.validated_topology_hashes == ("net-1", "net-2")
    calibrator.artifact.validate_runtime(model_hash="m", feature_schema_hash="f", topology_hash="net-1")
    with pytest.raises(ValueError, match="topology"):
        calibrator.artifact.validate_runtime(
            model_hash="m", feature_schema_hash="f", topology_hash="never-seen-topology"
        )
