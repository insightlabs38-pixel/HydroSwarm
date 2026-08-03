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
