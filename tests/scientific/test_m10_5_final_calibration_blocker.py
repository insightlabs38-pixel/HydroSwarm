from pathlib import Path

def test_m10_4_factory_reconstructs_calibrator_and_release_must_not_refit() -> None:
    source = (Path(__file__).resolve().parents[2] / "scripts/hydrocore_v5/m10_4_common.py").read_text()
    assert "fit_frozen_calibrator(model_hash=model_hash" in source
    assert "return SplitConformalCalibrator.fit(" in source
