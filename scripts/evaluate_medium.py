"""One-shot locked-test comparison for converged HydroCore-M and existing baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from hydroswarm.calibration.conformal import SplitConformalCalibrator
from hydroswarm.model import HydroCore, HydroMono
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA

from evaluate_learning import (
    _calibration_examples,
    _classification_metrics,
    _fuse,
    _hash,
    _model_report,
    _predict,
)
from train import load_dataset


def load_model(path: Path, *, architecture: str, variant: str) -> HydroCore:
    model: HydroCore = (
        HydroMono.from_variant(variant)
        if architecture == "hydromono"
        else HydroCore.from_variant(variant)
    )
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-directory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--medium-model", type=Path, required=True)
    parser.add_argument("--small-model", type=Path, required=True)
    parser.add_argument("--mono-model", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    test_path = args.tensor_directory / "test.jsonl"
    expected_test_hash = protocol["test_split_lock"]["tensor_manifest_sha256"]
    if _hash(test_path) != expected_test_hash:
        raise SystemExit("locked test tensor manifest changed; refusing evaluation")

    calibration = load_dataset(
        args.tensor_directory / "calibration.jsonl", split="calibration"
    )
    test = load_dataset(test_path, split="test")
    models = {
        "hydrocore_m": load_model(args.medium_model, architecture="hydrocore", variant="medium"),
        "hydrocore_s": load_model(args.small_model, architecture="hydrocore", variant="small"),
        "hydromono_s": load_model(args.mono_model, architecture="hydromono", variant="small"),
    }

    medium_calibration, _ = _predict(models["hydrocore_m"], calibration, batch_size=8)
    calibration_hybrid = _fuse(
        medium_calibration["classical"], medium_calibration["source"]
    )
    medium_hash = _hash(args.medium_model)
    calibration_manifest_hash = _hash(args.tensor_directory / "calibration.jsonl")
    calibrator = SplitConformalCalibrator.fit(
        _calibration_examples(calibration, calibration_hybrid),
        alpha=0.1,
        model_hash=medium_hash,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        dataset_manifest_hash=calibration_manifest_hash,
        minimum_group_size=10,
    )
    calibrator.save(args.calibration_output)

    predictions = {}
    timings = {}
    for name, model in models.items():
        predictions[name], timings[name] = _predict(
            model, test, batch_size=8 if name == "hydrocore_m" else 16
        )
    labels = predictions["hydrocore_m"]["label_source"].astype(int)
    classical = predictions["hydrocore_m"]["classical"]
    medium_hybrid = _fuse(classical, predictions["hydrocore_m"]["source"])
    small_hybrid = _fuse(classical, predictions["hydrocore_s"]["source"])
    calibration_report = calibrator.evaluate(
        _calibration_examples(test, medium_hybrid)
    )

    comparisons = {
        "classical": _classification_metrics(classical, labels),
        "hydrocore_m_neural": _model_report(predictions["hydrocore_m"]),
        "hydrocore_m_hybrid": _classification_metrics(medium_hybrid, labels),
        "hydrocore_s_hybrid": _classification_metrics(small_hybrid, labels),
        "hydromono_s": _model_report(predictions["hydromono_s"]),
    }
    medium_profile = np.mean([
        comparisons["hydrocore_m_neural"][key]
        for key in ("start_time_accuracy", "duration_accuracy", "strength_accuracy")
    ])
    small_profile_report = _model_report(predictions["hydrocore_s"])
    small_profile = np.mean([
        small_profile_report[key]
        for key in ("start_time_accuracy", "duration_accuracy", "strength_accuracy")
    ])
    medium_accuracy = comparisons["hydrocore_m_hybrid"]["top1"]["estimate"]
    small_accuracy = comparisons["hydrocore_s_hybrid"]["top1"]["estimate"]
    meaningful_gain = bool(
        medium_accuracy >= small_accuracy + 0.005
        or (medium_accuracy >= small_accuracy - 0.005 and medium_profile >= small_profile + 0.05)
    )
    parameter_counts = {name: model.parameter_count() for name, model in models.items()}
    report = {
        "schema_version": 1,
        "bootstrap_samples": 2_000,
        "test_split_sha256": expected_test_hash,
        "test_examples": len(test),
        "training_decisions_used_test": False,
        "model_sha256": {
            "hydrocore_m": medium_hash,
            "hydrocore_s": _hash(args.small_model),
            "hydromono_s": _hash(args.mono_model),
        },
        "comparisons": comparisons,
        "hydrocore_s_neural_profile": small_profile_report,
        "timing": timings,
        "parameter_counts": parameter_counts,
        "cost_adjusted": {
            name: {
                "top1_per_million_parameters": float(
                    comparisons[metric]["source_localization"]["top1"]["estimate"]
                    / (parameter_counts[name] / 1_000_000)
                ),
                "top1_per_mean_scenario_ms": float(
                    comparisons[metric]["source_localization"]["top1"]["estimate"]
                    / timings[name]["mean_scenario_latency_ms"]
                ),
            }
            for name, metric in (
                ("hydrocore_m", "hydrocore_m_neural"),
                ("hydromono_s", "hydromono_s"),
            )
        },
        "calibration": {
            "fit_split": "calibration",
            "artifact": str(args.calibration_output),
            "artifact_sha256": calibrator.artifact.artifact_hash,
            "fit_report": asdict(calibrator.artifact.report),
            "held_out_report": asdict(calibration_report),
        },
        "promotion_gate": {
            "meaningful_operational_gain": meaningful_gain,
            "medium_hybrid_top1_change_vs_small": medium_accuracy - small_accuracy,
            "medium_profile_macro_change_vs_small": float(medium_profile - small_profile),
            "recommendation": "promote_alongside_small" if meaningful_gain else "do_not_promote",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
