"""Evaluate trained neural, classical-signature, and hybrid source inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import psutil
from safetensors.torch import load_file
import torch
from torch.utils.data import DataLoader

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator
from hydroswarm.model import HydroCore, HydroMono
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.training import GovernedScenarioDataset, collate_scenarios

from train import load_dataset


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_accuracy(
    predictions: np.ndarray, labels: np.ndarray, *, seed: int = 2026, samples: int = 2_000
) -> dict[str, float]:
    correct = (predictions == labels).astype(float)
    rng = np.random.default_rng(seed)
    values = np.asarray([
        correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(samples)
    ])
    return {
        "estimate": float(correct.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def _classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    ranking = np.argsort(-probabilities, axis=1)
    reciprocal = []
    for index, label in enumerate(labels):
        rank = int(np.flatnonzero(ranking[index] == label)[0]) + 1
        reciprocal.append(1.0 / rank)
    return {
        "top1": _bootstrap_accuracy(ranking[:, 0], labels),
        "top3_accuracy": float(np.mean([label in row[:3] for label, row in zip(labels, ranking, strict=True)])),
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def _fuse(classical: np.ndarray, neural: np.ndarray, neural_weight: float = 0.6) -> np.ndarray:
    log_probability = (
        (1.0 - neural_weight) * np.log(np.clip(classical, 1e-8, 1.0))
        + neural_weight * np.log(np.clip(neural, 1e-8, 1.0))
    )
    log_probability -= log_probability.max(axis=1, keepdims=True)
    values = np.exp(log_probability)
    return values / values.sum(axis=1, keepdims=True)


@torch.no_grad()
def _predict(
    model: HydroCore,
    dataset: GovernedScenarioDataset,
    *,
    batch_size: int = 16,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "source": [], "start": [], "duration": [], "strength": [], "fault": [], "classical": []
    }
    labels: dict[str, list[np.ndarray]] = {
        "source": [], "start": [], "duration": [], "strength": [], "fault": []
    }
    process = psutil.Process()
    rss_start = process.memory_info().rss
    latencies = []
    for inputs, targets in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_scenarios
    ):
        started = time.perf_counter()
        output = model({key: value.float() if value.is_floating_point() else value for key, value in inputs.items()})
        latencies.append(time.perf_counter() - started)
        collected["source"].append(torch.softmax(output["source_node_logits"], -1).cpu().numpy())
        collected["start"].append(torch.softmax(output["start_time_logits"], -1).cpu().numpy())
        collected["duration"].append(torch.softmax(output["duration_logits"], -1).cpu().numpy())
        collected["strength"].append(torch.softmax(output["relative_strength_logits"], -1).cpu().numpy())
        collected["fault"].append(torch.sigmoid(output["sensor_fault_logits"]).cpu().numpy())
        collected["classical"].append(inputs["classical_prior"].cpu().numpy())
        target_names = {
            "source": "source_node",
            "start": "start_time",
            "duration": "duration",
            "strength": "relative_strength",
            "fault": "sensor_fault",
        }
        for key, target_name in target_names.items():
            labels[key].append(targets[target_name].cpu().numpy())
    result = {key: np.concatenate(value) for key, value in collected.items()}
    result.update({f"label_{key}": np.concatenate(value) for key, value in labels.items()})
    timings = {
        "batches": float(len(latencies)),
        "mean_batch_latency_ms": statistics.fmean(latencies) * 1000.0,
        "p95_batch_latency_ms": float(np.quantile(latencies, 0.95)) * 1000.0,
        "mean_scenario_latency_ms": sum(latencies) * 1000.0 / len(dataset),
        "rss_start_bytes": float(rss_start),
        "rss_end_bytes": float(process.memory_info().rss),
        "rss_delta_bytes": float(process.memory_info().rss - rss_start),
    }
    return result, timings


def _model_report(predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    source_labels = predictions["label_source"].astype(int)
    report = {
        "source_localization": _classification_metrics(predictions["source"], source_labels),
        "start_time_accuracy": float(
            (predictions["start"].argmax(1) == predictions["label_start"]).mean()
        ),
        "duration_accuracy": float(
            (predictions["duration"].argmax(1) == predictions["label_duration"]).mean()
        ),
        "strength_accuracy": float(
            (predictions["strength"].argmax(1) == predictions["label_strength"]).mean()
        ),
        "sensor_fault_element_accuracy": float(
            ((predictions["fault"] >= 0.5) == predictions["label_fault"]).mean()
        ),
    }
    return report


def _load_model(architecture: str, checkpoint: Path, *, variant: str = "small") -> HydroCore:
    model: HydroCore = HydroMono.from_variant(variant) if architecture == "hydromono" else HydroCore.from_variant(variant)
    model.load_state_dict(load_file(checkpoint / "model.safetensors", device="cpu"), strict=True)
    return model


def _calibration_examples(
    dataset: GovernedScenarioDataset, probabilities: np.ndarray
) -> list[CalibrationExample]:
    return [
        CalibrationExample(
            probabilities=tuple(map(float, probabilities[index])),
            true_index=int(example.targets["source_node"]),
            condition=example.stage.name,
            network_id=example.network_id,
        )
        for index, example in enumerate(dataset)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tensor-directory", default="tensors")
    parser.add_argument("--hydrocore-checkpoint", type=Path, required=True)
    parser.add_argument("--hydromono-checkpoint", type=Path)
    parser.add_argument("--hydrocore-medium-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/results/learning-evaluation.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    datasets = {
        split: load_dataset(
            args.corpus / args.tensor_directory / f"{split}.jsonl", split=split
        )
        for split in ("validation", "calibration", "test")
    }
    model = _load_model("hydrocore", args.hydrocore_checkpoint)
    predictions = {}
    timings = {}
    for split, dataset in datasets.items():
        predictions[split], timings[split] = _predict(model, dataset)
    classical_test = predictions["test"]["classical"]
    neural_test = predictions["test"]["source"]
    labels = predictions["test"]["label_source"].astype(int)
    hybrid_test = _fuse(classical_test, neural_test)
    checkpoint_path = args.hydrocore_checkpoint / "model.safetensors"
    checkpoint_hash = _hash(checkpoint_path)
    calibration_manifest_hash = _hash(
        args.corpus / args.tensor_directory / "calibration.jsonl"
    )
    calibrator = SplitConformalCalibrator.fit(
        _calibration_examples(datasets["calibration"], predictions["calibration"]["source"]),
        alpha=0.1,
        model_hash=checkpoint_hash,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        dataset_manifest_hash=calibration_manifest_hash,
        minimum_group_size=10,
    )
    calibration_path = args.output.parent / "hydrocore-s-calibration.json"
    calibrator.save(calibration_path)
    heldout_calibration = calibrator.evaluate(_calibration_examples(datasets["test"], neural_test))
    fresh_model = _load_model("hydrocore", args.hydrocore_checkpoint)
    clean_load_exact = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), fresh_model.state_dict().values(), strict=True)
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "corpus_report_sha256": _hash(args.corpus / "dataset-report.json"),
        "hydrocore_s_checkpoint_sha256": checkpoint_hash,
        "clean_checkpoint_reload_exact": clean_load_exact,
        "validation": _model_report(predictions["validation"]),
        "held_out_hydraulic_family": {
            "classical_signature": _classification_metrics(classical_test, labels),
            "hydrocore_s_neural": _model_report(predictions["test"]),
            "hybrid": _classification_metrics(hybrid_test, labels),
            "timing_and_rss": timings["test"],
            "neural_contribution_mean_l1": float(np.abs(hybrid_test - classical_test).sum(axis=1).mean()),
        },
        "calibration": {
            "artifact": str(calibration_path),
            "artifact_sha256": calibrator.artifact.artifact_hash,
            "fitted_report": asdict(calibrator.artifact.report),
            "held_out_report": asdict(heldout_calibration),
        },
    }
    classical_accuracy = report["held_out_hydraulic_family"]["classical_signature"]["top1"]["estimate"]
    hybrid_accuracy = report["held_out_hydraulic_family"]["hybrid"]["top1"]["estimate"]
    report["central_claim"] = {
        "hybrid_improves_over_classical": hybrid_accuracy > classical_accuracy,
        "absolute_top1_change": hybrid_accuracy - classical_accuracy,
        "statement": (
            "Hybrid improved held-out hydraulic-family top-1 accuracy over the classical signature baseline."
            if hybrid_accuracy > classical_accuracy
            else "This run did not establish a held-out hydraulic-family top-1 improvement over the classical signature baseline."
        ),
    }
    if args.hydromono_checkpoint:
        mono = _load_model("hydromono", args.hydromono_checkpoint)
        mono_predictions, mono_timing = _predict(mono, datasets["test"])
        report["held_out_hydraulic_family"]["hydromono"] = _model_report(mono_predictions)
        report["held_out_hydraulic_family"]["hydromono_timing_and_rss"] = mono_timing
        report["hydromono_checkpoint_sha256"] = _hash(args.hydromono_checkpoint / "model.safetensors")
    if args.hydrocore_medium_checkpoint:
        medium = _load_model("hydrocore", args.hydrocore_medium_checkpoint, variant="medium")
        medium_predictions, medium_timing = _predict(medium, datasets["test"], batch_size=8)
        report["held_out_hydraulic_family"]["hydrocore_m_partial"] = _model_report(
            medium_predictions
        )
        report["held_out_hydraulic_family"]["hydrocore_m_partial_timing_and_rss"] = (
            medium_timing
        )
        report["hydrocore_m_partial_checkpoint_sha256"] = _hash(
            args.hydrocore_medium_checkpoint / "model.safetensors"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
