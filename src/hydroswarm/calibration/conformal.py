"""Held-out split/Mondrian conformal calibration with artifact validity checks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hydroswarm.preprocessing.builder import NO_NORMALIZATION_SENTINEL


CALIBRATION_SCHEMA_VERSION = "hydroswarm-calibration-v1"


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    probabilities: tuple[float, ...]
    true_index: int
    condition: str
    network_id: str


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    coverage: float
    mean_set_size: float
    expected_calibration_error: float
    coverage_by_condition: Mapping[str, float]
    coverage_by_network: Mapping[str, float]
    examples: int


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    schema_version: str
    alpha: float
    model_hash: str
    feature_schema_hash: str
    dataset_manifest_hash: str
    global_scores: tuple[float, ...]
    mondrian_scores: Mapping[str, tuple[float, ...]]
    network_scores: Mapping[str, tuple[float, ...]]
    report: CalibrationReport
    #: core-issues.txt repair item 6: identity of the governed node/edge
    #: NormalizationStats (HydraulicFeatureBuilder.normalization_fingerprint)
    #: this calibration was fit against. Defaults to the explicit "no
    #: governed normalization" sentinel rather than an empty string, so
    #: every existing artifact (fit before this field existed) is
    #: unambiguously "fit with no normalization layer active" -- the
    #: literal truth for all of them -- rather than silently unset.
    normalization_hash: str = NO_NORMALIZATION_SENTINEL
    #: core-issues.txt repair item 10: identifies which fusion algorithm
    #: produced the probability vectors this calibration was fit against
    #: (see hydroswarm.inference.fusion's DYNAMIC_TRUST_FUSION_CONFIG /
    #: fixed_weight_fusion_config). None means "not declared" -- every
    #: artifact fit before this field existed was in fact fit on raw
    #: neural probabilities with no fusion step at all, which is not the
    #: same thing as either named config, so it is left unset rather than
    #: mislabeled as one of them.
    fusion_config_hash: str | None = None
    #: core-issues.txt repair item 10: the structural topology hashes
    #: (hydroswarm.data.scenarios.network_sha256; stable per named
    #: topology family, unlike the per-scenario randomized-hydraulics
    #: network_hash) actually present in the data this calibration was fit
    #: against. Empty means "not declared" -- validate_runtime skips the
    #: check entirely rather than treating an empty set as "every topology
    #: is unknown".
    validated_topology_hashes: tuple[str, ...] = ()

    @property
    def artifact_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def validate_runtime(
        self,
        *,
        model_hash: str,
        feature_schema_hash: str,
        dataset_manifest_hash: str | None = None,
        normalization_hash: str | None = None,
        fusion_config_hash: str | None = None,
        topology_hash: str | None = None,
    ) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported calibration artifact schema")
        if self.model_hash != model_hash or self.feature_schema_hash != feature_schema_hash:
            raise ValueError("calibration artifact is incompatible with model/features")
        if dataset_manifest_hash is not None and self.dataset_manifest_hash != dataset_manifest_hash:
            raise ValueError("calibration dataset manifest hash does not match")
        if normalization_hash is not None and self.normalization_hash != normalization_hash:
            raise ValueError(
                "calibration artifact was fit against a different governed normalization "
                "than the runtime feature builder is currently using"
            )
        if fusion_config_hash is not None and self.fusion_config_hash != fusion_config_hash:
            raise ValueError(
                "calibration artifact was fit against a different fusion configuration "
                "than the one currently producing fused probabilities"
            )
        if (
            topology_hash is not None
            and self.validated_topology_hashes
            and topology_hash not in self.validated_topology_hashes
        ):
            raise ValueError(
                "calibration artifact was never fit against this network's topology; "
                "an unknown topology must invalidate calibration"
            )


def _normalize(probabilities: Sequence[float]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a finite non-negative vector")
    if values.sum() <= 0:
        raise ValueError("probability vector has no mass")
    return values / values.sum()


def _quantile(scores: Sequence[float], alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("conformal scores are empty")
    rank = min(values.size, max(1, int(math.ceil((values.size + 1) * (1 - alpha)))))
    return float(np.partition(values, rank - 1)[rank - 1])


def expected_calibration_error(
    confidences: Sequence[float], correctness: Sequence[bool], *, bins: int = 10
) -> float:
    confidence = np.asarray(confidences, dtype=float)
    correct = np.asarray(correctness, dtype=float)
    if confidence.shape != correct.shape or confidence.ndim != 1 or bins <= 0:
        raise ValueError("confidence/correctness must be aligned vectors and bins positive")
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if selected.any():
            error += float(selected.mean()) * abs(float(correct[selected].mean() - confidence[selected].mean()))
    return error


class SplitConformalCalibrator:
    def __init__(self, artifact: CalibrationArtifact) -> None:
        self.artifact = artifact

    @classmethod
    def fit(
        cls,
        examples: Sequence[CalibrationExample],
        *,
        alpha: float = 0.1,
        model_hash: str,
        feature_schema_hash: str,
        dataset_manifest_hash: str,
        minimum_group_size: int = 10,
        normalization_hash: str = NO_NORMALIZATION_SENTINEL,
        fusion_config_hash: str | None = None,
        topology_hashes: Sequence[str] = (),
    ) -> SplitConformalCalibrator:
        if not 0 < alpha < 1 or not examples:
            raise ValueError("alpha must be within (0,1) and examples non-empty")
        scores: list[float] = []
        conditions: dict[str, list[float]] = {}
        networks: dict[str, list[float]] = {}
        confidence: list[float] = []
        correct: list[bool] = []
        for item in examples:
            probabilities = _normalize(item.probabilities)
            if not 0 <= item.true_index < probabilities.size:
                raise ValueError("calibration true index is outside probability vector")
            score = 1.0 - float(probabilities[item.true_index])
            scores.append(score)
            conditions.setdefault(item.condition, []).append(score)
            networks.setdefault(item.network_id, []).append(score)
            confidence.append(float(probabilities.max()))
            correct.append(int(probabilities.argmax()) == item.true_index)
        usable_conditions = {key: tuple(value) for key, value in conditions.items() if len(value) >= minimum_group_size}
        usable_networks = {key: tuple(value) for key, value in networks.items() if len(value) >= minimum_group_size}
        sorted_topology_hashes = tuple(sorted(set(topology_hashes)))
        temporary = CalibrationArtifact(
            CALIBRATION_SCHEMA_VERSION, alpha, model_hash, feature_schema_hash, dataset_manifest_hash,
            tuple(scores), usable_conditions, usable_networks,
            CalibrationReport(0.0, 0.0, 0.0, {}, {}, len(examples)),
            normalization_hash, fusion_config_hash, sorted_topology_hashes,
        )
        calibrator = cls(temporary)
        report = calibrator.evaluate(examples)
        return cls(CalibrationArtifact(
            CALIBRATION_SCHEMA_VERSION, alpha, model_hash, feature_schema_hash, dataset_manifest_hash,
            tuple(scores), usable_conditions, usable_networks, report,
            normalization_hash, fusion_config_hash, sorted_topology_hashes,
        ))

    def candidate_set(
        self,
        probabilities: Sequence[float],
        *,
        condition: str | None = None,
        network_id: str | None = None,
        ood_level: str = "NORMAL",
    ) -> tuple[int, ...]:
        if ood_level == "OUTSIDE_VALIDATED_RANGE":
            return ()
        values = _normalize(probabilities)
        scores = self.artifact.global_scores
        if network_id in self.artifact.network_scores:
            scores = self.artifact.network_scores[network_id]
        elif condition in self.artifact.mondrian_scores:
            scores = self.artifact.mondrian_scores[condition]
        threshold = _quantile(scores, self.artifact.alpha)
        return tuple(int(index) for index in np.flatnonzero(1.0 - values <= threshold))

    def evaluate(self, examples: Sequence[CalibrationExample]) -> CalibrationReport:
        covered: list[bool] = []
        sizes: list[int] = []
        condition_values: dict[str, list[bool]] = {}
        network_values: dict[str, list[bool]] = {}
        confidences: list[float] = []
        correctness: list[bool] = []
        for item in examples:
            selected = self.candidate_set(
                item.probabilities, condition=item.condition, network_id=item.network_id
            )
            hit = item.true_index in selected
            covered.append(hit)
            sizes.append(len(selected))
            condition_values.setdefault(item.condition, []).append(hit)
            network_values.setdefault(item.network_id, []).append(hit)
            values = _normalize(item.probabilities)
            confidences.append(float(values.max()))
            correctness.append(int(values.argmax()) == item.true_index)
        return CalibrationReport(
            coverage=float(np.mean(covered)), mean_set_size=float(np.mean(sizes)),
            expected_calibration_error=expected_calibration_error(confidences, correctness),
            coverage_by_condition={key: float(np.mean(value)) for key, value in condition_values.items()},
            coverage_by_network={key: float(np.mean(value)) for key, value in network_values.items()},
            examples=len(examples),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self.artifact), indent=2, sort_keys=True)
        target.write_text(payload, encoding="utf-8")
        target.with_suffix(target.suffix + ".sha256").write_text(
            hashlib.sha256(payload.encode()).hexdigest(), encoding="ascii"
        )

    @classmethod
    def load(cls, path: str | Path) -> SplitConformalCalibrator:
        target = Path(path)
        payload = target.read_text(encoding="utf-8")
        expected = target.with_suffix(target.suffix + ".sha256").read_text(encoding="ascii").strip()
        if hashlib.sha256(payload.encode()).hexdigest() != expected:
            raise ValueError("calibration artifact checksum mismatch")
        data = json.loads(payload)
        data["global_scores"] = tuple(data["global_scores"])
        data["mondrian_scores"] = {key: tuple(value) for key, value in data["mondrian_scores"].items()}
        data["network_scores"] = {key: tuple(value) for key, value in data["network_scores"].items()}
        if "validated_topology_hashes" in data:
            data["validated_topology_hashes"] = tuple(data["validated_topology_hashes"])
        data["report"] = CalibrationReport(**data["report"])
        return cls(CalibrationArtifact(**data))

