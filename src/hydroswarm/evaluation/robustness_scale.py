"""Frozen, locked-test-excluding robustness/scale tensor characterization.

This module deliberately evaluates only the governed validation and
development-holdout tensor populations named in the committed protocol.  It
does not know a path to the locked split, and rejects one if supplied through
the protocol.  It is an offline characterization of the shipping weights;
it does not mutate model, calibration, corpus, or runtime policy artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch

from hydroswarm.calibration import SplitConformalCalibrator
from hydroswarm.inference.fusion import ControlAction, jensen_shannon_divergence, uncertainty_control
from hydroswarm.runtime.v4_inference_bundle import load_v4_inference_bundle
from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology
from hydroswarm.training.ood_categories import OODCategory, OOD_CATEGORY_BEHAVIOR


REQUIRED_ROW_FIELDS = (
    "run_id", "git_commit", "model_sha256", "calibration_sha256", "feature_schema_sha256",
    "network_id", "network_sha256", "topology_class", "random_seed", "source_node",
    "sensor_count", "observation_count", "perturbation_type", "perturbation_level",
    "top1_correct", "top3_correct", "reciprocal_rank", "true_source_probability",
    "candidate_set_size", "candidate_contains_truth", "posterior_entropy", "calibrated",
    "ood_level", "disagreement_js", "evidence_sufficient", "planning_allowed", "control_action",
    "suppression_reasons", "samples_requested", "first_recommended_node",
    "expected_information_gain", "inference_ms", "analysis_ms", "sampling_ms", "planning_ms",
    "verification_ms", "total_workflow_ms", "process_rss_mb", "exact_simulator_calls", "outcome", "error_class",
    "runtime_mode",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def locked_test_status(repo_root: Path) -> bool:
    """Read the immutable freeze declaration without reading any split data."""
    return bool(_read_json(repo_root / "reports/results/v4/architecture-freeze.json")["locked_test_opened"])


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if protocol.get("locked_test", {}).get("excluded") is not True:
        raise ValueError("protocol must explicitly exclude locked evaluation")
    forbidden = set(protocol["locked_test"].get("forbidden_split_names", ()))
    populations = tuple(protocol.get("populations", ()))
    if not populations or any(name in forbidden or "locked" in name.lower() or name == "test" for name in populations):
        raise ValueError("protocol attempts to consume a locked-test population")
    return protocol


def deterministic_indices(dataset: ShardedScenarioDataset, *, population: str, limit: int, seed: int) -> list[int]:
    """Stable content-addressed stratification, independent of filesystem order."""
    keyed = []
    for index, entry in enumerate(dataset._entries):  # governed index metadata, not tensor internals
        value = hashlib.sha256(f"{seed}:{population}:{entry.scenario_id}".encode()).hexdigest()
        keyed.append((value, index))
    return [index for _key, index in sorted(keyed)[: min(limit, len(keyed))]]


def _entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return float(-(safe * np.log2(safe)).sum())


def _softmax(values: torch.Tensor) -> np.ndarray:
    return torch.softmax(values.float(), dim=-1).detach().cpu().numpy().astype(float)


def _condition(population: str) -> tuple[str, str, OODCategory]:
    if population == "validation":
        return "baseline", "nominal", OODCategory.NONE
    category = OODCategory[population.removeprefix("ood-")]
    return category.value.lower(), category.value, category


def _topology_class(category: OODCategory) -> str:
    return "unseen" if category is OODCategory.UNSEEN_TOPOLOGY else "governed_in_distribution"


def _candidate_probabilities(neural: np.ndarray, classical: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Offline tensor replay uses a transparent fixed blend only for metrics.

    Live dynamic trust requires raw hydraulic estimator residuals unavailable in
    sharded tensors. Calibration applicability/authority is still derived from
    the current deterministic policy, not from this proxy.
    """
    neural = neural.copy()
    classical = classical.copy()
    neural[~mask] = 0.0
    classical[~mask] = 0.0
    neural /= neural.sum() or 1.0
    classical /= classical.sum() or 1.0
    js = jensen_shannon_divergence(neural[mask], classical[mask])
    blend = np.sqrt(np.clip(neural, 1e-12, 1.0) * np.clip(classical, 1e-12, 1.0))
    blend[~mask] = 0.0
    blend /= blend.sum() or 1.0
    return blend, js


def _row(
    *, entry: Any, inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], output: Mapping[str, torch.Tensor],
    population: str, protocol: Mapping[str, Any], calibration: SplitConformalCalibrator, inference_ms: float, rss_mb: float,
) -> dict[str, Any]:
    perturbation_type, perturbation_level, category = _condition(population)
    behavior = OOD_CATEGORY_BEHAVIOR[category]
    mask = inputs["source_candidate_mask"].bool()[0].cpu().numpy()
    neural = _softmax(output["source_node_logits"][0])
    classical = inputs["classical_prior"][0].float().cpu().numpy().astype(float)
    posterior, js = _candidate_probabilities(neural, classical, mask)
    candidate_indices = calibration.candidate_set(
        posterior[mask], network_id=entry.network_id,
        ood_level="NORMAL" if behavior.calibration_valid else "OUTSIDE_VALIDATED_RANGE",
    ) if behavior.calibration_valid else ()
    source_positions = np.flatnonzero(mask)
    candidates = tuple(int(source_positions[index]) for index in candidate_indices)
    true_index = int(targets["source_node"][0].item()) if bool(targets["source_node_mask"][0]) else None
    ranked = np.argsort(-posterior)
    rank = int(np.where(ranked == true_index)[0][0]) + 1 if true_index is not None else None
    quality = inputs.get("quality_features")
    sensor_mask = inputs.get("sensor_mask")
    deployed_nodes = sensor_mask[0].bool().any(dim=0) if sensor_mask is not None else None
    sensor_count = int(deployed_nodes.sum().item()) if deployed_nodes is not None else None
    observation_count = int(sensor_mask[0].bool().sum().item()) if sensor_mask is not None else None
    healthy_fraction = 1.0
    if quality is not None and sensor_mask is not None and sensor_count:
        latest = quality[0, -1, :, 0].detach().cpu().numpy()
        deployed = deployed_nodes.detach().cpu().numpy().reshape(-1)
        if deployed.any():
            healthy_fraction = float(np.nanmean(latest[deployed] >= 0.5))
        else:
            healthy_fraction = 0.0
    ood_score = 0.0 if category is OODCategory.NONE else 1.0
    evidence_sufficient = bool(behavior.calibration_valid and len(candidates) in {1, 2, 3})
    suppression: list[str] = []
    if not behavior.calibration_valid:
        suppression.append(f"OOD_{category.value}")
        suppression.append("CALIBRATION_INVALID_OR_MISSING")
    if js >= 0.5:
        suppression.append("HIGH_CLASSICAL_NEURAL_DISAGREEMENT")
    if not evidence_sufficient:
        suppression.append("CANDIDATE_REGION_TOO_BROAD")
    planning_allowed = bool(evidence_sufficient and not suppression and behavior.planning_permitted)
    if planning_allowed:
        action = ControlAction.GENERATE_PLANS
    else:
        action = uncertainty_control(
            candidate_count=max(1, len(candidates)), disagreement_js=js, ood_score=ood_score,
            healthy_sensor_fraction=max(0.0, min(1.0, healthy_fraction)), sample_budget_remaining=5,
        )
        if action is ControlAction.GENERATE_PLANS:
            action = ControlAction.REQUEST_SAMPLE if healthy_fraction >= 0.25 else ControlAction.ABSTAIN
    node_ids = tuple(entry.topology.node_ids) if entry.topology else ()
    source_node = node_ids[true_index] if true_index is not None and true_index < len(node_ids) else None
    return {
        "run_id": hashlib.sha256(f"{population}:{entry.scenario_id}".encode()).hexdigest()[:16],
        "git_commit": protocol["tested_git_commit"], **protocol["identities"],
        "network_id": entry.network_id, "network_sha256": entry.topology.network_hash if entry.topology else None,
        "topology_class": _topology_class(category), "random_seed": entry.seed, "source_node": source_node,
        "sensor_count": sensor_count, "observation_count": observation_count,
        "perturbation_type": perturbation_type, "perturbation_level": perturbation_level,
        "top1_correct": None if rank is None else rank == 1, "top3_correct": None if rank is None else rank <= 3,
        "reciprocal_rank": None if rank is None else 1.0 / rank,
        "true_source_probability": None if true_index is None else float(posterior[true_index]),
        "candidate_set_size": len(candidates), "candidate_contains_truth": None if true_index is None else true_index in candidates,
        "posterior_entropy": _entropy(posterior[mask]), "calibrated": behavior.calibration_valid,
        "ood_level": "NORMAL" if category is OODCategory.NONE else "OUTSIDE_VALIDATED_RANGE",
        "disagreement_js": js, "evidence_sufficient": evidence_sufficient, "planning_allowed": planning_allowed,
        "control_action": action.value, "suppression_reasons": suppression,
        "samples_requested": 1 if action is ControlAction.REQUEST_SAMPLE else 0,
        # This study does not invoke the active-sampling engine.  A highest-
        # posterior source is not a sampling recommendation and must never be
        # presented as one.  Likewise, tensor-forward timing is inference
        # timing only, not pipeline-analysis or workflow time.
        "first_recommended_node": None, "expected_information_gain": None,
        "inference_ms": inference_ms, "analysis_ms": None, "sampling_ms": None, "planning_ms": None,
        "verification_ms": None, "total_workflow_ms": None, "process_rss_mb": rss_mb,
        "exact_simulator_calls": 0, "outcome": "SUPPRESSED" if not planning_allowed else "ADVISORY_ONLY",
        "error_class": None, "runtime_mode": "OFFLINE_TENSOR_REPLAY",
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Null-safe aggregate by predeclared perturbation condition."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["perturbation_level"]), []).append(row)
    summary: dict[str, Any] = {"runs": len(rows), "conditions": {}, "authority_invariant_failures": []}
    for name, group in sorted(groups.items()):
        def mean(field: str) -> float | None:
            values = [float(item[field]) for item in group if item.get(field) is not None]
            return float(statistics.fmean(values)) if values else None
        summary["conditions"][name] = {
            "runs": len(group), "top1": mean("top1_correct"), "top3": mean("top3_correct"), "mrr": mean("reciprocal_rank"),
            "candidate_set_size": mean("candidate_set_size"), "entropy": mean("posterior_entropy"),
            "planning_suppression_rate": 1.0 - (mean("planning_allowed") or 0.0),
            "median_inference_ms": statistics.median(float(item["inference_ms"]) for item in group if item["inference_ms"] is not None),
        }
    for row in rows:
        if not row["planning_allowed"] and row["control_action"] == ControlAction.GENERATE_PLANS.value:
            summary["authority_invariant_failures"].append(row["run_id"])
    return summary


def write_results(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in REQUIRED_ROW_FIELDS})


def run(repo_root: Path, *, protocol_path: Path, output_dir: Path, verify_only: bool = False) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if locked_test_status(repo_root):
        raise RuntimeError("locked_test_opened is true; characterization must not run")
    if verify_only:
        return {"locked_test_opened": False, "protocol_valid": True}
    bundle = load_v4_inference_bundle(repo_root / "models/hydrocore-v4-release")
    calibration = SplitConformalCalibrator(bundle.calibration) if bundle.calibration is not None else None
    if calibration is None:
        raise RuntimeError("shipping bundle calibration is required for characterization")
    rows: list[dict[str, Any]] = []
    process = psutil.Process(os.getpid())
    for population in protocol["populations"]:
        expected = "validation" if population == "validation" else "development_holdout"
        dataset = ShardedScenarioDataset(repo_root / protocol["corpus_root"] / population, expected_split=expected)
        dataset.verify_shard_checksums()
        indices = deterministic_indices(dataset, population=population, limit=int(protocol["max_rows_per_population"]), seed=int(protocol["random_seed"]))
        for index in indices:
            entry = dataset._entries[index]
            example = dataset[index]
            inputs, targets = collate_variable_topology([example])
            with torch.inference_mode():
                bundle.model(inputs)  # one predeclared warm-up per condition/row shape
                timings = []
                output: Mapping[str, torch.Tensor] | None = None
                for _ in range(int(protocol["timing_repetitions"])):
                    started = perf_counter()
                    output = bundle.model(inputs)
                    timings.append((perf_counter() - started) * 1000.0)
            assert output is not None
            rows.append(_row(entry=entry, inputs=inputs, targets=targets, output=output, population=population,
                             protocol=protocol, calibration=calibration, inference_ms=float(statistics.median(timings)),
                             rss_mb=process.memory_info().rss / (1024 * 1024)))
    summary = aggregate(rows)
    summary.update({
        "schema_version": "hydroswarm-robustness-scale-summary-v1", "locked_test_opened_before": False,
        "locked_test_opened_after": locked_test_status(repo_root), "platform": {
            "system": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "total_ram_mb": psutil.virtual_memory().total / (1024 * 1024),
        }, "known_limitations": [
            "Rows are offline replay of governed tensors; live WNTR pipeline, sampling, plan generation, and exact verification are not invoked.",
            "Candidate posterior is an offline fixed blend because raw hydraulic trust inputs are not stored in tensor shards; authority applicability remains the current deterministic policy.",
            "Null performance fields mean unavailable, not zero cost.",
        ],
    })
    if summary["locked_test_opened_after"]:
        raise RuntimeError("locked test became opened during campaign")
    write_results(rows, summary, output_dir)
    return summary
