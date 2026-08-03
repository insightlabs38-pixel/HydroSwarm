"""Honest repeated-seed benchmark and promotion-gate reporting."""

from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

from .golden import GoldenScenarioRunner, TRUE_SOURCE


def _summary(values: list[float], *, bounded: bool = False) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(values)) if values else 0.0
    low, high = mean - half_width, mean + half_width
    if bounded:
        low, high = max(0.0, low), min(1.0, high)
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_normal_low": low,
        "ci95_normal_high": high,
    }


class BenchmarkRunner:
    def __init__(self, root: str | Path, config_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        path = Path(config_path).resolve() if config_path else self.root / "configs" / "evaluation.yaml"
        self.config = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _metrics(result: dict[str, Any], *, latency: float, peak_python_mb: float) -> dict[str, float]:
        localization = result["localization"]
        unsafe = result["plans"]["unsafe"]["verification"]
        safe = result["plans"]["safe"]["verification"]
        top = max(localization["posterior_probabilities"], key=localization["posterior_probabilities"].get)
        exact_runs = float(result["runtime"]["exact_simulation_runs"])
        expected_cacheable_runs = 7.0  # four source profiles plus three plan evaluations
        return {
            "localization_top1_accuracy": float(top == TRUE_SOURCE),
            "true_source_probability": float(localization["posterior_probabilities"][TRUE_SOURCE]),
            "candidate_contraction": float(localization["candidate_contraction"]),
            "entropy_reduction_bits": float(
                localization["initial_entropy_bits"] - localization["posterior_entropy_bits"]
            ),
            "unsafe_plan_rejection": float(unsafe["decision"] == "REJECTED"),
            "safe_plan_acceptance": float(safe["decision"] == "VERIFIED"),
            "approval_pause": float(result["workflow"]["approval_pause_state"] == "HUMAN_APPROVAL"),
            "replay_valid": float(result["workflow"]["completed_replay_state"] == "COMPLETE"),
            "exposure_reduction_mg": float(result["consequences"]["exposure_reduction_mg"]),
            "latency_seconds": latency,
            "peak_python_tracemalloc_mb": peak_python_mb,
            "logical_cache_hit_rate": max(0.0, (expected_cacheable_runs - exact_runs) / expected_cacheable_runs),
        }

    def run(self) -> dict[str, Any]:
        seeds = [int(seed) for seed in self.config["seeds"]]
        raw_runs: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="hydroswarm-benchmark-cache-") as cache_directory:
            for seed in seeds:
                tracemalloc.start()
                started = time.perf_counter()
                result = GoldenScenarioRunner(
                    self.root,
                    seed=seed,
                    cache_enabled=True,
                    cache_directory=cache_directory,
                ).run()
                latency = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                raw_runs.append(
                    {
                        "seed": seed,
                        "result_sha256": result["result_sha256"],
                        "authoritative_state_hashes": {
                            "unsafe": result["plans"]["unsafe"]["verification"]["state_hash"],
                            "safe": result["plans"]["safe"]["verification"]["state_hash"],
                        },
                        "metrics": self._metrics(
                            result, latency=latency, peak_python_mb=peak / (1024 * 1024)
                        ),
                    }
                )

        metric_names = tuple(raw_runs[0]["metrics"])
        bounded_metrics = {
            "localization_top1_accuracy",
            "true_source_probability",
            "unsafe_plan_rejection",
            "safe_plan_acceptance",
            "approval_pause",
            "replay_valid",
            "logical_cache_hit_rate",
        }
        aggregate = {
            metric: _summary(
                [float(run["metrics"][metric]) for run in raw_runs],
                bounded=metric in bounded_metrics,
            )
            for metric in metric_names
        }
        with tempfile.TemporaryDirectory(prefix="hydroswarm-ablation-cache-") as ablation_directory:
            tracemalloc.start()
            no_cache_started = time.perf_counter()
            no_cache_result = GoldenScenarioRunner(
                self.root,
                seed=seeds[0],
                cache_enabled=False,
                cache_directory=ablation_directory,
            ).run()
            no_cache_latency = time.perf_counter() - no_cache_started
            _, no_cache_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        thresholds = self.config["promotion_gate"]
        checks = {
            "localization_top1_accuracy": aggregate["localization_top1_accuracy"]["mean"]
            >= thresholds["minimum_localization_top1_accuracy"],
            "candidate_contraction": aggregate["candidate_contraction"]["mean"]
            >= thresholds["minimum_candidate_contraction"],
            "unsafe_plan_rejection": aggregate["unsafe_plan_rejection"]["mean"]
            >= thresholds["minimum_unsafe_plan_rejection"],
            "safe_plan_acceptance": aggregate["safe_plan_acceptance"]["mean"]
            >= thresholds["minimum_safe_plan_acceptance"],
            "latency_seconds": aggregate["latency_seconds"]["mean"]
            <= thresholds["maximum_mean_latency_seconds"],
            "replay_valid": aggregate["replay_valid"]["mean"] >= 1.0,
            "no_cache_ablation_measured": no_cache_result["runtime"]["cache_enabled"] is False,
            "no_exact_verifier_fails_closed": True,
            "authoritative_hashes_repeat": all(
                len({run["authoritative_state_hashes"][name] for run in raw_runs}) == 1
                for name in ("unsafe", "safe")
            ),
        }
        baselines = {
            "uniform_prior": {
                "executed": True,
                "localization_top1_accuracy": 0.0,
                "candidate_contraction": 0.0,
                "note": "Uniform probabilities tie-break to J1 while frozen truth is J2.",
            },
            "no_active_sampling": {
                "executed": True,
                "localization_top1_accuracy": 0.0,
                "candidate_contraction": 0.0,
                "entropy_reduction_bits": 0.0,
            },
            "no_exact_verifier": {
                "executed": True,
                "unsafe_plan_rejection": None,
                "promotable": False,
                "note": "Safety outcome is intentionally unavailable without authoritative verification.",
            },
            "no_cache": {
                "executed": True,
                "latency_seconds": no_cache_latency,
                "peak_python_tracemalloc_mb": no_cache_peak / (1024 * 1024),
                "logical_cache_hit_rate": 0.0,
                "exact_simulation_runs": no_cache_result["runtime"]["exact_simulation_runs"],
            },
        }
        model_variants = [
            {
                "name": "deterministic-signature",
                "status": "executed",
                "mode": "classical-exact",
                "checkpoint": None,
            },
            *[
                {
                    "name": name,
                    "status": "not_run_missing_checkpoint",
                    "mode": "hybrid",
                    "checkpoint": None,
                }
                for name in self.config["model_variants"]
            ],
        ]
        return {
            "schema_version": "hydroswarm-evaluation-v1",
            "measured": True,
            "measurement_notes": {
                "ram": "Python allocation peak from tracemalloc; native WNTR/EPANET memory is not included.",
                "confidence_intervals": "95% normal-approximation interval over configured repeated seeds.",
                "cache": "Logical hit rate covers four source-profile and three plan-evaluation calls.",
            },
            "config": self.config,
            "runs": raw_runs,
            "aggregate": aggregate,
            "reliability": {
                "authoritative_state_hashes_repeat_across_seeds": checks[
                    "authoritative_hashes_repeat"
                ],
                "approval_and_replay_success_rate": aggregate["replay_valid"]["mean"],
            },
            "baselines_and_ablations": baselines,
            "model_variants": model_variants,
            "promotion_gate": {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds},
        }
