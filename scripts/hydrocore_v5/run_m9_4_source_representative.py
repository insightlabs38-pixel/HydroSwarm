"""Milestone 9.4: source-representative, exchangeability-corrected
re-evaluation of the frozen ARM_A/CURRENT and ARM_B2/STEP_MATCHED_INTERLEAVED
_MULTI_FAMILY HydroCore-S predictors (docs: see the M9.4 governing prompt;
follow-up to `reports/evaluation/hydrocore-v5/m9-3/m9-3-closure.json`).

FROZEN-CHECKPOINT RE-EVALUATION ONLY: no training, no tuning, no
architecture change. Reuses UNMODIFIED: `run_m9_0a_evaluate`'s checkpoint
loading / signature-library construction / `_evaluate_on_family` /
`_postprocess_rows` / `_calibration_examples`; `run_m7_topology`'s
`_classical_fit_library` / `_rank_metrics` / `_entropy_normalized`;
`hydroswarm.calibration.conformal.SplitConformalCalibrator` (alpha=0.1,
B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH construction, unchanged).

NEW in this file only: a full-source (no `EVAL_MAX_SOURCES` truncation)
scenario-generation policy (`_generate_m9_4_scenarios`), applied IDENTICALLY
to both the `calibration_m9_4` and `development_m9_4` roles (same stage/
event/degradation config, only seed range and split label differ) so the
two populations are exchangeable by construction -- the corrective for
M9.3's H2 (`CALIBRATION_DEVELOPMENT_SHIFT`, SUPPORTED) finding.

Section 8 (legacy-reproduction bridge): re-executes `run_m9_0a_evaluate`/
`run_m9_0a_decide`'s `main()` UNMODIFIED, with their output-path globals
monkeypatched to a scratch directory so the historical
`reports/evaluation/hydrocore-v5/m9-0a-*` artifacts are never touched.

Writes:
  reports/evaluation/hydrocore-v5/m9-4/m9-4-manifest.json
  reports/evaluation/hydrocore-v5/m9-4/m9-4-source-policy.json
  reports/evaluation/hydrocore-v5/m9-4/m9-4-representativeness-audit.json
  reports/evaluation/hydrocore-v5/m9-4/m9-4-legacy-reproduction.json
  reports/evaluation/hydrocore-v5/m9-4/m9-4-predictions.jsonl
  reports/evaluation/hydrocore-v5/m9-4/m9-4-calibration.json
"""

from __future__ import annotations

import importlib
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m9_4_common as m4  # noqa: E402
from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator, classify_runtime_condition  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage as ScenarioCurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.training.causal_prefix import CAUSAL_PREFIX_DEPTHS, ScenarioRecord, scenario_to_prefix_example, truncate_causal_prefix  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402

from run_m3_calibration import DEPTH_BUCKET_OF  # noqa: E402
from run_m7_topology import _classical_belief, _entropy_normalized, _infer, _rank_metrics  # noqa: E402
from run_m9_0_arm_b import FEATURE_KWARGS  # noqa: E402
from run_m9_0a_evaluate import (  # noqa: E402
    ARM_A_KNOWN_FAMILIES as _AA_KNOWN,
    ARM_B2_KNOWN_FAMILIES as _AB2_KNOWN,
    _arm_a_model,
    _arm_b2_model,
    _build_libraries,
    _evaluate_on_family,
    _library_for,
    _postprocess_rows,
)

assert set(_AA_KNOWN) == set(m4.ARM_A_KNOWN_FAMILIES)
assert set(_AB2_KNOWN) == set(m4.ARM_B2_KNOWN_FAMILIES)

#: Section 6: source nodes evaluated by the M9.0a legacy generator
#: (alphabetically-sorted, truncated to EVAL_MAX_SOURCES=4). Recomputed
#: live below (never hardcoded) but this is the definition being applied.
LEGACY_MAX_SOURCES = 4


# ---------------------------------------------------------------------------
# Section 4/5: full-source, exchangeable scenario generation.
# ---------------------------------------------------------------------------


def _generate_m9_4_scenarios(family: str, loader: Any, role: str) -> list[tuple[ScenarioRecord, dict[str, Any]]]:
    """Complete candidate source-junction set (NO truncation), REPEATS_PER_SOURCE
    draws each, IDENTICAL generation policy for calibration_m9_4 and
    development_m9_4 (only the seed range and split label differ) -- mirrors
    `run_m7_topology._generate_eval_scenarios`'s own policy exactly, minus
    the EVAL_MAX_SOURCES truncation."""

    junctions = m4.full_junction_list(family, loader)
    seed_base = m4.m9_4_seed_base(family, role)
    split = DatasetSplit.CALIBRATION if role == "calibration_m9_4" else DatasetSplit.DEVELOPMENT_HOLDOUT
    generator = WNTRScenarioGenerator()
    out: list[tuple[ScenarioRecord, dict[str, Any]]] = []
    for source_index, source in enumerate(junctions):
        for repeat in range(m4.REPEATS_PER_SOURCE):
            seed = seed_base + source_index * 1_000 + repeat
            network = loader()
            config = ScenarioGenerationConfig(
                seed=seed, network_id=family, network_family=family,
                split=split, stage=ScenarioCurriculumStage.OPERATIONAL,
                event_type=EventType.CONTAMINATION, source_node=source,
                sensor_count=min(len(junctions), 4), pipe_outage_probability=0.0,
            )
            scenario, randomized_network = generator.generate_with_network(network, config)
            context = build_feature_context(randomized_network)
            record = ScenarioRecord(scenario=scenario, network=randomized_network, feature_context=context)
            incident = scenario.manifest.incident
            covariates = {
                "family": family, "role": role, "source_node": source, "source_index": source_index,
                "repeat": repeat, "generator_seed": seed, "scenario_id": str(scenario.manifest.scenario_id),
                "legacy_included_source": source_index < LEGACY_MAX_SOURCES,
                "event_severity": incident.relative_strength, "contamination_start_minute": incident.start_minute,
                "contamination_duration_minutes": incident.duration_minutes, "demand_regime": incident.demand_regime,
                "profile": incident.profile, "sensor_count": len(scenario.manifest.sensor_nodes),
                "sensor_nodes": list(scenario.manifest.sensor_nodes),
                "missing_probability_config": config.missing_probability,
                "frozen_probability_config": config.frozen_probability,
                "communication_outage_probability_config": config.communication_outage_probability,
                "drift_per_hour_config": config.drift_per_hour,
                "unit_mismatch_probability_config": config.unit_mismatch_probability,
                "roughness_variation_fraction_config": config.roughness_variation_fraction,
                "tank_level_variation_fraction_config": config.tank_level_variation_fraction,
                "pipe_outage_probability_config": config.pipe_outage_probability,
                "network_sha256": scenario.manifest.network_sha256,
                "hydraulic_timestep_seconds": int(randomized_network.options.time.hydraulic_timestep),
                "pattern_timestep_seconds": int(randomized_network.options.time.pattern_timestep),
            }
            out.append((record, covariates))
    return out


def _build_m9_4_pools() -> dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]:
    pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]] = {}
    for family in m4.ALL_FAMILIES:
        loader = m4.ALL_FAMILY_LOADERS[family]
        for role in m4.M9_4_ROLES:
            print(f"generating {role} pool for {family}...", flush=True)
            pools[(family, role)] = _generate_m9_4_scenarios(family, loader, role)
    return pools


# ---------------------------------------------------------------------------
# Section 4 arithmetic sanity + source-policy artifact.
# ---------------------------------------------------------------------------


def _write_source_policy(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"repeats_per_source": m4.REPEATS_PER_SOURCE, "families": {}}
    for family in m4.ALL_FAMILIES:
        loader = m4.ALL_FAMILY_LOADERS[family]
        junctions = m4.full_junction_list(family, loader)
        legacy_subset = junctions[:LEGACY_MAX_SOURCES]
        newly_included = junctions[LEGACY_MAX_SOURCES:]
        payload["families"][family] = {
            "trained_or_unseen_development": "TRAINED_FAMILY" if family in m4.TRAINED_FAMILIES else "UNSEEN_DEVELOPMENT_FAMILY",
            "complete_source_junction_set": list(junctions),
            "n_sources": len(junctions),
            "legacy_included_source_set": list(legacy_subset),
            "newly_included_source_set": list(newly_included),
            "expected_incidents_per_role": len(junctions) * m4.REPEATS_PER_SOURCE,
            "roles": {
                role: {
                    "seed_base": m4.m9_4_seed_base(family, role),
                    "n_incidents": len(pools[(family, role)]),
                    "seed_range": [
                        min(cov["generator_seed"] for _r, cov in pools[(family, role)]),
                        max(cov["generator_seed"] for _r, cov in pools[(family, role)]),
                    ],
                }
                for role in m4.M9_4_ROLES
            },
        }
        for role in m4.M9_4_ROLES:
            expected = len(junctions) * m4.REPEATS_PER_SOURCE
            actual = len(pools[(family, role)])
            assert actual == expected, f"{family}/{role}: expected {expected} incidents, got {actual}"
    return payload


# ---------------------------------------------------------------------------
# Section 19: representativeness audit (calibration_m9_4 vs development_m9_4).
# ---------------------------------------------------------------------------


def _covariate_summary(covariates: list[dict[str, Any]]) -> dict[str, Any]:
    if not covariates:
        return {}
    numeric_keys = (
        "event_severity", "contamination_start_minute", "contamination_duration_minutes", "demand_regime",
        "sensor_count", "missing_probability_config", "frozen_probability_config",
        "communication_outage_probability_config", "drift_per_hour_config", "unit_mismatch_probability_config",
        "roughness_variation_fraction_config", "tank_level_variation_fraction_config",
        "pipe_outage_probability_config", "hydraulic_timestep_seconds", "pattern_timestep_seconds",
    )
    source_counts: dict[str, int] = {}
    for cov in covariates:
        source_counts[cov["source_node"]] = source_counts.get(cov["source_node"], 0) + 1
    return {
        "n": len(covariates),
        "source_node_counts": source_counts,
        "n_distinct_sources": len(source_counts),
        "numeric_means": {key: statistics.fmean(float(cov[key]) for cov in covariates) for key in numeric_keys},
    }


def _representativeness_audit(
    pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"families": {}, "checks": {}}
    all_pass = True
    for family in m4.ALL_FAMILIES:
        loader = m4.ALL_FAMILY_LOADERS[family]
        junctions = set(m4.full_junction_list(family, loader))
        cal_cov = [cov for _r, cov in pools[(family, "calibration_m9_4")]]
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_4")]]

        cal_sources = {cov["source_node"] for cov in cal_cov}
        dev_sources = {cov["source_node"] for cov in dev_cov}
        cal_seeds = {cov["generator_seed"] for cov in cal_cov}
        dev_seeds = {cov["generator_seed"] for cov in dev_cov}
        cal_scenario_ids = {cov["scenario_id"] for cov in cal_cov}
        dev_scenario_ids = {cov["scenario_id"] for cov in dev_cov}

        checks = {
            "all_sources_in_calibration": cal_sources == junctions,
            "all_sources_in_development": dev_sources == junctions,
            "no_zero_support_source": all(
                sum(1 for cov in cal_cov if cov["source_node"] == j) > 0
                and sum(1 for cov in dev_cov if cov["source_node"] == j) > 0
                for j in junctions
            ),
            "equal_repeats_per_source_calibration": len({
                sum(1 for cov in cal_cov if cov["source_node"] == j) for j in junctions
            }) <= 1,
            "equal_repeats_per_source_development": len({
                sum(1 for cov in dev_cov if cov["source_node"] == j) for j in junctions
            }) <= 1,
            "seed_disjoint_calibration_vs_development": cal_seeds.isdisjoint(dev_seeds),
            "no_scenario_id_overlap": cal_scenario_ids.isdisjoint(dev_scenario_ids),
            "no_first_n_truncation": len(cal_sources) == len(junctions) and len(dev_sources) == len(junctions),
        }
        family_pass = all(checks.values())
        all_pass = all_pass and family_pass
        report["families"][family] = {
            "checks": checks, "family_pass": family_pass,
            "calibration_m9_4": _covariate_summary(cal_cov),
            "development_m9_4": _covariate_summary(dev_cov),
        }
    report["checks"]["all_families_pass"] = all_pass
    report["representativeness_audit_passed"] = all_pass
    return report


# ---------------------------------------------------------------------------
# Section 8: legacy-reproduction bridge -- re-execute the frozen M9.0a
# evaluate/decide pipeline UNMODIFIED (output paths monkeypatched to a
# scratch dir so historical m9-0a-* artifacts are never overwritten).
# ---------------------------------------------------------------------------


def _run_legacy_reproduction() -> dict[str, Any]:
    scratch = m4.M9_4_DIR / "legacy-reproduction-scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    eval_mod = importlib.import_module("run_m9_0a_evaluate")
    decide_mod = importlib.import_module("run_m9_0a_decide")

    original_eval_paths = (eval_mod.OUT_RESULTS, eval_mod.OUT_TOPOLOGY, eval_mod.OUT_CALIBRATION)
    original_decide_paths = (
        decide_mod.RESULTS_PATH, decide_mod.TOPOLOGY_PATH, decide_mod.CALIBRATION_PATH,
        decide_mod.SUMMARY_PATH, decide_mod.BUDGET_PARITY_PATH,
    )
    try:
        eval_mod.OUT_RESULTS = scratch / "m9-0a-results.json"
        eval_mod.OUT_TOPOLOGY = scratch / "m9-0a-topology-generalization.json"
        eval_mod.OUT_CALIBRATION = scratch / "m9-0a-calibration.json"
        print("legacy reproduction: re-executing frozen run_m9_0a_evaluate.main() (scratch output)...", flush=True)
        started = time.time()
        assert eval_mod.main() == 0
        eval_wall = time.time() - started

        decide_mod.RESULTS_PATH = eval_mod.OUT_RESULTS
        decide_mod.TOPOLOGY_PATH = eval_mod.OUT_TOPOLOGY
        decide_mod.CALIBRATION_PATH = eval_mod.OUT_CALIBRATION
        decide_mod.SUMMARY_PATH = scratch / "m9-0a-summary.md"
        decide_mod.BUDGET_PARITY_PATH = scratch / "m9-0a-budget-parity.json"
        print("legacy reproduction: re-executing frozen run_m9_0a_decide.main() (scratch output)...", flush=True)
        assert decide_mod.main() == 0
    finally:
        eval_mod.OUT_RESULTS, eval_mod.OUT_TOPOLOGY, eval_mod.OUT_CALIBRATION = original_eval_paths
        (
            decide_mod.RESULTS_PATH, decide_mod.TOPOLOGY_PATH, decide_mod.CALIBRATION_PATH,
            decide_mod.SUMMARY_PATH, decide_mod.BUDGET_PARITY_PATH,
        ) = original_decide_paths

    results = json.loads((scratch / "m9-0a-results.json").read_text())
    topology = json.loads((scratch / "m9-0a-topology-generalization.json").read_text())
    calibration = json.loads((scratch / "m9-0a-calibration.json").read_text())

    known = decide_mod._known_network_summary(results)
    unseen = decide_mod._unseen_pooled_and_per_family(topology)
    pooled_gain_pp = (
        statistics.fmean(unseen["ARM_B2"]["pooled_mature_neural_top1"])
        - statistics.fmean(unseen["ARM_A"]["pooled_mature_neural_top1"])
    ) * 100
    per_seed_diffs = []
    for seed in m4.SEEDS:
        a_vals, b_vals = [], []
        for family in m4.UNSEEN_DEVELOPMENT_FAMILIES:
            rows_a = topology["arms"]["ARM_A"]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"][str(seed)]
            rows_b = topology["arms"]["ARM_B2"]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"][str(seed)]
            a_vals.extend(decide_mod._per_incident_top1(rows_a, "metrics_neural", "MATURE").values())
            b_vals.extend(decide_mod._per_incident_top1(rows_b, "metrics_neural", "MATURE").values())
        per_seed_diffs.append(statistics.fmean(b_vals) - statistics.fmean(a_vals))

    arm_a_coverage = tuple(
        calibration["arms"]["ARM_A"]["per_seed"][str(seed)]["known_family"].get("marginal_coverage")
        for seed in m4.SEEDS
    )
    arm_b2_coverage = tuple(
        calibration["arms"]["ARM_B2"]["per_seed"][str(seed)]["known_family"].get("marginal_coverage")
        for seed in m4.SEEDS
    )

    checks = {
        "pooled_gain_pp_within_tolerance": m4.relative_close(pooled_gain_pp, m4.LEGACY_POOLED_UNSEEN_MATURE_NEURAL_TOP1_GAIN_PP, rel_tol=1e-3),
        "per_seed_diffs_within_tolerance": all(
            m4.relative_close(actual, expected, rel_tol=1e-3)
            for actual, expected in zip(per_seed_diffs, m4.LEGACY_PER_SEED_POOLED_MATURE_DIFFS, strict=True)
        ),
        "arm_a_coverage_within_tolerance": all(
            m4.relative_close(actual, expected, rel_tol=1e-3)
            for actual, expected in zip(arm_a_coverage, m4.LEGACY_ARM_A_MARGINAL_COVERAGE, strict=True)
        ),
        "arm_b2_coverage_within_tolerance": all(
            m4.relative_close(actual, expected, rel_tol=1e-3)
            for actual, expected in zip(arm_b2_coverage, m4.LEGACY_ARM_B2_MARGINAL_COVERAGE, strict=True)
        ),
    }
    passed = all(checks.values())

    result = {
        "M9_4_LEGACY_REPRODUCTION": "PASS" if passed else "FAIL",
        "reproduction_method": (
            "Re-executed run_m9_0a_evaluate.main()/run_m9_0a_decide.main() UNMODIFIED "
            "(output-path globals monkeypatched to a scratch directory; historical "
            "reports/evaluation/hydrocore-v5/m9-0a-* artifacts were never written to)."
        ),
        "reproduced": {
            "pooled_unseen_mature_neural_top1_gain_pp": pooled_gain_pp,
            "per_seed_pooled_mature_diff": per_seed_diffs,
            "arm_a_known_family_marginal_coverage": list(arm_a_coverage),
            "arm_b2_known_family_marginal_coverage": list(arm_b2_coverage),
        },
        "legacy_recorded": {
            "pooled_unseen_mature_neural_top1_gain_pp": m4.LEGACY_POOLED_UNSEEN_MATURE_NEURAL_TOP1_GAIN_PP,
            "per_seed_pooled_mature_diff": list(m4.LEGACY_PER_SEED_POOLED_MATURE_DIFFS),
            "arm_a_known_family_marginal_coverage": list(m4.LEGACY_ARM_A_MARGINAL_COVERAGE),
            "arm_b2_known_family_marginal_coverage": list(m4.LEGACY_ARM_B2_MARGINAL_COVERAGE),
        },
        "checks": checks,
        "eval_wall_seconds": eval_wall,
        "known_network_summary_reproduced": known,
    }
    return result


# ---------------------------------------------------------------------------
# Section 5/16: calibration-example construction on the M9.4 calibration_m9_4
# pool (byte-for-byte the same CalibrationExample construction as
# run_m9_0a_evaluate._calibration_examples, applied to OUR pool).
# ---------------------------------------------------------------------------


def _calibration_examples_m9_4(model, family: str, library: Any, calibration_records: list[ScenarioRecord]) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    for depth in CAUSAL_PREFIX_DEPTHS:
        for record in calibration_records:
            example = scenario_to_prefix_example(
                record.scenario, record.network, library, depth, feature_context=record.feature_context, **FEATURE_KWARGS,
            )
            with torch.no_grad():
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
            probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
            truth = int(example.targets["source_node"].item())
            full_series = build_sensor_series(record.scenario, record.feature_context)
            truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
            condition = classify_runtime_condition(truncated_series)
            bucket = DEPTH_BUCKET_OF[depth]
            examples.append(CalibrationExample(
                probabilities=tuple(probs), true_index=truth, condition=condition, network_id=f"{family}:{bucket}",
            ))
    return examples


def _fit_calibrator_m9_4(
    arm: str, seed: int, libraries: dict[str, Any], loader_fn,
    pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]],
) -> SplitConformalCalibrator:
    known_families = m4.ARM_A_KNOWN_FAMILIES if arm == "ARM_A" else m4.ARM_B2_KNOWN_FAMILIES
    model, _sha = loader_fn(seed)
    model.eval()
    examples: list[CalibrationExample] = []
    for family in sorted(known_families):
        library = _library_for(libraries, family, arm)
        records = [record for record, _cov in pools[(family, "calibration_m9_4")]]
        examples.extend(_calibration_examples_m9_4(model, family, library, records))
    return SplitConformalCalibrator.fit(
        examples, alpha=m4.ALPHA, minimum_group_size=m4.MINIMUM_GROUP_SIZE,
        model_hash=f"m9-4-{arm}-seed{seed}", feature_schema_hash="n/a",
        dataset_manifest_hash=f"m9-4-{arm}-seed{seed}-calibration_m9_4-pool",
    )


# ---------------------------------------------------------------------------
# Full-source inference over development_m9_4 (both arms x 3 seeds).
# ---------------------------------------------------------------------------


def _mean(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        node: Any = row
        for key in keys:
            node = node[key]
        if node is not None:
            values.append(float(node))
    return statistics.fmean(values) if values else None


def _median(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        node: Any = row
        for key in keys:
            node = node[key]
        if node is not None:
            values.append(float(node))
    return statistics.median(values) if values else None


def _cal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "marginal_coverage": _mean(rows, "candidate_covered"),
        "mean_candidate_set_size": _mean(rows, "candidate_set_size"),
        "median_candidate_set_size": _median(rows, "candidate_set_size"),
        "singleton_rate": statistics.fmean(row["candidate_set_size"] == 1 for row in rows),
        "calibration_applicability_rate": _mean(rows, "calibration_applicable"),
        "by_maturity": {
            bucket: {
                "coverage": _mean([r for r in rows if r["depth_bucket"] == bucket], "candidate_covered"),
                "mean_set_size": _mean([r for r in rows if r["depth_bucket"] == bucket], "candidate_set_size"),
            }
            for bucket in ("EARLY", "MID", "MATURE")
        },
    }


def main() -> int:
    m4.M9_4_DIR.mkdir(parents=True, exist_ok=True)
    m4.M9_4_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    locked_before = m4.assert_locked_test_closed()
    start_commit = m4.current_commit()
    start_branch = m4.current_branch()
    assert start_branch == m4.FROZEN_BRANCH

    print("verifying checkpoint SHA256 (before)...", flush=True)
    checkpoint_identities: dict[str, Any] = {"ARM_A": {}, "ARM_B2": {}}
    for seed in m4.SEEDS:
        rec = json.loads((m4.RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
        path = rec["export_path"]
        sha = m4.checkpoint_sha256(path)
        assert sha == rec["checkpoint_sha256"], f"ARM_A seed{seed} checkpoint hash mismatch before inference"
        checkpoint_identities["ARM_A"][str(seed)] = {"export_path": path, "sha256_before": sha}
    for seed in m4.SEEDS:
        rec = json.loads((m4.RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        path = rec["training_summary"]["export_path"]
        sha = m4.checkpoint_sha256(path)
        assert sha == rec["training_summary"]["export_sha256"], f"ARM_B2 seed{seed} checkpoint hash mismatch before inference"
        checkpoint_identities["ARM_B2"][str(seed)] = {
            "export_path": path, "sha256_before": sha,
            "optimizer_steps": m4.ARM_B2_TOTAL_OPTIMIZER_STEPS_BY_SEED[seed],
        }

    print("building M9.4 full-source scenario pools...", flush=True)
    pools = _build_m9_4_pools()

    print("writing source-policy artifact...", flush=True)
    source_policy = _write_source_policy(pools)
    m4.M9_4_SOURCE_POLICY_PATH.write_text(json.dumps(source_policy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running representativeness audit...", flush=True)
    audit = _representativeness_audit(pools)
    m4.M9_4_REPRESENTATIVENESS_AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if not audit["representativeness_audit_passed"]:
        print("REPRESENTATIVENESS AUDIT FAILED -- see m9-4-representativeness-audit.json", flush=True)

    print("running Section 8 legacy-reproduction bridge (re-executing frozen M9.0a pipeline)...", flush=True)
    legacy = _run_legacy_reproduction()
    m4.M9_4_LEGACY_REPRODUCTION_PATH.write_text(json.dumps(legacy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"M9_4_LEGACY_REPRODUCTION = {legacy['M9_4_LEGACY_REPRODUCTION']}", flush=True)

    print("building signature libraries (reused unmodified from run_m9_0a_evaluate)...", flush=True)
    libraries = _build_libraries()

    print("running full-source inference over development_m9_4 (both arms x 3 seeds)...", flush=True)
    all_rows: dict[str, dict[int, list[dict[str, Any]]]] = {"ARM_A": {}, "ARM_B2": {}}
    #: keyed by (family, generator_seed) -- generator_seed is unique per
    #: incident within a family, so this is an O(1) lookup from an
    #: evaluated row (row["family"], row["seed"]) back to its covariates.
    dev_covariates: dict[tuple[str, int], dict[str, Any]] = {}
    for family in m4.ALL_FAMILIES:
        for record, cov in pools[(family, "development_m9_4")]:
            dev_covariates[(family, cov["generator_seed"])] = cov

    for arm, known_families, loader_fn in (
        ("ARM_A", m4.ARM_A_KNOWN_FAMILIES, _arm_a_model), ("ARM_B2", m4.ARM_B2_KNOWN_FAMILIES, _arm_b2_model),
    ):
        for seed in m4.SEEDS:
            print(f"  {arm} seed {seed}...", flush=True)
            model, sha_now = loader_fn(seed)
            model.eval()
            checkpoint_identities[arm][str(seed)]["sha256_during_inference"] = sha_now
            seed_rows: list[dict[str, Any]] = []
            for family in m4.ALL_FAMILIES:
                if family in ("branched-loop", "loop-grid") and arm == "ARM_A":
                    continue
                known = family in known_families
                library = _library_for(libraries, family, arm)
                dev_records = [record for record, _cov in pools[(family, "development_m9_4")]]
                eval_scenarios = [(r.scenario, r.network, r.feature_context) for r in dev_records]
                with torch.no_grad():
                    rows = _evaluate_on_family(model, family, library, eval_scenarios, known=known)
                seed_rows.extend(rows)
            all_rows[arm][seed] = seed_rows

    print("fitting B_DEPTH_AWARE calibrators on calibration_m9_4 (per arm x seed)...", flush=True)
    calibrators: dict[str, dict[int, SplitConformalCalibrator]] = {"ARM_A": {}, "ARM_B2": {}}
    for arm, loader_fn in (("ARM_A", _arm_a_model), ("ARM_B2", _arm_b2_model)):
        for seed in m4.SEEDS:
            print(f"  fitting {arm} seed {seed} calibrator...", flush=True)
            calibrators[arm][seed] = _fit_calibrator_m9_4(arm, seed, libraries, loader_fn, pools)

    print("post-processing rows (fusion + calibration application)...", flush=True)
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m4.SEEDS:
            _postprocess_rows(all_rows[arm][seed], calibrators[arm][seed])

    print("verifying checkpoint SHA256 (after) -- no mutation...", flush=True)
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m4.SEEDS:
            path = checkpoint_identities[arm][str(seed)]["export_path"]
            sha_after = m4.checkpoint_sha256(path)
            checkpoint_identities[arm][str(seed)]["sha256_after"] = sha_after
            assert sha_after == checkpoint_identities[arm][str(seed)]["sha256_before"], f"{arm} seed{seed} checkpoint mutated!"

    print("writing canonical predictions.jsonl...", flush=True)
    n_rows_written = 0
    with m4.M9_4_PREDICTIONS_PATH.open("w", encoding="utf-8") as fh:
        for arm in ("ARM_A", "ARM_B2"):
            for seed in m4.SEEDS:
                for row in all_rows[arm][seed]:
                    cov = dev_covariates.get((row["family"], row["seed"]))
                    out_row = {
                        "arm": arm, "predictor_seed": seed, "family": row["family"], "known": row["known"],
                        "depth": row["depth"], "depth_bucket": row["depth_bucket"],
                        "generator_seed": row["seed"], "truth_node": row["truth_node"], "truth_index": row["truth_index"],
                        "node_ids": row["node_ids"], "neural_probs": row["neural_probs"],
                        "classical_belief": row["classical_belief"], "hybrid_belief": row["hybrid_belief"],
                        "condition": row["condition"], "evidence_sufficiency": row["evidence_sufficiency"],
                        "healthy_sensor_fraction": row["healthy_sensor_fraction"], "missing_rate": row["missing_rate"],
                        "metrics_neural": row["metrics_neural"], "metrics_classical": row["metrics_classical"],
                        "metrics_hybrid": row["metrics_hybrid"], "nll_neural": row["nll_neural"],
                        "posterior_entropy_neural": row["posterior_entropy_neural"], "all_finite": row["all_finite"],
                        "calibration_source": row.get("calibration_source"), "calibration_group_key": row.get("calibration_group_key"),
                        "calibration_applicable": row.get("calibration_applicable"), "candidate_set_size": row.get("candidate_set_size"),
                        "candidate_set_includes_truth": row.get("candidate_set_includes_truth"),
                        "candidate_covered": row.get("candidate_covered"),
                        "source_node": cov["source_node"] if cov else row["truth_node"],
                        "source_index": cov["source_index"] if cov else None,
                        "repeat": cov["repeat"] if cov else None,
                        "scenario_id": cov["scenario_id"] if cov else None,
                        "legacy_included_source": cov["legacy_included_source"] if cov else None,
                        "incident_id": f"{row['family']}:{cov['source_node'] if cov else row['truth_node']}:{row['seed']}",
                    }
                    fh.write(json.dumps(out_row, sort_keys=True, default=str) + "\n")
                    n_rows_written += 1
    print(f"wrote {n_rows_written} rows to {m4.M9_4_PREDICTIONS_PATH}", flush=True)

    print("writing calibration report...", flush=True)
    calibration_report: dict[str, Any] = {"arms": {"ARM_A": {"per_seed": {}}, "ARM_B2": {"per_seed": {}}}}
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m4.SEEDS:
            rows_this_seed = all_rows[arm][seed]
            known_rows = [r for r in rows_this_seed if r["known"]]
            unseen_rows = [r for r in rows_this_seed if not r["known"]]
            per_family = {
                family: _cal_summary([r for r in known_rows if r["family"] == family])
                for family in (m4.ARM_A_KNOWN_FAMILIES if arm == "ARM_A" else m4.ARM_B2_KNOWN_FAMILIES)
            }
            calibration_report["arms"][arm]["per_seed"][str(seed)] = {
                "alpha": m4.ALPHA, "known_family": _cal_summary(known_rows),
                "known_family_per_family": per_family,
                "unseen_topology_calibration_transfer": _cal_summary(unseen_rows),
            }
        for family in (m4.ARM_A_KNOWN_FAMILIES if arm == "ARM_A" else m4.ARM_B2_KNOWN_FAMILIES):
            coverages = [
                calibration_report["arms"][arm]["per_seed"][str(seed)]["known_family_per_family"][family].get("marginal_coverage")
                for seed in m4.SEEDS
            ]
            coverages = [c for c in coverages if c is not None]
            calibration_report["arms"][arm].setdefault("aggregate_by_family", {})[family] = {
                "mean_marginal_coverage": statistics.fmean(coverages) if coverages else None,
                "min_marginal_coverage": min(coverages) if coverages else None,
                "n_seeds_passing_0_85": sum(1 for c in coverages if c >= m4.OPERATIONAL_COVERAGE_FLOOR),
                "all_3_seeds_pass_0_85": all(c >= m4.OPERATIONAL_COVERAGE_FLOOR for c in coverages) if len(coverages) == 3 else False,
            }
        coverages_overall = [
            calibration_report["arms"][arm]["per_seed"][str(seed)]["known_family"].get("marginal_coverage") for seed in m4.SEEDS
        ]
        coverages_overall = [c for c in coverages_overall if c is not None]
        calibration_report["arms"][arm]["aggregate"] = {
            "mean_marginal_coverage": statistics.fmean(coverages_overall) if coverages_overall else None,
            "min_marginal_coverage": min(coverages_overall) if coverages_overall else None,
            "max_marginal_coverage": max(coverages_overall) if coverages_overall else None,
            "n_seeds_passing_0_85": sum(1 for c in coverages_overall if c >= m4.OPERATIONAL_COVERAGE_FLOOR),
            "all_3_seeds_pass_0_85": all(c >= m4.OPERATIONAL_COVERAGE_FLOOR for c in coverages_overall) if len(coverages_overall) == 3 else False,
        }
    m4.M9_4_CALIBRATION_PATH.write_text(json.dumps(calibration_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    locked_after = m4.assert_locked_test_closed()

    manifest = {
        "milestone": "M9.4", "branch": start_branch, "start_commit": start_commit,
        "m9_0a_protocol_frozen_commit": m4.M9_0A_PROTOCOL_FROZEN_COMMIT,
        "m9_0a_results_commit": m4.M9_0A_RESULTS_COMMIT,
        "m9_0b_protocol_frozen_commit": m4.M9_0B_PROTOCOL_FROZEN_COMMIT,
        "m9_0b_results_commit": m4.M9_0B_RESULTS_COMMIT,
        "m9_3_closure_commit": m4.M9_3_CLOSURE_COMMIT,
        "checkpoint_identities": checkpoint_identities,
        "seeds": list(m4.SEEDS), "alpha": m4.ALPHA, "minimum_group_size": m4.MINIMUM_GROUP_SIZE,
        "coverage_floor": m4.OPERATIONAL_COVERAGE_FLOOR, "coverage_target": m4.NOMINAL_COVERAGE_TARGET,
        "families": list(m4.ALL_FAMILIES), "trained_families": list(m4.TRAINED_FAMILIES),
        "unseen_development_families": list(m4.UNSEEN_DEVELOPMENT_FAMILIES),
        "repeats_per_source": m4.REPEATS_PER_SOURCE, "depths": list(m4.DEPTHS),
        "m9_4_seed_bases": {f"{k[0]}|{k[1]}": v for k, v in m4.M9_4_SEED_BASES.items()},
        "bootstrap_resamples": m4.BOOTSTRAP_RESAMPLES, "bootstrap_seed": m4.BOOTSTRAP_SEED,
        "bootstrap_interval": m4.BOOTSTRAP_INTERVAL,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "no_training_performed": True, "no_predictor_modified": True,
        "legacy_reproduction": legacy["M9_4_LEGACY_REPRODUCTION"],
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "n_prediction_rows": n_rows_written,
    }
    m4.M9_4_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("done.", flush=True)
    print(json.dumps({
        "legacy_reproduction": legacy["M9_4_LEGACY_REPRODUCTION"],
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "n_prediction_rows": n_rows_written,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
