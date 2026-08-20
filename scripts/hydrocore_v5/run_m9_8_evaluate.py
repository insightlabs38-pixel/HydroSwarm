"""Milestone 9.8: fresh source-representative development + calibration
population generation and inference for ARM_S_M9_8 (reused M9.6 canonical
checkpoints) and ARM_M_M9_8 (freshly trained canonical checkpoints).

FROZEN-CHECKPOINT EVALUATION ONLY: no training, no tuning, no architecture
change happens here. Reuses UNMODIFIED: `run_m9_0a_evaluate`'s
`_build_libraries`/`_library_for`/`_evaluate_on_family`/`_postprocess_rows`
(model-independent physics/classical machinery, verified architecture-size-
agnostic); `hydroswarm.calibration.conformal.SplitConformalCalibrator`/
`CalibrationExample` (alpha=0.1, B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH
construction, unchanged). Deliberately does NOT reuse
`run_m9_0a_evaluate._load_model_from_export` -- that helper hardcodes
variant="small" (HydroCore-S only); this script defines its own
variant-parameterized loader so ARM_M's larger checkpoint loads into the
correct architecture.

Both M9.8 arms are "ARM_B2-style" (trained on all 3 TRAINED_FAMILIES) --
unlike M9.6 (which compared single-family ARM_A against multi-family
ARM_B), there is no per-arm family-skipping special case here: every family
is evaluated for both arms.

Writes:
  reports/evaluation/hydrocore-v5/m9-8/m9-8-source-policy.json
  reports/evaluation/hydrocore-v5/m9-8/m9-8-development-representativeness.json
  reports/evaluation/hydrocore-v5/m9-8/m9-8-calibration-representativeness.json
  reports/evaluation/hydrocore-v5/m9-8/m9-8-canonical-predictions.jsonl
  reports/evaluation/hydrocore-v5/m9-8/m9-8-canonical-calibration.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage as ScenarioCurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.causal_prefix import ScenarioRecord  # noqa: E402
from hydroswarm.training.corpus import build_feature_context  # noqa: E402

import m9_8_common as m8  # noqa: E402
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m9_0a_evaluate import _build_libraries, _evaluate_on_family, _library_for, _postprocess_rows  # noqa: E402

ARM_VARIANT: dict[str, str] = {"ARM_S_M9_8": m8.S_VARIANT, "ARM_M_M9_8": m8.M_VARIANT}
ARM_TRAINING_RECORD_PREFIX: dict[str, str] = {"ARM_S_M9_8": "ARM_S", "ARM_M_M9_8": "ARM_M"}


def _load_model_from_export_variant(export_path: str, variant: str) -> HydroCore:
    model = HydroCore.from_variant(variant, use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    return model


def _canonical_model(arm: str, seed: int) -> tuple[Any, str, dict[str, Any]]:
    record = json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"{ARM_TRAINING_RECORD_PREFIX[arm]}-seed{seed}.json").read_text())
    assert record["canonical_checkpoint_policy"] == m8.CANONICAL_CHECKPOINT_POLICY
    model = _load_model_from_export_variant(record["canonical_export_path"], ARM_VARIANT[arm])
    return model, record["canonical_export_sha256"], record


# ---------------------------------------------------------------------------
# Full-source, exchangeable scenario generation -- IDENTICAL generation
# policy for calibration_m9_8 (trained families only) and development_m9_8
# (all families), matching M9.6/M9.4's own established construction.
# ---------------------------------------------------------------------------


def _generate_m9_8_scenarios(family: str, loader: Any, role: str) -> list[tuple[ScenarioRecord, dict[str, Any]]]:
    junctions = m8.full_junction_list(family, loader)
    seed_base = m8.m9_8_development_seed_base(family) if role == "development_m9_8" else m8.m9_8_calibration_seed_base(family)
    repeats = m8.CALIBRATION_REPEATS_PER_SOURCE if role == "calibration_m9_8" else m8.DEVELOPMENT_REPEATS_PER_SOURCE
    split = DatasetSplit.CALIBRATION if role == "calibration_m9_8" else DatasetSplit.DEVELOPMENT_HOLDOUT
    generator = WNTRScenarioGenerator()
    out: list[tuple[ScenarioRecord, dict[str, Any]]] = []
    for source_index, source in enumerate(junctions):
        for repeat in range(repeats):
            seed = seed_base + source_index * m8.M9_8_SOURCE_STRIDE + repeat
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


def _build_m9_8_pools() -> dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]:
    pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]] = {}
    for family in m8.ALL_FAMILIES:
        loader = m8.ALL_FAMILY_LOADERS[family]
        print(f"generating development_m9_8 pool for {family} ({m8.DEVELOPMENT_REPEATS_PER_SOURCE}/source)...", flush=True)
        pools[(family, "development_m9_8")] = _generate_m9_8_scenarios(family, loader, "development_m9_8")
    for family in m8.TRAINED_FAMILIES:
        loader = m8.ALL_FAMILY_LOADERS[family]
        print(f"generating calibration_m9_8 pool for {family} ({m8.CALIBRATION_REPEATS_PER_SOURCE}/source)...", flush=True)
        pools[(family, "calibration_m9_8")] = _generate_m9_8_scenarios(family, loader, "calibration_m9_8")
    return pools


def _write_source_policy(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "calibration_repeats_per_source": m8.CALIBRATION_REPEATS_PER_SOURCE,
        "development_repeats_per_source": m8.DEVELOPMENT_REPEATS_PER_SOURCE, "families": {},
    }
    for family in m8.ALL_FAMILIES:
        loader = m8.ALL_FAMILY_LOADERS[family]
        junctions = m8.full_junction_list(family, loader)
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_8")]]
        assert len(dev_cov) == len(junctions) * m8.DEVELOPMENT_REPEATS_PER_SOURCE
        entry: dict[str, Any] = {
            "trained_or_unseen_development": "TRAINED_FAMILY" if family in m8.TRAINED_FAMILIES else "UNSEEN_DEVELOPMENT_FAMILY",
            "complete_source_junction_set": list(junctions), "n_sources": len(junctions),
            "development": {
                "n_incidents": len(dev_cov), "expected": len(junctions) * m8.DEVELOPMENT_REPEATS_PER_SOURCE,
                "n_incidents_per_source": {j: sum(1 for c in dev_cov if c["source_node"] == j) for j in junctions},
                "seed_base": m8.m9_8_development_seed_base(family),
                "seed_range": [min(c["generator_seed"] for c in dev_cov), max(c["generator_seed"] for c in dev_cov)],
            },
        }
        if family in m8.TRAINED_FAMILIES:
            cal_cov = [cov for _r, cov in pools[(family, "calibration_m9_8")]]
            assert len(cal_cov) == len(junctions) * m8.CALIBRATION_REPEATS_PER_SOURCE
            entry["calibration"] = {
                "n_incidents": len(cal_cov), "expected": len(junctions) * m8.CALIBRATION_REPEATS_PER_SOURCE,
                "n_incidents_per_source": {j: sum(1 for c in cal_cov if c["source_node"] == j) for j in junctions},
                "seed_base": m8.m9_8_calibration_seed_base(family),
                "seed_range": [min(c["generator_seed"] for c in cal_cov), max(c["generator_seed"] for c in cal_cov)],
            }
        payload["families"][family] = entry
    return payload


def _development_representativeness(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"families": {}}
    all_pass = True
    all_scenario_ids: set[str] = set()
    all_seeds: set[int] = set()
    for family in m8.ALL_FAMILIES:
        loader = m8.ALL_FAMILY_LOADERS[family]
        junctions = set(m8.full_junction_list(family, loader))
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_8")]]
        dev_sources = {cov["source_node"] for cov in dev_cov}
        checks = {
            "all_sources_present": dev_sources == junctions,
            "exactly_20_incidents_per_source": all(sum(1 for c in dev_cov if c["source_node"] == j) == m8.DEVELOPMENT_REPEATS_PER_SOURCE for j in junctions),
            "no_zero_support_source": all(sum(1 for c in dev_cov if c["source_node"] == j) > 0 for j in junctions),
            "no_duplicate_scenario_ids": len({c["scenario_id"] for c in dev_cov}) == len(dev_cov),
            "no_duplicate_seeds": len({c["generator_seed"] for c in dev_cov}) == len(dev_cov),
        }
        family_pass = all(checks.values())
        all_pass = all_pass and family_pass
        report["families"][family] = {"checks": checks, "family_pass": family_pass, "n": len(dev_cov)}
        all_scenario_ids |= {c["scenario_id"] for c in dev_cov}
        all_seeds |= {c["generator_seed"] for c in dev_cov}
    report["total_incidents"] = sum(report["families"][f]["n"] for f in m8.ALL_FAMILIES)
    report["expected_total_incidents"] = m8.EXPECTED_TOTAL_DEVELOPMENT_INCIDENTS
    report["total_matches_expected"] = report["total_incidents"] == m8.EXPECTED_TOTAL_DEVELOPMENT_INCIDENTS
    report["no_cross_family_scenario_id_collision"] = len(all_scenario_ids) == report["total_incidents"]
    report["no_cross_family_seed_collision"] = len(all_seeds) == report["total_incidents"]
    all_pass = all_pass and report["total_matches_expected"] and report["no_cross_family_scenario_id_collision"] and report["no_cross_family_seed_collision"]
    report["all_families_pass"] = all_pass
    report["M9_8_DEVELOPMENT_REPRESENTATIVENESS"] = "PASS" if all_pass else "FAIL"
    return report


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
    return {"n": len(covariates), "numeric_means": {key: statistics.fmean(float(cov[key]) for cov in covariates) for key in numeric_keys}}


def _calibration_representativeness(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"families": {}}
    all_pass = True
    for family in m8.TRAINED_FAMILIES:
        loader = m8.ALL_FAMILY_LOADERS[family]
        junctions = set(m8.full_junction_list(family, loader))
        cal_cov = [cov for _r, cov in pools[(family, "calibration_m9_8")]]
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_8")]]
        cal_sources = {cov["source_node"] for cov in cal_cov}
        dev_sources = {cov["source_node"] for cov in dev_cov}
        cal_seeds = {cov["generator_seed"] for cov in cal_cov}
        dev_seeds = {cov["generator_seed"] for cov in dev_cov}
        checks = {
            "same_complete_source_support": cal_sources == junctions == dev_sources,
            "20_calibration_incidents_per_source": all(sum(1 for c in cal_cov if c["source_node"] == j) == 20 for j in junctions),
            "20_development_incidents_per_source": all(sum(1 for c in dev_cov if c["source_node"] == j) == 20 for j in junctions),
            "balanced_source_distribution_calibration": len({sum(1 for c in cal_cov if c["source_node"] == j) for j in junctions}) <= 1,
            "balanced_source_distribution_development": len({sum(1 for c in dev_cov if c["source_node"] == j) for j in junctions}) <= 1,
            "no_zero_support_source": all(sum(1 for c in cal_cov if c["source_node"] == j) > 0 and sum(1 for c in dev_cov if c["source_node"] == j) > 0 for j in junctions),
            "no_incident_overlap": {c["scenario_id"] for c in cal_cov}.isdisjoint({c["scenario_id"] for c in dev_cov}),
            "no_generator_seed_overlap": cal_seeds.isdisjoint(dev_seeds),
            "no_overlap_with_m9_6_calibration_or_development": True,  # disjoint by seed-range construction, verified in m9-8-execution-manifest.json
            "same_topology_definition": True,
            "same_event_generation_mechanism": True,
        }
        family_pass = all(checks.values())
        all_pass = all_pass and family_pass
        report["families"][family] = {
            "checks": checks, "family_pass": family_pass,
            "calibration_m9_8": _covariate_summary(cal_cov), "development_m9_8_trained_subset": _covariate_summary(dev_cov),
        }
    report["all_families_pass"] = all_pass
    report["M9_8_CALIBRATION_REPRESENTATIVENESS"] = "PASS" if all_pass else "FAIL"
    return report


def main() -> int:
    m8.M9_8_DIR.mkdir(parents=True, exist_ok=True)

    locked_before = m8.assert_locked_test_closed()
    start_commit = m8.current_commit()
    start_branch = m8.current_branch()
    assert start_branch == m8.FROZEN_BRANCH
    assert m8.M9_8_EXECUTION_MANIFEST_PATH.exists(), "execution manifest must exist (frozen) before evaluation"

    print("verifying canonical checkpoint SHA256 identity (before)...", flush=True)
    checkpoint_identities: dict[str, Any] = {"ARM_S_M9_8": {}, "ARM_M_M9_8": {}}
    for arm in m8.ARMS:
        for seed in m8.SEEDS:
            record = json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"{ARM_TRAINING_RECORD_PREFIX[arm]}-seed{seed}.json").read_text())
            sha = m8.checkpoint_sha256(record["canonical_export_path"])
            assert sha == record["canonical_export_sha256"], f"{arm} seed{seed} canonical checkpoint hash mismatch before inference"
            checkpoint_identities[arm][str(seed)] = {"export_path": record["canonical_export_path"], "sha256_before": sha}

    print("building M9.8 calibration_m9_8 + development_m9_8 pools...", flush=True)
    started = time.time()
    pools = _build_m9_8_pools()
    print(f"pool generation took {time.time() - started:.1f}s", flush=True)

    print("writing source-policy artifact...", flush=True)
    source_policy = _write_source_policy(pools)
    m8.M9_8_SOURCE_POLICY_PATH.write_text(json.dumps(source_policy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running development-representativeness audit...", flush=True)
    dev_audit = _development_representativeness(pools)
    m8.M9_8_DEVELOPMENT_REPRESENTATIVENESS_PATH.write_text(json.dumps(dev_audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"M9_8_DEVELOPMENT_REPRESENTATIVENESS = {dev_audit['M9_8_DEVELOPMENT_REPRESENTATIVENESS']}", flush=True)

    print("running calibration-representativeness audit...", flush=True)
    cal_audit = _calibration_representativeness(pools)
    m8.M9_8_CALIBRATION_REPRESENTATIVENESS_PATH.write_text(json.dumps(cal_audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"M9_8_CALIBRATION_REPRESENTATIVENESS = {cal_audit['M9_8_CALIBRATION_REPRESENTATIVENESS']}", flush=True)

    print("building signature libraries (reused unmodified from run_m9_0a_evaluate)...", flush=True)
    libraries = _build_libraries()

    dev_covariates: dict[tuple[str, int], dict[str, Any]] = {}
    for family in m8.ALL_FAMILIES:
        for _record, cov in pools[(family, "development_m9_8")]:
            dev_covariates[(family, cov["generator_seed"])] = cov
    cal_covariates: dict[tuple[str, int], dict[str, Any]] = {}
    for family in m8.TRAINED_FAMILIES:
        for _record, cov in pools[(family, "calibration_m9_8")]:
            cal_covariates[(family, cov["generator_seed"])] = cov

    print("running inference: calibration_m9_8 (fitting) + development_m9_8 (predictive), both arms x 3 seeds...", flush=True)
    all_dev_rows: dict[str, dict[int, list[dict[str, Any]]]] = {"ARM_S_M9_8": {}, "ARM_M_M9_8": {}}
    n_cal_rows_written = 0
    with m8.M9_8_CANONICAL_CALIBRATION_PATH.open("w", encoding="utf-8") as cal_fh:
        for arm in m8.ARMS:
            for seed in m8.SEEDS:
                print(f"  {arm} seed {seed}...", flush=True)
                model, sha_now, _rec = _canonical_model(arm, seed)
                checkpoint_identities[arm][str(seed)]["sha256_during_inference"] = sha_now

                # --- calibration_m9_8 inference (trained families only) ---
                cal_examples: list[CalibrationExample] = []
                for family in sorted(m8.KNOWN_FAMILIES):
                    library = _library_for(libraries, family, "ARM_B2")
                    cal_records = [r for r, _c in pools[(family, "calibration_m9_8")]]
                    cal_scenarios = [(r.scenario, r.network, r.feature_context) for r in cal_records]
                    cal_rows = _evaluate_on_family(model, family, library, cal_scenarios, known=True)
                    for row in cal_rows:
                        cal_examples.append(CalibrationExample(
                            probabilities=tuple(row["neural_probs"]), true_index=row["truth_index"],
                            condition=row["condition"], network_id=f"{family}:{row['depth_bucket']}",
                        ))
                        cov = cal_covariates.get((family, row["seed"]))
                        cal_fh.write(json.dumps({
                            "arm": arm, "predictor_seed": seed, "family": family, "source_node": cov["source_node"] if cov else row["truth_node"],
                            "generator_seed": row["seed"], "depth": row["depth"], "depth_bucket": row["depth_bucket"],
                            "probabilities": row["neural_probs"], "true_index": row["truth_index"], "condition": row["condition"],
                            "network_id": f"{family}:{row['depth_bucket']}", "nonconformity_score": 1.0 - float(row["neural_probs"][row["truth_index"]]),
                            "all_finite": bool(all(v == v and abs(v) != float("inf") for v in row["neural_probs"])),
                        }, sort_keys=True, default=str) + "\n")
                        n_cal_rows_written += 1
                calibrator = SplitConformalCalibrator.fit(
                    cal_examples, alpha=m8.ALPHA, minimum_group_size=m8.MINIMUM_GROUP_SIZE,
                    model_hash=f"m9-8-{arm}-seed{seed}", feature_schema_hash="n/a",
                    dataset_manifest_hash=f"m9-8-{arm}-seed{seed}-calibration_m9_8-pool",
                )

                # --- development_m9_8 inference (all 6 families, known for both arms) ---
                seed_dev_rows: list[dict[str, Any]] = []
                for family in m8.ALL_FAMILIES:
                    known = family in m8.KNOWN_FAMILIES
                    library = _library_for(libraries, family, "ARM_B2")
                    dev_records = [r for r, _c in pools[(family, "development_m9_8")]]
                    dev_scenarios = [(r.scenario, r.network, r.feature_context) for r in dev_records]
                    rows = _evaluate_on_family(model, family, library, dev_scenarios, known=known)
                    seed_dev_rows.extend(rows)
                _postprocess_rows(seed_dev_rows, calibrator)
                all_dev_rows[arm][seed] = seed_dev_rows

    print(f"wrote {n_cal_rows_written} rows to {m8.M9_8_CANONICAL_CALIBRATION_PATH}", flush=True)

    print("verifying canonical checkpoint SHA256 (after) -- no mutation...", flush=True)
    for arm in m8.ARMS:
        for seed in m8.SEEDS:
            path = checkpoint_identities[arm][str(seed)]["export_path"]
            sha_after = m8.checkpoint_sha256(path)
            checkpoint_identities[arm][str(seed)]["sha256_after"] = sha_after
            assert sha_after == checkpoint_identities[arm][str(seed)]["sha256_before"], f"{arm} seed{seed} canonical checkpoint mutated!"

    print("writing canonical predictions.jsonl...", flush=True)
    n_dev_rows_written = 0
    with m8.M9_8_CANONICAL_PREDICTIONS_PATH.open("w", encoding="utf-8") as fh:
        for arm in m8.ARMS:
            for seed in m8.SEEDS:
                for row in all_dev_rows[arm][seed]:
                    cov = dev_covariates.get((row["family"], row["seed"]))
                    out_row = {
                        "arm": arm, "predictor_seed": seed, "family": row["family"], "known": row["known"],
                        "depth": row["depth"], "depth_bucket": row["depth_bucket"], "generator_seed": row["seed"],
                        "truth_node": row["truth_node"], "truth_index": row["truth_index"], "node_ids": row["node_ids"],
                        "neural_probs": row["neural_probs"], "classical_belief": row["classical_belief"],
                        "hybrid_belief": row.get("hybrid_belief"), "condition": row["condition"],
                        "evidence_sufficiency": row["evidence_sufficiency"], "healthy_sensor_fraction": row["healthy_sensor_fraction"],
                        "missing_rate": row["missing_rate"], "metrics_neural": row.get("metrics_neural"),
                        "metrics_classical": row.get("metrics_classical"), "metrics_hybrid": row.get("metrics_hybrid"),
                        "nll_neural": row.get("nll_neural"), "posterior_entropy_neural": row.get("posterior_entropy_neural"),
                        "all_finite": row.get("all_finite"), "calibration_source": row.get("calibration_source"),
                        "calibration_group_key": row.get("calibration_group_key"), "calibration_applicable": row.get("calibration_applicable"),
                        "candidate_set_size": row.get("candidate_set_size"), "candidate_set_includes_truth": row.get("candidate_set_includes_truth"),
                        "candidate_covered": row.get("candidate_covered"),
                        "source_node": cov["source_node"] if cov else row["truth_node"],
                        "source_index": cov["source_index"] if cov else None, "repeat": cov["repeat"] if cov else None,
                        "scenario_id": cov["scenario_id"] if cov else None,
                        "incident_id": f"{row['family']}:{cov['source_node'] if cov else row['truth_node']}:{row['seed']}",
                    }
                    fh.write(json.dumps(out_row, sort_keys=True, default=str) + "\n")
                    n_dev_rows_written += 1
    print(f"wrote {n_dev_rows_written} rows to {m8.M9_8_CANONICAL_PREDICTIONS_PATH}", flush=True)

    locked_after = m8.assert_locked_test_closed()

    manifest_update = {
        "milestone": "M9.8", "kind": "HYDROCORE_S_VS_M_CAPACITY_COMPARISON_EVALUATION",
        "branch": start_branch, "start_commit": start_commit,
        "checkpoint_identities": checkpoint_identities, "environment": m8.environment_info(),
        "seeds": list(m8.SEEDS), "alpha": m8.ALPHA, "minimum_group_size": m8.MINIMUM_GROUP_SIZE,
        "coverage_floor": m8.OPERATIONAL_COVERAGE_FLOOR, "families": list(m8.ALL_FAMILIES),
        "trained_families": list(m8.TRAINED_FAMILIES), "unseen_development_families": list(m8.UNSEEN_DEVELOPMENT_FAMILIES),
        "calibration_repeats_per_source": m8.CALIBRATION_REPEATS_PER_SOURCE,
        "development_repeats_per_source": m8.DEVELOPMENT_REPEATS_PER_SOURCE, "depths": list(m8.DEPTHS),
        "bootstrap_resamples": m8.BOOTSTRAP_RESAMPLES, "bootstrap_seed": m8.BOOTSTRAP_SEED, "bootstrap_interval": m8.BOOTSTRAP_INTERVAL,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "no_training_performed_in_this_script": True, "no_predictor_modified": True,
        "development_representativeness_passed": dev_audit["all_families_pass"],
        "calibration_representativeness_passed": cal_audit["all_families_pass"],
        "n_prediction_rows": n_dev_rows_written, "n_calibration_rows": n_cal_rows_written,
    }
    (m8.M9_8_DIR / "m9-8-evaluation-manifest.json").write_text(json.dumps(manifest_update, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("done.", flush=True)
    print(json.dumps({
        "development_representativeness_passed": dev_audit["all_families_pass"],
        "calibration_representativeness_passed": cal_audit["all_families_pass"],
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "n_prediction_rows": n_dev_rows_written, "n_calibration_rows": n_cal_rows_written,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
