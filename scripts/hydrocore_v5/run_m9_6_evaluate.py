"""Milestone 9.6: source-representative evaluation + calibration-population
generation for the freshly trained, exact-compute-parity ARM_A_M9_6/
ARM_B_M9_6 canonical (FINAL STEP 1350) checkpoints.

FROZEN-CHECKPOINT EVALUATION ONLY (relative to this script): no training,
no tuning, no architecture change happens here -- the checkpoints it reads
were produced by `run_m9_6_train_arm_a.py`/`run_m9_6_train_arm_b.py`.
Reuses UNMODIFIED: `run_m9_0a_evaluate`'s `_load_model_from_export`,
`_build_libraries`/`_library_for`, `_evaluate_on_family`/`_postprocess_rows`
(model-independent physics/classical machinery); `hydroswarm.calibration.
conformal.SplitConformalCalibrator`/`CalibrationExample` (alpha=0.1,
B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH construction, unchanged, independently
confirmed in M9.5R).

Mirrors `run_m9_4_source_representative.py`'s full-source (no
`EVAL_MAX_SOURCES` truncation), exchangeable calibration/development
generation policy, generalized to M9.6's own fresh seed namespace and its
6-family DEVELOPMENT_M9_6 scope (Section 16: unseen-family evaluation is
NOT optional for M9.6, unlike M9.5R).

Writes:
  reports/evaluation/hydrocore-v5/m9-6/m9-6-manifest.json
  reports/evaluation/hydrocore-v5/m9-6/m9-6-source-policy.json
  reports/evaluation/hydrocore-v5/m9-6/m9-6-development-representativeness.json
  reports/evaluation/hydrocore-v5/m9-6/m9-6-calibration-representativeness.json
  reports/evaluation/hydrocore-v5/m9-6/m9-6-canonical-predictions.jsonl
  reports/evaluation/hydrocore-v5/m9-6/m9-6-canonical-calibration.jsonl
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

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage as ScenarioCurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.training.causal_prefix import ScenarioRecord  # noqa: E402
from hydroswarm.training.corpus import build_feature_context  # noqa: E402

import m9_6_common as m6  # noqa: E402
from run_m9_0a_evaluate import _build_libraries, _evaluate_on_family, _library_for, _load_model_from_export, _postprocess_rows  # noqa: E402


def _library_for_m9_6(libraries: dict[str, Any], family: str, arm: str) -> Any:
    return _library_for(libraries, family, "ARM_A" if arm == "ARM_A_M9_6" else "ARM_B2")


def _canonical_model(arm: str, seed: int) -> tuple[Any, str, dict[str, Any]]:
    record = json.loads((m6.M9_6_TRAINING_RUNS_DIR / f"{arm}-seed{seed}.json").read_text())
    model = _load_model_from_export(record["canonical_export_path"])
    return model, record["canonical_export_sha256"], record


# ---------------------------------------------------------------------------
# Section 17/26: full-source, exchangeable scenario generation. IDENTICAL
# generation policy for calibration_m9_6 (trained families only) and
# development_m9_6 (all families) -- only seed range/split/repeat differ.
# ---------------------------------------------------------------------------


def _generate_m9_6_scenarios(family: str, loader: Any, role: str) -> list[tuple[ScenarioRecord, dict[str, Any]]]:
    junctions = m6.full_junction_list(family, loader)
    seed_base = m6.m9_6_seed_base(family, role)
    repeats = m6.CALIBRATION_REPEATS_PER_SOURCE if role == "calibration_m9_6" else m6.DEVELOPMENT_REPEATS_PER_SOURCE
    split = DatasetSplit.CALIBRATION if role == "calibration_m9_6" else DatasetSplit.DEVELOPMENT_HOLDOUT
    generator = WNTRScenarioGenerator()
    out: list[tuple[ScenarioRecord, dict[str, Any]]] = []
    for source_index, source in enumerate(junctions):
        for repeat in range(repeats):
            seed = seed_base + source_index * m6.M9_6_SOURCE_STRIDE + repeat
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


def _build_m9_6_pools() -> dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]:
    pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]] = {}
    for family in m6.ALL_FAMILIES:
        loader = m6.ALL_FAMILY_LOADERS[family]
        print(f"generating development_m9_6 pool for {family} ({m6.DEVELOPMENT_REPEATS_PER_SOURCE}/source)...", flush=True)
        pools[(family, "development_m9_6")] = _generate_m9_6_scenarios(family, loader, "development_m9_6")
    for family in m6.TRAINED_FAMILIES:
        loader = m6.ALL_FAMILY_LOADERS[family]
        print(f"generating calibration_m9_6 pool for {family} ({m6.CALIBRATION_REPEATS_PER_SOURCE}/source)...", flush=True)
        pools[(family, "calibration_m9_6")] = _generate_m9_6_scenarios(family, loader, "calibration_m9_6")
    return pools


# ---------------------------------------------------------------------------
# Section 8/26: source-policy artifact (calibration + development).
# ---------------------------------------------------------------------------


def _write_source_policy(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "calibration_repeats_per_source": m6.CALIBRATION_REPEATS_PER_SOURCE,
        "development_repeats_per_source": m6.DEVELOPMENT_REPEATS_PER_SOURCE, "families": {},
    }
    for family in m6.ALL_FAMILIES:
        loader = m6.ALL_FAMILY_LOADERS[family]
        junctions = m6.full_junction_list(family, loader)
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_6")]]
        assert len(dev_cov) == len(junctions) * m6.DEVELOPMENT_REPEATS_PER_SOURCE
        entry: dict[str, Any] = {
            "trained_or_unseen_development": "TRAINED_FAMILY" if family in m6.TRAINED_FAMILIES else "UNSEEN_DEVELOPMENT_FAMILY",
            "complete_source_junction_set": list(junctions), "n_sources": len(junctions),
            "development": {
                "n_incidents": len(dev_cov), "expected": len(junctions) * m6.DEVELOPMENT_REPEATS_PER_SOURCE,
                "n_incidents_per_source": {j: sum(1 for c in dev_cov if c["source_node"] == j) for j in junctions},
                "seed_base": m6.m9_6_seed_base(family, "development_m9_6"),
                "seed_range": [min(c["generator_seed"] for c in dev_cov), max(c["generator_seed"] for c in dev_cov)],
            },
        }
        if family in m6.TRAINED_FAMILIES:
            cal_cov = [cov for _r, cov in pools[(family, "calibration_m9_6")]]
            assert len(cal_cov) == len(junctions) * m6.CALIBRATION_REPEATS_PER_SOURCE
            entry["calibration"] = {
                "n_incidents": len(cal_cov), "expected": len(junctions) * m6.CALIBRATION_REPEATS_PER_SOURCE,
                "n_incidents_per_source": {j: sum(1 for c in cal_cov if c["source_node"] == j) for j in junctions},
                "seed_base": m6.m9_6_seed_base(family, "calibration_m9_6"),
                "seed_range": [min(c["generator_seed"] for c in cal_cov), max(c["generator_seed"] for c in cal_cov)],
            }
        payload["families"][family] = entry
    return payload


# ---------------------------------------------------------------------------
# Section 17: development-population audit (single population, evaluated
# identically for both arms -- source coverage/balance/no-zero-support).
# ---------------------------------------------------------------------------


def _development_representativeness(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"families": {}}
    all_pass = True
    for family in m6.ALL_FAMILIES:
        loader = m6.ALL_FAMILY_LOADERS[family]
        junctions = set(m6.full_junction_list(family, loader))
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_6")]]
        dev_sources = {cov["source_node"] for cov in dev_cov}
        checks = {
            "all_sources_present": dev_sources == junctions,
            "exactly_20_incidents_per_source": all(sum(1 for c in dev_cov if c["source_node"] == j) == m6.DEVELOPMENT_REPEATS_PER_SOURCE for j in junctions),
            "no_zero_support_source": all(sum(1 for c in dev_cov if c["source_node"] == j) > 0 for j in junctions),
            "no_duplicate_scenario_ids": len({c["scenario_id"] for c in dev_cov}) == len(dev_cov),
            "no_duplicate_seeds": len({c["generator_seed"] for c in dev_cov}) == len(dev_cov),
        }
        family_pass = all(checks.values())
        all_pass = all_pass and family_pass
        report["families"][family] = {"checks": checks, "family_pass": family_pass, "n": len(dev_cov)}
    report["all_families_pass"] = all_pass
    report["m9_6_development_representativeness_passed"] = all_pass
    return report


# ---------------------------------------------------------------------------
# Section 27: calibration-representativeness audit (calibration_m9_6 vs
# development_m9_6's trained-family subset).
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
    return {"n": len(covariates), "numeric_means": {key: statistics.fmean(float(cov[key]) for cov in covariates) for key in numeric_keys}}


def _calibration_representativeness(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"families": {}}
    all_pass = True
    for family in m6.TRAINED_FAMILIES:
        loader = m6.ALL_FAMILY_LOADERS[family]
        junctions = set(m6.full_junction_list(family, loader))
        cal_cov = [cov for _r, cov in pools[(family, "calibration_m9_6")]]
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_6")]]
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
            "same_topology_definition": True,  # same network loader used for both roles by construction
            "same_event_generation_mechanism": True,  # same ScenarioGenerationConfig fields (CONTAMINATION/OPERATIONAL) by construction
        }
        family_pass = all(checks.values())
        all_pass = all_pass and family_pass
        report["families"][family] = {
            "checks": checks, "family_pass": family_pass,
            "calibration_m9_6": _covariate_summary(cal_cov), "development_m9_6_trained_subset": _covariate_summary(dev_cov),
        }
    report["all_families_pass"] = all_pass
    report["M9_6_CALIBRATION_REPRESENTATIVENESS"] = "PASS" if all_pass else "FAIL"
    return report


def main() -> int:
    m6.M9_6_DIR.mkdir(parents=True, exist_ok=True)

    locked_before = m6.assert_locked_test_closed()
    start_commit = m6.current_commit()
    start_branch = m6.current_branch()
    assert start_branch == m6.FROZEN_BRANCH
    assert m6.M9_6_PROTOCOL_PATH.exists(), "protocol freeze artifact must exist before evaluation"

    print("verifying canonical checkpoint SHA256 identity (before)...", flush=True)
    checkpoint_identities: dict[str, Any] = {"ARM_A_M9_6": {}, "ARM_B_M9_6": {}}
    for arm in m6.ARMS:
        for seed in m6.SEEDS:
            record = json.loads((m6.M9_6_TRAINING_RUNS_DIR / f"{arm}-seed{seed}.json").read_text())
            sha = m6.checkpoint_sha256(record["canonical_export_path"])
            assert sha == record["canonical_export_sha256"], f"{arm} seed{seed} canonical checkpoint hash mismatch before inference"
            checkpoint_identities[arm][str(seed)] = {"export_path": record["canonical_export_path"], "sha256_before": sha}

    print("building M9.6 calibration_m9_6 + development_m9_6 pools...", flush=True)
    started = time.time()
    pools = _build_m9_6_pools()
    print(f"pool generation took {time.time() - started:.1f}s", flush=True)

    print("writing source-policy artifact...", flush=True)
    source_policy = _write_source_policy(pools)
    m6.M9_6_SOURCE_POLICY_PATH.write_text(json.dumps(source_policy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running development-representativeness audit...", flush=True)
    dev_audit = _development_representativeness(pools)
    m6.M9_6_DEVELOPMENT_REPRESENTATIVENESS_PATH.write_text(json.dumps(dev_audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running calibration-representativeness audit...", flush=True)
    cal_audit = _calibration_representativeness(pools)
    m6.M9_6_CALIBRATION_REPRESENTATIVENESS_PATH.write_text(json.dumps(cal_audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"M9_6_CALIBRATION_REPRESENTATIVENESS = {cal_audit['M9_6_CALIBRATION_REPRESENTATIVENESS']}", flush=True)

    print("building signature libraries (reused unmodified from run_m9_0a_evaluate)...", flush=True)
    libraries = _build_libraries()

    dev_covariates: dict[tuple[str, int], dict[str, Any]] = {}
    for family in m6.ALL_FAMILIES:
        for _record, cov in pools[(family, "development_m9_6")]:
            dev_covariates[(family, cov["generator_seed"])] = cov
    cal_covariates: dict[tuple[str, int], dict[str, Any]] = {}
    for family in m6.TRAINED_FAMILIES:
        for _record, cov in pools[(family, "calibration_m9_6")]:
            cal_covariates[(family, cov["generator_seed"])] = cov

    print("running inference: calibration_m9_6 (fitting) + development_m9_6 (predictive), both arms x 3 seeds...", flush=True)
    all_dev_rows: dict[str, dict[int, list[dict[str, Any]]]] = {"ARM_A_M9_6": {}, "ARM_B_M9_6": {}}
    calibrators: dict[str, dict[int, SplitConformalCalibrator]] = {"ARM_A_M9_6": {}, "ARM_B_M9_6": {}}
    n_cal_rows_written = 0
    with m6.M9_6_CANONICAL_CALIBRATION_PATH.open("w", encoding="utf-8") as cal_fh:
        for arm, known_families in (("ARM_A_M9_6", m6.ARM_A_KNOWN_FAMILIES), ("ARM_B_M9_6", m6.ARM_B_KNOWN_FAMILIES)):
            for seed in m6.SEEDS:
                print(f"  {arm} seed {seed}...", flush=True)
                model, sha_now, _rec = _canonical_model(arm, seed)
                checkpoint_identities[arm][str(seed)]["sha256_during_inference"] = sha_now

                # --- calibration_m9_6 inference (trained families only) ---
                cal_examples: list[CalibrationExample] = []
                for family in sorted(known_families):
                    library = _library_for_m9_6(libraries, family, arm)
                    cal_records = [r for r, _c in pools[(family, "calibration_m9_6")]]
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
                    cal_examples, alpha=m6.ALPHA, minimum_group_size=m6.MINIMUM_GROUP_SIZE,
                    model_hash=f"m9-6-{arm}-seed{seed}", feature_schema_hash="n/a",
                    dataset_manifest_hash=f"m9-6-{arm}-seed{seed}-calibration_m9_6-pool",
                )
                calibrators[arm][seed] = calibrator

                # --- development_m9_6 inference (all families known to this arm get known=True) ---
                seed_dev_rows: list[dict[str, Any]] = []
                for family in m6.ALL_FAMILIES:
                    if arm == "ARM_A_M9_6" and family in ("branched-loop", "loop-grid"):
                        # ARM_A is evaluated on golden-reference (known) and on the truly
                        # unseen-to-both families coastal-branch/tree-branch/dense-loop
                        # (known=False, Section 22's primary comparison set), mirroring
                        # M9.4's own scope exactly -- branched-loop/loop-grid are ARM_B-only
                        # trained families, not part of the unseen-macro-family gate.
                        continue
                    known = family in known_families
                    library = _library_for_m9_6(libraries, family, arm)
                    dev_records = [r for r, _c in pools[(family, "development_m9_6")]]
                    dev_scenarios = [(r.scenario, r.network, r.feature_context) for r in dev_records]
                    rows = _evaluate_on_family(model, family, library, dev_scenarios, known=known)
                    seed_dev_rows.extend(rows)
                _postprocess_rows(seed_dev_rows, calibrator)
                all_dev_rows[arm][seed] = seed_dev_rows

    print(f"wrote {n_cal_rows_written} rows to {m6.M9_6_CANONICAL_CALIBRATION_PATH}", flush=True)

    print("verifying canonical checkpoint SHA256 (after) -- no mutation...", flush=True)
    for arm in m6.ARMS:
        for seed in m6.SEEDS:
            path = checkpoint_identities[arm][str(seed)]["export_path"]
            sha_after = m6.checkpoint_sha256(path)
            checkpoint_identities[arm][str(seed)]["sha256_after"] = sha_after
            assert sha_after == checkpoint_identities[arm][str(seed)]["sha256_before"], f"{arm} seed{seed} canonical checkpoint mutated!"

    print("writing canonical predictions.jsonl...", flush=True)
    n_dev_rows_written = 0
    with m6.M9_6_CANONICAL_PREDICTIONS_PATH.open("w", encoding="utf-8") as fh:
        for arm in m6.ARMS:
            for seed in m6.SEEDS:
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
    print(f"wrote {n_dev_rows_written} rows to {m6.M9_6_CANONICAL_PREDICTIONS_PATH}", flush=True)

    locked_after = m6.assert_locked_test_closed()

    manifest = {
        "milestone": "M9.6", "kind": "EXACT_COMPUTE_PARITY_FINAL_HYDROCORE_S_CONFIRMATION",
        "branch": start_branch, "start_commit": start_commit,
        "checkpoint_identities": checkpoint_identities, "environment": m6.environment_info(),
        "seeds": list(m6.SEEDS), "alpha": m6.ALPHA, "minimum_group_size": m6.MINIMUM_GROUP_SIZE,
        "coverage_floor": m6.OPERATIONAL_COVERAGE_FLOOR, "families": list(m6.ALL_FAMILIES),
        "trained_families": list(m6.TRAINED_FAMILIES), "unseen_development_families": list(m6.UNSEEN_DEVELOPMENT_FAMILIES),
        "calibration_repeats_per_source": m6.CALIBRATION_REPEATS_PER_SOURCE,
        "development_repeats_per_source": m6.DEVELOPMENT_REPEATS_PER_SOURCE, "depths": list(m6.DEPTHS),
        "m9_6_seed_bases": {f"{k[0]}|{k[1]}": v for k, v in m6.M9_6_SEED_BASES.items()},
        "bootstrap_resamples": m6.BOOTSTRAP_RESAMPLES, "bootstrap_seed": m6.BOOTSTRAP_SEED, "bootstrap_interval": m6.BOOTSTRAP_INTERVAL,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "no_training_performed": True, "no_predictor_modified": True,
        "development_representativeness_passed": dev_audit["m9_6_development_representativeness_passed"],
        "calibration_representativeness_passed": cal_audit["all_families_pass"],
        "n_prediction_rows": n_dev_rows_written, "n_calibration_rows": n_cal_rows_written,
    }
    m6.M9_6_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("done.", flush=True)
    print(json.dumps({
        "development_representativeness_passed": dev_audit["m9_6_development_representativeness_passed"],
        "calibration_representativeness_passed": cal_audit["all_families_pass"],
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "n_prediction_rows": n_dev_rows_written, "n_calibration_rows": n_cal_rows_written,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
