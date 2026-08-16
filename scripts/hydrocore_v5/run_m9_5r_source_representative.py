"""Milestone 9.5R: independent, one-shot confirmation of HydroCore-S
calibration at the already-predeclared adequate support level (20
independent physical calibration incidents/source), for the frozen ARM_A/
CURRENT and ARM_B2/STEP_MATCHED_INTERLEAVED_MULTI_FAMILY HydroCore-S
predictors, using completely fresh, disjoint CALIBRATION_M9_5R and
DEVELOPMENT_M9_5R populations.

This is NOT a reinterpretation of M9.5 (which remains formally closed with
M9_5_DECISION=E). M9.5R does not repeat M9.5's 4/8/12/20 support curve; it
runs EXACTLY ONE primary calibration-support condition (20 repeats/source
for both calibration and development).

Reuses UNMODIFIED: `run_m7_topology`'s full-family loaders/`SEED_BASES`;
`run_m9_0a_evaluate`'s checkpoint loading (`_arm_a_model`/`_arm_b2_model`)
and signature-library construction (`_build_libraries`/`_library_for`);
`hydroswarm.calibration.conformal.SplitConformalCalibrator`/
`CalibrationExample`/`classify_runtime_condition` (alpha=0.1,
B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH construction, unchanged);
`hydroswarm.training.causal_prefix.scenario_to_prefix_example`/
`truncate_causal_prefix`/`CAUSAL_PREFIX_DEPTHS`.

Restricted to the 3 TRAINED families (golden-reference, branched-loop,
loop-grid) -- calibration validity concerns trained families only (governing
prompt Section 9/16); unseen-family inference is out of scope for M9.5R.

Writes:
  reports/evaluation/hydrocore-v5/m9-5r/m9-5r-manifest.json
  reports/evaluation/hydrocore-v5/m9-5r/m9-5r-source-policy.json
  reports/evaluation/hydrocore-v5/m9-5r/m9-5r-representativeness-audit.json
  reports/evaluation/hydrocore-v5/m9-5r/m9-5r-canonical-calibration.jsonl
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

import torch  # noqa: E402

import m9_5r_common as m5r  # noqa: E402
from hydroswarm.calibration.conformal import classify_runtime_condition  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage as ScenarioCurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.training.causal_prefix import CAUSAL_PREFIX_DEPTHS, ScenarioRecord, scenario_to_prefix_example, truncate_causal_prefix  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402

from run_m9_0a_evaluate import (  # noqa: E402
    ARM_A_KNOWN_FAMILIES as _AA_KNOWN,
    ARM_B2_KNOWN_FAMILIES as _AB2_KNOWN,
    _arm_a_model,
    _arm_b2_model,
    _build_libraries,
    _library_for,
)

assert set(_AA_KNOWN) == set(m5r.ARM_A_KNOWN_FAMILIES)
assert set(_AB2_KNOWN) == set(m5r.ARM_B2_KNOWN_FAMILIES)


# ---------------------------------------------------------------------------
# Section 8/9: single-support, full-source, exchangeable scenario generation
# -- IDENTICAL generation policy for calibration_m9_5r and development_m9_5r
# (only the seed range, split label, and role differ).
# ---------------------------------------------------------------------------


def _generate_m9_5r_scenarios(family: str, loader: Any, role: str, repeats: int) -> list[tuple[ScenarioRecord, dict[str, Any]]]:
    junctions = m5r.full_junction_list(family, loader)
    seed_base = m5r.m9_5r_seed_base(family, role)
    split = DatasetSplit.CALIBRATION if role == "calibration_m9_5r" else DatasetSplit.DEVELOPMENT_HOLDOUT
    generator = WNTRScenarioGenerator()
    out: list[tuple[ScenarioRecord, dict[str, Any]]] = []
    for source_index, source in enumerate(junctions):
        for repeat in range(repeats):
            seed = seed_base + source_index * m5r.M9_5R_SOURCE_STRIDE + repeat
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


def _build_m9_5r_pools() -> dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]:
    pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]] = {}
    for family in m5r.TRAINED_FAMILIES:
        loader = m5r.ALL_FAMILY_LOADERS[family]
        print(f"generating calibration_m9_5r pool for {family} ({m5r.CALIBRATION_REPEATS_PER_SOURCE}/source)...", flush=True)
        pools[(family, "calibration_m9_5r")] = _generate_m9_5r_scenarios(family, loader, "calibration_m9_5r", m5r.CALIBRATION_REPEATS_PER_SOURCE)
        print(f"generating development_m9_5r pool for {family} ({m5r.DEVELOPMENT_REPEATS_PER_SOURCE}/source)...", flush=True)
        pools[(family, "development_m9_5r")] = _generate_m9_5r_scenarios(family, loader, "development_m9_5r", m5r.DEVELOPMENT_REPEATS_PER_SOURCE)
    return pools


# ---------------------------------------------------------------------------
# Section 8: source-policy artifact.
# ---------------------------------------------------------------------------


def _write_source_policy(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "calibration_repeats_per_source": m5r.CALIBRATION_REPEATS_PER_SOURCE,
        "development_repeats_per_source": m5r.DEVELOPMENT_REPEATS_PER_SOURCE,
        "no_support_sweep": True, "families": {},
    }
    for family in m5r.TRAINED_FAMILIES:
        loader = m5r.ALL_FAMILY_LOADERS[family]
        junctions = m5r.full_junction_list(family, loader)
        cal_cov = [cov for _r, cov in pools[(family, "calibration_m9_5r")]]
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_5r")]]
        assert len(cal_cov) == len(junctions) * m5r.CALIBRATION_REPEATS_PER_SOURCE
        assert len(dev_cov) == len(junctions) * m5r.DEVELOPMENT_REPEATS_PER_SOURCE
        payload["families"][family] = {
            "trained_or_unseen_development": "TRAINED_FAMILY",
            "complete_source_junction_set": list(junctions), "n_sources": len(junctions),
            "calibration": {
                "n_incidents": len(cal_cov), "expected": len(junctions) * m5r.CALIBRATION_REPEATS_PER_SOURCE,
                "n_incidents_per_source": {j: sum(1 for c in cal_cov if c["source_node"] == j) for j in junctions},
                "seed_base": m5r.m9_5r_seed_base(family, "calibration_m9_5r"),
                "seed_range": [min(c["generator_seed"] for c in cal_cov), max(c["generator_seed"] for c in cal_cov)],
            },
            "development": {
                "n_incidents": len(dev_cov), "expected": len(junctions) * m5r.DEVELOPMENT_REPEATS_PER_SOURCE,
                "n_incidents_per_source": {j: sum(1 for c in dev_cov if c["source_node"] == j) for j in junctions},
                "seed_base": m5r.m9_5r_seed_base(family, "development_m9_5r"),
                "seed_range": [min(c["generator_seed"] for c in dev_cov), max(c["generator_seed"] for c in dev_cov)],
            },
        }
    return payload


# ---------------------------------------------------------------------------
# Section 10: representativeness/exchangeability audit.
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
        "n": len(covariates), "source_node_counts": source_counts, "n_distinct_sources": len(source_counts),
        "numeric_means": {key: statistics.fmean(float(cov[key]) for cov in covariates) for key in numeric_keys},
    }


def _representativeness_audit(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"families": {}, "checks": {}}
    all_pass = True
    for family in m5r.TRAINED_FAMILIES:
        loader = m5r.ALL_FAMILY_LOADERS[family]
        junctions = set(m5r.full_junction_list(family, loader))
        cal_cov = [cov for _r, cov in pools[(family, "calibration_m9_5r")]]
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_5r")]]

        cal_sources = {cov["source_node"] for cov in cal_cov}
        dev_sources = {cov["source_node"] for cov in dev_cov}
        cal_seeds = {cov["generator_seed"] for cov in cal_cov}
        dev_seeds = {cov["generator_seed"] for cov in dev_cov}
        cal_scenario_ids = {cov["scenario_id"] for cov in cal_cov}
        dev_scenario_ids = {cov["scenario_id"] for cov in dev_cov}

        checks = {
            "all_sources_in_calibration": cal_sources == junctions,
            "all_sources_in_development": dev_sources == junctions,
            "exactly_20_calibration_incidents_per_source": all(
                sum(1 for cov in cal_cov if cov["source_node"] == j) == m5r.CALIBRATION_REPEATS_PER_SOURCE for j in junctions
            ),
            "exactly_20_development_incidents_per_source": all(
                sum(1 for cov in dev_cov if cov["source_node"] == j) == m5r.DEVELOPMENT_REPEATS_PER_SOURCE for j in junctions
            ),
            "no_zero_support_source": all(
                sum(1 for cov in cal_cov if cov["source_node"] == j) > 0
                and sum(1 for cov in dev_cov if cov["source_node"] == j) > 0
                for j in junctions
            ),
            "balanced_source_node_proportions_calibration": len({
                sum(1 for cov in cal_cov if cov["source_node"] == j) for j in junctions
            }) <= 1,
            "balanced_source_node_proportions_development": len({
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
            "calibration_m9_5r": _covariate_summary(cal_cov),
            "development_m9_5r": _covariate_summary(dev_cov),
        }
    report["checks"]["all_families_pass"] = all_pass
    report["representativeness_audit_passed"] = all_pass
    return report


# ---------------------------------------------------------------------------
# Neural example computation (shared code path for calibration_m9_5r and
# development_m9_5r rows).
# ---------------------------------------------------------------------------


def _neural_rows_for_pool(
    model: Any, family: str, library: Any, pool: list[tuple[ScenarioRecord, dict[str, Any]]], *, role: str, arm: str, seed: int,
) -> list[dict[str, Any]]:
    from run_m9_0_arm_b import FEATURE_KWARGS

    rows: list[dict[str, Any]] = []
    for record, cov in pool:
        full_series = build_sensor_series(record.scenario, record.feature_context)
        for depth in CAUSAL_PREFIX_DEPTHS:
            example = scenario_to_prefix_example(
                record.scenario, record.network, library, depth, feature_context=record.feature_context, **FEATURE_KWARGS,
            )
            with torch.no_grad():
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
            probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
            truth = int(example.targets["source_node"].item())
            truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
            condition = classify_runtime_condition(truncated_series)
            bucket = m5r.depth_bucket_of(depth)
            all_finite = all(v == v and abs(v) != float("inf") for v in probs)  # v == v excludes NaN
            rows.append({
                "arm": arm, "predictor_seed": seed, "split": "calibration" if role == "calibration_m9_5r" else "development",
                "family": family, "incident_id": f"{family}:{cov['source_node']}:{cov['generator_seed']}",
                "source_node": cov["source_node"], "source_index": cov["source_index"], "repeat": cov["repeat"],
                "generator_seed": cov["generator_seed"], "depth": depth, "depth_bucket": bucket,
                "probabilities": probs, "true_index": truth, "condition": condition, "network_id": f"{family}:{bucket}",
                "nonconformity_score": 1.0 - float(probs[truth]), "all_finite": bool(all_finite),
            })
    return rows


def main() -> int:
    m5r.M9_5R_DIR.mkdir(parents=True, exist_ok=True)

    locked_before = m5r.assert_locked_test_closed()
    start_commit = m5r.current_commit()
    start_branch = m5r.current_branch()
    assert start_branch == m5r.FROZEN_BRANCH
    assert m5r.M9_5R_PROTOCOL_PATH.exists(), "protocol freeze artifact must exist BEFORE inference -- run write_m9_5r_protocol.py first"
    protocol_freeze_sha256 = m5r.checkpoint_sha256(str(m5r.M9_5R_PROTOCOL_PATH))
    protocol = json.loads(m5r.M9_5R_PROTOCOL_PATH.read_text())
    # NOTE: protocol["start_commit"] is the HEAD *before* the protocol-freeze
    # commit existed (a commit cannot embed its own SHA at authoring time --
    # same issue M9.4/M9.5's end_commit hit). The commit that actually
    # CONTAINS the frozen protocol file is `start_commit` here (current HEAD
    # at source-representative execution time), which is correct as long as
    # no commits landed between freezing the protocol and running this
    # script -- true for this milestone's governed sequence.
    protocol_frozen_at_commit = start_commit

    print("verifying checkpoint SHA256 (before)...", flush=True)
    checkpoint_identities: dict[str, Any] = {"ARM_A": {}, "ARM_B2": {}}
    for seed in m5r.SEEDS:
        rec = json.loads((m5r.RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
        path = rec["export_path"]
        sha = m5r.checkpoint_sha256(path)
        assert sha == rec["checkpoint_sha256"], f"ARM_A seed{seed} checkpoint hash mismatch before inference"
        checkpoint_identities["ARM_A"][str(seed)] = {"export_path": path, "sha256_before": sha}
    for seed in m5r.SEEDS:
        rec = json.loads((m5r.RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        path = rec["training_summary"]["export_path"]
        sha = m5r.checkpoint_sha256(path)
        assert sha == rec["training_summary"]["export_sha256"], f"ARM_B2 seed{seed} checkpoint hash mismatch before inference"
        checkpoint_identities["ARM_B2"][str(seed)] = {
            "export_path": path, "sha256_before": sha, "optimizer_steps": m5r.ARM_B2_TOTAL_OPTIMIZER_STEPS_BY_SEED[seed],
        }

    print("building M9.5R fresh calibration + development pools (support=20 only, trained families only)...", flush=True)
    started = time.time()
    pools = _build_m9_5r_pools()
    print(f"pool generation took {time.time() - started:.1f}s", flush=True)

    print("writing source-policy artifact...", flush=True)
    source_policy = _write_source_policy(pools)
    m5r.M9_5R_SOURCE_POLICY_PATH.write_text(json.dumps(source_policy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running representativeness audit...", flush=True)
    audit = _representativeness_audit(pools)
    m5r.M9_5R_REPRESENTATIVENESS_AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if not audit["representativeness_audit_passed"]:
        print("REPRESENTATIVENESS AUDIT FAILED -- see m9-5r-representativeness-audit.json", flush=True)

    print("building signature libraries (reused unmodified from run_m9_0a_evaluate)...", flush=True)
    libraries = _build_libraries()

    print("running inference over calibration_m9_5r and development_m9_5r (both arms x 3 seeds)...", flush=True)
    n_rows_written = 0
    with m5r.M9_5R_CANONICAL_CALIBRATION_PATH.open("w", encoding="utf-8") as fh:
        for arm, known_families, loader_fn in (
            ("ARM_A", m5r.ARM_A_KNOWN_FAMILIES, _arm_a_model), ("ARM_B2", m5r.ARM_B2_KNOWN_FAMILIES, _arm_b2_model),
        ):
            for seed in m5r.SEEDS:
                print(f"  {arm} seed {seed}...", flush=True)
                model, sha_now = loader_fn(seed)
                model.eval()
                checkpoint_identities[arm][str(seed)]["sha256_during_inference"] = sha_now
                for family in known_families:
                    library = _library_for(libraries, family, arm)
                    t0 = time.time()
                    cal_rows = _neural_rows_for_pool(model, family, library, pools[(family, "calibration_m9_5r")], role="calibration_m9_5r", arm=arm, seed=seed)
                    dev_rows = _neural_rows_for_pool(model, family, library, pools[(family, "development_m9_5r")], role="development_m9_5r", arm=arm, seed=seed)
                    print(f"    {family}: {len(cal_rows)} calibration rows, {len(dev_rows)} development rows ({time.time() - t0:.1f}s)", flush=True)
                    for row in cal_rows + dev_rows:
                        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                        n_rows_written += 1
                sha_after_seed = m5r.checkpoint_sha256(checkpoint_identities[arm][str(seed)]["export_path"])
                assert sha_after_seed == checkpoint_identities[arm][str(seed)]["sha256_before"], f"{arm} seed{seed} checkpoint mutated mid-inference!"
    print(f"wrote {n_rows_written} rows to {m5r.M9_5R_CANONICAL_CALIBRATION_PATH}", flush=True)

    print("verifying checkpoint SHA256 (after) -- no mutation...", flush=True)
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5r.SEEDS:
            path = checkpoint_identities[arm][str(seed)]["export_path"]
            sha_after = m5r.checkpoint_sha256(path)
            checkpoint_identities[arm][str(seed)]["sha256_after"] = sha_after
            assert sha_after == checkpoint_identities[arm][str(seed)]["sha256_before"], f"{arm} seed{seed} checkpoint mutated!"

    locked_after = m5r.assert_locked_test_closed()

    manifest = {
        "milestone": "M9.5R", "kind": "INDEPENDENT_CALIBRATION_CONFIRMATION",
        "branch": start_branch, "start_commit": start_commit,
        "protocol_frozen_at_commit": protocol_frozen_at_commit, "protocol_freeze_sha256": protocol_freeze_sha256,
        "m9_4_code_commit": m5r.M9_4_CODE_COMMIT, "m9_4_metadata_fix_commit": m5r.M9_4_METADATA_FIX_COMMIT,
        "m9_5_code_commit": m5r.M9_5_CODE_COMMIT, "m9_5_metadata_fix_commit": m5r.M9_5_METADATA_FIX_COMMIT,
        "m9_4_closure_sha256": m5r.checkpoint_sha256(str(m5r.M9_4_CLOSURE_PATH)),
        "m9_5_closure_sha256": m5r.checkpoint_sha256(str(m5r.M9_5_CLOSURE_PATH)),
        "m9_5_manifest_sha256": m5r.checkpoint_sha256(str(m5r.M9_5_MANIFEST_PATH)),
        "checkpoint_identities": checkpoint_identities, "environment": m5r.environment_info(),
        "seeds": list(m5r.SEEDS), "alpha": m5r.ALPHA, "minimum_group_size": m5r.MINIMUM_GROUP_SIZE,
        "coverage_floor": m5r.OPERATIONAL_COVERAGE_FLOOR, "coverage_target_nominal": m5r.NOMINAL_COVERAGE_TARGET,
        "trained_families": list(m5r.TRAINED_FAMILIES), "depths": list(m5r.DEPTHS),
        "calibration_repeats_per_source": m5r.CALIBRATION_REPEATS_PER_SOURCE,
        "development_repeats_per_source": m5r.DEVELOPMENT_REPEATS_PER_SOURCE,
        "no_support_sweep": True,
        "m9_5r_seed_bases": {f"{k[0]}|{k[1]}": v for k, v in m5r.M9_5R_SEED_BASES.items()},
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "no_training_performed": True, "no_predictor_modified": True,
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "n_canonical_rows": n_rows_written,
        "unseen_family_scope_note": (
            "Unseen-family (coastal-branch/tree-branch/dense-loop) inference is OUT OF SCOPE for M9.5R -- "
            "calibration validity concerns trained families only; every M9.5R gate/decision is defined "
            "purely in terms of trained-family cells."
        ),
    }
    m5r.M9_5R_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("done.", flush=True)
    print(json.dumps({
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "n_canonical_rows": n_rows_written,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
