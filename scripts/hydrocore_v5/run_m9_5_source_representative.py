"""Milestone 9.5: source-representative calibration-support confirmation
study for the frozen ARM_A/CURRENT and ARM_B2/STEP_MATCHED_INTERLEAVED_
MULTI_FAMILY HydroCore-S predictors (follow-up to
`reports/evaluation/hydrocore-v5/m9-4/m9-4-closure.json`,
`M9_4_DECISION="B"`).

CALIBRATION-SUPPORT / FROZEN-CHECKPOINT STUDY ONLY: no training, no tuning,
no architecture change, no calibration-method change. The only intervention
is a much larger, nested, source-representative calibration population
(4/8/12/20 independent physical incidents/source; 20 is the ONLY
promotion-relevant level) plus a fresh, disjoint development population.

Reuses UNMODIFIED: `run_m7_topology`'s full-family loaders/`SEED_BASES`;
`run_m9_0a_evaluate`'s checkpoint loading (`_arm_a_model`/`_arm_b2_model`)
and signature-library construction (`_build_libraries`/`_library_for`);
`hydroswarm.calibration.conformal.SplitConformalCalibrator`/
`CalibrationExample`/`classify_runtime_condition` (alpha=0.1,
B_DEPTH_AWARE/CURRENT_FAMILY_DEPTH construction, unchanged);
`hydroswarm.training.causal_prefix.scenario_to_prefix_example`/
`truncate_causal_prefix`/`CAUSAL_PREFIX_DEPTHS`.

NEW in this file only: a nested-support, full-source scenario-generation
policy (`_generate_m9_5_scenarios`) mirroring M9.4's
`_generate_m9_4_scenarios` (same generation policy, only the seed range,
role, and repeat count differ), restricted to the 3 TRAINED families
(Section 9/10: calibration validity is scoped to trained families; unseen
families are out of scope for M9.5, not just secondary -- see module-level
SCOPE note below).

SCOPE NOTE (deviation, documented): the governing prompt's Section 10 makes
unseen-family evaluation OPTIONAL ("may be evaluated only as secondary
diagnostics if already supported cleanly" / "just reuse a simple fixed pool
if you evaluate them at all"). Given M9.5's primary question is entirely
about TRAINED-family calibration validity (Section 1/5/9/16), and every
required gate/decision path (Sections 16-27) is defined purely in terms of
trained-family cells, this implementation OMITS unseen-family inference
entirely to keep the (already large, ~19k-forward-pass) run tractable. This
is flagged explicitly in m9-5-closure.json's limitations.

Runtime-feasibility note (Section 8): development support was evaluated at
the preferred 20 incidents/source BEFORE committing (see `_feasibility_check`
below) -- total physical incidents for calibration_m9_5 (support=20) +
development_m9_5 (20/source), 3 trained families only, is 760, roughly 2.6x
M9.4's 288-incident full run (which completed in well under an hour). This
was judged tractable and used as-is; no fallback to 10/source was needed.

Writes:
  reports/evaluation/hydrocore-v5/m9-5/m9-5-manifest.json
  reports/evaluation/hydrocore-v5/m9-5/m9-5-source-policy.json
  reports/evaluation/hydrocore-v5/m9-5/m9-5-representativeness-audit.json
  reports/evaluation/hydrocore-v5/m9-5/m9-5-canonical-calibration.jsonl
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

import m9_5_common as m5  # noqa: E402
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

assert set(_AA_KNOWN) == set(m5.ARM_A_KNOWN_FAMILIES)
assert set(_AB2_KNOWN) == set(m5.ARM_B2_KNOWN_FAMILIES)


# ---------------------------------------------------------------------------
# Section 4/7/8: nested-support, full-source, exchangeable scenario generation.
# ---------------------------------------------------------------------------


def _generate_m9_5_scenarios(family: str, loader: Any, role: str, repeats: int) -> list[tuple[ScenarioRecord, dict[str, Any]]]:
    """Complete candidate source-junction set (NO truncation), `repeats` draws
    each, IDENTICAL generation policy for calibration_m9_5 and
    development_m9_5 (only the seed range, split label, and repeat count
    differ) -- mirrors M9.4's `_generate_m9_4_scenarios` exactly, generalized
    to a configurable repeat count and M9.5's own seed-base table. Sources are
    enumerated in deterministic sorted order and repeats in range(repeats),
    so support_N (repeat < N) is by construction a prefix subset of any
    larger repeats value drawn from the SAME seed base."""

    junctions = m5.full_junction_list(family, loader)
    seed_base = m5.m9_5_seed_base(family, role)
    split = DatasetSplit.CALIBRATION if role == "calibration_m9_5" else DatasetSplit.DEVELOPMENT_HOLDOUT
    generator = WNTRScenarioGenerator()
    out: list[tuple[ScenarioRecord, dict[str, Any]]] = []
    for source_index, source in enumerate(junctions):
        for repeat in range(repeats):
            seed = seed_base + source_index * m5.M9_5_SOURCE_STRIDE + repeat
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


def _build_m9_5_pools() -> dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]:
    pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]] = {}
    for family in m5.TRAINED_FAMILIES:
        loader = m5.ALL_FAMILY_LOADERS[family]
        print(f"generating calibration_m9_5 pool for {family} (support={m5.PRIMARY_SUPPORT})...", flush=True)
        pools[(family, "calibration_m9_5")] = _generate_m9_5_scenarios(family, loader, "calibration_m9_5", m5.PRIMARY_SUPPORT)
        print(f"generating development_m9_5 pool for {family} ({m5.DEVELOPMENT_REPEATS_PER_SOURCE}/source)...", flush=True)
        pools[(family, "development_m9_5")] = _generate_m9_5_scenarios(family, loader, "development_m9_5", m5.DEVELOPMENT_REPEATS_PER_SOURCE)
    return pools


# ---------------------------------------------------------------------------
# Section 5/6/29#2: source-policy artifact -- nested support-level counts,
# seed bases/ranges, per-source repeat accounting.
# ---------------------------------------------------------------------------


def _write_source_policy(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "support_levels_repeats_per_source": list(m5.SUPPORT_LEVELS), "primary_support_repeats_per_source": m5.PRIMARY_SUPPORT,
        "development_repeats_per_source": m5.DEVELOPMENT_REPEATS_PER_SOURCE, "families": {},
    }
    for family in m5.TRAINED_FAMILIES:
        loader = m5.ALL_FAMILY_LOADERS[family]
        junctions = m5.full_junction_list(family, loader)
        cal_cov = [cov for _r, cov in pools[(family, "calibration_m9_5")]]
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_5")]]
        nested_counts = {}
        for level in m5.SUPPORT_LEVELS:
            subset = [c for c in cal_cov if c["repeat"] < level]
            nested_counts[str(level)] = {
                "n_incidents": len(subset),
                "expected": len(junctions) * level,
                "n_incidents_per_source": {j: sum(1 for c in subset if c["source_node"] == j) for j in junctions},
            }
            assert nested_counts[str(level)]["n_incidents"] == nested_counts[str(level)]["expected"]
        payload["families"][family] = {
            "trained_or_unseen_development": "TRAINED_FAMILY",
            "complete_source_junction_set": list(junctions), "n_sources": len(junctions),
            "nested_support_counts": nested_counts,
            "development": {
                "n_incidents": len(dev_cov), "expected": len(junctions) * m5.DEVELOPMENT_REPEATS_PER_SOURCE,
                "seed_base": m5.m9_5_seed_base(family, "development_m9_5"),
                "seed_range": [min(c["generator_seed"] for c in dev_cov), max(c["generator_seed"] for c in dev_cov)],
            },
            "calibration_seed_base": m5.m9_5_seed_base(family, "calibration_m9_5"),
            "calibration_seed_range": [min(c["generator_seed"] for c in cal_cov), max(c["generator_seed"] for c in cal_cov)],
        }
        assert len(dev_cov) == len(junctions) * m5.DEVELOPMENT_REPEATS_PER_SOURCE
    return payload


# ---------------------------------------------------------------------------
# Section 11: representativeness audit (calibration_m9_5 @ support=20 vs
# development_m9_5), trained families only.
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
    for family in m5.TRAINED_FAMILIES:
        loader = m5.ALL_FAMILY_LOADERS[family]
        junctions = set(m5.full_junction_list(family, loader))
        cal_all = [cov for _r, cov in pools[(family, "calibration_m9_5")]]
        cal_cov = [c for c in cal_all if c["repeat"] < m5.PRIMARY_SUPPORT]  # support=20 == full pool
        dev_cov = [cov for _r, cov in pools[(family, "development_m9_5")]]

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
            "calibration_m9_5_at_primary_support": _covariate_summary(cal_cov),
            "development_m9_5": _covariate_summary(dev_cov),
        }
    report["checks"]["all_families_pass"] = all_pass
    report["representativeness_audit_passed"] = all_pass
    return report


# ---------------------------------------------------------------------------
# Uniform neural-example computation (calibration_m9_5 AND development_m9_5
# use the SAME code path -- probabilities, true index, condition, network_id
# -- so nonconformity scores / candidate-set application are directly
# comparable between roles). This intentionally reuses only the model-facing
# example construction (`scenario_to_prefix_example`) and NOT
# `_evaluate_on_family`'s classical-belief/hybrid-fusion machinery, which
# Section 28's predictive sanity metrics (Top-1/Top-3/MRR/NLL/Brier/entropy,
# NEURAL only) do not require.
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
            bucket = m5.depth_bucket_of(depth)
            all_finite = all(v == v and abs(v) != float("inf") for v in probs)  # v == v excludes NaN
            rows.append({
                "arm": arm, "predictor_seed": seed, "split": "calibration" if role == "calibration_m9_5" else "development",
                "family": family, "incident_id": f"{family}:{cov['source_node']}:{cov['generator_seed']}",
                "source_node": cov["source_node"], "source_index": cov["source_index"], "repeat": cov["repeat"],
                "generator_seed": cov["generator_seed"], "depth": depth, "depth_bucket": bucket,
                "probabilities": probs, "true_index": truth, "condition": condition, "network_id": f"{family}:{bucket}",
                "nonconformity_score": 1.0 - float(probs[truth]), "all_finite": bool(all_finite),
            })
    return rows


def _feasibility_check(pools: dict[tuple[str, str], list[tuple[ScenarioRecord, dict[str, Any]]]]) -> dict[str, Any]:
    n_cal = sum(len(pools[(f, "calibration_m9_5")]) for f in m5.TRAINED_FAMILIES)
    n_dev = sum(len(pools[(f, "development_m9_5")]) for f in m5.TRAINED_FAMILIES)
    return {
        "development_repeats_per_source_used": m5.DEVELOPMENT_REPEATS_PER_SOURCE,
        "fallback_to_10_per_source_used": False,
        "n_calibration_incidents_support_20": n_cal, "n_development_incidents": n_dev,
        "n_physical_incidents_total": n_cal + n_dev,
        "m9_4_comparison_n_physical_incidents_total": 288,
        "note": (
            "20/source (preferred, Section 8) used directly for development_m9_5 -- judged tractable "
            "given M9.4's 288-incident full run completed well within an hour; no fallback to 10/source "
            "was required."
        ),
    }


def main() -> int:
    m5.M9_5_DIR.mkdir(parents=True, exist_ok=True)
    m5.M9_5_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    locked_before = m5.assert_locked_test_closed()
    start_commit = m5.current_commit()
    start_branch = m5.current_branch()
    assert start_branch == m5.FROZEN_BRANCH

    print("verifying checkpoint SHA256 (before)...", flush=True)
    checkpoint_identities: dict[str, Any] = {"ARM_A": {}, "ARM_B2": {}}
    for seed in m5.SEEDS:
        rec = json.loads((m5.RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
        path = rec["export_path"]
        sha = m5.checkpoint_sha256(path)
        assert sha == rec["checkpoint_sha256"], f"ARM_A seed{seed} checkpoint hash mismatch before inference"
        checkpoint_identities["ARM_A"][str(seed)] = {"export_path": path, "sha256_before": sha}
    for seed in m5.SEEDS:
        rec = json.loads((m5.RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        path = rec["training_summary"]["export_path"]
        sha = m5.checkpoint_sha256(path)
        assert sha == rec["training_summary"]["export_sha256"], f"ARM_B2 seed{seed} checkpoint hash mismatch before inference"
        checkpoint_identities["ARM_B2"][str(seed)] = {
            "export_path": path, "sha256_before": sha, "optimizer_steps": m5.ARM_B2_TOTAL_OPTIMIZER_STEPS_BY_SEED[seed],
        }

    print("building M9.5 nested-support calibration + fresh development pools (trained families only)...", flush=True)
    started = time.time()
    pools = _build_m9_5_pools()
    print(f"pool generation took {time.time() - started:.1f}s", flush=True)

    feasibility = _feasibility_check(pools)
    print(json.dumps(feasibility, indent=2), flush=True)

    print("writing source-policy artifact...", flush=True)
    source_policy = _write_source_policy(pools)
    m5.M9_5_SOURCE_POLICY_PATH.write_text(json.dumps(source_policy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running representativeness audit...", flush=True)
    audit = _representativeness_audit(pools)
    m5.M9_5_REPRESENTATIVENESS_AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if not audit["representativeness_audit_passed"]:
        print("REPRESENTATIVENESS AUDIT FAILED -- see m9-5-representativeness-audit.json", flush=True)

    print("building signature libraries (reused unmodified from run_m9_0a_evaluate)...", flush=True)
    libraries = _build_libraries()

    print("running inference over calibration_m9_5 (support=20) and development_m9_5 (both arms x 3 seeds)...", flush=True)
    n_rows_written = 0
    with m5.M9_5_CANONICAL_CALIBRATION_PATH.open("w", encoding="utf-8") as fh:
        for arm, known_families, loader_fn in (
            ("ARM_A", m5.ARM_A_KNOWN_FAMILIES, _arm_a_model), ("ARM_B2", m5.ARM_B2_KNOWN_FAMILIES, _arm_b2_model),
        ):
            for seed in m5.SEEDS:
                print(f"  {arm} seed {seed}...", flush=True)
                model, sha_now = loader_fn(seed)
                model.eval()
                checkpoint_identities[arm][str(seed)]["sha256_during_inference"] = sha_now
                for family in known_families:
                    library = _library_for(libraries, family, arm)
                    t0 = time.time()
                    cal_rows = _neural_rows_for_pool(model, family, library, pools[(family, "calibration_m9_5")], role="calibration_m9_5", arm=arm, seed=seed)
                    dev_rows = _neural_rows_for_pool(model, family, library, pools[(family, "development_m9_5")], role="development_m9_5", arm=arm, seed=seed)
                    print(f"    {family}: {len(cal_rows)} calibration rows, {len(dev_rows)} development rows ({time.time() - t0:.1f}s)", flush=True)
                    for row in cal_rows + dev_rows:
                        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                        n_rows_written += 1
                sha_after_seed = m5.checkpoint_sha256(checkpoint_identities[arm][str(seed)]["export_path"])
                assert sha_after_seed == checkpoint_identities[arm][str(seed)]["sha256_before"], f"{arm} seed{seed} checkpoint mutated mid-inference!"
    print(f"wrote {n_rows_written} rows to {m5.M9_5_CANONICAL_CALIBRATION_PATH}", flush=True)

    print("verifying checkpoint SHA256 (after) -- no mutation...", flush=True)
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5.SEEDS:
            path = checkpoint_identities[arm][str(seed)]["export_path"]
            sha_after = m5.checkpoint_sha256(path)
            checkpoint_identities[arm][str(seed)]["sha256_after"] = sha_after
            assert sha_after == checkpoint_identities[arm][str(seed)]["sha256_before"], f"{arm} seed{seed} checkpoint mutated!"

    locked_after = m5.assert_locked_test_closed()

    manifest = {
        "milestone": "M9.5", "kind": "SOURCE_REPRESENTATIVE_CALIBRATION_SUPPORT_CONFIRMATION",
        "branch": start_branch, "start_commit": start_commit,
        "m9_4_code_commit": m5.M9_4_CODE_COMMIT, "m9_4_metadata_fix_commit": m5.M9_4_METADATA_FIX_COMMIT,
        "m9_4_closure_sha256": m5.checkpoint_sha256(str(m5.M9_4_CLOSURE_PATH)),
        "m9_4_manifest_sha256": m5.checkpoint_sha256(str(m5.M9_4_MANIFEST_PATH)),
        "m9_4_source_policy_sha256": m5.checkpoint_sha256(str(m5.REPORT_DIR / "m9-4" / "m9-4-source-policy.json")),
        "checkpoint_identities": checkpoint_identities, "environment": m5.environment_info(),
        "seeds": list(m5.SEEDS), "alpha": m5.ALPHA, "minimum_group_size": m5.MINIMUM_GROUP_SIZE,
        "coverage_floor": m5.OPERATIONAL_COVERAGE_FLOOR, "coverage_target_nominal": m5.NOMINAL_COVERAGE_TARGET,
        "trained_families": list(m5.TRAINED_FAMILIES), "depths": list(m5.DEPTHS),
        "support_levels_repeats_per_source": list(m5.SUPPORT_LEVELS), "primary_support_repeats_per_source": m5.PRIMARY_SUPPORT,
        "development_repeats_per_source": m5.DEVELOPMENT_REPEATS_PER_SOURCE,
        "m9_5_seed_bases": {f"{k[0]}|{k[1]}": v for k, v in m5.M9_5_SEED_BASES.items()},
        "quantile_bootstrap_resamples": m5.QUANTILE_BOOTSTRAP_RESAMPLES, "quantile_bootstrap_seed": m5.QUANTILE_BOOTSTRAP_SEED,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "no_training_performed": True, "no_predictor_modified": True,
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "feasibility": feasibility, "n_canonical_rows": n_rows_written,
        "unseen_family_scope_note": (
            "Unseen-family (coastal-branch/tree-branch/dense-loop) inference was OMITTED per this module's "
            "documented SCOPE NOTE -- Section 10 makes it optional and calibration validity concerns trained "
            "families only; every M9.5 gate/decision is defined purely in terms of trained-family cells."
        ),
    }
    m5.M9_5_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("done.", flush=True)
    print(json.dumps({
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "n_canonical_rows": n_rows_written,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
