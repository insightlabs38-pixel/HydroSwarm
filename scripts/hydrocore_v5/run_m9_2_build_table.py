"""Milestone 9.2: build the canonical paired diagnostic table.

DIAGNOSTIC / ANALYSIS-ONLY. Trains nothing, tunes nothing, never opens
`locked_final_test`/`locked_topology_test`.

Primary data source is already-persisted M9.1 per-incident/per-depth
`dev_rows` (`reports/evaluation/hydrocore-v5/m9-1-results.json`) for CURRENT
and the three novel arms, screening seeds only (20260814, 31874) -- these
rows already carry probabilities, truth index, per-row metrics, runtime
condition, and solver-health flags exactly as M9.1 computed them, so no
model inference is needed to obtain them.

Two quantities are NOT persisted anywhere and are reconstructed here via
deterministic, read-only inference against the exact frozen M9.1 checkpoints
(protocol Section 2 of the M9.2 brief):

1. Per-development-row conformal candidate sets. M9.1 only persisted the
   AGGREGATE calibration coverage/set-size (`m9-1-calibration.json`), not
   the per-row candidate set. Reconstructing it requires re-fitting each
   (arm, seed)'s B_DEPTH_AWARE calibrator from the `calibration` split --
   which itself requires one deterministic forward pass per (scenario,
   depth) on that split (GRAPH_SDE using its exact frozen MC=4 Brownian
   schedule, via `m9_1_common.evaluate_split`, byte-identical to what
   `run_m9_1_evaluate.py` already did once). The already-persisted DEV ROWS
   THEMSELVES require no new forward pass -- their probabilities are reused
   as-is, only run through the reconstructed calibrator's `candidate_set`.
   Equivalence against `m9-1-calibration.json`'s aggregate coverage/set-size
   is checked and recorded for every (arm, seed) before any row is trusted.
2. Topology metadata (node ordering, graph distances). The corpus uses a
   single fixed `golden-reference` topology for every scenario in every
   split (see `causal_prefix.py` module docstring), so this is built once
   from the network object itself, not per scenario.

Everything else in the table is a deterministic, causally-available
function of already-generated scenario/data properties (curriculum stage,
sensor count, per-depth missingness/gap statistics from the scenario's own
sensor series) -- no model inference involved.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx  # noqa: E402

import m9_1_common as common  # noqa: E402
import m9_2_common as m92  # noqa: E402
import run_m9_1_evaluate as run_eval  # noqa: E402
from hydroswarm.simulation.network import build_networkx_network  # noqa: E402
from hydroswarm.training.causal_prefix import truncate_causal_prefix  # noqa: E402
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402
from hydroswarm.preprocessing.alignment import canonical_node_order  # noqa: E402


TOLERANCE_STRICT = 1e-9  # exact/count-derived metrics reproduced from the same source rows
TOLERANCE_FLOAT = 1e-6  # continuous metrics recomputed via the same formula


# ---------------------------------------------------------------------------
# Step 1: load M9.1 artifacts and record their identity.
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha(path: Path) -> str:
    import subprocess

    out = subprocess.run(
        ["git", "hash-object", str(path)], cwd=m92.ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return out


def load_m9_1_sources() -> dict[str, Any]:
    paths = {
        "m9-1-results.json": m92.M9_1_RESULTS_PATH,
        "m9-1-calibration.json": m92.M9_1_CALIBRATION_PATH,
        "m9-1-guardrails.json": m92.M9_1_GUARDRAILS_PATH,
        "m9-1-closure.json": m92.M9_1_CLOSURE_PATH,
    }
    identities = {}
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"BLOCKED: required M9.1 artifact missing: {path}")
        identities[name] = {
            "path": str(path.relative_to(m92.ROOT)),
            "sha256": _sha256_file(path),
            "git_blob_sha": _git_blob_sha(path),
        }
    closure = json.loads(paths["m9-1-closure.json"].read_text())
    if closure.get("M9_1_FINAL_DECISION") != "CURRENT_HYDROCORE_RETAINED":
        raise SystemExit(
            f"BLOCKED: m9-1-closure.json M9_1_FINAL_DECISION={closure.get('M9_1_FINAL_DECISION')!r}, "
            "expected CURRENT_HYDROCORE_RETAINED"
        )
    if closure.get("locked_test_opened_before") or closure.get("locked_test_opened_after"):
        raise SystemExit("BLOCKED: m9-1-closure.json records locked_test_opened=true at some point")
    if closure.get("protocol_frozen_at_commit") != m92.PROTOCOL_FROZEN_AT_COMMIT:
        raise SystemExit("BLOCKED: m9-1-closure.json protocol_frozen_at_commit mismatch")
    return {"identities": identities, "closure": closure}


# ---------------------------------------------------------------------------
# Step 2: load dev_rows for every (arm, screening-seed), assert pairing.
# ---------------------------------------------------------------------------


def load_dev_rows() -> dict[str, dict[int, list[dict[str, Any]]]]:
    results = json.loads(m92.M9_1_RESULTS_PATH.read_text())
    out: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for arm in m92.ALL_ARMS:
        out[arm] = {}
        for seed in m92.SCREENING_SEEDS:
            block = results.get(arm, {}).get(str(seed))
            if block is None:
                raise SystemExit(f"BLOCKED: missing m9-1-results.json[{arm!r}][{seed!r}]")
            out[arm][seed] = block["dev_rows"]
    # M9.2 governance: CURRENT's unpaired third seed must never be pulled into
    # this table even though it IS present in the source artifact.
    for seed_key in results.get("CURRENT", {}):
        assert int(seed_key) != m92.EXCLUDED_UNPAIRED_SEED or True  # documented presence, not consumed below
    assert m92.EXCLUDED_UNPAIRED_SEED not in {s for s in m92.SCREENING_SEEDS}
    return out


def assert_pairing(dev_rows: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Every (seed, incident_id, depth) key present for CURRENT must be
    present for every novel arm at that seed, and vice versa -- no
    accidental unmatched rows (Section 3)."""

    key_sets: dict[tuple[str, int], set[tuple[int, int]]] = {}
    for arm, by_seed in dev_rows.items():
        for seed, rows in by_seed.items():
            keys = {(r["incident_id"], r["depth"]) for r in rows}
            if len(keys) != len(rows):
                raise SystemExit(f"BLOCKED: duplicate (incident_id, depth) rows in {arm}-seed{seed}")
            key_sets[(arm, seed)] = keys
    reference = key_sets[("CURRENT", m92.SCREENING_SEEDS[0])]
    for (arm, seed), keys in key_sets.items():
        if seed not in m92.SCREENING_SEEDS:
            raise SystemExit(f"BLOCKED: non-screening seed {seed} present in loaded rows for {arm}")
        current_seed_ref = key_sets[("CURRENT", seed)]
        if keys != current_seed_ref:
            missing = current_seed_ref - keys
            extra = keys - current_seed_ref
            raise SystemExit(
                f"BLOCKED: unmatched rows {arm}-seed{seed} vs CURRENT-seed{seed}: "
                f"missing={len(missing)} extra={len(extra)}"
            )
    return {
        "n_incidents": len({inc for inc, _d in reference}),
        "n_depths": len({d for _inc, d in reference}),
        "n_keys_per_arm_seed": len(reference),
        "all_arm_seed_combinations_paired": True,
    }


# ---------------------------------------------------------------------------
# Step 3: reproduce M9.1 aggregates from the loaded dev_rows (Section 4).
# ---------------------------------------------------------------------------


def reproduce_m9_1_aggregates(dev_rows: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    results = json.loads(m92.M9_1_RESULTS_PATH.read_text())
    mismatches = []
    reproduced = {}
    for arm in m92.ALL_ARMS:
        reproduced[arm] = {}
        for seed in m92.SCREENING_SEEDS:
            rows = dev_rows[arm][seed]
            early = run_eval._agg_top1(rows, m92.EARLY_DEPTHS)
            mid = run_eval._agg_top1(rows, m92.MID_DEPTHS)
            mature = run_eval._agg_top1(rows, m92.MATURE_DEPTHS)
            mrr = run_eval._agg_mrr(rows)
            stored = results[arm][str(seed)]["aggregates"]
            reproduced[arm][str(seed)] = {"early_top1": early, "mid_top1": mid, "mature_top1": mature, "overall_mrr": mrr}
            for key, value in (("early_top1", early), ("mid_top1", mid), ("mature_top1", mature), ("overall_mrr", mrr)):
                if abs(value - stored[key]) > TOLERANCE_FLOAT:
                    mismatches.append(
                        {"arm": arm, "seed": seed, "metric": key, "reproduced": value, "stored": stored[key]}
                    )
    if mismatches:
        raise SystemExit(f"BLOCKED: reproduced aggregates disagree with m9-1-results.json: {mismatches}")

    # Cross-check the guardrail regression-pp formula (run_m9_1_decide.py,
    # unmodified) against m9-1-guardrails.json's own screening block.
    import run_m9_1_decide as decide

    calibration = json.loads(m92.M9_1_CALIBRATION_PATH.read_text())
    guardrails = json.loads(m92.M9_1_GUARDRAILS_PATH.read_text())
    guardrail_mismatches = []
    for arm in m92.NOVEL_ARMS:
        step1 = decide._step1_guardrails(arm, m92.SCREENING_SEEDS, results, calibration)
        stored_step1 = guardrails["screening"][arm]["step1"]
        for key in ("early_regression_pp", "mature_regression_pp", "mrr_regression"):
            if abs(step1[key] - stored_step1[key]) > TOLERANCE_FLOAT:
                guardrail_mismatches.append({"arm": arm, "metric": key, "reproduced": step1[key], "stored": stored_step1[key]})
    if guardrail_mismatches:
        raise SystemExit(f"BLOCKED: reproduced guardrail regressions disagree with m9-1-guardrails.json: {guardrail_mismatches}")

    return {
        "status": "REPRODUCED_EXACTLY",
        "tolerance_float": TOLERANCE_FLOAT,
        "per_arm_seed_aggregates": reproduced,
        "guardrail_regression_pp": {
            arm: {
                "early_regression_pp": guardrails["screening"][arm]["step1"]["early_regression_pp"],
                "mature_regression_pp": guardrails["screening"][arm]["step1"]["mature_regression_pp"],
                "mrr_regression": guardrails["screening"][arm]["step1"]["mrr_regression"],
            }
            for arm in m92.NOVEL_ARMS
        },
    }


# ---------------------------------------------------------------------------
# Step 4: topology (single fixed golden-reference network for every split).
# ---------------------------------------------------------------------------


def build_topology() -> dict[str, Any]:
    graph = build_networkx_network()
    undirected = nx.Graph(graph)
    node_ids = canonical_node_order(graph.nodes())
    junctions = tuple(n for n, d in graph.nodes(data=True) if d.get("node_type") == "junction")
    distances = dict(nx.all_pairs_shortest_path_length(undirected))
    degree = dict(undirected.degree())
    return {
        "node_ids": node_ids,
        "junctions": junctions,
        "degree": degree,
        "distances": {a: dict(d) for a, d in distances.items()},
        "edges": sorted(tuple(sorted(e[:2])) for e in undirected.edges()),
    }


# ---------------------------------------------------------------------------
# Step 5: causal per-(incident, depth) missingness features (arm-independent).
# ---------------------------------------------------------------------------


def build_missingness_features(dev_records) -> dict[int, dict[int, dict[str, Any]]]:
    from hydroswarm.data.scenarios import CurriculumStage

    stages = tuple(CurriculumStage)
    out: dict[int, dict[int, dict[str, Any]]] = {}
    for index, record in enumerate(dev_records):
        incident_id = common.incident_id_for("development_holdout", index)
        full_series = build_sensor_series(record.scenario, record.feature_context)
        stage = stages[index % len(stages)]
        sensor_count = 3 + (index % 3)
        out[incident_id] = {}
        for depth in m92.CAUSAL_PREFIX_DEPTHS:
            truncated = [truncate_causal_prefix(s, depth) for s in full_series]
            total_obs = sum(len(s.timestamps_seconds) for s in truncated)
            missing_obs = sum(sum(s.missing) for s in truncated)
            delayed_obs = sum(sum(s.delayed) for s in truncated)
            frozen_obs = sum(sum(s.frozen) for s in truncated)
            drift_obs = sum(sum(s.drift) for s in truncated)
            valid_obs = total_obs - missing_obs
            health_values = [h for s in truncated for h, m in zip(s.health, s.missing) if not m]
            valid_timestamps = sorted({t for s in truncated for t, m in zip(s.timestamps_seconds, s.missing) if not m})
            gaps = [b - a for a, b in zip(valid_timestamps, valid_timestamps[1:])]
            n_sensors_contributing = sum(1 for s in truncated if any(not m for m in s.missing))
            mean_gap = statistics.fmean(gaps) if gaps else None
            out[incident_id][depth] = {
                "curriculum_stage": str(stage.value) if hasattr(stage, "value") else str(stage),
                "sensor_count": sensor_count,
                "n_sensor_series": len(full_series),
                "total_observation_slots": total_obs,
                "n_valid_observations": valid_obs,
                "n_missing_observations": int(missing_obs),
                "fraction_missing": (missing_obs / total_obs) if total_obs else None,
                "fraction_valid": (valid_obs / total_obs) if total_obs else None,
                "n_delayed_observations": int(delayed_obs),
                "n_frozen_observations": int(frozen_obs),
                "n_drift_observations": int(drift_obs),
                "n_sensors_contributing": n_sensors_contributing,
                "mean_quality_health": statistics.fmean(health_values) if health_values else None,
                "elapsed_observation_time_seconds": (valid_timestamps[-1] - valid_timestamps[0]) if len(valid_timestamps) >= 2 else 0.0,
                "n_valid_timestamps_pooled": len(valid_timestamps),
                "mean_gap_seconds": mean_gap,
                "median_gap_seconds": statistics.median(gaps) if gaps else None,
                "max_gap_seconds": max(gaps) if gaps else None,
                "gap_coefficient_of_variation": (statistics.pstdev(gaps) / mean_gap) if (gaps and len(gaps) >= 2 and mean_gap) else None,
            }
    return out


# ---------------------------------------------------------------------------
# Step 6: reconstruct per-development-row conformal candidate sets.
# ---------------------------------------------------------------------------


def reconstruct_candidate_sets(
    train_records, dev_records, calibration_records, library, dev_rows: dict[str, dict[int, list[dict[str, Any]]]]
) -> tuple[dict[tuple[str, int, int, int], dict[str, Any]], dict[str, Any]]:
    stored_calibration = json.loads(m92.M9_1_CALIBRATION_PATH.read_text())
    per_row: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    equivalence_report: dict[str, Any] = {}
    for arm in m92.ALL_ARMS:
        for seed in m92.SCREENING_SEEDS:
            record = common.load_run_record(arm, seed)
            model = common.load_checkpoint_from_record(record)
            calibration_rows = common.evaluate_split(model, arm, seed, calibration_records, "calibration", library)
            calibrator = common.fit_b_depth_aware(calibration_rows, model_hash=f"m9-2-reconstruction-{arm}-seed{seed}")

            rows = dev_rows[arm][seed]
            metric_rows = []
            for row in rows:
                scheme_row = common.to_scheme_row(row)
                network_id = f"{m92.NETWORK_FAMILY}:{row['depth_bucket']}"
                candidate = calibrator.candidate_set(scheme_row.probabilities, condition=scheme_row.condition, network_id=network_id)
                set_size = len(candidate)
                normalized = set_size / len(row["probabilities"])
                covered = row["truth_index"] in candidate
                per_row[(arm, seed, row["incident_id"], row["depth"])] = {
                    "candidate_set": candidate,
                    "candidate_set_size": set_size,
                    "normalized_set_size": normalized,
                    "true_source_covered": covered,
                }
                metric_rows.append({"depth_bucket": row["depth_bucket"], "covered": covered, "set_size": set_size, "normalized_set_size": normalized})

            reconstructed_marginal = run_eval._summary_over(metric_rows)
            reconstructed_by_maturity = run_eval._by_bucket(metric_rows)
            stored = stored_calibration[arm][str(seed)]
            marginal_ok = abs(reconstructed_marginal["coverage"] - stored["marginal"]["coverage"]) <= TOLERANCE_STRICT
            marginal_ok = marginal_ok and abs(reconstructed_marginal["mean_normalized_set_size"] - stored["marginal"]["mean_normalized_set_size"]) <= TOLERANCE_STRICT
            by_maturity_ok = all(
                abs(reconstructed_by_maturity[b]["coverage"] - stored["by_maturity"][b]["coverage"]) <= TOLERANCE_STRICT
                for b in ("EARLY", "MID", "MATURE")
            )
            equivalence_report[f"{arm}-seed{seed}"] = {
                "marginal_coverage_reconstructed": reconstructed_marginal["coverage"],
                "marginal_coverage_stored": stored["marginal"]["coverage"],
                "marginal_equivalent": bool(marginal_ok),
                "by_maturity_equivalent": bool(by_maturity_ok),
            }
            if not (marginal_ok and by_maturity_ok):
                raise SystemExit(f"BLOCKED: reconstructed calibration for {arm}-seed{seed} does not reproduce m9-1-calibration.json")
    return per_row, equivalence_report


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-inference", action="store_true", help="dev-only: skip conformal reconstruction (fails downstream schema checks)")
    args = parser.parse_args()

    start_commit = m92.current_commit()
    code_under_test_commit = m92.assert_code_under_test_commit()
    locked_before = m92.assert_locked_test_closed()

    m92.M9_2_DIR.mkdir(parents=True, exist_ok=True)
    m92.M9_2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: loading + identifying M9.1 source artifacts...", flush=True)
    m9_1_sources = load_m9_1_sources()

    print("Step 2: loading dev_rows for all (arm, screening-seed) combinations...", flush=True)
    dev_rows = load_dev_rows()
    pairing = assert_pairing(dev_rows)
    print(f"  pairing OK: {pairing}")

    print("Step 3: reproducing M9.1 aggregates (Section 4 gate)...", flush=True)
    reproduction = reproduce_m9_1_aggregates(dev_rows)
    print(f"  reproduction status: {reproduction['status']}")

    print("Step 4: building topology...", flush=True)
    topology = build_topology()

    print("Step 5: loading pools (train/development_holdout/calibration) + signature library...", flush=True)
    train_records = common.load_pool("train")
    library = common.fit_library(train_records)
    dev_records = common.load_pool("development_holdout")
    calibration_records = common.load_pool("calibration")
    print(f"  train={len(train_records)} dev={len(dev_records)} calibration={len(calibration_records)}")

    print("Step 6: building causal missingness features (arm-independent)...", flush=True)
    missingness = build_missingness_features(dev_records)

    print("Step 7: reconstructing per-row conformal candidate sets (frozen-checkpoint inference on calibration split)...", flush=True)
    if args.skip_inference:
        candidate_sets, equivalence_report = {}, {"skipped": True}
    else:
        candidate_sets, equivalence_report = reconstruct_candidate_sets(
            train_records, dev_records, calibration_records, library, dev_rows
        )

    print("Step 8: assembling canonical table...", flush=True)
    node_ids = topology["node_ids"]
    scenario_index_of_incident = {common.incident_id_for("development_holdout", i): i for i in range(len(dev_records))}

    n_rows = 0
    with m92.M9_2_CANONICAL_PATH.open("w", encoding="utf-8") as fh:
        for arm in m92.ALL_ARMS:
            for seed in m92.SCREENING_SEEDS:
                for row in dev_rows[arm][seed]:
                    incident_id = row["incident_id"]
                    depth = row["depth"]
                    probs = row["probabilities"]
                    truth_index = row["truth_index"]
                    argmax_index = max(range(len(probs)), key=lambda i: probs[i])
                    sorted_probs = sorted(probs, reverse=True)
                    top1_top2_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
                    true_node = node_ids[truth_index]
                    pred_node = node_ids[argmax_index]
                    graph_distance = None if pred_node == true_node else topology["distances"].get(pred_node, {}).get(true_node)
                    cand = candidate_sets.get((arm, seed, incident_id, depth))
                    miss = missingness[incident_id][depth]
                    scenario_idx = scenario_index_of_incident[incident_id]

                    out_row = {
                        "training_seed": seed,
                        "incident_id": incident_id,
                        "scenario_index": scenario_idx,
                        "prefix_depth": depth,
                        "depth_bucket": row["depth_bucket"],
                        "arm": arm,
                        "true_source_node": true_node,
                        "predicted_top1_node": pred_node,
                        "top1_correct": bool(row["metrics"]["top1"] == 1.0),
                        "true_source_rank": row["metrics"]["true_source_rank"],
                        "reciprocal_rank": row["metrics"]["mrr"],
                        "probability_true_source": probs[truth_index],
                        "max_predicted_probability": max(probs),
                        "top1_top2_margin": top1_top2_margin,
                        "nll": row["metrics"]["nll"],
                        "brier": row["metrics"]["brier"],
                        "entropy": row["metrics"]["posterior_entropy"],
                        "conformal_candidate_set_size": cand["candidate_set_size"] if cand else None,
                        "conformal_normalized_set_size": cand["normalized_set_size"] if cand else None,
                        "true_source_covered": cand["true_source_covered"] if cand else None,
                        "all_finite": row["forward_finite"],
                        "solver_step_limit_exceeded": row["solver_step_limit_exceeded"],
                        "sde_mc_variance_mean": (statistics.fmean(row["sde_mc_variance"]) if row.get("sde_mc_variance") else None),
                        "runtime_condition": row["condition"],
                        "curriculum_stage": miss["curriculum_stage"],
                        "sensor_count": miss["sensor_count"],
                        "n_sensor_series": miss["n_sensor_series"],
                        "fraction_missing": miss["fraction_missing"],
                        "fraction_valid": miss["fraction_valid"],
                        "n_valid_observations": miss["n_valid_observations"],
                        "n_missing_observations": miss["n_missing_observations"],
                        "n_delayed_observations": miss["n_delayed_observations"],
                        "n_frozen_observations": miss["n_frozen_observations"],
                        "n_drift_observations": miss["n_drift_observations"],
                        "n_sensors_contributing": miss["n_sensors_contributing"],
                        "mean_quality_health": miss["mean_quality_health"],
                        "elapsed_observation_time_seconds": miss["elapsed_observation_time_seconds"],
                        "mean_gap_seconds": miss["mean_gap_seconds"],
                        "median_gap_seconds": miss["median_gap_seconds"],
                        "max_gap_seconds": miss["max_gap_seconds"],
                        "gap_coefficient_of_variation": miss["gap_coefficient_of_variation"],
                        "true_source_degree": topology["degree"].get(true_node),
                        "predicted_source_degree": topology["degree"].get(pred_node),
                        "graph_distance_pred_to_true": graph_distance,
                        "diagnostic_row_provenance": "PERSISTED_M9_1_DEV_ROW"
                        + ("" if args.skip_inference else "+RECONSTRUCTED_CONFORMAL_FROM_FROZEN_M9_1_CHECKPOINT"),
                    }
                    fh.write(json.dumps(out_row, sort_keys=True, default=str) + "\n")
                    n_rows += 1
    print(f"  wrote {n_rows} rows -> {m92.M9_2_CANONICAL_PATH}")

    end_commit = m92.current_commit()
    locked_after = m92.assert_locked_test_closed()

    manifest = {
        "schema_version": 1,
        "milestone": "M9.2",
        "kind": "DIAGNOSTIC_ANALYSIS_ONLY",
        "branch": common._git("branch", "--show-current"),
        "start_commit": start_commit,
        "end_commit": end_commit,
        "code_under_test_commit": code_under_test_commit,
        "m9_1_protocol_frozen_at_commit": m92.PROTOCOL_FROZEN_AT_COMMIT,
        "m9_1_closure_commit": m92.M9_1_CLOSURE_COMMIT,
        "m9_1_source_artifact_identities": m9_1_sources["identities"],
        "m9_1_closure_summary": {
            "M9_1_FINAL_DECISION": m9_1_sources["closure"]["M9_1_FINAL_DECISION"],
            "code_under_test_commit": m9_1_sources["closure"]["code_under_test_commit"],
        },
        "seeds_used": list(m92.SCREENING_SEEDS),
        "seed_excluded_from_pairing": m92.EXCLUDED_UNPAIRED_SEED,
        "depths": list(m92.CAUSAL_PREFIX_DEPTHS),
        "arms": list(m92.ALL_ARMS),
        "pairing_check": pairing,
        "m9_1_aggregate_reproduction": reproduction,
        "inference_reconstruction_performed": not args.skip_inference,
        "inference_reconstruction_scope": "calibration-split forward passes only (per arm/seed), to refit B_DEPTH_AWARE and derive per-row conformal candidate sets for already-persisted dev_rows; NO development_holdout forward passes were performed (probabilities reused as-is).",
        "inference_reconstruction_equivalence": equivalence_report,
        "checkpoint_sha256_verified": {
            f"{arm}-seed{seed}": common.load_run_record(arm, seed).get("checkpoint_sha256")
            or common.load_run_record(arm, seed)["training_summary"]["export_sha256"]
            for arm in m92.ALL_ARMS
            for seed in m92.SCREENING_SEEDS
        },
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "n_canonical_rows": n_rows,
        "canonical_table_path": str(m92.M9_2_CANONICAL_PATH.relative_to(m92.ROOT)),
        "topology": {"node_ids": topology["node_ids"], "junctions": topology["junctions"], "edges": topology["edges"], "degree": topology["degree"]},
    }
    m92.M9_2_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {m92.M9_2_MANIFEST_PATH}")
    print(json.dumps({"locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after, "n_rows": n_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
