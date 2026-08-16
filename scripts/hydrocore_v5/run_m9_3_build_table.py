"""Milestone 9.3: build the canonical per-example calibration diagnostic
table explaining why the STEP_MATCHED_INTERLEAVED_MULTI_FAMILY predictor
(ARM_B2, frozen M9.0a checkpoints) fails the >=0.85 known-family conformal
coverage floor despite its validated unseen-topology localization gain.

DIAGNOSTIC / ANALYSIS-ONLY. Trains nothing, tunes nothing, modifies no
checkpoint, never changes alpha (0.1), never opens
locked_final_test/locked_topology_test.

Reuses the EXACT frozen M9.0a/M9.0b machinery as a library (imported, not
reimplemented): `run_m9_0a_evaluate._evaluate_on_family` for known-family
development rows (both ARM_A and ARM_B2), `run_m9_0b_calibration_schemes`
for the four frozen Mondrian grouping schemes, `run_m7_topology` for
scenario-pool/inference primitives. Calibration-split rows (which M9.0a/
M9.0b never persisted with incident identity) are rebuilt with the SAME
scenario-to-example/inference call sequence, only additionally retaining
`incident_id` (`scenario.manifest.seed`) and node identity -- no new
scientific computation, same forward pass, same score formula.

Unseen-topology development rows are NOT re-inferred: they are read
directly from the already-persisted `m9-0a-topology-generalization.json`
(`per_incident_rows`), which already carries the CURRENT_FAMILY_DEPTH-style
(`family:depth_bucket`) calibration fields computed by M9.0a's own
`_postprocess_rows`. Diagnostic only (Section 4 of the M9.3 brief).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import m9_3_common as m93  # noqa: E402
import m9_0b_calibration_schemes as schemes  # noqa: E402
import run_m7_topology as m7  # noqa: E402
import run_m9_0a_evaluate as m9_0a_eval  # noqa: E402
import run_m9_0b_evaluate as m9_0b_eval  # noqa: E402
from run_m9_0_arm_b import FEATURE_KWARGS  # noqa: E402
from run_m3_calibration import DEPTH_BUCKET_OF  # noqa: E402
from hydroswarm.calibration.conformal import classify_runtime_condition, _quantile  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402

TOLERANCE_FLOAT = 1e-6


# ---------------------------------------------------------------------------
# Checkpoint provenance.
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_load_checkpoints() -> dict[str, Any]:
    """Verifies every ARM_A/ARM_B2 x seed checkpoint SHA256 against its
    recorded provenance (m8-7-runs / m9-0a-runs + cross-check against
    m9-0a-results.json) BEFORE any inference, per Section 2's requirement."""

    provenance: dict[str, Any] = {"ARM_A": {}, "ARM_B2": {}}
    m9_0a_results = json.loads(m93.M9_0A_RESULTS_PATH.read_text())
    for seed in m93.SEEDS:
        record = json.loads((m93.RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
        recorded = record["checkpoint_sha256"]
        cross_check = m9_0a_results["arms"]["ARM_A"]["per_seed"][str(seed)]["checkpoint_sha256"]
        on_disk = _sha256_file(Path(record["export_path"]))
        if not (recorded == cross_check == on_disk):
            raise SystemExit(f"BLOCKED: ARM_A seed{seed} checkpoint provenance mismatch: recorded={recorded} cross_check={cross_check} on_disk={on_disk}")
        provenance["ARM_A"][str(seed)] = {"export_path": record["export_path"], "sha256": recorded}

        record_b2 = json.loads((m93.RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        recorded_b2 = record_b2["training_summary"]["export_sha256"]
        cross_check_b2 = m9_0a_results["arms"]["ARM_B2"]["per_seed"][str(seed)]["checkpoint_sha256"]
        on_disk_b2 = _sha256_file(Path(record_b2["training_summary"]["export_path"]))
        if not (recorded_b2 == cross_check_b2 == on_disk_b2):
            raise SystemExit(f"BLOCKED: ARM_B2 seed{seed} checkpoint provenance mismatch: recorded={recorded_b2} cross_check={cross_check_b2} on_disk={on_disk_b2}")
        provenance["ARM_B2"][str(seed)] = {"export_path": record_b2["training_summary"]["export_path"], "sha256": recorded_b2}
    return provenance


def _load_model(export_path: str) -> HydroCore:
    from run_m8_7_arm import SHARED_MODEL_CONFIG

    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Incident-keyed row builders. Same computation as
# run_m9_0a_evaluate._evaluate_on_family / run_m9_0b_evaluate's calibration
# row builders -- retains incident_id (scenario.manifest.seed) and node
# identity, which those functions' own return types drop.
# ---------------------------------------------------------------------------


def _rank_and_score(probs: list[float], truth_index: int) -> dict[str, Any]:
    p_truth = probs[truth_index]
    max_prob = max(probs)
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    rank = sum(1 for p in probs if p > p_truth) + 1
    entropy = -sum(p * math.log2(p + 1e-12) for p in probs if p > 0) / max(1.0, math.log2(len(probs)))
    nll = -math.log(p_truth + 1e-9)
    brier = sum((p - (1.0 if i == truth_index else 0.0)) ** 2 for i, p in enumerate(probs))
    return {
        "top1_correct": bool(max_prob == p_truth and probs.index(max_prob) == truth_index),
        "true_source_rank": rank,
        "reciprocal_rank": 1.0 / rank,
        "probability_true_source": p_truth,
        "max_predicted_probability": max_prob,
        "top1_top2_margin": margin,
        "entropy": entropy,
        "nll": nll,
        "brier": brier,
        "nonconformity_score": 1.0 - p_truth,
    }


def build_calibration_rows(model: HydroCore, family: str, library: Any, calibration_pool: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in calibration_pool:
            incident_id = int(record.scenario.manifest.seed)
            for depth in CAUSAL_PREFIX_DEPTHS:
                example = scenario_to_prefix_example(
                    record.scenario, record.network, library, depth, feature_context=record.feature_context, **FEATURE_KWARGS,
                )
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                truth = int(example.targets["source_node"].item())
                full_series = build_sensor_series(record.scenario, record.feature_context)
                truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                condition = classify_runtime_condition(truncated_series)
                rows.append({
                    "split": "calibration", "family": family, "incident_id": incident_id, "prefix_depth": depth,
                    "probabilities": probs, "truth_index": truth, "condition": condition,
                })
    return rows


def build_development_rows(model: HydroCore, family: str, library: Any, eval_scenarios: list[tuple[Any, Any, Any]], *, known: bool) -> list[dict[str, Any]]:
    raw_rows = m9_0a_eval._evaluate_on_family(model, family, library, eval_scenarios, known=known)
    out = []
    for row in raw_rows:
        out.append({
            "split": "development", "family": family, "incident_id": int(row["seed"]), "prefix_depth": row["depth"],
            "depth_bucket": row["depth_bucket"], "probabilities": row["neural_probs"], "truth_index": row["truth_index"],
            "condition": row["condition"], "node_ids": row["node_ids"], "truth_node": row["truth_node"],
            "known": row["known"], "healthy_sensor_fraction": row["healthy_sensor_fraction"], "missing_rate": row["missing_rate"],
        })
    return out


# ---------------------------------------------------------------------------
# Scheme fitting + per-row application (all 4 M9.0b schemes), reused
# verbatim from m9_0b_calibration_schemes.
# ---------------------------------------------------------------------------


def _to_scheme_row(row: dict[str, Any]) -> schemes.SchemeRow:
    return schemes.SchemeRow(
        probabilities=tuple(row["probabilities"]), true_index=row["truth_index"],
        condition=row["condition"], family=row["family"], depth_bucket=row.get("depth_bucket") or DEPTH_BUCKET_OF[row["prefix_depth"]],
    )


def fit_all_schemes(calibration_rows: list[dict[str, Any]], *, model_hash: str) -> dict[str, Any]:
    scheme_rows = [_to_scheme_row(r) for r in calibration_rows]
    fitted: dict[str, Any] = {}
    for scheme in schemes.SCHEME_NAMES:
        if scheme == "HIERARCHICAL_CONSERVATIVE":
            fitted[scheme] = schemes.fit_hierarchical(scheme_rows, model_hash=model_hash)
        else:
            fitted[scheme] = schemes.fit_scheme(scheme, scheme_rows, model_hash=model_hash)
    return fitted


def apply_schemes_to_row(fitted: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    scheme_row = _to_scheme_row(row)
    out: dict[str, Any] = {}
    for scheme, calibrator in fitted.items():
        if scheme == "HIERARCHICAL_CONSERVATIVE":
            candidate, q_used, source, group = calibrator.candidate_set(scheme_row)
        else:
            candidate, source, group = schemes.candidate_set_for_scheme(scheme, calibrator, scheme_row)
            network_id_fn = schemes.NETWORK_ID_BUILDERS[scheme]
            _sel_source, _sel_group, sel_scores = calibrator.selection(condition=scheme_row.condition, network_id=network_id_fn(scheme_row))
            q_used = _quantile(sel_scores, m93.ALPHA)
        out[scheme] = {
            "candidate_set_size": len(candidate), "normalized_set_size": len(candidate) / len(row["probabilities"]),
            "covered": row["truth_index"] in candidate, "selection_source": source, "group_key": group,
            "quantile_used": q_used,
        }
    return out


def group_support_snapshot(calibrator: Any) -> dict[str, Any]:
    groups = {}
    for key, scores in calibrator.artifact.network_scores.items():
        n = len(scores)
        groups[str(key)] = {
            "n": n, "quantile": _quantile(scores, m93.ALPHA),
            "meets_minimum_group_size": n >= m93.MINIMUM_GROUP_SIZE,
            "finite_sample_resolution": m93.finite_sample_resolution(n),
        }
    return groups


# ---------------------------------------------------------------------------
# Topology metadata (generic wntr .to_graph() -- no invented graph, no
# invented distance definition).
# ---------------------------------------------------------------------------


def build_family_topology(family: str, loader) -> dict[str, Any]:
    network = loader()
    graph = network.to_graph().to_undirected()
    junctions = sorted(network.junction_name_list)
    degree = dict(graph.degree())
    return {"family": family, "n_nodes": graph.number_of_nodes(), "n_junctions": len(junctions), "junctions": junctions, "degree": degree}


# ---------------------------------------------------------------------------
# Reproduction gate (Section 5).
# ---------------------------------------------------------------------------


def reproduce_m9_0a_m9_0b(canonical_by_arm_seed_family: dict[str, Any]) -> dict[str, Any]:
    m9_0a_calibration = json.loads(m93.M9_0A_CALIBRATION_PATH.read_text())
    m9_0b_results = json.loads(m93.M9_0B_RESULTS_PATH.read_text())
    mismatches: list[dict[str, Any]] = []

    # M9.0b CURRENT_FAMILY_DEPTH scheme, ARM_B2, marginal coverage over
    # known (trained) families -- must match m9-0b-results.json exactly.
    reproduced_m9_0b: dict[str, Any] = {}
    for seed in m93.SEEDS:
        rows = canonical_by_arm_seed_family["ARM_B2"][seed]
        dev_rows = [r for r in rows if r["split"] == "development" and r["topology_family"] in m93.KNOWN_FAMILIES]
        n = len(dev_rows)
        covered = sum(1 for r in dev_rows if r["schemes"]["CURRENT_FAMILY_DEPTH"]["covered"])
        reproduced_coverage = covered / n if n else None
        stored = m9_0b_results["schemes"]["CURRENT_FAMILY_DEPTH"]["per_seed"][str(seed)]["overall"]["marginal_coverage"]
        reproduced_m9_0b[str(seed)] = {"reproduced": reproduced_coverage, "stored": stored, "n": n}
        if reproduced_coverage is None or abs(reproduced_coverage - stored) > TOLERANCE_FLOAT:
            mismatches.append({"gate": "M9_0B_CURRENT_FAMILY_DEPTH_MARGINAL", "seed": seed, "reproduced": reproduced_coverage, "stored": stored})

    # M9.0a known-network (golden-reference) neural Top-1 by maturity, ARM_A and ARM_B2.
    m9_0a_results = json.loads(m93.M9_0A_RESULTS_PATH.read_text())
    reproduced_m9_0a: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        reproduced_m9_0a[arm] = {}
        for seed in m93.SEEDS:
            rows = [r for r in canonical_by_arm_seed_family[arm][seed] if r["split"] == "development" and r["topology_family"] == "golden-reference"]
            for bucket, depths in (("EARLY", m93.EARLY_DEPTHS), ("MID", m93.MID_DEPTHS), ("MATURE", m93.MATURE_DEPTHS)):
                subset = [r for r in rows if r["prefix_depth"] in depths]
                if not subset:
                    continue
                reproduced_top1 = statistics.fmean(r["top1_correct"] for r in subset)
                stored = m9_0a_results["arms"][arm]["known_network_localization"][str(seed)][bucket]["neural"]["top1"]
                reproduced_m9_0a.setdefault(arm, {}).setdefault(str(seed), {})[bucket] = {"reproduced": reproduced_top1, "stored": stored}
                if abs(reproduced_top1 - stored) > TOLERANCE_FLOAT:
                    mismatches.append({"gate": "M9_0A_KNOWN_NETWORK_TOP1", "arm": arm, "seed": seed, "bucket": bucket, "reproduced": reproduced_top1, "stored": stored})

    status = "REPRODUCTION_FAILED" if mismatches else "EXACT_OR_WITHIN_DECLARED_FLOAT_TOLERANCE"
    return {
        "M9_0A_REPRODUCTION": "FAILED" if mismatches else "EXACT_OR_WITHIN_DECLARED_FLOAT_TOLERANCE",
        "M9_0B_REPRODUCTION": "FAILED" if mismatches else "EXACT_OR_WITHIN_DECLARED_FLOAT_TOLERANCE",
        "status": status,
        "tolerance_float": TOLERANCE_FLOAT,
        "mismatches": mismatches,
        "reproduced_m9_0b_current_family_depth_marginal": reproduced_m9_0b,
        "reproduced_m9_0a_known_network_top1": reproduced_m9_0a,
    }


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    start_commit = m93.current_commit()
    branch = m93.current_branch()
    if branch != m93.FROZEN_BRANCH:
        raise SystemExit(f"BLOCKED: must execute on {m93.FROZEN_BRANCH!r}, got {branch!r}")
    locked_before = m93.assert_locked_test_closed()

    m93.M9_3_DIR.mkdir(parents=True, exist_ok=True)
    m93.M9_3_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: verifying checkpoint provenance (SHA256, both arms, all 3 seeds)...", flush=True)
    provenance = verify_and_load_checkpoints()

    print("Step 2: building known-family signature libraries...", flush=True)
    arm_b2_libraries = m9_0b_eval._build_known_family_libraries()
    arm_a_golden_library = fit_pool_signature_library(build_scenario_pool("train", network_loader=build_wntr_network))

    print("Step 3: building calibration + development scenario pools (shared across seeds)...", flush=True)
    calibration_pools_b2 = {family: m9_0b_eval._calibration_pool_for(family, loader) for family, loader in m7.TRAINED_FAMILIES}
    arm_a_calibration_pool = build_scenario_pool("calibration", network_loader=build_wntr_network)
    eval_scenarios = {family: m7._generate_eval_scenarios(family, loader, m7.SEED_BASES[(family, "eval")]) for family, loader in m7.TRAINED_FAMILIES}

    print("Step 4: building family topology metadata (wntr .to_graph(), no invented distances)...", flush=True)
    topology_by_family = {family: build_family_topology(family, loader) for family, loader in m7.TRAINED_FAMILIES}

    m9_0a_topology = json.loads(m93.M9_0A_TOPOLOGY_PATH.read_text())

    canonical_by_arm_seed_family: dict[str, dict[int, list[dict[str, Any]]]] = {"ARM_A": {}, "ARM_B2": {}}
    all_group_support: dict[str, Any] = {"ARM_A": {}, "ARM_B2": {}}
    n_rows_written = 0

    with m93.M9_3_CANONICAL_PATH.open("w", encoding="utf-8") as fh:
        for arm, known_families, pool_map, library_map in (
            ("ARM_A", m93.ARM_A_KNOWN_FAMILIES, {"golden-reference": arm_a_calibration_pool}, {"golden-reference": arm_a_golden_library}),
            ("ARM_B2", m93.ARM_B2_KNOWN_FAMILIES, calibration_pools_b2, arm_b2_libraries),
        ):
            for seed in m93.SEEDS:
                print(f"{arm} seed {seed}: loading checkpoint + regenerating calibration/development rows...", flush=True)
                model = _load_model(provenance[arm][str(seed)]["export_path"])

                calibration_rows: list[dict[str, Any]] = []
                development_rows: list[dict[str, Any]] = []
                for family in known_families:
                    calibration_rows.extend(build_calibration_rows(model, family, library_map[family], pool_map[family]))
                    development_rows.extend(build_development_rows(model, family, library_map[family], eval_scenarios[family], known=True))

                fitted = fit_all_schemes(calibration_rows, model_hash=f"m9-3-{arm}-seed{seed}")
                all_group_support[arm][str(seed)] = {
                    scheme: group_support_snapshot(cal) for scheme, cal in fitted.items() if scheme != "HIERARCHICAL_CONSERVATIVE"
                }

                seed_rows: list[dict[str, Any]] = []
                for row in calibration_rows + development_rows:
                    score_block = _rank_and_score(row["probabilities"], row["truth_index"])
                    scheme_block = apply_schemes_to_row(fitted, row)
                    out_row = {
                        "predictor_arm": arm, "training_seed": seed, "split": row["split"],
                        "topology_family": row["family"], "incident_id": row["incident_id"], "prefix_depth": row["prefix_depth"],
                        "depth_bucket": row.get("depth_bucket") or DEPTH_BUCKET_OF[row["prefix_depth"]],
                        "condition": row["condition"], "known_family": row["family"] in known_families,
                        **score_block,
                        "schemes": scheme_block,
                        "topology_n_nodes": topology_by_family.get(row["family"], {}).get("n_nodes"),
                        "topology_n_junctions": topology_by_family.get(row["family"], {}).get("n_junctions"),
                    }
                    if row["split"] == "development":
                        out_row["true_source_node"] = row["truth_node"]
                        predicted_index = row["probabilities"].index(max(row["probabilities"]))
                        out_row["predicted_top1_node"] = row["node_ids"][predicted_index]
                        out_row["true_source_degree"] = topology_by_family.get(row["family"], {}).get("degree", {}).get(row["truth_node"])
                        out_row["missing_rate"] = row["missing_rate"]
                        out_row["healthy_sensor_fraction"] = row["healthy_sensor_fraction"]
                    fh.write(json.dumps(out_row, sort_keys=True, default=str) + "\n")
                    seed_rows.append(out_row)
                    n_rows_written += 1

                # Diagnostic-only unseen-topology rows, reusing ALREADY PERSISTED
                # M9.0a per-incident rows (no new inference) -- Section 4.
                for family in m93.UNSEEN_FAMILIES:
                    persisted = m9_0a_topology["arms"][arm]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"][str(seed)]
                    for row in persisted:
                        score_block = _rank_and_score(row["neural_probs"], row["truth_index"])
                        out_row = {
                            "predictor_arm": arm, "training_seed": seed, "split": "development_unseen_diagnostic",
                            "topology_family": family, "incident_id": int(row["seed"]), "prefix_depth": row["depth"],
                            "depth_bucket": row["depth_bucket"], "condition": row["condition"], "known_family": False,
                            **score_block,
                            "schemes": {
                                "CURRENT_FAMILY_DEPTH": {
                                    "candidate_set_size": row["candidate_set_size"], "covered": row["candidate_covered"],
                                    "selection_source": row["calibration_source"], "group_key": row["calibration_group_key"],
                                    "normalized_set_size": row["candidate_set_size"] / len(row["neural_probs"]), "quantile_used": None,
                                }
                            },
                            "true_source_node": row["truth_node"], "predicted_top1_node": row["node_ids"][row["neural_probs"].index(max(row["neural_probs"]))],
                            "topology_n_nodes": len(row["node_ids"]), "topology_n_junctions": None, "true_source_degree": None,
                            "missing_rate": row["missing_rate"], "healthy_sensor_fraction": row["healthy_sensor_fraction"],
                        }
                        fh.write(json.dumps(out_row, sort_keys=True, default=str) + "\n")
                        n_rows_written += 1

                canonical_by_arm_seed_family[arm][seed] = seed_rows

    print(f"  wrote {n_rows_written} rows -> {m93.M9_3_CANONICAL_PATH}")

    print("Step 5: reproducing M9.0a/M9.0b aggregates (Section 5 gate)...", flush=True)
    reproduction = reproduce_m9_0a_m9_0b(canonical_by_arm_seed_family)
    m93.M9_3_REPRODUCTION_PATH.write_text(json.dumps(reproduction, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"  reproduction status: {reproduction['status']}")
    if reproduction["status"] != "EXACT_OR_WITHIN_DECLARED_FLOAT_TOLERANCE":
        print("BLOCKED: reproduction failed, see m9-3-reproduction.json. Stopping before interpretation.")
        raise SystemExit(2)

    print("Step 6: writing group-support snapshot + manifest...", flush=True)
    m93.M9_3_SUPPORT_ANALYSIS_PATH.write_text(json.dumps({"group_support_by_scheme": all_group_support}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    end_commit = m93.current_commit()
    locked_after = m93.assert_locked_test_closed()

    manifest = {
        "schema_version": 1, "milestone": "M9.3", "kind": "DIAGNOSTIC_ANALYSIS_ONLY",
        "branch": branch, "start_commit": start_commit, "end_commit": end_commit,
        "m9_0a_protocol_frozen_commit": m93.M9_0A_PROTOCOL_FROZEN_COMMIT, "m9_0a_results_commit": m93.M9_0A_RESULTS_COMMIT,
        "m9_0b_protocol_frozen_commit": m93.M9_0B_PROTOCOL_FROZEN_COMMIT, "m9_0b_results_commit": m93.M9_0B_RESULTS_COMMIT,
        "m9_2_closure_commit": m93.M9_2_CLOSURE_COMMIT,
        "checkpoint_sha256": provenance,
        "seeds": list(m93.SEEDS), "alpha": m93.ALPHA, "known_families": list(m93.KNOWN_FAMILIES), "unseen_families_diagnostic_only": list(m93.UNSEEN_FAMILIES),
        "depths": list(m93.DEPTHS), "operational_coverage_floor": m93.OPERATIONAL_COVERAGE_FLOOR,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "m9_0a_m9_0b_reproduction_status": reproduction["status"],
        "n_canonical_rows": n_rows_written,
        "canonical_table_path": str(m93.M9_3_CANONICAL_PATH.relative_to(m93.ROOT)),
        "topology_by_family": topology_by_family,
        "no_training_performed": True, "no_predictor_modified": True,
    }
    m93.M9_3_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {m93.M9_3_MANIFEST_PATH}")
    print(json.dumps({"locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after, "n_rows": n_rows_written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
