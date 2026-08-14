"""Capability diagnostic Section 13: data-generator/runtime network parity.

For each governed topology family (golden-reference, branched-loop,
loop-grid) plus the coastal-branch unseen-dev topology, this script builds
TWO independently-constructed network objects and checks whether they are
actually the same network by hash, not just by name:

  - "scenario_network": however `hydroswarm.data.scenarios.
    WNTRScenarioGenerator`/corpus-building scripts construct this family
    today (`build_wntr_network()` for golden-reference --
    src/hydroswarm/simulation/network.py, a PROGRAMMATIC build; a plain
    `wntr.network.WaterNetworkModel(path)` file load for the other three,
    exactly matching `scripts/generate_cycle_b_corpus.py`'s
    `TRAIN_TOPOLOGIES`/`DEV_OOD_TOPOLOGY`).
  - "production_network": exactly how `hydroswarm.runtime.v4_defaults.
    V4PipelineFactory.__call__` builds its internal network (line ~269:
    `wntr.network.WaterNetworkModel(str(network_path))`), which becomes
    `pipeline.simulator.network` -- and, critically, `src/hydroswarm/
    api/app.py`'s REAL `perform_analysis` (line ~433) passes exactly
    `pipeline.simulator.network` (not a freshly-built network) as the
    `network` argument to `pipeline.analyze()` for every real incident.

For golden-reference specifically, these are TWO DIFFERENT CONSTRUCTION
PATHS for what is meant to be the same physical network -- this script
checks, empirically, whether they hash identically and, if not, what the
measured downstream effect is on the `classical_prior` model-input feature
and final localization accuracy through the REAL `pipeline.analyze()`
method (not assumed, computed).

No locked-test access: only fresh WNTRScenarioGenerator-generated scenarios
(seed family 20260813, reusing the SAME N=20 parity-scenario seeds
`scripts/capability_diagnostic/train_serve_parity_full.py` already
established for Sections 6/11-13, per the protocol doc's own framing of
"Pressure/sensor-series/network parity (Sections 11-13): ... plus a scan
over the same N=20 parity scenarios").
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import wntr  # noqa: E402

from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.classical.signature_policy import KNOWN_TRAINING_TOPOLOGY_HASHES, resolve_signature_mode  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.runtime.paths import resolve_v4_bundle_dir  # noqa: E402
from hydroswarm.runtime.v4_defaults import V4PipelineFactory  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.corpus import (  # noqa: E402
    build_feature_context,
    build_sensor_series,
    model_input_classical_prior,
    resolve_model_input_signature_library,
    scenario_to_example,
)

from generate_cycle_b_corpus import DEV_OOD_TOPOLOGY, TRAIN_TOPOLOGIES  # noqa: E402

PARITY_SEEDS = [20260813_00 + i for i in range(20)]

#: (family, scenario_network_loader, production_network_path, is_governed_training_family)
FAMILIES: tuple[tuple[str, Any, Path, bool], ...] = (
    ("golden-reference", build_wntr_network, ROOT / "data" / "frozen" / "golden_network.inp", True),
    ("branched-loop", dict(TRAIN_TOPOLOGIES)["branched-loop"], ROOT / "data" / "topology-transfer" / "branched-loop.inp", True),
    ("loop-grid", dict(TRAIN_TOPOLOGIES)["loop-grid"], ROOT / "data" / "topologies" / "loop-grid.inp", True),
    ("coastal-branch", DEV_OOD_TOPOLOGY[1], ROOT / "data" / "topologies" / "coastal-branch.inp", False),
)


def _rank_metrics(belief: dict[str, float], truth: str) -> dict[str, Any]:
    if not belief or sum(belief.values()) <= 0:
        return {"top1": None, "top3": None, "reciprocal_rank": None}
    return {
        "top1": localization_top_k(belief, truth, k=1),
        "top3": localization_top_k(belief, truth, k=3),
        "reciprocal_rank": mean_reciprocal_rank([belief], [truth]),
    }


def _family_identity(family: str, scenario_loader: Any, production_path: Path) -> dict[str, Any]:
    scenario_network = scenario_loader()
    production_network = wntr.network.WaterNetworkModel(str(production_path))

    file_sha256 = hashlib.sha256(production_path.read_bytes()).hexdigest()
    scenario_hash = network_sha256(scenario_network)
    production_hash = network_sha256(production_network)
    scenario_state_hash = HydraulicSimulator(scenario_network).state_hash()
    production_state_hash = HydraulicSimulator(production_network).state_hash()

    scenario_mode = resolve_signature_mode(scenario_hash)
    production_mode = resolve_signature_mode(production_hash)

    return {
        "family": family,
        "production_inp_path": str(production_path.relative_to(ROOT)),
        "production_inp_file_sha256": file_sha256,
        "scenario_network_sha256": scenario_hash,
        "production_network_sha256": production_hash,
        "structural_hashes_match": scenario_hash == production_hash,
        "scenario_network_state_hash": scenario_state_hash,
        "production_network_state_hash": production_state_hash,
        "state_hashes_match": scenario_state_hash == production_state_hash,
        "scenario_network_signature_mode": scenario_mode,
        "production_network_signature_mode": production_mode,
        "signature_modes_match": scenario_mode == production_mode,
        "scenario_hash_in_known_training_hashes": scenario_hash in KNOWN_TRAINING_TOPOLOGY_HASHES,
        "production_hash_in_known_training_hashes": production_hash in KNOWN_TRAINING_TOPOLOGY_HASHES,
    }, scenario_network, production_network


def _measure_downstream_effect(
    family: str, scenario_network: Any, production_network: Any, seeds: list[int], governed: bool
) -> dict[str, Any]:
    """For a family where scenario_network and production_network may
    differ, actually run the REAL frozen pipeline.analyze() both ways
    (mirroring production's real `network = pipeline.simulator.network`
    call pattern vs. every prior diagnostic script's habit of passing a
    freshly-built scenario network) and measure the real effect on the
    model-input classical_prior feature and final localization accuracy.
    """
    factory = V4PipelineFactory(resolve_v4_bundle_dir())

    # Build one pipeline per network arm since V4PipelineFactory binds its
    # internal simulator/state_estimator to whatever network_path it is
    # given -- so we build it once against the PRODUCTION path (matching
    # real deployment), then call analyze() with EACH network object as
    # the `network` argument, exactly mirroring the two different call
    # patterns found in the codebase (production's app.py vs. every prior
    # diagnostic script).
    NETWORK_PATHS = {
        "golden-reference": ROOT / "data" / "frozen" / "golden_network.inp",
        "branched-loop": ROOT / "data" / "topology-transfer" / "branched-loop.inp",
        "loop-grid": ROOT / "data" / "topologies" / "loop-grid.inp",
        "coastal-branch": ROOT / "data" / "topologies" / "coastal-branch.inp",
    }
    pipeline = factory(None, NETWORK_PATHS[family])
    context = build_feature_context(scenario_network)
    generator = WNTRScenarioGenerator()

    records: list[dict[str, Any]] = []
    for seed in seeds:
        config = ScenarioGenerationConfig(
            seed=seed, network_id=family, network_family=family,
            split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
            event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
        )
        try:
            scenario = generator.generate(scenario_network, config)
            truth = scenario.manifest.incident.source_nodes[0]
            series = build_sensor_series(scenario, context)

            # --- classical_prior comparison (the actual model-input feature) ---
            junctions_scn = tuple(sorted(scenario_network.junction_name_list))
            junctions_prod = tuple(sorted(production_network.junction_name_list))
            lib_scn, ref_ts_scn, mode_scn = resolve_model_input_signature_library(network_sha256(scenario_network), junctions_scn, scenario_network)
            lib_prod, ref_ts_prod, mode_prod = resolve_model_input_signature_library(network_sha256(production_network), junctions_prod, production_network)
            prior_scn = model_input_classical_prior(lib_scn, junctions_scn, series, ref_ts_scn)
            prior_prod = model_input_classical_prior(lib_prod, junctions_prod, series, ref_ts_prod)
            common_nodes = sorted(set(prior_scn) & set(prior_prod))
            classical_prior_max_abs_diff = max((abs(prior_scn[n] - prior_prod[n]) for n in common_nodes), default=None)

            # --- full pipeline.analyze() comparison, exactly mirroring the
            # two real call patterns in this codebase ---
            result_scenario_arm = pipeline.analyze(uuid.uuid4(), scenario_network, series)
            result_production_arm = pipeline.analyze(uuid.uuid4(), production_network, series)

            records.append({
                "seed": seed,
                "signature_mode_scenario_arm": mode_scn,
                "signature_mode_production_arm": mode_prod,
                "classical_prior_max_abs_diff": classical_prior_max_abs_diff,
                "scenario_arm": {
                    "fused": _rank_metrics(dict(result_scenario_arm.fused_belief), truth),
                    "ood_level": result_scenario_arm.ood_level.value,
                },
                "production_arm": {
                    "fused": _rank_metrics(dict(result_production_arm.fused_belief), truth),
                    "ood_level": result_production_arm.ood_level.value,
                },
            })
        except Exception as exc:  # noqa: BLE001
            records.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})

    ok = [r for r in records if "error" not in r]
    scenario_top1 = [r["scenario_arm"]["fused"]["top1"] for r in ok if r["scenario_arm"]["fused"]["top1"] is not None]
    production_top1 = [r["production_arm"]["fused"]["top1"] for r in ok if r["production_arm"]["fused"]["top1"] is not None]
    mode_mismatches = sum(1 for r in ok if r["signature_mode_scenario_arm"] != r["signature_mode_production_arm"])
    max_prior_diffs = [r["classical_prior_max_abs_diff"] for r in ok if r["classical_prior_max_abs_diff"] is not None]

    return {
        "n_scenarios": len(seeds),
        "n_ok": len(ok),
        "n_errors": len(records) - len(ok),
        "n_signature_mode_mismatches": mode_mismatches,
        "mean_classical_prior_max_abs_diff": (sum(max_prior_diffs) / len(max_prior_diffs)) if max_prior_diffs else None,
        "max_classical_prior_max_abs_diff": max(max_prior_diffs) if max_prior_diffs else None,
        "scenario_arm_top1_mean": (sum(scenario_top1) / len(scenario_top1)) if scenario_top1 else None,
        "production_arm_top1_mean": (sum(production_top1) / len(production_top1)) if production_top1 else None,
        "top1_delta_production_minus_scenario": (
            (sum(production_top1) / len(production_top1)) - (sum(scenario_top1) / len(scenario_top1))
            if scenario_top1 and production_top1 else None
        ),
        "records": records,
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed for this diagnostic"

    per_family: dict[str, Any] = {}
    findings: dict[str, Any] = {}

    for family, loader, prod_path, governed in FAMILIES:
        identity, scenario_network, production_network = _family_identity(family, loader, prod_path)
        entry: dict[str, Any] = {"identity": identity, "governed_training_family": governed}

        # scenario_to_example TopologyMetadata sanity check ((iv) in the
        # protocol): training's own construction is entirely self-consistent
        # (never touches production_network) -- confirm that directly.
        try:
            junctions = tuple(sorted(scenario_network.junction_name_list))
            scenario_hash = identity["scenario_network_sha256"]
            library, _ref_ts, mode = resolve_model_input_signature_library(scenario_hash, junctions, scenario_network)
            generator = WNTRScenarioGenerator()
            config = ScenarioGenerationConfig(
                seed=PARITY_SEEDS[0], network_id=family, network_family=family,
                split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
                event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
            )
            scenario = generator.generate(scenario_network, config)
            example = scenario_to_example(scenario, scenario_network, library)
            entry["topology_metadata_check"] = {
                "scenario_manifest_network_sha256": scenario.manifest.network_sha256,
                "matches_scenario_network_sha256": scenario.manifest.network_sha256 == scenario_hash,
                "example_topology_hash": example.topology.topology_hash if example.topology else None,
                "example_network_hash": example.topology.network_hash if example.topology else None,
                "example_topology_hash_matches_scenario_hash": (
                    example.topology.topology_hash == scenario_hash if example.topology else False
                ),
                "signature_mode_used_for_training": mode,
            }
        except Exception as exc:  # noqa: BLE001
            entry["topology_metadata_check"] = {"error": f"{type(exc).__name__}: {exc}"}

        # Downstream empirical effect: full for golden-reference (where a
        # mismatch was found by direct hash computation), a lighter N=5
        # scan for the other families to confirm no mismatch materializes.
        n_scan = 20 if not identity["structural_hashes_match"] else 5
        entry["downstream_effect"] = _measure_downstream_effect(
            family, scenario_network, production_network, PARITY_SEEDS[:n_scan], governed
        )

        status = "PASS" if identity["structural_hashes_match"] and identity["signature_modes_match"] else "FAIL"
        entry["verdict"] = status
        per_family[family] = entry

    mismatched_families = [f for f, e in per_family.items() if e["verdict"] == "FAIL"]

    if mismatched_families:
        findings["CAP-DATA-01"] = {
            "title": "Golden-reference network built programmatically for scenario generation/training diverges "
            "(by structural hash) from the .inp-file-loaded network production actually serves against, causing "
            "REAL LIVE incidents to silently resolve RUNTIME_GENERATED_IMPORTED_NETWORK signature mode instead of "
            "the governed GOVERNED_KNOWN_NETWORK mode the model was trained under.",
            "root_cause": (
                "hydroswarm.simulation.network.build_wntr_network() constructs golden-reference programmatically "
                "(used by every scenario-generation/training/most-diagnostic-script call site this session). "
                "src/hydroswarm/runtime/v4_defaults.py V4PipelineFactory.__call__ (line ~269) instead loads "
                "data/frozen/golden_network.inp via wntr.network.WaterNetworkModel(str(network_path)) -- an "
                "EPANET-format round trip. src/hydroswarm/api/app.py's real perform_analysis (line ~433) passes "
                "exactly `network = pipeline.simulator.network` (the FILE-LOADED network) into pipeline.analyze() "
                "for every real production incident -- never the programmatically-built network. The two networks' "
                "pipe length/diameter/elevation floats differ only at the ~1e-9 relative (EPANET unit round-trip) "
                "level -- not a real hydraulic difference -- but hydroswarm.data.scenarios.network_sha256() hashes "
                "exact JSON float reprs, so this negligible numerical noise produces a COMPLETELY different hash. "
                "hydroswarm.classical.signature_policy.KNOWN_TRAINING_TOPOLOGY_FAMILY_BY_HASH is keyed by the "
                "PROGRAMMATIC build's hash only, so the production-real, file-loaded network's hash is absent from "
                "it, and resolve_signature_mode() returns RUNTIME_GENERATED_IMPORTED_NETWORK for every real "
                "golden-reference incident."
            ),
            "measured_effect": {
                family: {
                    "scenario_network_sha256": per_family[family]["identity"]["scenario_network_sha256"],
                    "production_network_sha256": per_family[family]["identity"]["production_network_sha256"],
                    "scenario_signature_mode": per_family[family]["identity"]["scenario_network_signature_mode"],
                    "production_signature_mode": per_family[family]["identity"]["production_network_signature_mode"],
                    "n_signature_mode_mismatches_of_n_scenarios": (
                        f"{per_family[family]['downstream_effect']['n_signature_mode_mismatches']}/"
                        f"{per_family[family]['downstream_effect']['n_ok']}"
                    ),
                    "mean_classical_prior_max_abs_diff": per_family[family]["downstream_effect"]["mean_classical_prior_max_abs_diff"],
                    "scenario_arm_top1_mean": per_family[family]["downstream_effect"]["scenario_arm_top1_mean"],
                    "production_arm_top1_mean": per_family[family]["downstream_effect"]["production_arm_top1_mean"],
                    "top1_delta_production_vs_scenario": per_family[family]["downstream_effect"]["top1_delta_production_minus_scenario"],
                }
                for family in mismatched_families
            },
            "taxonomy": "CAP-DATA",
            "severity": "HIGH if top1_delta_production_vs_scenario is large and classical_prior_max_abs_diff is "
            "non-trivial (feeds directly into the trained model's expected input distribution); see measured_effect "
            "above for the actual numbers from this run, not an assumed severity.",
        }

    report = {
        "schema_version": 1,
        "section": "13_data_generator_runtime_network_parity",
        "locked_test_opened_before": locked_before,
        "seed_family": "20260813_00.. (same N=20 parity scenarios as Sections 6/11)",
        "families": per_family,
        "mismatched_families": mismatched_families,
        "cap_findings": findings,
        "summary": {
            family: {
                "structural_hashes_match": e["identity"]["structural_hashes_match"],
                "state_hashes_match": e["identity"]["state_hashes_match"],
                "signature_modes_match": e["identity"]["signature_modes_match"],
                "verdict": e["verdict"],
            }
            for family, e in per_family.items()
        },
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "network-parity.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if findings:
        print(json.dumps(findings, indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
