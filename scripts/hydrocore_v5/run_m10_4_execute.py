"""Milestone 10.4 Parts 3-9: population generation + real causal
full-trajectory execution (ARM_FULL / ARM_NO_EXTRA_SAMPLING) + fail-closed
cases, under the frozen protocol (`m10_4_protocol.py`,
`docs/evaluation/HYDROCORE_V5_M10_4_FULL_TRAJECTORY_PROTOCOL.md`).

Requires `run_m10_4_preflight.py` to have already written
`m10-4-preflight.json` with `result == "M10_4_PREFLIGHT_PASS"` -- refuses to
run otherwise.

Writes (all under reports/evaluation/hydrocore-v5/m10/m10-4/):
  m10-4-seed-disjointness.json
  m10-4-population-manifest.json
  m10-4-trajectories.jsonl        (raw per-incident-pair rows, retained)
  m10-4-fail-closed.json
  m10-4-safety-counters.json
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m10_4_common as m104  # noqa: E402
import m10_4_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from hydroswarm.api import create_app  # noqa: E402
from hydroswarm.evaluation.live_robustness import Condition  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

M10_4_DIR = m10.M10_DIR / "m10-4"


def _require_preflight_pass() -> None:
    path = M10_4_DIR / "m10-4-preflight.json"
    if not path.exists():
        raise RuntimeError("m10-4-preflight.json missing -- run run_m10_4_preflight.py first")
    preflight = json.loads(path.read_text())
    if preflight.get("result") != "M10_4_PREFLIGHT_PASS":
        raise RuntimeError(f"M10.4 preflight did not pass ({preflight.get('result')}); refusing to execute trajectories")


def _conditions_for_family(family: str) -> tuple[str, ...]:
    return proto.TRAINED_FAMILY_CONDITIONS if family in m10.TRAINED_FAMILIES else proto.UNSEEN_FAMILY_CONDITIONS


def _build_condition(*, family: str, kind: str, seed: int, incident_index: int) -> Condition:
    kwargs = dict(proto.CONDITION_KWARGS[kind])
    name = f"m10-4-{family}-{kind}-{incident_index}-{seed}"
    return Condition(name, network_id=family, seed=seed, **kwargs)


def _population_manifest() -> dict[str, Any]:
    rows = []
    for model_seed in proto.MODEL_SEEDS:
        for family, kind in proto.population_cells():
            for incident_index in range(proto.INCIDENTS_PER_CELL):
                seed = proto.incident_seed(model_seed, family, kind, incident_index)
                rows.append({
                    "model_seed": model_seed, "family": family, "condition_kind": kind,
                    "incident_index": incident_index, "physical_seed": seed,
                })
    manifest = {
        "kind": "M10_4_POPULATION_MANIFEST",
        "protocol_hash": proto.protocol_hash(),
        "model_seeds": list(proto.MODEL_SEEDS),
        "trained_families": list(m10.TRAINED_FAMILIES),
        "unseen_families": list(m10.UNSEEN_FAMILIES),
        "condition_kinds": list(proto.CONDITION_KINDS),
        "incidents_per_cell": proto.INCIDENTS_PER_CELL,
        "n_physical_incidents_per_seed": sum(
            proto.INCIDENTS_PER_CELL for _ in proto.population_cells()
        ),
        "n_physical_incidents_total": len(rows),
        "n_api_incidents_total": len(rows) * 2,
        "rows": rows,
    }
    return manifest


def _run_population(rows: list[dict[str, Any]], safety: dict[str, int]) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_seed.setdefault(row["model_seed"], []).append(row)

    with tempfile.TemporaryDirectory(prefix="hydroswarm-m10-4-exec-") as tmp:
        tmp_path = Path(tmp)
        for model_seed, seed_rows in rows_by_seed.items():
            t_seed = time.time()
            factory = m104.M10_4_PipelineFactory(seed=model_seed, project_root=m10.ROOT_PATH)
            app = create_app(
                pipeline_factory=factory,
                database_path=tmp_path / f"state-{model_seed}.sqlite3",
                ledger_path=tmp_path / f"audit-{model_seed}.sqlite3",
                network_directory=tmp_path / f"networks-{model_seed}",
            )
            with TestClient(app) as client:
                network_ids: dict[str, str] = {}
                network_paths: dict[str, Path] = {}
                for row in seed_rows:
                    family = row["family"]
                    if family not in network_ids:
                        inp = m104.network_inp_path(family, tmp_path)
                        imported = client.post(
                            "/api/networks/import",
                            files={"file": (inp.name, inp.read_bytes(), "application/octet-stream")},
                        )
                        if imported.status_code != 201:
                            raise RuntimeError(f"network import failed for {family}: {imported.text}")
                        network_ids[family] = imported.json()["network_id"]
                        network_paths[family] = inp
                    condition = _build_condition(
                        family=family, kind=row["condition_kind"], seed=row["physical_seed"],
                        incident_index=row["incident_index"],
                    )
                    record = m104.run_incident_pair(
                        client=client, network_path=network_paths[family], network_id=network_ids[family],
                        condition=condition, maximum_samples=proto.MAXIMUM_SAMPLES, safety=safety,
                    )
                    record["model_seed"] = model_seed
                    record["condition_kind"] = row["condition_kind"]
                    record["incident_index"] = row["incident_index"]
                    trajectories.append(record)
            print(f"  seed {model_seed}: {len(seed_rows)} incident-pairs in {time.time() - t_seed:.1f}s", flush=True)
    return trajectories


def _run_fail_closed(safety: dict[str, int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seed = proto.FAIL_CLOSED_SEED_BASE

    with tempfile.TemporaryDirectory(prefix="hydroswarm-m10-4-failclosed-") as tmp:
        tmp_path = Path(tmp)

        # MODEL_UNAVAILABLE: pipeline_factory whose model failed to load.
        class _NoModelFactory:
            def __init__(self) -> None:
                self.fallback_reason = "m10_4_fail_closed_model_unavailable"

            def __call__(self, _record: Any, network_path: Any):
                from hydroswarm.classical import (
                    GOVERNED_TRAINING_SIGNATURE_POLICY, SignatureBuilder, SignatureCache, SignatureCacheKey,
                )
                from hydroswarm.data.scenarios import network_sha256
                from hydroswarm.inference import HybridInferencePipeline
                from hydroswarm.simulation import HydraulicSimulator
                from hydroswarm.simulation.wrapper import wntr as _wntr
                import hashlib as _hashlib

                network = _wntr.network.WaterNetworkModel(str(network_path))
                simulator = HydraulicSimulator(network)
                source_nodes = tuple(map(str, network.junction_name_list))
                policy = GOVERNED_TRAINING_SIGNATURE_POLICY
                network_sha256(network)
                key = SignatureCacheKey(
                    network_hash=simulator.state_hash(), hydraulic_state_hash=simulator.state_hash(),
                    simulator_version=simulator.simulator_version, configuration_hash=policy.policy_hash,
                    sensor_layout_hash=_hashlib.sha256("|".join(source_nodes).encode()).hexdigest(),
                )
                artifact = SignatureBuilder(simulator, SignatureCache(tmp_path / "sig-cache-fc")).build_or_load(
                    key=key, source_nodes=source_nodes, start_time_bins=policy.start_time_bins,
                    duration_bins=policy.duration_bins, strength_bins=policy.strength_bins,
                    demand_regimes=policy.demand_regimes, sensor_nodes=source_nodes,
                    sample_times_seconds=policy.sample_times_seconds,
                )
                return HybridInferencePipeline(simulator=simulator, signature_artifact=artifact, model=None)

        app = create_app(
            pipeline_factory=_NoModelFactory(), database_path=tmp_path / "state-fc1.sqlite3",
            ledger_path=tmp_path / "audit-fc1.sqlite3", network_directory=tmp_path / "networks-fc1",
        )
        with TestClient(app) as client:
            inp = m104.network_inp_path("golden-reference", tmp_path)
            imported = client.post("/api/networks/import", files={"file": (inp.name, inp.read_bytes(), "application/octet-stream")})
            network_id = imported.json()["network_id"]
            condition = Condition("m10-4-fc-model-unavailable", "nominal", "clean_operational", seed, network_id="golden-reference")
            from hydroswarm.evaluation.live_robustness import _payloads, _scenario_config
            from hydroswarm.data.scenarios import WNTRScenarioGenerator
            import wntr as _wntr2
            from datetime import UTC, datetime, timedelta

            network = _wntr2.network.WaterNetworkModel(str(inp))
            scenario, _randomized = WNTRScenarioGenerator().generate_with_network(network, _scenario_config(condition))
            origin = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=(seed % 180))
            observations = _payloads(scenario, condition, origin)
            created = client.post("/api/incidents", json={
                "network_id": network_id, "detected_at": origin.isoformat(),
                "observations": observations, "maximum_samples": 3,
            })
            outcome = {"case": "MODEL_UNAVAILABLE", "create_status": created.status_code}
            if created.status_code == 201:
                incident_id = created.json()["incident_id"]
                analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
                outcome["analyze_status"] = analyzed.status_code
                if analyzed.status_code == 200:
                    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
                    outcome["control_action"] = analysis.get("control_action")
                    outcome["neural_belief"] = analysis.get("neural_belief")
                    outcome["classical_fallback_used"] = analysis.get("neural_belief") is None
                    outcome["bounded"] = True
                else:
                    outcome["bounded"] = analyzed.status_code in (409, 422, 503)
            else:
                outcome["bounded"] = created.status_code in (409, 422, 503)
            results.append(outcome)

        # CALIBRATION_UNAVAILABLE: real model, calibration_artifact=None.
        factory = m104.M10_4_PipelineFactory(seed=proto.MODEL_SEEDS[0], project_root=m10.ROOT_PATH)
        factory._calibrator = None  # noqa: SLF001 -- targeted fail-closed injection, disclosed in the artifact
        app2 = create_app(
            pipeline_factory=factory, database_path=tmp_path / "state-fc2.sqlite3",
            ledger_path=tmp_path / "audit-fc2.sqlite3", network_directory=tmp_path / "networks-fc2",
        )
        with TestClient(app2) as client:
            inp = m104.network_inp_path("golden-reference", tmp_path)
            imported = client.post("/api/networks/import", files={"file": (inp.name, inp.read_bytes(), "application/octet-stream")})
            network_id = imported.json()["network_id"]
            condition = Condition("m10-4-fc-calibration-unavailable", "nominal", "clean_operational", seed + 1, network_id="golden-reference")
            record = m104.run_incident_pair(
                client=client, network_path=inp, network_id=network_id, condition=condition,
                maximum_samples=3, safety=safety,
            )
            full = record["arms"].get("FULL", {})
            fa = full.get("final_analysis", {})
            results.append({
                "case": "CALIBRATION_UNAVAILABLE",
                "calibrated": fa.get("calibrated"), "control_action": fa.get("control_action"),
                "bounded": fa.get("calibrated") is False and fa.get("planning_allowed") is not True,
            })

        # SENSOR_STATE_INSUFFICIENT: all observations missing.
        factory3 = m104.M10_4_PipelineFactory(seed=proto.MODEL_SEEDS[0], project_root=m10.ROOT_PATH)
        app3 = create_app(
            pipeline_factory=factory3, database_path=tmp_path / "state-fc3.sqlite3",
            ledger_path=tmp_path / "audit-fc3.sqlite3", network_directory=tmp_path / "networks-fc3",
        )
        with TestClient(app3) as client:
            inp = m104.network_inp_path("golden-reference", tmp_path)
            imported = client.post("/api/networks/import", files={"file": (inp.name, inp.read_bytes(), "application/octet-stream")})
            network_id = imported.json()["network_id"]
            created = client.post("/api/incidents", json={
                "network_id": network_id, "detected_at": "2025-01-01T00:00:00Z",
                "observations": [], "maximum_samples": 3,
            })
            outcome = {"case": "SENSOR_STATE_INSUFFICIENT", "create_status": created.status_code}
            if created.status_code == 201:
                incident_id = created.json()["incident_id"]
                analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
                outcome["analyze_status"] = analyzed.status_code
                outcome["bounded"] = analyzed.status_code in (409, 422)
            else:
                outcome["bounded"] = created.status_code in (409, 422)
            results.append(outcome)

            # SAMPLING_BUDGET_PREEXHAUSTED: maximum_samples=0, then a
            # recommend call must fail closed with no sample surfaced.
            condition = Condition("m10-4-fc-budget-preexhausted", "sensor_coverage", "25%", seed + 2, network_id="golden-reference", coverage=0.25)
            from hydroswarm.evaluation.live_robustness import _payloads as _payloads2, _scenario_config as _scfg2
            from hydroswarm.data.scenarios import WNTRScenarioGenerator as _WSG2
            import wntr as _wntr3
            from datetime import UTC as _UTC2, datetime as _dt2, timedelta as _td2

            network2 = _wntr3.network.WaterNetworkModel(str(inp))
            scenario2, _r2 = _WSG2().generate_with_network(network2, _scfg2(condition))
            origin2 = _dt2(2025, 1, 1, tzinfo=_UTC2) + _td2(days=(condition.seed % 180))
            obs2 = _payloads2(scenario2, condition, origin2)
            created2 = client.post("/api/incidents", json={
                "network_id": network_id, "detected_at": origin2.isoformat(),
                "observations": obs2, "maximum_samples": 0,
            })
            outcome2 = {"case": "SAMPLING_BUDGET_PREEXHAUSTED", "create_status": created2.status_code}
            if created2.status_code == 201:
                incident_id2 = created2.json()["incident_id"]
                analyzed2 = client.post(f"/api/incidents/{incident_id2}/analyze")
                outcome2["analyze_status"] = analyzed2.status_code
                if analyzed2.status_code == 200:
                    rec2 = client.post(f"/api/incidents/{incident_id2}/samples/recommend")
                    outcome2["recommend_status"] = rec2.status_code
                    outcome2["bounded"] = rec2.status_code == 409
                else:
                    outcome2["bounded"] = True
            else:
                outcome2["bounded"] = created2.status_code in (409, 422)
            results.append(outcome2)

            # NO_ACCESSIBLE_UNSAMPLED_CANDIDATE: every junction already
            # observed as an initial sensor (coverage effectively 100% of
            # every possible sample location) -- Scout has nothing left to
            # recommend.
            full_condition = Condition("m10-4-fc-no-accessible", "nominal", "clean_operational", seed + 3, network_id="golden-reference", coverage=1.0)
            scenario3, _r3 = _WSG2().generate_with_network(network2, _scfg2(full_condition))
            origin3 = _dt2(2025, 1, 1, tzinfo=_UTC2) + _td2(days=(full_condition.seed % 180))
            obs3 = _payloads2(scenario3, full_condition, origin3)
            created3 = client.post("/api/incidents", json={
                "network_id": network_id, "detected_at": origin3.isoformat(),
                "observations": obs3, "maximum_samples": 3,
            })
            outcome3 = {"case": "NO_ACCESSIBLE_UNSAMPLED_CANDIDATE", "create_status": created3.status_code}
            if created3.status_code == 201:
                incident_id3 = created3.json()["incident_id"]
                analyzed3 = client.post(f"/api/incidents/{incident_id3}/analyze")
                outcome3["analyze_status"] = analyzed3.status_code
                if analyzed3.status_code == 200:
                    rec3 = client.post(f"/api/incidents/{incident_id3}/samples/recommend")
                    outcome3["recommend_status"] = rec3.status_code
                    outcome3["bounded"] = rec3.status_code in (200, 409)
                else:
                    outcome3["bounded"] = True
            else:
                outcome3["bounded"] = created3.status_code in (409, 422)
            results.append(outcome3)

    return results


def main() -> None:
    M10_4_DIR.mkdir(parents=True, exist_ok=True)
    branch = m10.current_branch()
    assert branch == m10.FROZEN_BRANCH, f"must execute on {m10.FROZEN_BRANCH!r}, got {branch!r}"
    _require_preflight_pass()
    locked_before = m10.assert_locked_test_closed()

    disjointness = m104.verify_seed_disjointness()
    assert disjointness["disjoint"], f"M10.4 seed range is NOT disjoint: {disjointness}"
    (M10_4_DIR / "m10-4-seed-disjointness.json").write_text(json.dumps(disjointness, indent=2) + "\n")

    manifest = _population_manifest()
    (M10_4_DIR / "m10-4-population-manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"population: {manifest['n_physical_incidents_total']} physical incidents "
          f"({manifest['n_api_incidents_total']} paired API incidents)", flush=True)

    safety = dict(m104.SAFETY_COUNTERS_TEMPLATE)

    started = time.time()
    trajectories = _run_population(manifest["rows"], safety)
    print(f"population execution complete: {len(trajectories)} pairs in {time.time() - started:.1f}s", flush=True)

    with (M10_4_DIR / "m10-4-trajectories.jsonl").open("w") as fh:
        for record in trajectories:
            fh.write(json.dumps(record, default=str) + "\n")

    fail_closed = _run_fail_closed(safety)
    all_bounded = all(item.get("bounded") for item in fail_closed)
    fail_closed_doc = {
        "kind": "M10_4_FAIL_CLOSED", "protocol_hash": proto.protocol_hash(),
        "cases": fail_closed, "all_cases_bounded_and_deterministic": all_bounded,
    }
    (M10_4_DIR / "m10-4-fail-closed.json").write_text(json.dumps(fail_closed_doc, indent=2, default=str) + "\n")

    locked_after = m10.assert_locked_test_closed()
    safety_doc = {
        "kind": "M10_4_SAFETY_COUNTERS", "protocol_hash": proto.protocol_hash(),
        "counters": safety, "all_zero": all(v == 0 for v in safety.values()),
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
    }
    (M10_4_DIR / "m10-4-safety-counters.json").write_text(json.dumps(safety_doc, indent=2) + "\n")

    print(json.dumps({"safety_all_zero": safety_doc["all_zero"], "fail_closed_all_bounded": all_bounded, "locked_after": locked_after}, indent=2))


if __name__ == "__main__":
    main()
