"""M11.6A-1 -- write the frozen design-freeze artifacts.

Generates the additive design-input/design-freeze artifacts under
`reports/evaluation/hydrocore-v5/m11/m11-6a/design-freeze/` from the
authoritative `m11_6a_design` module, so the JSON artifacts and the code can
never drift. Does NOT materialize a locked dataset, derive a final seed, or
touch any locked data.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m11_6a_design as design  # noqa: E402
import m11_6a_topology as topology  # noqa: E402
import run_m11_6_locked_evaluation as evaluator  # noqa: E402

DESIGN_DIR = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-6a/design-freeze"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def prior_data_inventory() -> dict[str, Any]:
    """Mechanical inventory of everything already used for training/validation/
    calibration/development_holdout/ood_development/M9/M10/M11.5. This is the
    additive design-input artifact (task Section 3): provenance/hashes/ranges,
    NOT copied historical outcome tables.
    """

    topology_files = {
        "golden-reference": ROOT / "data/frozen/golden_network.inp",
        "branched-loop": ROOT / "data/topology-transfer/branched-loop.inp",
        "loop-grid": ROOT / "data/topologies/loop-grid.inp",
        "coastal-branch": ROOT / "data/topologies/coastal-branch.inp",
    }
    frozen_files = {
        "data/frozen/manifest.json": ROOT / "data/frozen/manifest.json",
        "data/frozen/golden_scenario.json": ROOT / "data/frozen/golden_scenario.json",
        "data/frozen/live_example_network.inp": ROOT / "data/frozen/live_example_network.inp",
        "data/frozen/live_example_scenario.json": ROOT / "data/frozen/live_example_scenario.json",
    }
    return {
        "kind": "M11_6A_PRIOR_DATA_INVENTORY",
        "note": "provenance/hashes/ranges only -- establishes the formal NON-OVERLAP rule; contains no historical outcome tables and no candidate locked examples",
        "topology_families": design.PRIOR_TOPOLOGY_SIGNATURES,
        "topology_files": {
            name: {"path": relative(path), "sha256": sha256_file(path)}
            for name, path in topology_files.items()
        },
        "frozen_regression_fixtures": {
            name: {"path": relative(path), "sha256": sha256_file(path)}
            for name, path in frozen_files.items()
        },
        "seed_namespaces": design.PRIOR_SEED_RANGES,
        "split_roles_used": ["train", "validation", "calibration", "development_holdout", "ood_development"],
        "locked_splits_never_used": ["locked_final_test", "locked_topology_test"],
        "no_relabeling": (
            "data/frozen regression/golden fixtures and every M0-M11.5 "
            "development/validation/OOD population MUST NOT be relabeled as the "
            "final locked evaluation (task Section 0)."
        ),
    }


def design_preflight() -> dict[str, Any]:
    return {
        "kind": "M11_6A_DESIGN_PREFLIGHT",
        "repository": "insightlabs38-pixel/HydroSwarm",
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "expected_blocker_head": "6c84b3b710f50c9bf58f6f76d881af7f32e0710b",
        "blocker_state": "M11_6_LOCKED_EVALUATION_BLOCKED_NO_DATASET",
        "m11_5_state": "M11_5_FULL_VALIDATION_PASS",
        "m11_5_matrix_green": True,
        "m11_6_preconditions_satisfied": True,
        "tuning_closed": True,
        "locked_open_count_at_start": 0,
        "locked_test_opened": False,
        "locked_evaluation_authorized": False,
        "authorization_consumed": False,
        "frozen_finalist_identity_verified": evaluator.verify_finalist_identity(),
        "design_hash": design.design_hash(),
        "worktree_clean": git("status", "--porcelain") == "",
    }


def design_freeze_record(preflight: dict[str, Any]) -> dict[str, Any]:
    design_files = [
        ROOT / "scripts/hydrocore_v5/m11_6a_design.py",
        ROOT / "scripts/hydrocore_v5/m11_6a_topology.py",
        ROOT / "scripts/hydrocore_v5/run_m11_6a_materialize.py",
        ROOT / "scripts/hydrocore_v5/run_m11_6_locked_evaluation.py",
        ROOT / "docs/evaluation/HYDROCORE_V5_M11_6A_LOCKED_EVALUATION_DESIGN_FREEZE.md",
    ]
    return {
        "kind": "M11_6A_DESIGN_FREEZE",
        "milestone": "M11.6A-1",
        "schema_version": design.DESIGN_SCHEMA_VERSION,
        "design_frozen": True,
        "dataset_materialized": False,
        "locked_manifest_created": False,
        "final_locked_seed_derived": False,
        "finalist_evaluated_on_locked": False,
        "locked_open_count": 0,
        "locked_test_opened": False,
        "locked_evaluation_authorized": False,
        "authorization_consumed": False,
        "next_action": "M11_6A_2_MATERIALIZE_FROM_CORRECTED_FROZEN_DESIGN",
        "design_hash": design.design_hash(),
        "seed_derivation_rule_version": design.SEED_RULE_VERSION,
        "seed_derivation_formula": design.seed_derivation_spec()["master_formula"],
        "design_file_hashes": {relative(path): sha256_file(path) for path in design_files},
        "m11_6_blocker_closure_preserved": "M11_6_LOCKED_EVALUATION_BLOCKED_NO_DATASET",
        "superseded_design_freeze_commits": list(design.SUPERSEDED_DESIGN_FREEZE_COMMITS),
        "materialization_must_use_this_commit": True,
        "manifest_file_sha256_binding": (
            "Authorization binds to materialization_manifest_file_sha256 (SHA-256 "
            "of the exact committed manifest FILE bytes); the canonical-dict hash "
            "is recorded separately as manifest_canonical_hash and is never the "
            "authorization binding."
        ),
        "preflight": preflight,
        "finalist_identity": {
            "system": evaluator.FINALIST["system"],
            "seed": evaluator.FINALIST["seed"],
            "checkpoint_sha256": evaluator.FINALIST["checkpoint"],
            "release_manifest_sha256": evaluator.FINALIST["manifest"],
            "calibration_sha256": evaluator.FINALIST["calibration"],
            "calibration_artifact_hash": evaluator.FINALIST["calibration_artifact"],
        },
        "does_not_claim_real_locked_manifest_hash": True,
    }


def write_json(name: str, payload: dict[str, Any]) -> Path:
    path = DESIGN_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def main() -> None:
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    preflight = design_preflight()
    freeze = design_freeze_record(preflight)

    artifacts: list[tuple[str, dict[str, Any]]] = [
        ("m11-6a-design-preflight.json", preflight),
        ("m11-6a-prior-data-inventory.json", prior_data_inventory()),
        ("m11-6a-population-spec.json", {"kind": "M11_6A_POPULATION_SPEC", **design.population_spec()}),
        ("m11-6a-seed-derivation-spec.json", {"kind": "M11_6A_SEED_DERIVATION_SPEC", **design.seed_derivation_spec()}),
        ("m11-6a-topology-novelty-spec.json", {
            "kind": "M11_6A_TOPOLOGY_NOVELTY_SPEC",
            "novelty": design.topology_novelty_spec(),
            "generator": topology.topology_spec(),
        }),
        ("m11-6a-nonoverlap-spec.json", {"kind": "M11_6A_NONOVERLAP_SPEC", **design.non_overlap_spec()}),
        ("m11-6a-manifest-schema.json", {"kind": "M11_6A_MANIFEST_SCHEMA", **design.manifest_schema()}),
        ("m11-6a-metrics.json", {"kind": "M11_6A_METRICS", "metrics": design.METRICS}),
        ("m11-6a-gate-provenance.json", {"kind": "M11_6A_GATE_PROVENANCE", **design.GATE_PROVENANCE}),
        ("m11-6a-evaluator-contract.json", evaluator.evaluator_contract()),
        ("m11-6a-exactly-once-contract.json", {"kind": "M11_6A_EXACTLY_ONCE_CONTRACT", **design.exactly_once_contract()}),
        ("m11-6a-design-freeze.json", freeze),
    ]
    for name, payload in artifacts:
        write_json(name, payload)

    print(json.dumps({
        "design_dir": relative(DESIGN_DIR),
        "design_frozen": freeze["design_frozen"],
        "design_hash": design.design_hash(),
        "final_locked_seed_derived": freeze["final_locked_seed_derived"],
        "locked_test_opened": freeze["locked_test_opened"],
        "artifacts_written": [name for name, _ in artifacts],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
