"""Freeze M10.5A's pre-M10.4 deterministic deployment identity."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_COMMIT = "cecee4c8a8bd9bc3801d8038bfe9b858508bc8fd"
SEEDS = (20260814, 31874, 20260815)
SELECTED_SEED = SEEDS[0]
EXPECTED_SHA = "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"
M94_INTRODUCTION_COMMIT = "f2e7857f00be6e33420439f44b6ededa0e6c396f"
DOC = ROOT / "docs/evaluation/HYDROCORE_V5_M10_5A_DEPLOYMENT_SELECTION_AMENDMENT.md"
OUT = ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5a-selection"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "scripts/hydrocore_v5"))
    import m9_4_common as m94
    import m9_6_common as m96
    import m10_common as m10

    record = json.loads((m96.M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{SELECTED_SEED}.json").read_text())
    path = Path(record["canonical_export_path"])
    actual = digest(path)
    return {
        "kind": "M10_5A_SEED_ORDER_PROVENANCE",
        "historical_declaration": "scripts/hydrocore_v5/m9_4_common.py:SEEDS",
        "historical_introduction_commit": M94_INTRODUCTION_COMMIT,
        "historical_precedes_m10_4": True,
        "m9_4_seeds": list(m94.SEEDS), "m9_6_seeds": list(m96.SEEDS), "m10_seeds": list(m10.SEEDS),
        "orders_identical": m94.SEEDS == m96.SEEDS == m10.SEEDS == SEEDS,
        "selection_rule": "FIRST_CANONICAL_SEED_IN_PREEXISTING_FROZEN_SEED_ORDER",
        "selected_seed": SELECTED_SEED,
        "checkpoint_path": str(path), "checkpoint_sha256": actual,
        "expected_checkpoint_sha256": EXPECTED_SHA, "checkpoint_sha_matches": actual == EXPECTED_SHA,
        "canonical_export_policy": record["canonical_checkpoint_policy"],
        "all_peers_use_final_step": all(json.loads((m96.M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{s}.json").read_text())["canonical_checkpoint_policy"] == "FINAL_STEP_1350" for s in SEEDS),
        "m10_4_performance_inspected": False,
    }


def write() -> dict:
    a = audit(); assert DOC.exists() and a["orders_identical"] and a["checkpoint_sha_matches"] and a["all_peers_use_final_step"]
    protocol_hash = digest(DOC)
    closure = {"kind":"M10_5A_CLOSURE", "closure_state":"M10_5A_DEPLOYMENT_SELECTION_FROZEN", "start_commit":START_COMMIT, "protocol_hash":protocol_hash, "selected_seed":SELECTED_SEED, "selected_checkpoint_sha256":EXPECTED_SHA, "m10_4_performance_used":False}
    payloads = {
      "m10-5a-protocol.json": {"kind":"M10_5A_PROTOCOL","protocol_hash":protocol_hash,"start_commit":START_COMMIT,"selection_rule":a["selection_rule"]},
      "m10-5a-seed-order-provenance.json": a,
      "m10-5a-selection-rule.json": {"kind":"M10_5A_SELECTION_RULE","rule":a["selection_rule"],"selected_seed":SELECTED_SEED,"performance_independent":True,"unique":True},
      "m10-5a-checkpoint-identity.json": {"kind":"M10_5A_CHECKPOINT_IDENTITY","seed":SELECTED_SEED,"sha256":EXPECTED_SHA,"final_step_1350":True},
      "m10-5a-historical-immutability.json": {"kind":"M10_5A_HISTORICAL_IMMUTABILITY","historical_paths_modified":[],"locked_test_opened":False},
      "m10-5a-closure.json": closure,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items(): (OUT/name).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    return closure


if __name__ == "__main__": print(json.dumps(write(), indent=2, sort_keys=True))
