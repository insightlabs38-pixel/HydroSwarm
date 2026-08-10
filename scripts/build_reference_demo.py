"""SUB-4 (submission.txt SS6.4): generator entry point for the governed
REFERENCE INCIDENT artifact.

Runs the real, frozen, WNTR-backed golden scenario (same one
`scripts/run_golden.py` runs), builds the milestone-by-milestone
reference-incident artifact from it, and writes:

  artifacts/reference-demo/reference-incident-v1.json
  artifacts/reference-demo/manifest.json

Deterministic: same frozen golden inputs + same code version -> the same
semantic artifact (`artifact_sha256` is stable across runs on unchanged
inputs -- `generated_at` is the only field expected to differ run to run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation import GoldenScenarioRunner, build_reference_incident_artifact  # noqa: E402
from hydroswarm.networks import network_topology_metadata  # noqa: E402
from hydroswarm.training.artifacts import git_commit_hash  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts" / "reference-demo"
    )
    args = parser.parse_args(argv)

    golden_result = GoldenScenarioRunner(ROOT, seed=args.seed).run()

    # The exact frozen network golden.py just ran against (written by
    # freeze_golden_inputs) -- reused via the same metadata extraction a
    # real network import computes, so the artifact's topology can never
    # silently drift from a hand-authored fixture (submission.txt's "do not
    # mix demo fixture data into live UI state").
    import wntr

    frozen_network_path = ROOT / "data" / "frozen" / "golden_network.inp"
    network_topology = network_topology_metadata(wntr.network.WaterNetworkModel(str(frozen_network_path)))

    artifact = build_reference_incident_artifact(
        golden_result,
        generator="scripts/build_reference_demo.py",
        source_commit=git_commit_hash(ROOT),
        network_topology=network_topology,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.out_dir / "reference-incident-v1.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "reference_id": artifact["reference_id"],
        "artifact_file": artifact_path.name,
        "artifact_sha256": artifact["artifact_sha256"],
        "golden_result_hash": artifact["golden_result_hash"],
        "final_event_hash": artifact["final_event_hash"],
        "event_count": artifact["event_count"],
        "milestone_count": len(artifact["milestones"]),
        "generated_at": artifact["generated_at"],
        "source_commit": artifact["source_commit"],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {artifact_path} ({len(artifact['milestones'])} milestones)")
    print(f"wrote {manifest_path}")
    print(f"artifact_sha256={artifact['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
