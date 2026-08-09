"""core-issues2.txt Phase 7: the trajectory-corpus generation script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_trajectory_corpus  # noqa: E402

from test_run_corpus_gates import _build_mini_corpus  # noqa: E402

#: Every test in this module runs many real WNTR/EPANET verifications
#: (audited call count >=10 each) -- see pyproject.toml's full_simulation
#: marker docstring.
pytestmark = pytest.mark.full_simulation


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("mini-corpus-for-trajectories")
    _build_mini_corpus(output)
    return output


def test_generates_a_shard_and_report_with_zero_errors(mini_corpus, tmp_path) -> None:
    output = tmp_path / "trajectories"
    exit_code = generate_trajectory_corpus.main(
        [
            "--corpus-dir", str(mini_corpus),
            "--output", str(output),
            "--split", "train",
            "--limit", "3",
            "--signature-cache-dir", str(tmp_path / "sigcache"),
        ]
    )
    assert exit_code == 0

    shard_path = output / "train.jsonl"
    records = [json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 3

    report = json.loads((output / "train-report.json").read_text(encoding="utf-8"))
    assert report["errors_this_run"] == 0
    assert report["scenarios_processed_this_run"] == 3
    assert report["total_in_shard"] == 3


def test_every_record_carries_the_full_governed_bundle(mini_corpus, tmp_path) -> None:
    output = tmp_path / "trajectories"
    generate_trajectory_corpus.main(
        [
            "--corpus-dir", str(mini_corpus),
            "--output", str(output),
            "--split", "train",
            "--limit", "2",
            "--signature-cache-dir", str(tmp_path / "sigcache"),
        ]
    )
    records = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        assert record["scenario_id"]
        assert record["ood_category"] in {
            "NONE", "UNSEEN_TOPOLOGY", "EXTREME_DEMAND", "TANK_STATE_SHIFT",
            "ROUGHNESS_MISMATCH", "SEVERE_MISSINGNESS", "FROZEN_DRIFTING_SENSOR",
        }
        assert "next_step" in record["targets"]
        assert "ood_class" in record["targets"]
        assert "sensor_reconstruction" in record["targets"]
        assert len(record["scout"]["steps"]) >= 1
        assert len(record["strategist"]["steps"][0]["labels"]) >= 2


def test_rerun_is_resumable_and_does_not_duplicate_records(mini_corpus, tmp_path) -> None:
    output = tmp_path / "trajectories"
    args = [
        "--corpus-dir", str(mini_corpus),
        "--output", str(output),
        "--split", "train",
        "--limit", "3",
        "--signature-cache-dir", str(tmp_path / "sigcache"),
    ]
    generate_trajectory_corpus.main(args)
    first_report = json.loads((output / "train-report.json").read_text(encoding="utf-8"))
    assert first_report["scenarios_processed_this_run"] == 3

    generate_trajectory_corpus.main(args)
    second_report = json.loads((output / "train-report.json").read_text(encoding="utf-8"))
    assert second_report["scenarios_processed_this_run"] == 0  # nothing new to do
    assert second_report["total_in_shard"] == 3

    records = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 3  # not duplicated
    assert len({record["scenario_id"] for record in records}) == 3


def test_rerun_with_a_larger_limit_only_processes_the_new_scenarios(mini_corpus, tmp_path) -> None:
    output = tmp_path / "trajectories"
    base_args = [
        "--corpus-dir", str(mini_corpus),
        "--output", str(output),
        "--split", "train",
        "--signature-cache-dir", str(tmp_path / "sigcache"),
    ]
    generate_trajectory_corpus.main([*base_args, "--limit", "2"])
    generate_trajectory_corpus.main([*base_args, "--limit", "4"])

    report = json.loads((output / "train-report.json").read_text(encoding="utf-8"))
    assert report["scenarios_processed_this_run"] == 2
    assert report["total_in_shard"] == 4
