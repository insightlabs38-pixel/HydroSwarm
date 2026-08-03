from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_measured_evaluation_is_repeated_seed_and_honest_about_missing_models() -> None:
    result = json.loads(
        (ROOT / "reports" / "results" / "evaluation_results.json").read_text(encoding="utf-8")
    )
    assert result["measured"] is True
    assert len(result["runs"]) == len(result["config"]["seeds"]) >= 3
    assert result["aggregate"]["localization_top1_accuracy"]["n"] >= 3
    assert result["baselines_and_ablations"]["no_exact_verifier"]["promotable"] is False
    missing = [item for item in result["model_variants"] if item["status"] == "not_run_missing_checkpoint"]
    assert {item["name"] for item in missing} == {"small", "medium", "large"}
    assert result["promotion_gate"]["passed"] == all(result["promotion_gate"]["checks"].values())

