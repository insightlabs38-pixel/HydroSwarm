"""Structural non-degeneracy gates for a Strategist trajectory corpus
(important-issues.txt requirement 15).

Run against a `cycle-b2-trajectories-vN`-style directory (produced by
`scripts/generate_trajectory_corpus.py`, one `{split}.jsonl` file per split,
each line the JSON serialization of one `IncidentTrajectory` -- see
`_incident_trajectory_to_json` there for the exact shape this script reads:
`record["strategist"]["steps"][0]["labels"]`/`["targets"]`).

This exists because Phase 12 Stage E discovered, only by directly checking
real numbers rather than trusting suspiciously-good loss curves, that every
valid Strategist candidate across 1000 validation scenarios had an IDENTICAL
`plan_value` of exactly 1.0 -- `HydraulicSimulator.evaluate_plan()` never
computed contamination-exposure consequences, so `cost` (and therefore
`plan_value`/`regret`) had zero real variance anywhere in the corpus. These
gates make that specific failure mode impossible to silently reintroduce:
a corpus that fails any gate here must not be used for Strategist training.

Gates:

1. plan_value_variance         valid-candidate plan_value has nonzero
                                variance across the corpus (not every
                                candidate mechanically tied at 1.0).
2. exposure_variance           valid-candidate contaminant_mass_consumed_mg
                                (consequence_vector[0]) has nonzero variance.
3. per_scenario_cost_variation at least one scenario has more than one
                                distinct plan_value among its OWN valid
                                candidates (proves within-scenario ranking is
                                even possible, not just across-scenario).
4. no_action_not_universally_identical
                                NO_ACTION and every other valid candidate are
                                NOT consequence-identical in every scenario
                                (a corpus where they always match cannot
                                teach the Strategist that action ever helps).
5. some_valid_plan_improves_exposure
                                at least one scenario has a valid non-
                                NO_ACTION candidate with strictly lower
                                contaminant_mass_consumed_mg than NO_ACTION.

Also reports plan_value/exposure_proxy distributions by action_template and
by split -- not a pass/fail gate, but required reporting
(important-issues.txt requirement 15: "Report target distributions by
template and split").
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


class GateFailure(Exception):
    pass


def _iter_scenarios(jsonl_path: Path):
    with jsonl_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _step0_labels_and_targets(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    steps = record.get("strategist", {}).get("steps", [])
    if not steps:
        return None
    return steps[0]["labels"]


def _collect(trajectory_dir: Path, splits: list[str]) -> dict[str, Any]:
    """Single streaming pass building everything every gate below needs, so
    a multi-gigabyte corpus is never loaded into memory more than once."""

    all_plan_values: list[float] = []
    all_exposures: list[float] = []
    scenarios_with_cost_variation = 0
    scenarios_total = 0
    scenarios_with_step0 = 0
    no_action_always_identical = True
    any_valid_plan_beats_no_action = False
    by_template: dict[str, dict[str, list[float]]] = {}
    by_split: dict[str, dict[str, list[float]]] = {}

    for split in splits:
        jsonl_path = trajectory_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue
        split_bucket = by_split.setdefault(split, {"plan_value": [], "exposure": []})
        for record in _iter_scenarios(jsonl_path):
            scenarios_total += 1
            labels = _step0_labels_and_targets(record)
            if not labels:
                continue
            scenarios_with_step0 += 1

            valid = [label for label in labels if label["plan_validity"]]
            valid_values = [label["plan_value"] for label in valid if label["plan_value"] is not None]
            if len(set(round(v, 9) for v in valid_values)) > 1:
                scenarios_with_cost_variation += 1

            no_action = next((label for label in labels if label["is_no_response_comparator"]), None)
            no_action_vector = (
                no_action["consequence_vector"] if no_action is not None and no_action["plan_validity"] else None
            )

            for label in valid:
                template = label["action_template"]
                template_bucket = by_template.setdefault(template, {"plan_value": [], "exposure": []})
                if label["plan_value"] is not None:
                    all_plan_values.append(label["plan_value"])
                    template_bucket["plan_value"].append(label["plan_value"])
                    split_bucket["plan_value"].append(label["plan_value"])
                vector = label["consequence_vector"]
                if vector is not None:
                    exposure = vector[0]
                    all_exposures.append(exposure)
                    template_bucket["exposure"].append(exposure)
                    split_bucket["exposure"].append(exposure)
                    if no_action_vector is not None and not label["is_no_response_comparator"]:
                        if tuple(vector) != tuple(no_action_vector):
                            no_action_always_identical = False
                        if exposure < no_action_vector[0]:
                            any_valid_plan_beats_no_action = True

    return {
        "scenarios_total": scenarios_total,
        "scenarios_with_step0": scenarios_with_step0,
        "all_plan_values": all_plan_values,
        "all_exposures": all_exposures,
        "scenarios_with_cost_variation": scenarios_with_cost_variation,
        "no_action_always_identical": no_action_always_identical,
        "any_valid_plan_beats_no_action": any_valid_plan_beats_no_action,
        "by_template": by_template,
        "by_split": by_split,
    }


def _variance(values: list[float]) -> float:
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def gate_plan_value_variance(stats: dict[str, Any], report: dict[str, Any]) -> None:
    variance = _variance(stats["all_plan_values"])
    report["plan_value_variance"] = variance
    if not stats["all_plan_values"]:
        raise GateFailure("no valid candidates with plan_value found in corpus")
    if variance < 1e-9:
        raise GateFailure(
            f"plan_value has effectively zero variance ({variance:.3e}) across "
            f"{len(stats['all_plan_values'])} valid candidates -- every candidate is mechanically tied"
        )


def gate_exposure_variance(stats: dict[str, Any], report: dict[str, Any]) -> None:
    variance = _variance(stats["all_exposures"])
    report["exposure_variance"] = variance
    if not stats["all_exposures"]:
        raise GateFailure("no valid candidates with a consequence_vector found in corpus")
    if variance < 1e-9:
        raise GateFailure(
            f"contaminant_mass_consumed_mg has effectively zero variance ({variance:.3e}) across "
            f"{len(stats['all_exposures'])} valid candidates -- exposure is not being measured"
        )


def gate_per_scenario_cost_variation(stats: dict[str, Any], report: dict[str, Any]) -> None:
    report["scenarios_with_cost_variation"] = stats["scenarios_with_cost_variation"]
    report["scenarios_with_step0"] = stats["scenarios_with_step0"]
    if stats["scenarios_with_cost_variation"] == 0:
        raise GateFailure(
            "no scenario has valid-candidate cost variation -- every scenario's own valid "
            "candidates are tied, so no within-scenario ranking signal exists"
        )


def gate_no_action_not_universally_identical(stats: dict[str, Any], report: dict[str, Any]) -> None:
    report["no_action_always_identical"] = stats["no_action_always_identical"]
    if stats["no_action_always_identical"]:
        raise GateFailure(
            "NO_ACTION and every active plan are consequence-identical in every scenario -- "
            "the corpus cannot teach the Strategist that acting ever changes the outcome"
        )


def gate_some_valid_plan_improves_exposure(stats: dict[str, Any], report: dict[str, Any]) -> None:
    report["any_valid_plan_beats_no_action"] = stats["any_valid_plan_beats_no_action"]
    if not stats["any_valid_plan_beats_no_action"]:
        raise GateFailure("no valid plan improves exposure relative to NO_ACTION anywhere in the corpus")


def _distribution_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


GATES = (
    gate_plan_value_variance,
    gate_exposure_variance,
    gate_per_scenario_cost_variation,
    gate_no_action_not_universally_identical,
    gate_some_valid_plan_improves_exposure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "validation", "calibration", "development_holdout"]
    )
    parser.add_argument("--report", type=Path, default=None, help="optional path to write the JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = _collect(args.trajectory_dir, args.splits)

    report: dict[str, Any] = {
        "trajectory_dir": str(args.trajectory_dir),
        "splits": args.splits,
        "scenarios_total": stats["scenarios_total"],
        "scenarios_with_strategist_step0": stats["scenarios_with_step0"],
        "distributions_by_template": {
            template: {
                "plan_value": _distribution_summary(buckets["plan_value"]),
                "exposure_mg": _distribution_summary(buckets["exposure"]),
            }
            for template, buckets in sorted(stats["by_template"].items())
        },
        "distributions_by_split": {
            split: {
                "plan_value": _distribution_summary(buckets["plan_value"]),
                "exposure_mg": _distribution_summary(buckets["exposure"]),
            }
            for split, buckets in sorted(stats["by_split"].items())
        },
        "gates": {},
    }

    failures: list[str] = []
    for gate in GATES:
        name = gate.__name__.removeprefix("gate_")
        try:
            gate(stats, report)
        except GateFailure as error:
            failures.append(f"{name}: {error}")
            report["gates"][name] = {"passed": False, "detail": str(error)}
        else:
            report["gates"][name] = {"passed": True}

    report["passed"] = not failures

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k not in ("distributions_by_template", "distributions_by_split")}, indent=2, sort_keys=True))

    if failures:
        print(f"\n{len(failures)}/{len(GATES)} gate(s) FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"\n{len(GATES)}/{len(GATES)} gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
