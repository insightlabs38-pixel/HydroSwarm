"""Assemble immutable post-remediation summaries from measured artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/evaluation/capability-remediation"


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> int:
    temporal = read("temporal-capability.json")
    topology = read("topology-transfer.json")
    component = read("component-decomposition.json")
    sampling = read("sampling.json")
    live_rows = read("live-capability-results.json")
    live = read("live-capability-summary.json")
    fit = read("calibration-fit.json")
    calibration = {
        "schema_version": 1,
        "artifact": "models/hydrocore-v4-release/calibration.json",
        "fit_population": "existing designated calibration split only",
        "alpha": fit["alpha"], "coverage": fit["coverage"],
        "mean_candidate_size": fit["mean_set_size"],
        "coverage_by_condition": fit.get("coverage_by_condition"),
        "coverage_by_network": fit.get("coverage_by_network"),
        "locked_test_opened": False,
    }
    ood = {
        "schema_version": 1,
        "canonical_supported": {
            key: {"calibrated": value["calibrated"], "ood_normal": value["ood_normal"]}
            for key, value in topology["conditions"].items() if key.startswith("known_")
        },
        "unseen_development": topology["conditions"]["unseen_development_coastal"],
        "expectation": "Unknown topology remains calibration-inapplicable and non-NORMAL OOD.",
        "locked_test_opened": False,
    }
    reasons = Counter(reason for row in live_rows for reason in (row.get("suppression_reasons") or []))
    suppression = {
        "schema_version": 1, "population": "four canonical causal LIVE development runs",
        "counts": dict(sorted(reasons.items())),
        "calibration_invalid_rate": sum(not bool(row.get("calibrated")) for row in live_rows) / len(live_rows),
        "ood_caution_rate": sum(row.get("ood_level") != "NORMAL" for row in live_rows) / len(live_rows),
        "locked_test_opened": False,
    }
    # ``calibration.json`` is the actual regenerated conformal artifact,
    # deliberately preserved beside these reports.  Never replace it with a
    # prose/metric summary.
    (OUT / "calibration-summary.json").write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "ood.json").write_text(json.dumps(ood, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "suppression.json").write_text(json.dumps(suppression, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "live-capability.json").write_text(json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1, "locked_test_opened": False,
        "temporal": temporal["checkpoints"], "calibration": calibration,
        "ood": ood, "sampling": sampling["strategies"],
        "components": component["known_network_component_summary"],
        "topology": topology["conditions"], "live": live,
        "finding": {
            "CAP-REM-01": "Early causal prefixes (one step top-1 0.15; three step 0.45) are materially weaker than later causal evidence; investigate causal-prefix training before model scaling.",
            "CAP-REM-02": "EIG improves entropy relative to random but neither strategy reached planning eligibility within three samples in the sparse-evidence development slice; safety gates were not retuned.",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("campaign", "metric", "value", "n", "source"))
        writer.writeheader()
        for checkpoint, values in temporal["checkpoints"].items():
            for metric in ("top1", "top3", "mrr", "planning_eligible"):
                writer.writerow({"campaign": f"temporal:{checkpoint}", "metric": metric, "value": values[metric], "n": values["n"], "source": "temporal-capability.json"})
        for strategy, values in sampling["strategies"].items():
            for metric, value in values.items():
                if metric != "n":
                    writer.writerow({"campaign": f"sampling:{strategy}", "metric": metric, "value": value, "n": values["n"], "source": "sampling.json"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
