"""Run the governed target/label audit (overnight-plan.txt Task 0.5).

Loads each split of a canonical-tensor JSONL corpus, computes structural
label-quality checks and sanity baselines, and writes a machine-readable
report. Baseline accuracies are only computed for --decision-split values
(train/validation/calibration by default); the locked test split, if
provided, is included solely for leakage detection and structural counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydroswarm.training import audit_corpus, load_scenario_examples_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/learning-v1/tensors-canonical-v3"),
        help="directory containing <split>.jsonl files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "calibration", "test"],
        help="split names to load as <corpus-dir>/<split>.jsonl",
    )
    parser.add_argument(
        "--decision-splits",
        nargs="+",
        default=["train", "validation", "calibration"],
        help="splits eligible for sanity-baseline accuracy computation; "
        "locked test data should never appear here",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits = {
        name: load_scenario_examples_jsonl(args.corpus_dir / f"{name}.jsonl")
        for name in args.splits
        if (args.corpus_dir / f"{name}.jsonl").exists()
    }
    missing = set(args.splits) - set(splits)
    if missing:
        print(f"warning: missing split files, skipped: {sorted(missing)}")
    report = audit_corpus(splits, decision_splits=args.decision_splits)
    report["corpus_dir"] = str(args.corpus_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
