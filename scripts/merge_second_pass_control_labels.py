"""core-issues4.txt Section F step 6b: join persisted second-pass control
labels into a new, versioned, trainable control corpus.

scripts/persist_second_pass_control_labels.py (Section F step 6a) already
streams SecondPassControlLabel rows to
data/learning-v2/cycle-b2-control-v2/second-pass-labels/{split}.jsonl. This
script is the corpus-merge half: it joins those rows by scenario_id onto
data/learning-v2/cycle-b2's own tensors-normalized inputs (unchanged), and
ALSO recomputes event_cause via the corrected, post-Phase-6.4
hydroswarm.training.corpus._event_cause classifier against each scenario's
real GeneratedScenario -- deliberately NOT cycle-b2's own stored
event_cause tensor, which is a protected artifact carrying the documented
~5% pre-fix HYDRAULIC_MISMATCH mislabel (see corpus.py's own
UNSUPPORTED_EVENT_CAUSES docstring and the handoff report's Phase 8
section).

Never mutates data/learning-v2/cycle-b2: writes only under
data/learning-v2/cycle-b2-control-v2/. The new corpus's own scenarios/ and
normalization/ directories are read-only relative symlinks into cycle-b2
(inputs and topology are byte-identical -- only event_cause/
evidence_sufficiency/next_step targets differ), so
scripts/run_corpus_gates.py's existing topology-provenance/deterministic-
replay/normalization-ownership gates re-verify real, unmodified provenance
rather than a second, separately-trusted copy of it.

Only train and validation are merged. Calibration remains calibration-
owned (core-issues4.txt Section F: "must not become a training split") and
second-pass labels were only ever generated for train/validation.

Usage:

    python scripts/merge_second_pass_control_labels.py \
        --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
        --second-pass-dir data/learning-v2/cycle-b2-control-v2/second-pass-labels \
        --output-dir data/learning-v2/cycle-b2-control-v2
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from hydroswarm.data.scenarios import DatasetSplit, load_generated_scenarios
from hydroswarm.training.corpus import EVENT_CAUSE_INDEX, _event_cause
from hydroswarm.training.data import ScenarioExample
from hydroswarm.training.label_audit import audit_corpus
from hydroswarm.training.sharded_data import ShardedScenarioDataset, write_shards
from hydroswarm.training.targets_v2 import EventCause, NextStep

_MERGED_SPLITS = ("train", "validation")
_NEXT_STEP_INDEX: dict[str, int] = {member.value: index for index, member in enumerate(NextStep)}


def _load_second_pass_rows(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows[record["scenario_id"]] = record
    return rows


def _load_corrected_event_cause(corpus_root: Path, split: DatasetSplit) -> dict[str, EventCause]:
    """Recomputes event_cause per real GeneratedScenario via the corrected
    classifier -- see module docstring for why this must not reuse
    cycle-b2's own stored event_cause tensor."""

    scenarios = load_generated_scenarios(corpus_root / "scenarios", split)
    return {str(scenario.manifest.scenario_id): _event_cause(scenario) for scenario in scenarios}


def _symlink_shared_asset(source: Path, target: Path) -> None:
    """Read-only relative symlink -- never a copy or a mutation of `source`."""

    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(source, start=target.parent)
    target.symlink_to(relative, target_is_directory=source.is_dir())


#: cycle-b2 (and every Cycle-B-style corpus) carries two parallel tensor
#: variants: "tensors" (raw, pre-normalization features -- what
#: scripts/run_corpus_gates.py's gate_normalization_ownership refits
#: NormalizationStats from and compares byte-for-byte against
#: normalization/*.json) and "tensors-normalized" (post-NormalizationStats.
#: transform() features -- what Stage-A was actually trained/calibrated
#: against, and therefore what second-pass control-label generation and
#: control-head training must consume). Merging only one variant and
#: naming its output directory "tensors" breaks gate 6 (it would refit
#: normalization from ALREADY-normalized data and compare against an
#: artifact fit from raw data -- an apples-to-oranges mismatch, not a real
#: normalization-ownership violation). Both variants are merged here, each
#: writing to the correspondingly-named output directory, so downstream
#: consumers keep the exact same "tensors" vs "tensors-normalized" contract
#: cycle-b2 itself already establishes.
MERGED_TENSOR_DIRNAMES = ("tensors", "tensors-normalized")


def merge_split(
    corpus_root: Path,
    tensors_dirname: str,
    second_pass_dir: Path,
    output_dir: Path,
    *,
    split: str,
    second_pass_rows: dict[str, dict[str, Any]],
    corrected_event_cause: dict[str, EventCause],
) -> tuple[list[ScenarioExample], dict[str, Any]]:
    dataset = ShardedScenarioDataset(corpus_root / tensors_dirname / split, expected_split=split)
    dataset.verify_shard_checksums()

    merged_examples: list[ScenarioExample] = []
    matched = 0
    event_cause_changed = 0
    unmatched_scenario_ids: list[str] = []
    for position in range(len(dataset)):
        example = dataset[position]
        row = second_pass_rows.get(example.scenario_id)
        cause = corrected_event_cause.get(example.scenario_id)
        if row is None or cause is None:
            unmatched_scenario_ids.append(example.scenario_id)
            merged_examples.append(example)
            continue

        new_cause_index = torch.tensor(EVENT_CAUSE_INDEX[cause])
        old_cause_index = example.targets.get("event_cause")
        if old_cause_index is not None and int(old_cause_index) != int(new_cause_index):
            event_cause_changed += 1

        new_targets = {
            "event_cause": new_cause_index,
            "evidence_sufficiency": torch.tensor(bool(row["evidence_sufficiency"])),
            "next_step": torch.tensor(_NEXT_STEP_INDEX[row["next_step"]]),
        }
        merged_examples.append(dataclasses.replace(example, targets={**example.targets, **new_targets}))
        matched += 1

    split_output_dir = output_dir / tensors_dirname / split
    manifest = write_shards(merged_examples, split_output_dir)

    return merged_examples, {
        "split": split,
        "tensors_dirname": tensors_dirname,
        "examples_total": len(merged_examples),
        "examples_matched": matched,
        "examples_unmatched_second_pass_label": unmatched_scenario_ids,
        "event_cause_changed_by_phase_6_4_fix": event_cause_changed,
        "shard_manifest": manifest,
    }


def _validate_new_targets(examples: list[ScenarioExample]) -> dict[str, list[dict[str, Any]]]:
    """Real, independent range/finiteness checks on the three merged
    targets -- label_audit.audit_corpus's existing _impossible_labels does
    not know about event_cause/evidence_sufficiency/next_step, so this
    supplements it rather than silently trusting construction alone."""

    impossible: list[dict[str, Any]] = []
    non_finite: list[dict[str, Any]] = []
    event_cause_count = len(EventCause)
    next_step_count = len(NextStep)
    for example in examples:
        cause = int(example.targets["event_cause"])
        if not (0 <= cause < event_cause_count):
            impossible.append({"scenario_id": example.scenario_id, "reason": "event_cause out of range", "value": cause})
        step = int(example.targets["next_step"])
        if not (0 <= step < next_step_count):
            impossible.append({"scenario_id": example.scenario_id, "reason": "next_step out of range", "value": step})
        sufficiency = example.targets["evidence_sufficiency"]
        if not torch.isfinite(sufficiency.float()).all():
            non_finite.append({"scenario_id": example.scenario_id, "tensor": "evidence_sufficiency"})
        if int(sufficiency) not in (0, 1):
            impossible.append(
                {"scenario_id": example.scenario_id, "reason": "evidence_sufficiency not boolean", "value": int(sufficiency)}
            )
    return {"impossible_labels": impossible, "finite_value_violations": non_finite}


def build_label_audit(
    merged_by_split: dict[str, list[ScenarioExample]],
) -> dict[str, Any]:
    audit = audit_corpus(merged_by_split, decision_splits=_MERGED_SPLITS)
    for split, examples in merged_by_split.items():
        extra = _validate_new_targets(examples)
        audit["splits"][split]["impossible_labels"] = (
            audit["splits"][split]["impossible_labels"] + extra["impossible_labels"]
        )
        audit["splits"][split]["finite_value_violations"] = (
            audit["splits"][split]["finite_value_violations"] + extra["finite_value_violations"]
        )
    return audit


def _label_distribution(merged_by_split: dict[str, list[ScenarioExample]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    event_cause_names = [member.value for member in EventCause]
    next_step_names = [member.value for member in NextStep]
    for split, examples in merged_by_split.items():
        event_cause_counts = [0] * len(EventCause)
        next_step_counts = [0] * len(NextStep)
        evidence_sufficient_true = 0
        for example in examples:
            event_cause_counts[int(example.targets["event_cause"])] += 1
            next_step_counts[int(example.targets["next_step"])] += 1
            evidence_sufficient_true += int(example.targets["evidence_sufficiency"])
        report[split] = {
            "count": len(examples),
            "event_cause": dict(zip(event_cause_names, event_cause_counts, strict=True)),
            "next_step": dict(zip(next_step_names, next_step_counts, strict=True)),
            "evidence_sufficiency_true": evidence_sufficient_true,
            "evidence_sufficiency_false": len(examples) - evidence_sufficient_true,
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-root", type=Path, default=Path("data/learning-v2/cycle-b2"))
    parser.add_argument("--second-pass-dir", type=Path, default=Path("data/learning-v2/cycle-b2-control-v2/second-pass-labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/learning-v2/cycle-b2-control-v2"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Loaded once per split (dirname-independent -- the same scenario_id ->
    # (event_cause, evidence_sufficiency, next_step) join applies whether
    # the underlying features are raw or normalized).
    second_pass_by_split = {split: _load_second_pass_rows(args.second_pass_dir / f"{split}.jsonl") for split in _MERGED_SPLITS}
    corrected_event_cause_by_split = {
        split: _load_corrected_event_cause(args.corpus_root, DatasetSplit(split)) for split in _MERGED_SPLITS
    }

    merged_by_split: dict[str, list[ScenarioExample]] = {}
    merge_reports: dict[str, Any] = {}
    for tensors_dirname in MERGED_TENSOR_DIRNAMES:
        for split in _MERGED_SPLITS:
            examples, report = merge_split(
                args.corpus_root,
                tensors_dirname,
                args.second_pass_dir,
                args.output_dir,
                split=split,
                second_pass_rows=second_pass_by_split[split],
                corrected_event_cause=corrected_event_cause_by_split[split],
            )
            # gate_normalization_ownership and every other run_corpus_gates.py
            # gate reads corpus_dir / "tensors" (the raw variant) -- the
            # label audit/leakage/distribution reports below describe that
            # variant specifically, matching what the gates actually check.
            if tensors_dirname == "tensors":
                merged_by_split[split] = examples
            merge_reports[f"{tensors_dirname}/{split}"] = report
            print(
                f"{tensors_dirname}/{split}: merged {report['examples_matched']}/{report['examples_total']} "
                f"({len(report['examples_unmatched_second_pass_label'])} unmatched, "
                f"{report['event_cause_changed_by_phase_6_4_fix']} event_cause labels changed by the Phase 6.4 fix)"
            )

    # scenarios/ and normalization/ are byte-identical to cycle-b2's own
    # (only targets changed above) -- symlinked, not copied, so
    # run_corpus_gates.py's topology-provenance/deterministic-replay/
    # normalization-ownership gates re-verify real, unmodified cycle-b2
    # provenance rather than trusting a second copy of it.
    _symlink_shared_asset(args.corpus_root / "scenarios", args.output_dir / "scenarios")
    _symlink_shared_asset(args.corpus_root / "normalization", args.output_dir / "normalization")

    label_audit = build_label_audit(merged_by_split)
    (args.output_dir / "label-audit.json").write_text(
        json.dumps(label_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    label_distribution = _label_distribution(merged_by_split)
    (args.output_dir / "label-distribution-report.json").write_text(
        json.dumps(label_distribution, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    leakage_report = label_audit["cross_split_leakage"]
    (args.output_dir / "leakage-report.json").write_text(
        json.dumps(leakage_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Teacher-checkpoint / calibration / control-policy identity: read back
    # from step 6a's own already-computed manifests rather than
    # recomputing (single source of truth for these hashes).
    source_manifests = {
        split: json.loads((args.second_pass_dir / f"{split}.manifest.json").read_text(encoding="utf-8"))
        for split in _MERGED_SPLITS
    }
    teacher_hashes = {source_manifests[split]["teacher_checkpoint_hash"] for split in _MERGED_SPLITS}
    calibration_hashes = {source_manifests[split]["calibration_hash"] for split in _MERGED_SPLITS}
    control_policy_hashes = {source_manifests[split]["control_policy_hash"] for split in _MERGED_SPLITS}
    if len(teacher_hashes) != 1 or len(calibration_hashes) != 1 or len(control_policy_hashes) != 1:
        raise ValueError(
            "train/validation second-pass labels were generated from different "
            f"teacher/calibration/policy identities: {source_manifests}"
        )

    corpus_manifest = {
        "schema_version": 1,
        "corpus_name": "cycle-b2-control-v2",
        "source_corpus_root": str(args.corpus_root),
        "source_tensors_dirnames": list(MERGED_TENSOR_DIRNAMES),
        "merged_splits": list(_MERGED_SPLITS),
        "teacher_checkpoint_hash": next(iter(teacher_hashes)),
        "calibration_hash": next(iter(calibration_hashes)),
        "control_policy_version": source_manifests["train"]["control_policy_version"],
        "control_policy_hash": next(iter(control_policy_hashes)),
        "event_cause_recomputed_via": "hydroswarm.training.corpus._event_cause (post-Phase-6.4 fix)",
        "merge_reports": merge_reports,
    }
    manifest_hash = hashlib.sha256(
        json.dumps(corpus_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    corpus_manifest["corpus_manifest_hash"] = manifest_hash
    (args.output_dir / "manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({"corpus_manifest_hash": manifest_hash, "label_distribution": label_distribution}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
