"""Rebuild HydroCore manifests from raw scenarios using production feature semantics."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from hydroswarm.data.scenarios import DatasetSplit, load_generated_scenarios
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.corpus import (
    build_feature_context,
    fit_signature_library,
    scenario_to_example,
    write_tensor_manifest,
)

from prepare_training_corpus import HELD_OUT_FAMILY, TRAINING_FAMILIES


def main() -> int:
    root = Path("data/learning-v1")
    output = root / "tensors-canonical-v3"
    report_path = root / "canonical-tensor-report-v3.json"
    if report_path.exists():
        raise SystemExit(f"refusing to replace completed canonical corpus: {report_path}")
    definitions = {
        item.name: item for item in (*TRAINING_FAMILIES, HELD_OUT_FAMILY)
    }
    networks = {
        name: build_wntr_network(definition) for name, definition in definitions.items()
    }
    contexts = {
        name: build_feature_context(network) for name, network in networks.items()
    }
    scenarios = {
        split: load_generated_scenarios(root / "scenarios", split)
        for split in DatasetSplit
    }
    node_ids = tuple(sorted(next(iter(networks.values())).junction_name_list))
    signatures = fit_signature_library(scenarios[DatasetSplit.TRAIN], node_ids)
    started = time.perf_counter()
    hashes = {}
    shape_counts: Counter[str] = Counter()
    for split, split_scenarios in scenarios.items():
        examples = [
            scenario_to_example(
                scenario,
                networks[scenario.manifest.network_family],
                signatures,
                feature_context=contexts[scenario.manifest.network_family],
            )
            for scenario in split_scenarios
        ]
        for example in examples:
            shape_counts[str(tuple(example.inputs["node_features"].shape))] += 1
        path = output / f"{split.value}.jsonl"
        hashes[split.value] = write_tensor_manifest(path, examples)
    report = {
        "schema_version": 1,
        "feature_schema_version": DEFAULT_FEATURE_SCHEMA.version,
        "feature_schema_sha256": DEFAULT_FEATURE_SCHEMA.fingerprint,
        "source_scenario_report_sha256": hashlib.sha256(
            (root / "dataset-report.json").read_bytes()
        ).hexdigest(),
        "split_counts": {split.value: len(items) for split, items in scenarios.items()},
        "manifest_sha256": hashes,
        "node_feature_shape_counts": dict(shape_counts),
        "build_seconds": time.perf_counter() - started,
        "runtime_builder": "hydroswarm.preprocessing.HydraulicFeatureBuilder",
        "canonical_runtime_compatible": True,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
