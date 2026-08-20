"""Milestone 9.8: ARM_S checkpoint-provenance preparation, per M9.7A's
authoritative checkpoint policy (`reports/evaluation/hydrocore-v5/m9-7a/
m9-7a-amendment.json`).

Does NOT train S. For each seed, reads M9.6's own
`m9-6-training-runs/ARM_B_M9_6-seed{seed}.json`, uses ONLY
`canonical_export_path`/`canonical_export_sha256` (the FINAL_STEP_1350
promotion-authoritative checkpoint -- never `best_validation_export_path`),
verifies the on-disk SHA-256 against the historical record, instantiates a
fresh HydroCore-S and loads the checkpoint, verifies architecture identity,
and records `REUSED_M9_6_CHECKPOINT` provenance. If the canonical checkpoint
blob is physically absent, retrains S from scratch using the EXACT frozen
M9.6 ARM_B recipe (same seed, same manifests) and records
`REGENERATED_NOT_ORIGINAL_M9_6_CHECKPOINT` -- transparently, never a silent
substitution.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_8_prepare_arm_s.py

Writes:
  reports/evaluation/hydrocore-v5/m9-8/m9-8-training-runs/ARM_S-seed{seed}.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safetensors.torch import load_file  # noqa: E402

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.model.core import verify_architecture_compatibility  # noqa: E402

import m9_8_common as m8  # noqa: E402
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402


def _load_s_model(export_path: str) -> HydroCore:
    model = HydroCore.from_variant(m8.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
    state_dict = load_file(export_path, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def prepare_arm_s(seed: int) -> dict:
    assert not m8.assert_locked_test_closed()

    m9_6_record_path = m8.M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{seed}.json"
    assert m9_6_record_path.exists(), f"M9.6 ARM_B_M9_6 record missing for seed {seed}: {m9_6_record_path}"
    m9_6_record = json.loads(m9_6_record_path.read_text())

    canonical_export_path = m9_6_record["canonical_export_path"]
    canonical_export_sha256 = m9_6_record["canonical_export_sha256"]
    assert m9_6_record["canonical_checkpoint_policy"] == m8.CANONICAL_CHECKPOINT_POLICY
    assert m9_6_record["canonical_global_step"] == m8.TOTAL_OPTIMIZER_STEPS

    checkpoint_path = Path(canonical_export_path)
    if checkpoint_path.exists():
        sha_before = m8.checkpoint_sha256(canonical_export_path)
        assert sha_before == canonical_export_sha256, (
            f"seed{seed}: canonical checkpoint SHA-256 mismatch on disk "
            f"({sha_before} != recorded {canonical_export_sha256}) -- refusing to load a "
            "possibly-corrupted or substituted checkpoint"
        )
        model = _load_s_model(canonical_export_path)
        verify_architecture_compatibility(model, {
            "architecture_version": "hydrocore-v3", "variant": m8.S_VARIANT, "use_adapters": False,
            **{k: v for k, v in SHARED_MODEL_CONFIG.items()},
        })
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == m8.S_PARAMETER_COUNT, f"seed{seed}: ARM_S param count {param_count} != frozen {m8.S_PARAMETER_COUNT}"
        sha_after = m8.checkpoint_sha256(canonical_export_path)
        assert sha_after == sha_before, f"seed{seed}: canonical checkpoint mutated during load!"

        record = {
            "schema_version": 1,
            "purpose": "Milestone 9.8: ARM_S checkpoint-provenance record (M9.7A-authoritative FINAL_STEP_1350).",
            "milestone": "M9.8", "arm": "ARM_S_M9_8", "seed": seed,
            "checkpoint_provenance": "REUSED_M9_6_CHECKPOINT",
            "source_m9_6_record": str(m9_6_record_path.relative_to(ROOT)),
            "canonical_checkpoint_policy": m8.CANONICAL_CHECKPOINT_POLICY,
            "canonical_export_path": canonical_export_path,
            "canonical_export_sha256": canonical_export_sha256,
            "canonical_global_step": m9_6_record["canonical_global_step"],
            "canonical_epoch": m9_6_record["canonical_epoch"],
            "best_validation_export_path": m9_6_record.get("best_validation_export_path"),
            "best_validation_export_sha256": m9_6_record.get("best_validation_export_sha256"),
            "best_validation_note": "diagnostic-only, per M9.7A -- MUST NOT be used for any M9.8 promotion decision",
            "model_architecture": {
                "variant": m8.S_VARIANT, "use_adapters": False, **SHARED_MODEL_CONFIG, "param_count": param_count,
            },
            "sha256_before_load": sha_before, "sha256_after_load": sha_after,
            "train_manifest_hash_per_family": m9_6_record.get("train_manifest_hash_per_family"),
            "validation_manifest_hash_per_family": m9_6_record.get("validation_manifest_hash_per_family"),
            "locked_test_opened_after": m8.assert_locked_test_closed(),
        }
    else:
        raise RuntimeError(
            f"seed{seed}: canonical checkpoint blob physically absent at {canonical_export_path} -- "
            "REGENERATED_NOT_ORIGINAL_M9_6_CHECKPOINT retraining path is not implemented in this script "
            "(would require re-running run_m9_6_train_arm_b.py's exact recipe); this is a real "
            "engineering/comparability blocker, not silently handled here."
        )

    m8.M9_8_TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = m8.M9_8_TRAINING_RUNS_DIR / f"ARM_S-seed{seed}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return record


def main() -> int:
    results = {}
    for seed in m8.SEEDS:
        print(f"preparing ARM_S seed {seed}...", flush=True)
        record = prepare_arm_s(seed)
        results[seed] = record["checkpoint_provenance"]
        print(json.dumps({
            "seed": seed, "checkpoint_provenance": record["checkpoint_provenance"],
            "canonical_export_sha256": record["canonical_export_sha256"],
            "param_count": record["model_architecture"]["param_count"],
        }, indent=2))
    print("ARM_S preparation complete:", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
