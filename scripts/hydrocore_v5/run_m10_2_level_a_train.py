"""Milestone 10.2 Scout refit -- Level A: frozen-backbone, specialist-side
Scout training under the frozen protocol
(`docs/evaluation/HYDROCORE_V5_M10_2_SCOUT_REFIT_PROTOCOL.md`,
`m10_2_refit_protocol.py`).

Does NOT run the true M10.2 learned-vs-deterministic Scout comparison. Ends
by writing the Level-A representation-sufficiency gate result; this task's
own orchestration (not this script) decides whether to escalate to Level B.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-2-refit/m10-2-refit-gradient-coverage.json
  reports/evaluation/hydrocore-v5/m10/m10-2-refit/m10-2-refit-level-a.json
  reports/evaluation/hydrocore-v5/m10/m10-2-refit/checkpoints/level-a-seed{seed}/
    model.safetensors, checkpoint_identity.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import m10_2_refit_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import _degradation_probabilities, fit_pool_signature_library  # noqa: E402
from hydroswarm.training.gradient_coverage import compute_gradient_coverage, require_gradient_coverage  # noqa: E402
from hydroswarm.training.losses import compute_multitask_loss  # noqa: E402
from hydroswarm.training.scenario_reconstruction import ScenarioReconstructionError, reconstruct_scenario_network  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402
from hydroswarm.training.scout_targets import ScoutTrainingExample, build_scout_training_examples  # noqa: E402
from hydroswarm.training.trainer import set_deterministic_seed  # noqa: E402

M10_2_REFIT_DIR = m10.M10_DIR / "m10-2-refit"
CHECKPOINTS_DIR = M10_2_REFIT_DIR / "checkpoints"
SIGNATURE_CACHE_DIR = ROOT / "experiments" / "cache" / "m10-2-refit-signatures"

TASK_WEIGHTS: dict[str, float] = {
    "sample_node": 1.0, "information_gain": 0.5, "candidate_reduction": 0.5, "should_continue_sampling": 0.5,
}


# --------------------------------------------------------------------------
# Corpus assembly (built once, reused across all 3 seeds -- independent of
# teacher checkpoint weights).
# --------------------------------------------------------------------------


def _build_pool_and_examples(*, seed_base: int, count: int, network, loader, input_library, artifact) -> list[ScoutTrainingExample]:
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=proto.FAMILY, seed_base=seed_base, count=count,
        source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    examples: list[ScoutTrainingExample] = []
    skipped = 0
    for record in pool:
        try:
            reconstruction = reconstruct_scenario_network(
                network, record.scenario.manifest, degradation_policy=_degradation_probabilities, original=record.scenario,
            )
        except ScenarioReconstructionError:
            skipped += 1
            continue
        node_ids = tuple(sorted(network.node_name_list))
        examples.extend(
            build_scout_training_examples(
                record.scenario, record.network, input_library, artifact, node_ids, proto.DEPTH,
                reconstruction=reconstruction, maximum_samples=proto.MAXIMUM_SAMPLES,
                noise_scale_mg_l=proto.NOISE_SCALE_MG_L, feature_context=record.feature_context,
            )
        )
    return examples


def _collate(examples: list[ScoutTrainingExample]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    keys = set(examples[0].state.batch)
    batch = {key: torch.cat([ex.state.batch[key] for ex in examples], dim=0) for key in keys}
    target_keys = set(examples[0].targets)
    targets = {key: torch.stack([ex.targets[key] for ex in examples], dim=0) for key in target_keys}
    return batch, targets


# --------------------------------------------------------------------------
# Level-A training loop (frozen protocol: Adam lr=1e-3, batch_size=8,
# epochs=20, final-epoch checkpoint selection, exact 18-parameter allowlist).
# --------------------------------------------------------------------------


def _apply_level_a_allowlist(model: HydroCore) -> None:
    allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowlist)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == allowlist, f"allowlist mismatch: extra={trainable - allowlist}, missing={allowlist - trainable}"


def _train_level_a(model: HydroCore, train_examples: list[ScoutTrainingExample], *, seed: int) -> dict[str, Any]:
    _apply_level_a_allowlist(model)
    set_deterministic_seed(seed, deterministic=False)  # frozen backbone + tiny heads: determinism via seeded shuffle only
    trainable_params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=proto.LEARNING_RATE, weight_decay=proto.WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    n = len(train_examples)
    epoch_losses: list[float] = []
    model.train()
    for epoch in range(proto.EPOCHS):
        order = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, proto.BATCH_SIZE):
            indices = order[start:start + proto.BATCH_SIZE]
            batch_examples = [train_examples[i] for i in indices]
            inputs, targets = _collate(batch_examples)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            result = compute_multitask_loss(output, targets, task_weights=TASK_WEIGHTS)
            if not torch.isfinite(result.total):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            result.total.backward()
            optimizer.step()
            total_loss += float(result.total)
            n_batches += 1
        epoch_losses.append(total_loss / max(1, n_batches))
    model.eval()
    return {"epoch_losses": epoch_losses, "final_epoch": proto.EPOCHS - 1, "n_train_examples": n}


# --------------------------------------------------------------------------
# Checkpoint / provenance.
# --------------------------------------------------------------------------


def _save_refit_checkpoint(
    model: HydroCore, *, seed: int, teacher_sha256: str, train_manifest_hash: str, validation_manifest_hash: str,
    gradient_coverage_hash: str, level: str,
) -> dict[str, Any]:
    out_dir = CHECKPOINTS_DIR / f"level-a-seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    model_path = out_dir / "model.safetensors"
    save_file(tensors, model_path)
    model_sha256 = __import__("hashlib").sha256(model_path.read_bytes()).hexdigest()
    identity = {
        "kind": "M10_2_SCOUT_REFIT_CHECKPOINT_IDENTITY",
        "parent_m9_6_checkpoint_sha256": teacher_sha256,
        "refit_level": level,
        "trainable_parameter_allowlist": list(proto.LEVEL_A_PARAMETER_ALLOWLIST),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_state_schema_version": proto.TRAINING_STATE_SCHEMA_VERSION,
        "target_schema_version": proto.TARGET_SCHEMA_VERSION,
        "train_manifest_hash": train_manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
        "seed": seed,
        "optimizer_config_hash": __import__("hashlib").sha256(
            json.dumps({"optimizer": proto.OPTIMIZER, "lr": proto.LEARNING_RATE, "weight_decay": proto.WEIGHT_DECAY,
                        "batch_size": proto.BATCH_SIZE, "epochs": proto.EPOCHS}, sort_keys=True).encode()
        ).hexdigest(),
        "checkpoint_selection_policy": proto.CHECKPOINT_SELECTION,
        "gradient_coverage_certificate_hash": gradient_coverage_hash,
        "git_commit": m10.current_commit(),
        "model_sha256": model_sha256,
        "never_call_this_m9_6": True,
    }
    (out_dir / "checkpoint_identity.json").write_text(json.dumps(identity, indent=2, default=str) + "\n")
    return identity


def _manifest_hash(examples: list[ScoutTrainingExample], seed_base: int, count: int) -> str:
    payload = {"seed_base": seed_base, "count": count, "n_examples": len(examples)}
    return __import__("hashlib").sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> None:
    M10_2_REFIT_DIR.mkdir(parents=True, exist_ok=True)
    branch = m10.current_branch()
    assert branch == m10.FROZEN_BRANCH
    locked_before = m10.assert_locked_test_closed()

    family, loader = TRAINED_FAMILIES[0]
    assert family == proto.FAMILY
    network = loader()

    print("fitting input signature library and building heavy signature artifact...", flush=True)
    train_pool_for_library = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=proto.TRAIN_SEED_BASE, count=proto.TRAIN_COUNT,
        source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    input_library = fit_pool_signature_library(train_pool_for_library)

    cache = SignatureCache(str(SIGNATURE_CACHE_DIR))
    key = SignatureCacheKey(
        network_hash="m10-2-refit-golden-reference", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="m10-2-refit-cfg1", sensor_layout_hash="m10-2-refit-layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)

    t0 = time.time()
    print("building TRAIN examples...", flush=True)
    train_examples: list[ScoutTrainingExample] = []
    for record in train_pool_for_library:
        try:
            reconstruction = reconstruct_scenario_network(
                network, record.scenario.manifest, degradation_policy=_degradation_probabilities, original=record.scenario,
            )
        except ScenarioReconstructionError:
            continue
        node_ids = tuple(sorted(network.node_name_list))
        train_examples.extend(
            build_scout_training_examples(
                record.scenario, record.network, input_library, artifact, node_ids, proto.DEPTH,
                reconstruction=reconstruction, maximum_samples=proto.MAXIMUM_SAMPLES,
                noise_scale_mg_l=proto.NOISE_SCALE_MG_L, feature_context=record.feature_context,
            )
        )
    print(f"TRAIN: {len(train_examples)} examples from {len(train_pool_for_library)} scenarios, {time.time()-t0:.1f}s", flush=True)

    print("building VALIDATION examples...", flush=True)
    t0 = time.time()
    validation_examples = _build_pool_and_examples(
        seed_base=proto.VALIDATION_SEED_BASE, count=proto.VALIDATION_COUNT, network=network, loader=loader,
        input_library=input_library, artifact=artifact,
    )
    print(f"VALIDATION: {len(validation_examples)} examples, {time.time()-t0:.1f}s", flush=True)

    train_manifest_hash = _manifest_hash(train_examples, proto.TRAIN_SEED_BASE, proto.TRAIN_COUNT)
    validation_manifest_hash = _manifest_hash(validation_examples, proto.VALIDATION_SEED_BASE, proto.VALIDATION_COUNT)

    corpus_summary = {
        "kind": "M10_2_REFIT_CORPUS_SUMMARY",
        "n_train_scenarios": len(train_pool_for_library),
        "n_train_examples": len(train_examples),
        "n_train_examples_with_real_recommendation": sum(1 for ex in train_examples if ex.label.sample_node_id is not None),
        "n_validation_examples": len(validation_examples),
        "n_validation_examples_with_real_recommendation": sum(1 for ex in validation_examples if ex.label.sample_node_id is not None),
        "train_manifest_hash": train_manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
    }
    print(json.dumps(corpus_summary, indent=2))

    gradient_coverage_docs: dict[str, Any] = {}
    level_a_results: dict[str, Any] = {}
    for seed in m10.SEEDS:
        print(f"=== Level A, seed {seed} ===", flush=True)
        record = m10.canonical_s_checkpoint(seed)
        model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
        model.load_state_dict(load_file(record["canonical_export_path"], device="cpu"), strict=True)

        train_result = _train_level_a(model, train_examples, seed=seed)
        print(f"  final-epoch mean loss: {train_result['epoch_losses'][-1]:.4f}", flush=True)

        allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
        parameter_groups = {
            "sample_node": [n for n in allowlist if n.startswith("sample_node_head")],
            "information_gain": [n for n in allowlist if n.startswith("information_gain_head")],
            "candidate_reduction": [n for n in allowlist if n.startswith("candidate_reduction_head")],
            "should_continue_sampling": [n for n in allowlist if n.startswith("should_continue_sampling_head")],
        }
        batch_examples = train_examples[: proto.BATCH_SIZE]
        inputs, targets = _collate(batch_examples)
        certificates = compute_gradient_coverage(
            model, lambda m, inputs=inputs: m(inputs), targets, task_weights=TASK_WEIGHTS,
            parameter_groups=parameter_groups, min_valid_target_count=1, verify_parameter_update=True,
            update_lr=proto.LEARNING_RATE,
        )
        cert_doc = {task: cert.to_dict() for task, cert in certificates.items()}
        gradient_coverage_hash = __import__("hashlib").sha256(
            json.dumps(cert_doc, sort_keys=True, default=str).encode()
        ).hexdigest()
        gradient_coverage_docs[str(seed)] = {"certificates": cert_doc, "hash": gradient_coverage_hash}
        try:
            require_gradient_coverage(certificates)
            gradient_coverage_passed = True
        except Exception as error:  # noqa: BLE001
            gradient_coverage_passed = False
            print(f"  GRADIENT COVERAGE FAILED: {error}", flush=True)

        # Mechanical allowlist-exactness re-assertion (post-training).
        trainable_now = {name for name, p in model.named_parameters() if p.requires_grad}
        assert trainable_now == allowlist

        identity = _save_refit_checkpoint(
            model, seed=seed, teacher_sha256=record["canonical_export_sha256"],
            train_manifest_hash=train_manifest_hash, validation_manifest_hash=validation_manifest_hash,
            gradient_coverage_hash=gradient_coverage_hash, level="A",
        )
        level_a_results[str(seed)] = {
            "train_epoch_losses": train_result["epoch_losses"],
            "n_train_examples": train_result["n_train_examples"],
            "gradient_coverage_passed": gradient_coverage_passed,
            "checkpoint_identity": identity,
        }

    locked_after = m10.assert_locked_test_closed()
    doc = {
        "kind": "M10_2_REFIT_LEVEL_A_TRAINING",
        "branch": branch, "commit": m10.current_commit(),
        "protocol_hash": proto.protocol_hash(),
        "corpus_summary": corpus_summary,
        "per_seed": level_a_results,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    (M10_2_REFIT_DIR / "m10-2-refit-level-a.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    (M10_2_REFIT_DIR / "m10-2-refit-gradient-coverage.json").write_text(json.dumps(gradient_coverage_docs, indent=2, default=str) + "\n")
    print("Level A training complete.")


if __name__ == "__main__":
    main()
