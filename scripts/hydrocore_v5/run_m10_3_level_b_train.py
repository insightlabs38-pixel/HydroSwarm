"""M10.3A Strategist refit -- Level B: ONE predeclared, bounded partial
shared-backbone unfreeze (Level-A allowlist + backbone[3] + final_norm),
executed ONLY because the Level-A gate mechanically triggered escalation
(`m10-3-refit-closure.json`: `M10_3_LEVEL_B_ESCALATION_TRIGGERED`) under
the frozen protocol
(`docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md`,
`m10_3_refit_protocol.py` Section 8).

Warm-starts from the SAME canonical M9.6 teacher checkpoint Level A used
(never from Level A's own outcome-tuned weights) -- per the frozen
protocol's own independent-comparability requirement. Reuses the IDENTICAL
TRAIN/VALIDATION population Level A used (same seed base/count).

Does NOT run the true M10.3 learned-vs-deterministic Strategist
comparison. Ends by writing gradient-coverage certificates and per-seed
Level-B checkpoints; `run_m10_3_level_b_gate.py` (not this script) decides
competence/M9-preservation.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-level-b-gradient-coverage.json
  reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-level-b.json
  reports/evaluation/hydrocore-v5/m10/m10-3-refit/checkpoints/level-b-seed{seed}/
    model.safetensors, checkpoint_identity.json
"""

from __future__ import annotations

import hashlib
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

import m10_3_refit_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import fit_pool_signature_library, scenario_to_prefix_example  # noqa: E402
from hydroswarm.training.gradient_coverage import compute_gradient_coverage, require_gradient_coverage  # noqa: E402
from hydroswarm.training.losses import compute_multitask_loss  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402
from hydroswarm.training.strategist_candidate_corpus import (  # noqa: E402
    StrategistCandidateExample,
    build_strategist_candidate_example,
)
from hydroswarm.training.trainer import set_deterministic_seed  # noqa: E402

M10_3_REFIT_DIR = m10.M10_DIR / "m10-3-refit"
CHECKPOINTS_DIR = M10_3_REFIT_DIR / "checkpoints"
SIGNATURE_CACHE_DIR = ROOT / "experiments" / "cache" / "m10-3-refit-signatures"


class CorpusExample:
    __slots__ = ("scenario_id", "inputs", "targets", "real_plan_count")

    def __init__(self, scenario_id: str, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], real_plan_count: int) -> None:
        self.scenario_id = scenario_id
        self.inputs = inputs
        self.targets = targets
        self.real_plan_count = real_plan_count


def _build_corpus(*, seed_base: int, count: int, network: Any, loader, input_library, artifact) -> list[CorpusExample]:
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=proto.FAMILY, seed_base=seed_base, count=count,
        source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    node_ids = tuple(sorted(network.node_name_list))
    edge_ids = tuple(
        (network.get_link(name).start_node_name, network.get_link(name).end_node_name)
        for name in sorted(network.link_name_list)
    )
    examples: list[CorpusExample] = []
    skipped_no_candidates = 0
    for record in pool:
        candidate: StrategistCandidateExample | None = build_strategist_candidate_example(
            record.scenario, record.network, record.feature_context, artifact, node_ids, edge_ids,
            maximum_plans=proto.MAXIMUM_PLAN_COUNT,
        )
        if candidate is None:
            skipped_no_candidates += 1
            continue
        sentinel_example = scenario_to_prefix_example(
            record.scenario, record.network, input_library, proto.DEPTH, feature_context=record.feature_context,
        )
        sentinel_batch = {key: value.unsqueeze(0) for key, value in sentinel_example.inputs.items()}
        merged_inputs = {**sentinel_batch, **candidate.batch}
        examples.append(CorpusExample(candidate.scenario_id, merged_inputs, candidate.targets, candidate.real_plan_count))
    print(f"  built {len(examples)}/{len(pool)} examples ({skipped_no_candidates} skipped, no candidates)", flush=True)
    return examples


def _collate(examples: list[CorpusExample]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    input_keys = set(examples[0].inputs)
    inputs = {key: torch.cat([ex.inputs[key] for ex in examples], dim=0) for key in input_keys}
    target_keys = set(examples[0].targets)
    targets = {key: torch.stack([ex.targets[key] for ex in examples], dim=0) for key in target_keys}
    return inputs, targets


LEVEL_B_ALLOWLIST: frozenset[str] = frozenset(proto.LEVEL_A_PARAMETER_ALLOWLIST) | frozenset(proto.LEVEL_B_EXTRA_PARAMETER_ALLOWLIST)


def _apply_level_b_allowlist(model: HydroCore) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in LEVEL_B_ALLOWLIST)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == LEVEL_B_ALLOWLIST, f"allowlist mismatch: extra={trainable - LEVEL_B_ALLOWLIST}, missing={LEVEL_B_ALLOWLIST - trainable}"


def _train_level_b(model: HydroCore, train_examples: list[CorpusExample], *, seed: int) -> dict[str, Any]:
    _apply_level_b_allowlist(model)
    set_deterministic_seed(seed, deterministic=False)
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
            result = compute_multitask_loss(output, targets, task_weights=proto.TASK_WEIGHTS)
            if not torch.isfinite(result.total):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            result.total.backward()
            optimizer.step()
            total_loss += float(result.total)
            n_batches += 1
        epoch_losses.append(total_loss / max(1, n_batches))
    model.eval()
    return {"epoch_losses": epoch_losses, "final_epoch": proto.EPOCHS - 1, "n_train_examples": n}


def _save_refit_checkpoint(
    model: HydroCore, *, seed: int, teacher_sha256: str, train_manifest_hash: str, validation_manifest_hash: str,
    gradient_coverage_hash: str, level: str,
) -> dict[str, Any]:
    out_dir = CHECKPOINTS_DIR / f"level-b-seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    model_path = out_dir / "model.safetensors"
    save_file(tensors, model_path)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    identity = {
        "kind": "M10_3_STRATEGIST_REFIT_CHECKPOINT_IDENTITY",
        "parent_m9_6_checkpoint_sha256": teacher_sha256,
        "refit_level": level,
        "trainable_parameter_allowlist": sorted(LEVEL_B_ALLOWLIST),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "candidate_training_schema_version": proto.CANDIDATE_TRAINING_SCHEMA_VERSION,
        "target_keys": list(proto.STRATEGIST_TARGET_KEYS),
        "train_manifest_hash": train_manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
        "seed": seed,
        "optimizer_config_hash": hashlib.sha256(
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


def _manifest_hash(examples: list[CorpusExample], seed_base: int, count: int) -> str:
    payload = {"seed_base": seed_base, "count": count, "n_examples": len(examples),
               "scenario_ids": sorted(ex.scenario_id for ex in examples)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> None:
    M10_3_REFIT_DIR.mkdir(parents=True, exist_ok=True)
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
        network_hash="m10-3-refit-golden-reference", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="m10-3-refit-cfg1", sensor_layout_hash="m10-3-refit-layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)

    t0 = time.time()
    print("building TRAIN examples...", flush=True)
    train_examples = _build_corpus(
        seed_base=proto.TRAIN_SEED_BASE, count=proto.TRAIN_COUNT, network=network, loader=loader,
        input_library=input_library, artifact=artifact,
    )
    print(f"TRAIN: {len(train_examples)} examples, {time.time()-t0:.1f}s", flush=True)

    print("building VALIDATION examples...", flush=True)
    t0 = time.time()
    validation_examples = _build_corpus(
        seed_base=proto.VALIDATION_SEED_BASE, count=proto.VALIDATION_COUNT, network=network, loader=loader,
        input_library=input_library, artifact=artifact,
    )
    print(f"VALIDATION: {len(validation_examples)} examples, {time.time()-t0:.1f}s", flush=True)

    train_manifest_hash = _manifest_hash(train_examples, proto.TRAIN_SEED_BASE, proto.TRAIN_COUNT)
    validation_manifest_hash = _manifest_hash(validation_examples, proto.VALIDATION_SEED_BASE, proto.VALIDATION_COUNT)

    corpus_summary = {
        "kind": "M10_3_REFIT_CORPUS_SUMMARY",
        "n_train_scenarios": proto.TRAIN_COUNT,
        "n_train_examples": len(train_examples),
        "n_validation_scenarios": proto.VALIDATION_COUNT,
        "n_validation_examples": len(validation_examples),
        "train_manifest_hash": train_manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
    }
    print(json.dumps(corpus_summary, indent=2))

    gradient_coverage_docs: dict[str, Any] = {}
    level_b_results: dict[str, Any] = {}
    for seed in m10.SEEDS:
        print(f"=== Level B, seed {seed} ===", flush=True)
        record = m10.canonical_s_checkpoint(seed)
        model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
        model.load_state_dict(load_file(record["canonical_export_path"], device="cpu"), strict=True)

        train_result = _train_level_b(model, train_examples, seed=seed)
        print(f"  final-epoch mean loss: {train_result['epoch_losses'][-1]:.4f}", flush=True)

        shared_params = [name for name in LEVEL_B_ALLOWLIST if name.startswith("candidate_plan_encoder")
                          or name.startswith("backbone.3") or name == "final_norm.weight"]
        task_head_prefix = {
            "plan_validity": "plan_validity_head",
            "plan_value": "plan_value_head",
            "exposure_proxy": "consequence_proxy_heads.exposure_proxy",
            "pressure_risk_proxy": "consequence_proxy_heads.pressure_risk_proxy",
            "service_loss_proxy": "consequence_proxy_heads.service_loss_proxy",
            "containment_time_proxy": "consequence_proxy_heads.containment_time_proxy",
            "plan_regret_proxy": "consequence_proxy_heads.plan_regret_proxy",
        }
        parameter_groups = {
            task: shared_params + [name for name in LEVEL_B_ALLOWLIST if name.startswith(prefix)]
            for task, prefix in task_head_prefix.items()
        }
        batch_examples = train_examples[: proto.BATCH_SIZE]
        inputs, targets = _collate(batch_examples)
        certificates = compute_gradient_coverage(
            model, lambda m, inputs=inputs: m(inputs), targets, task_weights=proto.TASK_WEIGHTS,
            parameter_groups=parameter_groups, min_valid_target_count=1, verify_parameter_update=True,
            update_lr=proto.LEARNING_RATE,
        )
        cert_doc = {task: cert.to_dict() for task, cert in certificates.items()}
        gradient_coverage_hash = hashlib.sha256(
            json.dumps(cert_doc, sort_keys=True, default=str).encode()
        ).hexdigest()
        gradient_coverage_docs[str(seed)] = {"certificates": cert_doc, "hash": gradient_coverage_hash}
        try:
            require_gradient_coverage(certificates)
            gradient_coverage_passed = True
        except Exception as error:  # noqa: BLE001
            gradient_coverage_passed = False
            print(f"  GRADIENT COVERAGE FAILED: {error}", flush=True)

        trainable_now = {name for name, p in model.named_parameters() if p.requires_grad}
        assert trainable_now == LEVEL_B_ALLOWLIST

        identity = _save_refit_checkpoint(
            model, seed=seed, teacher_sha256=record["canonical_export_sha256"],
            train_manifest_hash=train_manifest_hash, validation_manifest_hash=validation_manifest_hash,
            gradient_coverage_hash=gradient_coverage_hash, level="B",
        )
        level_b_results[str(seed)] = {
            "train_epoch_losses": train_result["epoch_losses"],
            "n_train_examples": train_result["n_train_examples"],
            "gradient_coverage_passed": gradient_coverage_passed,
            "checkpoint_identity": identity,
        }

    locked_after = m10.assert_locked_test_closed()
    doc = {
        "kind": "M10_3_REFIT_LEVEL_B_TRAINING",
        "branch": branch, "commit": m10.current_commit(),
        "protocol_hash": proto.protocol_hash(),
        "corpus_summary": corpus_summary,
        "per_seed": level_b_results,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    (M10_3_REFIT_DIR / "m10-3-refit-level-b.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    (M10_3_REFIT_DIR / "m10-3-refit-level-b-gradient-coverage.json").write_text(json.dumps(gradient_coverage_docs, indent=2, default=str) + "\n")
    print("Level B training complete.")


if __name__ == "__main__":
    main()
