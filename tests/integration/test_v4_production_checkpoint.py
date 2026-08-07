"""core-issues4.txt Section H: real (not synthetic-shape) v4 checkpoint
integration coverage --

- "one real multi-topology v4 training smoke run": trains a real
  HydroCore("small", event_control_heads=True) for real steps against real
  multi-topology examples drawn from data/learning-v2/cycle-b2-control-v2
  (branched-loop, golden-reference, loop-grid all present in Cycle B2).
- "checkpoint save/resume/reload test": the Trainer-level generic
  checkpoint/resume path, exercised against this real run (not a synthetic
  fixture -- test_event_control_smoke_screening.py's own pattern, reused
  here specifically for the v4 identity layer).
- "production-factory construction test using a real v4 artifact": wraps
  the resulting real, trained model in an actual v4 identity
  (build_checkpoint_identity), writes it through the real save_v4_checkpoint
  path, and reconstructs it via load_v4_checkpoint -- proving the complete
  v4 identity contract (Section A/B) round-trips a genuinely trained model,
  not only the tiny random-weight fixtures test_checkpoint_identity.py uses.

Deliberately does NOT depend on any specific background/long-running
experiment job's output path (those are ephemeral, not committed, and can
be cleaned up) -- everything this test needs, it trains itself, in seconds,
from the real committed data/learning-v2/cycle-b2-control-v2 corpus.
"""

from __future__ import annotations

from pathlib import Path

import torch

from hydroswarm.model.core import HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.training import (
    GovernedScenarioDataset,
    ShardedScenarioDataset,
    Trainer,
    TrainingConfig,
    collate_variable_topology,
)
from hydroswarm.training.checkpoint_identity import build_checkpoint_identity, load_v4_checkpoint, save_v4_checkpoint

_CORPUS_ROOT = Path("data/learning-v2/cycle-b2-control-v2/tensors-normalized")
_OVERRIDES = dict(prior_mode="feature_only", event_control_heads=True, action_vocabulary_size=ACTION_TEMPLATE_COUNT)


def _real_multi_topology_subset(split: str, count: int) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(_CORPUS_ROOT / split, expected_split=split)
    # cycle-b2's shards are grouped by network_id in generation order (the
    # first N examples of a split are typically all one topology family) --
    # stride across the full split so a small subset genuinely spans
    # multiple real topologies, rather than sampling only the first count
    # contiguous rows.
    stride = max(1, len(dataset) // count)
    indices = list(range(0, len(dataset), stride))[:count]
    examples = [dataset[index] for index in indices]
    families = {example.network_id for example in examples}
    assert len(families) >= 2, f"expected a real multi-topology subset, got only {families}"
    return GovernedScenarioDataset(examples, expected_split=split)


def test_real_multi_topology_v4_training_save_resume_and_production_reload(tmp_path: Path) -> None:
    train = _real_multi_topology_subset("train", 24)
    validation = _real_multi_topology_subset("validation", 8)

    run_root = tmp_path / "run"
    config = TrainingConfig(
        seed=20260807,
        epochs=2,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=2,
        checkpoint_every_epochs=1,
        maximum_runtime_seconds=120.0,
        gradnorm_logging=False,
    )
    trainer = Trainer(
        HydroCore.from_variant("small", **_OVERRIDES),
        train,
        validation_dataset=validation,
        config=config,
        run_root=run_root,
        workdir=".",
        collate_fn=collate_variable_topology,
    )
    first = trainer.fit()
    assert first.epochs_completed == 2
    assert torch.isfinite(torch.tensor(first.best_validation_loss))

    # --- checkpoint save/resume/reload -----------------------------------
    resumed_config = TrainingConfig(**{**config.as_dict(), "epochs": 3})
    resumed_trainer = Trainer(
        HydroCore.from_variant("small", **_OVERRIDES),
        train,
        validation_dataset=validation,
        config=resumed_config,
        run_root=run_root,
        workdir=".",
        collate_fn=collate_variable_topology,
    )
    resumed = resumed_trainer.fit(resume_from=Path(first.final_checkpoint))
    assert resumed.epochs_completed == 3
    assert resumed.global_steps > first.global_steps

    # --- production-factory construction from a REAL v4 artifact ---------
    from safetensors.torch import load_file

    trained_model = HydroCore.from_variant("small", **_OVERRIDES)
    trained_model.load_state_dict(load_file(str(Path(resumed.final_checkpoint) / "model.safetensors")), strict=True)
    trained_model.eval()

    validation_examples = [validation[index] for index in range(len(validation))]
    inputs, _targets = collate_variable_topology(validation_examples)
    with torch.no_grad():
        pre_save_output = trained_model(inputs)

    identity = build_checkpoint_identity(
        trained_model,
        normalization_hash="cycle-b2-control-v2-tensors-normalized",
        fusion_policy_hash="fixed-weight-v1:neural=0.6",
        source_corpus_manifest_hashes=(train.manifest_hash, validation.manifest_hash),
        trained_outputs=frozenset({"source_node", "evidence_sufficiency", "next_step"}),
        # This is a real trained artifact but has not been through a
        # promotion-gate evaluation in this test -- validated/runtime_enabled
        # correctly stay empty (core-issues2.txt Phase 10: "Unvalidated
        # heads must contribute exactly zero to runtime decisions").
        validated_outputs=frozenset(),
        runtime_enabled_outputs=frozenset(),
    )

    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    v4_dir = tmp_path / "v4-artifact"
    save_v4_checkpoint(
        v4_dir,
        model=trained_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=resumed.epochs_completed,
        global_step=resumed.global_steps,
        best_validation_loss=resumed.best_validation_loss,
        identity=identity,
        resolved_training_config=resumed_config.as_dict(),
        dataset_manifest_hashes={"train": train.manifest_hash, "validation": validation.manifest_hash},
        task_weights=resumed_config.task_weights,
        workdir=".",
    )
    assert (v4_dir / "checkpoint_identity.json").exists()
    assert (v4_dir / "artifact_manifest.json").exists()

    reloaded_model, reloaded_identity, _metadata = load_v4_checkpoint(v4_dir)
    reloaded_model.eval()
    assert reloaded_identity.fingerprint() == identity.fingerprint()
    assert reloaded_identity == identity

    with torch.no_grad():
        post_reload_output = reloaded_model(inputs)

    for key in ("source_node_logits", "evidence_sufficiency", "next_step_logits"):
        assert torch.allclose(pre_save_output[key], post_reload_output[key], atol=1e-6), (
            f"{key} did not round-trip exactly through save_v4_checkpoint/load_v4_checkpoint"
        )
