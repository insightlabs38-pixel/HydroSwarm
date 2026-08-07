from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import signal
from pathlib import Path

import pytest
import torch

from hydroswarm.model import HydroCore
from hydroswarm.training import (
    CurriculumStage,
    GovernedScenarioDataset,
    ScenarioExample,
    ShardedScenarioDataset,
    Trainer,
    TrainingConfig,
    collate_variable_topology,
    write_shards,
)


def _tiny_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        edge_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        latent_tokens=64,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
    )


def _smoke_examples() -> list[ScenarioExample]:
    examples = []
    for index in range(2):
        generator = torch.Generator().manual_seed(100 + index)
        examples.append(
            ScenarioExample(
                scenario_id=f"smoke-{index}",
                network_id="Net1",
                split="train",
                seed=index,
                seed_family=f"smoke-family-{index}",
                stage=CurriculumStage.CLEAN,
                inputs={
                    "node_features": torch.randn(3, 3, generator=generator),
                    "temporal_features": torch.randn(2, 3, 2, generator=generator),
                    "quality_features": torch.randn(2, 3, 2, generator=generator),
                    "travel_time": torch.tensor([0.0, 1.0, 2.0]),
                    "reservoir_reachability": torch.tensor([1.0, 0.5, 0.0]),
                    "demand_centrality": torch.tensor([0.1, 0.2, 0.3]),
                    "node_mask": torch.ones(3, dtype=torch.bool),
                },
                targets={"source_node": torch.tensor(index)},
            )
        )
    return examples


def _dataset() -> GovernedScenarioDataset:
    return GovernedScenarioDataset(_smoke_examples(), expected_split="train")


def _variable_topology_dataset() -> GovernedScenarioDataset:
    # Two examples with genuinely different node counts (3 vs 5), the way
    # a real multi-topology corpus (e.g. Cycle A) mixes examples from
    # different networks within one split -- collate_scenarios (plain
    # torch.stack) cannot batch these; collate_variable_topology pads to
    # the batch's max node/edge count instead.
    examples = []
    for index, nodes in enumerate((3, 5)):
        generator = torch.Generator().manual_seed(200 + index)
        edges = [(node, node + 1) for node in range(nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long)
        examples.append(
            ScenarioExample(
                scenario_id=f"vartopo-{index}",
                network_id=f"net-{nodes}",
                split="train",
                seed=index,
                seed_family=f"vartopo-family-{index}",
                stage=CurriculumStage.CLEAN,
                inputs={
                    "node_features": torch.randn(nodes, 3, generator=generator),
                    "temporal_features": torch.randn(2, nodes, 2, generator=generator),
                    "quality_features": torch.randn(2, nodes, 2, generator=generator),
                    "edge_index": edge_index,
                    "edge_features": torch.randn(len(edges), 2, generator=generator),
                    "source_candidate_mask": torch.ones(nodes, dtype=torch.bool),
                },
                targets={"source_node": torch.tensor(0)},
            )
        )
    return GovernedScenarioDataset(examples, expected_split="train")


def _config(epochs: int) -> TrainingConfig:
    return TrainingConfig(
        seed=7,
        epochs=epochs,
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=0,
        checkpoint_every_epochs=1,
        early_stopping_patience=0,
        maximum_runtime_seconds=60,
        minimum_free_disk_gb=0,
        gradnorm_logging=True,
    )


def test_cpu_smoke_training_checkpoint_export_and_resume(tmp_path: Path) -> None:
    first = Trainer(
        _tiny_model(),
        _dataset(),
        config=_config(1),
        run_root=tmp_path / "runs",
        workdir=tmp_path,
    ).fit()
    checkpoint = Path(first.final_checkpoint)
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "optimizer_state.pt").is_file()
    assert Path(first.export_path).is_file()
    assert json.loads((Path(first.run_directory) / "status.json").read_text())["state"] == "COMPLETED"

    resumed = Trainer(
        _tiny_model(),
        _dataset(),
        config=_config(2),
        run_root=tmp_path / "runs",
        workdir=tmp_path,
    ).fit(resume_from=checkpoint)
    assert resumed.epochs_completed == 2
    assert resumed.global_steps > first.global_steps
    assert Path(resumed.final_checkpoint, "model.safetensors").is_file()


def test_trainer_accepts_a_custom_collate_fn_for_variable_topology_batches(tmp_path: Path) -> None:
    # Bundle E smoke-job prerequisite: a genuinely multi-topology corpus
    # (different node counts within one split) requires
    # collate_variable_topology, which the pre-existing Trainer had no way
    # to select -- it hardcoded collate_scenarios (plain torch.stack),
    # which raises on mismatched shapes. This proves the override actually
    # reaches the DataLoader and produces a normal, finite-loss run.
    summary = Trainer(
        _tiny_model(),
        _variable_topology_dataset(),
        config=_config(1),
        run_root=tmp_path / "runs",
        workdir=tmp_path,
        collate_fn=collate_variable_topology,
    ).fit()
    assert Path(summary.final_checkpoint, "model.safetensors").is_file()
    assert math.isfinite(summary.best_validation_loss)


def test_runtime_budget_exports_best_completed_epoch(tmp_path: Path, monkeypatch) -> None:
    trainer = Trainer(
        _tiny_model(),
        _dataset(),
        config=_config(2),
        run_root=tmp_path / "budget-runs",
        workdir=tmp_path,
    )
    original = trainer._train_epoch
    calls = 0

    def budgeted_epoch(dataset, *, epoch: int, started: float) -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("maximum training runtime exceeded")
        return original(dataset, epoch=epoch, started=started)

    monkeypatch.setattr(trainer, "_train_epoch", budgeted_epoch)
    summary = trainer.fit()

    assert summary.stop_reason == "runtime_budget"
    assert summary.epochs_completed == 1
    assert summary.best_epoch == 0
    assert summary.final_checkpoint == ""
    assert Path(summary.export_path).is_file()
    # core-issues.txt repair item 11: final_checkpoint being "" must not
    # mean losing the periodic checkpoint this run did save (epoch 0, with
    # checkpoint_every_epochs=1), and export_path's own SHA-256 must be
    # recorded alongside it rather than left for the caller to recompute
    # (or, as the real bug was, never recorded at all).
    assert summary.last_resumable_checkpoint != ""
    assert (Path(summary.last_resumable_checkpoint) / "model.safetensors").is_file()
    assert summary.export_sha256 == hashlib.sha256(Path(summary.export_path).read_bytes()).hexdigest()
    status = json.loads((Path(summary.run_directory) / "status.json").read_text())
    assert status["state"] == "COMPLETED"
    assert status["stop_reason"] == "runtime_budget"


def test_sigterm_stops_cleanly_and_saves_a_resumable_checkpoint(tmp_path: Path) -> None:
    trainer = Trainer(
        _tiny_model(),
        _dataset(),
        config=_config(5),
        run_root=tmp_path / "sigterm-runs",
        workdir=tmp_path,
    )
    trainer.install_signal_handlers()
    original = trainer._train_epoch
    calls = 0

    def epoch_that_receives_sigterm(dataset, *, epoch: int, started: float) -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return original(dataset, epoch=epoch, started=started)

    trainer._train_epoch = epoch_that_receives_sigterm
    summary = trainer.fit()

    assert trainer._shutdown_requested is True
    assert summary.stop_reason == "runtime_budget"
    assert summary.epochs_completed == 1
    assert Path(summary.export_path).is_file()
    status = json.loads((Path(summary.run_directory) / "status.json").read_text())
    assert status["state"] == "COMPLETED"


def test_trainer_trains_directly_from_a_sharded_scenario_dataset(tmp_path: Path) -> None:
    # core-issues.txt repair item 12: Trainer must accept a lazy,
    # disk-backed ShardedScenarioDataset directly -- not require every
    # caller to first materialize it into a GovernedScenarioDataset (an
    # in-memory list of every ScenarioExample) purely to satisfy a type
    # hint. This also exercises verify_shard_checksums as the Stage 2/3/4
    # scripts now call it.
    write_shards(_smoke_examples(), tmp_path / "shards", shard_size=1)
    sharded = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    sharded.verify_shard_checksums()

    summary = Trainer(
        _tiny_model(),
        sharded,
        config=_config(1),
        run_root=tmp_path / "runs",
        workdir=tmp_path,
    ).fit()

    assert Path(summary.export_path).is_file()
    assert math.isfinite(summary.best_validation_loss)


def test_validation_produces_a_per_epoch_per_task_history(tmp_path: Path) -> None:
    # core-issues3.txt Phase 11.4: "Log gradient norms and primary-task
    # regressions" -- validation_history.jsonl accumulates ONE entry per
    # epoch (unlike epoch_summary.json, which atomic_json always
    # overwrites, so only the latest epoch survives there), each carrying
    # the per-task mean validation loss, not just the overall scalar.
    trainer = Trainer(
        _tiny_model(),
        _dataset(),
        config=_config(2),
        run_root=tmp_path / "runs",
        validation_dataset=_dataset(),
        workdir=tmp_path,
    )
    trainer.fit()
    history = [
        json.loads(line)
        for line in (trainer.artifacts.path / "validation_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [entry["epoch"] for entry in history] == [0, 1]
    for entry in history:
        assert math.isfinite(entry["validation_task_losses"]["source_node"])
        # The smoke dataset carries only one task, so the overall mean
        # equals that task's own mean exactly (up to float summation order).
        assert entry["validation_loss"] == pytest.approx(
            entry["validation_task_losses"]["source_node"], abs=1e-4
        )
    summary = json.loads((trainer.artifacts.path / "epoch_summary.json").read_text())
    assert "source_node" in summary["validation_task_losses"]


def test_gradient_conflict_logging_is_off_by_default_and_calls_the_diagnostic_only_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    import hydroswarm.training.trainer as trainer_module

    calls = []
    real = trainer_module.task_gradient_conflict

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "task_gradient_conflict", _spy)

    off = Trainer(_tiny_model(), _dataset(), config=_config(1), run_root=tmp_path / "off", workdir=tmp_path)
    off.fit()
    assert calls == []
    off_metrics = [
        json.loads(line)
        for line in (off.artifacts.path / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(entry["task_gradient_conflict"] == {} for entry in off_metrics)

    config = dataclasses.replace(_config(1), gradient_conflict_logging=True)
    on = Trainer(_tiny_model(), _dataset(), config=config, run_root=tmp_path / "on", workdir=tmp_path)
    on.fit()
    assert len(calls) > 0


def test_gradnorm_logging_only_runs_on_the_configured_batch_interval(tmp_path: Path) -> None:
    # core-issues.txt repair item 11: task_gradient_norms is expensive (one
    # extra torch.autograd.grad call per task loss, every batch it runs
    # on); gradnorm_log_every_n_batches=2 over this dataset's 2 batches per
    # epoch must compute it for batch 0 only, not batch 1.
    config = dataclasses.replace(_config(1), gradnorm_log_every_n_batches=2)
    trainer = Trainer(
        _tiny_model(), _dataset(), config=config, run_root=tmp_path / "runs", workdir=tmp_path,
    )
    trainer.fit()
    metrics = [
        json.loads(line)
        for line in (trainer.artifacts.path / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_batch = {entry["batch"]: entry["task_gradient_norms"] for entry in metrics if entry["epoch"] == 0}
    assert by_batch[0] != {}
    assert by_batch[1] == {}
