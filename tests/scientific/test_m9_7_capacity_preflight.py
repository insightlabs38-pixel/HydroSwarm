"""Milestone 9.7 Section 9: engineering-only preflight for HydroCore-M.

Proves, BEFORE any HydroCore-M development-performance number is observed,
that the governed HydroCore-M capacity config (`MODEL_VARIANTS
["small_v5_capacity_m"]`, `reports/evaluation/hydrocore-v5/m9-7/
m9-7-selected-m-architecture.json`) is a mechanically correct, drop-in
capacity scale of the frozen HydroCore-S recipe. No development_holdout
data, no accuracy number, and no locked split is touched anywhere in this
module -- every check uses synthetic/tiny/train-side-shaped tensors or the
model's own structural properties (parameter count, output shapes,
permutation equivariance, causal masking, gradient wiring, checkpoint
round-trip, scheduler/optimizer-step representability, calibration-interface
acceptance).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator
from hydroswarm.evaluation.live_robustness import locked_test_opened
from hydroswarm.model import HydroCore
from hydroswarm.model.core import MODEL_VARIANTS
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.training.checkpoint import load_checkpoint, save_checkpoint
from hydroswarm.training.config import TrainingConfig
from hydroswarm.training.data import CurriculumStage, ScenarioExample, TopologyMetadata
from hydroswarm.training.permutation import measure_equivariance, permute_example
from hydroswarm.training.trainer import _scheduler
from hydroswarm.training.variable_collate import collate_variable_topology

ROOT = Path(__file__).resolve().parents[2]

SHARED_MODEL_CONFIG = dict(
    prior_mode="feature_only",
    event_control_heads=True,
    scout_control_heads=True,
    strategist_mode="candidate_conditioned",
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True,
    ood_category_head=True,
)
AGE_FIX_ONLY_MODEL_KWARGS = dict(
    temporal_feature_dim=6, quality_feature_dim=4, elapsed_time_normalization="window_relative"
)

S_TOTAL_PARAMETERS = 4_182_612
M_TOTAL_PARAMETERS = 13_919_572
TARGET_LOW, TARGET_HIGH = 12_000_000, 16_000_000

NODE_FEATURE_DIM = 19
EDGE_FEATURE_DIM = 13
TEMPORAL_FEATURE_DIM = 6
QUALITY_FEATURE_DIM = 4
PLAN_FEATURE_DIM = 6


def _build_s() -> HydroCore:
    return HydroCore.from_variant(
        "small", use_adapters=False, **AGE_FIX_ONLY_MODEL_KWARGS, **SHARED_MODEL_CONFIG
    )


def _build_m() -> HydroCore:
    return HydroCore.from_variant(
        "small_v5_capacity_m", use_adapters=False, **AGE_FIX_ONLY_MODEL_KWARGS, **SHARED_MODEL_CONFIG
    )


def _synthetic_example(
    *, seed: int, nodes: int, steps: int = 3, plans: int = 4, source_local_index: int = 0
) -> ScenarioExample:
    generator = torch.Generator().manual_seed(seed)
    edges = [(i, i + 1) for i in range(nodes - 1)] + [(nodes - 1, 0)]  # ring, matches test_permutation.py
    edge_index = torch.tensor(edges, dtype=torch.long).T
    node_ids = tuple(f"J{i}" for i in range(nodes))
    topology = TopologyMetadata(
        topology_hash=f"topo-{seed}",
        network_hash=f"net-{seed}",
        node_ids=node_ids,
        edge_ids=tuple((node_ids[a], node_ids[b]) for a, b in edges),
        source_candidate_ids=node_ids,
        hydraulic_state_hash=f"state-{seed}",
        signature_library_hash=f"sig-{seed}",
        target_schema_version="targets_v1",
        feature_schema_version="hydroswarm-features-v2",
    )
    return ScenarioExample(
        scenario_id=f"m9-7-synthetic-{seed}",
        network_id=f"net-{seed}",
        split="train",
        seed=seed,
        seed_family=f"family-{seed}",
        stage=CurriculumStage.CLEAN,
        inputs={
            "node_features": torch.randn(nodes, NODE_FEATURE_DIM, generator=generator),
            "temporal_features": torch.randn(steps, nodes, TEMPORAL_FEATURE_DIM, generator=generator),
            "quality_features": torch.randn(steps, nodes, QUALITY_FEATURE_DIM, generator=generator),
            "edge_index": edge_index,
            "edge_features": torch.randn(len(edges), EDGE_FEATURE_DIM, generator=generator),
            "travel_time": torch.rand(nodes, generator=generator),
            "reservoir_reachability": torch.rand(nodes, generator=generator),
            "demand_centrality": torch.rand(nodes, generator=generator),
            "source_candidate_mask": torch.ones(nodes, dtype=torch.bool),
            "node_mask": torch.ones(nodes, dtype=torch.bool),
            "plan_template_ids": torch.randint(0, ACTION_TEMPLATE_COUNT, (plans,), generator=generator),
            "plan_target_type": torch.randint(0, 3, (plans,), generator=generator),
            "plan_mask": torch.ones(plans, dtype=torch.bool),
            "plan_features": torch.randn(plans, PLAN_FEATURE_DIM, generator=generator),
        },
        targets={
            "source_node": torch.tensor(source_local_index),
            "sensor_fault": torch.rand(nodes, generator=generator) > 0.8,
            "event_presence": torch.tensor(1.0),
        },
        topology=topology,
    )


# ---------------------------------------------------------------------------
# Locked-split guard (checks 21/22): asserted at both collection edges of
# this module, matching every other v5 scientific test's convention.
# ---------------------------------------------------------------------------


def test_locked_test_unopened_before_module() -> None:
    assert locked_test_opened(ROOT) is False


# ---------------------------------------------------------------------------
# Check 1/2: instantiation + parameter count in the frozen intended range.
# ---------------------------------------------------------------------------


def test_hydrocore_m_instantiates_successfully() -> None:
    model = _build_m()
    assert isinstance(model, HydroCore)
    assert model.variant_name == "small_v5_capacity_m"


def test_hydrocore_m_registered_in_model_variants() -> None:
    assert "small_v5_capacity_m" in MODEL_VARIANTS
    variant = MODEL_VARIANTS["small_v5_capacity_m"]
    assert (variant.d_model, variant.nhead, variant.dim_feedforward, variant.num_layers, variant.latent_tokens) == (
        352, 11, 1056, 4, 64,
    )
    assert variant.modality_layers == 1


def test_hydrocore_m_parameter_count_in_frozen_range() -> None:
    model = _build_m()
    total = sum(p.numel() for p in model.parameters())
    assert total == M_TOTAL_PARAMETERS
    assert TARGET_LOW <= total <= TARGET_HIGH


def test_hydrocore_s_parameter_count_unchanged() -> None:
    """The already-frozen M9.6 HydroCore-S recipe must not be altered by this milestone."""
    model = _build_s()
    total = sum(p.numel() for p in model.parameters())
    assert total == S_TOTAL_PARAMETERS


# ---------------------------------------------------------------------------
# Check 3/4/10: S output shapes == M output shapes; same feature schema
# accepted; no head accidentally changed dimension or semantics.
# ---------------------------------------------------------------------------


def test_s_and_m_accept_identical_batch_schema_and_produce_identical_output_shapes() -> None:
    example = _synthetic_example(seed=1, nodes=6)
    inputs, _targets = collate_variable_topology([example])

    s_model, m_model = _build_s().eval(), _build_m().eval()
    with torch.no_grad():
        s_output = s_model(inputs)
        m_output = m_model(inputs)

    assert set(s_output.keys()) == set(m_output.keys())
    for key in s_output:
        s_shape, m_shape = tuple(s_output[key].shape), tuple(m_output[key].shape)
        if key in ("hidden_state", "latent_state"):
            # These two are the only genuinely d_model-shaped outputs (raw
            # backbone/latent state, not a semantic prediction head) --
            # everything else below must be identical between S and M.
            assert s_shape[:-1] == m_shape[:-1], f"{key}: non-width dims differ {s_shape} vs {m_shape}"
            continue
        assert s_shape == m_shape, f"{key}: shape differs S={s_shape} M={m_shape}"


def test_incident_level_head_widths_are_class_counts_not_d_model() -> None:
    """Every head's trailing dim must be its fixed semantic class/output
    count, matching M9.7 Section 4's 'no head accidentally changes
    dimension or semantics' requirement -- must be identical for S and M
    despite their very different d_model."""
    example = _synthetic_example(seed=2, nodes=5)
    inputs, _targets = collate_variable_topology([example])
    s_model, m_model = _build_s().eval(), _build_m().eval()
    with torch.no_grad():
        s_output = s_model(inputs)
        m_output = m_model(inputs)

    fixed_width_keys = {
        "source_region_logits": 3,
        "start_time_logits": 12,
        "duration_logits": 8,
        "relative_strength_logits": 4,
        "ood_logits": 3,
        "ood_category_logits": 11,
        "event_cause_logits": 5,
        "next_step_logits": 4,
        "action_logits": ACTION_TEMPLATE_COUNT,
        "plan_validity_logits": 2,
        "sentinel": 2,
        "scout": 2,
        "strategist": 3,
    }
    for key, expected_width in fixed_width_keys.items():
        assert s_output[key].shape[-1] == expected_width, key
        assert m_output[key].shape[-1] == expected_width, key


# ---------------------------------------------------------------------------
# Check 5: variable topology still works, for both S and M.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build_model", [_build_s, _build_m])
def test_variable_topology_still_works(build_model) -> None:
    model = build_model().eval()
    small = _synthetic_example(seed=3, nodes=4)
    large = _synthetic_example(seed=4, nodes=11)
    with torch.no_grad():
        small_out = model(collate_variable_topology([small])[0])
        large_out = model(collate_variable_topology([large])[0])
    assert small_out["source_node_logits"].shape == (1, 4)
    assert large_out["source_node_logits"].shape == (1, 11)
    # A genuine incident-level classification has no node axis, unlike
    # source_node_logits, and must not scale with node count either.
    assert small_out["source_region_logits"].shape == (1, 3)
    assert large_out["source_region_logits"].shape == (1, 3)


# ---------------------------------------------------------------------------
# Check 6: node permutation invariance/equivariance -- same guarantee S
# already has (tests/unit/test_permutation.py), reproduced against M.
# ---------------------------------------------------------------------------


def test_hydrocore_m_graph_permutation_equivariance() -> None:
    model = _build_m().eval()
    example = _synthetic_example(seed=5, nodes=6, source_local_index=2)
    permutation = [3, 1, 4, 0, 5, 2]

    report = measure_equivariance(model, example, permutation, collate_fn=collate_variable_topology, atol=1e-3)
    assert report.non_equivariant_keys == ()
    assert report.predicted_source_agrees


def test_hydrocore_m_source_region_logits_invariant_under_permutation() -> None:
    model = _build_m().eval()
    example = _synthetic_example(seed=6, nodes=6, source_local_index=1)
    permutation = [5, 4, 3, 2, 1, 0]
    permuted = permute_example(example, permutation)

    with torch.no_grad():
        original = model(collate_variable_topology([example])[0])["source_region_logits"]
        again = model(collate_variable_topology([permuted])[0])["source_region_logits"]
    assert torch.allclose(original, again, atol=1e-4)


# ---------------------------------------------------------------------------
# Check 7: causality preserved -- content placed after a masked-out
# timestep must never influence the pooled temporal representation, for
# both S and M (this is TemporalEncoder's own masked-mean mechanism,
# unaffected by d_model, but re-verified end-to-end against the real M
# config here rather than assumed).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build_model", [_build_s, _build_m])
def test_causal_masking_ignores_content_beyond_the_valid_window(build_model) -> None:
    model = build_model().eval()
    example = _synthetic_example(seed=7, nodes=5, steps=4)
    inputs, _targets = collate_variable_topology([example])

    # Only the first 2 of 4 timesteps are "observed" -- everything at or
    # beyond index 2 must be causally invisible to the model regardless of
    # what value is placed there.
    sensor_mask = torch.zeros(1, 4, 5, dtype=torch.bool)
    sensor_mask[:, :2, :] = True

    inputs_a = dict(inputs)
    inputs_a["sensor_mask"] = sensor_mask
    inputs_a["quality_mask"] = sensor_mask.clone()

    inputs_b = dict(inputs)
    inputs_b["temporal_features"] = inputs["temporal_features"].clone()
    inputs_b["temporal_features"][:, 2:, :, :] = torch.randn_like(inputs_b["temporal_features"][:, 2:, :, :]) * 999.0
    inputs_b["quality_features"] = inputs["quality_features"].clone()
    inputs_b["quality_mask"] = sensor_mask.clone()
    inputs_b["sensor_mask"] = sensor_mask

    with torch.no_grad():
        output_a = model(inputs_a)
        output_b = model(inputs_b)

    assert torch.allclose(output_a["source_node_logits"], output_b["source_node_logits"], atol=1e-4)


# ---------------------------------------------------------------------------
# Check 8: AGE_FIX_ONLY preserved -- both S and M are constructed with the
# same model-side elapsed_time_normalization; the feature-builder-side
# unobserved_age_sentinel="fixed" fix is architecture-independent by
# construction (HydraulicFeatureBuilder never receives a model instance).
# ---------------------------------------------------------------------------


def test_age_fix_only_model_side_config_identical_for_s_and_m() -> None:
    s_config = _build_s().architecture_config()
    m_config = _build_m().architecture_config()
    assert s_config["elapsed_time_normalization"] == "window_relative"
    assert m_config["elapsed_time_normalization"] == "window_relative"
    assert s_config["temporal_feature_dim"] == m_config["temporal_feature_dim"] == 6
    assert s_config["quality_feature_dim"] == m_config["quality_feature_dim"] == 4


# ---------------------------------------------------------------------------
# Check 9: every supervised head receives a gradient.
# ---------------------------------------------------------------------------


#: Parameters that are structurally inactive under the frozen M9.6
#: SHARED_MODEL_CONFIG/AGE_FIX_ONLY combination regardless of what batch
#: content is supplied -- not a wiring gap, and identical for S and M:
#: `plan_query_tokens` is only read in `strategist_mode="anonymous_queries"`
#: (the frozen config uses "candidate_conditioned"); `prior_logit_scale` is
#: only read when `prior_mode in ("logit_only", "feature_and_logit")` (the
#: frozen config uses "feature_only").
KNOWN_INACTIVE_UNDER_FROZEN_CONFIG = frozenset({"plan_query_tokens", "prior_logit_scale"})


def test_all_supervised_heads_receive_gradients() -> None:
    model = _build_m().train()
    example = _synthetic_example(seed=8, nodes=6, plans=3)
    inputs, _targets = collate_variable_topology([example])
    batch_size, nodes = inputs["node_features"].shape[:2]
    generator = torch.Generator().manual_seed(9)
    # Activate every optional-context projection (residual/prior/role/
    # previous_actions/verifier_feedback) so this check exercises the full
    # parameter set a real training batch can reach, not just the fields
    # _synthetic_example happens to always populate.
    inputs["residual_features"] = torch.randn(batch_size, nodes, 4, generator=generator)
    inputs["classical_prior"] = torch.softmax(torch.randn(batch_size, nodes, generator=generator), dim=-1)
    inputs["role_features"] = torch.randn(batch_size, 8, generator=generator)
    inputs["previous_actions"] = torch.randn(batch_size, 8, generator=generator)
    inputs["verifier_feedback"] = torch.randn(batch_size, 8, generator=generator)

    output = model(inputs)
    # Sum every produced output tensor (not compute_multitask_loss, which
    # only scores tasks that HAVE a target -- this milestone's concern is
    # architecture WIRING, i.e. every head that HydroCore.forward() actually
    # produces an output for must sit on a differentiable path to that
    # output, regardless of which tasks a given training run happens to
    # supply labels for). plan_* tensors were supplied above so the
    # candidate_conditioned plan-scoring heads are included too.
    # tanh-bounded before summing: several heads mask padded/invalid
    # positions with torch.finfo(dtype).min, and summing enough such
    # near--3.4e38 entries directly overflows to -inf -- tanh squashes
    # every element into (-1, 1) (still differentiable everywhere, still
    # zero only at exactly 0) without changing which parameters are on a
    # differentiable path to `total`.
    total = sum(torch.tanh(value.float()).sum() for value in output.values())
    assert torch.isfinite(total)
    total.backward()

    ungraded = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None and name not in KNOWN_INACTIVE_UNDER_FROZEN_CONFIG
    ]
    assert ungraded == [], f"parameters with no gradient after backward: {ungraded}"


# ---------------------------------------------------------------------------
# Checks 11/12: interleaved topology microbatching + gradient accumulation,
# reusing the real M9.0a/M9.6 step function against HydroCore-M unmodified.
# ---------------------------------------------------------------------------


def test_interleaved_microbatching_and_gradient_accumulation_work_for_hydrocore_m() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5"))
    from run_m9_0a_arm_b2 import step_matched_interleaved_optimizer_step  # noqa: PLC0415

    model = _build_m().train()
    config = TrainingConfig.from_yaml(str(ROOT / "configs" / "training-v5-causal.yaml"), require_complete_task_weights=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    families = ("golden-reference", "branched-loop", "loop-grid", "golden-reference")  # 4-slot rotation, matching M9.6
    slot_batches = []
    for index, family in enumerate(families):
        example = _synthetic_example(seed=100 + index, nodes=5)
        inputs, targets = collate_variable_topology([example])
        slot_batches.append((family, (inputs, targets)))

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    result = step_matched_interleaved_optimizer_step(model, optimizer, slot_batches, config=config)
    after = {name: param.detach().clone() for name, param in model.named_parameters()}

    assert len(result["slot_losses"]) == 4
    assert math.isfinite(result["gradient_norm"])
    # A single optimizer.step() (one accumulated update from 4 microbatches)
    # must have moved at least one parameter -- gradient accumulation into
    # ONE update, not 4 separate updates.
    changed = sum(1 for name in before if not torch.equal(before[name], after[name]))
    assert changed > 0


# ---------------------------------------------------------------------------
# Check 13/14: exact 1350-step configuration representable for both S and
# M; scheduler totals match (scheduler is parameter-count-independent).
# ---------------------------------------------------------------------------


def test_1350_step_budget_representable_for_both_s_and_m() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5"))
    import m9_6_common as m6  # noqa: PLC0415

    assert m6.TOTAL_OPTIMIZER_STEPS == 1350
    assert sum(m6.ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == 1350
    # This budget is a pure training-config fact, independent of which
    # model (S or M) consumes it -- representable identically for both.


def test_scheduler_trajectory_is_identical_for_s_and_m() -> None:
    config = TrainingConfig.from_yaml(str(ROOT / "configs" / "training-v5-causal.yaml"), require_complete_task_weights=True)
    s_model, m_model = _build_s(), _build_m()
    s_optimizer = torch.optim.AdamW(s_model.parameters(), lr=config.learning_rate)
    m_optimizer = torch.optim.AdamW(m_model.parameters(), lr=config.learning_rate)

    s_scheduler = _scheduler(s_optimizer, config, total_steps=1500)
    m_scheduler = _scheduler(m_optimizer, config, total_steps=1500)

    s_trajectory, m_trajectory = [], []
    for _ in range(1350):
        s_trajectory.append(s_optimizer.param_groups[0]["lr"])
        m_trajectory.append(m_optimizer.param_groups[0]["lr"])
        s_optimizer.step()
        m_optimizer.step()
        s_scheduler.step()
        m_scheduler.step()

    assert s_trajectory == m_trajectory


# ---------------------------------------------------------------------------
# Checks 15/16: checkpoint save/load works; resume does not alter
# architecture/config identity.
# ---------------------------------------------------------------------------


def test_hydrocore_m_checkpoint_save_load_round_trip(tmp_path: Path) -> None:
    model = _build_m()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    original_config = model.architecture_config()
    directory = tmp_path / "m9-7-m-checkpoint"
    save_checkpoint(directory, model=model, optimizer=optimizer, scheduler=scheduler, epoch=0, global_step=1, best_validation_loss=1.0)

    reloaded = _build_m()
    load_checkpoint(directory, model=reloaded)

    assert reloaded.architecture_config() == original_config
    assert sum(p.numel() for p in reloaded.parameters()) == M_TOTAL_PARAMETERS
    for (name_a, param_a), (name_b, param_b) in zip(model.named_parameters(), reloaded.named_parameters(), strict=True):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


# ---------------------------------------------------------------------------
# Checks 17/18/19/20: calibration interface accepts M outputs without any
# conformal-code modification; B_DEPTH_AWARE grouping / alpha=0.1 /
# source-representative support policy are all model-agnostic by
# construction (they operate on posterior probability vectors and string
# group keys, never on a model instance or its width).
# ---------------------------------------------------------------------------


def test_calibration_interface_accepts_hydrocore_m_posteriors_unmodified() -> None:
    model = _build_m().eval()
    examples = [_synthetic_example(seed=200 + i, nodes=5, source_local_index=i % 5) for i in range(12)]

    calibration_examples = []
    for index, example in enumerate(examples):
        inputs, _targets = collate_variable_topology([example])
        with torch.no_grad():
            posterior = torch.softmax(model(inputs)["source_node_logits"][0], dim=-1).numpy()
        depth_bucket = "EARLY" if index % 2 == 0 else "MATURE"
        calibration_examples.append(
            CalibrationExample(
                probabilities=posterior,
                true_index=0,
                condition=f"golden-reference:{depth_bucket}",  # B_DEPTH_AWARE network_id convention, unchanged
                network_id="golden-reference",
            )
        )

    calibrator = SplitConformalCalibrator.fit(
        calibration_examples,
        alpha=0.1,
        model_hash="m9-7-hydrocore-m-smoke",
        feature_schema_hash="hydroswarm-features-v2",
        dataset_manifest_hash="m9-7-synthetic",
        minimum_group_size=1,
    )
    assert calibrator.artifact.alpha == 0.1


def test_locked_test_unopened_after_module() -> None:
    assert locked_test_opened(ROOT) is False
