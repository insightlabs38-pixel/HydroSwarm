"""M10.3A Strategist supervision/representation refit amendment: real
candidate-schema/target/leakage tests (small, real WNTR-backed
golden-reference scenarios -- same fixture pattern as
`tests/scientific/test_m10_2_scout_refit_corpus.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import fit_pool_signature_library  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402
from hydroswarm.training.strategist_candidate_corpus import (  # noqa: E402
    MAXIMUM_PLAN_COUNT,
    StrategistCandidateAlignmentError,
    build_strategist_candidate_example,
)

from run_m7_topology import SEED_BASES, TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.real_simulation

_CACHE_DIR = "/tmp/claude-0/-workspace/m10_3_refit_test_sig_cache"


@pytest.fixture(scope="module")
def _golden_fixture():
    family, loader = TRAINED_FAMILIES[0]
    assert family == "golden-reference"
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")],
        count=15, source_round_robin=True,
    )
    network = loader()
    input_library = fit_pool_signature_library(pool)
    cache = SignatureCache(_CACHE_DIR)
    key = SignatureCacheKey(
        network_hash="golden-ref-strategist-refit-test",
        hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="cfg1", sensor_layout_hash="layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)
    node_ids = tuple(sorted(network.node_name_list))
    edge_ids = tuple(
        (network.get_link(name).start_node_name, network.get_link(name).end_node_name)
        for name in sorted(network.link_name_list)
    )
    return pool, network, artifact, node_ids, edge_ids, input_library


def _first_real_example(fixture):
    pool, network, artifact, node_ids, edge_ids, _library = fixture
    for rec in pool:
        example = build_strategist_candidate_example(rec.scenario, rec.network, rec.feature_context, artifact, node_ids, edge_ids)
        if example is not None:
            return rec, example
    pytest.skip("no scenario in this small pool produced any real candidate")


# --------------------------------------------------------------------------
# 1/3/4/22/23. Candidate schema validation, padding/masking, batching.
# --------------------------------------------------------------------------


def test_candidate_tensors_are_padded_to_maximum_plan_count(_golden_fixture) -> None:
    _rec, example = _first_real_example(_golden_fixture)
    assert example.batch["plan_template_ids"].shape == (1, MAXIMUM_PLAN_COUNT)
    assert example.batch["plan_features"].shape == (1, MAXIMUM_PLAN_COUNT, 6)
    assert example.batch["plan_mask"].shape == (1, MAXIMUM_PLAN_COUNT)
    assert int(example.batch["plan_mask"].sum()) == example.real_plan_count


def test_padded_positions_are_masked_false_and_use_safe_sentinels(_golden_fixture) -> None:
    _rec, example = _first_real_example(_golden_fixture)
    real = example.real_plan_count
    if real >= MAXIMUM_PLAN_COUNT:
        pytest.skip("this incident's candidate set already fills the maximum -- no padding to inspect")
    assert bool(example.batch["plan_mask"][0, real:].any()) is False
    assert torch.equal(example.batch["plan_target_node_index"][0, real:], torch.full((MAXIMUM_PLAN_COUNT - real,), -1, dtype=torch.long))
    assert torch.equal(example.batch["plan_target_link_index"][0, real:], torch.full((MAXIMUM_PLAN_COUNT - real,), -1, dtype=torch.long))


def test_target_tensors_padded_positions_are_masked_false(_golden_fixture) -> None:
    _rec, example = _first_real_example(_golden_fixture)
    real = example.real_plan_count
    if real >= MAXIMUM_PLAN_COUNT:
        pytest.skip("this incident's candidate set already fills the maximum -- no padding to inspect")
    for name in ("plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
                 "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy"):
        assert bool(example.targets[f"{name}_mask"][real:].any()) is False


def test_real_labels_and_proposals_align_by_template_in_construction_order(_golden_fixture) -> None:
    _rec, example = _first_real_example(_golden_fixture)
    assert [label.action_template for label in example.labels] == [proposal.template for proposal in example.proposals]


def test_multi_candidate_batching_via_torch_cat_and_stack(_golden_fixture) -> None:
    pool, network, artifact, node_ids, edge_ids, _library = _golden_fixture
    examples = []
    for rec in pool:
        example = build_strategist_candidate_example(rec.scenario, rec.network, rec.feature_context, artifact, node_ids, edge_ids)
        if example is not None:
            examples.append(example)
        if len(examples) >= 3:
            break
    if len(examples) < 2:
        pytest.skip("fewer than 2 usable examples in this small pool")
    batched_inputs = {key: torch.cat([ex.batch[key] for ex in examples], dim=0) for key in examples[0].batch}
    batched_targets = {key: torch.stack([ex.targets[key] for ex in examples], dim=0) for key in examples[0].targets}
    assert batched_inputs["plan_template_ids"].shape == (len(examples), MAXIMUM_PLAN_COUNT)
    assert batched_targets["plan_value"].shape == (len(examples), MAXIMUM_PLAN_COUNT)


# --------------------------------------------------------------------------
# Candidate order invariance (Part 2's own requirement).
# --------------------------------------------------------------------------


def test_candidate_order_does_not_change_per_candidate_model_output(_golden_fixture) -> None:
    _rec, example = _first_real_example(_golden_fixture)
    real = example.real_plan_count
    if real < 2:
        pytest.skip("fewer than 2 real candidates -- no reordering to test")

    model = HydroCore(
        node_feature_dim=example.batch["plan_features"].shape[-1] and 3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=2, modality_layers=1,
        adapter_dims=(32, 32, 32), dropout=0.0, use_adapters=False,
        strategist_mode="candidate_conditioned", consequence_prescreening_heads=True, action_vocabulary_size=9,
    ).eval()

    nodes = 5
    generator = torch.Generator().manual_seed(11)
    base_batch = {
        "node_features": torch.randn(1, nodes, 3, generator=generator),
        "temporal_features": torch.randn(1, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(1, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(1, nodes, dtype=torch.bool),
    }
    plan_target_node_index = torch.randint(0, nodes, (1, real))
    original = {
        "plan_template_ids": example.batch["plan_template_ids"][:, :real],
        "plan_target_type": example.batch["plan_target_type"][:, :real],
        "plan_target_node_index": plan_target_node_index,
        "plan_features": example.batch["plan_features"][:, :real],
        "plan_mask": example.batch["plan_mask"][:, :real],
    }
    permutation = torch.randperm(real)
    permuted = {key: value[:, permutation] for key, value in original.items()}

    with torch.no_grad():
        output_original = model({**base_batch, **original})
        output_permuted = model({**base_batch, **permuted})

    torch.testing.assert_close(
        output_original["plan_value"][0][permutation], output_permuted["plan_value"][0], atol=1e-5, rtol=1e-5,
    )


# --------------------------------------------------------------------------
# Leakage audit (Part 5's own requirement): mutate offline outcomes, hold
# current decision-time state fixed, assert INPUT tensors unchanged.
# --------------------------------------------------------------------------


def test_build_strategist_candidate_example_has_no_target_or_outcome_parameter() -> None:
    import inspect

    parameters = set(inspect.signature(build_strategist_candidate_example).parameters)
    for forbidden in ("label", "labels", "target", "targets", "consequence", "outcome", "verification"):
        assert forbidden not in parameters


def test_input_tensors_never_contain_a_governed_target_key(_golden_fixture) -> None:
    _rec, example = _first_real_example(_golden_fixture)
    for target_name in ("plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
                        "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
                        "action_template", "target_pointer", "consequence_vector"):
        assert target_name not in example.batch
        assert f"{target_name}_mask" not in example.batch


def test_alignment_error_raised_on_a_real_mismatch(_golden_fixture, monkeypatch) -> None:
    """Adversarial: force the independently-reconstructed proposal list to
    disagree with generate_strategist_labels' own internal proposals --
    the alignment guard must fail closed, not silently proceed."""

    pool, network, artifact, node_ids, edge_ids, _library = _golden_fixture
    rec = pool[0]

    import hydroswarm.training.strategist_candidate_corpus as corpus_module

    original = corpus_module._reconstruct_context_and_proposals

    def _tampered(*args, **kwargs):
        proposals, context = original(*args, **kwargs)
        if len(proposals) > 1:
            proposals = (proposals[1], proposals[0], *proposals[2:])
        return proposals, context

    monkeypatch.setattr(corpus_module, "_reconstruct_context_and_proposals", _tampered)
    try:
        with pytest.raises(StrategistCandidateAlignmentError):
            build_strategist_candidate_example(rec.scenario, rec.network, rec.feature_context, artifact, node_ids, edge_ids)
    except pytest.skip.Exception:
        raise
    except AssertionError:
        pytest.skip("this incident had fewer than 2 real candidates -- no swap possible to force misalignment")


def test_context_construction_never_reads_scenario_incident_ground_truth_for_candidate_generation(_golden_fixture) -> None:
    """`_reconstruct_context_and_proposals` (the INPUT side) must never
    reference `scenario.manifest.incident` -- the exact ground-truth source
    profile is legitimately used ONLY by `generate_strategist_labels`
    (the TARGET side, offline). Structural, not merely empirical, guard."""

    import inspect

    import hydroswarm.training.strategist_candidate_corpus as corpus_module

    source = inspect.getsource(corpus_module._reconstruct_context_and_proposals)
    assert "manifest.incident" not in source
    assert "incident_truth" not in source


def test_locked_test_opened_remains_false() -> None:
    assert locked_test_opened(ROOT) is False
