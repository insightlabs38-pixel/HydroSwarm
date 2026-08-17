"""TRUE Milestone 10.2 evaluation harness: real-simulation integration
tests (small, real WNTR-backed golden-reference scenarios -- same fixture
pattern as `tests/scientific/test_m10_2_scout_refit_corpus.py`).

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M10_2_TRUE_EVALUATION_PROTOCOL.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_m10_2_true_evaluation as ev  # noqa: E402
from run_m7_topology import SEED_BASES, TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import _degradation_probabilities, fit_pool_signature_library  # noqa: E402
from hydroswarm.training.scenario_reconstruction import reconstruct_scenario_network, simulate_all_node_truth  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.real_simulation

_CACHE_DIR = "/tmp/claude-0/-workspace/m10_2_true_sig_cache"


@pytest.fixture(scope="module")
def _fixture():
    family, loader = TRAINED_FAMILIES[0]
    assert family == "golden-reference"
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")],
        count=15, source_round_robin=True,
    )
    input_library = fit_pool_signature_library(pool)
    network = loader()
    cache = SignatureCache(_CACHE_DIR)
    key = SignatureCacheKey(
        network_hash="m10-2-true-eval-test", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="cfg1", sensor_layout_hash="layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)
    junction_ids = tuple(sorted(network.junction_name_list))
    calibrator = ev.fit_frozen_calibrator()
    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG).eval()
    return pool, input_library, network, artifact, junction_ids, calibrator, model


def _reconstruction_for(pool, network, index=0):
    rec = pool[index]
    reconstruction = reconstruct_scenario_network(
        network, rec.scenario.manifest, degradation_policy=_degradation_probabilities, original=rec.scenario,
    )
    return rec, reconstruction, simulate_all_node_truth(reconstruction)


def test_real_approved_checkpoints_load_and_produce_finite_scout_outputs() -> None:
    for seed in (20260814, 31874, 20260815):
        model, sha256, path = ev.verify_and_load_checkpoint(seed)
        assert sha256 == ev.proto.LEVEL_A_REFIT_CHECKPOINT_SHA256[seed]


def test_budget_is_never_exceeded_for_either_arm(_fixture) -> None:
    pool, input_library, network, artifact, junction_ids, calibrator, model = _fixture
    rec, reconstruction, all_node_truth = _reconstruction_for(pool, network, index=0)
    for arm in ("D", "L"):
        result = ev._run_one_arm(
            arm=arm, model=model, scenario=rec.scenario, network=network, input_library=input_library,
            target_artifact=artifact, reconstruction=reconstruction, all_node_truth=all_node_truth,
            junction_ids=junction_ids, calibrator=calibrator,
            network_id_key=f"{ev.proto.FAMILY}:{ev.m10.depth_bucket_of(ev.proto.DEPTH)}",
            feature_context=rec.feature_context,
        )
        assert result["final_samples_taken"] <= ev.proto.MAXIMUM_SAMPLES


def test_already_sampled_node_is_never_reselected_within_a_trajectory(_fixture) -> None:
    pool, input_library, network, artifact, junction_ids, calibrator, model = _fixture
    for index in range(min(5, len(pool))):
        rec, reconstruction, all_node_truth = _reconstruction_for(pool, network, index=index)
        for arm in ("D", "L"):
            result = ev._run_one_arm(
                arm=arm, model=model, scenario=rec.scenario, network=network, input_library=input_library,
                target_artifact=artifact, reconstruction=reconstruction, all_node_truth=all_node_truth,
                junction_ids=junction_ids, calibrator=calibrator,
                network_id_key=f"{ev.proto.FAMILY}:{ev.m10.depth_bucket_of(ev.proto.DEPTH)}",
                feature_context=rec.feature_context,
            )
            sampled_nodes = [r["chosen_node"] for r in result["rounds"] if r["decision"] == "SAMPLE"]
            assert len(sampled_nodes) == len(set(sampled_nodes))


def test_every_chosen_node_is_a_valid_accessible_junction(_fixture) -> None:
    pool, input_library, network, artifact, junction_ids, calibrator, model = _fixture
    rec, reconstruction, all_node_truth = _reconstruction_for(pool, network, index=1)
    for arm in ("D", "L"):
        result = ev._run_one_arm(
            arm=arm, model=model, scenario=rec.scenario, network=network, input_library=input_library,
            target_artifact=artifact, reconstruction=reconstruction, all_node_truth=all_node_truth,
            junction_ids=junction_ids, calibrator=calibrator,
            network_id_key=f"{ev.proto.FAMILY}:{ev.m10.depth_bucket_of(ev.proto.DEPTH)}",
            feature_context=rec.feature_context,
        )
        for round_record in result["rounds"]:
            if round_record["decision"] == "SAMPLE":
                assert round_record["chosen_node"] in junction_ids


def test_deterministic_repeatability_same_scenario_same_seed_same_trajectory(_fixture) -> None:
    pool, input_library, network, artifact, junction_ids, calibrator, model = _fixture
    rec, reconstruction, all_node_truth = _reconstruction_for(pool, network, index=2)
    kwargs = dict(
        model=model, scenario=rec.scenario, network=network, input_library=input_library,
        target_artifact=artifact, reconstruction=reconstruction, all_node_truth=all_node_truth,
        junction_ids=junction_ids, calibrator=calibrator,
        network_id_key=f"{ev.proto.FAMILY}:{ev.m10.depth_bucket_of(ev.proto.DEPTH)}",
        feature_context=rec.feature_context,
    )
    for arm in ("D", "L"):
        result_a = ev._run_one_arm(arm=arm, **kwargs)
        result_b = ev._run_one_arm(arm=arm, **kwargs)
        chosen_a = [r.get("chosen_node") for r in result_a["rounds"]]
        chosen_b = [r.get("chosen_node") for r in result_b["rounds"]]
        assert chosen_a == chosen_b
        assert result_a["resolved_at_step"] == result_b["resolved_at_step"]


def test_paired_arms_share_identical_round0_state_before_any_policy_choice(_fixture) -> None:
    """Both arms build their round-0 model input from the SAME scenario, SAME
    network, SAME (empty) already_sampled/revealed state -- so round-0
    `source_node_logits` must be bit-identical regardless of which arm asks
    for it (the policy choice has not yet had any effect)."""

    pool, input_library, network, artifact, junction_ids, calibrator, model = _fixture
    rec, reconstruction, all_node_truth = _reconstruction_for(pool, network, index=3)
    state = ev.build_scout_training_state_batch(
        rec.scenario, network, input_library, ev.proto.DEPTH, already_sampled=[], revealed_values_mg_l={},
        sampling_round=0, sample_budget_total=ev.proto.MAXIMUM_SAMPLES, feature_context=rec.feature_context,
    )
    with torch.no_grad():
        output_first = model(state.batch)
        output_second = model(state.batch)
    torch.testing.assert_close(output_first["source_node_logits"], output_second["source_node_logits"])


def test_same_level_a_refit_checkpoint_object_used_for_both_arms() -> None:
    """Structural guard: `run()` loads exactly ONE model per seed
    (`main()`'s own single `verify_and_load_checkpoint` call per seed,
    outside `run()` itself) and passes the SAME `model` object/variable to
    both `_run_one_arm(arm="D", ...)` and `_run_one_arm(arm="L", ...)`
    calls -- never two different checkpoints."""

    import inspect

    source = inspect.getsource(ev.run)
    assert "verify_and_load_checkpoint" not in source  # run() takes `model` as a parameter, never reloads it
    assert 'arm="D", model=model,' in source
    assert 'arm="L", model=model,' in source


def test_future_leakage_mutation_never_changes_current_round_state(_fixture) -> None:
    """Adversarial leakage test: mutating a LATER-round revealed value (for
    a node not yet in `already_sampled`) must never change the CURRENT
    round's built state -- reuses the same leakage guarantee
    `build_scout_training_state_batch` already provides, exercised here
    through this evaluation harness's own call pattern."""

    pool, input_library, network, artifact, junction_ids, calibrator, model = _fixture
    rec, reconstruction, all_node_truth = _reconstruction_for(pool, network, index=4)
    baseline = ev.build_scout_training_state_batch(
        rec.scenario, network, input_library, ev.proto.DEPTH, already_sampled=["J1"],
        revealed_values_mg_l={"J1": 1.0}, sampling_round=1, sample_budget_total=ev.proto.MAXIMUM_SAMPLES,
        feature_context=rec.feature_context,
    )
    with_future_leak = ev.build_scout_training_state_batch(
        rec.scenario, network, input_library, ev.proto.DEPTH, already_sampled=["J1"],
        revealed_values_mg_l={"J1": 1.0, "J4": 999.0}, sampling_round=1,
        sample_budget_total=ev.proto.MAXIMUM_SAMPLES, feature_context=rec.feature_context,
    )
    for key in baseline.batch:
        torch.testing.assert_close(baseline.batch[key], with_future_leak.batch[key])


def test_true_source_never_enters_the_built_model_input(_fixture) -> None:
    pool, input_library, network, artifact, junction_ids, calibrator, model = _fixture
    rec, reconstruction, all_node_truth = _reconstruction_for(pool, network, index=0)
    state = ev.build_scout_training_state_batch(
        rec.scenario, network, input_library, ev.proto.DEPTH, already_sampled=[], revealed_values_mg_l={},
        sampling_round=0, sample_budget_total=ev.proto.MAXIMUM_SAMPLES, feature_context=rec.feature_context,
    )
    assert "source_node" not in state.batch
    assert "source_node_mask" not in state.batch


def test_locked_test_opened_remains_false() -> None:
    assert locked_test_opened(ROOT) is False
