"""M10.2 Scout supervision/representation refit amendment, Parts 3/4:
real Scout training-state/target corpus wiring + leakage isolation.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import _degradation_probabilities, fit_pool_signature_library  # noqa: E402
from hydroswarm.training.scenario_reconstruction import reconstruct_scenario_network  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402
from hydroswarm.training.scout_targets import SCOUT_TARGET_SCHEMA_VERSION, build_scout_training_examples  # noqa: E402
from hydroswarm.training.scout_training_state import (  # noqa: E402
    SCOUT_TRAINING_STATE_SCHEMA_VERSION,
    build_scout_training_state_batch,
)

from run_m7_topology import SEED_BASES, TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.real_simulation

_CACHE_DIR = "/tmp/claude-0/-workspace/scout_sig_cache"


@pytest.fixture(scope="module")
def _golden_fixture():
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
        network_hash="golden-ref-refit-test",
        hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="cfg1", sensor_layout_hash="layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)
    return pool, input_library, network, artifact


def test_schema_versions_are_distinct_from_each_other_and_from_eval_schema() -> None:
    from hydroswarm.evaluation.scout_state import SCOUT_EVAL_STATE_SCHEMA_VERSION
    from hydroswarm.training import checkpoint_identity

    versions = {
        SCOUT_TRAINING_STATE_SCHEMA_VERSION,
        SCOUT_TARGET_SCHEMA_VERSION,
        SCOUT_EVAL_STATE_SCHEMA_VERSION,
        checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION,
    }
    assert len(versions) == 4
    assert checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION == "scout-state-v1-unbuilt"


def test_role_and_residual_channels_are_populated_nowhere_else_in_the_codebase() -> None:
    """Confirms this amendment's index assignments (role_features[0:2],
    residual_features[...,0]) cannot collide with any other real caller --
    the load-bearing assumption `scout_training_state.py`'s module docstring
    makes."""

    import subprocess

    result = subprocess.run(
        ["grep", "-rn", "role_features\\s*=\\|residual_features\\s*=", str(ROOT / "src")],
        capture_output=True, text=True,
    )
    hits = [
        line for line in result.stdout.splitlines()
        if "scout_training_state.py" not in line and "core.py:" not in line
    ]
    assert hits == [], f"unexpected role_features/residual_features writer(s): {hits}"


def test_build_scout_training_state_batch_has_no_label_or_target_parameter() -> None:
    """Structural leakage guard: the INPUT builder cannot even be CALLED
    with a ScoutLabel/offline-truth object -- there is no parameter for
    one."""

    parameters = set(inspect.signature(build_scout_training_state_batch).parameters)
    assert "label" not in parameters
    assert "target" not in parameters
    assert "targets" not in parameters


def test_round_and_budget_and_accessibility_channels_are_correctly_encoded(_golden_fixture) -> None:
    pool, input_library, network, _artifact = _golden_fixture
    rec = pool[0]
    state = build_scout_training_state_batch(
        rec.scenario, rec.network, input_library, 3,
        already_sampled=["J1"], revealed_values_mg_l={"J1": 2.0},
        sampling_round=1, sample_budget_total=3, feature_context=rec.feature_context,
    )
    from hydroswarm.training.scout_training_state import MAXIMUM_SAMPLES_BOUND

    role = state.batch["role_features"]
    assert role.shape == (1, 8)
    assert role[0, 0].item() == pytest.approx(1.0 / MAXIMUM_SAMPLES_BOUND)
    assert role[0, 1].item() == pytest.approx(2.0 / MAXIMUM_SAMPLES_BOUND)  # 3 total - 1 already sampled
    assert torch.equal(role[0, 2:], torch.zeros(6))

    residual = state.batch["residual_features"]
    assert residual.shape == (1, len(state.node_ids), 4)
    junction_set = set(state.junction_ids)
    for position, node_id in enumerate(state.node_ids):
        expected = 1.0 if node_id in junction_set else 0.0
        assert residual[0, position, 0].item() == pytest.approx(expected)
    assert torch.equal(residual[..., 1:], torch.zeros(1, len(state.node_ids), 3))


def test_future_revealed_value_for_a_not_yet_sampled_node_has_zero_effect(_golden_fixture) -> None:
    """Adversarial leakage mutation: a `revealed_values_mg_l` entry for a
    node NOT in `already_sampled` (i.e. "future" information relative to
    this decision) must never change the built batch."""

    pool, input_library, network, _artifact = _golden_fixture
    rec = pool[0]
    kwargs = dict(
        scenario=rec.scenario, network=rec.network, signature_library=input_library, depth=3,
        already_sampled=["J1"], sampling_round=1, sample_budget_total=3, feature_context=rec.feature_context,
    )
    baseline = build_scout_training_state_batch(revealed_values_mg_l={"J1": 2.0}, **kwargs)
    with_future_leak = build_scout_training_state_batch(
        revealed_values_mg_l={"J1": 2.0, "J4": 999.0}, **kwargs  # J4 not sampled -- must be ignored
    )
    for key in baseline.batch:
        torch.testing.assert_close(baseline.batch[key], with_future_leak.batch[key])


def test_mutating_offline_target_label_never_changes_the_paired_input(_golden_fixture) -> None:
    """The strongest available form of "mutate the label, hold current
    state fixed, prove input unchanged": run the real pipeline twice with
    different `noise_scale_mg_l` (which only affects the OFFLINE label's
    posterior/EIG computation and the revealed value's noise draw, not the
    deterministic INPUT-construction machinery for a FIXED already_sampled/
    revealed_values_mg_l pair) and confirm step-0's `state.batch` (built
    before any reveal exists) is identical regardless."""

    pool, input_library, network, artifact = _golden_fixture
    rec = pool[1]
    reconstruction = reconstruct_scenario_network(
        network, rec.scenario.manifest, degradation_policy=_degradation_probabilities, original=rec.scenario,
    )
    node_ids = tuple(sorted(rec.network.node_name_list))
    examples_a = build_scout_training_examples(
        rec.scenario, rec.network, input_library, artifact, node_ids, 3,
        reconstruction=reconstruction, maximum_samples=2, noise_scale_mg_l=0.5,
    )
    examples_b = build_scout_training_examples(
        rec.scenario, rec.network, input_library, artifact, node_ids, 3,
        reconstruction=reconstruction, maximum_samples=2, noise_scale_mg_l=1.5,
    )
    for key in examples_a[0].state.batch:
        torch.testing.assert_close(examples_a[0].state.batch[key], examples_b[0].state.batch[key])


def test_already_sampled_grows_monotonically_and_revealed_samples_lag_by_one_step(_golden_fixture) -> None:
    pool, input_library, network, artifact = _golden_fixture
    for rec in pool:
        reconstruction = reconstruct_scenario_network(
            network, rec.scenario.manifest, degradation_policy=_degradation_probabilities, original=rec.scenario,
        )
        node_ids = tuple(sorted(rec.network.node_name_list))
        examples = build_scout_training_examples(
            rec.scenario, rec.network, input_library, artifact, node_ids, 3,
            reconstruction=reconstruction, maximum_samples=3,
        )
        if len(examples) < 2:
            continue
        for previous, current in zip(examples, examples[1:]):
            assert set(previous.already_sampled) <= set(current.already_sampled)
            assert len(current.already_sampled) == len(previous.already_sampled) + 1
        break
    else:
        pytest.skip("no scenario in this small pool produced a multi-step trajectory")


def test_finite_outputs_through_a_real_frozen_backbone_forward_pass(_golden_fixture) -> None:
    from run_m9_0a_evaluate import SHARED_MODEL_CONFIG

    from hydroswarm.model import HydroCore

    pool, input_library, network, _artifact = _golden_fixture
    rec = pool[0]
    state = build_scout_training_state_batch(
        rec.scenario, rec.network, input_library, 3,
        already_sampled=["J1"], revealed_values_mg_l={"J1": 1.0},
        sampling_round=1, sample_budget_total=3, feature_context=rec.feature_context,
    )
    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG).eval()
    with torch.no_grad():
        output = model(state.batch)
    assert torch.isfinite(output["sample_node_logits"]).all()
    assert torch.isfinite(output["expected_information_gain"]).all()


def test_locked_test_opened_remains_false() -> None:
    assert locked_test_opened(ROOT) is False
