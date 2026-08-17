"""M10.2 preflight regression tests: empirically re-verify (not merely by
source reading) that the M9.6 training corpus never included Scout targets,
and exercise `hydroswarm.evaluation.scout_state` end-to-end against a real
`HydraulicFeatureBuilder` -> `HydroCore.forward()` batch.

Frozen correction document:
docs/evaluation/HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.evaluation.scout_readiness import M9_6_OBSERVED_CORPUS_TARGET_KEYS  # noqa: E402
from hydroswarm.evaluation.scout_state import (  # noqa: E402
    assert_finite_scout_outputs,
    build_scout_evaluation_state,
    decode_learned_scout_recommendation,
)
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.causal_prefix import fit_pool_signature_library, scenario_to_prefix_example  # noqa: E402

from run_m7_topology import SEED_BASES, TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.real_simulation

#: golden-reference is the smallest trained family; 15 scenarios gives
#: fit_pool_signature_library round-robin source coverage (matches the
#: existing SMALL_POOL_COUNT convention in test_interleaved_topology_m9_0.py).
_SMALL_POOL_COUNT = 15


@pytest.fixture(scope="module")
def _golden_pool_and_library():
    family, loader = TRAINED_FAMILIES[0]
    assert family == "golden-reference"
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")],
        count=_SMALL_POOL_COUNT, source_round_robin=True,
    )
    library = fit_pool_signature_library(pool)
    return pool, library


def test_m9_6_training_corpus_never_included_scout_targets(_golden_pool_and_library) -> None:
    """Direct, empirical re-verification of
    hydroswarm.evaluation.scout_readiness's central finding: the ONLY
    targets-generating function M9.6 training ever called
    (scenario_to_prefix_example, via CausalPrefixDatasetView) never produces
    sample_node/information_gain/candidate_reduction/should_continue_sampling
    -- checked against a real generated scenario, not merely by reading
    causal_prefix.py's source."""

    pool, library = _golden_pool_and_library
    record = pool[0]
    example = scenario_to_prefix_example(record.scenario, record.network, library, 3, feature_context=record.feature_context)
    observed_keys = frozenset(example.targets.keys())
    assert observed_keys == M9_6_OBSERVED_CORPUS_TARGET_KEYS
    scout_keys = {"sample_node", "information_gain", "candidate_reduction", "should_continue_sampling"}
    assert observed_keys.isdisjoint(scout_keys)


def test_scout_evaluation_state_end_to_end_against_a_real_hydraulic_feature_batch(_golden_pool_and_library) -> None:
    """Builds a real HydroBatch via the same causal-prefix pipeline M9.6
    training used, wraps it in a ScoutEvaluationState, runs it through a
    real (freshly-initialized, NOT the frozen M9.6 weights -- this is an
    interface/shape/finiteness check, not a scientific comparison) HydroCore
    forward pass with scout_control_heads=True, and decodes a
    recommendation -- proving the schema/adapter genuinely composes with the
    real feature pipeline end to end, independent of the separate
    untrained-heads finding."""

    pool, library = _golden_pool_and_library
    record = pool[0]
    example = scenario_to_prefix_example(record.scenario, record.network, library, 3, feature_context=record.feature_context)
    node_ids = example.topology.node_ids
    batch = {key: value.unsqueeze(0) for key, value in example.inputs.items()}

    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG).eval()
    state = build_scout_evaluation_state(
        node_ids=[node_ids],
        batch=batch,
        already_sampled=[[node_ids[0]]],
        sampling_round=[1],
        sample_budget_total=[3],
    )
    with torch.no_grad():
        output = model(state.batch)
    assert_finite_scout_outputs(output)
    recommendation = decode_learned_scout_recommendation(output, state)
    assert recommendation.node_id != node_ids[0]
    if recommendation.node_id is not None:
        assert recommendation.node_id in node_ids
    assert recommendation.promotable is False


def test_locked_test_opened_remains_false_after_real_simulation_use() -> None:
    assert locked_test_opened(ROOT) is False
