"""Milestone 7 (experiments.txt): topology diversity and OOD-aware fusion.

Part A -- training diversity. Compares the frozen Milestone-1 winner ("CURRENT":
trained on golden-reference only, exactly the predictor M3-M6 already froze via
`run_m3_calibration._freeze_predictor`) against a NEW "EXPANDED" arm trained with
the IDENTICAL architecture/config/seeds but exposed to two additional, structurally
distinct topology families during training.

Architectural constraint discovered and worked around (documented, not silently
assumed): `hydroswarm.training.causal_prefix.scenario_to_prefix_example` requires
every example in one `CausalPrefixDatasetView` to share ONE `SignatureLibrary`
(`if junction_ids != signature_library.node_ids: raise ValueError`), and
`collate_scenarios` requires every example in one micro-batch to share identical
node-count-dependent tensor shapes. Neither is a v5-vs-v4 architecture change (the
model itself is topology-agnostic -- see `scripts/evaluate_topology_transfer.py`,
which already runs the SHIPPED v4 checkpoint zero-shot on branched-loop/loop-grid/
coastal-branch); it is a training-time DATA-PIPELINE constraint the existing
`CausalPrefixDatasetView`/`Trainer` machinery was never exercised against because
Milestones 1-6 only ever trained on one family. Mixing families within a single
micro-batch is therefore not attempted (that is real Milestone-8/9 infrastructure
scope -- "Use PyG only if there is a measured problem", not assumed here). Instead
EXPANDED training is a SEQUENTIAL CURRICULUM across three `Trainer.fit(resume_from=...)`
phases (golden-reference epochs 0-6, branched-loop epochs 7-13, loop-grid epochs
14-19), each phase internally family-pure (so every micro-batch is shape-consistent)
and each resuming the SAME model/optimizer/scheduler state from the previous phase's
checkpoint via `Trainer`'s existing, unmodified resume mechanism -- verified directly
this session (see the smoke test this script's history is built on: a 2-epoch
tree-branch phase resumed cleanly into a 4-epoch dense-loop phase with a DIFFERENT
junction count, no shape errors). `Trainer`/`CausalPrefixDatasetView`/
`scenario_to_prefix_example` are used completely UNMODIFIED; this script only adds
NEW, additive per-family pool/library construction alongside them. No frozen file
is edited.

Total training-scenario BUDGET is held constant between arms (600 train / 100
validation / 150 calibration, matching Milestone 1's own SPLIT_SEED_RANGES) so the
comparison isolates topology DIVERSITY, not data VOLUME: CURRENT gets 600
golden-reference scenarios; EXPANDED gets 200 each of golden-reference/branched-loop/
loop-grid. `early_stopping_patience` is overridden to 0 (disabled) for EXPANDED only
(a documented compute-cost/mechanics override, not a scientific one, matching
Milestone 1's own `GRADNORM_LOG_EVERY_N_BATCHES` override precedent): the
`best_validation_loss` carried across a `resume_from` phase boundary is measured
against a DIFFERENT family's validation set than produced it, so "stale epochs"
counted across a family switch is not a meaningful convergence signal and would
otherwise truncate phases 2/3 almost immediately.

Two seeds (20260814, 31874), matching every M1-M6 screening arm's own two-seed
convention ("Two seeds for rapid screening of any arm/ablation" --
HYDROCORE_V5_EXPERIMENT_PROTOCOL.md Section 2). CURRENT's two seeds are the
EXISTING frozen Milestone-1 arm-A checkpoints (zero retraining cost, byte-identical
to what M3-M6 already used). Per-seed results are always preserved individually,
never collapsed to a mean before being recorded.

Development-unseen evaluation pool (never trained by EITHER arm): the existing
`coastal-branch` (development-only control, per LIVE_ROBUSTNESS_PROTOCOL.md) plus
two NEW small topologies built locally below -- `tree-branch` (pure tree, cycle_rank
0) and `dense-loop` (heavily looped, cycle_rank 3) -- chosen after directly measuring
structural descriptors (junction count, cycle_rank, degree distribution) against
every existing committed topology to guarantee genuine, non-overlapping structural
diversity (branched/looped/mixed, sizes 4-8 junctions, cycle_rank 0-3), not roughness
perturbations of a known family. Both new topologies were verified to solve cleanly
in a real WNTR/EPANET simulation before use (see this session's smoke test). The
locked topology set is never touched -- no committed or generated file in this
script uses a "locked_final_test"/"locked_topology_test" token, and no such file is
read (this repo currently has no locked-topology fixture materialized at all: see
`hydroswarm.evaluation.live_robustness.locked_test_opened`, whose guard this script
still asserts before and after, unchanged).

Also evaluated on branched-loop/loop-grid (known to EXPANDED, unseen to CURRENT):
this is the actual crux comparison for Part A's own question -- does exposure
during training help specifically on the families that exposure covers, without
regressing the shared golden-reference family.

Part B -- OOD-aware fusion. Reuses the SAME captured per-row (neural_logits,
classical_belief, TrustFeatures inputs) from Part A's evaluation pass -- no new
model inference -- to compare four fusion policies purely offline, mirroring
`scripts/capability_diagnostic/component_fusion_analysis.py`'s own "mined entirely
from an already-computed dataset, production fusion.py never touched" convention:
  - classical-only
  - neural-only
  - CURRENT_DYNAMIC (the real, unmodified `hydroswarm.inference.fusion.
    fuse_source_probabilities` / `dynamic_classical_trust`)
  - NOVELTY_AWARE (a new, LOCAL-ONLY function defined in this script that adds a
    predeclared `NOVELTY_TRUST_BOOST=0.25` push toward classical trust when the
    row's topology family is unseen by the model being evaluated -- never applied
    to `src/hydroswarm/inference/fusion.py` itself; diagnostic-only, exactly like
    every other capability_diagnostic fusion analysis in this repo).
Topology "novelty" here is a FAMILY-level binary signal (was this family in this
specific model's OWN training data), not `OODDetector.topology_novelty`'s
exact-hash check -- documented explicitly because with this repo's default nonzero
`roughness_variation_fraction`/`tank_level_variation_fraction`, per-scenario
network_sha256 varies even WITHIN one trained family, which would make an
exact-hash check spuriously report near-100% novelty for scenarios the model
genuinely trained on. The concept (binary known/unknown gate on structural
familiarity) is the same one `OODDetector.topology_novelty`'s own docstring
describes ("maximum novelty for a graph outside the explicitly validated ... set");
this script only swaps the granularity appropriate to comparing TRAINED-FAMILY
membership rather than exact-hash registry membership.

TrustFeatures inputs this lightweight (M1-M6-style, no release-bundle) harness can
supply for real: healthy_sensor_fraction/missing_rate (directly from SensorSeries),
neural_entropy/classical_entropy (directly from the computed belief vectors).
hydraulic_uncertainty/normalized_residual use documented proxies, since the full
production `HydraulicStateEstimator` is not wired into this harness (never was in
M1-M6 either): hydraulic_uncertainty = 1 - evidence_sufficiency (the model's own
real head output); normalized_residual = the classical library's own log1p residual
at its argmax candidate, min-max normalized across this run's own evaluation batch.
Diagnostic-only, consistent with component_fusion_analysis.py's explicit
"production fusion policy untouched" scope -- this experiment's job is to report
whether the novelty-aware idea shows a real, honestly-measured signal, not to ship
a new fusion formula.

Frozen elsewhere, unchanged here: architecture (SHARED_MODEL_CONFIG), alpha=0.1,
K=3 (MAXIMUM_EVALUATION_HYPOTHESES), the B_DEPTH_AWARE calibration SCHEME (refit
per-model on that model's own calibration pool, exactly like every M4-M6 script
already refits it -- never loaded from a serialized artifact, matching established
convention).

Writes:
  reports/evaluation/hydrocore-v5/m7-training-diversity.json   (Part A)
  reports/evaluation/hydrocore-v5/m7-ood-fusion.json           (Part B)
  reports/evaluation/hydrocore-v5/m7-summary.md
  reports/evaluation/hydrocore-v5/m7-runs/EXPANDED-seed{seed}.json (per-seed training provenance)
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import (  # noqa: E402
    CalibrationExample,
    SplitConformalCalibrator,
    classify_runtime_condition,
)
from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage as ScenarioCurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.inference.fusion import (  # noqa: E402
    FusionDiagnostics,
    TrustFeatures,
    dynamic_classical_trust,
    fuse_source_probabilities,
    jensen_shannon_divergence,
)
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.preprocessing import HydraulicFeatureBuilder, SensorSeries  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.simulation.wrapper import MAXIMUM_EVALUATION_HYPOTHESES  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    CausalPrefixDatasetView,
    ScenarioRecord,
    build_scenario_pool,
    fit_pool_signature_library,
    full_history_policy,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.corpus import (  # noqa: E402
    aligned_observations_from_series,
    build_feature_context,
    build_sensor_series,
    model_input_classical_prior,
)
from hydroswarm.training.trainer import Trainer, set_deterministic_seed  # noqa: E402

import wntr  # noqa: E402

from run_m1_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m3_calibration import DEPTH_BUCKET_OF  # noqa: E402

OUT_TRAINING_DIVERSITY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m7-training-diversity.json"
OUT_OOD_FUSION = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m7-ood-fusion.json"
OUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m7-summary.md"
RUNS_ROOT = ROOT / "experiments" / "runs" / "hydrocore-v5-causal"
M7_RUNS_SUMMARY_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m7-runs"

ALPHA = 0.1
K_MAX_CANDIDATES = MAXIMUM_EVALUATION_HYPOTHESES
assert K_MAX_CANDIDATES == 3
EPS = 1e-9

SEEDS: tuple[int, ...] = (20260814, 31874)
EVAL_DEPTHS: tuple[int, ...] = (3, 25)  # EARLY, MATURE (DEPTH_BUCKET_OF).

#: Total-budget-matched per-family scenario counts for the EXPANDED arm
#: (3 families x these counts == CURRENT arm's own 600/100/150 totals).
TRAIN_PER_FAMILY = 200
VALIDATION_PER_FAMILY = 33  # 33+33+34 = 100
CALIBRATION_PER_FAMILY = 50
#: Cumulative epoch boundaries for the 3-phase sequential curriculum
#: (config.epochs=20, unchanged from configs/training-v5-causal.yaml).
PHASE_EPOCH_BOUNDARIES: tuple[int, ...] = (7, 14, 20)

#: Small, deterministic, coverage-guaranteed pool used ONLY to fit a
#: classical SignatureLibrary for a family a given model never trained on
#: (real WNTR-simulated per-source physics, never a neural-weight update --
#: see module docstring). source_node is explicitly round-robinned (unlike
#: the train/validation/calibration pools below, which follow
#: build_scenario_pool's own established random-source convention) because
#: this pool is deliberately small and every junction's coverage is the
#: entire point of fitting a signature library at all.
CLASSICAL_FIT_ONLY_COUNT = 40

#: Evaluation-metrics pool: up to this many distinct source junctions, each
#: repeated this many seeds, matching topology_generalization_decomposition.py's
#: "8 fresh, non-locked, seed-distinct scenarios" precedent (scaled up
#: slightly since this milestone also splits by depth).
EVAL_MAX_SOURCES = 4
EVAL_SEED_REPEATS = 4

#: Part B: predeclared novelty-aware fusion trust push (documented, not
#: fit/tuned against any result -- chosen once, before this run, as a
#: moderate push in fuse_source_probabilities' own trust units).
NOVELTY_TRUST_BOOST = 0.25

DIURNAL_PATTERN = (
    0.62, 0.56, 0.52, 0.50, 0.55, 0.72,
    1.00, 1.22, 1.18, 1.05, 0.98, 1.02,
    1.08, 1.04, 0.98, 0.96, 1.08, 1.28,
    1.38, 1.24, 1.08, 0.94, 0.80, 0.70,
)


# ---------------------------------------------------------------------------
# New synthetic topologies (development-unseen only -- never trained by
# either arm). Both verified this session to solve cleanly under a real
# wntr.sim.EpanetSimulator run (min/max pressure finite and physically
# sane) before being used anywhere else in this script.
# ---------------------------------------------------------------------------


def build_tree_branch_network() -> Any:
    """Pure branched tree (cycle_rank=0, 5 junctions, single branch point at
    J2) -- distinct in kind (not degree) from every other committed
    topology, none of which is loop-free."""

    model = wntr.network.WaterNetworkModel()
    model.name = "tree-branch"
    model.add_pattern("diurnal", list(DIURNAL_PATTERN))
    model.add_reservoir("R1", base_head=128.0, coordinates=(0.0, 0.0))
    for name, x, y, elevation, demand in (
        ("J1", 900.0, 0.0, 98.0, 0.0075),
        ("J2", 1800.0, 0.0, 102.0, 0.0085),
        ("J3", 2600.0, 700.0, 105.0, 0.0060),
        ("J4", 2600.0, -700.0, 104.0, 0.0090),
        ("J5", 3400.0, -700.0, 108.0, 0.0055),
    ):
        model.add_junction(name, base_demand=demand, demand_pattern="diurnal", elevation=elevation, coordinates=(x, y))
    for name, start, end, length, diameter, roughness in (
        ("P_R1_J1", "R1", "J1", 900.0, 0.30, 118.0),
        ("P_J1_J2", "J1", "J2", 900.0, 0.28, 115.0),
        ("P_J2_J3", "J2", "J3", 950.0, 0.22, 112.0),
        ("P_J2_J4", "J2", "J4", 950.0, 0.22, 112.0),
        ("P_J4_J5", "J4", "J5", 850.0, 0.18, 110.0),
    ):
        model.add_pipe(name, start, end, length=length, diameter=diameter, roughness=roughness, minor_loss=0.0, initial_status="OPEN")
    model.options.time.pattern_timestep = 3_600
    model.options.time.hydraulic_timestep = 3_600
    model.options.time.quality_timestep = 300
    model.options.time.duration = 24 * 3_600
    return model


def build_dense_loop_network() -> Any:
    """Heavily looped network (cycle_rank=3, 6 junctions, two hub nodes at
    degree 3 and 5) -- more densely looped than every other committed
    topology (golden-reference and loop-grid both max out at cycle_rank 2-3
    with lower peak degree), giving a genuinely distinct degree
    distribution, not merely a bigger version of an existing family."""

    model = wntr.network.WaterNetworkModel()
    model.name = "dense-loop"
    model.add_pattern("diurnal", list(DIURNAL_PATTERN))
    model.add_reservoir("R1", base_head=130.0, coordinates=(0.0, 0.0))
    for name, x, y, elevation, demand in (
        ("J1", 1000.0, 0.0, 97.0, 0.0070),
        ("J2", 2000.0, 500.0, 100.0, 0.0080),
        ("J3", 2800.0, 0.0, 103.0, 0.0065),
        ("J4", 2000.0, -500.0, 101.0, 0.0075),
        ("J5", 1200.0, 900.0, 99.0, 0.0060),
        ("J6", 1200.0, -900.0, 98.0, 0.0068),
    ):
        model.add_junction(name, base_demand=demand, demand_pattern="diurnal", elevation=elevation, coordinates=(x, y))
    for name, start, end, length, diameter, roughness in (
        ("P_R1_J1", "R1", "J1", 1000.0, 0.32, 118.0),
        ("P_J1_J2", "J1", "J2", 1100.0, 0.25, 114.0),
        ("P_J2_J3", "J2", "J3", 900.0, 0.22, 112.0),
        ("P_J3_J4", "J3", "J4", 900.0, 0.22, 112.0),
        ("P_J4_J1", "J4", "J1", 1100.0, 0.25, 114.0),
        ("P_J1_J5", "J1", "J5", 750.0, 0.20, 110.0),
        ("P_J5_J2", "J5", "J2", 800.0, 0.20, 110.0),
        ("P_J1_J6", "J1", "J6", 750.0, 0.20, 110.0),
        ("P_J6_J4", "J6", "J4", 800.0, 0.20, 110.0),
    ):
        model.add_pipe(name, start, end, length=length, diameter=diameter, roughness=roughness, minor_loss=0.0, initial_status="OPEN")
    model.options.time.pattern_timestep = 3_600
    model.options.time.hydraulic_timestep = 3_600
    model.options.time.quality_timestep = 300
    model.options.time.duration = 24 * 3_600
    return model


#: (family_name, network_loader, .inp path or None for a programmatic build).
#: golden-reference is EXPANDED's phase-1 family and CURRENT's only family.
#: branched-loop/loop-grid are EXPANDED's phase-2/3 families (known to
#: EXPANDED, unseen to CURRENT). coastal-branch/tree-branch/dense-loop are
#: unseen to BOTH arms (development-unseen evaluation pool).
TRAINED_FAMILIES: tuple[tuple[str, Any], ...] = (
    ("golden-reference", build_wntr_network),
    ("branched-loop", lambda: wntr.network.WaterNetworkModel(str(ROOT / "data/topology-transfer/branched-loop.inp"))),
    ("loop-grid", lambda: wntr.network.WaterNetworkModel(str(ROOT / "data/topologies/loop-grid.inp"))),
)
UNSEEN_FAMILIES: tuple[tuple[str, Any], ...] = (
    ("coastal-branch", lambda: wntr.network.WaterNetworkModel(str(ROOT / "data/topologies/coastal-branch.inp"))),
    ("tree-branch", build_tree_branch_network),
    ("dense-loop", build_dense_loop_network),
)
ALL_EVAL_FAMILIES: tuple[tuple[str, Any], ...] = TRAINED_FAMILIES + UNSEEN_FAMILIES


# ---------------------------------------------------------------------------
# Family-scoped scenario pools. A NEW, additive parallel to
# hydroswarm.training.causal_prefix.build_scenario_pool (never modified):
# that function hardcodes network_id=network_family=NETWORK_FAMILY
# ("golden-reference") regardless of which network_loader is passed, so
# reusing it as-is for a different topology would silently mislabel
# provenance. This copies its (Milestone-0-established, deterministic)
# stage-cycling/degradation-probability construction, parametrized by
# family name, count, and a family-specific seed base so seeds never
# collide across families or across the train/validation/calibration/
# classical-fit-only pools below.
# ---------------------------------------------------------------------------


def _degradation_probabilities(stage: ScenarioCurriculumStage) -> dict[str, float]:
    degraded = stage in {ScenarioCurriculumStage.DEGRADED, ScenarioCurriculumStage.SHIFT, ScenarioCurriculumStage.ADVERSARIAL}
    shifted = stage in {ScenarioCurriculumStage.SHIFT, ScenarioCurriculumStage.ADVERSARIAL}
    return {
        "missing_probability": 0.08 if stage != ScenarioCurriculumStage.CLEAN else 0.0,
        "frozen_probability": 0.06 if degraded else 0.01,
        "communication_outage_probability": 0.06 if degraded else 0.01,
        "unit_mismatch_probability": 0.02 if shifted else 0.0,
    }


def _family_scenario_pool(
    split: str, *, network_loader, family: str, seed_base: int, count: int, source_round_robin: bool = False,
) -> list[ScenarioRecord]:
    """Family-labeled counterpart of `build_scenario_pool`, restricted to
    ONE topology family per call so every record shares one junction set
    (required by `scenario_to_prefix_example`'s signature-library check).
    `source_round_robin=True` (used only by the small classical-fit-only
    pools) explicitly cycles `source_node` through every junction in order
    to GUARANTEE `fit_signature_library` sees every candidate source at
    least once even at a small count; the larger train/validation/
    calibration pools below keep `build_scenario_pool`'s own convention of
    leaving `source_node=None` (generator-random, seeded, deterministic --
    statistically guaranteed coverage at these pool sizes: at 200
    scenarios over at most 8 junctions, the probability any junction draws
    zero sources is astronomically small)."""

    generator = WNTRScenarioGenerator()
    stages = tuple(ScenarioCurriculumStage)
    records: list[ScenarioRecord] = []
    network_probe = network_loader()
    junctions = tuple(sorted(network_probe.junction_name_list))
    for index in range(count):
        stage = stages[index % len(stages)]
        seed = seed_base + index * 100
        network = network_loader()
        source_node = junctions[index % len(junctions)] if source_round_robin else None
        config = ScenarioGenerationConfig(
            seed=seed,
            network_id=family,
            network_family=family,
            split=DatasetSplit(split),
            stage=stage,
            event_type=EventType.CONTAMINATION,
            source_node=source_node,
            sensor_count=min(len(junctions), 3 + (index % 3)),
            pipe_outage_probability=0.0,
            **_degradation_probabilities(stage),
        )
        scenario, randomized_network = generator.generate_with_network(network, config)
        context = build_feature_context(randomized_network)
        records.append(ScenarioRecord(scenario=scenario, network=randomized_network, feature_context=context))
    return records


#: Distinct seed bases per family/role so no two pools anywhere in this
#: script ever draw the same (seed, network) pair. Ranges chosen well
#: outside SPLIT_SEED_RANGES (900_000_000-903_999_999) and every other
#: script's reserved ranges this session (M6 used 960-970M, capability
#: diagnostics used 202608xx_xx) to avoid any risk of accidental reuse.
SEED_BASES: dict[tuple[str, str], int] = {}
_next_seed_base = 940_000_000
for _family_name, _ in ALL_EVAL_FAMILIES:
    for _role in ("train", "validation", "calibration", "classical_fit", "eval"):
        SEED_BASES[(_family_name, _role)] = _next_seed_base
        _next_seed_base += 1_000_000


def _classical_fit_library(family: str, network_loader):
    """Fit a fresh SignatureLibrary for `family` from a small, source-
    round-robined, TRAIN-labeled pool -- real WNTR-simulated per-source
    physics, computed fresh for whatever network is being evaluated,
    never a neural-weight update (see module docstring's "never use
    development_holdout labels to train" note: this is the classical arm's
    own physics fitting step, structurally identical to what
    fit_pool_signature_library already does for golden-reference in every
    M1-M6 script, just applied to a family none of those scripts ever
    needed)."""

    records = _family_scenario_pool(
        "train", network_loader=network_loader, family=family,
        seed_base=SEED_BASES[(family, "classical_fit")], count=CLASSICAL_FIT_ONLY_COUNT,
        source_round_robin=True,
    )
    return fit_pool_signature_library(records), records


# ---------------------------------------------------------------------------
# CURRENT arm: the existing frozen Milestone-1 winner. Zero retraining --
# reuses the exact checkpoint M3-M6 already froze via _freeze_predictor,
# for BOTH seeds (both seeds' checkpoints are still present in
# experiments/runs/hydrocore-v5-causal/A-seed*/ from this session's own M1
# run; per-seed provenance is read directly off each seed's own M1 summary
# rather than assumed).
# ---------------------------------------------------------------------------


def _current_arm_checkpoint(seed: int) -> str:
    path = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-runs" / f"A-seed{seed}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    export_path = record["training_summary"]["export_path"]
    if not Path(export_path).exists():
        raise FileNotFoundError(
            f"CURRENT arm's frozen Milestone-1 seed-{seed} checkpoint is missing on disk "
            f"({export_path}) -- experiments/runs/ is gitignored and does not survive across "
            "sessions; re-run scripts/hydrocore_v5/run_m1_arm.py --arm A --seed "
            f"{seed} first."
        )
    return export_path


def _load_model(export_path: str) -> HydroCore:
    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# EXPANDED arm: sequential 3-phase curriculum (see module docstring for why
# this is the correct, minimal-risk way to expose one model to 3 topology
# families given CausalPrefixDatasetView/collate_scenarios' existing
# single-family-per-micro-batch constraint). Trainer/CausalPrefixDatasetView/
# scenario_to_prefix_example are used completely unmodified.
# ---------------------------------------------------------------------------


def _train_expanded_arm(seed: int) -> dict[str, Any]:
    config_path = ROOT / "configs" / "training-v5-causal.yaml"
    base_config = TrainingConfig.from_yaml(str(config_path), require_complete_task_weights=True)
    assert base_config.epochs == PHASE_EPOCH_BOUNDARIES[-1], (
        f"PHASE_EPOCH_BOUNDARIES must sum to configs/training-v5-causal.yaml's epochs "
        f"({base_config.epochs}); got boundaries {PHASE_EPOCH_BOUNDARIES}"
    )

    families = [name for name, _ in TRAINED_FAMILIES]
    counts = {"train": TRAIN_PER_FAMILY, "validation": VALIDATION_PER_FAMILY, "calibration": CALIBRATION_PER_FAMILY}
    # Milestone-1 total-budget parity: 3 x 33/33/34 validation == 100 (matches
    # SPLIT_SEED_RANGES["validation"][1]). Only the LAST family gets the +1.
    validation_counts = [VALIDATION_PER_FAMILY, VALIDATION_PER_FAMILY, VALIDATION_PER_FAMILY + 1]

    phase_pools: dict[str, dict[str, list[ScenarioRecord]]] = {}
    phase_libraries: dict[str, Any] = {}
    for family_index, (family, loader) in enumerate(TRAINED_FAMILIES):
        pools = {
            "train": _family_scenario_pool(
                "train", network_loader=loader, family=family,
                seed_base=SEED_BASES[(family, "train")], count=counts["train"],
            ),
            "validation": _family_scenario_pool(
                "validation", network_loader=loader, family=family,
                seed_base=SEED_BASES[(family, "validation")], count=validation_counts[family_index],
            ),
            "calibration": _family_scenario_pool(
                "calibration", network_loader=loader, family=family,
                seed_base=SEED_BASES[(family, "calibration")], count=counts["calibration"],
            ),
        }
        phase_pools[family] = pools
        phase_libraries[family] = fit_pool_signature_library(pools["train"])

    run_root_base = RUNS_ROOT / f"EXPANDED-seed{seed}"
    resume_from: str | None = None
    phase_summaries: list[dict[str, Any]] = []
    model: HydroCore | None = None
    started = time.time()
    for phase_index, (family, _loader) in enumerate(TRAINED_FAMILIES):
        phase_epochs = PHASE_EPOCH_BOUNDARIES[phase_index]
        phase_config = replace(base_config, seed=seed, gradnorm_log_every_n_batches=5, early_stopping_patience=0, epochs=phase_epochs)
        set_deterministic_seed(phase_config.seed, deterministic=phase_config.deterministic)

        library = phase_libraries[family]
        train_view = CausalPrefixDatasetView(
            phase_pools[family]["train"], expected_split="train", signature_library=library,
            depth_policy=full_history_policy, base_seed=phase_config.seed, batch_size=phase_config.batch_size,
        )
        validation_view = CausalPrefixDatasetView(
            phase_pools[family]["validation"], expected_split="validation", signature_library=library,
            depth_policy=full_history_policy, base_seed=phase_config.seed, batch_size=phase_config.batch_size,
        )

        model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
        phase_run_root = run_root_base / f"phase{phase_index + 1}-{family}"
        trainer = Trainer(model, train_view, config=phase_config, run_root=phase_run_root, validation_dataset=validation_view)
        summary = trainer.fit(resume_from=resume_from)
        resume_from = summary.last_resumable_checkpoint or summary.final_checkpoint

        phase_summaries.append({
            "phase": phase_index + 1,
            "family": family,
            "epoch_boundary": phase_epochs,
            "train_scenario_count": len(phase_pools[family]["train"]),
            "validation_scenario_count": len(phase_pools[family]["validation"]),
            "signature_library_manifest_hash": library.manifest_hash,
            "train_manifest_hash": train_view.manifest_hash,
            "validation_manifest_hash": validation_view.manifest_hash,
            "training_summary": asdict(summary),
        })

    assert model is not None
    wall_seconds = time.time() - started
    final_export_path = phase_summaries[-1]["training_summary"]["export_path"]

    record = {
        "schema_version": 1,
        "purpose": "Milestone 7 Part A: EXPANDED topology-family training arm (sequential curriculum).",
        "arm": "EXPANDED",
        "seed": seed,
        "trained_families": families,
        "phase_epoch_boundaries": list(PHASE_EPOCH_BOUNDARIES),
        "total_train_scenarios": sum(len(phase_pools[f]["train"]) for f in families),
        "total_validation_scenarios": sum(len(phase_pools[f]["validation"]) for f in families),
        "total_calibration_scenarios": sum(len(phase_pools[f]["calibration"]) for f in families),
        "wall_seconds": wall_seconds,
        "phases": phase_summaries,
        "final_export_path": final_export_path,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    M7_RUNS_SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    (M7_RUNS_SUMMARY_ROOT / f"EXPANDED-seed{seed}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return {
        "export_path": final_export_path,
        "calibration_pools": {family: phase_pools[family]["calibration"] for family in families},
        "libraries": phase_libraries,
        "record": record,
    }


# ---------------------------------------------------------------------------
# Frozen B_DEPTH_AWARE calibration scheme (Milestone 3's decision), refit
# per-model on that model's OWN calibration pool -- exactly the "refit
# identically, never load a serialized artifact" convention every M4-M6
# script already follows. `network_id` is encoded as f"{family}:{bucket}"
# (Milestone-6 convention), now genuinely varying by FAMILY as well as
# bucket, which is what makes SplitConformalCalibrator.selection() able to
# report NETWORK_SPECIFIC vs CONDITION_SPECIFIC/GLOBAL fallback per family
# below -- the real "calibration applicability" signal Part A's metrics
# table needs.
# ---------------------------------------------------------------------------


def _fit_model_calibrator(model: HydroCore, calibration_pools: dict[str, list[ScenarioRecord]], libraries: dict[str, Any]) -> SplitConformalCalibrator:
    examples: list[CalibrationExample] = []
    with torch.no_grad():
        for family, records in calibration_pools.items():
            library = libraries[family]
            for depth in CAUSAL_PREFIX_DEPTHS:
                for record in records:
                    scenario = record.scenario
                    example = scenario_to_prefix_example(scenario, record.network, library, depth, feature_context=record.feature_context)
                    output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                    probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                    truth = int(example.targets["source_node"].item())
                    full_series = build_sensor_series(scenario, record.feature_context)
                    truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                    condition = classify_runtime_condition(truncated_series)
                    bucket = DEPTH_BUCKET_OF[depth]
                    examples.append(CalibrationExample(
                        probabilities=tuple(probs), true_index=truth, condition=condition,
                        network_id=f"{family}:{bucket}",
                    ))
    return SplitConformalCalibrator.fit(
        examples, alpha=ALPHA, model_hash="m7-topology", feature_schema_hash="n/a",
        dataset_manifest_hash="m7-calibration-pool",
    )


# ---------------------------------------------------------------------------
# Inference + per-row capture (shared by Part A's metrics table and Part
# B's offline fusion sweep -- one inference pass, no re-running the model).
# ---------------------------------------------------------------------------


def _infer(model: HydroCore, network, feature_context, series: Sequence[SensorSeries], library, target_timestamps) -> dict[str, Any]:
    classical_prior = model_input_classical_prior(library, list(library.node_ids), series, target_timestamps)
    window_steps = max(len(item.timestamps_seconds) for item in series)
    built = HydraulicFeatureBuilder().build(
        network, feature_context.graph, feature_context.state, series,
        classical_prior=classical_prior, window_steps=window_steps,
    )
    with torch.no_grad():
        output = model(built.batch)
    neural_probs = torch.softmax(output["source_node_logits"][0], dim=-1)
    neural_logits = output["source_node_logits"][0].detach().numpy()
    node_ids = list(built.node_ids)
    evidence_sufficiency = float(output["evidence_sufficiency"].reshape(-1)[0].item())
    return {
        "neural_probs": neural_probs.tolist(),
        "neural_logits": neural_logits,
        "node_ids": node_ids,
        "evidence_sufficiency": evidence_sufficiency,
        "condition": classify_runtime_condition(series),
    }


def _classical_belief(library, node_ids: Sequence[str], series: Sequence[SensorSeries], target_timestamps) -> tuple[list[float], float]:
    """Returns (belief over node_ids, residual_raw) where residual_raw is
    the classical library's own log1p-MSE residual at its argmax
    candidate -- the SAME quantity `SignatureLibrary.posterior_from_observations`
    computes internally per candidate (mirrored here, not reimplemented
    differently, to avoid any risk of drifting from the governed
    algorithm: `comparable = valid & isfinite(signature)`,
    `mean((log1p(observed)[comparable] - signature[comparable])**2)`),
    read out for the WINNING candidate only, as the real, data-derived
    input to TrustFeatures.normalized_residual (min-max normalized across
    the full evaluation batch by the caller, see module docstring)."""

    # Aligned onto library.node_ids (the library's own, junction-only
    # dimensionality it was fit against -- e.g. golden-reference has 4
    # junctions but built.node_ids/`node_ids` here has 6 graph nodes
    # including the reservoir/tank), matching _infer's own
    # model_input_classical_prior call exactly; re-keyed onto the
    # caller's `node_ids` ordering by NAME (not position) below, so
    # non-junction nodes simply get 0.0 belief.
    prior = model_input_classical_prior(library, list(library.node_ids), series, target_timestamps)
    belief = [float(prior.get(node_id, 0.0)) for node_id in node_ids]
    argmax_node = node_ids[int(np.argmax(belief))]
    if argmax_node not in library.signatures:
        return belief, 0.0
    observed, valid = aligned_observations_from_series(list(library.node_ids), series, target_timestamps)
    transformed = np.log1p(np.nan_to_num(observed, nan=0.0))
    signature = library.signatures[argmax_node]
    comparable = valid & np.isfinite(signature)
    if not comparable.any():
        return belief, 0.0
    residual_raw = float(np.mean((transformed[comparable] - signature[comparable]) ** 2))
    return belief, residual_raw


def _rank_metrics(belief: Sequence[float], truth_index: int) -> dict[str, Any]:
    row_probs = {index: float(value) for index, value in enumerate(belief)}
    return {
        "top1": localization_top_k(row_probs, truth_index, k=1),
        "top3": localization_top_k(row_probs, truth_index, k=3),
        "mrr": mean_reciprocal_rank([row_probs], [truth_index]),
    }


def _entropy_normalized(probs: Sequence[float]) -> float:
    values = np.clip(np.asarray(probs, dtype=np.float64), EPS, None)
    values = values / values.sum()
    raw_entropy = float(-np.sum(values * np.log2(values)))
    return raw_entropy / max(1.0, math.log2(len(probs)))


# ---------------------------------------------------------------------------
# Novelty-aware fusion (Part B). LOCAL-ONLY: never applied to
# src/hydroswarm/inference/fusion.py. Mirrors fuse_source_probabilities'
# own body exactly, swapping only the trust computation -- see module
# docstring for NOVELTY_TRUST_BOOST's provenance and the "diagnostic-only"
# scope this shares with component_fusion_analysis.py.
# ---------------------------------------------------------------------------


def novelty_aware_classical_trust(features: TrustFeatures, disagreement_js: float, topology_novelty: float) -> float:
    base_trust = dynamic_classical_trust(features, disagreement_js)
    return float(np.clip(base_trust + NOVELTY_TRUST_BOOST * topology_novelty, 0.05, 0.95))


def fuse_source_probabilities_novelty_aware(
    neural_logits: np.ndarray, classical_probabilities: np.ndarray, physical_mask: np.ndarray,
    features: TrustFeatures, topology_novelty: float,
) -> tuple[np.ndarray, FusionDiagnostics]:
    logits = np.asarray(neural_logits, dtype=np.float64)
    classical = np.asarray(classical_probabilities, dtype=np.float64)
    classical = classical / max(float(classical.sum()), EPS)
    mask = np.asarray(physical_mask, dtype=bool)
    shifted = logits - np.max(logits)
    neural = np.exp(shifted)
    neural = neural / neural.sum()
    divergence = jensen_shannon_divergence(neural, classical)
    trust = novelty_aware_classical_trust(features, divergence, topology_novelty)
    fused_logits = logits + trust * np.log(classical + 1e-12)
    fused_logits[~mask] = -np.inf
    fused_logits -= np.max(fused_logits[mask])
    fused = np.zeros_like(fused_logits)
    fused[mask] = np.exp(fused_logits[mask])
    fused /= fused.sum()
    return fused, FusionDiagnostics(classical_trust=trust, disagreement_js=divergence, masked_nodes=int((~mask).sum()))


# ---------------------------------------------------------------------------
# Evaluation-scenario generation and the full per-model, per-family
# evaluation pass. Development-only (DatasetSplit.DEVELOPMENT_HOLDOUT):
# never used for training or calibration fitting.
# ---------------------------------------------------------------------------


def _generate_eval_scenarios(family: str, loader, seed_base: int) -> list[tuple[Any, Any, Any]]:
    network_probe = loader()
    junctions = tuple(sorted(network_probe.junction_name_list))
    sources = junctions[: min(EVAL_MAX_SOURCES, len(junctions))]
    generator = WNTRScenarioGenerator()
    out = []
    for source_index, source in enumerate(sources):
        for repeat in range(EVAL_SEED_REPEATS):
            seed = seed_base + source_index * 1_000 + repeat
            network = loader()
            config = ScenarioGenerationConfig(
                seed=seed, network_id=family, network_family=family,
                split=DatasetSplit.DEVELOPMENT_HOLDOUT, stage=ScenarioCurriculumStage.OPERATIONAL,
                event_type=EventType.CONTAMINATION, source_node=source,
                sensor_count=min(len(junctions), 4), pipe_outage_probability=0.0,
            )
            scenario, randomized_network = generator.generate_with_network(network, config)
            context = build_feature_context(randomized_network)
            out.append((scenario, randomized_network, context))
    return out


def _evaluate_model_on_family(
    model: HydroCore, family: str, loader, library, *, known: bool,
) -> list[dict[str, Any]]:
    """One inference pass per (scenario, depth); captures everything Part
    A's metrics table and Part B's offline fusion sweep both need, so
    Part B re-uses these rows rather than re-running the model (see module
    docstring's "mined entirely from an already-computed dataset"
    convention)."""

    scenarios = _generate_eval_scenarios(family, loader, SEED_BASES[(family, "eval")])
    rows: list[dict[str, Any]] = []
    for scenario, network, context in scenarios:
        truth_node = scenario.manifest.incident.source_nodes[0]
        full_series = build_sensor_series(scenario, context)
        # FIXED full-history reference grid (matches library.signatures'
        # own shape, fit against full, untruncated scenarios) -- NOT the
        # depth-truncated series' own (shrinking) timestamps. Mirrors
        # run_m6_temporal.py's own `target_timestamps` convention exactly
        # (a single fixed grid threaded through every _infer call
        # regardless of the evidence depth/cadence being tested): feeding
        # a depth-shrunk target grid instead crashes
        # SignatureLibrary.posterior_from_observations with a shape
        # mismatch against the (fixed-length) fitted signature array --
        # confirmed directly this session (ValueError: operands could not
        # be broadcast together with shapes (3,4) (25,4)).
        reference_timestamps = full_series[0].timestamps_seconds if full_series else ()
        for depth in EVAL_DEPTHS:
            series = [truncate_causal_prefix(item, depth) for item in full_series]
            result = _infer(model, network, context, series, library, reference_timestamps)
            classical_belief, residual_raw = _classical_belief(library, result["node_ids"], series, reference_timestamps)
            node_ids = result["node_ids"]
            if truth_node not in node_ids:
                continue
            truth_index = node_ids.index(truth_node)
            healthy = statistics.fmean(item.health[-1] for item in series) if series else 1.0
            missing = statistics.fmean(float(item.missing[-1]) for item in series) if series else 0.0
            rows.append({
                "family": family, "known": known, "depth": depth, "depth_bucket": DEPTH_BUCKET_OF[depth],
                "seed": scenario.manifest.seed, "truth_node": truth_node, "truth_index": truth_index,
                "node_ids": node_ids, "neural_probs": result["neural_probs"], "neural_logits": result["neural_logits"],
                "classical_belief": classical_belief, "condition": result["condition"],
                "evidence_sufficiency": result["evidence_sufficiency"], "residual_raw": residual_raw,
                "healthy_sensor_fraction": healthy, "missing_rate": missing,
            })
    return rows


def _postprocess_rows(rows: list[dict[str, Any]], calibrator: SplitConformalCalibrator) -> None:
    """In-place: adds normalized_residual (batch min-max), TrustFeatures-
    derived fields, CURRENT_DYNAMIC and NOVELTY_AWARE fused beliefs
    (topology_novelty = 0.0 for `known` rows, 1.0 otherwise -- see module
    docstring's family-level novelty operationalization), calibration
    candidate sets/applicability (via the real, frozen B_DEPTH_AWARE
    SplitConformalCalibrator.selection()), and top1/top3/mrr for all three
    (neural/classical/hybrid) arms plus the novelty-aware arm."""

    if not rows:
        return
    residuals = np.asarray([row["residual_raw"] for row in rows], dtype=np.float64)
    lo, hi = float(residuals.min()), float(residuals.max())
    span = hi - lo
    for row, residual_raw in zip(rows, residuals, strict=True):
        normalized_residual = 0.0 if span <= EPS else float((residual_raw - lo) / span)
        neural_probs = row["neural_probs"]
        classical_belief = row["classical_belief"]
        topology_novelty = 0.0 if row["known"] else 1.0
        features = TrustFeatures(
            healthy_sensor_fraction=float(np.clip(row["healthy_sensor_fraction"], 0.0, 1.0)),
            missing_rate=float(np.clip(row["missing_rate"], 0.0, 1.0)),
            normalized_residual=normalized_residual,
            hydraulic_uncertainty=float(np.clip(1.0 - row["evidence_sufficiency"], 0.0, 1.0)),
            neural_entropy=_entropy_normalized(neural_probs),
            classical_entropy=_entropy_normalized(classical_belief),
            ood_score=topology_novelty,
        )
        node_ids = row["node_ids"]
        mask = np.ones(len(node_ids), dtype=bool)
        classical_array = np.asarray(classical_belief, dtype=np.float64)
        classical_array = classical_array / max(float(classical_array.sum()), EPS)
        current_fused, current_diag = fuse_source_probabilities(row["neural_logits"], classical_array, mask, features)
        novelty_fused, novelty_diag = fuse_source_probabilities_novelty_aware(
            row["neural_logits"], classical_array, mask, features, topology_novelty,
        )
        bucket = row["depth_bucket"]
        network_id = f"{row['family']}:{bucket}"
        source, group_key, _scores = calibrator.selection(condition=row["condition"], network_id=network_id)
        candidate = calibrator.candidate_set(neural_probs, condition=row["condition"], network_id=network_id)

        row["normalized_residual"] = normalized_residual
        row["trust_features"] = asdict(features)
        row["current_dynamic_belief"] = current_fused.tolist()
        row["current_dynamic_trust"] = current_diag.classical_trust
        row["novelty_aware_belief"] = novelty_fused.tolist()
        row["novelty_aware_trust"] = novelty_diag.classical_trust
        row["calibration_source"] = source
        row["calibration_group_key"] = group_key
        row["calibration_applicable"] = source == "NETWORK_SPECIFIC"
        row["candidate_set_size"] = len(candidate)
        row["candidate_set_includes_truth"] = row["truth_index"] in candidate
        truth_index = row["truth_index"]
        row["metrics_neural"] = _rank_metrics(neural_probs, truth_index)
        row["metrics_classical"] = _rank_metrics(classical_belief, truth_index)
        row["metrics_current_dynamic"] = _rank_metrics(current_fused, truth_index)
        row["metrics_novelty_aware"] = _rank_metrics(novelty_fused, truth_index)


# ---------------------------------------------------------------------------
# Part A: per-family, per-depth-bucket metrics table.
# ---------------------------------------------------------------------------


def _mean(rows: Sequence[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        node = row
        for key in keys:
            node = node[key]
        if node is not None:
            values.append(float(node))
    return statistics.fmean(values) if values else None


def _aggregate_family_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    families = sorted({row["family"] for row in rows})
    for family in families:
        table[family] = {}
        for bucket in ("EARLY", "MATURE"):
            subset = [row for row in rows if row["family"] == family and row["depth_bucket"] == bucket]
            if not subset:
                continue
            table[family][bucket] = {
                "n": len(subset),
                "known_to_this_model": subset[0]["known"],
                "neural": {metric: _mean(subset, "metrics_neural", metric) for metric in ("top1", "top3", "mrr")},
                "classical": {metric: _mean(subset, "metrics_classical", metric) for metric in ("top1", "top3", "mrr")},
                "hybrid_current_dynamic": {metric: _mean(subset, "metrics_current_dynamic", metric) for metric in ("top1", "top3", "mrr")},
                "candidate_set_size_mean": _mean(subset, "candidate_set_size"),
                "candidate_set_coverage": _mean(subset, "candidate_set_includes_truth"),
                "topology_novelty": 0.0 if subset[0]["known"] else 1.0,
                "calibration_applicability_rate": _mean(subset, "calibration_applicable"),
            }
    return table


# ---------------------------------------------------------------------------
# Part B: OOD-aware fusion. Gated on Part A's own finding (classical
# transfers better than neural on THIS model's unseen dev families).
# ---------------------------------------------------------------------------


def _fusion_sweep(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return {
        "n": len(rows),
        "classical_only": {metric: _mean(rows, "metrics_classical", metric) for metric in ("top1", "top3", "mrr")},
        "neural_only": {metric: _mean(rows, "metrics_neural", metric) for metric in ("top1", "top3", "mrr")},
        "current_dynamic_fusion": {metric: _mean(rows, "metrics_current_dynamic", metric) for metric in ("top1", "top3", "mrr")},
        "novelty_aware_fusion": {metric: _mean(rows, "metrics_novelty_aware", metric) for metric in ("top1", "top3", "mrr")},
        "mean_current_dynamic_trust": _mean(rows, "current_dynamic_trust"),
        "mean_novelty_aware_trust": _mean(rows, "novelty_aware_trust"),
        "calibration_applicability_rate": _mean(rows, "calibration_applicable"),
    }


def _part_b_for_model(model_label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    unseen_rows = [row for row in rows if not row["known"]]
    known_rows = [row for row in rows if row["known"]]
    unseen_classical_top1 = _mean(unseen_rows, "metrics_classical", "top1")
    unseen_neural_top1 = _mean(unseen_rows, "metrics_neural", "top1")
    classical_transfers_better = (
        unseen_classical_top1 is not None and unseen_neural_top1 is not None
        and unseen_classical_top1 > unseen_neural_top1
    )

    per_unseen_family = {}
    for family, _loader in UNSEEN_FAMILIES:
        family_rows = [row for row in unseen_rows if row["family"] == family]
        per_unseen_family[family] = _fusion_sweep(family_rows)

    novelty_gain_per_family = {}
    for family, sweep in per_unseen_family.items():
        if sweep is None:
            continue
        current = sweep["current_dynamic_fusion"]["top1"]
        novelty = sweep["novelty_aware_fusion"]["top1"]
        if current is not None and novelty is not None:
            novelty_gain_per_family[family] = novelty - current
    repeated_unseen_family_gain = bool(novelty_gain_per_family) and all(
        gain > 0 for gain in novelty_gain_per_family.values()
    )

    known_sweep = _fusion_sweep(known_rows)
    known_regression_pp = None
    no_meaningful_known_regression = True
    if known_sweep is not None:
        current_known = known_sweep["current_dynamic_fusion"]["top1"]
        novelty_known = known_sweep["novelty_aware_fusion"]["top1"]
        if current_known is not None and novelty_known is not None:
            known_regression_pp = (current_known - novelty_known) * 100
            no_meaningful_known_regression = known_regression_pp <= 0.5  # predeclared tolerance, matches M1's rounding-noise bar order of magnitude

    unseen_calibration_applicability = _mean(unseen_rows, "calibration_applicable")
    preserved_ood_truthfulness = bool(
        unseen_calibration_applicability is not None and unseen_calibration_applicability == 0.0
    )

    promotion = {
        "repeated_unseen_family_gain": repeated_unseen_family_gain,
        "no_meaningful_known_network_regression": no_meaningful_known_regression,
        "preserved_ood_truthfulness": preserved_ood_truthfulness,
        "promoted": bool(
            classical_transfers_better and repeated_unseen_family_gain
            and no_meaningful_known_regression and preserved_ood_truthfulness
        ),
    }

    return {
        "model": model_label,
        "gate_classical_transfers_better_on_unseen": classical_transfers_better,
        "unseen_classical_top1": unseen_classical_top1,
        "unseen_neural_top1": unseen_neural_top1,
        "pooled_unseen_fusion_sweep": _fusion_sweep(unseen_rows),
        "per_unseen_family_fusion_sweep": per_unseen_family,
        "novelty_aware_gain_pp_per_family": {family: gain * 100 for family, gain in novelty_gain_per_family.items()},
        "known_family_fusion_sweep": known_sweep,
        "known_network_regression_pp": known_regression_pp,
        "unseen_calibration_applicability_rate": unseen_calibration_applicability,
        "promotion": promotion,
        "note": (
            "Fusion sweep compares classical-only/neural-only/CURRENT_DYNAMIC "
            "(real fuse_source_probabilities)/NOVELTY_AWARE (this script's own local-only "
            "variant, never applied to src/hydroswarm/inference/fusion.py) purely offline over "
            "already-computed Part-A rows -- no new model inference this section. "
            "'gate_classical_transfers_better_on_unseen'=False means the fusion sweep below is "
            "reported for completeness but the milestone's own Part B trigger condition "
            "('If classical inference consistently transfers better on unseen topologies') did "
            "not hold for this model, so 'promoted' is forced False regardless of the sweep's "
            "own numbers."
        ),
    }


# ---------------------------------------------------------------------------
# Per-seed orchestration: build both arms' models/libraries/calibrators,
# evaluate both on every family in ALL_EVAL_FAMILIES.
# ---------------------------------------------------------------------------


def _run_seed(seed: int, shared_unseen_libraries: dict[str, Any]) -> dict[str, Any]:
    print(f"[m7] seed {seed}: loading CURRENT arm (frozen Milestone-1 checkpoint)...", flush=True)
    current_export_path = _current_arm_checkpoint(seed)
    current_model = _load_model(current_export_path)

    golden_train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    golden_library = fit_pool_signature_library(golden_train_records)
    golden_calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)

    current_libraries = {"golden-reference": golden_library, **shared_unseen_libraries}
    current_libraries["branched-loop"], _ = _classical_fit_library("branched-loop", TRAINED_FAMILIES[1][1])
    current_libraries["loop-grid"], _ = _classical_fit_library("loop-grid", TRAINED_FAMILIES[2][1])

    print(f"[m7] seed {seed}: fitting CURRENT arm's B_DEPTH_AWARE calibrator...", flush=True)
    current_calibrator = _fit_model_calibrator(
        current_model, {"golden-reference": golden_calibration_records}, {"golden-reference": golden_library},
    )

    current_known = {"golden-reference"}
    current_rows: list[dict[str, Any]] = []
    for family, loader in ALL_EVAL_FAMILIES:
        print(f"[m7] seed {seed}: evaluating CURRENT arm on {family}...", flush=True)
        rows = _evaluate_model_on_family(
            current_model, family, loader, current_libraries[family], known=family in current_known,
        )
        current_rows.extend(rows)
    _postprocess_rows(current_rows, current_calibrator)

    print(f"[m7] seed {seed}: training EXPANDED arm (3-phase sequential curriculum)...", flush=True)
    expanded = _train_expanded_arm(seed)
    expanded_model = _load_model(expanded["export_path"])
    expanded_known = {name for name, _ in TRAINED_FAMILIES}
    expanded_libraries = {**expanded["libraries"], **shared_unseen_libraries}

    print(f"[m7] seed {seed}: fitting EXPANDED arm's B_DEPTH_AWARE calibrator...", flush=True)
    expanded_calibrator = _fit_model_calibrator(expanded_model, expanded["calibration_pools"], expanded["libraries"])

    expanded_rows: list[dict[str, Any]] = []
    for family, loader in ALL_EVAL_FAMILIES:
        print(f"[m7] seed {seed}: evaluating EXPANDED arm on {family}...", flush=True)
        rows = _evaluate_model_on_family(
            expanded_model, family, loader, expanded_libraries[family], known=family in expanded_known,
        )
        expanded_rows.extend(rows)
    _postprocess_rows(expanded_rows, expanded_calibrator)

    return {
        "seed": seed,
        "current": {
            "export_path": current_export_path,
            "known_families": sorted(current_known),
            "family_table": _aggregate_family_table(current_rows),
            "part_b": _part_b_for_model("CURRENT", current_rows),
            "rows": current_rows,
        },
        "expanded": {
            "export_path": expanded["export_path"],
            "known_families": sorted(expanded_known),
            "family_table": _aggregate_family_table(expanded_rows),
            "part_b": _part_b_for_model("EXPANDED", expanded_rows),
            "rows": expanded_rows,
            "training_record": expanded["record"],
        },
    }


def _strip_rows_for_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-seed 'rows' carry full per-scenario provenance (useful for
    debugging/re-derivation) but are large and duplicate what family_table
    already aggregates; kept in a SEPARATE per-seed-rows file rather than
    inflating the main results JSON, matching M6's own "aggregate first,
    raw rows separately if needed" convention (M6 kept aggregates only in
    its main JSON)."""

    import copy

    trimmed = copy.deepcopy(payload)
    for arm in ("current", "expanded"):
        trimmed[arm].pop("rows", None)
    return trimmed


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for Milestone 7"
    locked_before = locked_test_opened(ROOT)
    started = time.time()

    print("[m7] fitting shared classical-fit-only libraries for unseen dev families...", flush=True)
    shared_unseen_libraries: dict[str, Any] = {}
    for family, loader in UNSEEN_FAMILIES:
        library, _records = _classical_fit_library(family, loader)
        shared_unseen_libraries[family] = library

    per_seed_full: list[dict[str, Any]] = []
    for seed in SEEDS:
        per_seed_full.append(_run_seed(seed, shared_unseen_libraries))

    wall_seconds = time.time() - started
    locked_after = locked_test_opened(ROOT)

    per_seed_trimmed = [_strip_rows_for_json(entry) for entry in per_seed_full]

    training_diversity_report = {
        "schema_version": 1,
        "section": "milestone_7_part_a_training_diversity",
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "seeds": list(SEEDS),
        "wall_seconds": wall_seconds,
        "current_arm_families": ["golden-reference"],
        "expanded_arm_families": [name for name, _ in TRAINED_FAMILIES],
        "unseen_dev_families": [name for name, _ in UNSEEN_FAMILIES],
        "eval_depths": list(EVAL_DEPTHS),
        "per_seed": [
            {
                "seed": entry["seed"],
                "current_family_table": entry["current"]["family_table"],
                "expanded_family_table": entry["expanded"]["family_table"],
                "current_export_path": entry["current"]["export_path"],
                "expanded_export_path": entry["expanded"]["export_path"],
            }
            for entry in per_seed_trimmed
        ],
    }

    ood_fusion_report = {
        "schema_version": 1,
        "section": "milestone_7_part_b_ood_aware_fusion",
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "novelty_trust_boost": NOVELTY_TRUST_BOOST,
        "per_seed": [
            {"seed": entry["seed"], "current": entry["current"]["part_b"], "expanded": entry["expanded"]["part_b"]}
            for entry in per_seed_trimmed
        ],
    }

    OUT_TRAINING_DIVERSITY.write_text(json.dumps(training_diversity_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUT_OOD_FUSION.write_text(json.dumps(ood_fusion_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    # -- Summary markdown --
    lines = ["# Milestone 7 summary: topology diversity and OOD-aware fusion", ""]
    lines.append(f"Seeds: {list(SEEDS)}. Wall seconds: {wall_seconds:.1f}.")
    lines.append("")
    lines.append("## Part A -- known-network (golden-reference) regression check")
    lines.append("")
    lines.append("| seed | arm | bucket | neural top1 | classical top1 | hybrid top1 |")
    lines.append("|---|---|---|---|---|---|")
    for entry in per_seed_trimmed:
        for arm_label, arm_key in (("CURRENT", "current"), ("EXPANDED", "expanded")):
            table = entry[arm_key]["family_table"].get("golden-reference", {})
            for bucket, stats in table.items():
                lines.append(
                    f"| {entry['seed']} | {arm_label} | {bucket} | {stats['neural']['top1']:.3f} | "
                    f"{stats['classical']['top1']:.3f} | {stats['hybrid_current_dynamic']['top1']:.3f} |"
                )
    lines.append("")
    lines.append("## Part A -- unseen dev-family generalization (MATURE bucket, hybrid top1)")
    lines.append("")
    lines.append("| seed | family | CURRENT (unseen) | EXPANDED (unseen unless noted) |")
    lines.append("|---|---|---|---|")
    for entry in per_seed_trimmed:
        for family, _ in UNSEEN_FAMILIES + TRAINED_FAMILIES[1:]:
            current_stats = entry["current"]["family_table"].get(family, {}).get("MATURE")
            expanded_stats = entry["expanded"]["family_table"].get(family, {}).get("MATURE")
            expanded_known = family in entry["expanded"]["known_families"]
            current_val = f"{current_stats['hybrid_current_dynamic']['top1']:.3f}" if current_stats else "n/a"
            expanded_val = f"{expanded_stats['hybrid_current_dynamic']['top1']:.3f}{'*' if expanded_known else ''}" if expanded_stats else "n/a"
            lines.append(f"| {entry['seed']} | {family} | {current_val} | {expanded_val} |")
    lines.append("")
    lines.append("(* = family EXPANDED trained on; unmarked = unseen to that arm.)")
    lines.append("")
    lines.append("## Part B -- OOD-aware fusion promotion decision")
    lines.append("")
    lines.append("| seed | model | gate (classical>neural, unseen) | promoted |")
    lines.append("|---|---|---|---|")
    for entry in per_seed_trimmed:
        for arm_label, arm_key in (("CURRENT", "current"), ("EXPANDED", "expanded")):
            part_b = entry[arm_key]["part_b"]
            lines.append(
                f"| {entry['seed']} | {arm_label} | {part_b['gate_classical_transfers_better_on_unseen']} | "
                f"{part_b['promotion']['promoted']} |"
            )
    lines.append("")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "wall_seconds": wall_seconds,
        "locked_test_opened_after": locked_after,
        "seeds": list(SEEDS),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
