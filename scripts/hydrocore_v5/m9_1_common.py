"""Shared, governed helpers for the Milestone 9.1 scientific runner.

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_1_PROTOCOL.md`, including its
2026-08-16 Section-21 addendum. This module implements NO new scientific
policy of its own -- every constant/threshold/formula below is copied
verbatim from the frozen document (section citations given inline) or from
the already-frozen preflight/correction artifacts it cites. It exists only
to avoid re-deriving the same pinned constants in every M9.1 script.

Two runner-level (non-scientific) engineering decisions this module makes,
both logged here for the avoidance of doubt (neither touches a metric,
threshold, split, seed, or promotion rule -- Section 21 amendments are not
required for either):

1. GRAPH_SDE per-Monte-Carlo-draw seeding (protocol Section 9). `HydroCore
   .forward` (frozen, not modified by this milestone) calls
   `self.temporal_dynamics(...)` with exactly nine positional arguments and
   no keyword pass-through, so `GraphSDEDynamics.forward`'s `seed=` keyword
   can never be reached through an ordinary `model(batch)` call. This module
   supplies the seed for one evaluation call by temporarily rebinding the
   `GraphSDEDynamics` INSTANCE's `forward` attribute to
   `functools.partial(bound_forward, seed=...)` for the duration of exactly
   one `model(batch)` call (see `sde_forward_seed`), then restores the
   original bound method. Verified byte-exact against a direct
   `dynamics(*args, seed=...)` call before being relied on anywhere (see
   `tests/scientific/test_m9_1_runner.py`). Zero lines of
   `continuous_time.py`/`core.py` are touched.
2. Brownian `incident_id` for CALIBRATION-split rows (protocol Section 9
   defines the formula only in terms of `development_holdout`, since that is
   the split under discussion in the guardrail/primary-metric context; the
   milestone's own researcher-degrees-of-freedom audit, Section 20, frames
   the pinned rule as "the existing deterministic scenario-generation seed,
   not a new ID scheme" in general, not literally restricted to one split
   name). GRAPH_SDE calibration-fitting forward passes also require a
   concrete seed, so this module applies the SAME already-existing
   convention to whichever split is being processed:
   `SPLIT_SEED_RANGES[split][0] + index * 100`, where `index` is the
   scenario's own position in `build_scenario_pool(split, ...)`'s returned
   list -- the exact integer that pool-builder itself already computes
   internally for that scenario, for both `development_holdout` and
   `calibration`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydroswarm.calibration.conformal import classify_runtime_condition  # noqa: E402
from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.model.continuous_time import GraphCDEDynamics, GraphODEDynamics, GraphSDEDynamics  # noqa: E402
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    SPLIT_SEED_RANGES,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402

from m9_0b_calibration_schemes import MINIMUM_GROUP_SIZE as _M9_0B_MIN_GROUP_SIZE, SchemeRow, fit_scheme  # noqa: E402
from run_m8_7_arm import ARM_DEFINITIONS as M8_7_ARM_DEFINITIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Section 21(c): the two mandatory commit pins, plus branch/paths (Section
# header + Section 19).
# ---------------------------------------------------------------------------

PROTOCOL_FROZEN_AT_COMMIT = "0f05be1d47258a8c3d19e3a0d0e1122e3e560069"
#: Section 21(a): first superseding pin (dated 2026-08-16) -- added the
#: frozen GRAPH_ODE max_num_steps bound, no other model change.
CODE_UNDER_TEST_COMMIT_FLOOR_V1 = "154605180f2a950d86452cfc8ec7202990aba8cf"
#: Section 21 (dated 2026-08-17): second superseding pin. M9.7 commit
#: 475874d ("feat(v5-causal): add governed HydroCore-M capacity config")
#: registers MODEL_VARIANTS["small_v5_capacity_m"] in
#: src/hydroswarm/model/core.py -- the ONLY commit touching any
#: FROZEN_UNCHANGED_PATHS file between CODE_UNDER_TEST_COMMIT_FLOOR_V1 and
#: this pin (verified: `git log CODE_UNDER_TEST_COMMIT_FLOOR_V1..475874d --
#: <FROZEN_UNCHANGED_PATHS>` lists exactly this one commit). Audited and
#: proven additive-only before being accepted as the new floor -- see
#: reports/evaluation/hydrocore-v5/m9-1-provenance-refresh/ and this
#: document's own 2026-08-17 amendment (Section 21):
#:   - `MODEL_VARIANTS["small"]` line is byte-identical (git diff of the
#:     hunk adds exactly one new dict entry, changes zero existing lines);
#:   - continuous_time.py and training-v5-causal.yaml have a literal empty
#:     diff between the two pins (nothing to audit there at all);
#:   - MODEL_VARIANTS is looked up strictly by key
#:     (`MODEL_VARIANTS[variant.lower()]`, core.py), never iterated, so an
#:     added key cannot change "small"'s resolved configuration or any
#:     GRAPH_ODE/CDE/SDE construction path;
#:   - mechanically confirmed: constructing "small" at this commit with
#:     M9.1's own frozen `SHARED_MODEL_CONFIG`/`CURRENT_MODEL_KWARGS`
#:     yields exactly `CURRENT_BASELINE_TOTAL_PARAMS` (4,182,612) trainable
#:     parameters, unchanged.
#: `assert_code_under_test_commit` accepts this commit OR a later commit on
#: the same branch that changes nothing under the three paths below (the
#: header's own re-preflight-trigger rule) -- ANY future change to those
#: paths beyond this pin still trips the guard exactly as before; this is a
#: narrow, audited exception for one specific historical commit, not a
#: relaxation of the rule itself.
CODE_UNDER_TEST_COMMIT_FLOOR = "475874d8977d0952e8fc3626eb2bd6580cc3c2f7"
FROZEN_UNCHANGED_PATHS = (
    "src/hydroswarm/model/continuous_time.py",
    "src/hydroswarm/model/core.py",
    "configs/training-v5-causal.yaml",
)
FROZEN_BRANCH = "exp/hydrocore-v5-causal"

CONFIG_PATH = ROOT / "configs" / "training-v5-causal.yaml"
RUN_ROOT = ROOT / "experiments" / "runs" / "hydrocore-v5-causal-m9-1"
REPORT_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5"
RUNS_DIR = REPORT_DIR / "m9-1-runs"
M8_7_RUNS_DIR = REPORT_DIR / "m8-7-runs"
RESULTS_PATH = REPORT_DIR / "m9-1-results.json"
CALIBRATION_PATH = REPORT_DIR / "m9-1-calibration.json"
GUARDRAILS_PATH = REPORT_DIR / "m9-1-guardrails.json"
SUMMARY_PATH = REPORT_DIR / "m9-1-summary.md"
CLOSURE_PATH = REPORT_DIR / "m9-1-closure.json"

# ---------------------------------------------------------------------------
# Section 1: arms. Section 5: seeds.
# ---------------------------------------------------------------------------

NOVEL_ARMS = ("GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE")
ALL_ARMS = ("CURRENT", "GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE")
SCREENING_SEEDS = (20260814, 31874)
CONFIRMATION_SEED = 20260815
REPRESENTATIVE_SEED = 31874  # Step 2's screening-stage pairing seed (Section 12).
ALL_M8_7_SEEDS = (20260814, 31874, 20260815)  # CURRENT's three reusable checkpoint seeds.

# ---------------------------------------------------------------------------
# Section 2: frozen entry recipe.
# ---------------------------------------------------------------------------

FEATURE_KWARGS: dict[str, Any] = {"unobserved_age_sentinel": "fixed", "include_relative_gap_feature": False}
ALPHA = 0.1
NETWORK_FAMILY = "golden-reference"
MINIMUM_GROUP_SIZE = 10
assert MINIMUM_GROUP_SIZE == _M9_0B_MIN_GROUP_SIZE

SHARED_MODEL_CONFIG: dict[str, Any] = dict(
    prior_mode="feature_only",
    event_control_heads=True,
    scout_control_heads=True,
    strategist_mode="candidate_conditioned",
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True,
    ood_category_head=True,
)
CURRENT_MODEL_KWARGS: dict[str, Any] = M8_7_ARM_DEFINITIONS["AGE_FIX_ONLY"]["model_kwargs"]
GRADNORM_LOG_EVERY_N_BATCHES = 5

# ---------------------------------------------------------------------------
# Section 4: depth grid / maturity buckets.
# ---------------------------------------------------------------------------

FULL_HISTORY_DEPTH = 25
EARLY_DEPTHS = (1, 2, 3)
MID_DEPTHS = (4, 6)
MATURE_DEPTHS = (12, 25)
assert tuple(CAUSAL_PREFIX_DEPTHS) == (1, 2, 3, 4, 6, 12, 25)
DEPTH_BUCKET_OF: dict[int, str] = {
    **{d: "EARLY" for d in EARLY_DEPTHS},
    **{d: "MID" for d in MID_DEPTHS},
    **{d: "MATURE" for d in MATURE_DEPTHS},
}
#: Section 11.1: the primary metric's own depth-bucket composition.
PRIMARY_METRIC_DEPTHS = EARLY_DEPTHS + MID_DEPTHS

# ---------------------------------------------------------------------------
# Section 6: frozen architecture widths (verbatim from
# m9-1-preflight-correction-results.json's parameter_matching.corrected).
# ---------------------------------------------------------------------------

ARM_MLP_WIDTH: dict[str, int] = {"GRAPH_ODE": 574, "GRAPH_CDE": 214, "GRAPH_SDE": 464}
CURRENT_BASELINE_TOTAL_PARAMS = 4182612

# ---------------------------------------------------------------------------
# Section 7: solver configuration.
# ---------------------------------------------------------------------------

ODE_MAX_NUM_STEPS = 2000
ODE_RTOL = 1e-3
ODE_ATOL = 1e-4
CDE_STEP_SIZE = 0.25
SDE_DT = 0.05
SDE_DIFFUSION_SCALE = 0.1

# ---------------------------------------------------------------------------
# Section 9: GRAPH_SDE Monte Carlo evaluation policy.
# ---------------------------------------------------------------------------

SDE_MC_COUNT = 4


def brownian_seed(predictor_training_seed: int, incident_id: int, prefix_depth: int, mc_index: int) -> int:
    """Frozen formula, protocol Section 9, reproduced verbatim."""

    key = f"{predictor_training_seed}:{incident_id}:{prefix_depth}:{mc_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31)


def incident_id_for(split: str, index: int) -> int:
    """Deterministic per-scenario generation seed for `split`, the SAME
    integer `build_scenario_pool` itself computes internally for that
    scenario -- see module docstring item 2 for why this is applied to both
    `development_holdout` and `calibration`."""

    start_seed, _count = SPLIT_SEED_RANGES[split]
    return start_seed + index * 100


# ---------------------------------------------------------------------------
# Section 12: guardrail thresholds (frozen, verbatim).
# ---------------------------------------------------------------------------

GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP = 5.0
GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP = 3.0
GUARDRAIL_MAX_MRR_REGRESSION = 0.05
GUARDRAIL_MIN_COVERAGE = 0.85
#: Section 11.2 / M9.0b Section 10's own normalized candidate-set-size bar.
GUARDRAIL_MAX_NORMALIZED_SET_SIZE = 0.5

# ---------------------------------------------------------------------------
# Section 14: statistical procedure (frozen, verbatim, reusing
# run_m8_7_bootstrap.py's own already-tested resampling loop).
# ---------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_INTERVAL = 0.90

EPS = 1e-9


# ---------------------------------------------------------------------------
# Section 19(d) / 21(c): SHA and lock-state assertions.
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def current_commit() -> str:
    return _git("rev-parse", "HEAD")


def _github_pr_context() -> tuple[str | None, str | None]:
    """Return the trusted PR source ref/SHA for a GitHub detached merge checkout."""
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return None, None
    ref, sha = os.environ.get("GITHUB_HEAD_REF"), None
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path:
        try:
            event = json.loads(Path(event_path).read_text())
            head = event.get("pull_request", {}).get("head", {})
            ref = head.get("ref", ref)
            sha = head.get("sha")
        except (OSError, json.JSONDecodeError):
            return None, None
    return ref, sha


def _accepted_checkout_identity(
    *, head: str, branch: str, pr_ref: str | None, pr_head: str | None,
    is_ancestor: callable,
) -> bool:
    if branch:
        return branch == FROZEN_BRANCH
    return pr_ref == FROZEN_BRANCH and bool(pr_head) and is_ancestor(pr_head, head)


def assert_code_under_test_commit() -> str:
    """Section header + Section 21(a): the executing commit must be
    `CODE_UNDER_TEST_COMMIT_FLOOR` or a later commit on `FROZEN_BRANCH` that
    changes nothing under `FROZEN_UNCHANGED_PATHS`. Returns the verified
    commit SHA (the `code_under_test_commit` value every artifact records)."""

    head = current_commit()
    branch = _git("branch", "--show-current")
    pr_ref, pr_head = _github_pr_context()
    def is_ancestor(candidate: str, descendant: str) -> bool:
        try:
            _git("merge-base", "--is-ancestor", candidate, descendant)
            return True
        except subprocess.CalledProcessError:
            return False
    if not _accepted_checkout_identity(head=head, branch=branch, pr_ref=pr_ref, pr_head=pr_head, is_ancestor=is_ancestor):
        raise AssertionError(f"must execute on {FROZEN_BRANCH!r} or its verified GitHub PR merge, got {branch!r}")
    merge_base = _git("merge-base", CODE_UNDER_TEST_COMMIT_FLOOR, head)
    if merge_base != CODE_UNDER_TEST_COMMIT_FLOOR:
        raise AssertionError(
            f"HEAD {head} is not a descendant of the pinned floor commit {CODE_UNDER_TEST_COMMIT_FLOOR}"
        )
    if head != CODE_UNDER_TEST_COMMIT_FLOOR:
        diff = _git("diff", "--name-only", CODE_UNDER_TEST_COMMIT_FLOOR, head, "--", *FROZEN_UNCHANGED_PATHS)
        if diff.strip():
            raise AssertionError(
                f"HEAD {head} changes frozen paths vs {CODE_UNDER_TEST_COMMIT_FLOOR}: {diff.strip()}"
            )
    return head


def assert_locked_test_closed() -> bool:
    opened = locked_test_opened(ROOT)
    if opened:
        raise AssertionError("locked_final_test/locked_topology_test must remain unopened for M9.1")
    return opened


# ---------------------------------------------------------------------------
# GRAPH_SDE call-time seed injection (module docstring item 1).
# ---------------------------------------------------------------------------


@contextmanager
def sde_forward_seed(dynamics: GraphSDEDynamics, seed: int) -> Iterator[None]:
    original_forward = dynamics.forward
    dynamics.forward = functools.partial(original_forward, seed=seed)  # type: ignore[method-assign]
    try:
        yield
    finally:
        # Bound methods are not cached/singleton -- `dynamics.forward`
        # allocates a fresh bound-method object on every access, so an
        # identity check against `original_forward` here would spuriously
        # fail even on a fully correct restore. Deleting the instance
        # override is sufficient: attribute lookup falls back to the class
        # method again, byte-for-byte equivalent to the pre-patch state
        # (verified behaviorally in tests/scientific/test_m9_1_runner.py).
        del dynamics.__dict__["forward"]


# ---------------------------------------------------------------------------
# Model construction (Section 6, Section 2's SHARED_MODEL_CONFIG).
# ---------------------------------------------------------------------------


def build_current_model() -> HydroCore:
    return HydroCore.from_variant("small", use_adapters=False, **CURRENT_MODEL_KWARGS, **SHARED_MODEL_CONFIG)


def build_dynamics(arm: str) -> torch.nn.Module:
    baseline = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    common = dict(
        temporal_feature_dim=baseline.temporal_feature_dim,
        quality_feature_dim=baseline.quality_feature_dim,
        d_model=baseline.d_model,
        edge_feature_dim=baseline.edge_feature_dim,
        mlp_width=ARM_MLP_WIDTH[arm],
    )
    if arm == "GRAPH_ODE":
        dynamics = GraphODEDynamics(**common, rtol=ODE_RTOL, atol=ODE_ATOL, max_num_steps=ODE_MAX_NUM_STEPS)
        assert dynamics.max_num_steps == ODE_MAX_NUM_STEPS
        return dynamics
    if arm == "GRAPH_CDE":
        return GraphCDEDynamics(**common, step_size=CDE_STEP_SIZE)
    if arm == "GRAPH_SDE":
        return GraphSDEDynamics(**common, diffusion_scale=SDE_DIFFUSION_SCALE, dt=SDE_DT)
    raise ValueError(arm)


def build_novel_model(arm: str) -> HydroCore:
    dynamics = build_dynamics(arm)
    return HydroCore.from_variant("small", use_adapters=False, temporal_dynamics=dynamics, **SHARED_MODEL_CONFIG)


def build_model(arm: str) -> HydroCore:
    return build_current_model() if arm == "CURRENT" else build_novel_model(arm)


# ---------------------------------------------------------------------------
# Section 15: per-(arm, seed) run records.
# ---------------------------------------------------------------------------


def run_record_path(arm: str, seed: int) -> Path:
    return RUNS_DIR / f"{arm}-seed{seed}.json"


def load_run_record(arm: str, seed: int) -> dict[str, Any]:
    return json.loads(run_record_path(arm, seed).read_text())


def load_checkpoint_from_record(record: dict[str, Any]) -> HydroCore:
    from safetensors.torch import load_file

    arm = record["arm"]
    model = build_model(arm)
    export_path = record["training_summary"]["export_path"] if arm != "CURRENT" else record["export_path"]
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Section 11.1/11.2: per-row metric block, reused verbatim from
# evaluate_m1_depths.py's own formula set.
# ---------------------------------------------------------------------------


def per_row_metrics(probs_tensor: torch.Tensor, truth: int) -> dict[str, float]:
    row_probs = {position: float(value) for position, value in enumerate(probs_tensor.tolist())}
    p_truth = row_probs.get(truth, 0.0)
    onehot = torch.zeros_like(probs_tensor)
    onehot[truth] = 1.0
    rank = int((probs_tensor > p_truth).sum().item()) + 1
    return {
        "top1": localization_top_k(row_probs, truth, k=1),
        "top3": localization_top_k(row_probs, truth, k=3),
        "mrr": mean_reciprocal_rank([row_probs], [truth]),
        "nll": -__import__("math").log(p_truth + EPS),
        "brier": float(torch.sum((probs_tensor - onehot) ** 2)),
        "posterior_entropy": float(-torch.sum(probs_tensor * torch.log(probs_tensor + EPS))),
        "true_source_rank": rank,
    }


# ---------------------------------------------------------------------------
# Bootstrap (Section 14), reused verbatim from run_m8_7_bootstrap.py's own
# already-tested resampling loop.
# ---------------------------------------------------------------------------


def paired_bootstrap(
    candidate_per_incident: Sequence[float],
    control_per_incident: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    interval: float = BOOTSTRAP_INTERVAL,
) -> dict[str, Any]:
    if len(candidate_per_incident) != len(control_per_incident):
        raise ValueError("paired bootstrap requires equal-length, same-incident-order sequences")
    paired_diff = np.asarray(candidate_per_incident) - np.asarray(control_per_incident)
    observed_mean_diff = float(paired_diff.mean())
    rng = np.random.default_rng(seed)
    n = len(paired_diff)
    resample_means = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = paired_diff[idx].mean()
    lower_pct = (1 - interval) / 2 * 100
    upper_pct = (1 - (1 - interval) / 2) * 100
    ci_lower = float(np.percentile(resample_means, lower_pct))
    ci_upper = float(np.percentile(resample_means, upper_pct))
    return {
        "n_incidents": n,
        "resamples": resamples,
        "bootstrap_seed": seed,
        "interval": interval,
        "observed_mean_diff": observed_mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_entirely_positive": bool(ci_lower > 0),
        "ci_entirely_non_positive": bool(ci_upper <= 0),
    }


# ---------------------------------------------------------------------------
# Scenario pools / signature library (Section 3, reused verbatim machinery).
# ---------------------------------------------------------------------------


def load_pool(split: str) -> list[Any]:
    return build_scenario_pool(split, network_loader=build_wntr_network)


def fit_library(train_records: Sequence[Any]) -> Any:
    return fit_pool_signature_library(train_records)


# ---------------------------------------------------------------------------
# Row-level evaluation (single scenario, single depth), handling GRAPH_SDE's
# frozen 4-MC-mean policy (Section 9) uniformly with the other three arms.
# ---------------------------------------------------------------------------


def evaluate_row(
    model: HydroCore,
    arm: str,
    seed: int,
    record: Any,
    incident_id: int,
    library: Any,
    depth: int,
) -> dict[str, Any]:
    example = scenario_to_prefix_example(
        record.scenario, record.network, library, depth, feature_context=record.feature_context, **FEATURE_KWARGS,
    )
    inputs = {key: value.unsqueeze(0) for key, value in example.inputs.items()}
    truth = int(example.targets["source_node"].item())

    mc_draw_probs: list[list[float]] | None = None
    mc_variance: list[float] | None = None
    solver_step_limit_exceeded = False
    try:
        if arm == "GRAPH_SDE":
            draws = []
            for mc_index in range(SDE_MC_COUNT):
                call_seed = brownian_seed(seed, incident_id, depth, mc_index)
                with sde_forward_seed(model.temporal_dynamics, call_seed), torch.no_grad():
                    output = model(inputs)
                draws.append(torch.softmax(output["source_node_logits"][0], dim=-1))
            stacked = torch.stack(draws, dim=0)
            probs_tensor = stacked.mean(dim=0)
            mc_draw_probs = [d.tolist() for d in draws]
            mc_variance = stacked.var(dim=0, unbiased=False).tolist()
        else:
            with torch.no_grad():
                output = model(inputs)
            probs_tensor = torch.softmax(output["source_node_logits"][0], dim=-1)
    except AssertionError as error:
        # torchdiffeq's dopri5 stepper enforces max_num_steps via a bare
        # assert (see run_m9_1_train_arm.py's identical catch) -- protocol
        # Section 7: recorded as SOLVER_STEP_LIMIT_EXCEEDED, never retried.
        if "max_num_steps exceeded" not in str(error):
            raise
        solver_step_limit_exceeded = True
        nodes = int(inputs["node_mask"].shape[1])
        probs_tensor = torch.full((nodes,), float("nan"))  # never used for a metric: overridden by the flag below.

    if solver_step_limit_exceeded:
        metrics = {"top1": 0.0, "top3": 0.0, "mrr": 0.0, "nll": float("inf"), "brier": float("nan"), "posterior_entropy": float("nan"), "true_source_rank": -1}
    else:
        metrics = per_row_metrics(probs_tensor, truth)
    full_series = build_sensor_series(record.scenario, record.feature_context)
    truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
    condition = classify_runtime_condition(truncated_series)

    return {
        "incident_id": incident_id,
        "depth": depth,
        "depth_bucket": DEPTH_BUCKET_OF[depth],
        "truth_index": truth,
        "condition": condition,
        "probabilities": probs_tensor.tolist(),
        "metrics": metrics,
        "sde_mc_draws": mc_draw_probs,
        "sde_mc_variance": mc_variance,
        "solver_step_limit_exceeded": solver_step_limit_exceeded,
        "forward_finite": False if solver_step_limit_exceeded else bool(torch.isfinite(probs_tensor).all()),
    }


def evaluate_split(model: HydroCore, arm: str, seed: int, records: Sequence[Any], split: str, library: Any) -> list[dict[str, Any]]:
    """One row per (scenario, depth) over every depth in CAUSAL_PREFIX_DEPTHS,
    for the full `records` population of `split` -- Section 4's own
    "whatever the deterministic pool function returns IS the evaluation
    population" convention."""

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        incident_id = incident_id_for(split, index)
        for depth in CAUSAL_PREFIX_DEPTHS:
            row = evaluate_row(model, arm, seed, record, incident_id, library, depth)
            row["scenario_index"] = index
            rows.append(row)
    return rows


def to_scheme_row(row: dict[str, Any]) -> SchemeRow:
    return SchemeRow(
        probabilities=tuple(row["probabilities"]),
        true_index=row["truth_index"],
        condition=row["condition"],
        family=NETWORK_FAMILY,
        depth_bucket=row["depth_bucket"],
    )


def fit_b_depth_aware(rows: Sequence[dict[str, Any]], *, model_hash: str) -> Any:
    scheme_rows = [to_scheme_row(row) for row in rows]
    return fit_scheme("CURRENT_FAMILY_DEPTH", scheme_rows, alpha=ALPHA, model_hash=model_hash, minimum_group_size=MINIMUM_GROUP_SIZE)


def primary_metric_per_incident(rows_by_depth: dict[int, dict[str, Any]]) -> float:
    """Section 11.1: mean of top1(depth) over PRIMARY_METRIC_DEPTHS only,
    for one development_holdout incident."""

    return statistics.fmean(rows_by_depth[d]["metrics"]["top1"] for d in PRIMARY_METRIC_DEPTHS)


# ---------------------------------------------------------------------------
# Artifact merge helpers (never silently drop unrelated prior entries).
# ---------------------------------------------------------------------------


def merge_nested_json(path: Path, top_key: str, inner_key: str, value: Any) -> dict[str, Any]:
    """Sets existing[top_key][inner_key] = value without disturbing any
    other top_key/inner_key entry already on disk -- the mechanism this
    milestone uses so re-running evaluate/calibrate/decide for one
    (arm, seed) or one stage never silently overwrites another's results."""

    existing: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    existing.setdefault(top_key, {})[inner_key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return existing
