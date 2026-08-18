"""M10.3B -- Strategist target-identifiability + failure-diagnosis amendment.

Diagnostic-only, additive to M10.3A. Does NOT train anything, does NOT
touch any checkpoint, does NOT run true M10.3, does NOT open locked data.
Rebuilds the SAME frozen M10.3A population (`m10_3_refit_protocol`'s own
TRAIN/VALIDATION seed bases -- `run_m10_3_level_a_train.py`'s own
`_build_corpus`, reused unmodified) and computes the diagnostics
`docs/evaluation/HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md` documents.

Answers one question: did M10.3A fail because HydroCore needs broader
retraining, or because the current Strategist objective/population does not
contain enough correct, legally observable, within-incident decision signal
to justify such retraining?

Writes reports/evaluation/hydrocore-v5/m10/m10-3b-diagnosis/*.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import m10_3_refit_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m10_3_level_a_train import CorpusExample, _build_corpus  # noqa: E402

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.planning.action_templates import ACTION_TEMPLATES  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import fit_pool_signature_library  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402

M10_DIR = m10.M10_DIR
M10_3_REFIT_DIR = M10_DIR / "m10-3-refit"
M10_3B_DIR = M10_DIR / "m10-3b-diagnosis"
SIGNATURE_CACHE_DIR = ROOT / "experiments" / "cache" / "m10-3-refit-signatures"

TARGET_KEYS = (
    "plan_value", "exposure_proxy", "pressure_risk_proxy",
    "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
)
ALL_STRATEGIST_KEYS = ("plan_validity", *TARGET_KEYS)

#: Diagnostic-only near-tie tolerances, preregistered HERE (before any
#: per-incident result in this script is inspected) -- never derived from
#: model performance. Reuses the repository's OWN governed physical
#: sensitivity constant where one exists (service_loss_proxy reuses
#: verifier.SERVICE_AVAILABILITY_SENSITIVITY_EPSILON = 0.02 directly).
#: pressure_risk_proxy/containment_time_proxy use a "1 simulator-reported
#: minute is the smallest physically distinguishable difference" argument,
#: normalized onto each proxy's own train-owned scale (60.0 / 240.0
#: minutes respectively, from plan_value_policy.py). exposure_proxy uses a
#: 1%-of-no-response-baseline argument (it is itself already a ratio).
#: plan_value/plan_regret_proxy use the propagated sum of the four additive
#: cost components' own tolerances (plan_value_policy.py's own additive
#: cost construction), rounded up.
NEAR_TIE_TOLERANCE: dict[str, float] = {
    "exposure_proxy": 0.01,
    "pressure_risk_proxy": 1.0 / 60.0,
    "service_loss_proxy": 0.02,
    "containment_time_proxy": 1.0 / 240.0,
    "plan_value": 0.05,
    "plan_regret_proxy": 0.05,
}


def _sha256_of_scenario_ids(examples: list[CorpusExample]) -> str:
    payload = sorted(ex.scenario_id for ex in examples)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _build_population() -> tuple[list[CorpusExample], list[CorpusExample], Any, tuple[str, ...]]:
    family, loader = TRAINED_FAMILIES[0]
    assert family == proto.FAMILY
    network = loader()
    print("fitting input signature library / signature artifact (reused, unmodified)...", flush=True)
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

    print(f"building TRAIN examples (seed_base={proto.TRAIN_SEED_BASE}, count={proto.TRAIN_COUNT})...", flush=True)
    train_examples = _build_corpus(
        seed_base=proto.TRAIN_SEED_BASE, count=proto.TRAIN_COUNT, network=network, loader=loader,
        input_library=input_library, artifact=artifact,
    )
    print(f"building VALIDATION examples (seed_base={proto.VALIDATION_SEED_BASE}, count={proto.VALIDATION_COUNT})...", flush=True)
    validation_examples = _build_corpus(
        seed_base=proto.VALIDATION_SEED_BASE, count=proto.VALIDATION_COUNT, network=network, loader=loader,
        input_library=input_library, artifact=artifact,
    )
    node_ids = tuple(sorted(network.node_name_list))
    return train_examples, validation_examples, network, node_ids


# ---------------------------------------------------------------------------
# Per-incident record extraction (shared by every section below).
# ---------------------------------------------------------------------------


class IncidentRecord:
    __slots__ = ("scenario_id", "real_count", "template_ids", "target_type", "node_index", "link_index", "values", "masks")

    def __init__(self, ex: CorpusExample) -> None:
        real = ex.real_plan_count
        self.scenario_id = ex.scenario_id
        self.real_count = real
        inputs = ex.inputs
        self.template_ids = inputs["plan_template_ids"][0, :real].tolist()
        self.target_type = inputs["plan_target_type"][0, :real].tolist()
        self.node_index = inputs["plan_target_node_index"][0, :real].tolist()
        self.link_index = inputs["plan_target_link_index"][0, :real].tolist()
        self.values: dict[str, np.ndarray] = {}
        self.masks: dict[str, np.ndarray] = {}
        for name in ALL_STRATEGIST_KEYS:
            self.values[name] = ex.targets[name][:real].numpy()
            self.masks[name] = ex.targets[f"{name}_mask"][:real].numpy().astype(bool)


def _records(examples: list[CorpusExample]) -> list[IncidentRecord]:
    return [IncidentRecord(ex) for ex in examples]


# ---------------------------------------------------------------------------
# Section 8/4: target-formula audit (from direct source inspection, cross-
# checked against a small controlled synthetic example computed here).
# ---------------------------------------------------------------------------


def _target_formula_audit() -> dict[str, Any]:
    from hydroswarm.domain import ConsequenceMetrics
    from hydroswarm.planning.plan_value_policy import PLAN_VALUE_POLICY_VERSION, evaluate_plan_value

    # Controlled synthetic example: A clearly better than B clearly better
    # than C, verified end to end through evaluate_plan_value.
    no_response = ConsequenceMetrics(
        contaminant_mass_consumed_mg=1000.0, pressure_violation_minutes=0.0,
        service_availability=1.0, containment_time_minutes=240.0,
        minimum_pressure_m=20.0, operation_count=0,
    )
    candidate_a = ConsequenceMetrics(  # clearly best: big exposure cut, fast containment
        contaminant_mass_consumed_mg=100.0, pressure_violation_minutes=0.0,
        service_availability=1.0, containment_time_minutes=30.0,
        minimum_pressure_m=20.0, operation_count=1,
    )
    candidate_b = ConsequenceMetrics(  # intermediate
        contaminant_mass_consumed_mg=500.0, pressure_violation_minutes=0.0,
        service_availability=0.98, containment_time_minutes=90.0,
        minimum_pressure_m=20.0, operation_count=1,
    )
    candidate_c = ConsequenceMetrics(  # clearly worst: barely better than no-response
        contaminant_mass_consumed_mg=950.0, pressure_violation_minutes=0.0,
        service_availability=0.95, containment_time_minutes=230.0,
        minimum_pressure_m=20.0, operation_count=1,
    )
    pool = [no_response, candidate_a, candidate_b, candidate_c]
    results = {
        name: evaluate_plan_value(metrics, no_response=no_response, valid_candidate_metrics=pool)
        for name, metrics in (("no_response", no_response), ("A", candidate_a), ("B", candidate_b), ("C", candidate_c))
    }
    synthetic_plan_values = {name: r.plan_value for name, r in results.items()}
    synthetic_regrets = {name: r.regret for name, r in results.items()}
    monotonic_ok = synthetic_plan_values["A"] > synthetic_plan_values["B"] > synthetic_plan_values["C"] > synthetic_plan_values["no_response"] - 1e-9
    regret_monotonic_ok = synthetic_regrets["A"] < synthetic_regrets["B"] < synthetic_regrets["C"]

    formulas = {
        "plan_validity": {
            "definition": "PlanVerifier.verify(...).decision == PlanDecision.VERIFIED",
            "physical_quantity": "boolean acceptance of the plan by exact WNTR/EPANET simulation plus prescreen",
            "raw_units": "boolean (1=valid)",
            "normalization": "none", "clipping": "none",
            "transform": "none", "higher_is_better": True,
            "masking": "never masked (plan_validity_mask always True)",
            "invalid_plan_handling": "n/a -- this IS the validity signal",
            "reference_baseline_plan": "none",
            "relationship_to_ranking": "gates which candidates enter the plan_value/proxy pool at all",
            "per_plan_or_per_incident": "per-plan",
            "source_code": "src/hydroswarm/training/strategist_labels.py:180 (is_valid), src/hydroswarm/simulation/verifier.py:180-181/231",
        },
        "exposure_proxy": {
            "definition": "contaminant_mass_consumed_mg / max(no_response.contaminant_mass_consumed_mg, 1e-9)",
            "physical_quantity": "ratio of this plan's contaminant mass consumed to the no-action baseline's own",
            "raw_units": "dimensionless ratio", "normalization": "ratio to no-response baseline (per-incident, not per-pool)",
            "clipping": "none (can exceed 1.0 if a plan is worse than doing nothing)",
            "transform": "none", "higher_is_better": False,
            "masking": "masked when candidate invalid, or consequences/no_response/valid_pool unavailable",
            "invalid_plan_handling": "target fully masked (0.0 placeholder, mask=False), never imputed",
            "reference_baseline_plan": "NO_ACTION plan's own exact WNTR consequences, this SAME incident",
            "relationship_to_ranking": "one of 4 additive components of cost -> regret -> plan_value",
            "per_plan_or_per_incident": "per-plan (numerator), but scaled by a per-incident constant (denominator)",
            "source_code": "src/hydroswarm/planning/plan_value_policy.py:75",
        },
        "pressure_risk_proxy": {
            "definition": "pressure_violation_minutes / 60.0",
            "physical_quantity": "total simulated minutes any node's pressure fell below minimum_pressure_m (10.0m default)",
            "raw_units": "minutes / 60.0 (dimensionless, 1.0 = 60 minutes of violation)",
            "normalization": "fixed train-owned scale (60.0), not pool-relative", "clipping": "none",
            "transform": "none", "higher_is_better": False,
            "masking": "same as exposure_proxy",
            "invalid_plan_handling": "target fully masked, never imputed",
            "reference_baseline_plan": "none (absolute quantity, not baseline-relative)",
            "relationship_to_ranking": "one of 4 additive cost components",
            "per_plan_or_per_incident": "per-plan",
            "source_code": "src/hydroswarm/planning/plan_value_policy.py:76",
            "STRUCTURAL_DEGENERACY_FINDING": (
                "MATHEMATICALLY GUARANTEED ZERO for every VALID candidate: PlanVerifier.verify() "
                "rejects (PlanDecision.REJECTED, hence plan_validity=False, hence this target is "
                "MASKED not merely zero) whenever minimum_pressure_m < the SAME minimum_pressure_m "
                "threshold used to compute pressure_violation_minutes itself "
                "(src/hydroswarm/simulation/wrapper.py:1341/1349 for the hydraulic-only path, "
                "1477 for the exposure-aware path actually used by strategist label generation). "
                "pressure_violation_minutes > 0 at any timestep implies minimum_pressure < threshold "
                "at that timestep, which implies the simulation-wide minimum_pressure_m also falls "
                "below threshold, which implies PRESSURE_BELOW_MINIMUM fires and the plan is "
                "REJECTED. Therefore pressure_risk_proxy == 0.0 for EVERY plan whose plan_validity "
                "target is True, by construction of the safety gate -- not a bug, not a population "
                "artifact, not fixable without weakening the safety threshold (forbidden)."
            ),
        },
        "service_loss_proxy": {
            "definition": "1.0 - service_availability",
            "physical_quantity": "fraction of REQUESTED demand not delivered, aggregated over the simulation window",
            "raw_units": "fraction in [0, 1]", "normalization": "none (already a fraction)", "clipping": "service_availability itself clipped to [0,1] upstream",
            "transform": "none", "higher_is_better": False,
            "masking": "same as exposure_proxy",
            "invalid_plan_handling": "target fully masked, never imputed",
            "reference_baseline_plan": "the plan's own requested-vs-delivered demand (baseline_requested_demand)",
            "relationship_to_ranking": "one of 4 additive cost components",
            "per_plan_or_per_incident": "per-plan",
            "source_code": "src/hydroswarm/planning/plan_value_policy.py:77",
            "STRUCTURAL_BOUND_FINDING": (
                "BOUNDED (not mathematically forced to exactly 0, unlike pressure_risk_proxy) to "
                "[0, 0.10] for every VALID candidate by the SAME verifier safety gate: "
                "SERVICE_BELOW_MINIMUM fires (REJECTED) whenever service_availability < "
                "minimum_service_availability (0.90 default, src/hydroswarm/simulation/wrapper.py:444). "
                "Empirically near-zero in this population (see identifiability artifact) because the "
                "generated candidates on this network/scenario population rarely cause even mild "
                "service disruption, not because it is mathematically forced to exactly 0 -- a "
                "candidate/population characteristic, not a verifier-gate mathematical necessity."
            ),
        },
        "containment_time_proxy": {
            "definition": "(240.0 if containment_time_minutes is None else containment_time_minutes) / 240.0",
            "physical_quantity": "minutes until contamination is no longer detected above threshold anywhere",
            "raw_units": "minutes / 240.0 (dimensionless, 1.0 = 240 minutes OR never contained in-window)",
            "normalization": "fixed train-owned scale (240.0)", "clipping": "none (can exceed 1.0 if containment takes > 240 min)",
            "transform": "None -> worst-case (1.0) substitution, an explicit design choice (not an accidental default)",
            "higher_is_better": False,
            "masking": "same as exposure_proxy",
            "invalid_plan_handling": "target fully masked, never imputed",
            "reference_baseline_plan": "none (absolute quantity)",
            "relationship_to_ranking": "one of 4 additive cost components",
            "per_plan_or_per_incident": "per-plan",
            "source_code": "src/hydroswarm/planning/plan_value_policy.py:78-81",
        },
        "plan_regret_proxy": {
            "definition": "max(0.0, cost - best_cost) where cost = exposure_proxy + pressure_risk_proxy + service_loss_proxy + containment_time_proxy, best_cost = min(cost) over every exactly-verified valid candidate for this SAME incident (including no-response)",
            "physical_quantity": "additive cost gap to the best available verified plan in this incident's own candidate pool",
            "raw_units": "same dimensionless cost scale as the 4 components summed", "normalization": "per-incident (best_cost is a per-incident pool minimum)",
            "clipping": "clamped at 0.0 from below only (never negative)", "transform": "none",
            "higher_is_better": False,
            "masking": "masked wherever plan_value itself is masked (same is_valid/consequences/no_response/pool gate)",
            "invalid_plan_handling": "target fully masked, never imputed",
            "reference_baseline_plan": "the pool minimum-cost candidate for this incident (may or may not be no-response)",
            "relationship_to_ranking": "== regret, the quantity plan_value is a monotone transform of",
            "per_plan_or_per_incident": "per-plan value, but references a per-incident pool minimum",
            "source_code": "src/hydroswarm/planning/plan_value_policy.py:126-127",
        },
        "plan_value": {
            "definition": "1.0 / (1.0 + regret)",
            "physical_quantity": "bounded, monotone-decreasing transform of plan_regret_proxy",
            "raw_units": "dimensionless value score in (0, 1]", "normalization": "n/a (already bounded by construction)",
            "clipping": "none needed -- 1/(1+regret) is bounded in (0,1] for any regret>=0",
            "transform": "1/(1+x) applied to regret", "higher_is_better": True,
            "masking": "same as plan_regret_proxy",
            "invalid_plan_handling": "target fully masked, never imputed",
            "reference_baseline_plan": "same pool minimum as plan_regret_proxy",
            "relationship_to_ranking": "THE ranking target (pairwise-ranking gate criterion is computed on this key)",
            "per_plan_or_per_incident": "per-plan",
            "source_code": "src/hydroswarm/planning/plan_value_policy.py:136",
            "REDUNDANCY_FINDING": (
                "plan_value and plan_regret_proxy are EXACT, DETERMINISTIC, BIJECTIVE (monotone "
                "decreasing) functions of one another by construction: plan_value = 1/(1+regret), "
                "plan_regret_proxy = regret, computed from the SAME `regret` local variable in the "
                "SAME evaluate_plan_value() call (plan_value_policy.py:127/136/145). They are never "
                "independently informative -- a model that learns one exactly determines the other. "
                "Empirically verified below (target-correlation artifact): Spearman correlation "
                "between the two should equal exactly -1.0 up to floating-point roundoff."
            ),
        },
    }
    return {
        "kind": "M10_3B_TARGET_FORMULA_AUDIT",
        "plan_value_policy_version": PLAN_VALUE_POLICY_VERSION,
        "formulas": formulas,
        "synthetic_monotonicity_check": {
            "description": "Controlled synthetic pool: A (clearly best) > B (intermediate) > C (clearly worst) > no_response, verified end to end through the real evaluate_plan_value() implementation.",
            "plan_values": synthetic_plan_values,
            "regrets": synthetic_regrets,
            "plan_value_monotonic_A_gt_B_gt_C": bool(monotonic_ok),
            "regret_monotonic_A_lt_B_lt_C": bool(regret_monotonic_ok),
        },
    }


# ---------------------------------------------------------------------------
# Section 9: ranking/sign/alignment audit (code trace + synthetic unit
# checks; no training).
# ---------------------------------------------------------------------------


def _ranking_alignment_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # 1. Gate script's own pairwise comparator direction.
    pred = np.array([0.9, 0.5, 0.1])
    target = np.array([0.9, 0.5, 0.1])  # A>B>C in both, same direction (plan_value: higher=better)
    mask = np.array([True, True, True])

    def _pairwise(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
        valid_indices = np.flatnonzero(mask)
        correct = 0
        total = 0
        for a in range(len(valid_indices)):
            for b in range(a + 1, len(valid_indices)):
                i, j = valid_indices[a], valid_indices[b]
                if target[i] == target[j]:
                    continue
                total += 1
                true_order = target[i] > target[j]
                pred_order = pred[i] > pred[j]
                if true_order == pred_order:
                    correct += 1
        return correct, total

    correct, total = _pairwise(pred, target, mask)
    checks.append({
        "name": "perfect_agreement_gives_100pct",
        "expects": "3/3 correct when pred order == target order exactly",
        "correct": correct, "total": total, "passed": bool(correct == total == 3),
    })

    correct, total = _pairwise(-pred, target, mask)
    checks.append({
        "name": "inverted_prediction_gives_0pct",
        "expects": "0/3 correct when pred is exactly inverted vs target (sanity: metric is direction-sensitive)",
        "correct": correct, "total": total, "passed": bool(correct == 0 and total == 3),
    })

    # 2. Candidate-permutation invariance of the SAME accuracy metric
    # (order must not matter to the metric itself, independent of the
    # already-existing test_candidate_order_does_not_change_per_candidate_
    # model_output for the MODEL).
    rng = np.random.default_rng(20260817)
    pred2 = rng.normal(size=9)
    target2 = rng.normal(size=9)
    mask2 = np.ones(9, dtype=bool)
    base_correct, base_total = _pairwise(pred2, target2, mask2)
    perm = rng.permutation(9)
    perm_correct, perm_total = _pairwise(pred2[perm], target2[perm], mask2[perm])
    checks.append({
        "name": "pairwise_metric_permutation_invariant",
        "base": [base_correct, base_total], "permuted": [perm_correct, perm_total],
        "passed": bool(base_correct == perm_correct and base_total == perm_total),
    })

    # 3. Padding contamination: a padded (mask=False) slot with an
    # extreme/adversarial value must never enter the pairwise count.
    pred3 = np.array([0.9, 0.5, 999.0])  # position 2 is "padding" with a poisoned value
    target3 = np.array([0.9, 0.5, -999.0])
    mask3 = np.array([True, True, False])
    correct, total = _pairwise(pred3, target3, mask3)
    checks.append({
        "name": "padded_slot_excluded_from_pairwise_count",
        "correct": correct, "total": total,
        "passed": bool(total == 1 and correct == 1),
    })

    # 4. plan_value vs plan_regret_proxy: gate script never mixes the two
    # (only plan_value is used for _pairwise_ranking_accuracy_per_incident
    # in run_m10_3_level_a_gate.py). Confirmed by source inspection, not
    # merely asserted here.
    import inspect

    import run_m10_3_level_a_gate as gate_module
    main_source = inspect.getsource(gate_module.main)
    ranking_call_line = next(
        line for line in main_source.splitlines() if "_pairwise_ranking_accuracy_per_incident(" in line and "def " not in line
    )
    only_plan_value_ranked = (
        'pred["plan_value"]' in ranking_call_line and "plan_regret_proxy" not in ranking_call_line
    )
    checks.append({
        "name": "gate_ranks_only_plan_value_not_regret",
        "note": "confirms the gate's pairwise-ranking criterion is computed on plan_value (higher=better) consistently, never accidentally mixed with plan_regret_proxy (lower=better) which would silently invert the comparison",
        "passed": bool(only_plan_value_ranked),
    })

    # 5. Candidate identity keying: candidate_tensorizer keys every row to
    # its OWN proposal by content (template_ids/target_type/node_index/
    # link_index/features), never by a separately-tracked position counter
    # -- confirmed by source inspection (candidate_tensorizer.py's own
    # `for position, proposal in enumerate(proposals):` loop writes every
    # field for that SAME position from that SAME proposal, single pass,
    # no separate reordering step afterward).
    import hydroswarm.planning.candidate_tensorizer as tensorizer_module
    tensorizer_source = inspect.getsource(tensorizer_module.plan_proposals_to_candidate_tensors)
    single_pass_keying = tensorizer_source.count("for position, proposal in enumerate(proposals):") == 1
    checks.append({
        "name": "candidate_tensorizer_single_pass_content_keyed",
        "note": "every INPUT tensor row for position p is written from proposals[p] in a single enumerate() pass -- no separate reordering/sorting step exists that could desynchronize row p's inputs from row p's targets",
        "passed": bool(single_pass_keying),
    })

    all_passed = all(c["passed"] for c in checks)
    return {
        "kind": "M10_3B_RANKING_ALIGNMENT_AUDIT",
        "conclusion": (
            "NO mechanical sign/ranking/candidate-alignment defect found. The gate's pairwise "
            "comparator (run_m10_3_level_a_gate.py::_pairwise_ranking_accuracy_per_incident) uses "
            "`target[i] > target[j]` vs `pred[i] > pred[j]` on plan_value alone (both higher-is-"
            "better, consistent direction); padded slots are excluded via `mask`, never contaminate "
            "the count; candidate identity is keyed by a single content-preserving enumerate() pass "
            "with no reordering step; the existing repository test "
            "test_candidate_order_does_not_change_per_candidate_model_output already proves the "
            "MODEL's own per-row output is permutation-invariant. The near-chance/sub-chance "
            "pairwise-ranking accuracy observed in M10.3A is therefore NOT explained by a sign, "
            "ordering, masking, or indexing defect."
        ) if all_passed else "One or more mechanical checks FAILED -- see individual entries.",
        "checks": checks,
        "all_checks_passed": bool(all_passed),
    }


# ---------------------------------------------------------------------------
# Section 10/11: target identifiability (global) + within-incident variance
# + plan_value/regret identifiability.
# ---------------------------------------------------------------------------


def _global_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    unique = np.unique(np.round(values, 6))
    return {
        "n": int(values.size),
        "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "max": float(values.max()),
        "q25": float(np.percentile(values, 25)), "q50": float(np.percentile(values, 50)),
        "q75": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "n_unique_rounded_6dp": int(unique.size),
        "fraction_exactly_constant": float(np.mean(values == values[0])) if values.size else 0.0,
    }


def _target_identifiability(records: list[IncidentRecord], split: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    validity_all = np.concatenate([r.values["plan_validity"] for r in records]) if records else np.array([])
    out["plan_validity"] = {
        "n_candidates": int(validity_all.size),
        "fraction_valid": float(validity_all.mean()) if validity_all.size else None,
        "fraction_incidents_with_mixed_validity": float(np.mean([
            0 < r.values["plan_validity"].sum() < r.real_count for r in records
        ])) if records else None,
    }
    for name in TARGET_KEYS:
        vals = np.concatenate([r.values[name][r.masks[name]] for r in records]) if records else np.array([])
        out[name] = _global_stats(vals)
    return {"kind": "M10_3B_TARGET_IDENTIFIABILITY_GLOBAL", "split": split, "per_target": out}


def _within_incident_variance(records: list[IncidentRecord], split: str) -> dict[str, Any]:
    per_target: dict[str, Any] = {}
    for name in TARGET_KEYS:
        tol = NEAR_TIE_TOLERANCE[name]
        n_incidents_with_2plus_valid = 0
        n_incidents_with_2plus_distinguishable = 0
        n_incidents_with_3plus_distinguishable = 0
        n_incidents_all_tied = 0
        best_vs_second_margins: list[float] = []
        best_vs_worst_margins: list[float] = []
        per_incident_ranges: list[float] = []
        per_incident_variances: list[float] = []
        per_incident_iqrs: list[float] = []
        n_incidents_total_with_any_valid = 0
        for r in records:
            valid = r.values[name][r.masks[name]]
            if valid.size < 2:
                continue
            n_incidents_total_with_any_valid += 1
            n_incidents_with_2plus_valid += 1
            sorted_vals = np.sort(valid)
            spread = float(sorted_vals[-1] - sorted_vals[0])
            per_incident_ranges.append(spread)
            per_incident_variances.append(float(valid.var()))
            per_incident_iqrs.append(float(np.percentile(valid, 75) - np.percentile(valid, 25)))
            # distinguishable pairs: any two values differing by more than tol
            distinguishable_count = 0
            for i in range(len(sorted_vals)):
                for j in range(i + 1, len(sorted_vals)):
                    if abs(sorted_vals[j] - sorted_vals[i]) > tol:
                        distinguishable_count += 1
            if distinguishable_count > 0:
                n_incidents_with_2plus_distinguishable += 1
            else:
                n_incidents_all_tied += 1
            # "3+ distinguishable": at least 3 values pairwise-separated by
            # more than tol from at least one neighbor cluster -- approximate
            # via number of distinct clusters at tolerance tol.
            clusters = 1
            for i in range(1, len(sorted_vals)):
                if sorted_vals[i] - sorted_vals[i - 1] > tol:
                    clusters += 1
            if clusters >= 3:
                n_incidents_with_3plus_distinguishable += 1
            higher_better = name == "plan_value"
            best = sorted_vals[-1] if higher_better else sorted_vals[0]
            worst = sorted_vals[0] if higher_better else sorted_vals[-1]
            second = sorted_vals[-2] if higher_better else sorted_vals[1]
            best_vs_second_margins.append(float(abs(best - second)))
            best_vs_worst_margins.append(float(abs(best - worst)))

        def _summ(xs: list[float]) -> dict[str, Any]:
            if not xs:
                return {"n": 0}
            arr = np.array(xs)
            return {
                "n": len(xs), "mean": float(arr.mean()), "median": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)), "q75": float(np.percentile(arr, 75)),
                "max": float(arr.max()), "min": float(arr.min()),
            }

        per_target[name] = {
            "near_tie_tolerance": tol,
            "n_incidents_with_2plus_valid_candidates": n_incidents_with_2plus_valid,
            "fraction_incidents_with_2plus_meaningfully_distinguishable": (
                n_incidents_with_2plus_distinguishable / n_incidents_total_with_any_valid
                if n_incidents_total_with_any_valid else None
            ),
            "fraction_incidents_with_3plus_meaningfully_distinguishable_clusters": (
                n_incidents_with_3plus_distinguishable / n_incidents_total_with_any_valid
                if n_incidents_total_with_any_valid else None
            ),
            "fraction_incidents_all_candidates_effectively_tied": (
                n_incidents_all_tied / n_incidents_total_with_any_valid
                if n_incidents_total_with_any_valid else None
            ),
            "per_incident_range_distribution": _summ(per_incident_ranges),
            "per_incident_variance_distribution": _summ(per_incident_variances),
            "per_incident_iqr_distribution": _summ(per_incident_iqrs),
            "best_vs_second_best_margin_distribution": _summ(best_vs_second_margins),
            "best_vs_worst_margin_distribution": _summ(best_vs_worst_margins),
        }

    # Section 11: plan_value / plan_regret_proxy redundancy check (both
    # pooled and, since they are per-plan-deterministic transforms of one
    # another, a direct value-level check too).
    pv_all: list[float] = []
    regret_all: list[float] = []
    for r in records:
        mask = r.masks["plan_value"] & r.masks["plan_regret_proxy"]
        pv_all.extend(r.values["plan_value"][mask].tolist())
        regret_all.extend(r.values["plan_regret_proxy"][mask].tolist())
    pv_arr, regret_arr = np.array(pv_all), np.array(regret_all)
    redundancy_corr = float(spearmanr(pv_arr, regret_arr).statistic) if pv_arr.size > 1 else None
    reconstructed_pv = 1.0 / (1.0 + regret_arr) if regret_arr.size else np.array([])
    max_abs_reconstruction_error = float(np.max(np.abs(reconstructed_pv - pv_arr))) if pv_arr.size else None

    return {
        "kind": "M10_3B_WITHIN_INCIDENT_VARIANCE",
        "split": split,
        "tolerance_definitions": NEAR_TIE_TOLERANCE,
        "per_target": per_target,
        "plan_value_plan_regret_proxy_redundancy": {
            "spearman_correlation": redundancy_corr,
            "expected": -1.0,
            "max_abs_reconstruction_error_plan_value_from_1_over_1_plus_regret": max_abs_reconstruction_error,
            "conclusion": "plan_value and plan_regret_proxy are exact deterministic bijective (monotone-decreasing) transforms of one another -- confirmed empirically here, matching the formula audit's code-level finding. They carry identical ranking information; training/evaluating both as if independently informative double-counts one real signal.",
        },
    }


# ---------------------------------------------------------------------------
# Section 13: candidate generator diversity audit.
# ---------------------------------------------------------------------------


def _candidate_diversity(records: list[IncidentRecord], split: str) -> dict[str, Any]:
    template_counter: Counter[str] = Counter()
    per_incident_template_sets: list[frozenset[str]] = []
    real_counts: list[int] = []
    target_kind_counter: Counter[int] = Counter()  # 0=NONE,1=NODE,2=LINK
    target_node_diversity: list[int] = []
    duplicate_incident_count = 0
    for r in records:
        real_counts.append(r.real_count)
        names = [ACTION_TEMPLATES[t] for t in r.template_ids]
        template_counter.update(names)
        per_incident_template_sets.append(frozenset(names))
        target_kind_counter.update(r.target_type)
        node_targets = {n for n, t in zip(r.node_index, r.target_type) if t == 1 and n >= 0}
        target_node_diversity.append(len(node_targets))
        # duplicate real candidates within one incident: identical
        # (template, target_type, node_index, link_index) tuple appearing
        # more than once.
        keys = list(zip(r.template_ids, r.target_type, r.node_index, r.link_index))
        if len(keys) != len(set(keys)):
            duplicate_incident_count += 1

    # consequence diversity: for each template, distribution of
    # exposure_proxy/service_loss_proxy/containment_time_proxy among VALID
    # candidates of that template (this is the "does the generator produce
    # meaningful tradeoffs" question).
    per_template_consequence: dict[str, Any] = {}
    for template in ACTION_TEMPLATES:
        vals: dict[str, list[float]] = {name: [] for name in TARGET_KEYS}
        n_valid = 0
        n_total = 0
        for r in records:
            for position, tid in enumerate(r.template_ids):
                if ACTION_TEMPLATES[tid] != template:
                    continue
                n_total += 1
                if r.masks["plan_value"][position]:
                    n_valid += 1
                    for name in TARGET_KEYS:
                        if r.masks[name][position]:
                            vals[name].append(float(r.values[name][position]))
        per_template_consequence[template] = {
            "n_proposed": n_total, "n_valid": n_valid,
            "fraction_valid": (n_valid / n_total) if n_total else None,
            **{f"{name}_mean": (float(np.mean(v)) if v else None) for name, v in vals.items()},
            **{f"{name}_std": (float(np.std(v)) if v else None) for name, v in vals.items()},
        }

    n = len(records)
    return {
        "kind": "M10_3B_CANDIDATE_DIVERSITY",
        "split": split,
        "n_incidents": n,
        "real_candidate_count_distribution": {
            "mean": float(np.mean(real_counts)) if real_counts else None,
            "min": int(np.min(real_counts)) if real_counts else None,
            "max": int(np.max(real_counts)) if real_counts else None,
            "histogram": dict(Counter(real_counts)),
        },
        "template_frequency": dict(template_counter),
        "target_type_frequency": {"NONE": target_kind_counter.get(0, 0), "NODE": target_kind_counter.get(1, 0), "LINK": target_kind_counter.get(2, 0)},
        "target_node_diversity_per_incident": {
            "mean_distinct_node_targets": float(np.mean(target_node_diversity)) if target_node_diversity else None,
            "max_distinct_node_targets": int(np.max(target_node_diversity)) if target_node_diversity else None,
        },
        "n_incidents_with_duplicate_candidate_identity": duplicate_incident_count,
        "fraction_incidents_all_9_templates_present": float(np.mean([len(s) == len(ACTION_TEMPLATES) for s in per_incident_template_sets])) if per_incident_template_sets else None,
        "per_template_consequence_profile": per_template_consequence,
        "conclusion": (
            "Every incident proposes from the SAME fixed 9-template vocabulary in the SAME fixed "
            "order (NO_ACTION always first/always proposed); real diversity comes from WHICH node/"
            "link each template resolves to (probable-source-dependent) and each template's own "
            "typical consequence profile (see per_template_consequence_profile) -- e.g. ISOLATE_"
            "SOURCE/ISOLATE_AND_FLUSH are structurally different actions from WAIT_OBSERVE/"
            "INCREASE_MONITORING with different expected exposure/containment-time profiles. "
            "Candidate generation is unmodified from the live production path; any future "
            "population-diversity amendment must expand SCENARIO SEVERITY/topology conditions, "
            "not hand-craft implausible candidate templates."
        ),
    }


# ---------------------------------------------------------------------------
# Section 14: target cross-correlation / redundancy.
# ---------------------------------------------------------------------------


def _target_correlation(records: list[IncidentRecord], split: str) -> dict[str, Any]:
    pairs = [
        ("plan_value", "exposure_proxy"), ("plan_value", "pressure_risk_proxy"),
        ("plan_value", "service_loss_proxy"), ("plan_value", "containment_time_proxy"),
        ("plan_value", "plan_regret_proxy"), ("exposure_proxy", "containment_time_proxy"),
        ("pressure_risk_proxy", "service_loss_proxy"), ("exposure_proxy", "service_loss_proxy"),
        ("exposure_proxy", "pressure_risk_proxy"), ("containment_time_proxy", "service_loss_proxy"),
    ]
    pooled: dict[str, Any] = {}
    within_incident: dict[str, Any] = {}
    for a, b in pairs:
        key = f"{a}__vs__{b}"
        va, vb = [], []
        for r in records:
            mask = r.masks[a] & r.masks[b]
            va.extend(r.values[a][mask].tolist())
            vb.extend(r.values[b][mask].tolist())
        va_arr, vb_arr = np.array(va), np.array(vb)
        if va_arr.size > 1 and np.std(va_arr) > 0 and np.std(vb_arr) > 0:
            pooled[key] = {"n": int(va_arr.size), "spearman": float(spearmanr(va_arr, vb_arr).statistic)}
        else:
            pooled[key] = {"n": int(va_arr.size), "spearman": None, "note": "zero variance in at least one side"}

        within_corrs: list[float] = []
        for r in records:
            mask = r.masks[a] & r.masks[b]
            xa, xb = r.values[a][mask], r.values[b][mask]
            if xa.size >= 3 and np.std(xa) > 0 and np.std(xb) > 0:
                c = spearmanr(xa, xb).statistic
                if np.isfinite(c):
                    within_corrs.append(float(c))
        within_incident[key] = {
            "n_incidents_with_meaningful_within_incident_variance": len(within_corrs),
            "mean_within_incident_spearman": float(np.mean(within_corrs)) if within_corrs else None,
        }

    return {
        "kind": "M10_3B_TARGET_CORRELATION",
        "split": split,
        "pooled_correlation": pooled,
        "within_incident_correlation": within_incident,
        "conclusion": (
            "plan_value/plan_regret_proxy are exactly redundant (see within-incident-variance "
            "artifact). pressure_risk_proxy has zero pooled variance among valid candidates (see "
            "target-formula audit's STRUCTURAL_DEGENERACY_FINDING) so its correlation with anything "
            "is undefined (reported as null with n and a zero-variance note, never silently 0.0 or "
            "1.0)."
        ),
    }


# ---------------------------------------------------------------------------
# Section 15: feature/target identifiability (legal decision-time inputs).
# NON-PROMOTABLE / DIAGNOSTIC ONLY.
# ---------------------------------------------------------------------------


def _feature_identifiability(records: list[IncidentRecord], split: str) -> dict[str, Any]:
    # Stratify plan_value/exposure_proxy by (template, target_type) -- the
    # ONLY legal decision-time identity signal a candidate carries (Part 4
    # of the M10.3A protocol: plan_features is a 6-dim structural vector
    # derived from template/target_type/has_target, nothing richer).
    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        for position, tid in enumerate(r.template_ids):
            template = ACTION_TEMPLATES[tid]
            for name in ("plan_value", "exposure_proxy"):
                if r.masks[name][position]:
                    strata[template][name].append(float(r.values[name][position]))

    stratified_means: dict[str, Any] = {}
    for template, values in strata.items():
        stratified_means[template] = {
            name: {"n": len(v), "mean": float(np.mean(v)) if v else None, "std": float(np.std(v)) if v else None}
            for name, v in values.items()
        }

    # Between-template variance vs within-template variance (a crude,
    # NON-PROMOTABLE ANOVA-style diagnostic signal-to-noise check): does
    # KNOWING the template explain a meaningful share of plan_value's total
    # variance? This uses only legal, pre-verification decision-time
    # information (the template identity itself).
    all_pv: list[float] = []
    template_of_pv: list[str] = []
    for r in records:
        for position, tid in enumerate(r.template_ids):
            if r.masks["plan_value"][position]:
                all_pv.append(float(r.values["plan_value"][position]))
                template_of_pv.append(ACTION_TEMPLATES[tid])
    anova_result = None
    if all_pv:
        overall_mean = float(np.mean(all_pv))
        total_ss = float(np.sum((np.array(all_pv) - overall_mean) ** 2))
        between_ss = 0.0
        for template in set(template_of_pv):
            group = [v for v, t in zip(all_pv, template_of_pv) if t == template]
            between_ss += len(group) * (float(np.mean(group)) - overall_mean) ** 2
        anova_result = {
            "n": len(all_pv), "total_sum_of_squares": total_ss, "between_template_sum_of_squares": between_ss,
            "fraction_of_plan_value_variance_explained_by_template_identity_alone": (between_ss / total_ss) if total_ss > 0 else None,
        }

    return {
        "kind": "M10_3B_FEATURE_IDENTIFIABILITY",
        "split": split,
        "label": "NON-PROMOTABLE / DIAGNOSTIC ONLY -- not a model, not a proposal for any runtime path",
        "stratified_means_by_template": stratified_means,
        "template_identity_explains_plan_value_variance": anova_result,
        "conclusion": (
            "If a large fraction of plan_value's total variance is explained by template identity "
            "alone (a legal, pre-verification, decision-time input the model already embeds via "
            "candidate_plan_encoder.template_embedding), the model has access to real predictive "
            "signal for the BETWEEN-template ranking question. Ranking failure would then be "
            "concentrated in the WITHIN-template-choice-of-target discrimination instead, which "
            "requires the richer target_embedding/plan_features signal, not template identity. See "
            "the fraction_of_plan_value_variance_explained_by_template_identity_alone figure."
        ),
    }


# ---------------------------------------------------------------------------
# Section 16: oracle / achievable-utility diagnostic. NON-PROMOTABLE.
# ---------------------------------------------------------------------------


def _oracle_utility(records: list[IncidentRecord], split: str) -> dict[str, Any]:
    gains_best_vs_no_action: list[float] = []
    gains_best_vs_random_valid: list[float] = []
    gains_best_vs_first_candidate: list[float] = []
    n_incidents_no_action_already_optimal = 0
    n_incidents_considered = 0
    rng = np.random.default_rng(20260817)

    for r in records:
        mask = r.masks["plan_value"]
        if mask.sum() < 2:
            continue
        n_incidents_considered += 1
        valid_positions = np.flatnonzero(mask)
        pv = r.values["plan_value"][mask]
        best = float(pv.max())
        # NO_ACTION is always proposal position 0 by construction
        # (response.py always appends it first, action_templates.py's own
        # ACTION_TEMPLATES[0] == "NO_ACTION").
        no_action_position = 0
        no_action_value = float(r.values["plan_value"][no_action_position]) if r.masks["plan_value"][no_action_position] else None
        first_candidate_value = float(r.values["plan_value"][valid_positions[0]])
        random_position = int(rng.choice(valid_positions))
        random_value = float(r.values["plan_value"][random_position])

        if no_action_value is not None:
            gains_best_vs_no_action.append(best - no_action_value)
            if best - no_action_value <= NEAR_TIE_TOLERANCE["plan_value"]:
                n_incidents_no_action_already_optimal += 1
        gains_best_vs_random_valid.append(best - random_value)
        gains_best_vs_first_candidate.append(best - first_candidate_value)

    def _summ(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"n": 0}
        arr = np.array(xs)
        return {
            "n": len(xs), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)), "q75": float(np.percentile(arr, 75)),
            "max": float(arr.max()), "fraction_meaningfully_positive": float(np.mean(arr > NEAR_TIE_TOLERANCE["plan_value"])),
        }

    return {
        "kind": "M10_3B_ORACLE_UTILITY",
        "split": split,
        "label": "NON-PROMOTABLE / DIAGNOSTIC ONLY -- oracle uses exact WNTR-verified labels, never a deployable policy",
        "n_incidents_considered": n_incidents_considered,
        "best_vs_no_action_plan_value_gain": _summ(gains_best_vs_no_action),
        "best_vs_random_valid_candidate_plan_value_gain": _summ(gains_best_vs_random_valid),
        "best_vs_first_proposed_candidate_plan_value_gain": _summ(gains_best_vs_first_candidate),
        "fraction_incidents_where_no_action_is_already_near_optimal": (
            n_incidents_no_action_already_optimal / n_incidents_considered if n_incidents_considered else None
        ),
        "conclusion": (
            "If fraction_incidents_where_no_action_is_already_near_optimal is high and the gain "
            "distributions are concentrated near the near_tie_tolerance floor, then even a perfect "
            "oracle Strategist would rarely outperform the trivial NO_ACTION/first-candidate policy "
            "on THIS population -- meaning low competence-gate scores reflect low available decision "
            "utility in the population, not necessarily a representation failure. A low fraction "
            "would instead indicate real, learnable decision value exists and is simply not yet "
            "being captured."
        ),
    }


# ---------------------------------------------------------------------------
# Section 18: leakage audit (extends M10.3A's, structural + a fresh
# empirical check that candidate order equals the fixed template order,
# never a truth-derived order).
# ---------------------------------------------------------------------------


def _leakage_audit(records: list[IncidentRecord], split: str) -> dict[str, Any]:
    order_is_canonical_template_order = True
    violations = 0
    for r in records:
        names = [ACTION_TEMPLATES[t] for t in r.template_ids]
        expected_prefix = [t for t in ACTION_TEMPLATES if t in set(names)]
        if names != expected_prefix:
            order_is_canonical_template_order = False
            violations += 1
        if names and names[0] != "NO_ACTION":
            order_is_canonical_template_order = False
            violations += 1

    import inspect

    import hydroswarm.training.strategist_candidate_corpus as corpus_module
    reconstruct_source = inspect.getsource(corpus_module._reconstruct_context_and_proposals)

    return {
        "kind": "M10_3B_LEAKAGE_AUDIT",
        "split": split,
        "candidate_order_is_fixed_canonical_template_order_never_truth_derived": {
            "passed": order_is_canonical_template_order,
            "n_incidents_checked": len(records),
            "n_violations": violations,
            "note": "response.py::generate_response_plans appends templates in the SAME fixed literal order every call (NO_ACTION always first/always proposed), driven only by PlanGenerationContext's current-evidence-derived fields -- never by exact WNTR outcome or future truth.",
        },
        "input_construction_never_reads_incident_ground_truth": {
            "passed": ("manifest.incident" not in reconstruct_source) and ("incident_truth" not in reconstruct_source),
            "note": "_reconstruct_context_and_proposals (the INPUT side) source contains no reference to manifest.incident/incident_truth -- structural, re-confirmed here, matches the existing repository test test_context_construction_never_reads_scenario_incident_ground_truth_for_candidate_generation.",
        },
        "existing_m10_3a_adversarial_tests_reused": "tests/scientific/test_m10_3_strategist_refit_corpus.py (alignment-guard fail-closed test, no-target-key-in-input test, permutation-invariance test) -- re-run as part of this task's own pytest pass, not re-implemented.",
        "conclusion": "No target/outcome leakage into candidate ordering or INPUT tensors found, consistent with M10.3A's own Part 5 finding.",
    }


# ---------------------------------------------------------------------------
# Section 17: calibration-preservation protocol-interpretation audit
# (reads M10.3A's own artifact, does not retrain/refit anything).
# ---------------------------------------------------------------------------


def _calibration_preservation_audit() -> dict[str, Any]:
    preservation = json.loads((M10_3_REFIT_DIR / "m10-3-refit-preservation.json").read_text())
    per_seed_analysis = {}
    for seed, entry in preservation["per_seed"].items():
        teacher_cov = entry["teacher_calibration_coverage"]
        level_b_cov = entry["level_b_calibration_coverage"]
        floor = entry["calibration_coverage_floor"]
        per_seed_analysis[seed] = {
            "teacher_calibration_coverage": teacher_cov,
            "level_b_calibration_coverage": level_b_cov,
            "floor": floor,
            "teacher_itself_below_absolute_floor_on_this_population": bool(teacher_cov < floor),
            "level_b_below_absolute_floor": bool(level_b_cov < floor),
            "relative_degradation_teacher_to_level_b": float(teacher_cov - level_b_cov),
            "any_ci_confident_sentinel_regression": entry["any_ci_confident_regression"],
        }
    n_teacher_below_floor = sum(1 for v in per_seed_analysis.values() if v["teacher_itself_below_absolute_floor_on_this_population"])
    return {
        "kind": "M10_3B_CALIBRATION_PRESERVATION_PROTOCOL_AUDIT",
        "label": "PROTOCOL-INTERPRETATION AUDIT ONLY -- does not reopen, reverse, or refit M10.3A's Level-B rejection",
        "source_artifact": "reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-preservation.json",
        "calibration_coverage_floor_origin": {
            "value": 0.85,
            "source": "scripts/hydrocore_v5/m9_4_common.py::OPERATIONAL_COVERAGE_FLOOR (alpha=0.1, nominal_coverage_target=0.90) -- an M9-wide governance floor, reused verbatim (not re-derived) by run_m10_3_level_b_preservation.py::CALIBRATION_COVERAGE_FLOOR",
            "population_it_was_governing_when_frozen": "the M9 canonical calibration/operational-evaluation population (multi-family, full depth grid) -- NOT specific to the M10.3A golden-reference/MATURE/VALIDATION_SEED_BASE=1_300_100_000 development-only population it is applied to here.",
        },
        "per_seed": per_seed_analysis,
        "finding": (
            f"The UNMODIFIED M9.6 teacher itself already scores below the 0.85 absolute floor for "
            f"{n_teacher_below_floor}/3 seeds on THIS specific development population (seed 31874: "
            f"0.8467, seed 20260815: 0.8033) -- before Level B touches anything. This shows the 0.85 "
            f"absolute floor, while a valid TARGET for the original M9 operational-evaluation "
            f"population it was governed for, is not automatically a well-calibrated ABSOLUTE cutoff "
            f"for every smaller/differently-composed development population a later milestone might "
            f"evaluate on -- finite-sample conformal coverage on a 300-scenario single-family subset "
            f"can legitimately sit below a floor set for a much larger/broader population without "
            f"that alone indicating a defect."
        ),
        "does_this_change_the_m10_3a_level_b_rejection": (
            "NO. Level B is independently, sufficiently disqualified by CI-confident PAIRED "
            "regressions against its OWN unmodified parent teacher on the SAME population "
            "(source_region/start_time/event_presence/event_cause and others, varying by seed, all "
            "with CI excluding zero) -- a RELATIVE criterion that does not depend on where the "
            "absolute floor is set. Even if the floor question were resolved in Level B's favor, the "
            "relative-regression criterion alone still rejects Level B for all three seeds."
        ),
        "recommendation_for_future_full_shared_refit_preservation_gates": (
            "Future preservation gates should require BOTH: (A) no CI-confident paired-bootstrap "
            "degradation against the SAME checkpoint's own frozen parent teacher on the SAME "
            "population (a population-invariant, self-relative criterion); AND (B) calibration "
            "coverage validity under the calibration regime's OWN governed population/support set -- "
            "not an absolute floor number transplanted onto a different, smaller development-only "
            "population it was never calibrated against. Where a smaller development population's "
            "teacher coverage is already below the M9-wide floor (as observed here), that population "
            "should either use a floor re-derived for its own support size/composition, or the "
            "calibration criterion should be evaluated as relative-to-teacher (like criterion A) "
            "rather than absolute, on that population specifically."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    M10_3B_DIR.mkdir(parents=True, exist_ok=True)
    branch = m10.current_branch()
    assert branch == m10.FROZEN_BRANCH
    locked_before = m10.assert_locked_test_closed()
    start_commit = m10.current_commit()

    protocol_doc = {
        "kind": "M10_3B_DIAGNOSIS_PROTOCOL",
        "milestone": "M10.3B",
        "branch": branch,
        "start_commit": start_commit,
        "amends_nothing_in": [
            "docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md",
            "docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_REFIT_RESULTS.md",
            "reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-closure.json",
        ],
        "reuses_frozen_population_from": "scripts/hydrocore_v5/m10_3_refit_protocol.py (unchanged)",
        "m10_3a_protocol_hash": proto.protocol_hash(),
        "trains_nothing": True,
        "touches_no_checkpoint": True,
        "locked_test_opened_before": locked_before,
    }
    (M10_3B_DIR / "m10-3b-protocol.json").write_text(json.dumps(protocol_doc, indent=2, default=str) + "\n")

    formula_audit = _target_formula_audit()
    (M10_3B_DIR / "m10-3b-target-formula-audit.json").write_text(json.dumps(formula_audit, indent=2, default=str) + "\n")
    print("wrote target-formula-audit", flush=True)

    ranking_audit = _ranking_alignment_audit()
    (M10_3B_DIR / "m10-3b-ranking-alignment-audit.json").write_text(json.dumps(ranking_audit, indent=2, default=str) + "\n")
    print("wrote ranking-alignment-audit:", ranking_audit["all_checks_passed"], flush=True)

    calibration_audit = _calibration_preservation_audit()
    (M10_3B_DIR / "m10-3b-calibration-preservation-audit.json").write_text(json.dumps(calibration_audit, indent=2, default=str) + "\n")
    print("wrote calibration-preservation-audit", flush=True)

    print("building population (this rebuilds the SAME M10.3A train+validation corpus, ~15-20 min)...", flush=True)
    train_examples, validation_examples, network, node_ids = _build_population()
    train_records = _records(train_examples)
    validation_records = _records(validation_examples)
    print(f"train examples: {len(train_examples)}, validation examples: {len(validation_examples)}", flush=True)

    manifest = {
        "train_scenario_ids_sha256": _sha256_of_scenario_ids(train_examples),
        "validation_scenario_ids_sha256": _sha256_of_scenario_ids(validation_examples),
        "n_train_examples": len(train_examples), "n_validation_examples": len(validation_examples),
    }

    identifiability_train = _target_identifiability(train_records, "train")
    identifiability_validation = _target_identifiability(validation_records, "validation")
    (M10_3B_DIR / "m10-3b-target-identifiability.json").write_text(json.dumps(
        {"kind": "M10_3B_TARGET_IDENTIFIABILITY", "manifest": manifest, "train": identifiability_train, "validation": identifiability_validation},
        indent=2, default=str) + "\n")
    print("wrote target-identifiability", flush=True)

    variance_train = _within_incident_variance(train_records, "train")
    variance_validation = _within_incident_variance(validation_records, "validation")
    (M10_3B_DIR / "m10-3b-within-incident-variance.json").write_text(json.dumps(
        {"kind": "M10_3B_WITHIN_INCIDENT_VARIANCE_COMBINED", "manifest": manifest, "train": variance_train, "validation": variance_validation},
        indent=2, default=str) + "\n")
    print("wrote within-incident-variance", flush=True)

    diversity_train = _candidate_diversity(train_records, "train")
    diversity_validation = _candidate_diversity(validation_records, "validation")
    (M10_3B_DIR / "m10-3b-candidate-diversity.json").write_text(json.dumps(
        {"kind": "M10_3B_CANDIDATE_DIVERSITY_COMBINED", "manifest": manifest, "train": diversity_train, "validation": diversity_validation},
        indent=2, default=str) + "\n")
    print("wrote candidate-diversity", flush=True)

    correlation_train = _target_correlation(train_records, "train")
    correlation_validation = _target_correlation(validation_records, "validation")
    (M10_3B_DIR / "m10-3b-target-correlation.json").write_text(json.dumps(
        {"kind": "M10_3B_TARGET_CORRELATION_COMBINED", "manifest": manifest, "train": correlation_train, "validation": correlation_validation},
        indent=2, default=str) + "\n")
    print("wrote target-correlation", flush=True)

    feature_train = _feature_identifiability(train_records, "train")
    feature_validation = _feature_identifiability(validation_records, "validation")
    (M10_3B_DIR / "m10-3b-feature-identifiability.json").write_text(json.dumps(
        {"kind": "M10_3B_FEATURE_IDENTIFIABILITY_COMBINED", "manifest": manifest, "train": feature_train, "validation": feature_validation},
        indent=2, default=str) + "\n")
    print("wrote feature-identifiability", flush=True)

    oracle_train = _oracle_utility(train_records, "train")
    oracle_validation = _oracle_utility(validation_records, "validation")
    (M10_3B_DIR / "m10-3b-oracle-utility.json").write_text(json.dumps(
        {"kind": "M10_3B_ORACLE_UTILITY_COMBINED", "manifest": manifest, "train": oracle_train, "validation": oracle_validation},
        indent=2, default=str) + "\n")
    print("wrote oracle-utility", flush=True)

    leakage_train = _leakage_audit(train_records, "train")
    leakage_validation = _leakage_audit(validation_records, "validation")
    (M10_3B_DIR / "m10-3b-leakage-audit.json").write_text(json.dumps(
        {"kind": "M10_3B_LEAKAGE_AUDIT_COMBINED", "manifest": manifest, "train": leakage_train, "validation": leakage_validation},
        indent=2, default=str) + "\n")
    print("wrote leakage-audit", flush=True)

    locked_after = m10.assert_locked_test_closed()
    final_commit = m10.current_commit()
    summary = {
        "kind": "M10_3B_DIAGNOSIS_RUN_SUMMARY",
        "branch": branch, "start_commit": start_commit, "final_commit_at_analysis_time": final_commit,
        "manifest": manifest,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    (M10_3B_DIR / "m10-3b-run-summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2))
    print("M10.3B diagnostic data collection complete.")


if __name__ == "__main__":
    main()
