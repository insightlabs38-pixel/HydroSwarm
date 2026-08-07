"""core-issues3.txt Phase 12 Stage E: Strategist-policy comparison.

Compares (Stage E's exact required set):

- ``exact_all``: exact verification of all bounded candidates (the oracle --
  picks the true best VALID plan, having exactly verified every real
  candidate);
- ``deterministic_heuristic``: the pre-existing production heuristic
  (hydroswarm.planning.response.generate_response_plans' own fixed
  per-template predicted_value/predicted_validity, ranked and shortlisted
  exactly the way hydroswarm.planning.response.prescreen_top_plans already
  does in production);
- ``learned_ordering``: HydroCore's trained candidate-conditioned
  plan_value_head alone, ranking every real candidate and checking only its
  single top pick;
- ``learned_prescreen``: HydroCore's trained plan_validity_head AND
  plan_value_head together (P(valid) * predicted value), shortlisted to the
  same top-K exact-verification budget as ``deterministic_heuristic`` for a
  fair simulator-call comparison.

"WNTR remains authoritative in every learned variant" (core-issues3.txt
Phase 12 Stage E) -- no policy here ever ACTS on a raw predicted score.
Every policy's final selection is always the ground-truth, already-exactly-
WNTR-verified `plan_validity`/`plan_value` of whichever candidate(s) it
chose to "check" (see the "What 'simulator calls' means here" section
below); an unchecked candidate is never treated as the answer.

## What "simulator calls" means here (a real, documented scoping decision,
## not a shortcut)

data/learning-v2/cycle-b2-trajectories-v3/strategist-tensors-normalized
(scripts/build_strategist_candidate_dataset.py, Phase 10.3) already stores
the FULL bounded candidate set's exact WNTR-verified plan_validity/
plan_value/consequence-proxy targets for every real training-label
scenario -- Phase 3.1's own repair requires training-label generation to
verify every proposal generate_response_plans returns, never a
heuristically prescreened subset, precisely so a policy comparison like
this one is possible without re-running WNTR. Every value this script reads
is a genuine WNTR/EPANET output, exactly as governed everywhere else in
this project; what varies PER POLICY is only which subset of those
already-computed exact values that policy would have had to request in a
real, budget-constrained deployment (`prescreen_top_plans`' own
`maximum_exact_simulations` concept). "simulator_calls" reports the size of
that subset, honestly-scoped as "how many of these already-exact values did
this policy need to look at before selecting one" -- not a live
re-simulation count. This mirrors run_stage_d_scout_policy_comparison.py's
own "real WNTR-derived truth, never a training loss value" discipline for
its policies' entropy-reduction/agreement metrics, applied here to the
Strategist's own already-governed exact targets instead.

Every real validation scenario in this corpus carries all 9 canonical
candidates (matches ACTION_TEMPLATE_COUNT; confirmed by
scripts/train_strategist_heads.py's own smoke run: 9000 valid plan_validity
targets / 1000 validation scenarios == 9), so `exact_all`'s simulator-call
count is always 9 here -- the real, governed, upper bound the other three
policies are being measured for savings against.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from safetensors.torch import load_file

from hydroswarm.model import HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT, ACTION_TEMPLATES
from hydroswarm.planning.response import PlanGenerationContext, generate_response_plans
from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology

SEED = 20260807
POLICIES = ("exact_all", "deterministic_heuristic", "learned_ordering", "learned_prescreen")
#: Matches hydroswarm.planning.response.prescreen_top_plans' own default
#: exact-simulation budget -- both `deterministic_heuristic` and
#: `learned_prescreen` are measured against the SAME budget the real
#: production heuristic already uses, for a fair simulator-call comparison.
SHORTLIST_K = 3
#: Matches prescreen_top_plans' own eligibility filter.
ELIGIBILITY_THRESHOLD = 0.5


# --- deterministic heuristic: the real per-template scores, not re-derived -


def build_heuristic_template_scores() -> dict[str, tuple[float, float]]:
    """Extract generate_response_plans' own fixed per-template
    (predicted_value, predicted_validity) pair for every one of the 9
    canonical templates, by actually calling the real function with a
    dummy-but-structurally-complete context (real node/link ID strings,
    none of which are looked up against any real network -- the function
    itself never touches network state, only context.* tuple membership/
    length) rather than hand-copying its hardcoded numbers into a second
    table here. Keeps this script's heuristic baseline provably identical
    to the one actually deployed in hydroswarm.planning.response, per this
    codebase's own established aversion to duplicated-list drift (see
    reports/results/v4/pre-freeze-implementation-handoff.md's repeated
    "two lists drift apart" defect class)."""

    context = PlanGenerationContext(
        incident_id=uuid4(),
        model_version="stage-e-heuristic-lookup",
        probable_source_nodes=("J1",),
        isolatable_links=("L1", "L2"),
        downstream_flush_nodes=("J2",),
        critical_demand_nodes=("J3",),
        monitor_nodes=("J1",),
        sampled_nodes=frozenset(),
        maximum_actions=8,
    )
    proposals = generate_response_plans(context, maximum_plans=ACTION_TEMPLATE_COUNT)
    scores = {proposal.template: (proposal.predicted_value, proposal.predicted_validity) for proposal in proposals}
    missing = set(ACTION_TEMPLATES) - set(scores)
    if missing:
        raise RuntimeError(
            f"dummy PlanGenerationContext did not elicit all {ACTION_TEMPLATE_COUNT} canonical templates "
            f"(missing: {sorted(missing)}) -- widen the dummy context's fields"
        )
    return scores


# --- per-scenario candidate view over the already-exact governed tensors ---


class ScenarioCandidates:
    """One scenario's real (plan_mask=True) candidates, both ground-truth
    (already exact-WNTR-verified, from the tensor targets) and, once a
    model is supplied, learned-predicted."""

    def __init__(self, template_names: list[str], valid: list[bool], value: list[float], value_mask: list[bool]) -> None:
        self.template_names = template_names
        self.valid = valid
        self.value = value
        self.value_mask = value_mask
        self.count = len(template_names)

    def ground_truth_value(self, position: int) -> float:
        return self.value[position] if self.value_mask[position] else 0.0

    def no_action_index(self) -> int | None:
        for position, name in enumerate(self.template_names):
            if name == "NO_ACTION":
                return position
        return None


def _scenario_candidates(inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], row: int) -> ScenarioCandidates:
    mask = inputs["plan_mask"][row].bool()
    positions = mask.nonzero(as_tuple=True)[0].tolist()
    template_ids = inputs["plan_template_ids"][row]
    valid_t = targets["plan_validity"][row].bool()
    value_t = targets["plan_value"][row].float()
    value_mask_t = targets["plan_value_mask"][row].bool()
    return ScenarioCandidates(
        template_names=[ACTION_TEMPLATES[int(template_ids[position])] for position in positions],
        valid=[bool(valid_t[position]) for position in positions],
        value=[float(value_t[position]) for position in positions],
        value_mask=[bool(value_mask_t[position]) for position in positions],
    )


# --- policies: rank the real candidates, then "check" (ground-truth-look
# up) only a bounded shortlist -----------------------------------------


def _select_from_shortlist(candidates: ScenarioCandidates, order: list[int]) -> dict[str, Any]:
    """Common selection logic for every policy: check `order` (already the
    policy's own ranked shortlist, longest = SHORTLIST_K) in rank order,
    against ground truth; select the first-encountered VALID candidate with
    the highest ground-truth plan_value seen so far; fall back to
    NO_ACTION's own ground-truth value (never a fabricated 0.0 unless
    NO_ACTION itself is genuinely absent from this scenario's real
    candidate set, which Phase 3.1 guarantees it never is)."""

    checked = 0
    best_position: int | None = None
    best_value = float("-inf")
    first_checked_valid: bool | None = None
    for position in order:
        checked += 1
        is_valid = candidates.valid[position]
        if first_checked_valid is None:
            first_checked_valid = is_valid
        if is_valid:
            value = candidates.ground_truth_value(position)
            if value > best_value:
                best_value = value
                best_position = position

    if best_position is None:
        no_action_index = candidates.no_action_index()
        selected_value = candidates.ground_truth_value(no_action_index) if no_action_index is not None else 0.0
        selected_template = "NO_ACTION"
        selected_valid = True if no_action_index is not None else False
    else:
        selected_value = best_value
        selected_template = candidates.template_names[best_position]
        selected_valid = True

    return {
        "simulator_calls": checked,
        "selected_template": selected_template,
        "selected_valid": selected_valid,
        "selected_value": selected_value,
        "first_checked_valid": first_checked_valid,
        "fell_back_to_no_action": best_position is None,
    }


def policy_exact_all(candidates: ScenarioCandidates) -> dict[str, Any]:
    """Oracle: every real candidate is exactly verified (Phase 3.1 -- this
    IS what the stored dataset already did, unconditionally), so its
    "shortlist" is the entire real candidate set, ranked by ground-truth
    value among valid candidates (order does not affect the selection, only
    the reported simulator-call count -- see module docstring)."""

    order = sorted(range(candidates.count), key=lambda position: -candidates.ground_truth_value(position))
    return _select_from_shortlist(candidates, order)


def policy_deterministic_heuristic(candidates: ScenarioCandidates, heuristic_scores: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Mirrors hydroswarm.planning.response.prescreen_top_plans' own
    filter+sort exactly: eligible = predicted_validity >= 0.5; sort by
    (is NO_ACTION, -(predicted_value * predicted_validity)); take the top
    SHORTLIST_K."""

    eligible = [
        position
        for position in range(candidates.count)
        if heuristic_scores[candidates.template_names[position]][1] >= ELIGIBILITY_THRESHOLD
    ]
    eligible.sort(
        key=lambda position: (
            candidates.template_names[position] == "NO_ACTION",
            -(heuristic_scores[candidates.template_names[position]][0] * heuristic_scores[candidates.template_names[position]][1]),
        )
    )
    return _select_from_shortlist(candidates, eligible[:SHORTLIST_K])


def policy_learned_ordering(candidates: ScenarioCandidates, predicted_value: list[float], predicted_valid_probability: list[float]) -> dict[str, Any]:
    """Pure ranking by the trained plan_value_head alone (ignores predicted
    validity entirely) -- checks only its single top pick (K=1). Tests
    whether ranking by predicted value alone, with no eligibility filter,
    already tends to find a genuinely valid, good plan zero-shot."""

    del predicted_valid_probability
    order = sorted(range(candidates.count), key=lambda position: -predicted_value[position])
    return _select_from_shortlist(candidates, order[:1])


def policy_learned_prescreen(candidates: ScenarioCandidates, predicted_value: list[float], predicted_valid_probability: list[float]) -> dict[str, Any]:
    """The trained plan_validity_head AND plan_value_head together --
    P(valid) * predicted value, with the SAME eligibility filter and
    exact-verification budget (SHORTLIST_K) as
    policy_deterministic_heuristic, for a fair simulator-call comparison
    between the learned and the pre-existing production prescreener."""

    eligible = [position for position in range(candidates.count) if predicted_valid_probability[position] >= ELIGIBILITY_THRESHOLD]
    eligible.sort(key=lambda position: -(predicted_value[position] * predicted_valid_probability[position]))
    return _select_from_shortlist(candidates, eligible[:SHORTLIST_K])


# --- orchestration -------------------------------------------------------


def load_strategist_model(checkpoint_path: Path) -> HydroCore:
    model = HydroCore.from_variant(
        "small",
        prior_mode="feature_only",
        strategist_mode="candidate_conditioned",
        action_vocabulary_size=ACTION_TEMPLATE_COUNT,
        consequence_prescreening_heads=True,
    )
    model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
    model.eval()
    return model


def default_strategist_checkpoint() -> Path:
    report_path = Path("reports/results/v4/strategist-heads-training.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = report.get("result")
    if not result:
        raise RuntimeError(f"{report_path} has no successful 'result' -- run scripts/train_strategist_heads.py first")
    return Path(result["final_checkpoint"]) / "model.safetensors"


def _predicted_scores(model: HydroCore, inputs: dict[str, torch.Tensor], row: int, count: int) -> tuple[list[float], list[float]]:
    with torch.no_grad():
        output = model(inputs)
    value = output["plan_value"][row, :count].tolist()
    valid_probability = torch.softmax(output["plan_validity_logits"][row, :count].float(), dim=-1)[:, 1].tolist()
    return value, valid_probability


def run(*, corpus_root: Path, split: str, strategist_checkpoint: Path, limit: int, batch_size: int) -> dict[str, Any]:
    started = time.perf_counter()
    heuristic_scores = build_heuristic_template_scores()
    model = load_strategist_model(strategist_checkpoint)

    dataset = ShardedScenarioDataset(corpus_root / split, expected_split=split)
    dataset.verify_shard_checksums()
    total = len(dataset)
    evaluate_count = min(limit, total) if limit else total
    # Stride across the full split (matches run_stage_d_scout_policy_comparison.py's
    # own established convention -- avoids a curriculum-ordered prefix bias).
    indices = list(range(total))
    if evaluate_count < total:
        stride = max(1, total // evaluate_count)
        indices = indices[::stride][:evaluate_count]

    per_policy_records: dict[str, list[dict[str, Any]]] = {name: [] for name in POLICIES}
    skipped_no_real_candidates = 0

    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        batch_examples = [dataset[index] for index in batch_indices]
        inputs, targets = collate_variable_topology(batch_examples)
        with torch.no_grad():
            output = model(inputs)
        batch_value = output["plan_value"].float()
        batch_valid_probability = torch.softmax(output["plan_validity_logits"].float(), dim=-1)[..., 1]

        for row in range(len(batch_examples)):
            candidates = _scenario_candidates(inputs, targets, row)
            if candidates.count == 0:
                skipped_no_real_candidates += 1
                continue
            predicted_value = batch_value[row, : candidates.count].tolist()
            predicted_valid_probability = batch_valid_probability[row, : candidates.count].tolist()

            per_policy_records["exact_all"].append(policy_exact_all(candidates))
            per_policy_records["deterministic_heuristic"].append(policy_deterministic_heuristic(candidates, heuristic_scores))
            per_policy_records["learned_ordering"].append(policy_learned_ordering(candidates, predicted_value, predicted_valid_probability))
            per_policy_records["learned_prescreen"].append(policy_learned_prescreen(candidates, predicted_value, predicted_valid_probability))

        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"  ... {min(batch_start + batch_size, len(indices))}/{len(indices)} scenarios processed ({time.perf_counter() - started:.0f}s)")

    oracle_value_by_index = [record["selected_value"] for record in per_policy_records["exact_all"]]

    summary: dict[str, Any] = {}
    for name in POLICIES:
        records = per_policy_records[name]
        if not records:
            continue
        regret = [oracle_value_by_index[index] - record["selected_value"] for index, record in enumerate(records)]
        matched_oracle = [abs(regret_value) < 1e-9 for regret_value in regret]
        summary[name] = {
            "scenarios": len(records),
            "mean_simulator_calls": float(np.mean([record["simulator_calls"] for record in records])),
            "selected_valid_rate": float(np.mean([record["selected_valid"] for record in records])),
            "found_non_no_action_plan_rate": float(np.mean([record["selected_template"] != "NO_ACTION" for record in records])),
            "fell_back_to_no_action_rate": float(np.mean([record["fell_back_to_no_action"] for record in records])),
            "first_checked_was_valid_rate": float(
                np.mean([record["first_checked_valid"] for record in records if record["first_checked_valid"] is not None])
            ),
            "mean_regret_vs_oracle": float(np.mean(regret)),
            "matched_oracle_best_rate": float(np.mean(matched_oracle)),
        }

    return {
        "scenarios_requested": len(indices),
        "scenarios_evaluated": len(indices) - skipped_no_real_candidates,
        "skipped_no_real_candidates": skipped_no_real_candidates,
        "shortlist_k": SHORTLIST_K,
        "eligibility_threshold": ELIGIBILITY_THRESHOLD,
        "heuristic_template_scores": {name: list(score) for name, score in heuristic_scores.items()},
        "policies": summary,
        "wall_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus-root", type=Path, default=Path("data/learning-v2/cycle-b2-trajectories-v3/strategist-tensors-normalized")
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--strategist-checkpoint", type=Path, default=None, help="defaults to reading strategist-heads-training.json")
    parser.add_argument("--limit", type=int, default=1000, help="number of scenarios to evaluate (0 = entire split)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/stage-e-strategist-comparison.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    strategist_checkpoint = args.strategist_checkpoint or default_strategist_checkpoint()

    started = time.perf_counter()
    try:
        result = run(
            corpus_root=args.corpus_root,
            split=args.split,
            strategist_checkpoint=strategist_checkpoint,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        failure: str | None = None
        print(f"OK ({result['wall_seconds']:.1f}s)")
    except Exception as error:  # noqa: BLE001 -- record and report, matching this repo's established smoke-job pattern
        result = None
        failure = f"{type(error).__name__}: {error}"
        print(f"FAILED: {failure}")

    report = {
        "schema_version": 1,
        "stage": "core-issues3.txt Phase 12 Stage E: Strategist-policy comparison",
        "strategist_checkpoint": str(strategist_checkpoint),
        "seed": SEED,
        "wall_seconds": time.perf_counter() - started,
        "result": result,
        "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if result is not None:
        print(json.dumps(result["policies"], indent=2))
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
