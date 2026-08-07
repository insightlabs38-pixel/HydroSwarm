"""core-issues3.txt Phase 12 Stage D: Scout-policy comparison.

Compares (Stage D's exact required set):

- random accessible sampling;
- fixed deterministic order;
- classical EIG (hydroswarm.training.scout_labels.generate_scout_label's
  own policy, i.e. hydroswarm.sampling.active.rank_sample_locations's
  top choice);
- learned Scout (scripts/train_scout_heads.py's trained sample_node_head);
- classical EIG plus learned residual/rerank.

"Do not promote a learned Scout merely because its supervised loss
decreases" -- every metric here is computed from real WNTR-derived truth
(hydroswarm.training.scenario_reconstruction.simulate_all_node_truth) and
the real classical posterior (hydroswarm.classical.signatures.
localize_with_signatures), never from a training loss value.

## A real, documented architectural limitation this script works within

HydroCore's Scout input has no "already_sampled"/revealed-evidence
conditioning (core-issues3.txt Phase 10.2's own documented scoping
decision, reports/results/v4/pre-freeze-implementation-handoff.md's
Phase 10.2 section) -- the trained model can only make ONE well-supported
decision: its recommendation from the scenario's ORIGINAL evidence.
Re-running the same model on the same input at step 2+ would
deterministically recommend the same node again, which is not a real
"keep sampling" policy.

Given this, this script runs two genuinely different comparisons rather
than pretending all 5 policies support the same evaluation:

1. STEP-0 comparison (all 5 policies, `--mode step0`): each policy picks
   ONE node from the scenario's original evidence; that node is revealed
   once; realized entropy reduction (via localize_with_signatures, before
   vs. after) and agreement with classical EIG's own top pick are
   measured. Fair for every policy, including learned_scout/
   classical_plus_residual.
2. MULTI-STEP operational trajectory comparison (`--mode trajectory`,
   random / fixed_order / classical_eig ONLY -- the 3 policies that need
   no model conditioning to make a well-defined step-N decision): runs up
   to --maximum-samples steps, revealing genuinely new evidence each step,
   tracking localize_with_signatures' source posterior/candidate set at
   every step. Reports median samples to resolution, resolved-within-
   {1,2,3}, and the candidate-set-size trajectory.

learned_scout/classical_plus_residual are NOT evaluated in mode
"trajectory" -- this is a documented, real limitation (see above), not an
oversight or silent omission.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from generate_cycle_b_corpus import _degradation_probabilities  # noqa: E402
from generate_trajectory_corpus import _train_topology_loaders  # noqa: E402

from hydroswarm.classical.metrics import entropy  # noqa: E402
from hydroswarm.classical.signatures import (  # noqa: E402
    SignatureArtifact,
    SignatureCache,
    SignatureCacheKey,
    localize_with_signatures,
)
from hydroswarm.data.scenarios import DatasetSplit, GeneratedScenario, load_generated_scenarios  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.sampling.active import SampleCandidate  # noqa: E402
from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology  # noqa: E402
from hydroswarm.training.corpus import _hydraulic_state_hash, build_feature_context  # noqa: E402
from hydroswarm.training.scenario_reconstruction import (  # noqa: E402
    ReconstructedScenarioContext,
    ScenarioReconstructionError,
    reconstruct_scenario_network,
    simulate_all_node_truth,
)
from hydroswarm.training.scout_labels import (  # noqa: E402
    ScoutLabel,
    build_signature_artifact_for_network,
    generate_scout_label,
    reindex_to_signature_grid,
)
from hydroswarm.training.scout_trajectory import reveal_sample_measurement  # noqa: E402

SEED = 20260807
POLICIES = (
    "random",
    "fixed_order",
    "classical_eig",
    "learned_scout",
    "classical_plus_residual",
)
TRAJECTORY_POLICIES = ("random", "fixed_order", "classical_eig")


# --- setup: per-topology signature artifacts (real WNTR simulation, cached) ---


def build_topology_artifacts(corpus_dir: Path, signature_cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Fit (or load, if cached) one SignatureArtifact per topology family
    from that topology's real TRAIN scenarios -- mirrors
    scripts/generate_trajectory_corpus.py's own setup exactly (same
    functions, same train-only-fitting discipline), since this script
    needs the identical artifacts for the classical/simulation-based
    policies to be scientifically comparable to that corpus's own labels."""

    loaders = _train_topology_loaders()
    networks = {name: loader() for name, loader in loaders.items()}
    train_scenarios_by_family: dict[str, list[GeneratedScenario]] = {name: [] for name in loaders}
    for scenario in load_generated_scenarios(corpus_dir / "scenarios", DatasetSplit.TRAIN):
        family = scenario.manifest.network_family
        if family in train_scenarios_by_family:
            train_scenarios_by_family[family].append(scenario)

    result: dict[str, dict[str, Any]] = {}
    for family, network in networks.items():
        scenarios = train_scenarios_by_family[family]
        if not scenarios:
            continue
        context = build_feature_context(network)
        cache = SignatureCache(signature_cache_dir)
        key = SignatureCacheKey(
            network_hash=scenarios[0].manifest.network_sha256,
            hydraulic_state_hash=_hydraulic_state_hash(context.state),
            simulator_version=scenarios[0].manifest.simulator_version,
            configuration_hash="stage-d-scout-policy-comparison-v1",
            sensor_layout_hash="all-junctions",
        )
        artifact = build_signature_artifact_for_network(network, cache, key=key)
        result[family] = {"network": network, "artifact": artifact}
    return result


# --- policies: given a step's classical-EIG-ranked candidates, choose one ---


def _accessible_unsampled(label: ScoutLabel, already_sampled: set[str]) -> list[SampleCandidate]:
    return [candidate for candidate in label.candidates if candidate.accessible and candidate.node_id not in already_sampled]


def policy_random(label: ScoutLabel, already_sampled: set[str], rng: np.random.Generator) -> str | None:
    pool = _accessible_unsampled(label, already_sampled)
    if not pool:
        return None
    return str(rng.choice([candidate.node_id for candidate in pool]))


def policy_fixed_order(label: ScoutLabel, already_sampled: set[str], rng: np.random.Generator) -> str | None:
    del rng
    pool = _accessible_unsampled(label, already_sampled)
    if not pool:
        return None
    return min(candidate.node_id for candidate in pool)


def policy_classical_eig(label: ScoutLabel, already_sampled: set[str], rng: np.random.Generator) -> str | None:
    """Deliberately does NOT just return `label.sample_node_id` -- that
    value already folds in generate_scout_label's own SEPARATE
    "should we bother sampling at all" stop-threshold decision
    (`minimum_information_gain_bits`), so it can be None even when
    `label.candidates` still has a well-defined top-ranked choice,
    silently making this policy answer a different question ("should we
    sample?") than every other policy here ("given we ARE taking one more
    sample, which node?"). `label.candidates` is already EIG-sorted
    (rank_sample_locations' own `sorted(..., key=expected_information_gain_bits,
    reverse=True)`), so its first accessible/unsampled entry IS the
    classical-EIG top choice, independent of the stop threshold."""

    del rng
    pool = _accessible_unsampled(label, already_sampled)
    return pool[0].node_id if pool else None


def make_policy_learned(model: HydroCore, inputs: dict[str, torch.Tensor], node_ids: tuple[str, ...]) -> Callable:
    """`node_ids` MUST be the canonical full-topology node order
    `sample_node_logits` is indexed against (`example.topology.node_ids` --
    see hydroswarm.training.scout_trajectory.build_scout_trajectory's own
    docstring for why `sorted(artifact.sensor_nodes)` is a DIFFERENT,
    junction-only ordering that silently disagrees with this space
    whenever the network has any reservoir or tank). `label.candidates`
    (junction-only) is always a subset of this canonical space, so looking
    up each candidate's canonical index is always well-defined."""

    with torch.no_grad():
        logits = model(inputs)["sample_node_logits"][0]

    def policy(label: ScoutLabel, already_sampled: set[str], rng: np.random.Generator) -> str | None:
        del rng
        pool = {candidate.node_id for candidate in _accessible_unsampled(label, already_sampled)}
        if not pool:
            return None
        masked = torch.tensor(
            [logits[index].item() if node in pool else float("-inf") for index, node in enumerate(node_ids)]
        )
        best_index = int(torch.argmax(masked))
        return node_ids[best_index] if node_ids[best_index] in pool else None

    return policy


def make_policy_classical_plus_residual(
    model: HydroCore, inputs: dict[str, torch.Tensor], node_ids: tuple[str, ...]
) -> Callable:
    """Rank fusion: classical EIG rank + learned expected_information_gain
    rank, lower combined rank wins -- a simple, standard rerank strategy
    (avoids needing to calibrate two differently-scaled score magnitudes
    against each other). `node_ids` has the same canonical-node-space
    requirement as make_policy_learned -- see its docstring."""

    with torch.no_grad():
        predicted_gain = model(inputs)["expected_information_gain"][0]
    node_index = {node: index for index, node in enumerate(node_ids)}

    def policy(label: ScoutLabel, already_sampled: set[str], rng: np.random.Generator) -> str | None:
        del rng
        pool = _accessible_unsampled(label, already_sampled)
        if not pool:
            return None
        classical_rank = {candidate.node_id: rank for rank, candidate in enumerate(pool)}
        residual_ranked = sorted(pool, key=lambda candidate: -float(predicted_gain[node_index[candidate.node_id]]))
        residual_rank = {candidate.node_id: rank for rank, candidate in enumerate(residual_ranked)}
        best = min(pool, key=lambda candidate: classical_rank[candidate.node_id] + residual_rank[candidate.node_id])
        return best.node_id

    return policy


# --- step-0 comparison (all 5 policies) -------------------------------------


def _posterior_entropy(scenario, artifact: SignatureArtifact, revealed: dict[str, float]) -> float:
    observed, valid = reindex_to_signature_grid(scenario, artifact.sensor_nodes, artifact.sample_times_seconds)
    if revealed:
        observed = observed.copy()
        valid = valid.copy()
        positions = {node: index for index, node in enumerate(artifact.sensor_nodes)}
        for node, value in revealed.items():
            column = positions.get(node)
            if column is None:
                continue
            observed[:, column] = value
            valid[:, column] = True
    result = localize_with_signatures(observed, artifact, observation_mask=valid, noise_scale=0.5)
    return float(entropy(list(result.source_probabilities.values())))


def evaluate_step0(
    scenario,
    artifact: SignatureArtifact,
    reconstruction: ReconstructedScenarioContext,
    policies: dict[str, Callable],
    *,
    noise_scale_mg_l: float,
) -> dict[str, Any]:
    label = generate_scout_label(scenario, artifact, noise_scale_mg_l=noise_scale_mg_l)
    all_node_truth = simulate_all_node_truth(reconstruction)
    before_entropy = _posterior_entropy(scenario, artifact, {})
    rng = np.random.default_rng(int(str(scenario.manifest.scenario_id).replace("-", "")[:16], 16))
    # The reference for "agrees_with_classical_eig" -- NOT label.sample_node_id,
    # which also folds in generate_scout_label's separate stop-threshold
    # decision (see policy_classical_eig's own docstring for why that
    # would make this comparison silently answer the wrong question).
    classical_eig_choice = policy_classical_eig(label, set(), rng)

    per_policy: dict[str, Any] = {}
    for name, policy in policies.items():
        chosen = policy(label, set(), rng)
        if chosen is None:
            per_policy[name] = {"chosen_node": None, "entropy_reduction_bits": 0.0, "agrees_with_classical_eig": None}
            continue
        revealed_value = reveal_sample_measurement(reconstruction, all_node_truth, chosen, 0, noise_scale_mg_l=noise_scale_mg_l)
        after_entropy = _posterior_entropy(scenario, artifact, {chosen: revealed_value})
        per_policy[name] = {
            "chosen_node": chosen,
            "entropy_reduction_bits": before_entropy - after_entropy,
            "agrees_with_classical_eig": chosen == classical_eig_choice,
        }
    return {"before_entropy_bits": before_entropy, "policies": per_policy}


# --- multi-step operational trajectory comparison (3 policies only) --------


def run_policy_trajectory(
    scenario,
    artifact: SignatureArtifact,
    reconstruction: ReconstructedScenarioContext,
    policy: Callable,
    *,
    maximum_samples: int,
    noise_scale_mg_l: float,
) -> dict[str, Any]:
    truth_source = scenario.manifest.incident.source_nodes[0] if scenario.manifest.incident.source_nodes else None
    all_node_truth = simulate_all_node_truth(reconstruction)
    rng = np.random.default_rng(int(str(scenario.manifest.scenario_id).replace("-", "")[:16], 16))

    already_sampled: list[str] = []
    revealed: dict[str, float] = {}
    steps: list[dict[str, Any]] = []
    for step_index in range(maximum_samples + 1):
        label = generate_scout_label(
            scenario, artifact, already_sampled=already_sampled, revealed_samples=revealed, noise_scale_mg_l=noise_scale_mg_l
        )
        observed, valid = reindex_to_signature_grid(scenario, artifact.sensor_nodes, artifact.sample_times_seconds)
        if revealed:
            observed = observed.copy()
            valid = valid.copy()
            positions = {node: index for index, node in enumerate(artifact.sensor_nodes)}
            for node, value in revealed.items():
                column = positions.get(node)
                if column is None:
                    continue
                observed[:, column] = value
                valid[:, column] = True
        localization = localize_with_signatures(observed, artifact, observation_mask=valid, noise_scale=noise_scale_mg_l)
        ranked_sources = sorted(localization.source_probabilities.items(), key=lambda item: -item[1])
        top1 = ranked_sources[0][0] if ranked_sources else None
        top3 = {node for node, _probability in ranked_sources[:3]}
        # candidate_regions partitions the cumulative_mass candidate set
        # into hydraulically-connected components; total candidate-set
        # size is the sum of every region's own node count.
        candidate_set_size = sum(len(region) for region in localization.candidate_regions)
        steps.append(
            {
                "step": step_index,
                "samples_taken": len(already_sampled),
                "entropy_bits": float(entropy(list(localization.source_probabilities.values()))),
                "candidate_set_size": candidate_set_size,
                "resolved_top1": top1 == truth_source,
                "resolved_top3": truth_source in top3,
            }
        )
        if step_index >= maximum_samples:
            break
        chosen = policy(label, set(already_sampled), rng)
        if chosen is None:
            break
        revealed[chosen] = reveal_sample_measurement(reconstruction, all_node_truth, chosen, step_index, noise_scale_mg_l=noise_scale_mg_l)
        already_sampled = already_sampled + [chosen]

    resolved_step = next((step["step"] for step in steps if step["resolved_top1"]), None)
    return {"truth_source": truth_source, "steps": steps, "resolved_at_step": resolved_step}


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


# --- orchestration -----------------------------------------------------


def load_scout_model(checkpoint_path: Path) -> HydroCore:
    from safetensors.torch import load_file

    model = HydroCore.from_variant("small", prior_mode="feature_only", scout_control_heads=True)
    model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
    model.eval()
    return model


def default_scout_checkpoint() -> Path:
    report_path = Path("reports/results/v4/scout-heads-training.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = report.get("result")
    if not result:
        raise RuntimeError(f"{report_path} has no successful 'result' -- run scripts/train_scout_heads.py first")
    return Path(result["final_checkpoint"]) / "model.safetensors"


def build_example_index(scout_corpus_root: Path, split: str) -> dict[str, Any]:
    dataset = ShardedScenarioDataset(scout_corpus_root / split, expected_split=split)
    return {dataset[index].scenario_id: dataset[index] for index in range(len(dataset))}


def run(
    *,
    corpus_dir: Path,
    scout_corpus_root: Path,
    split: str,
    signature_cache_dir: Path,
    scout_checkpoint: Path,
    limit: int,
    maximum_samples: int,
    noise_scale_mg_l: float,
    mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    topology_artifacts = build_topology_artifacts(corpus_dir, signature_cache_dir)
    print(f"signature artifacts ready for topologies: {sorted(topology_artifacts)}")

    model = load_scout_model(scout_checkpoint)
    example_index = build_example_index(scout_corpus_root, split)

    scenarios = list(load_generated_scenarios(corpus_dir / "scenarios", DatasetSplit(split)))
    scenarios = [scenario for scenario in scenarios if scenario.manifest.network_family in topology_artifacts]
    # Stride across the full split rather than taking a curriculum-ordered
    # prefix (test_v4_production_checkpoint.py's own established
    # convention) -- a corpus's scenarios are typically curriculum-ordered
    # (CLEAN stage first), so the first `limit` scenarios alone would skew
    # toward already-easily-resolved cases with little real information to
    # gain from any policy, understating every policy's real spread.
    if limit < len(scenarios):
        stride = max(1, len(scenarios) // limit)
        scenarios = scenarios[::stride][:limit]
    print(f"evaluating {len(scenarios)} scenarios (families: {sorted(topology_artifacts)})")

    step0_records: list[dict[str, Any]] = []
    trajectory_records: dict[str, list[dict[str, Any]]] = {name: [] for name in TRAJECTORY_POLICIES}
    skipped_no_example = 0
    skipped_reconstruction_error = 0

    for index, scenario in enumerate(scenarios):
        scenario_id = str(scenario.manifest.scenario_id)
        family = scenario.manifest.network_family
        example = example_index.get(scenario_id)
        if example is None:
            skipped_no_example += 1
            continue
        network = topology_artifacts[family]["network"]
        artifact = topology_artifacts[family]["artifact"]
        try:
            reconstruction = reconstruct_scenario_network(
                network, scenario.manifest, degradation_policy=_degradation_probabilities, original=scenario
            )
        except ScenarioReconstructionError:
            skipped_reconstruction_error += 1
            continue

        node_ids = example.topology.node_ids
        inputs, _targets = collate_variable_topology([example])
        policies = {
            "random": policy_random,
            "fixed_order": policy_fixed_order,
            "classical_eig": policy_classical_eig,
            "learned_scout": make_policy_learned(model, inputs, node_ids),
            "classical_plus_residual": make_policy_classical_plus_residual(model, inputs, node_ids),
        }

        if mode in ("step0", "both"):
            step0_records.append(evaluate_step0(scenario, artifact, reconstruction, policies, noise_scale_mg_l=noise_scale_mg_l))

        if mode in ("trajectory", "both"):
            for name in TRAJECTORY_POLICIES:
                trajectory_records[name].append(
                    run_policy_trajectory(
                        scenario, artifact, reconstruction, policies[name],
                        maximum_samples=maximum_samples, noise_scale_mg_l=noise_scale_mg_l,
                    )
                )

        if (index + 1) % 25 == 0:
            print(f"  ... {index + 1}/{len(scenarios)} scenarios processed ({time.perf_counter() - started:.0f}s)")

    report: dict[str, Any] = {
        "scenarios_requested": len(scenarios),
        "scenarios_evaluated": len(scenarios) - skipped_no_example - skipped_reconstruction_error,
        "skipped_no_matching_example": skipped_no_example,
        "skipped_reconstruction_error": skipped_reconstruction_error,
        "wall_seconds": time.perf_counter() - started,
    }

    if step0_records:
        summary: dict[str, Any] = {}
        for name in POLICIES:
            reductions = [record["policies"][name]["entropy_reduction_bits"] for record in step0_records if record["policies"][name]["chosen_node"] is not None]
            agreements = [record["policies"][name]["agrees_with_classical_eig"] for record in step0_records if record["policies"][name]["agrees_with_classical_eig"] is not None]
            no_recommendation = sum(1 for record in step0_records if record["policies"][name]["chosen_node"] is None)
            summary[name] = {
                "mean_entropy_reduction_bits": float(np.mean(reductions)) if reductions else None,
                "agreement_with_classical_eig_rate": float(np.mean(agreements)) if agreements else None,
                "no_recommendation_count": no_recommendation,
                "recommendation_count": len(reductions),
            }
        report["step0_comparison"] = {"scenarios": len(step0_records), "policies": summary}

    if any(trajectory_records.values()):
        trajectory_summary: dict[str, Any] = {}
        for name, records in trajectory_records.items():
            resolved_steps = [record["resolved_at_step"] for record in records if record["resolved_at_step"] is not None]
            never_resolved = sum(1 for record in records if record["resolved_at_step"] is None)
            resolved_within = {
                k: sum(1 for step in resolved_steps if step <= k) / len(records) if records else None
                for k in (1, 2, 3)
            }
            mean_candidate_set_by_step: dict[int, float] = {}
            for step_index in range(maximum_samples + 1):
                sizes = [record["steps"][step_index]["candidate_set_size"] for record in records if len(record["steps"]) > step_index]
                if sizes:
                    mean_candidate_set_by_step[step_index] = float(np.mean(sizes))
            trajectory_summary[name] = {
                "trajectories": len(records),
                "median_samples_to_resolution": median([float(step) for step in resolved_steps]),
                "never_resolved_count": never_resolved,
                "resolved_within_k_samples": resolved_within,
                "mean_candidate_set_size_by_step": mean_candidate_set_by_step,
            }
        report["trajectory_comparison"] = {
            "maximum_samples": maximum_samples,
            "policies_evaluated": list(trajectory_records),
            "policies_excluded": [name for name in POLICIES if name not in TRAJECTORY_POLICIES],
            "exclusion_reason": "no already_sampled/revealed-evidence conditioning in HydroCore's Scout input "
            "(core-issues3.txt Phase 10.2's documented scoping decision) -- see module docstring",
            "policies": trajectory_summary,
        }

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/learning-v2/cycle-b2"))
    parser.add_argument(
        "--scout-corpus-root", type=Path,
        default=Path("data/learning-v2/cycle-b2-trajectories-v3/scout-tensors-normalized"),
    )
    parser.add_argument("--split", type=str, default="validation", choices=[s.value for s in DatasetSplit])
    parser.add_argument("--signature-cache-dir", type=Path, default=Path("experiments/cache/signatures"))
    parser.add_argument("--scout-checkpoint", type=Path, default=None, help="defaults to reading scout-heads-training.json")
    parser.add_argument("--limit", type=int, default=150, help="number of scenarios to evaluate")
    parser.add_argument("--maximum-samples", type=int, default=3)
    parser.add_argument("--noise-scale-mg-l", type=float, default=0.5)
    parser.add_argument("--mode", choices=("step0", "trajectory", "both"), default="both")
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/stage-d-scout-policy-comparison.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scout_checkpoint = args.scout_checkpoint or default_scout_checkpoint()

    started = time.perf_counter()
    try:
        result = run(
            corpus_dir=args.corpus_dir,
            scout_corpus_root=args.scout_corpus_root,
            split=args.split,
            signature_cache_dir=args.signature_cache_dir,
            scout_checkpoint=scout_checkpoint,
            limit=args.limit,
            maximum_samples=args.maximum_samples,
            noise_scale_mg_l=args.noise_scale_mg_l,
            mode=args.mode,
        )
        failure: str | None = None
        print(f"OK ({result['wall_seconds']:.1f}s)")
    except Exception as error:  # noqa: BLE001 -- record and report, matching this repo's established smoke-job pattern
        result = None
        failure = f"{type(error).__name__}: {error}"
        print(f"FAILED: {failure}")

    report = {
        "schema_version": 1,
        "stage": "core-issues3.txt Phase 12 Stage D: Scout-policy comparison",
        "scout_checkpoint": str(scout_checkpoint),
        "seed": SEED,
        "wall_seconds": time.perf_counter() - started,
        "result": result,
        "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if result is not None:
        print(json.dumps(result, indent=2)[:4000])
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
