"""HydroCore-v5 Milestone 9.1 PREFLIGHT (engineering-correctness only).

Frozen protocol: docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md.
Builds the CURRENT baseline and parameter-matched GRAPH_ODE/GRAPH_CDE/
GRAPH_SDE candidates (src/hydroswarm/model/continuous_time.py), runs
forward/backward finiteness, a bounded (<=20 optimizer step) mini-overfit
sanity check, and latency/memory benchmarking (including SDE Monte-Carlo
path-count stabilization). Uses ONLY synthetic tensors -- no
development_holdout, no calibration split, no locked data. Reports NO
accuracy metric of any kind (no top1/MRR/loss-as-a-ranking-signal): initial
and final loss are reported solely to demonstrate optimization mechanics
function, never to compare or rank architectures.

Writes:
  reports/evaluation/hydrocore-v5/m9-1-preflight-results.json
  reports/evaluation/hydrocore-v5/m9-1-solver-feasibility.json
  reports/evaluation/hydrocore-v5/m9-1-parameter-matching.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model.continuous_time import (  # noqa: E402
    GraphCDEDynamics,
    GraphODEDynamics,
    GraphSDEDynamics,
    _TORCHCDE_IMPORT_ERROR,
    _TORCHDIFFEQ_IMPORT_ERROR,
    _TORCHSDE_IMPORT_ERROR,
    torchcde,
    torchdiffeq,
    torchsde,
)
from hydroswarm.model.core import HydroCore  # noqa: E402
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT  # noqa: E402

REPORT_DIR = ROOT / "reports/evaluation/hydrocore-v5"

NODES = 6
EDGES = 8
NODE_FEATURE_DIM = 19
EDGE_FEATURE_DIM = 13
STEPS = 6
BATCH = 2

SHARED_MODEL_CONFIG: dict[str, Any] = dict(
    prior_mode="feature_only",
    event_control_heads=True,
    scout_control_heads=True,
    strategist_mode="candidate_conditioned",
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True,
    ood_category_head=True,
)


def _synthetic_batch(seed: int) -> dict[str, torch.Tensor]:
    """Hand-built synthetic HydroBatch tensors (preflight protocol Section
    1 data policy) -- never development_holdout/calibration/locked data."""

    torch.manual_seed(seed)
    node_features = torch.randn(BATCH, NODES, NODE_FEATURE_DIM)
    temporal_features = torch.randn(BATCH, STEPS, NODES, 6)
    quality_features = torch.randn(BATCH, STEPS, NODES, 4)
    timestamps = torch.stack(
        [torch.arange(STEPS).float() * 3600.0 for _ in range(BATCH)]
    )
    edge_index = torch.randint(0, NODES, (BATCH, 2, EDGES))
    edge_features = torch.randn(BATCH, EDGES, EDGE_FEATURE_DIM)
    return dict(
        node_features=node_features,
        temporal_features=temporal_features,
        quality_features=quality_features,
        timestamps=timestamps,
        edge_index=edge_index,
        edge_features=edge_features,
        node_mask=torch.ones(BATCH, NODES, dtype=torch.bool),
        sensor_mask=torch.ones(BATCH, STEPS, NODES, dtype=torch.bool),
        quality_mask=torch.ones(BATCH, STEPS, NODES, dtype=torch.bool),
        edge_mask=torch.ones(BATCH, EDGES, dtype=torch.bool),
    )


def _build_baseline() -> HydroCore:
    return HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)


def _parameter_matching() -> dict[str, Any]:
    baseline = _build_baseline()
    baseline_total = baseline.parameter_report().total
    tq_params = sum(p.numel() for p in baseline.temporal_encoder.parameters()) + sum(
        p.numel() for p in baseline.quality_encoder.parameters()
    )
    edge_feature_dim = baseline.edge_feature_dim
    temporal_feature_dim = baseline.temporal_feature_dim
    quality_feature_dim = baseline.quality_feature_dim
    d_model = baseline.d_model

    def full_model_params(dynamics: torch.nn.Module) -> int:
        return baseline_total - tq_params + sum(p.numel() for p in dynamics.parameters())

    def search(build):
        best = None
        for width in range(4, 600, 2):
            dynamics = build(width)
            total = full_model_params(dynamics)
            delta_pct = (total - baseline_total) / baseline_total * 100.0
            if best is None or abs(delta_pct) < abs(best[2]):
                best = (width, total, delta_pct)
        return best

    ode_width, ode_total, ode_delta = search(
        lambda w: GraphODEDynamics(
            temporal_feature_dim, quality_feature_dim, d_model=d_model,
            edge_feature_dim=edge_feature_dim, mlp_width=w,
        )
    )
    cde_width, cde_total, cde_delta = search(
        lambda w: GraphCDEDynamics(
            temporal_feature_dim, quality_feature_dim, d_model=d_model,
            edge_feature_dim=edge_feature_dim, mlp_width=w,
        )
    )
    sde_width, sde_total, sde_delta = search(
        lambda w: GraphSDEDynamics(
            temporal_feature_dim, quality_feature_dim, d_model=d_model,
            edge_feature_dim=edge_feature_dim, mlp_width=w,
        )
    )
    return {
        "baseline_total_params": baseline_total,
        "baseline_temporal_and_quality_params_replaced": tq_params,
        "arms": {
            "GRAPH_ODE": {"mlp_width": ode_width, "total_params": ode_total, "delta_percent": ode_delta},
            "GRAPH_CDE": {"mlp_width": cde_width, "total_params": cde_total, "delta_percent": cde_delta},
            "GRAPH_SDE": {"mlp_width": sde_width, "total_params": sde_total, "delta_percent": sde_delta},
        },
        "every_novel_arm_within_5_percent": all(
            abs(d) <= 5.0 for d in (ode_delta, cde_delta, sde_delta)
        ),
    }


def _build_model(arm: str, matching: dict[str, Any]) -> tuple[HydroCore, torch.nn.Module | None]:
    baseline = _build_baseline()
    if arm == "CURRENT":
        return baseline, None
    common = dict(
        temporal_feature_dim=baseline.temporal_feature_dim,
        quality_feature_dim=baseline.quality_feature_dim,
        d_model=baseline.d_model,
        edge_feature_dim=baseline.edge_feature_dim,
        mlp_width=matching["arms"][arm]["mlp_width"],
    )
    if arm == "GRAPH_ODE":
        dynamics: torch.nn.Module = GraphODEDynamics(**common)
    elif arm == "GRAPH_CDE":
        dynamics = GraphCDEDynamics(**common)
    elif arm == "GRAPH_SDE":
        dynamics = GraphSDEDynamics(**common)
    else:
        raise ValueError(arm)
    model = HydroCore.from_variant(
        "small", use_adapters=False, temporal_dynamics=dynamics, **SHARED_MODEL_CONFIG
    )
    return model, dynamics


def _finite_forward_backward(model: HydroCore, batch: dict[str, torch.Tensor], sde_seed: int | None) -> dict[str, Any]:
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    output = model(batch)
    logits = output["source_node_logits"]
    finite_forward = bool(torch.isfinite(logits).all())
    target = torch.zeros(BATCH, NODES)
    target[:, 0] = 1.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), target)
    finite_loss = bool(torch.isfinite(loss))
    loss.backward()
    dynamics_grad = 0.0
    if model.temporal_dynamics is not None:
        dynamics_grad = sum(
            p.grad.abs().sum().item() for p in model.temporal_dynamics.parameters() if p.grad is not None
        )
    else:
        dynamics_grad = sum(
            p.grad.abs().sum().item()
            for module in (model.temporal_encoder, model.quality_encoder)
            for p in module.parameters()
            if p.grad is not None
        )
    head_grad = sum(
        p.grad.abs().sum().item() for p in model.source_node_head.parameters() if p.grad is not None
    )
    all_finite_grads = all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    return {
        "forward_finite": finite_forward,
        "loss_finite": finite_loss,
        "backward_grads_finite": bool(all_finite_grads),
        "temporal_dynamics_grad_nonzero": dynamics_grad > 0.0,
        "output_head_grad_nonzero": head_grad > 0.0,
    }


def _mini_overfit(model: HydroCore, batch: dict[str, torch.Tensor], steps: int = 20) -> dict[str, Any]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    target = torch.zeros(BATCH, NODES)
    target[:, 0] = 1.0
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        output = model(batch)
        logits = output["source_node_logits"].squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return {
        "optimizer_steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "all_finite": all(torch.isfinite(torch.tensor(losses))),
    }


def _benchmark_latency(
    model: HydroCore, batch: dict[str, torch.Tensor], *, repeats: int = 15, warmup: int = 3
) -> dict[str, Any]:
    model.eval()
    for _ in range(warmup):
        with torch.no_grad():
            model(batch)
    forward_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.no_grad():
            model(batch)
        forward_times.append(time.perf_counter() - start)

    fb_times = []
    for _ in range(repeats):
        for p in model.parameters():
            p.grad = None
        start = time.perf_counter()
        output = model(batch)
        loss = output["source_node_logits"].sum()
        loss.backward()
        fb_times.append(time.perf_counter() - start)

    def stats(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        p90_index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
        return {
            "median_seconds": statistics.median(ordered),
            "p90_seconds": ordered[p90_index],
        }

    return {
        "forward": stats(forward_times),
        "forward_backward": stats(fb_times),
    }


def _sde_mc_stabilization(dynamics: GraphSDEDynamics, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mc_count in (1, 4, 8):
        outputs = []
        start = time.perf_counter()
        for path_index in range(mc_count):
            with torch.no_grad():
                temporal, _ = dynamics(
                    batch["temporal_features"], batch["quality_features"], batch["sensor_mask"],
                    batch["quality_mask"], batch["timestamps"], batch["node_mask"],
                    batch["edge_index"], batch["edge_features"], batch["edge_mask"],
                    seed=20260814 + path_index,
                )
            outputs.append(temporal)
        elapsed = time.perf_counter() - start
        stacked = torch.stack(outputs, dim=0)
        mean_summary = stacked.mean(dim=0).abs().mean().item()
        variance_estimate = stacked.var(dim=0, unbiased=False).mean().item()
        results[f"mc_{mc_count}"] = {
            "runtime_seconds": elapsed,
            "mean_latent_summary": mean_summary,
            "variance_estimate": variance_estimate,
        }
    return results


def _solver_feasibility(matching: dict[str, Any], gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dependency_rows = {
        "GRAPH_ODE": {
            "package": "torchdiffeq",
            "version": getattr(torchdiffeq, "__version__", "0.2.5") if torchdiffeq else None,
            "license": "MIT",
            "why_selected": "canonical differentiable ODE solver for PyTorch; avoids hand-rolling a numerical integrator",
            "dependency_feasible": _TORCHDIFFEQ_IMPORT_ERROR is None,
        },
        "GRAPH_CDE": {
            "package": "torchcde",
            "version": getattr(torchcde, "__version__", "0.2.5") if torchcde else None,
            "license": "Apache-2.0",
            "why_selected": "canonical differentiable CDE library (built on torchdiffeq); provides linear-interpolation control-path construction",
            "dependency_feasible": _TORCHCDE_IMPORT_ERROR is None,
        },
        "GRAPH_SDE": {
            "package": "torchsde",
            "version": getattr(torchsde, "__version__", "0.2.6") if torchsde else None,
            "license": "Apache-2.0",
            "why_selected": "canonical differentiable SDE solver library; provides seeded Brownian-path construction for reproducibility/stochasticity tests",
            "dependency_feasible": _TORCHSDE_IMPORT_ERROR is None,
        },
    }
    verdicts: dict[str, Any] = {"CURRENT": "BASELINE_VALID"}
    for arm in ("GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE"):
        checklist = gates[arm]
        passed = all(checklist.values())
        verdicts[arm] = "PREFLIGHT_PASS" if passed else "PREFLIGHT_BLOCKED"
    return {
        "dependencies": dependency_rows,
        "parameter_matching_summary": matching["arms"],
        "every_novel_arm_within_5_percent": matching["every_novel_arm_within_5_percent"],
        "gate_checklists": gates,
        "arm_verdicts": verdicts,
    }


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for the M9.1 preflight"
    locked_before = locked_test_opened(ROOT)

    matching = _parameter_matching()

    results: dict[str, Any] = {"arms": {}}
    gates: dict[str, dict[str, Any]] = {}

    for arm in ("CURRENT", "GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE"):
        model, dynamics = _build_model(arm, matching)
        finite_batch = _synthetic_batch(seed=42)
        finiteness = _finite_forward_backward(model, finite_batch, sde_seed=42 if arm == "GRAPH_SDE" else None)

        overfit_batch = _synthetic_batch(seed=7)
        overfit = _mini_overfit(model, overfit_batch, steps=20)

        latency_batch = _synthetic_batch(seed=99)
        latency = _benchmark_latency(model, latency_batch)

        arm_result: dict[str, Any] = {
            "finiteness": finiteness,
            "mini_overfit_sanity": overfit,
            "latency": latency,
        }

        if arm == "GRAPH_SDE" and dynamics is not None:
            arm_result["mc_stabilization"] = _sde_mc_stabilization(dynamics, _synthetic_batch(seed=55))

        results["arms"][arm] = arm_result

        if arm != "CURRENT":
            gates[arm] = {
                "parameter_count_within_5_percent": abs(matching["arms"][arm]["delta_percent"]) <= 5.0,
                "forward_finite": finiteness["forward_finite"],
                "backward_finite": finiteness["backward_grads_finite"],
                "loss_finite": finiteness["loss_finite"],
                "temporal_dynamics_gradient_nonzero": finiteness["temporal_dynamics_grad_nonzero"],
                "output_head_gradient_nonzero": finiteness["output_head_grad_nonzero"],
                "mini_overfit_finite": overfit["all_finite"],
            }

    baseline_latency = results["arms"]["CURRENT"]["latency"]["forward"]["median_seconds"]
    for arm in ("GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE"):
        arm_latency = results["arms"][arm]["latency"]["forward"]["median_seconds"]
        ratio = arm_latency / baseline_latency if baseline_latency > 0 else float("inf")
        results["arms"][arm]["latency"]["ratio_to_current"] = ratio
        # Thresholds frozen in docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md
        # Section 11, predeclared before measurement: PRACTICAL <=3x, EXPENSIVE_
        # BUT_TESTABLE 3x-15x, PREFLIGHT_BLOCKED beyond that -- an architecture
        # is never rejected merely for being slower, only for cost pathological
        # enough to threaten the planned three-seed scientific run.
        feasibility = "PRACTICAL" if ratio <= 3.0 else ("EXPENSIVE_BUT_TESTABLE" if ratio <= 15.0 else "PREFLIGHT_BLOCKED")
        results["arms"][arm]["latency"]["feasibility_classification"] = feasibility

    locked_after = locked_test_opened(ROOT)
    results["locked_test_opened_before"] = locked_before
    results["locked_test_opened_after"] = locked_after
    assert not locked_after, "locked test must remain closed for the M9.1 preflight"

    solver_feasibility = _solver_feasibility(matching, gates)
    solver_feasibility["locked_test_opened_before"] = locked_before
    solver_feasibility["locked_test_opened_after"] = locked_after

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "m9-1-preflight-results.json").write_text(json.dumps(results, indent=2, default=str))
    (REPORT_DIR / "m9-1-solver-feasibility.json").write_text(json.dumps(solver_feasibility, indent=2, default=str))
    (REPORT_DIR / "m9-1-parameter-matching.json").write_text(json.dumps(matching, indent=2, default=str))

    print("M9.1 preflight complete.")
    print(json.dumps(solver_feasibility["arm_verdicts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
