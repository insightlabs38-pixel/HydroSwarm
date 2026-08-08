"""core-issues3.txt Phase 15: produce a real v4 checkpoint identity for the
runtime loader to reconstruct.

Gap this closes: `hydroswarm.training.checkpoint_identity`'s entire
save/load/governance machinery (Phase 9.1/9.2, `core-issues4.txt` Section
B/C) has existed and been unit-tested in isolation, but no training script
in this project (`run_stage_f_training.py`, `train_strategist_heads.py`,
etc.) has ever actually called `save_v4_checkpoint` -- every real trained
checkpoint on disk is a plain `ExperimentRegistry`/`Trainer` export with no
`checkpoint_identity.json`. Phase 15's runtime loader has nothing real to
load. This script builds one, post-hoc, from the best available already-
trained checkpoint (Stage F `no_adapters`-seed20260810 -- Stage F's own
measured, direction-consistent winner) and this pass's Phase 13/14
evaluation results.

**This is deliberately NOT a promotion.** `trained_outputs` records every
output that received real governed supervision; `validated_outputs` and
`runtime_enabled_outputs` are set MUCH more conservatively, straight from
`reports/results/v4/phase14-promotion-gates.md`'s own per-output verdicts
-- Scout's entire head group and `ood_category` are deliberately excluded
from all three sets (Scout's own promotion requirement failed outright;
`ood_category` never received a real training gradient), and Strategist's
outputs are excluded from `runtime_enabled_outputs` specifically because
Phase 14 found only one trained seed (gate 7 requires >= 2). Nothing here
changes which model the live production pipeline (`runtime/defaults.py`'s
`DefaultPipelineFactory`, still `models/hydrocore-s-learning-v1.safetensors`)
actually serves -- see this project's own restriction against overwriting
`data/learning-v1`/current checkpoints.

The optimizer/scheduler `save_v4_checkpoint` requires are freshly
constructed here (matching `hydroswarm.training.trainer.Trainer`'s own
AdamW + LambdaLR construction), NOT the real Stage-F run's actual optimizer
state, which was never captured by the plain export path this checkpoint
originally came from -- documented honestly in `resolved_training_config`
below; a caller that needs a genuine resumable optimizer state must resume
from the original Stage-F run directory instead, not this identity-only
reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from hydroswarm.model import HydroCore
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.training.checkpoint_identity import build_checkpoint_identity, save_v4_checkpoint
from hydroswarm.training.trainer import TrainingConfig, _scheduler

SHARED_MODEL_CONFIG: dict[str, Any] = {
    "prior_mode": "feature_only",
    "event_control_heads": True,
    "scout_control_heads": True,
    "strategist_mode": "candidate_conditioned",
    "action_vocabulary_size": 9,
    "consequence_prescreening_heads": True,
    "ood_category_head": True,
}

#: Every output that received real governed supervision this run (matches
#: Stage F's own "21 real tasks" plus ood_class, which is present in the
#: architecture even though it never reached train-split gradient -- see
#: TRAINED vs VALIDATED distinction below).
TRAINED_OUTPUTS: frozenset[str] = frozenset({
    "source_node", "source_region", "start_time", "duration", "relative_strength",
    "event_presence", "event_cause", "sensor_fault", "evidence_sufficiency",
    "sample_node", "information_gain", "candidate_reduction", "should_continue_sampling",
    "plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
    "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
    "next_step",
    "sensor_reconstruction", "travel_time",
})

#: reports/results/v4/phase14-promotion-gates.md's own per-output verdicts,
#: restricted to outputs that PASSED gates 1-4 with real evidence.
#: Excludes: sensor_fault (degenerate eval population, gate 4
#: indeterminate), the whole Scout group (gate 4 FAILS -- learned_scout is
#: worse than random), ood_category (gate 3 FAILS -- zero real gradient),
#: action_logits/action_pointer_logits (not in TRAINED_OUTPUTS at all under
#: strategist_mode=candidate_conditioned).
VALIDATED_OUTPUTS: frozenset[str] = frozenset({
    "event_presence", "event_cause", "start_time", "relative_strength",
    "evidence_sufficiency", "next_step",
    "plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
    "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
})

#: Of VALIDATED_OUTPUTS, only the Sentinel/control-family outputs are
#: promoted to runtime_enabled -- Strategist's outputs are held back
#: because Phase 14's gate 7 (>= 2 finalist seeds) is not yet satisfied
#: (only one seed, v4-strategist-heads-v4corpus-corrected, has been
#: trained). `duration` stays diagnostic-only (Phase 14: "flag as
#: lower-confidence", accuracy 0.50 vs ~33% chance -- real signal but the
#: weakest of the three profile heads).
RUNTIME_ENABLED_OUTPUTS: frozenset[str] = frozenset({
    "event_presence", "event_cause", "relative_strength",
    "evidence_sufficiency", "next_step",
})

#: diagnostic_only_outputs is reserved for outputs with NO real governed
#: supervision at all (output_governance.validate_output_governance
#: forbids overlap with trained_outputs) -- duration/sensor_fault/
#: source_region WERE really trained (see TRAINED_OUTPUTS), just not
#: validated/runtime-enabled per Phase 14's findings, so they simply stay
#: out of VALIDATED_OUTPUTS/RUNTIME_ENABLED_OUTPUTS above instead.
DIAGNOSTIC_ONLY_OUTPUTS: frozenset[str] = frozenset()
TRAINING_ONLY_OUTPUTS: frozenset[str] = frozenset({"sensor_reconstruction", "travel_time"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path("experiments/runs/stage-f/no_adapters-seed20260810/20260808T041727Z-de5f4b0e/model-export.safetensors"),
    )
    parser.add_argument("--use-adapters", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model = HydroCore.from_variant("small", use_adapters=args.use_adapters, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(str(args.source_checkpoint), device="cpu"), strict=True)
    model.eval()

    identity = build_checkpoint_identity(
        model,
        normalization_hash="none",  # joint-v4 tensors are pre-normalized at corpus-build time, not by a runtime feature-builder artifact
        fusion_policy_hash="fixed_weight_fusion-v1:neural_weight=0.6",
        source_corpus_manifest_hashes=("data/learning-v2/cycle-b2-joint-v4",),
        trained_outputs=TRAINED_OUTPUTS,
        validated_outputs=VALIDATED_OUTPUTS,
        runtime_enabled_outputs=RUNTIME_ENABLED_OUTPUTS,
        diagnostic_only_outputs=DIAGNOSTIC_ONLY_OUTPUTS,
        training_only_outputs=TRAINING_ONLY_OUTPUTS,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = _scheduler(optimizer, TrainingConfig(seed=20260810), total_steps=1)

    output_path = save_v4_checkpoint(
        args.output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=16,
        global_step=-1,  # real per-step count from the original Stage F run was not preserved by its plain-export path
        best_validation_loss=5.3512,  # stage-f-adapters-comparison.json: no_adapters-seed20260810
        identity=identity,
        resolved_training_config={
            "note": "identity constructed post-hoc from an already-trained plain export; "
            "optimizer/scheduler state above is FRESH, not the real Stage F run's own state -- "
            "see this script's module docstring",
            "source_checkpoint": str(args.source_checkpoint),
            "source_run": "scripts/run_stage_f_training.py, arm=no_adapters, seed=20260810",
            "shared_model_config": SHARED_MODEL_CONFIG,
            "use_adapters": args.use_adapters,
        },
        dataset_manifest_hashes={"joint_v4_corpus": "data/learning-v2/cycle-b2-joint-v4"},
        task_weights={name: 1.0 for name in TRAINED_OUTPUTS},
        calibration_hash=None,
    )

    print(f"wrote v4 checkpoint identity to {output_path}")
    print(f"identity fingerprint: {identity.fingerprint()}")
    print(f"runtime_enabled_outputs: {sorted(RUNTIME_ENABLED_OUTPUTS)}")
    print(f"feature_schema_hash: {DEFAULT_FEATURE_SCHEMA.fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
