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
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from hydroswarm.inference.fusion import DYNAMIC_TRUST_FUSION_CONFIG
from hydroswarm.model import HydroCore
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.runtime.v4_normalization import load_runtime_normalization_bundle
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
#: Stage F's own 19 real gradient-receiving tasks plus ood_class, which is
#: present in the architecture even though it never reached train-split
#: gradient -- see TRAINED vs VALIDATED distinction below).
#:
#: core-issues5.txt delta item 4 (P0 governance fix), Section C: does NOT
#: include "sensor_reconstruction"/"travel_time" -- SHARED_MODEL_CONFIG
#: above never sets `auxiliary_heads=True` (matching
#: scripts/run_stage_f_training.py's own real model config, which also
#: never sets it, so it took HydroCore's AUXILIARY_HEADS_DEFAULT=False),
#: which means the real Stage-F run's model never even PHYSICALLY
#: CONSTRUCTED sensor_reconstruction_head/travel_time_head
#: (hydroswarm.model.core.HydroCore.__init__: those heads only exist
#: `if self.auxiliary_heads`) -- claiming them "trained" was a real,
#: previously-undetected defect (claiming an output was trained merely
#: because the architecture COULD support it, not because the selected
#: run actually did), confirmed by their total absence from Phase 13's own
#: metrics report and Stage F's comparison results. main() asserts this
#: stays true against the actual constructed model rather than trusting
#: this comment alone.
TRAINED_OUTPUTS: frozenset[str] = frozenset({
    "source_node", "source_region", "start_time", "duration", "relative_strength",
    "event_presence", "event_cause", "sensor_fault", "evidence_sufficiency",
    "sample_node", "information_gain", "candidate_reduction", "should_continue_sampling",
    "plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
    "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
    "next_step",
})

#: core-issues5.txt delta item 4, Section C: heads that exist in the
#: architecture (auxiliary_heads=True would build them) but were NOT
#: physically constructed by this run's real model config -- must never
#: appear in TRAINED_OUTPUTS/TRAINING_ONLY_OUTPUTS/VALIDATED_OUTPUTS/
#: RUNTIME_ENABLED_OUTPUTS for a checkpoint identity built from this
#: SHARED_MODEL_CONFIG. Checked for real in main(), not just documented
#: here.
AUXILIARY_HEAD_OUTPUTS: frozenset[str] = frozenset({"sensor_reconstruction", "travel_time", "future_concentration"})

#: reports/results/v4/phase14-promotion-gates.md's own per-output verdicts,
#: restricted to outputs that PASSED gates 1-4 with real evidence.
#: Excludes: sensor_fault (degenerate eval population, gate 4
#: indeterminate), the whole Scout group (gate 4 FAILS -- learned_scout is
#: worse than random), ood_category (gate 3 FAILS -- zero real gradient),
#: action_logits/action_pointer_logits (not in TRAINED_OUTPUTS at all under
#: strategist_mode=candidate_conditioned).
#:
#: core-issues5.txt delta item 2: "source_node" included -- Phase 14's own
#: table records it as PASS on every gate ("top1 0.72 vs. classical-only
#: baseline already in production fusion"; "0.7247/0.7149 Stage-A;
#: 0.7205/0.7331 Stage-F, all close"), with a note to "re-verify under v4
#: metadata in Phase 15." Leaving it out of VALIDATED_OUTPUTS while
#: hydroswarm.inference.pipeline.HybridInferencePipeline.analyze()
#: unconditionally consumed source_node_logits regardless of
#: runtime_enabled_outputs was a real metadata/behavior contradiction, not
#: a deliberate exclusion -- see that method's own governance comment.
VALIDATED_OUTPUTS: frozenset[str] = frozenset({
    "source_node",
    "event_presence", "event_cause", "start_time", "relative_strength",
    "evidence_sufficiency", "next_step",
    "plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
    "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
})

#: Of VALIDATED_OUTPUTS, only the Sentinel/control-family outputs (plus
#: source_node -- delta item 2) are promoted to runtime_enabled --
#: Strategist's outputs are held back because Phase 14's gate 7 (>= 2
#: finalist seeds) is not yet satisfied (only one seed,
#: v4-strategist-heads-v4corpus-corrected, has been trained). `duration`
#: stays diagnostic-only (Phase 14: "flag as lower-confidence", accuracy
#: 0.50 vs ~33% chance -- real signal but the weakest of the three profile
#: heads).
RUNTIME_ENABLED_OUTPUTS: frozenset[str] = frozenset({
    "source_node",
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
#: core-issues5.txt delta item 4, Section C: empty, not
#: {"sensor_reconstruction", "travel_time"} -- see AUXILIARY_HEAD_OUTPUTS'
#: own comment above for why those heads were never even physically
#: constructed by this run's real model config, let alone trained.
TRAINING_ONLY_OUTPUTS: frozenset[str] = frozenset()


def _real_source_corpus_manifest_hashes(joint_corpus_dir: Path) -> tuple[str, ...]:
    """core-issues5.txt delta item 4, Section B: real content
    hashes/fingerprints for the joint-v4 corpus this checkpoint was
    actually trained on -- NOT the bare path string the identity
    previously recorded under this field. Reads
    `<joint_corpus_dir>/checksums.json`'s own whole-corpus
    `dataset_fingerprint_sha256` plus every individual population's
    `index_sha256` from `<joint_corpus_dir>/source-manifest-hashes.json`
    (both already real, already-committed governed artifacts --
    scripts/build_stage_f_joint_corpus.py's own output -- not computed
    fresh here)."""

    checksums = json.loads((joint_corpus_dir / "checksums.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((joint_corpus_dir / "source-manifest-hashes.json").read_text(encoding="utf-8"))

    hashes: list[str] = [checksums["dataset_fingerprint_sha256"]]

    def walk(node: Any) -> None:
        if isinstance(node, dict) and "index_sha256" in node:
            hashes.append(node["index_sha256"])
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(source_manifest)
    return tuple(dict.fromkeys(hashes))  # de-duplicated, order-preserving


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path("experiments/runs/stage-f/no_adapters-seed20260810/20260808T041727Z-de5f4b0e/model-export.safetensors"),
    )
    parser.add_argument("--use-adapters", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810"))
    parser.add_argument(
        "--normalization-dir",
        type=Path,
        default=Path("data/learning-v2/cycle-b2/normalization"),
        help=(
            "the real, committed, train-split-fit node/edge NormalizationStats artifact "
            "cycle-b2/tensors-normalized (and therefore cycle-b2-joint-v4, Stage F's actual "
            "training corpus) was built from -- core-issues5.txt Section 3"
        ),
    )
    parser.add_argument(
        "--joint-corpus-dir",
        type=Path,
        default=Path("data/learning-v2/cycle-b2-joint-v4"),
        help=(
            "Stage F's actual training corpus -- source of real content hashes for "
            "source_corpus_manifest_hashes/dataset_manifest_hashes (core-issues5.txt delta item 4, "
            "Section B), not merely this path string"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model = HydroCore.from_variant("small", use_adapters=args.use_adapters, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(str(args.source_checkpoint), device="cpu"), strict=True)
    model.eval()

    # core-issues5.txt delta item 4, Section C: fail loudly if
    # TRAINED_OUTPUTS/TRAINING_ONLY_OUTPUTS ever claim an auxiliary output
    # the ACTUAL constructed model (not merely SHARED_MODEL_CONFIG's
    # intent) does not physically have -- checked against the real model
    # instance, not just the config dict, so a HydroCore default change
    # cannot silently reintroduce this class of defect unnoticed.
    claimed_auxiliary_outputs = AUXILIARY_HEAD_OUTPUTS & (TRAINED_OUTPUTS | TRAINING_ONLY_OUTPUTS)
    if claimed_auxiliary_outputs and not model.auxiliary_heads:
        raise SystemExit(
            f"refusing to build a checkpoint identity that claims {sorted(claimed_auxiliary_outputs)} as "
            "trained/training-only while the constructed model has auxiliary_heads=False (those heads were "
            "never physically constructed, let alone supervised) -- core-issues5.txt delta item 4, Section C"
        )

    # core-issues5.txt Section 3 (P0 blocker): this checkpoint's training
    # corpus (cycle-b2-joint-v4) really was built with governed node/edge
    # normalization applied (see scripts/rebuild_normalized_shards.py) --
    # "none" was a real defect (train/serve normalization skew), not an
    # honest description of an unnormalized model. Load the real artifact
    # and record its actual fingerprint so the runtime loader can verify it.
    normalization_bundle = load_runtime_normalization_bundle(args.normalization_dir)

    # core-issues5.txt delta item 4, Section B: real content hashes, not
    # the corpus directory path.
    source_corpus_manifest_hashes = _real_source_corpus_manifest_hashes(args.joint_corpus_dir)
    dataset_fingerprint = source_corpus_manifest_hashes[0]

    identity = build_checkpoint_identity(
        model,
        normalization_hash=normalization_bundle.fingerprint,
        # core-issues5.txt delta item 4, Section A: the real canonical
        # dynamic-trust-fusion policy identity V4 serving actually uses
        # (hydroswarm.inference.fusion.DYNAMIC_TRUST_FUSION_CONFIG, the
        # same constant hydroswarm.runtime.v4_defaults.V4PipelineFactory
        # and HybridInferencePipeline.fusion_config_hash are keyed against)
        # -- NOT a hand-written fixed_weight_fusion-v1 string describing a
        # fusion policy V4 serving does not run. One canonical
        # constant/function, not two independently-maintained literals.
        fusion_policy_hash=DYNAMIC_TRUST_FUSION_CONFIG,
        source_corpus_manifest_hashes=source_corpus_manifest_hashes,
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
            # Paths recorded separately from the real hashes above
            # (core-issues5.txt delta item 4, Section B: "Record paths
            # separately if useful").
            "source_corpus_path": str(args.joint_corpus_dir),
            "normalization_dir": str(args.normalization_dir),
            "shared_model_config": SHARED_MODEL_CONFIG,
            "use_adapters": args.use_adapters,
        },
        dataset_manifest_hashes={"joint_v4_corpus": dataset_fingerprint},
        task_weights={name: 1.0 for name in TRAINED_OUTPUTS},
        calibration_hash=None,
    )

    print(f"wrote v4 checkpoint identity to {output_path}")
    print(f"identity fingerprint: {identity.fingerprint()}")
    print(f"runtime_enabled_outputs: {sorted(RUNTIME_ENABLED_OUTPUTS)}")
    print(f"feature_schema_hash: {DEFAULT_FEATURE_SCHEMA.fingerprint}")
    print(f"normalization_hash: {normalization_bundle.fingerprint} (from {args.normalization_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
