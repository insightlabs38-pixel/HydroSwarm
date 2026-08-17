"""Milestone 10.2 Scout refit -- Part 5: the frozen refit protocol as
importable, hashable DATA (not prose). `docs/evaluation/
HYDROCORE_V5_M10_2_SCOUT_REFIT_PROTOCOL.md` is the authoritative prose
version; this module is what the actual Level-A/B execution scripts import,
so the code that runs and the frozen document that authorizes it cannot
silently drift apart -- `protocol_hash()` gives every execution artifact a
concrete value to cite.

Frozen before any Level-A result is inspected. Do not edit after Level-A
executes; if a genuine defect is found, fix the executing code to correctly
implement this SAME frozen data, or freeze a new, separately-dated
amendment -- never silently edit this module's values after seeing results.
"""

from __future__ import annotations

import hashlib
import json

TEACHER_CHECKPOINT_SHA256: dict[int, str] = {
    20260814: "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
    31874: "527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332",
    20260815: "b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c",
}

TRAINING_STATE_SCHEMA_VERSION = "scout-training-state-v1"
TARGET_SCHEMA_VERSION = "scout-target-v1"

FAMILY = "golden-reference"
TRAIN_SEED_BASE = 1_200_000_000
TRAIN_COUNT = 250
VALIDATION_SEED_BASE = 1_200_100_000
VALIDATION_COUNT = 100
SOURCE_ROUND_ROBIN = True

DEPTH = 25
MAXIMUM_SAMPLES = 3
NOISE_SCALE_MG_L = 0.5

LEVEL_A_PARAMETER_ALLOWLIST: tuple[str, ...] = (
    "role_projection.weight", "role_projection.bias",
    "residual_projection.weight", "residual_projection.bias",
    "sample_node_head.network.0.weight", "sample_node_head.network.0.bias",
    "sample_node_head.network.1.weight", "sample_node_head.network.1.bias",
    "information_gain_head.0.weight",
    "information_gain_head.1.weight", "information_gain_head.1.bias",
    "candidate_reduction_head.0.weight",
    "candidate_reduction_head.1.weight", "candidate_reduction_head.1.bias",
    "should_continue_sampling_head.network.0.weight", "should_continue_sampling_head.network.0.bias",
    "should_continue_sampling_head.network.1.weight", "should_continue_sampling_head.network.1.bias",
)

LEVEL_B_ADDITIONAL_PARAMETER_PREFIXES: tuple[str, ...] = ("backbone.3.", "final_norm.")

OPTIMIZER = "Adam"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 8
EPOCHS = 20
CHECKPOINT_SELECTION = "FINAL_EPOCH"

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.90
BOOTSTRAP_SEED = 20260819

GATE_MIN_SUPPORT = 20

SCOUT_TASKS: tuple[str, ...] = ("sample_node", "information_gain", "candidate_reduction", "should_continue_sampling")


def _payload() -> dict[str, object]:
    return {
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "training_state_schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "family": FAMILY,
        "train_seed_base": TRAIN_SEED_BASE,
        "train_count": TRAIN_COUNT,
        "validation_seed_base": VALIDATION_SEED_BASE,
        "validation_count": VALIDATION_COUNT,
        "source_round_robin": SOURCE_ROUND_ROBIN,
        "depth": DEPTH,
        "maximum_samples": MAXIMUM_SAMPLES,
        "noise_scale_mg_l": NOISE_SCALE_MG_L,
        "level_a_parameter_allowlist": list(LEVEL_A_PARAMETER_ALLOWLIST),
        "level_b_additional_parameter_prefixes": list(LEVEL_B_ADDITIONAL_PARAMETER_PREFIXES),
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_ci": BOOTSTRAP_CI,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gate_min_support": GATE_MIN_SUPPORT,
        "scout_tasks": list(SCOUT_TASKS),
    }


def protocol_hash() -> str:
    encoded = json.dumps(_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def to_json_doc() -> dict[str, object]:
    return {"kind": "M10_2_REFIT_PROTOCOL", "milestone": "M10.2-refit", **_payload(), "protocol_hash": protocol_hash()}
