"""Canonical identifiers for HydroCore's runtime-consumed neural sub-tasks.

core-issues.txt repair item 8: a checkpoint's ``trained_tasks`` metadata
records which of these tasks were actually optimized against real labels.
Corpus generation (``hydroswarm.training.corpus.scenario_to_example``)
currently only ever writes targets_v2 "sentinel"-category values
(source_node, source_region, event_presence, event_cause, start_time,
duration, relative_strength, sensor_fault, evidence_sufficiency); Scout,
Strategist, and learned-OOD target generators do not exist yet, so their
heads never receive a real gradient and stay at random initialization.
``HybridInferencePipeline`` uses ``trained_tasks`` to force those heads'
outputs out of any runtime decision, leaving deterministic sampling,
planning, and OOD logic authoritative (see repair item 8). Adding a task
here without a corresponding real target generator landing would silently
reopen exactly the bug this module exists to close.
"""

from __future__ import annotations

#: Every runtime task HybridInferencePipeline knows how to gate.
RUNTIME_TASKS = frozenset({"sentinel", "scout", "strategist", "ood"})

#: Tasks the current corpus/training pipeline actually supervises with real
#: labels. Update only alongside a genuine Scout/Strategist/OOD target
#: generator landing (Phase 3+, out of scope for this repair pass).
CORPUS_SUPERVISED_TASKS = frozenset({"sentinel"})


def validate_tasks(tasks: frozenset[str], *, label: str) -> None:
    unknown = tasks - RUNTIME_TASKS
    if unknown:
        raise ValueError(f"{label} contains unknown runtime tasks: {sorted(unknown)}")
