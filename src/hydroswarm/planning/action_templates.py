"""Canonical bounded action-template vocabulary (core-issues3.txt Phase 3.5).

hydroswarm.planning.response.generate_response_plans() produces at most
this set, in this exact append order (its own internal `templates` list
literal). Centralized here as the single source of truth for anything that
needs the template count, ordering, or a stable schema hash -- target
encoding, model construction, runtime mapping, checkpoint metadata,
evaluation -- so a future change to response.py's template list cannot
silently drift out of sync with any of those consumers.

Previously duplicated (correctly, but only as a comment-documented
mirror) in hydroswarm.training.strategist_trajectory; that module now
imports from here instead of maintaining its own copy.
"""

from __future__ import annotations

import hashlib
import json

ACTION_TEMPLATES: tuple[str, ...] = (
    "NO_ACTION",
    "ISOLATE_SOURCE",
    "FLUSH_DOWNSTREAM",
    "ISOLATE_AND_FLUSH",
    "PROTECT_CRITICAL",
    "INCREASE_MONITORING",
    "REQUEST_SAMPLE",
    "WAIT_OBSERVE",
    "ALTERNATE_VALVE_CUT",
)

ACTION_TEMPLATE_COUNT = len(ACTION_TEMPLATES)
ACTION_TEMPLATE_INDEX: dict[str, int] = {name: index for index, name in enumerate(ACTION_TEMPLATES)}

#: Templates whose primary action targets a link (CLOSE_PIPE/OPEN_PIPE),
#: not a node. Anything not listed here that does have a target targets a
#: node; NO_ACTION and WAIT_OBSERVE have no target at all. Kept as an
#: explicit set (not inferred from ActionType at label-generation time
#: only) so target-space resolution is table-driven and testable on its
#: own, independent of any one plan's actual generated actions.
LINK_TARGET_TEMPLATES = frozenset({"ISOLATE_SOURCE", "ISOLATE_AND_FLUSH", "ALTERNATE_VALVE_CUT"})

#: Deterministic fingerprint of the vocabulary itself -- changes if the
#: template set or order ever changes, so checkpoint/corpus metadata can
#: detect a vocabulary mismatch instead of silently misinterpreting
#: target_pointer/action_template indices encoded under a different order.
ACTION_TEMPLATE_SCHEMA_HASH = hashlib.sha256(
    json.dumps(list(ACTION_TEMPLATES), separators=(",", ":")).encode()
).hexdigest()
