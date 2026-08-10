"""SUB-4 (submission.txt SS6): build the governed REFERENCE INCIDENT
artifact from the real, frozen, WNTR-backed golden workflow.

Turns `GoldenScenarioRunner`'s 21-event chain into a progressive narrative
of milestone snapshots a frontend can step through without ever guessing
"if event N then hide/reveal X" against the monolithic completed object.
Each milestone's `incident_view` is built directly from the real domain
objects `GoldenScenarioRunner` already produced (OperationalPlan/
PlanVerification dumps, measured probabilities, real event hashes) --
never fabricated -- and is stage-correct by construction: a milestone's
view only includes fields whose value the golden workflow had actually
computed by that milestone's last event, everything else stays `None`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "hydroswarm-reference-incident-v1"
REFERENCE_ID = "reference-incident-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Milestone:
    index: int
    milestone_id: str
    label: str
    event_sequence_start: int
    event_sequence_end: int
    auto_advance: bool
    pause_reason: str | None
    #: SUB-12.1 P1: generalizes "what happens when a judge/operator acts on
    #: this pause" beyond the single hard-coded "it must be approval"
    #: assumption -- there are two distinct pauses in the real reference
    #: narrative (collecting the next evidence sample, then approving the
    #: verified plan), and a future pass may add more. None for every
    #: auto_advance milestone; always paired with pause_action_label.
    pause_action: str | None
    pause_action_label: str | None
    narrative: str
    highlight: str
    incident_view: dict[str, Any] = field(default_factory=dict)


def _controller_state_after(events: list[dict[str, Any]], end: int) -> str:
    """The real FSM `from_state` recorded on the last event in this
    milestone's range -- never a hand-picked label. `from_state` is the
    state the controller was in when it processed that event."""
    (event,) = [item for item in events if item["sequence"] == end]
    return str(event["from_state"])


def build_reference_incident_artifact(
    golden_result: dict[str, Any],
    *,
    generator: str,
    source_commit: str,
    network_topology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full reference-incident artifact (pre-hash) from a real
    `GoldenScenarioRunner().run()` result. Raises `AssertionError` if a
    constructed milestone would leak data not yet available at that stage
    -- see `_assert_no_future_leakage` -- rather than silently shipping a
    narrative that lies about when evidence became known.

    `network_topology` is the real frozen golden network's node/link/
    coordinate metadata (`hydroswarm.networks.network_topology_metadata`),
    identical in shape to what a live network import computes. Optional
    only so callers that don't need a renderable map (e.g. hash/consistency
    checks) can omit it; the frontend REFERENCE INCIDENT experience needs
    it to draw the operational map without borrowing the hand-authored
    DEMO_FALLBACK fixture's topology."""

    events = golden_result["workflow"]["events"]
    localization = golden_result["localization"]
    plans = golden_result["plans"]

    unsafe_plan_dump = plans["unsafe"]["plan"]
    safe_plan_dump = plans["safe"]["plan"]
    unsafe_verification_dump = plans["unsafe"]["verification"]
    safe_verification_dump = plans["safe"]["verification"]
    unsafe_plan_id = unsafe_plan_dump["plan_id"]
    safe_plan_id = safe_plan_dump["plan_id"]

    base_view = {
        # Every generated plan carries the real incident_id (OperationalPlan
        # field) -- reused here rather than re-deriving/guessing it, since
        # no top-level event payload carries it.
        "incident_id": safe_plan_dump["incident_id"],
        "network_id": "golden-network-v1",
        "ood_level": "NORMAL",
    }

    def view(**overrides: Any) -> dict[str, Any]:
        merged = dict(base_view)
        merged.update(
            {
                "candidates": None,
                "candidate_region": None,
                "evidence_sufficient": None,
                "recommended_sample": None,
                "sample_observation": None,
                "plans": None,
                "selected_plan_id": None,
                "approved_plan_id": None,
                "approval_pending": False,
                "final_event_hash": None,
            }
        )
        merged.update(overrides)
        return merged

    milestones: list[Milestone] = []

    def add(
        milestone_id: str,
        label: str,
        start: int,
        end: int,
        *,
        auto_advance: bool,
        pause_reason: str | None,
        narrative: str,
        highlight: str,
        incident_view: dict[str, Any],
        pause_action: str | None = None,
        pause_action_label: str | None = None,
    ) -> None:
        assert (pause_action is None) == (pause_action_label is None), (
            "pause_action and pause_action_label must be set together"
        )
        assert auto_advance or pause_action is not None, (
            f"milestone {milestone_id!r} pauses (auto_advance=False) but declares no "
            "pause_action -- the frontend would have nothing to tell the operator to do"
        )
        milestones.append(
            Milestone(
                index=len(milestones),
                milestone_id=milestone_id,
                label=label,
                event_sequence_start=start,
                event_sequence_end=end,
                auto_advance=auto_advance,
                pause_reason=pause_reason,
                pause_action=pause_action,
                pause_action_label=pause_action_label,
                narrative=narrative,
                highlight=highlight,
                incident_view=dict(
                    incident_view, controller_state=_controller_state_after(events, end)
                ),
            )
        )

    add(
        "alert",
        "Incident detected",
        0,
        2,
        auto_advance=True,
        pause_reason=None,
        narrative="A contamination alert opens the incident. The network model and "
        "incident record load and validate before any analysis runs.",
        highlight="incident_opened",
        incident_view=view(),
    )
    add(
        "initial_uncertainty",
        "Initial source uncertainty",
        3,
        6,
        auto_advance=True,
        pause_reason=None,
        narrative="Data quality passes and the hydraulic state is estimated. With only "
        "the initial sensor reading, every candidate source is equally likely.",
        highlight="broad_candidate_set",
        incident_view=view(
            candidates=localization["initial_probabilities"],
            candidate_region=localization["initial_credible_region"],
            evidence_sufficient=False,
        ),
    )
    add(
        "evidence_insufficient",
        "Evidence insufficient to plan",
        7,
        7,
        auto_advance=True,
        pause_reason=None,
        narrative="The current evidence cannot narrow the candidate set enough to "
        "safely plan a response. Active sampling is required before proceeding.",
        highlight="evidence_gate_blocks_planning",
        incident_view=view(
            candidates=localization["initial_probabilities"],
            candidate_region=localization["initial_credible_region"],
            evidence_sufficient=False,
        ),
    )
    add(
        "sample_recommended",
        "Next sample selected",
        8,
        8,
        auto_advance=False,
        pause_reason="HydroSwarm pauses here rather than assuming a sample was taken -- "
        "evidence collection is a real, deliberate step, not a foregone conclusion.",
        pause_action="COLLECT_REFERENCE_SAMPLE",
        pause_action_label="Collect reference sample",
        narrative=f"Deterministic active sampling selects {localization['sample_node']} -- the "
        "measured signature split with the largest expected information gain.",
        highlight="sample_recommendation",
        incident_view=view(
            candidates=localization["initial_probabilities"],
            candidate_region=localization["initial_credible_region"],
            evidence_sufficient=False,
            recommended_sample={
                "node_id": localization["sample_node"],
                "expected_information_gain_bits": localization["information_gain_by_node_bits"][
                    localization["sample_node"]
                ],
            },
        ),
    )
    add(
        "sample_received",
        "Sample arrives",
        9,
        9,
        auto_advance=True,
        pause_reason=None,
        narrative=f"A real measurement arrives from {localization['sample_node']}. The "
        "posterior has not been recomputed yet -- candidate probabilities are still the "
        "pre-sample values.",
        highlight="raw_sample_received",
        incident_view=view(
            candidates=localization["initial_probabilities"],
            candidate_region=localization["initial_credible_region"],
            evidence_sufficient=False,
            recommended_sample={
                "node_id": localization["sample_node"],
                "expected_information_gain_bits": localization["information_gain_by_node_bits"][
                    localization["sample_node"]
                ],
            },
            sample_observation={
                "node_id": localization["sample_node"],
                "concentration_mg_l": localization["sample_observed_mg_l"],
            },
        ),
    )
    add(
        "posterior_contracted",
        "Posterior contracts",
        10,
        12,
        auto_advance=True,
        pause_reason=None,
        narrative="The measurement updates the posterior. The candidate region "
        f"contracts from {len(localization['initial_credible_region'])} nodes to "
        f"{len(localization['posterior_credible_region'])}, and evidence is now sufficient to plan.",
        highlight="posterior_contraction",
        incident_view=view(
            candidates=localization["posterior_probabilities"],
            candidate_region=localization["posterior_credible_region"],
            evidence_sufficient=True,
            sample_observation={
                "node_id": localization["sample_node"],
                "concentration_mg_l": localization["sample_observed_mg_l"],
            },
        ),
    )
    add(
        "plans_generated",
        "Response plans generated",
        13,
        13,
        auto_advance=True,
        pause_reason=None,
        narrative="Two candidate response plans are generated. Neither has undergone "
        "exact WNTR/EPANET verification yet.",
        highlight="unverified_plan_candidates",
        incident_view=view(
            candidates=localization["posterior_probabilities"],
            candidate_region=localization["posterior_credible_region"],
            evidence_sufficient=True,
            plans=[
                {"plan": unsafe_plan_dump, "verification": None},
                {"plan": safe_plan_dump, "verification": None},
            ],
        ),
    )
    add(
        "unsafe_plan_rejected",
        "Unsafe plan rejected on verification",
        14,
        15,
        auto_advance=True,
        pause_reason=None,
        narrative="Exact WNTR/EPANET verification rejects the feeder-closure plan for "
        f"violating a hard constraint ({', '.join(unsafe_verification_dump['rejection_codes'])}). "
        "The alternative plan has not been verified yet.",
        highlight="unsafe_plan_rejected",
        incident_view=view(
            candidates=localization["posterior_probabilities"],
            candidate_region=localization["posterior_credible_region"],
            evidence_sufficient=True,
            plans=[
                {"plan": unsafe_plan_dump, "verification": unsafe_verification_dump},
                {"plan": safe_plan_dump, "verification": None},
            ],
        ),
    )
    add(
        "safe_plan_verified",
        "Safe plan verified",
        16,
        17,
        auto_advance=True,
        pause_reason=None,
        narrative="Exact verification confirms the flush plan is safe: no constraint "
        "violations, and it is now eligible for human approval.",
        highlight="safe_plan_verified",
        incident_view=view(
            candidates=localization["posterior_probabilities"],
            candidate_region=localization["posterior_credible_region"],
            evidence_sufficient=True,
            plans=[
                {"plan": unsafe_plan_dump, "verification": unsafe_verification_dump},
                {"plan": safe_plan_dump, "verification": safe_verification_dump},
            ],
        ),
    )
    add(
        "human_approval_boundary",
        "Awaiting human approval",
        18,
        18,
        auto_advance=False,
        pause_reason="HydroSwarm never executes a response autonomously -- a verified "
        "plan pauses here until a human operator approves it.",
        pause_action="APPROVE_REFERENCE_PLAN",
        pause_action_label="Approve plan",
        narrative="The verified flush plan is selected and presented for approval. "
        "No action has been taken.",
        highlight="human_approval_required",
        incident_view=view(
            candidates=localization["posterior_probabilities"],
            candidate_region=localization["posterior_credible_region"],
            evidence_sufficient=True,
            plans=[
                {"plan": unsafe_plan_dump, "verification": unsafe_verification_dump},
                {"plan": safe_plan_dump, "verification": safe_verification_dump},
            ],
            selected_plan_id=safe_plan_id,
            approval_pending=True,
        ),
    )
    add(
        "completed",
        "Response approved and simulated",
        19,
        20,
        auto_advance=True,
        pause_reason=None,
        narrative="A human operator approves the verified plan. The approved response "
        "is simulated end to end, and the full, hash-chained event ledger is available "
        "for replay.",
        highlight="incident_closed",
        incident_view=view(
            candidates=localization["posterior_probabilities"],
            candidate_region=localization["posterior_credible_region"],
            evidence_sufficient=True,
            plans=[
                {"plan": unsafe_plan_dump, "verification": unsafe_verification_dump},
                {"plan": safe_plan_dump, "verification": safe_verification_dump},
            ],
            selected_plan_id=safe_plan_id,
            approved_plan_id=safe_plan_id,
            approval_pending=False,
            final_event_hash=golden_result["workflow"]["final_event_hash"],
        ),
    )

    assert milestones[-1].event_sequence_end == golden_result["workflow"]["event_count"] - 1, (
        "milestones must cover every event through the workflow's last event -- "
        "otherwise the reference narrative silently truncates the real golden trace"
    )
    _assert_contiguous_coverage(milestones, golden_result["workflow"]["event_count"])
    _assert_no_future_leakage(milestones, safe_plan_id=safe_plan_id, unsafe_plan_id=unsafe_plan_id)
    _assert_pauses_are_well_formed(milestones)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "reference_id": REFERENCE_ID,
        "title": "HydroSwarm reference incident",
        "description": "A checksummed replay of HydroSwarm's frozen, WNTR-backed golden "
        "incident workflow -- source localization, active sampling, plan verification, "
        "and human approval, end to end.",
        "generator": generator,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "source_artifacts": sorted(golden_result["fixture_manifest"]["artifacts"]),
        "source_artifact_hashes": {
            name: entry["sha256"] for name, entry in golden_result["fixture_manifest"]["artifacts"].items()
        },
        # SUB-12.1 P0 #2A: the real frozen network file's own hash, distinct
        # from golden_result_hash below (a hash of the *entire* golden
        # result payload -- localization, plans, consequences, events --
        # not the network specifically). The frontend's provenance.networkHash
        # must use this field, never golden_result_hash, or it mislabels one
        # artifact's hash as another's.
        "network_sha256": golden_result["fixture_manifest"]["artifacts"]["golden_network.inp"]["sha256"],
        "golden_result_hash": golden_result["result_sha256"],
        "final_event_hash": golden_result["workflow"]["final_event_hash"],
        "event_count": golden_result["workflow"]["event_count"],
        "network_topology": network_topology,
        "milestones": [
            {
                "index": milestone.index,
                "milestone_id": milestone.milestone_id,
                "label": milestone.label,
                "controller_state": milestone.incident_view["controller_state"],
                "event_sequence_start": milestone.event_sequence_start,
                "event_sequence_end": milestone.event_sequence_end,
                "auto_advance": milestone.auto_advance,
                "pause_reason": milestone.pause_reason,
                "pause_action": milestone.pause_action,
                "pause_action_label": milestone.pause_action_label,
                "incident_view": milestone.incident_view,
                "highlight": milestone.highlight,
                "narrative": milestone.narrative,
            }
            for milestone in milestones
        ],
    }
    # generated_at is a wall-clock timestamp, not semantic content -- excluded
    # from the hash (matching golden.py's own result_sha256, which excludes
    # its "runtime" timing block) so the acceptance criterion "same input
    # hashes + same code version -> same semantic reference artifact" holds
    # across repeated runs, not just within one process.
    reproducible_payload = {key: value for key, value in artifact.items() if key != "generated_at"}
    artifact["artifact_sha256"] = hashlib.sha256(_canonical(reproducible_payload).encode("utf-8")).hexdigest()
    return artifact


def _assert_contiguous_coverage(milestones: list[Milestone], event_count: int) -> None:
    expected_next = 0
    for milestone in milestones:
        assert milestone.event_sequence_start == expected_next, (
            f"milestone {milestone.milestone_id!r} starts at event "
            f"{milestone.event_sequence_start}, expected {expected_next} -- gap or overlap "
            "in event coverage"
        )
        assert milestone.event_sequence_end >= milestone.event_sequence_start
        expected_next = milestone.event_sequence_end + 1
    assert expected_next == event_count, (
        f"milestones cover events up to {expected_next - 1}, but the golden workflow has "
        f"{event_count} events (0..{event_count - 1})"
    )


#: submission.txt SUB-12.1 P1: the two real, meaningful human interactions
#: in the reference narrative -- collecting the next evidence sample, then
#: approving the verified plan. A milestone pauses if and only if it is
#: one of these two; every other milestone auto-advances. Declared once
#: here (not re-derived from auto_advance flags scattered across the
#: add() calls above) so _assert_pauses_are_well_formed can catch a
#: future milestone that pauses without a registered, intentional reason.
EXPECTED_PAUSE_ACTIONS = {
    "sample_recommended": "COLLECT_REFERENCE_SAMPLE",
    "human_approval_boundary": "APPROVE_REFERENCE_PLAN",
}


def _assert_pauses_are_well_formed(milestones: list[Milestone]) -> None:
    """Every milestone that pauses (auto_advance=False) must declare the
    exact pause_action EXPECTED_PAUSE_ACTIONS says it should, and no
    milestone outside that set may pause unannounced. Prevents both
    directions of drift: a future milestone silently gaining a pause the
    frontend has no button for, and a milestone silently losing the pause
    a judge/operator interaction depends on."""
    for milestone in milestones:
        expected_action = EXPECTED_PAUSE_ACTIONS.get(milestone.milestone_id)
        if expected_action is None:
            assert milestone.auto_advance, (
                f"milestone {milestone.milestone_id!r} pauses but is not in "
                "EXPECTED_PAUSE_ACTIONS -- add it there if this pause is intentional"
            )
            assert milestone.pause_action is None
            assert milestone.pause_action_label is None
        else:
            assert not milestone.auto_advance, (
                f"milestone {milestone.milestone_id!r} is expected to pause "
                f"(pause_action={expected_action!r}) but auto_advance is True"
            )
            assert milestone.pause_action == expected_action
            assert milestone.pause_action_label, (
                f"milestone {milestone.milestone_id!r} has a pause_action but no "
                "human-readable pause_action_label for the frontend to render"
            )


def _assert_no_future_leakage(milestones: list[Milestone], *, safe_plan_id: str, unsafe_plan_id: str) -> None:
    """Fail loudly rather than silently shipping a narrative that reveals
    a plan's verification outcome, the recommended sample, or the
    approval state before the corresponding event actually occurred."""

    by_id = {milestone.milestone_id: milestone for milestone in milestones}

    assert by_id["alert"].incident_view["candidates"] is None
    assert by_id["initial_uncertainty"].incident_view["candidates"] is not None
    assert by_id["initial_uncertainty"].incident_view["evidence_sufficient"] is False

    assert by_id["sample_recommended"].incident_view["recommended_sample"] is not None
    assert by_id["evidence_insufficient"].incident_view["recommended_sample"] is None, (
        "sample recommendation must not appear before the sample_recommended milestone"
    )
    assert by_id["sample_recommended"].incident_view["sample_observation"] is None, (
        "the sample value must not appear until the operator actually collects it "
        "(sample_received) -- sample_recommended is a real pause, not a formality"
    )

    # The raw sample must arrive before the posterior visibly contracts --
    # candidates at "sample_received" must still equal the PRE-sample
    # (initial) probabilities, not the posterior.
    sample_received_view = by_id["sample_received"].incident_view
    assert sample_received_view["sample_observation"] is not None
    assert sample_received_view["candidates"] == by_id["initial_uncertainty"].incident_view["candidates"], (
        "posterior probabilities leaked into the sample_received milestone before the "
        "posterior_contracted milestone actually computed them"
    )

    assert by_id["posterior_contracted"].incident_view["candidates"] != sample_received_view["candidates"]
    assert by_id["posterior_contracted"].incident_view["evidence_sufficient"] is True

    plans_generated_view = by_id["plans_generated"].incident_view
    assert plans_generated_view["plans"] is not None
    for entry in plans_generated_view["plans"]:
        assert entry["verification"] is None, (
            "a plan must not show a verification decision before its verification milestone"
        )

    unsafe_rejected_view = by_id["unsafe_plan_rejected"].incident_view
    unsafe_entry, safe_entry = unsafe_rejected_view["plans"]
    assert unsafe_entry["plan"]["plan_id"] == unsafe_plan_id
    assert unsafe_entry["verification"]["decision"] == "REJECTED"
    assert safe_entry["plan"]["plan_id"] == safe_plan_id
    assert safe_entry["verification"] is None, (
        "the safe plan's VERIFIED decision leaked into the unsafe_plan_rejected milestone"
    )

    safe_verified_view = by_id["safe_plan_verified"].incident_view
    assert safe_verified_view["plans"][1]["verification"]["decision"] == "VERIFIED"

    approval_view = by_id["human_approval_boundary"].incident_view
    assert approval_view["selected_plan_id"] == safe_plan_id
    assert approval_view["approval_pending"] is True
    assert approval_view["approved_plan_id"] is None, (
        "approval must not be shown as granted before the human_approved event"
    )
    assert approval_view["final_event_hash"] is None, (
        "final_event_hash must not appear before the workflow actually completes"
    )

    completed_view = by_id["completed"].incident_view
    assert completed_view["approved_plan_id"] == safe_plan_id
    assert completed_view["approval_pending"] is False
    assert completed_view["final_event_hash"], "completed milestone must carry the real final event hash"
