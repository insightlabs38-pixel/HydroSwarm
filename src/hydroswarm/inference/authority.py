"""Decision Authority / Applicability Certificate contract (core-issues5.txt
Section 13, P1 product/architecture feature).

One stable backend contract making HydroSwarm's authority hierarchy
explicit for every significant result, so the frontend never has to infer
authority (is this deterministic? calibrated? simulator-verified? actually
enabled at all?) from loosely related fields scattered across other
endpoints. Pure functions over already-computed results
(`IncidentAnalysisResult`, `PlanVerification`) -- this module makes no
inference calls and holds no state of its own.
"""

from __future__ import annotations

from collections.abc import Sequence

from hydroswarm.domain import (
    ApplicabilityStatus,
    AuthorityLevel,
    DecisionCertificate,
    DecisionProvenance,
    OODLevel,
    PlanDecision,
    PlanVerification,
)
from hydroswarm.inference.results import IncidentAnalysisResult

__all__ = [
    "build_decision_certificates",
    "ood_certificate",
    "plan_consequence_certificate",
    "scout_certificate",
    "source_localization_certificate",
]


def _analysis_provenance(analysis: IncidentAnalysisResult) -> DecisionProvenance:
    hashes = analysis.provenance_hashes
    return DecisionProvenance(
        model=hashes.get("model"),
        calibration=hashes.get("calibration"),
        network=hashes.get("network"),
        evidence=hashes.get("evidence"),
    )


def source_localization_certificate(analysis: IncidentAnalysisResult) -> DecisionCertificate:
    top_node = max(analysis.fused_belief.items(), key=lambda item: item[1])[0] if analysis.fused_belief else None
    applicability = ApplicabilityStatus.VALIDATED
    if analysis.ood_level != OODLevel.NORMAL:
        applicability = ApplicabilityStatus.OOD
    elif not analysis.calibrated:
        applicability = ApplicabilityStatus.CALIBRATION_UNAVAILABLE
    return DecisionCertificate(
        name="source_localization",
        value={"top_node": top_node, "belief": dict(analysis.fused_belief)},
        source="FUSED_CLASSICAL_NEURAL" if analysis.neural_belief is not None else "CLASSICAL_ONLY",
        authority=(
            AuthorityLevel.CALIBRATED_ADVISORY if analysis.calibrated else AuthorityLevel.ADVISORY
        ),
        applicability=applicability,
        enabled=bool(analysis.fused_belief),
        calibrated=analysis.calibrated,
        suppression_reasons=analysis.planning_suppression_reasons,
        provenance=_analysis_provenance(analysis),
    )


def scout_certificate(analysis: IncidentAnalysisResult) -> DecisionCertificate:
    """Learned Scout remains disabled unless separately promoted (Phase 14:
    learned_scout's realized entropy reduction is negative, worse than
    random sampling) -- the classical expected-information-gain
    recommendation is the real, deterministic, currently-authoritative
    source regardless."""

    sample_result = analysis.sample_result
    recommended = sample_result.recommended_node if sample_result is not None else None
    return DecisionCertificate(
        name="scout_recommendation",
        value=recommended,
        source="CLASSICAL_EIG",
        authority=AuthorityLevel.DETERMINISTIC,
        applicability=(
            ApplicabilityStatus.VALIDATED
            if sample_result is not None
            else ApplicabilityStatus.INSUFFICIENT_EVIDENCE
        ),
        enabled=sample_result is not None,
        calibrated=False,
        suppression_reasons=("LEARNED_SCOUT_SUPPRESSED:FAILED_PROMOTION_GATE",),
        provenance=_analysis_provenance(analysis),
    )


def ood_certificate(analysis: IncidentAnalysisResult) -> DecisionCertificate:
    """Deterministic OOD severity remains authoritative regardless of
    whether the learned 11-category ood_category head is ever promoted --
    see checkpoint_identity.py Section D item 7 and core-issues5.txt
    Section 18.1."""

    return DecisionCertificate(
        name="ood_state",
        value=analysis.ood_level.value,
        source="DETERMINISTIC_CONTROLLER",
        authority=AuthorityLevel.DETERMINISTIC,
        applicability=(
            ApplicabilityStatus.OOD
            if analysis.ood_level != OODLevel.NORMAL
            else ApplicabilityStatus.VALIDATED
        ),
        enabled=True,
        calibrated=False,
        suppression_reasons=("LEARNED_OOD_CATEGORY_SUPPRESSED:NOT_PROMOTED",),
        provenance=_analysis_provenance(analysis),
    )


def plan_consequence_certificate(
    verification: PlanVerification, *, provenance: DecisionProvenance | None = None
) -> DecisionCertificate:
    """core-issues5.txt Section 10's verification_status (CURRENT/STALE) is
    the applicability signal here -- a plan verified under evidence that
    has since changed is real simulator output, but no longer trustworthy
    for approval."""

    if verification.decision == PlanDecision.ABSTAINED:
        authority = AuthorityLevel.UNAVAILABLE
        suppression_reasons = (
            (f"ABSTAINED:{verification.abstention_reason.value}",)
            if verification.abstention_reason is not None
            else ("ABSTAINED",)
        )
    else:
        authority = AuthorityLevel.SIMULATOR_VERIFIED
        suppression_reasons = verification.rejection_codes

    applicability = (
        ApplicabilityStatus.STALE
        if verification.verification_status == "STALE"
        else ApplicabilityStatus.VALIDATED
    )
    return DecisionCertificate(
        name=f"plan_consequence:{verification.plan_id}",
        value=verification.consequences.model_dump(mode="json") if verification.consequences else None,
        source="WNTR_EPANET",
        authority=authority,
        applicability=applicability,
        enabled=verification.decision != PlanDecision.ABSTAINED,
        calibrated=False,
        suppression_reasons=suppression_reasons,
        provenance=provenance or DecisionProvenance(network=verification.state_hash),
    )


def build_decision_certificates(
    analysis: IncidentAnalysisResult | None,
    *,
    verifications: Sequence[PlanVerification] = (),
) -> tuple[DecisionCertificate, ...]:
    """The complete set of certificates for one incident's current state.
    `analysis` is None before the first analyze() call -- returns an empty
    tuple rather than fabricating placeholder certificates for results
    that do not exist yet."""

    if analysis is None:
        return tuple(
            plan_consequence_certificate(verification) for verification in verifications
        )
    return (
        source_localization_certificate(analysis),
        scout_certificate(analysis),
        ood_certificate(analysis),
        *(plan_consequence_certificate(verification) for verification in verifications),
    )
