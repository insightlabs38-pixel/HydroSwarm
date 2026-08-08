"""Evidence Value / Stop Certificate (core-issues5.txt Section 15, P1
product feature).

Converts the incident's current sampling/abstention state into a legible
decision-support signal -- "EVIDENCE SUFFICIENT" or "STOP: SAMPLE BUDGET
EXHAUSTED", not a silent absence of a sample recommendation. Uses only the
currently authoritative DETERMINISTIC sampling logic
(`hydroswarm.sampling.rank_sample_locations`, already run inside
`HybridInferencePipeline.analyze()` and captured on `sample_result`) --
learned Scout is not required, and is not promoted regardless (see
`hydroswarm.inference.authority.scout_certificate`).
"""

from __future__ import annotations

from hydroswarm.domain import EvidenceCertificate, EvidenceCertificateStatus
from hydroswarm.inference.fusion import ControlAction
from hydroswarm.inference.results import IncidentAnalysisResult

__all__ = ["build_evidence_certificate"]


def _message(
    status: EvidenceCertificateStatus,
    *,
    candidate_set_size: int,
    expected_information_gain_bits: float | None,
    sample_budget_remaining: int,
) -> str:
    if status == EvidenceCertificateStatus.EVIDENCE_SUFFICIENT:
        gain = f"{expected_information_gain_bits:.2f} bits" if expected_information_gain_bits is not None else "unavailable"
        return (
            f"EVIDENCE SUFFICIENT\nCandidate region: {candidate_set_size} node(s)\n"
            f"Additional expected information gain: {gain}\nPlanning gate satisfied"
        )
    if status == EvidenceCertificateStatus.STOP_BUDGET_EXHAUSTED:
        return "STOP: SAMPLE BUDGET EXHAUSTED\nDecision remains uncertain\nPlanning suppressed"
    if status == EvidenceCertificateStatus.STOP_NO_USEFUL_CANDIDATE:
        return "STOP: NO USEFUL SAMPLE CANDIDATE REMAINS\nDecision remains uncertain\nPlanning suppressed"
    if status == EvidenceCertificateStatus.STOP_ABSTAIN:
        return "STOP: ABSTAINING\nEvidence is too ambiguous or out of distribution\nPlanning suppressed"
    return (
        f"CONTINUE SAMPLING\nCandidate region: {candidate_set_size} node(s)\n"
        f"Sample budget remaining: {sample_budget_remaining}"
    )


def build_evidence_certificate(
    analysis: IncidentAnalysisResult,
    *,
    sample_budget_remaining: int,
    already_sampled_nodes: tuple[str, ...] = (),
) -> EvidenceCertificate:
    sample_result = analysis.sample_result
    candidate_nodes = analysis.conformal_candidate_nodes
    posterior_entropy_bits = (
        analysis.posterior_history[-1].entropy_bits if analysis.posterior_history else 0.0
    )

    recommended_node: str | None = None
    expected_information_gain_bits: float | None = None
    expected_candidate_reduction: float | None = None
    recommended_node_accessible: bool | None = None
    if sample_result is not None:
        recommended_node = sample_result.recommended_node
        if sample_result.ranked:
            top = sample_result.ranked[0]
            expected_information_gain_bits = top.expected_information_gain_bits
            expected_candidate_reduction = top.expected_candidate_reduction
            recommended_node_accessible = top.accessible

    if analysis.evidence_sufficient and analysis.control_action != ControlAction.REQUEST_SAMPLE:
        status = EvidenceCertificateStatus.EVIDENCE_SUFFICIENT
    elif analysis.control_action == ControlAction.ABSTAIN:
        status = EvidenceCertificateStatus.STOP_ABSTAIN
    elif sample_budget_remaining <= 0:
        status = EvidenceCertificateStatus.STOP_BUDGET_EXHAUSTED
    elif sample_result is not None and sample_result.stop:
        status = EvidenceCertificateStatus.STOP_NO_USEFUL_CANDIDATE
    else:
        status = EvidenceCertificateStatus.CONTINUE_SAMPLING

    message = _message(
        status,
        candidate_set_size=len(candidate_nodes),
        expected_information_gain_bits=expected_information_gain_bits,
        sample_budget_remaining=sample_budget_remaining,
    )
    return EvidenceCertificate(
        status=status,
        stop=status != EvidenceCertificateStatus.CONTINUE_SAMPLING,
        message=message,
        posterior_entropy_bits=posterior_entropy_bits,
        candidate_set_size=len(candidate_nodes),
        candidate_nodes=candidate_nodes,
        recommended_sample_node=recommended_node,
        expected_information_gain_bits=expected_information_gain_bits,
        expected_candidate_reduction=expected_candidate_reduction,
        sample_budget_remaining=max(0, sample_budget_remaining),
        already_sampled_nodes=already_sampled_nodes,
        recommended_node_accessible=recommended_node_accessible,
    )
