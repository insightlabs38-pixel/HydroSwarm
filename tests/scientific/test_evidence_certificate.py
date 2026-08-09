"""core-issues5.txt Section 15 (P1 product feature): Evidence Value / Stop
Certificate. Builds a real IncidentAnalysisResult via test_hybrid_
pipeline.py's own fixtures, then uses dataclasses.replace to exercise each
status branch against otherwise-realistic data."""

from __future__ import annotations

import pytest

from dataclasses import replace
from uuid import uuid4

from hydroswarm.domain import EvidenceCertificateStatus
from hydroswarm.inference.evidence_certificate import build_evidence_certificate
from hydroswarm.inference.fusion import ControlAction

from test_hybrid_pipeline import PriorFollowingModel, _pipeline, _series


pytestmark = pytest.mark.real_simulation


def _analysis():
    pipeline, network = _pipeline(PriorFollowingModel())
    return pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])


def test_evidence_sufficient_status_and_message() -> None:
    analysis = replace(_analysis(), evidence_sufficient=True, control_action=ControlAction.GENERATE_PLANS)
    certificate = build_evidence_certificate(analysis, sample_budget_remaining=3)

    assert certificate.status == EvidenceCertificateStatus.EVIDENCE_SUFFICIENT
    assert certificate.stop is True
    assert "EVIDENCE SUFFICIENT" in certificate.message
    assert "Planning gate satisfied" in certificate.message


def test_continue_sampling_status_when_evidence_insufficient_and_budget_remains() -> None:
    analysis = replace(_analysis(), evidence_sufficient=False, control_action=ControlAction.REQUEST_SAMPLE)
    certificate = build_evidence_certificate(analysis, sample_budget_remaining=2)

    assert certificate.status == EvidenceCertificateStatus.CONTINUE_SAMPLING
    assert certificate.stop is False
    assert certificate.sample_budget_remaining == 2
    assert "CONTINUE SAMPLING" in certificate.message


def test_stop_budget_exhausted_status_and_message() -> None:
    analysis = replace(_analysis(), evidence_sufficient=False, control_action=ControlAction.REQUEST_SAMPLE)
    certificate = build_evidence_certificate(analysis, sample_budget_remaining=0)

    assert certificate.status == EvidenceCertificateStatus.STOP_BUDGET_EXHAUSTED
    assert certificate.stop is True
    assert "SAMPLE BUDGET EXHAUSTED" in certificate.message
    assert certificate.sample_budget_remaining == 0


def test_stop_abstain_status_takes_priority_over_remaining_budget() -> None:
    analysis = replace(_analysis(), evidence_sufficient=False, control_action=ControlAction.ABSTAIN)
    certificate = build_evidence_certificate(analysis, sample_budget_remaining=5)

    assert certificate.status == EvidenceCertificateStatus.STOP_ABSTAIN
    assert certificate.stop is True
    assert "ABSTAINING" in certificate.message


def test_already_sampled_nodes_and_candidate_set_are_reported() -> None:
    analysis = _analysis()
    certificate = build_evidence_certificate(
        analysis, sample_budget_remaining=3, already_sampled_nodes=("J1", "J2")
    )
    assert certificate.already_sampled_nodes == ("J1", "J2")
    assert certificate.candidate_set_size == len(analysis.conformal_candidate_nodes)
    assert certificate.candidate_nodes == analysis.conformal_candidate_nodes


def test_negative_budget_is_clamped_to_zero_not_reported_negative() -> None:
    analysis = replace(_analysis(), evidence_sufficient=False, control_action=ControlAction.REQUEST_SAMPLE)
    certificate = build_evidence_certificate(analysis, sample_budget_remaining=-1)
    assert certificate.sample_budget_remaining == 0
    assert certificate.status == EvidenceCertificateStatus.STOP_BUDGET_EXHAUSTED


# core-issues5.txt delta item 8 (P1 fix): candidate_region_calibrated must
# truthfully distinguish a real calibrated conformal set from the
# uncalibrated credible-region fallback -- never presented as
# interchangeable, and the uncalibrated case must still carry a real,
# non-empty current snapshot rather than an implied empty region.


def test_calibrated_analysis_reports_a_calibrated_candidate_region() -> None:
    analysis = _analysis()
    assert analysis.calibrated is True
    certificate = build_evidence_certificate(analysis, sample_budget_remaining=3)

    assert certificate.candidate_region_calibrated is True
    assert "calibrated conformal set" in certificate.message
    assert "UNCALIBRATED" not in certificate.message
    # The real current candidate snapshot, not fabricated.
    assert certificate.candidate_nodes == analysis.conformal_candidate_nodes
    assert certificate.candidate_set_size == len(analysis.conformal_candidate_nodes)


def test_uncalibrated_analysis_reports_an_uncalibrated_credible_region_not_conformal_coverage() -> None:
    """analysis.calibrated=False must never be silently presented as if
    candidate_nodes carried conformal coverage's statistical guarantee --
    and the region itself must still be the real current credible-region
    snapshot (HybridInferencePipeline._credible_nodes), not an empty
    region implied merely because calibration is unavailable."""

    base = _analysis()
    analysis = replace(base, calibrated=False)
    assert analysis.conformal_candidate_nodes  # a real, non-empty snapshot exists
    certificate = build_evidence_certificate(analysis, sample_budget_remaining=3)

    assert certificate.candidate_region_calibrated is False
    assert "UNCALIBRATED credible region" in certificate.message
    assert "calibrated conformal set" not in certificate.message
    # Still the real current snapshot -- not emptied out.
    assert certificate.candidate_nodes == analysis.conformal_candidate_nodes
    assert certificate.candidate_set_size == len(analysis.conformal_candidate_nodes)
    assert certificate.candidate_set_size > 0


def test_uncalibrated_stop_budget_exhausted_message_does_not_claim_calibration_either() -> None:
    base = replace(_analysis(), evidence_sufficient=False, control_action=ControlAction.REQUEST_SAMPLE)
    analysis = replace(base, calibrated=False)
    certificate = build_evidence_certificate(analysis, sample_budget_remaining=0)

    assert certificate.candidate_region_calibrated is False
    assert certificate.status == EvidenceCertificateStatus.STOP_BUDGET_EXHAUSTED
