"""core-issues5.txt Section 15 (P1 product feature): Evidence Value / Stop
Certificate. Builds a real IncidentAnalysisResult via test_hybrid_
pipeline.py's own fixtures, then uses dataclasses.replace to exercise each
status branch against otherwise-realistic data."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from hydroswarm.domain import EvidenceCertificateStatus
from hydroswarm.inference.evidence_certificate import build_evidence_certificate
from hydroswarm.inference.fusion import ControlAction

from test_hybrid_pipeline import PriorFollowingModel, _pipeline, _series


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
