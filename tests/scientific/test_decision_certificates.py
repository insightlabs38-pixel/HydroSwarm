"""core-issues5.txt Section 13 (P1 product/architecture feature): Decision
Authority / Applicability Certificate contract. Reuses test_hybrid_
pipeline.py's own PriorFollowingModel/_pipeline/_series fixtures to build a
real IncidentAnalysisResult rather than hand-constructing every nested
dataclass."""

from __future__ import annotations

import pytest

from uuid import uuid4

from hydroswarm.domain import ApplicabilityStatus, AuthorityLevel, PlanDecision, PlanVerification
from hydroswarm.inference import build_decision_certificates
from hydroswarm.inference.authority import plan_consequence_certificate

from test_hybrid_pipeline import PriorFollowingModel, _pipeline, _series


def _analysis():
    pipeline, network = _pipeline(PriorFollowingModel())
    return pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])


def _verification(*, decision: PlanDecision, **kwargs) -> PlanVerification:
    from hydroswarm.domain import ConsequenceMetrics

    defaults = dict(
        plan_id=uuid4(),
        decision=decision,
        simulator="test",
        simulator_version="1.0",
        state_hash="a" * 64,
    )
    if decision == PlanDecision.VERIFIED and "consequences" not in kwargs:
        kwargs["consequences"] = ConsequenceMetrics(
            minimum_pressure_m=20.0, service_availability=1.0, operation_count=1
        )
    if decision == PlanDecision.REJECTED and "rejection_codes" not in kwargs:
        kwargs["rejection_codes"] = ("PRESSURE_BELOW_MINIMUM",)
    if decision == PlanDecision.ABSTAINED and "abstention_reason" not in kwargs:
        from hydroswarm.domain import AbstentionReason

        kwargs["abstention_reason"] = AbstentionReason.SIMULATION_TIMEOUT
    defaults.update(kwargs)
    return PlanVerification(**defaults)


def test_no_analysis_yet_produces_no_incident_level_certificates() -> None:
    certificates = build_decision_certificates(None)
    assert certificates == ()


def test_no_analysis_yet_still_surfaces_existing_plan_verifications() -> None:
    verification = _verification(decision=PlanDecision.VERIFIED)
    certificates = build_decision_certificates(None, verifications=(verification,))
    assert len(certificates) == 1
    assert certificates[0].name == f"plan_consequence:{verification.plan_id}"


@pytest.mark.real_simulation
def test_source_localization_certificate_reflects_calibration() -> None:
    analysis = _analysis()
    certificates = build_decision_certificates(analysis)
    localization = next(c for c in certificates if c.name == "source_localization")
    assert localization.source == "FUSED_CLASSICAL_NEURAL"
    assert localization.calibrated == analysis.calibrated
    assert localization.authority == (
        AuthorityLevel.CALIBRATED_ADVISORY if analysis.calibrated else AuthorityLevel.ADVISORY
    )
    assert localization.value["top_node"] in analysis.fused_belief
    assert localization.provenance.model == analysis.provenance_hashes.get("model")


@pytest.mark.real_simulation
def test_scout_certificate_always_reports_learned_scout_suppressed() -> None:
    analysis = _analysis()
    certificates = build_decision_certificates(analysis)
    scout = next(c for c in certificates if c.name == "scout_recommendation")
    assert scout.source == "CLASSICAL_EIG"
    assert scout.authority == AuthorityLevel.DETERMINISTIC
    assert "LEARNED_SCOUT_SUPPRESSED:FAILED_PROMOTION_GATE" in scout.suppression_reasons


@pytest.mark.real_simulation
def test_ood_certificate_always_reports_learned_ood_category_suppressed() -> None:
    analysis = _analysis()
    certificates = build_decision_certificates(analysis)
    ood = next(c for c in certificates if c.name == "ood_state")
    assert ood.source == "DETERMINISTIC_CONTROLLER"
    assert ood.authority == AuthorityLevel.DETERMINISTIC
    assert ood.value == analysis.ood_level.value
    assert "LEARNED_OOD_CATEGORY_SUPPRESSED:NOT_PROMOTED" in ood.suppression_reasons


@pytest.mark.real_simulation
def test_ood_certificate_unaffected_by_extreme_ood_category_even_when_runtime_enabled() -> None:
    """M10.1 readiness-review gap: closes the loop between the pipeline-
    level isolation proof (test_hybrid_pipeline_v4_gating.py's
    test_extreme_ood_category_logits_cannot_move_deterministic_ood_
    severity_*) and the Decision Authority contract actually surfaced to
    callers -- the certificate must keep reporting DETERMINISTIC/
    NOT_PROMOTED with an unchanged value even when a v4 identity runtime-
    enables ood_category AND the underlying head is pinned to an extreme,
    maximally-confident, different SUPPORTED category."""

    from test_hybrid_pipeline_v4_gating import EventAwareModel, ExtremeOODCategoryModel

    moderate_pipeline, network = _pipeline(EventAwareModel())
    moderate_pipeline.runtime_enabled_outputs = frozenset({"ood_category"})
    moderate = moderate_pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    extreme_pipeline, network = _pipeline(ExtremeOODCategoryModel())
    extreme_pipeline.runtime_enabled_outputs = frozenset({"ood_category"})
    extreme = extreme_pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    assert moderate.semantic_predictions.ood_category != extreme.semantic_predictions.ood_category

    for analysis in (moderate, extreme):
        certificates = build_decision_certificates(analysis)
        ood = next(c for c in certificates if c.name == "ood_state")
        assert ood.source == "DETERMINISTIC_CONTROLLER"
        assert ood.authority == AuthorityLevel.DETERMINISTIC
        assert ood.value == analysis.ood_level.value
        assert "LEARNED_OOD_CATEGORY_SUPPRESSED:NOT_PROMOTED" in ood.suppression_reasons

    moderate_certificate = next(c for c in build_decision_certificates(moderate) if c.name == "ood_state")
    extreme_certificate = next(c for c in build_decision_certificates(extreme) if c.name == "ood_state")
    assert moderate_certificate.value == extreme_certificate.value


def test_verified_plan_certificate_is_simulator_verified_and_current() -> None:
    verification = _verification(decision=PlanDecision.VERIFIED)
    certificate = plan_consequence_certificate(verification)
    assert certificate.source == "WNTR_EPANET"
    assert certificate.authority == AuthorityLevel.SIMULATOR_VERIFIED
    assert certificate.applicability == ApplicabilityStatus.VALIDATED
    assert certificate.enabled is True
    assert certificate.value is not None


def test_rejected_plan_certificate_carries_rejection_codes() -> None:
    verification = _verification(decision=PlanDecision.REJECTED)
    certificate = plan_consequence_certificate(verification)
    assert certificate.authority == AuthorityLevel.SIMULATOR_VERIFIED
    assert certificate.suppression_reasons == ("PRESSURE_BELOW_MINIMUM",)


def test_abstained_plan_certificate_is_unavailable_not_verified() -> None:
    verification = _verification(decision=PlanDecision.ABSTAINED)
    certificate = plan_consequence_certificate(verification)
    assert certificate.authority == AuthorityLevel.UNAVAILABLE
    assert certificate.enabled is False
    assert certificate.suppression_reasons == ("ABSTAINED:SIMULATION_TIMEOUT",)


def test_stale_verification_certificate_reflects_stale_applicability() -> None:
    verification = _verification(decision=PlanDecision.VERIFIED, verification_status="STALE")
    certificate = plan_consequence_certificate(verification)
    assert certificate.applicability == ApplicabilityStatus.STALE
    # A stale verification is still real simulator output -- it must not be
    # silently reclassified as UNAVAILABLE or lose its authority level,
    # only its applicability.
    assert certificate.authority == AuthorityLevel.SIMULATOR_VERIFIED


@pytest.mark.real_simulation
def test_full_certificate_set_includes_every_plan_verification() -> None:
    analysis = _analysis()
    verified = _verification(decision=PlanDecision.VERIFIED)
    rejected = _verification(decision=PlanDecision.REJECTED)
    certificates = build_decision_certificates(analysis, verifications=(verified, rejected))
    names = {c.name for c in certificates}
    assert "source_localization" in names
    assert "scout_recommendation" in names
    assert "ood_state" in names
    assert f"plan_consequence:{verified.plan_id}" in names
    assert f"plan_consequence:{rejected.plan_id}" in names
    assert len(certificates) == 5
