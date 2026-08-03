import torch

from hydroswarm.explanation import (
    ConstrainedLanguageDecoder,
    EvidenceBundle,
    ExplanationIntent,
    deterministic_operational_summary,
    explain,
    remove_one_sensor_sensitivity,
)


def evidence() -> EvidenceBundle:
    return EvidenceBundle(
        source_node="J1", source_probability=0.82, candidate_region=("J1", "J2"),
        candidate_coverage=0.91, recommended_sample="J3", information_gain_bits=0.7,
        candidates_before=6, candidates_after=2, selected_plan="Plan B", rejected_plan="Plan A",
        rejection_codes=("PRESSURE_BELOW_MINIMUM",), exposure_reduction_mg=120.0,
        pressure_violation_minutes=0.0, service_availability=0.98, disagreement_js=0.08,
        ood_level="NORMAL", approval_pending=True, abstention_reason=None,
        supporting_sensors=("S1", "S2"), removed_candidates={"J4": "negative sensor inconsistent"},
    )


def test_fixed_intents_and_summary_use_only_structured_facts() -> None:
    item = explain(ExplanationIntent.WHY_PLAN_REJECTED, evidence())
    assert "PRESSURE_BELOW_MINIMUM" in item.text
    summary = deterministic_operational_summary(evidence())
    assert "OPERATOR APPROVAL PENDING" in summary
    assert "98.0%" in summary
    scores = remove_one_sensor_sensitivity(
        ["S1", "S2"], {"J1": 0.8, "J2": 0.2},
        lambda sensor: {"J1": 0.5, "J2": 0.5} if sensor == "S1" else {"J1": 0.8, "J2": 0.2},
    )
    assert scores["S1"] > scores["S2"]


def test_language_decoder_detaches_verified_operational_memory() -> None:
    decoder = ConstrainedLanguageDecoder(vocab_size=64, d_model=32, nhead=4, dim_feedforward=64, num_layers=1)
    latent = torch.randn(2, 3, 16, requires_grad=True)
    output = decoder(torch.tensor([[1, 2], [3, 4]]), latent)
    output.sum().backward()
    assert latent.grad is None
    assert output.shape == (2, 2, 64)
