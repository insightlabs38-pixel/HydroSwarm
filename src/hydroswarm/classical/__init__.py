"""Classical hydraulic inference and operational metrics."""

from .dynamic_graph import (
    DirectedHydraulicGraph,
    HydraulicLink,
    NodeId,
    build_dynamic_graph,
)
from .metrics import (
    AbstentionMetrics,
    CandidateSetMetrics,
    PressureViolationMetrics,
    abstention_quality,
    candidate_set_metrics,
    entropy,
    information_gain_per_sample,
    localization_top_k,
    mean_reciprocal_rank,
    pressure_violations,
)
from .prior import BayesianPosterior, SignatureLibrary, bayesian_source_posterior
from .screening import (
    CandidateFeatures,
    ScreeningResult,
    SensorObservation,
    screen_candidates,
)
from .state_estimation import (
    EstimatedHydraulicState,
    HydraulicStateEstimator,
    OperationalTelemetry,
    StateResidualReport,
    ValueRange,
)
from .signatures import (
    CandidateExplanation,
    ClassicalLocalizationResult,
    SignatureArtifact,
    SignatureBuilder,
    SignatureCache,
    SignatureCacheKey,
    SourceHypothesis,
    classify_trace_residual,
    localize_with_signatures,
)
from .signature_registry import SignatureRegistry, SignatureRegistryEntry, SignatureRegistryError

__all__ = [
    "AbstentionMetrics",
    "BayesianPosterior",
    "CandidateFeatures",
    "CandidateExplanation",
    "ClassicalLocalizationResult",
    "CandidateSetMetrics",
    "DirectedHydraulicGraph",
    "HydraulicLink",
    "HydraulicStateEstimator",
    "EstimatedHydraulicState",
    "NodeId",
    "PressureViolationMetrics",
    "OperationalTelemetry",
    "ScreeningResult",
    "SensorObservation",
    "SignatureLibrary",
    "SignatureArtifact",
    "SignatureBuilder",
    "SignatureCache",
    "SignatureCacheKey",
    "SignatureRegistry",
    "SignatureRegistryEntry",
    "SignatureRegistryError",
    "SourceHypothesis",
    "StateResidualReport",
    "ValueRange",
    "abstention_quality",
    "bayesian_source_posterior",
    "build_dynamic_graph",
    "candidate_set_metrics",
    "classify_trace_residual",
    "entropy",
    "information_gain_per_sample",
    "localization_top_k",
    "localize_with_signatures",
    "mean_reciprocal_rank",
    "pressure_violations",
    "screen_candidates",
]
