"""Classical-neural fusion, calibration, and uncertainty control."""

from hydroswarm.inference.fusion import (
    ControlAction,
    FusionDiagnostics,
    TrustFeatures,
    conformal_candidate_set,
    fuse_source_probabilities,
    jensen_shannon_divergence,
    uncertainty_control,
)
from hydroswarm.calibration import CalibrationArtifact, SplitConformalCalibrator
from hydroswarm.inference.ood import OODComponents, OODDetector, OODReference
from hydroswarm.inference.pipeline import HybridInferencePipeline, HybridPipeline
from hydroswarm.inference.results import (
    EvidenceChange,
    EvidenceSnapshot,
    HybridRuntimeMode,
    IncidentAnalysisResult,
    PosteriorSnapshot,
    SemanticPredictions,
)

__all__ = [
    "ControlAction",
    "CalibrationArtifact",
    "EvidenceChange",
    "EvidenceSnapshot",
    "FusionDiagnostics",
    "HybridInferencePipeline",
    "HybridPipeline",
    "HybridRuntimeMode",
    "IncidentAnalysisResult",
    "OODComponents",
    "OODDetector",
    "OODReference",
    "PosteriorSnapshot",
    "SemanticPredictions",
    "SplitConformalCalibrator",
    "TrustFeatures",
    "conformal_candidate_set",
    "fuse_source_probabilities",
    "jensen_shannon_divergence",
    "uncertainty_control",
]
