"""Classical-neural fusion, calibration, and uncertainty control."""

from hydroswarm.inference.fusion import (
    DYNAMIC_TRUST_FUSION_CONFIG,
    ControlAction,
    FusionDiagnostics,
    TrustFeatures,
    conformal_candidate_set,
    fixed_weight_fusion,
    fixed_weight_fusion_config,
    fuse_source_probabilities,
    jensen_shannon_divergence,
    uncertainty_control,
)
from hydroswarm.calibration import CalibrationArtifact, SplitConformalCalibrator
from hydroswarm.inference.ood import OODComponents, OODDetector, OODReference
from hydroswarm.inference.pipeline import MODEL_VERSION, HybridInferencePipeline, HybridPipeline
from hydroswarm.inference.results import (
    EvidenceChange,
    EvidenceSnapshot,
    HybridRuntimeMode,
    IncidentAnalysisResult,
    PosteriorSnapshot,
    SemanticPredictions,
)

__all__ = [
    "DYNAMIC_TRUST_FUSION_CONFIG",
    "ControlAction",
    "CalibrationArtifact",
    "EvidenceChange",
    "EvidenceSnapshot",
    "FusionDiagnostics",
    "HybridInferencePipeline",
    "HybridPipeline",
    "HybridRuntimeMode",
    "IncidentAnalysisResult",
    "MODEL_VERSION",
    "OODComponents",
    "OODDetector",
    "OODReference",
    "PosteriorSnapshot",
    "SemanticPredictions",
    "SplitConformalCalibrator",
    "TrustFeatures",
    "conformal_candidate_set",
    "fixed_weight_fusion",
    "fixed_weight_fusion_config",
    "fuse_source_probabilities",
    "jensen_shannon_divergence",
    "uncertainty_control",
]
