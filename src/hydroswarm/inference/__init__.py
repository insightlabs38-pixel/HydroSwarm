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

__all__ = [
    "ControlAction",
    "FusionDiagnostics",
    "TrustFeatures",
    "conformal_candidate_set",
    "fuse_source_probabilities",
    "jensen_shannon_divergence",
    "uncertainty_control",
]

