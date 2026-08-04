"""HydroSwarm neural model public API."""

from .adapters import BottleneckAdapter, RoleHead
from .core import (
    ARCHITECTURE_VERSION,
    MODEL_VARIANTS,
    PRIOR_MODES,
    ArchitectureCompatibilityError,
    HydroBatch,
    HydroCore,
    HydroMono,
    HydroOutput,
    ModelVariant,
    NoAdapterHydroCore,
    ParameterReport,
    PriorMode,
    verify_architecture_compatibility,
)
from .encoders import GraphStructuralEncoder, QualityEncoder, StaticFeatureEncoder, TemporalEncoder
from .layers import EdgeAwareGraphConv, LatentHydraulicBlock

__all__ = [
    "ARCHITECTURE_VERSION",
    "MODEL_VARIANTS",
    "PRIOR_MODES",
    "ArchitectureCompatibilityError",
    "BottleneckAdapter",
    "EdgeAwareGraphConv",
    "GraphStructuralEncoder",
    "HydroBatch",
    "HydroCore",
    "HydroMono",
    "HydroOutput",
    "LatentHydraulicBlock",
    "ModelVariant",
    "NoAdapterHydroCore",
    "ParameterReport",
    "PriorMode",
    "QualityEncoder",
    "RoleHead",
    "StaticFeatureEncoder",
    "TemporalEncoder",
    "verify_architecture_compatibility",
]
