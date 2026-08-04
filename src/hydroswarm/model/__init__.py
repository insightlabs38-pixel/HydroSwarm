"""HydroSwarm neural model public API."""

from .adapters import BottleneckAdapter, RoleHead
from .core import (
    ARCHITECTURE_VERSION,
    INCIDENT_POOLING_MODES,
    MESSAGE_DIRECTIONS,
    MODEL_VARIANTS,
    PRIOR_MODES,
    ArchitectureCompatibilityError,
    HydroBatch,
    HydroCore,
    HydroMono,
    HydroOutput,
    IncidentPooling,
    MessageDirection,
    ModelVariant,
    NoAdapterHydroCore,
    ParameterReport,
    PriorMode,
    verify_architecture_compatibility,
)
from .encoders import GraphStructuralEncoder, QualityEncoder, StaticFeatureEncoder, TemporalEncoder
from .layers import DualChannelGraphConv, EdgeAwareGraphConv, LatentHydraulicBlock

__all__ = [
    "ARCHITECTURE_VERSION",
    "INCIDENT_POOLING_MODES",
    "MESSAGE_DIRECTIONS",
    "MODEL_VARIANTS",
    "PRIOR_MODES",
    "ArchitectureCompatibilityError",
    "BottleneckAdapter",
    "DualChannelGraphConv",
    "EdgeAwareGraphConv",
    "GraphStructuralEncoder",
    "HydroBatch",
    "HydroCore",
    "HydroMono",
    "HydroOutput",
    "IncidentPooling",
    "LatentHydraulicBlock",
    "MessageDirection",
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
