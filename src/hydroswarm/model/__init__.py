"""HydroSwarm neural model public API."""

from .adapters import BottleneckAdapter, RoleHead
from .core import (
    MODEL_VARIANTS,
    HydroBatch,
    HydroCore,
    HydroMono,
    HydroOutput,
    ModelVariant,
    NoAdapterHydroCore,
    ParameterReport,
)
from .encoders import GraphStructuralEncoder, QualityEncoder, StaticFeatureEncoder, TemporalEncoder
from .layers import EdgeAwareGraphConv, LatentHydraulicBlock

__all__ = [
    "MODEL_VARIANTS",
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
    "QualityEncoder",
    "RoleHead",
    "StaticFeatureEncoder",
    "TemporalEncoder",
]
