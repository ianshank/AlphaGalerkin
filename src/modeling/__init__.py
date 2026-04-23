"""Neural architectures and layers for AlphaGalerkin.

Public re-export surface. External consumers (e.g. Mouse-Droid-AGI's
sensor fusion head) import from this module rather than the underlying
files. Constructor signatures listed in
`docs/architecture/ADR-mouse-droid-fusion-integration.md` are
considered stable for the duration of the integration window.
"""

from src.modeling.attention import GalerkinAttention, HybridAttention, SoftmaxAttention
from src.modeling.embeddings import ContinuousEmbedding, FourierFeatures
from src.modeling.fnet import FNetBlock, FNetMixing, FNetStack, GalerkinFNetHybrid
from src.modeling.model import AlphaGalerkinModel
from src.modeling.multiscale_fourier import (
    AdaptiveFourierFeatures,
    FourierFeaturesConfig,
    MultiScaleFourierFeatures,
    PositionalEncoding,
    ProgressiveFourierFeatures,
    SpatialPositionalEncoding,
)
from src.modeling.stability import StabilityGuard, StableGalerkinInitializer

__all__ = [
    "AdaptiveFourierFeatures",
    "AlphaGalerkinModel",
    "ContinuousEmbedding",
    "FNetBlock",
    "FNetMixing",
    "FNetStack",
    "FourierFeatures",
    "FourierFeaturesConfig",
    "GalerkinAttention",
    "GalerkinFNetHybrid",
    "HybridAttention",
    "MultiScaleFourierFeatures",
    "PositionalEncoding",
    "ProgressiveFourierFeatures",
    "SoftmaxAttention",
    "SpatialPositionalEncoding",
    "StabilityGuard",
    "StableGalerkinInitializer",
]
