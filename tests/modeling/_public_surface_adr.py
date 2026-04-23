"""Source of truth for the Mouse-Droid-AGI fusion-head integration ADR.

The 18 classes below make up the stable public surface of ``src.modeling``
declared in ``docs/architecture/ADR-mouse-droid-fusion-integration.md``.
Each entry pairs the class (imported from the **top-level** package
``src.modeling`` — never submodule-deep) with the canonical submodule
path listed in the ADR table.

The contract test and the golden regenerator both import from this file
so there is exactly one list to keep in sync with the ADR.
"""

from __future__ import annotations

from typing import NamedTuple

from src.modeling import (
    AdaptiveFourierFeatures,
    AlphaGalerkinModel,
    ContinuousEmbedding,
    FNetBlock,
    FNetMixing,
    FNetStack,
    FourierFeatures,
    FourierFeaturesConfig,
    GalerkinAttention,
    GalerkinFNetHybrid,
    HybridAttention,
    MultiScaleFourierFeatures,
    PositionalEncoding,
    ProgressiveFourierFeatures,
    SoftmaxAttention,
    SpatialPositionalEncoding,
    StabilityGuard,
    StableGalerkinInitializer,
)


class SurfaceEntry(NamedTuple):
    """One row of the ADR stable-surface table."""

    cls: type
    expected_module: str


PUBLIC_SURFACE: tuple[SurfaceEntry, ...] = (
    SurfaceEntry(GalerkinAttention, "src.modeling.attention"),
    SurfaceEntry(SoftmaxAttention, "src.modeling.attention"),
    SurfaceEntry(HybridAttention, "src.modeling.attention"),
    SurfaceEntry(FNetBlock, "src.modeling.fnet"),
    SurfaceEntry(FNetMixing, "src.modeling.fnet"),
    SurfaceEntry(FNetStack, "src.modeling.fnet"),
    SurfaceEntry(GalerkinFNetHybrid, "src.modeling.fnet"),
    SurfaceEntry(MultiScaleFourierFeatures, "src.modeling.multiscale_fourier"),
    SurfaceEntry(AdaptiveFourierFeatures, "src.modeling.multiscale_fourier"),
    SurfaceEntry(ProgressiveFourierFeatures, "src.modeling.multiscale_fourier"),
    SurfaceEntry(PositionalEncoding, "src.modeling.multiscale_fourier"),
    SurfaceEntry(SpatialPositionalEncoding, "src.modeling.multiscale_fourier"),
    SurfaceEntry(FourierFeaturesConfig, "src.modeling.multiscale_fourier"),
    SurfaceEntry(StabilityGuard, "src.modeling.stability"),
    SurfaceEntry(StableGalerkinInitializer, "src.modeling.stability"),
    SurfaceEntry(AlphaGalerkinModel, "src.modeling.model"),
    SurfaceEntry(ContinuousEmbedding, "src.modeling.embeddings"),
    SurfaceEntry(FourierFeatures, "src.modeling.embeddings"),
)
"""The 18 ADR-frozen classes. Order matches the ADR table for human diff-ability."""
