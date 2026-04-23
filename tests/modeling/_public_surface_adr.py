"""Source of truth for the Mouse-Droid-AGI fusion-head integration ADR.

The 18 classes below make up the stable public surface of ``src.modeling``
declared in ``docs/architecture/ADR-mouse-droid-fusion-integration.md``.
Each entry pairs the top-level class *name* with the canonical submodule
path listed in the ADR table.

Names are stored as strings (not direct imports) so that a missing or
renamed export surfaces as an actionable assertion failure with the
custom remediation message in ``test_public_surface_contract.py``,
rather than an opaque ``ImportError`` during test collection.

The contract test and the golden regenerator both consume this list via
``tests.modeling._signature_utils.resolve_class``.
"""

from __future__ import annotations

from typing import NamedTuple


class SurfaceEntry(NamedTuple):
    """One row of the ADR stable-surface table."""

    class_name: str
    expected_module: str


# The 18 ADR-frozen classes. Order matches the ADR table for human diff-ability.
PUBLIC_SURFACE: tuple[SurfaceEntry, ...] = (
    SurfaceEntry("GalerkinAttention", "src.modeling.attention"),
    SurfaceEntry("SoftmaxAttention", "src.modeling.attention"),
    SurfaceEntry("HybridAttention", "src.modeling.attention"),
    SurfaceEntry("FNetBlock", "src.modeling.fnet"),
    SurfaceEntry("FNetMixing", "src.modeling.fnet"),
    SurfaceEntry("FNetStack", "src.modeling.fnet"),
    SurfaceEntry("GalerkinFNetHybrid", "src.modeling.fnet"),
    SurfaceEntry("MultiScaleFourierFeatures", "src.modeling.multiscale_fourier"),
    SurfaceEntry("AdaptiveFourierFeatures", "src.modeling.multiscale_fourier"),
    SurfaceEntry("ProgressiveFourierFeatures", "src.modeling.multiscale_fourier"),
    SurfaceEntry("PositionalEncoding", "src.modeling.multiscale_fourier"),
    SurfaceEntry("SpatialPositionalEncoding", "src.modeling.multiscale_fourier"),
    SurfaceEntry("FourierFeaturesConfig", "src.modeling.multiscale_fourier"),
    SurfaceEntry("StabilityGuard", "src.modeling.stability"),
    SurfaceEntry("StableGalerkinInitializer", "src.modeling.stability"),
    SurfaceEntry("AlphaGalerkinModel", "src.modeling.model"),
    SurfaceEntry("ContinuousEmbedding", "src.modeling.embeddings"),
    SurfaceEntry("FourierFeatures", "src.modeling.embeddings"),
)
