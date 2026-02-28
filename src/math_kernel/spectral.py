"""Spectral filtering and resolution adaptation for zero-shot transfer.

When transferring a model trained on one resolution to another, we need
to handle spectral aliasing. This module provides tools for proper
spectral filtering to enable resolution-independent inference.

Supports both PyTorch and JAX backends. The original PyTorch classes
(SpectralFilter, ResolutionAdapter) are kept unchanged for backward
compatibility.  JAX-backed equivalents (JaxSpectralFilter,
JaxResolutionAdapter) are provided when JAX/Flax are installed.
Factory functions select the right class based on a ``backend`` parameter.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional JAX / Flax imports
# ---------------------------------------------------------------------------
try:
    import flax.linen as fnn
    import jax  # noqa: F401 – used transitively
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

__all__ = [
    "SpectralFilter",
    "ResolutionAdapter",
    "create_spectral_filter",
    "create_resolution_adapter",
    "HAS_JAX",
]


# ===================================================================
# PyTorch implementations (unchanged for backward compatibility)
# ===================================================================


class SpectralFilter(nn.Module):
    """Low-pass spectral filter for anti-aliasing.

    When increasing resolution, high-frequency components that weren't
    present in training can cause artifacts. This filter attenuates
    frequencies above a cutoff to prevent aliasing.

    The filter is applied in the Fourier domain:
        filtered = iFFT(FFT(x) * H(f))

    where H(f) is a smooth low-pass filter.
    """

    def __init__(
        self,
        cutoff_ratio: float = 0.5,
        filter_type: str = "gaussian",
        learnable: bool = False,
    ) -> None:
        """Initialize spectral filter.

        Args:
            cutoff_ratio: Cutoff frequency as ratio of Nyquist (0 to 1).
            filter_type: Type of filter ("gaussian", "butterworth", "ideal").
            learnable: Whether cutoff is learnable.

        """
        super().__init__()
        self.filter_type = filter_type

        if learnable:
            self.cutoff_ratio = nn.Parameter(torch.tensor(cutoff_ratio))
        else:
            self.register_buffer("cutoff_ratio", torch.tensor(cutoff_ratio))

    def _create_filter_2d(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> Float[Tensor, "h w"]:
        """Create 2D frequency filter.

        Args:
            height: Spatial height.
            width: Spatial width (for rfft2, this is width // 2 + 1).
            device: Target device.

        Returns:
            2D filter mask.

        """
        # Frequency coordinates (normalized to [-0.5, 0.5] for full FFT)
        # For rfft2, we only need positive frequencies in the last dimension
        fy = torch.fft.fftfreq(height, device=device)
        fx = torch.fft.rfftfreq(width, device=device)

        # Create 2D frequency grid
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

        # Normalized frequency magnitude
        freq_magnitude = torch.sqrt(fy_grid**2 + fx_grid**2)

        # Apply filter based on type
        cutoff = self.cutoff_ratio * 0.5  # Scale to Nyquist

        if self.filter_type == "gaussian":
            # Gaussian filter: exp(-f^2 / (2 * cutoff^2))
            sigma = cutoff / 2.0
            filter_mask = torch.exp(-(freq_magnitude**2) / (2 * sigma**2 + 1e-8))

        elif self.filter_type == "butterworth":
            # Butterworth filter: 1 / (1 + (f/cutoff)^(2*order))
            order = 4
            filter_mask = 1.0 / (1.0 + (freq_magnitude / (cutoff + 1e-8)) ** (2 * order))

        elif self.filter_type == "ideal":
            # Ideal low-pass (sharp cutoff)
            filter_mask = (freq_magnitude <= cutoff).float()

        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        return filter_mask

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch channels height width"]:
        """Apply spectral filtering.

        Args:
            x: Input tensor in spatial domain.

        Returns:
            Filtered tensor.

        """
        # Apply 2D real FFT
        x_freq = torch.fft.rfft2(x)

        # Create and apply filter
        height, width = x.shape[-2:]
        filter_mask = self._create_filter_2d(height, width, x.device)

        # Broadcast filter to batch and channels
        x_freq_filtered = x_freq * filter_mask.unsqueeze(0).unsqueeze(0)

        # Inverse FFT
        x_filtered = torch.fft.irfft2(x_freq_filtered, s=(height, width))

        return x_filtered


class ResolutionAdapter(nn.Module):
    """Adapter for resolution-independent inference.

    Handles the transition between training resolution and inference
    resolution by:
    1. Spectral interpolation of positional features
    2. Anti-aliasing filtering
    3. Normalization adjustment for Monte Carlo integrals
    """

    def __init__(
        self,
        base_resolution: int | None = None,
        filter_cutoff: float = 0.5,
    ) -> None:
        """Initialize resolution adapter.

        Args:
            base_resolution: Training resolution (None = adaptive).
            filter_cutoff: Spectral filter cutoff ratio.

        """
        super().__init__()
        self.base_resolution = base_resolution
        self.spectral_filter = SpectralFilter(cutoff_ratio=filter_cutoff)

    def adapt_features(
        self,
        features: Float[Tensor, "batch n d"],
        source_size: int,
        target_size: int,
    ) -> Float[Tensor, "batch m d"]:
        """Adapt features from source to target resolution.

        Uses spectral interpolation to resize feature maps while
        preserving frequency content up to the Nyquist limit.

        Args:
            features: Features at source resolution.
            source_size: Source board size.
            target_size: Target board size.

        Returns:
            Adapted features at target resolution.

        """
        batch, n, d = features.shape

        if source_size == target_size:
            return features

        # Reshape to 2D spatial format
        features_2d = rearrange(features, "b (h w) d -> b d h w", h=source_size, w=source_size)

        # Spectral interpolation
        features_freq = torch.fft.rfft2(features_2d)

        # Target frequency dimensions
        target_freq_h = target_size
        target_freq_w = target_size // 2 + 1

        # Create output frequency tensor
        out_freq = torch.zeros(
            batch,
            d,
            target_freq_h,
            target_freq_w,
            dtype=features_freq.dtype,
            device=features.device,
        )

        # Copy frequencies (zero-pad or truncate)
        src_h, src_w = features_freq.shape[-2:]
        copy_h = min(src_h // 2, target_freq_h // 2)
        copy_w = min(src_w, target_freq_w)

        # Copy low frequencies (positive)
        out_freq[:, :, :copy_h, :copy_w] = features_freq[:, :, :copy_h, :copy_w]

        # Copy low frequencies (negative, wrap-around)
        if copy_h > 0:
            out_freq[:, :, -copy_h:, :copy_w] = features_freq[:, :, -copy_h:, :copy_w]

        # Inverse FFT to target resolution
        features_target = torch.fft.irfft2(out_freq, s=(target_size, target_size))

        # Apply anti-aliasing filter if upsampling
        if target_size > source_size:
            # Cutoff based on source resolution
            cutoff_ratio = source_size / target_size
            features_target = self._apply_adaptive_filter(features_target, cutoff_ratio)

        # Reshape back to sequence format
        features_out = rearrange(features_target, "b d h w -> b (h w) d")

        # Normalize for Monte Carlo integral consistency
        # FFT-based interpolation naturally preserves frequency content
        # Scale to maintain feature magnitude across resolutions:
        # - Upsampling (target > source): ratio > 1, compensates for energy dilution
        # - Downsampling (target < source): ratio < 1, maintains relative proportions
        scale_factor = target_size / source_size
        features_out = features_out * scale_factor

        return features_out

    def _apply_adaptive_filter(
        self,
        x: Float[Tensor, "batch channels height width"],
        cutoff_ratio: float,
    ) -> Float[Tensor, "batch channels height width"]:
        """Apply filter with adaptive cutoff.

        Args:
            x: Input tensor.
            cutoff_ratio: Adaptive cutoff ratio.

        Returns:
            Filtered tensor.

        """
        # Temporarily modify filter cutoff
        original_cutoff = self.spectral_filter.cutoff_ratio.clone()
        self.spectral_filter.cutoff_ratio.data.fill_(cutoff_ratio)

        result = self.spectral_filter(x)

        # Restore original cutoff
        self.spectral_filter.cutoff_ratio.data.copy_(original_cutoff)

        return result

    def forward(
        self,
        features: Float[Tensor, "batch n d"],
        source_size: int,
        target_size: int,
    ) -> Float[Tensor, "batch m d"]:
        """Forward pass (alias for adapt_features)."""
        return self.adapt_features(features, source_size, target_size)


# ===================================================================
# JAX / Flax implementations (guarded by HAS_JAX)
# ===================================================================

if HAS_JAX:

    class JaxSpectralFilter(fnn.Module):
        """JAX/Flax low-pass spectral filter for anti-aliasing.

        When increasing resolution, high-frequency components that weren't
        present in training can cause artifacts.  This filter attenuates
        frequencies above a cutoff to prevent aliasing.

        The filter is applied in the Fourier domain:
            filtered = iFFT(FFT(x) * H(f))

        where H(f) is a smooth low-pass filter.

        This is the Flax equivalent of :class:`SpectralFilter`.

        Attributes:
            cutoff_ratio: Cutoff frequency as ratio of Nyquist (0 to 1).
            filter_type: Type of filter (``"gaussian"``, ``"butterworth"``,
                ``"ideal"``).
            learnable: Whether the cutoff is a learnable parameter.

        """

        cutoff_ratio: float = 0.5
        filter_type: str = "gaussian"
        learnable: bool = False

        @fnn.compact  # type: ignore[untyped-decorator]
        def __call__(
            self,
            x: Any,
        ) -> Any:
            """Apply spectral filtering.

            Args:
                x: JAX array of shape ``(batch, channels, height, width)``
                    in the spatial domain.

            Returns:
                Filtered JAX array of the same shape.

            """
            height, width = x.shape[-2], x.shape[-1]

            # Resolve cutoff_ratio (learnable or fixed)
            if self.learnable:
                cutoff_val = self.param(
                    "cutoff_ratio",
                    lambda _rng: jnp.array(self.cutoff_ratio),
                )
            else:
                cutoff_val = jnp.array(self.cutoff_ratio)

            # Create filter (pass filter_type explicitly since static method
            # cannot access self)
            filter_mask = self._create_filter_2d(
                height, width, cutoff_val, filter_type=self.filter_type
            )

            # Apply 2D real FFT
            x_freq = jnp.fft.rfft2(x)

            # Broadcast filter to batch and channels
            x_freq_filtered = x_freq * filter_mask[jnp.newaxis, jnp.newaxis, ...]

            # Inverse FFT
            x_filtered = jnp.fft.irfft2(x_freq_filtered, s=(height, width))

            return x_filtered

        @staticmethod
        def _create_filter_2d(
            height: int,
            width: int,
            cutoff_ratio: Any,
            filter_type: str | None = None,
        ) -> Any:
            """Create 2D frequency filter.

            This is a static helper; ``filter_type`` defaults to the
            instance attribute when called from ``__call__``.

            Args:
                height: Spatial height.
                width: Spatial width.
                cutoff_ratio: Cutoff frequency as ratio of Nyquist.
                filter_type: Override filter type (used by classmethod
                    callers).

            Returns:
                JAX array of shape ``(height, width // 2 + 1)`` containing
                the filter mask.

            """
            # Frequency coordinates
            fy = jnp.fft.fftfreq(height)
            fx = jnp.fft.rfftfreq(width)

            # Create 2D frequency grid
            fy_grid, fx_grid = jnp.meshgrid(fy, fx, indexing="ij")

            # Normalized frequency magnitude
            freq_magnitude = jnp.sqrt(fy_grid**2 + fx_grid**2)

            # Apply filter based on type
            cutoff = cutoff_ratio * 0.5  # Scale to Nyquist

            # Default to gaussian since static methods can't access self
            _ft = filter_type if filter_type is not None else "gaussian"

            if _ft == "gaussian":
                sigma = cutoff / 2.0
                filter_mask = jnp.exp(-(freq_magnitude**2) / (2 * sigma**2 + 1e-8))
            elif _ft == "butterworth":
                order = 4
                filter_mask = 1.0 / (1.0 + (freq_magnitude / (cutoff + 1e-8)) ** (2 * order))
            elif _ft == "ideal":
                filter_mask = (freq_magnitude <= cutoff).astype(jnp.float32)
            else:
                raise ValueError(f"Unknown filter type: {_ft}")

            return filter_mask

    class JaxResolutionAdapter(fnn.Module):
        """JAX/Flax adapter for resolution-independent inference.

        Handles the transition between training resolution and inference
        resolution by:
        1. Spectral interpolation of positional features
        2. Anti-aliasing filtering
        3. Normalization adjustment for Monte Carlo integrals

        This is the Flax equivalent of :class:`ResolutionAdapter`.

        Attributes:
            base_resolution: Training resolution (``None`` = adaptive).
            filter_cutoff: Spectral filter cutoff ratio.
            filter_type: Filter type for anti-aliasing (``"gaussian"``,
                ``"butterworth"``, ``"ideal"``).

        """

        base_resolution: int | None = None
        filter_cutoff: float = 0.5
        filter_type: str = "gaussian"

        @fnn.compact  # type: ignore[untyped-decorator]
        def __call__(
            self,
            features: Any,
            source_size: int,
            target_size: int,
        ) -> Any:
            """Adapt features from source to target resolution.

            Uses spectral interpolation to resize feature maps while
            preserving frequency content up to the Nyquist limit.

            Args:
                features: JAX array of shape ``(batch, n, d)`` at source
                    resolution.
                source_size: Source board size.
                target_size: Target board size.

            Returns:
                Adapted features at target resolution, shape
                ``(batch, target_size^2, d)``.

            """
            batch, _n, d = features.shape

            if source_size == target_size:
                return features

            # Reshape to 2D spatial format: (batch, d, h, w)
            features_2d = features.reshape(batch, source_size, source_size, d).transpose(0, 3, 1, 2)

            # Spectral interpolation
            features_freq = jnp.fft.rfft2(features_2d)

            # Target frequency dimensions
            target_freq_h = target_size
            target_freq_w = target_size // 2 + 1

            # Create output frequency tensor (zero-padded)
            out_freq = jnp.zeros(
                (batch, d, target_freq_h, target_freq_w),
                dtype=features_freq.dtype,
            )

            # Copy frequencies (zero-pad or truncate)
            src_h, src_w = features_freq.shape[-2], features_freq.shape[-1]
            copy_h = min(src_h // 2, target_freq_h // 2)
            copy_w = min(src_w, target_freq_w)

            # Copy low frequencies (positive)
            out_freq = out_freq.at[:, :, :copy_h, :copy_w].set(
                features_freq[:, :, :copy_h, :copy_w]
            )

            # Copy low frequencies (negative, wrap-around)
            if copy_h > 0:
                out_freq = out_freq.at[:, :, -copy_h:, :copy_w].set(
                    features_freq[:, :, -copy_h:, :copy_w]
                )

            # Inverse FFT to target resolution
            features_target = jnp.fft.irfft2(out_freq, s=(target_size, target_size))

            # Apply anti-aliasing filter if upsampling
            if target_size > source_size:
                cutoff_ratio = source_size / target_size
                filter_mask = JaxSpectralFilter._create_filter_2d(
                    target_size,
                    target_size,
                    jnp.array(cutoff_ratio),
                    filter_type=self.filter_type,
                )
                features_freq_target = jnp.fft.rfft2(features_target)
                features_freq_target = (
                    features_freq_target * filter_mask[jnp.newaxis, jnp.newaxis, ...]
                )
                features_target = jnp.fft.irfft2(features_freq_target, s=(target_size, target_size))

            # Reshape back to sequence format: (batch, d, h, w) -> (batch, h*w, d)
            features_out = features_target.transpose(0, 2, 3, 1).reshape(
                batch, target_size * target_size, d
            )

            # Normalize for Monte Carlo integral consistency
            scale_factor = target_size / source_size
            features_out = features_out * scale_factor

            return features_out


# ===================================================================
# Factory functions
# ===================================================================


def create_spectral_filter(
    cutoff_ratio: float = 0.5,
    filter_type: str = "gaussian",
    learnable: bool = False,
    backend: str = "torch",
) -> Any:
    """Create a spectral filter instance for the specified backend.

    Args:
        cutoff_ratio: Cutoff frequency as ratio of Nyquist (0 to 1).
        filter_type: Type of filter (``"gaussian"``, ``"butterworth"``,
            ``"ideal"``).
        learnable: Whether the cutoff frequency is learnable.
        backend: Backend framework to use (``"torch"`` or ``"jax"``).

    Returns:
        A ``SpectralFilter`` (PyTorch ``nn.Module``) when ``backend="torch"``,
        or a ``JaxSpectralFilter`` (Flax ``nn.Module``) when ``backend="jax"``.

    Raises:
        ImportError: If ``backend="jax"`` but JAX/Flax are not installed.
        ValueError: If an unknown backend name is provided.

    """
    log = logger.bind(
        factory="create_spectral_filter",
        backend=backend,
        cutoff_ratio=cutoff_ratio,
        filter_type=filter_type,
        learnable=learnable,
    )

    if backend == "jax":
        if not HAS_JAX:
            raise ImportError(
                "JAX and Flax are required for the 'jax' backend. "
                "Install with: pip install 'alphagalerkin[jax]'"
            )
        log.info("factory.created", cls="JaxSpectralFilter")
        return JaxSpectralFilter(
            cutoff_ratio=cutoff_ratio,
            filter_type=filter_type,
            learnable=learnable,
        )

    if backend == "torch":
        log.info("factory.created", cls="SpectralFilter")
        return SpectralFilter(
            cutoff_ratio=cutoff_ratio,
            filter_type=filter_type,
            learnable=learnable,
        )

    msg = f"Unknown backend: {backend!r}. Supported: 'torch', 'jax'"
    raise ValueError(msg)


def create_resolution_adapter(
    base_resolution: int | None = None,
    filter_cutoff: float = 0.5,
    filter_type: str = "gaussian",
    backend: str = "torch",
) -> Any:
    """Create a resolution adapter instance for the specified backend.

    Args:
        base_resolution: Training resolution (``None`` = adaptive).
        filter_cutoff: Spectral filter cutoff ratio.
        filter_type: Filter type for anti-aliasing (only used for
            ``"jax"`` backend; the PyTorch version uses its internal
            ``SpectralFilter`` default).
        backend: Backend framework to use (``"torch"`` or ``"jax"``).

    Returns:
        A ``ResolutionAdapter`` (PyTorch ``nn.Module``) when
        ``backend="torch"``, or a ``JaxResolutionAdapter`` (Flax
        ``nn.Module``) when ``backend="jax"``.

    Raises:
        ImportError: If ``backend="jax"`` but JAX/Flax are not installed.
        ValueError: If an unknown backend name is provided.

    """
    log = logger.bind(
        factory="create_resolution_adapter",
        backend=backend,
        base_resolution=base_resolution,
        filter_cutoff=filter_cutoff,
        filter_type=filter_type,
    )

    if backend == "jax":
        if not HAS_JAX:
            raise ImportError(
                "JAX and Flax are required for the 'jax' backend. "
                "Install with: pip install 'alphagalerkin[jax]'"
            )
        log.info("factory.created", cls="JaxResolutionAdapter")
        return JaxResolutionAdapter(
            base_resolution=base_resolution,
            filter_cutoff=filter_cutoff,
            filter_type=filter_type,
        )

    if backend == "torch":
        log.info("factory.created", cls="ResolutionAdapter")
        return ResolutionAdapter(
            base_resolution=base_resolution,
            filter_cutoff=filter_cutoff,
        )

    msg = f"Unknown backend: {backend!r}. Supported: 'torch', 'jax'"
    raise ValueError(msg)
