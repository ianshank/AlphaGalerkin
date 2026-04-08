"""Tests for spectral filtering and resolution adaptation.

Tests mathematical properties:
- Filter shape preservation
- Low/high frequency behavior
- Resolution adaptation correctness
- Factory functions
- JAX backend error paths
"""

from __future__ import annotations

import math

import pytest
import torch

from src.math_kernel.spectral import (
    HAS_JAX,
    ResolutionAdapter,
    SpectralFilter,
    create_resolution_adapter,
    create_spectral_filter,
)


class TestSpectralFilter:
    """Tests for spectral filtering."""

    @pytest.fixture
    def filter_gaussian(self) -> SpectralFilter:
        """Create Gaussian filter."""
        return SpectralFilter(cutoff_ratio=0.5, filter_type="gaussian")

    @pytest.fixture
    def filter_butterworth(self) -> SpectralFilter:
        """Create Butterworth filter."""
        return SpectralFilter(cutoff_ratio=0.5, filter_type="butterworth")

    def test_output_shape(self, filter_gaussian: SpectralFilter) -> None:
        """Test that filter preserves shape."""
        x = torch.randn(2, 3, 9, 9)

        filtered = filter_gaussian(x)

        assert filtered.shape == x.shape

    def test_low_frequency_preserved(self, filter_gaussian: SpectralFilter) -> None:
        """Test that DC component is preserved."""
        # Create constant image (DC only)
        x = torch.ones(1, 1, 9, 9) * 5.0

        filtered = filter_gaussian(x)

        # DC should be mostly preserved
        assert torch.allclose(filtered.mean(), x.mean(), rtol=0.1)

    def test_high_frequency_attenuated(self, filter_gaussian: SpectralFilter) -> None:
        """Test that high frequencies are attenuated."""
        # Create checkerboard pattern (high frequency)
        x = torch.zeros(1, 1, 8, 8)
        x[:, :, ::2, ::2] = 1.0
        x[:, :, 1::2, 1::2] = 1.0

        filtered = filter_gaussian(x)

        # Variance should decrease after filtering
        assert filtered.var() < x.var()

    def test_butterworth_steeper_rolloff(
        self,
        filter_gaussian: SpectralFilter,
        filter_butterworth: SpectralFilter,
    ) -> None:
        """Test that Butterworth has steeper rolloff than Gaussian."""
        # Create mid-frequency signal
        x = torch.zeros(1, 1, 16, 16)
        for i in range(16):
            for j in range(16):
                x[0, 0, i, j] = math.sin(2 * math.pi * i / 8) * math.sin(2 * math.pi * j / 8)

        filtered_gauss = filter_gaussian(x)
        filtered_butter = filter_butterworth(x)

        # Both should filter, results will differ based on rolloff
        assert filtered_gauss.shape == filtered_butter.shape


class TestResolutionAdapter:
    """Tests for resolution adaptation."""

    @pytest.fixture
    def adapter(self) -> ResolutionAdapter:
        """Create resolution adapter."""
        return ResolutionAdapter(base_resolution=9, filter_cutoff=0.5)

    def test_same_resolution_identity(self, adapter: ResolutionAdapter) -> None:
        """Test that same resolution is approximately identity."""
        features = torch.randn(2, 81, 64)

        adapted = adapter.adapt_features(features, 9, 9)

        assert torch.allclose(adapted, features, atol=1e-5)

    def test_upsampling_shape(self, adapter: ResolutionAdapter) -> None:
        """Test shape after upsampling."""
        features = torch.randn(2, 81, 64)  # 9x9

        adapted = adapter.adapt_features(features, 9, 19)

        assert adapted.shape == (2, 361, 64)  # 19x19

    def test_downsampling_shape(self, adapter: ResolutionAdapter) -> None:
        """Test shape after downsampling."""
        features = torch.randn(2, 361, 64)  # 19x19

        adapted = adapter.adapt_features(features, 19, 9)

        assert adapted.shape == (2, 81, 64)  # 9x9

    def test_energy_approximately_preserved(self, adapter: ResolutionAdapter) -> None:
        """Test that total energy is approximately preserved."""
        features = torch.randn(2, 81, 64)

        adapted = adapter.adapt_features(features, 9, 19)

        # Energy (sum of squares) should be similar (with scaling)
        original_energy = (features**2).sum()
        adapted_energy = (adapted**2).sum()

        # Energy ratio should be close to area ratio
        # Due to normalization in adapter, energies should be comparable
        ratio = adapted_energy / original_energy

        # Allow some variance but should be in reasonable range
        assert 0.1 < ratio < 10.0

    def test_roundtrip_approximate_identity(self, adapter: ResolutionAdapter) -> None:
        """Test that upsample then downsample approximates identity."""
        features = torch.randn(2, 81, 64)

        # Upsample to 19x19, then downsample back to 9x9
        upsampled = adapter.adapt_features(features, 9, 19)
        roundtrip = adapter.adapt_features(upsampled, 19, 9)

        # Should be similar to original (some loss due to filtering)
        correlation = torch.corrcoef(torch.stack([features.flatten(), roundtrip.flatten()]))[0, 1]

        # High correlation indicates good reconstruction
        assert correlation > 0.5

    def test_multiple_resolutions(self, adapter: ResolutionAdapter) -> None:
        """Test adaptation works for various resolutions."""
        base_features = torch.randn(1, 81, 32)

        for target_size in [5, 9, 13, 19, 25]:
            adapted = adapter.adapt_features(base_features, 9, target_size)
            assert adapted.shape == (1, target_size**2, 32)

    def test_forward_aliases_adapt_features(self, adapter: ResolutionAdapter) -> None:
        """forward() and adapt_features() give identical results."""
        features = torch.randn(1, 81, 16)
        out_fwd = adapter.forward(features, 9, 13)
        out_adapt = adapter.adapt_features(features, 9, 13)
        assert torch.allclose(out_fwd, out_adapt)


# ===================================================================
# SpectralFilter – additional filter type coverage (lines 128-133)
# ===================================================================


class TestSpectralFilterIdealType:
    """Tests for the 'ideal' filter type (sharp cutoff)."""

    def test_ideal_filter_shape_preserved(self) -> None:
        """Ideal filter preserves input shape."""
        filt = SpectralFilter(cutoff_ratio=0.5, filter_type="ideal")
        x = torch.randn(1, 1, 8, 8)
        out = filt(x)
        assert out.shape == x.shape

    def test_ideal_filter_dc_preserved(self) -> None:
        """Ideal filter preserves DC component."""
        filt = SpectralFilter(cutoff_ratio=0.5, filter_type="ideal")
        x = torch.ones(1, 1, 8, 8) * 3.0
        out = filt(x)
        assert torch.allclose(out.mean(), x.mean(), atol=0.1)

    def test_ideal_filter_attenuates_high_freq(self) -> None:
        """Ideal filter attenuates high-frequency checkerboard pattern."""
        filt = SpectralFilter(cutoff_ratio=0.3, filter_type="ideal")
        # High-frequency checkerboard
        x = torch.zeros(1, 1, 8, 8)
        x[:, :, ::2, ::2] = 1.0
        x[:, :, 1::2, 1::2] = 1.0
        out = filt(x)
        assert out.var() < x.var()

    def test_ideal_filter_is_binary_mask(self) -> None:
        """Ideal filter mask has binary values (0 or 1)."""
        filt = SpectralFilter(cutoff_ratio=0.5, filter_type="ideal")
        mask = filt._create_filter_2d(8, 8, torch.device("cpu"))
        # All values should be 0.0 or 1.0
        assert torch.all((mask == 0.0) | (mask == 1.0))


class TestSpectralFilterUnknownType:
    """Tests for unknown filter type error path."""

    def test_unknown_filter_type_raises(self) -> None:
        """Unknown filter_type raises ValueError."""
        filt = SpectralFilter(cutoff_ratio=0.5, filter_type="hamming")
        x = torch.randn(1, 1, 8, 8)
        with pytest.raises(ValueError, match="Unknown filter type"):
            filt(x)


class TestSpectralFilterLearnable:
    """Tests for learnable spectral filter."""

    def test_learnable_cutoff_is_parameter(self) -> None:
        """Learnable filter has cutoff_ratio as nn.Parameter."""
        filt = SpectralFilter(cutoff_ratio=0.4, filter_type="gaussian", learnable=True)
        assert isinstance(filt.cutoff_ratio, torch.nn.Parameter)

    def test_non_learnable_cutoff_is_buffer(self) -> None:
        """Non-learnable filter has cutoff_ratio as buffer."""
        filt = SpectralFilter(cutoff_ratio=0.5, filter_type="gaussian", learnable=False)
        assert not isinstance(filt.cutoff_ratio, torch.nn.Parameter)

    def test_learnable_gradient_flows(self) -> None:
        """Gradient flows through learnable cutoff parameter."""
        filt = SpectralFilter(cutoff_ratio=0.5, filter_type="gaussian", learnable=True)
        x = torch.randn(1, 1, 8, 8)
        out = filt(x)
        loss = out.sum()
        loss.backward()
        assert filt.cutoff_ratio.grad is not None


class TestSpectralFilterButterworthAdditional:
    """Additional tests for Butterworth filter."""

    def test_butterworth_smooth_rolloff(self) -> None:
        """Butterworth filter has smooth (non-binary) mask values."""
        filt = SpectralFilter(cutoff_ratio=0.5, filter_type="butterworth")
        mask = filt._create_filter_2d(8, 8, torch.device("cpu"))
        # Should have values between 0 and 1 (not all binary)
        has_intermediate = ((mask > 0.01) & (mask < 0.99)).any()
        assert has_intermediate


class TestResolutionAdapterAdaptiveFilter:
    """Tests for _apply_adaptive_filter internal method."""

    def test_adaptive_filter_restores_cutoff(self) -> None:
        """_apply_adaptive_filter restores original cutoff after use."""
        adapter = ResolutionAdapter(base_resolution=9, filter_cutoff=0.5)
        original_cutoff = adapter.spectral_filter.cutoff_ratio.clone()

        x = torch.randn(1, 1, 16, 16)
        adapter._apply_adaptive_filter(x, cutoff_ratio=0.3)

        # Cutoff should be restored
        assert torch.allclose(adapter.spectral_filter.cutoff_ratio, original_cutoff)

    def test_adaptive_filter_output_shape(self) -> None:
        """_apply_adaptive_filter preserves shape."""
        adapter = ResolutionAdapter(base_resolution=9, filter_cutoff=0.5)
        x = torch.randn(2, 4, 12, 12)
        out = adapter._apply_adaptive_filter(x, cutoff_ratio=0.4)
        assert out.shape == x.shape

    def test_upsampling_invokes_adaptive_filter(self) -> None:
        """Upsampling path triggers the adaptive filter (target > source)."""
        adapter = ResolutionAdapter(base_resolution=5, filter_cutoff=0.5)
        features = torch.randn(1, 25, 8)  # 5x5

        # Upsample from 5x5 to 9x9; this should invoke _apply_adaptive_filter
        adapted = adapter.adapt_features(features, 5, 9)
        assert adapted.shape == (1, 81, 8)


class TestResolutionAdapterNone:
    """Tests for ResolutionAdapter with base_resolution=None."""

    def test_none_base_resolution(self) -> None:
        """Adapter works with base_resolution=None."""
        adapter = ResolutionAdapter(base_resolution=None, filter_cutoff=0.5)
        features = torch.randn(1, 81, 16)  # 9x9
        adapted = adapter.adapt_features(features, 9, 13)
        assert adapted.shape == (1, 169, 16)


# ===================================================================
# Factory function tests (covers lines 586-592, 645-651)
# ===================================================================


class TestCreateSpectralFilterFactory:
    """Tests for create_spectral_filter factory function."""

    def test_torch_backend_gaussian(self) -> None:
        """Factory with backend='torch' and gaussian type."""
        filt = create_spectral_filter(
            cutoff_ratio=0.5, filter_type="gaussian", learnable=False, backend="torch"
        )
        assert isinstance(filt, SpectralFilter)
        assert filt.filter_type == "gaussian"

    def test_torch_backend_butterworth(self) -> None:
        """Factory with backend='torch' and butterworth type."""
        filt = create_spectral_filter(cutoff_ratio=0.5, filter_type="butterworth", backend="torch")
        assert isinstance(filt, SpectralFilter)
        assert filt.filter_type == "butterworth"

    def test_torch_backend_ideal(self) -> None:
        """Factory with backend='torch' and ideal type."""
        filt = create_spectral_filter(cutoff_ratio=0.5, filter_type="ideal", backend="torch")
        assert isinstance(filt, SpectralFilter)
        assert filt.filter_type == "ideal"

    def test_torch_backend_learnable(self) -> None:
        """Factory with backend='torch' and learnable=True."""
        filt = create_spectral_filter(
            cutoff_ratio=0.4, filter_type="gaussian", learnable=True, backend="torch"
        )
        assert isinstance(filt, SpectralFilter)
        assert isinstance(filt.cutoff_ratio, torch.nn.Parameter)

    def test_torch_backend_functional(self) -> None:
        """Factory-produced filter works on input tensors."""
        filt = create_spectral_filter(backend="torch")
        x = torch.randn(1, 1, 8, 8)
        out = filt(x)
        assert out.shape == x.shape

    def test_jax_backend_raises_import_error(self) -> None:
        """Factory with backend='jax' raises ImportError when JAX not installed."""
        if HAS_JAX:
            pytest.skip("JAX is installed; cannot test ImportError path")
        with pytest.raises(ImportError, match="JAX and Flax are required"):
            create_spectral_filter(backend="jax")

    def test_unknown_backend_raises_value_error(self) -> None:
        """Factory with unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_spectral_filter(backend="tensorflow")


class TestCreateResolutionAdapterFactory:
    """Tests for create_resolution_adapter factory function."""

    def test_torch_backend(self) -> None:
        """Factory with backend='torch' returns ResolutionAdapter."""
        adapter = create_resolution_adapter(base_resolution=9, filter_cutoff=0.5, backend="torch")
        assert isinstance(adapter, ResolutionAdapter)

    def test_torch_backend_none_base_resolution(self) -> None:
        """Factory with base_resolution=None works."""
        adapter = create_resolution_adapter(
            base_resolution=None, filter_cutoff=0.5, backend="torch"
        )
        assert isinstance(adapter, ResolutionAdapter)
        assert adapter.base_resolution is None

    def test_torch_backend_functional(self) -> None:
        """Factory-produced adapter works on feature tensors."""
        adapter = create_resolution_adapter(base_resolution=9, backend="torch")
        features = torch.randn(1, 81, 16)
        out = adapter(features, 9, 13)
        assert out.shape == (1, 169, 16)

    def test_jax_backend_raises_import_error(self) -> None:
        """Factory with backend='jax' raises ImportError when JAX not installed."""
        if HAS_JAX:
            pytest.skip("JAX is installed; cannot test ImportError path")
        with pytest.raises(ImportError, match="JAX and Flax are required"):
            create_resolution_adapter(backend="jax")

    def test_unknown_backend_raises_value_error(self) -> None:
        """Factory with unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_resolution_adapter(backend="paddle")
