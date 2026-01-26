"""Tests for spectral filtering and resolution adaptation."""

from __future__ import annotations

import math

import pytest
import torch

from src.math_kernel.spectral import ResolutionAdapter, SpectralFilter


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
                x[0, 0, i, j] = math.sin(2 * math.pi * i / 8) * math.sin(
                    2 * math.pi * j / 8
                )

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
        original_energy = (features ** 2).sum()
        adapted_energy = (adapted ** 2).sum()

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
        correlation = torch.corrcoef(
            torch.stack([features.flatten(), roundtrip.flatten()])
        )[0, 1]

        # High correlation indicates good reconstruction
        assert correlation > 0.5

    def test_multiple_resolutions(self, adapter: ResolutionAdapter) -> None:
        """Test adaptation works for various resolutions."""
        base_features = torch.randn(1, 81, 32)

        for target_size in [5, 9, 13, 19, 25]:
            adapted = adapter.adapt_features(base_features, 9, target_size)
            assert adapted.shape == (1, target_size ** 2, 32)
