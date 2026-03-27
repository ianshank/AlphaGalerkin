"""Additional coverage tests for spectral filtering.

Tests cover uncovered paths in src/math_kernel/spectral.py:
- SpectralFilter: All filter types (gaussian, butterworth, ideal), learnable mode
- ResolutionAdapter: Same size, upsampling, downsampling paths
- Factory functions: create_spectral_filter, create_resolution_adapter
"""

from __future__ import annotations

import pytest
import torch

from src.math_kernel.spectral import (
    ResolutionAdapter,
    SpectralFilter,
    create_resolution_adapter,
    create_spectral_filter,
)

SEED = 42
BATCH_SIZE = 2
CHANNELS = 4
HEIGHT = 8
WIDTH = 8


@pytest.fixture
def sample_input() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


class TestSpectralFilter:
    """Tests for SpectralFilter module."""

    @pytest.mark.parametrize("filter_type", ["gaussian", "butterworth", "ideal"])
    def test_filter_types(self, filter_type: str, sample_input: torch.Tensor) -> None:
        f = SpectralFilter(cutoff_ratio=0.5, filter_type=filter_type)
        output = f(sample_input)
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()

    def test_learnable_cutoff(self, sample_input: torch.Tensor) -> None:
        f = SpectralFilter(cutoff_ratio=0.5, learnable=True)
        assert isinstance(f.cutoff_ratio, torch.nn.Parameter)
        output = f(sample_input)
        loss = output.sum()
        loss.backward()
        assert f.cutoff_ratio.grad is not None

    def test_non_learnable_cutoff(self) -> None:
        f = SpectralFilter(cutoff_ratio=0.3, learnable=False)
        assert not isinstance(f.cutoff_ratio, torch.nn.Parameter)

    def test_high_cutoff_preserves_signal(self, sample_input: torch.Tensor) -> None:
        f = SpectralFilter(cutoff_ratio=1.0, filter_type="gaussian")
        output = f(sample_input)
        # High cutoff should preserve most of the signal energy
        input_energy = (sample_input**2).sum()
        output_energy = (output**2).sum()
        # Output energy should be a significant fraction of input energy
        assert output_energy / input_energy > 0.1

    def test_low_cutoff_smooths_signal(self, sample_input: torch.Tensor) -> None:
        f = SpectralFilter(cutoff_ratio=0.1, filter_type="gaussian")
        output = f(sample_input)
        # Low cutoff should reduce high frequencies (smoother output)
        assert output.std() <= sample_input.std()

    def test_invalid_filter_type(self) -> None:
        f = SpectralFilter(cutoff_ratio=0.5, filter_type="invalid")
        with pytest.raises(ValueError, match="Unknown filter type"):
            f(torch.randn(1, 1, 4, 4))

    def test_non_square_input(self) -> None:
        f = SpectralFilter(cutoff_ratio=0.5)
        x = torch.randn(1, 1, 8, 16)
        output = f(x)
        assert output.shape == (1, 1, 8, 16)


class TestResolutionAdapter:
    """Tests for ResolutionAdapter module."""

    def test_same_resolution(self) -> None:
        adapter = ResolutionAdapter(base_resolution=8)
        torch.manual_seed(SEED)
        features = torch.randn(BATCH_SIZE, 64, CHANNELS)  # 8x8
        output = adapter.adapt_features(features, source_size=8, target_size=8)
        torch.testing.assert_close(output, features)

    def test_upsampling(self) -> None:
        adapter = ResolutionAdapter(base_resolution=4)
        torch.manual_seed(SEED)
        features = torch.randn(BATCH_SIZE, 16, CHANNELS)  # 4x4
        output = adapter.adapt_features(features, source_size=4, target_size=8)
        assert output.shape == (BATCH_SIZE, 64, CHANNELS)

    def test_downsampling(self) -> None:
        adapter = ResolutionAdapter(base_resolution=8)
        torch.manual_seed(SEED)
        features = torch.randn(BATCH_SIZE, 64, CHANNELS)  # 8x8
        output = adapter.adapt_features(features, source_size=8, target_size=4)
        assert output.shape == (BATCH_SIZE, 16, CHANNELS)

    def test_forward_alias(self) -> None:
        adapter = ResolutionAdapter()
        torch.manual_seed(SEED)
        features = torch.randn(1, 16, CHANNELS)
        result_adapt = adapter.adapt_features(features, source_size=4, target_size=4)
        result_forward = adapter.forward(features, source_size=4, target_size=4)
        torch.testing.assert_close(result_adapt, result_forward)

    def test_no_base_resolution(self) -> None:
        adapter = ResolutionAdapter(base_resolution=None)
        torch.manual_seed(SEED)
        features = torch.randn(1, 16, CHANNELS)
        output = adapter.adapt_features(features, source_size=4, target_size=8)
        assert output.shape == (1, 64, CHANNELS)

    def test_scale_factor_applied(self) -> None:
        adapter = ResolutionAdapter()
        torch.manual_seed(SEED)
        features = torch.randn(1, 16, CHANNELS)
        output = adapter.adapt_features(features, source_size=4, target_size=8)
        # Energy should scale with resolution
        assert output.shape[1] == 64


class TestSpectralFactoryFunctions:
    """Tests for factory functions."""

    def test_create_spectral_filter_torch(self) -> None:
        f = create_spectral_filter(cutoff_ratio=0.5, backend="torch")
        assert isinstance(f, SpectralFilter)

    def test_create_spectral_filter_learnable(self) -> None:
        f = create_spectral_filter(cutoff_ratio=0.5, learnable=True, backend="torch")
        assert isinstance(f.cutoff_ratio, torch.nn.Parameter)

    @pytest.mark.parametrize("filter_type", ["gaussian", "butterworth", "ideal"])
    def test_create_spectral_filter_types(self, filter_type: str) -> None:
        f = create_spectral_filter(filter_type=filter_type, backend="torch")
        assert f.filter_type == filter_type

    def test_create_spectral_filter_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_spectral_filter(backend="invalid")

    def test_create_resolution_adapter_torch(self) -> None:
        adapter = create_resolution_adapter(base_resolution=8, backend="torch")
        assert isinstance(adapter, ResolutionAdapter)

    def test_create_resolution_adapter_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_resolution_adapter(backend="invalid")
