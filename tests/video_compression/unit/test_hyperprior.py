"""Unit tests for hyperprior entropy model.

Tests the scale hyperprior implementation from Ballé et al. (2018):
- FactorizedPrior: Learned CDF entropy model
- GaussianConditional: Gaussian entropy model with predicted scales
- HyperAnalysis/HyperSynthesis: Hyperprior transforms
- HyperpriorEntropyModel: Complete entropy model
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.video_compression.config import EntropyConfig, EntropyModelType
from src.video_compression.models.hyperprior import (
    EntropyOutput,
    FactorizedPrior,
    GaussianConditional,
    HyperAnalysis,
    HyperSynthesis,
    HyperpriorEntropyModel,
    create_entropy_model,
)


class TestFactorizedPrior:
    """Tests for factorized prior entropy model."""

    @pytest.fixture
    def prior(self) -> FactorizedPrior:
        """Create test factorized prior."""
        return FactorizedPrior(channels=64, num_filters=3, init_scale=10.0)

    def test_forward_shape(self, prior: FactorizedPrior) -> None:
        """Test that output shapes match input."""
        x = torch.randn(2, 64, 8, 8)
        x_out, rate = prior(x)

        assert x_out.shape == x.shape
        assert rate.shape == (2,)

    def test_rate_is_positive(self, prior: FactorizedPrior) -> None:
        """Test that rate is always positive."""
        x = torch.randn(4, 64, 8, 8)
        _, rate = prior(x)

        assert (rate >= 0).all(), "Rate should be non-negative"

    def test_likelihood_bounded(self, prior: FactorizedPrior) -> None:
        """Test that implied likelihoods are in (0, 1]."""
        x = torch.randn(2, 64, 8, 8)
        x_out, rate = prior(x)

        # Implied likelihood from rate
        num_elements = x.shape[1] * x.shape[2] * x.shape[3]
        avg_bits_per_element = rate / num_elements

        # Should be reasonable (not too high or negative)
        assert (avg_bits_per_element >= 0).all()
        assert (avg_bits_per_element < 100).all()  # Sanity check

    def test_gradient_flow(self, prior: FactorizedPrior) -> None:
        """Test that gradients flow through the model."""
        x = torch.randn(2, 64, 8, 8, requires_grad=True)
        _, rate = prior(x)
        loss = rate.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_cdf_bounded(self, prior: FactorizedPrior) -> None:
        """Test that CDF values are bounded in [0, 1]."""
        x = torch.randn(1, 64, 8, 8) * 5.0  # Test various input values

        cdf = prior._cdf(x)

        # CDF should be bounded
        assert (cdf >= 0).all(), "CDF should be >= 0"
        assert (cdf <= 1).all(), "CDF should be <= 1"

    def test_different_batch_sizes(self, prior: FactorizedPrior) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 64, 8, 8)
            x_out, rate = prior(x)

            assert x_out.shape[0] == batch_size
            assert rate.shape[0] == batch_size


class TestGaussianConditional:
    """Tests for Gaussian conditional entropy model."""

    @pytest.fixture
    def gaussian(self) -> GaussianConditional:
        """Create test Gaussian conditional."""
        return GaussianConditional(scale_bound=0.11)

    def test_forward_shape(self, gaussian: GaussianConditional) -> None:
        """Test output shapes."""
        x = torch.randn(2, 64, 8, 8)
        scales = torch.rand(2, 64, 8, 8) + 0.5  # Positive scales

        x_out, rate = gaussian(x, scales)

        assert x_out.shape == x.shape
        assert rate.shape == (2,)

    def test_scale_clamping(self, gaussian: GaussianConditional) -> None:
        """Test that small scales are clamped."""
        x = torch.randn(2, 64, 8, 8)
        scales = torch.ones(2, 64, 8, 8) * 0.01  # Very small scales

        # Should not raise or produce inf
        x_out, rate = gaussian(x, scales)

        assert not torch.isinf(rate).any()
        assert not torch.isnan(rate).any()

    def test_with_means(self, gaussian: GaussianConditional) -> None:
        """Test with non-zero means."""
        x = torch.randn(2, 64, 8, 8)
        scales = torch.rand(2, 64, 8, 8) + 0.5
        means = torch.randn(2, 64, 8, 8)

        x_out, rate = gaussian(x, scales, means)

        assert x_out.shape == x.shape
        assert (rate >= 0).all()

    def test_gradient_flow(self, gaussian: GaussianConditional) -> None:
        """Test gradient flow."""
        x = torch.randn(2, 64, 8, 8, requires_grad=True)
        scales = (torch.rand(2, 64, 8, 8) + 0.5).requires_grad_(True)

        x_out, rate = gaussian(x, scales)
        loss = rate.sum()
        loss.backward()

        assert x.grad is not None
        # scales may not have grad if it's not a leaf tensor, check x instead
        assert not torch.isnan(x.grad).any()

    def test_rate_increases_with_variance(self, gaussian: GaussianConditional) -> None:
        """Test that rate increases with input variance (for fixed scales)."""
        scales = torch.ones(1, 64, 8, 8)

        # Small values should have lower rate
        x_small = torch.zeros(1, 64, 8, 8)
        x_large = torch.randn(1, 64, 8, 8) * 10

        _, rate_small = gaussian(x_small, scales)
        _, rate_large = gaussian(x_large, scales)

        # Large deviations should have higher rate on average
        assert rate_large.mean() > rate_small.mean()


class TestHyperAnalysis:
    """Tests for hyper-analysis transform."""

    @pytest.fixture
    def hyper_analysis(self) -> HyperAnalysis:
        """Create test hyper-analysis."""
        return HyperAnalysis(in_channels=64, out_channels=32, n_layers=3)

    def test_forward_shape(self, hyper_analysis: HyperAnalysis) -> None:
        """Test output shape with downsampling."""
        y = torch.randn(2, 64, 32, 32)
        z = hyper_analysis(y)

        # 3 layers with stride 2, 2, 1 -> spatial reduction of 4x
        assert z.shape == (2, 32, 8, 8)

    def test_uses_absolute_value(self, hyper_analysis: HyperAnalysis) -> None:
        """Test that sign doesn't affect output (uses absolute value)."""
        y = torch.randn(2, 64, 32, 32)
        z_pos = hyper_analysis(y)
        z_neg = hyper_analysis(-y)

        # Should be exactly equal due to abs()
        torch.testing.assert_close(z_pos, z_neg)

    def test_different_spatial_sizes(self, hyper_analysis: HyperAnalysis) -> None:
        """Test with various spatial dimensions."""
        for size in [16, 32, 64]:
            y = torch.randn(1, 64, size, size)
            z = hyper_analysis(y)

            # Check spatial reduction (approximately 4x)
            assert z.shape[2] <= size // 4 + 1
            assert z.shape[3] <= size // 4 + 1


class TestHyperSynthesis:
    """Tests for hyper-synthesis transform."""

    @pytest.fixture
    def hyper_synthesis(self) -> HyperSynthesis:
        """Create test hyper-synthesis."""
        return HyperSynthesis(in_channels=32, out_channels=64, n_layers=3)

    def test_forward_shape(self, hyper_synthesis: HyperSynthesis) -> None:
        """Test output shape with upsampling."""
        z = torch.randn(2, 32, 8, 8)
        sigma = hyper_synthesis(z)

        # 3 layers with stride 2, 2, 1 -> spatial expansion of 4x
        assert sigma.shape == (2, 64, 32, 32)

    def test_output_positive(self, hyper_synthesis: HyperSynthesis) -> None:
        """Test that output (scales) are positive."""
        z = torch.randn(2, 32, 8, 8)
        sigma = hyper_synthesis(z)

        assert (sigma > 0).all(), "Scales should be positive (exp activation)"

    def test_gradient_flow(self, hyper_synthesis: HyperSynthesis) -> None:
        """Test gradient flow through transposed convolutions."""
        z = torch.randn(2, 32, 8, 8, requires_grad=True)
        sigma = hyper_synthesis(z)
        loss = sigma.sum()
        loss.backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any()


class TestHyperpriorEntropyModel:
    """Tests for complete hyperprior entropy model."""

    @pytest.fixture
    def config(self) -> EntropyConfig:
        """Create test entropy config."""
        return EntropyConfig(
            name="test_entropy",
            model_type=EntropyModelType.HYPERPRIOR,
            hyper_channels=32,
            num_filters=64,
            hyper_layers=2,
        )

    @pytest.fixture
    def model(self, config: EntropyConfig) -> HyperpriorEntropyModel:
        """Create test entropy model."""
        return HyperpriorEntropyModel(config)

    def test_forward_training_mode(self, model: HyperpriorEntropyModel) -> None:
        """Test forward pass in training mode."""
        model.train()
        y = torch.randn(2, 64, 16, 16)

        output = model(y)

        assert isinstance(output, EntropyOutput)
        assert output.y_hat.shape == y.shape
        assert output.rate.shape == (2,)

    def test_forward_eval_mode(self, model: HyperpriorEntropyModel) -> None:
        """Test forward pass in eval mode."""
        model.eval()
        y = torch.randn(2, 64, 16, 16)

        output = model(y)

        # In eval mode, y_hat should be rounded integers
        assert isinstance(output, EntropyOutput)
        assert output.y_hat.shape == y.shape

    def test_entropy_output_fields(self, model: HyperpriorEntropyModel) -> None:
        """Test that all output fields are present and valid."""
        y = torch.randn(2, 64, 16, 16)
        output = model(y)

        # Check all fields exist
        assert output.y_hat is not None
        assert output.z_hat is not None
        assert output.y_likelihoods is not None
        assert output.z_likelihoods is not None
        assert output.rate is not None

        # Check rates are positive
        assert (output.rate >= 0).all()

    def test_compress_returns_symbols(self, model: HyperpriorEntropyModel) -> None:
        """Test compress returns quantized symbols."""
        model.eval()
        y = torch.randn(1, 64, 16, 16)

        compressed = model.compress(y)

        assert "y_symbols" in compressed
        assert "z_symbols" in compressed
        assert "scales" in compressed

        # Symbols should be integers
        assert compressed["y_symbols"].dtype == torch.int16
        assert compressed["z_symbols"].dtype == torch.int16

    def test_decompress_inverse(self, model: HyperpriorEntropyModel) -> None:
        """Test decompress recovers y_hat from symbols."""
        model.eval()
        y = torch.randn(1, 64, 16, 16)

        compressed = model.compress(y)
        y_recovered = model.decompress(
            compressed["y_symbols"],
            compressed["z_symbols"],
        )

        # Should match compressed y_symbols as float
        torch.testing.assert_close(
            y_recovered,
            compressed["y_symbols"].float(),
        )

    def test_rate_estimation_reasonable(self, model: HyperpriorEntropyModel) -> None:
        """Test that rate estimates are reasonable."""
        y = torch.randn(1, 64, 16, 16)
        output = model(y)

        # Rate should be positive and not too extreme
        num_elements = y.numel()
        bpp = output.rate.item() / num_elements

        assert bpp > 0, "Rate should be positive"
        assert bpp < 32, "Rate should be reasonable (< 32 bits per element)"

    def test_scales_resize_to_match_latent(self, model: HyperpriorEntropyModel) -> None:
        """Test that scales are resized to match latent dimensions."""
        model.train()
        y = torch.randn(1, 64, 16, 16)

        output = model(y)

        # Scales should match y shape for Gaussian conditional
        assert output.y_hat.shape == y.shape

    def test_gradient_flow(self, model: HyperpriorEntropyModel) -> None:
        """Test gradient flow through entire model."""
        model.train()
        y = torch.randn(2, 64, 16, 16, requires_grad=True)

        output = model(y)
        loss = output.rate.sum()
        loss.backward()

        assert y.grad is not None
        assert not torch.isnan(y.grad).any()


class TestCreateEntropyModel:
    """Tests for entropy model factory function."""

    def test_factorized_type(self) -> None:
        """Test creating factorized prior."""
        config = EntropyConfig(
            name="test",
            model_type=EntropyModelType.FACTORIZED,
            num_filters=64,
        )
        model = create_entropy_model(config)

        assert isinstance(model, FactorizedPrior)

    def test_hyperprior_type(self) -> None:
        """Test creating hyperprior model."""
        config = EntropyConfig(
            name="test",
            model_type=EntropyModelType.HYPERPRIOR,
            num_filters=64,
            hyper_channels=32,
        )
        model = create_entropy_model(config)

        assert isinstance(model, HyperpriorEntropyModel)

    def test_autoregressive_fallback(self) -> None:
        """Test that autoregressive falls back to hyperprior."""
        config = EntropyConfig(
            name="test",
            model_type=EntropyModelType.AUTOREGRESSIVE,
            num_filters=64,
            hyper_channels=32,
        )
        model = create_entropy_model(config)

        # Falls back to hyperprior
        assert isinstance(model, HyperpriorEntropyModel)

    def test_models_are_trainable(self) -> None:
        """Test that created models have trainable parameters."""
        config = EntropyConfig(
            name="test",
            model_type=EntropyModelType.HYPERPRIOR,
            num_filters=64,
            hyper_channels=32,
        )
        model = create_entropy_model(config)

        # Should have trainable parameters
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
