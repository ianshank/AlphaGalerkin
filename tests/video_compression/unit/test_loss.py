"""Unit tests for video compression loss functions.

Tests rate-distortion and compression losses:
- DistortionLoss: MSE, MS-SSIM, and mixed metrics
- RDLoss: Rate-distortion tradeoff
- CompressionLoss: Complete loss with perceptual component
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.video_compression.training.loss import (
    CompressionLoss,
    DistortionLoss,
    LossOutput,
    RDLoss,
)


class TestDistortionLoss:
    """Tests for DistortionLoss module."""

    @pytest.fixture
    def images(self) -> tuple[Tensor, Tensor]:
        """Create test image pair."""
        # Target and slightly noisy prediction
        target = torch.rand(2, 3, 64, 64)
        pred = target + torch.randn_like(target) * 0.1
        pred = pred.clamp(0, 1)
        return pred, target

    def test_mse_mode(self, images: tuple[Tensor, Tensor]) -> None:
        """Test MSE-only distortion."""
        pred, target = images
        loss = DistortionLoss(metric="mse")

        distortion, mse, ms_ssim_loss = loss(pred, target)

        assert distortion.shape == ()  # Scalar
        assert mse.shape == ()
        assert ms_ssim_loss is None
        # MSE should equal distortion in MSE mode
        torch.testing.assert_close(distortion, mse)

    def test_ms_ssim_mode(self, images: tuple[Tensor, Tensor]) -> None:
        """Test MS-SSIM-only distortion."""
        pred, target = images
        loss = DistortionLoss(metric="ms_ssim")

        distortion, mse, ms_ssim_loss = loss(pred, target)

        assert distortion.shape == ()
        assert mse.shape == ()
        assert ms_ssim_loss is not None
        # MS-SSIM should equal distortion in MS-SSIM mode
        torch.testing.assert_close(distortion, ms_ssim_loss)

    def test_mixed_mode(self, images: tuple[Tensor, Tensor]) -> None:
        """Test mixed MSE + MS-SSIM distortion."""
        pred, target = images
        weight = 0.84
        loss = DistortionLoss(metric="mixed", ms_ssim_weight=weight)

        distortion, mse, ms_ssim_loss = loss(pred, target)

        assert distortion.shape == ()
        assert mse.shape == ()
        assert ms_ssim_loss is not None

        # Check mixed formula
        expected = weight * ms_ssim_loss + (1 - weight) * mse
        torch.testing.assert_close(distortion, expected)

    def test_gradient_flow(self, images: tuple[Tensor, Tensor]) -> None:
        """Test gradient flow through all modes."""
        pred, target = images
        pred.requires_grad_(True)

        for mode in ["mse", "ms_ssim", "mixed"]:
            loss = DistortionLoss(metric=mode)
            distortion, _, _ = loss(pred, target)
            distortion.backward(retain_graph=True)

            assert pred.grad is not None
            assert not torch.isnan(pred.grad).any()
            pred.grad.zero_()

    def test_identical_images_low_loss(self) -> None:
        """Test that identical images produce low loss."""
        target = torch.rand(1, 3, 64, 64)
        pred = target.clone()

        for mode in ["mse", "ms_ssim", "mixed"]:
            loss = DistortionLoss(metric=mode)
            distortion, mse, _ = loss(pred, target)

            # MSE should be ~0 for identical images
            assert mse.item() < 1e-6

    def test_different_images_high_loss(self) -> None:
        """Test that different images produce higher loss."""
        target = torch.zeros(1, 3, 64, 64)
        pred = torch.ones(1, 3, 64, 64)

        loss = DistortionLoss(metric="mse")
        distortion, mse, _ = loss(pred, target)

        # MSE should be 1.0 for completely different images
        assert mse.item() > 0.9


class TestRDLoss:
    """Tests for RDLoss module."""

    @pytest.fixture
    def loss_fn(self) -> RDLoss:
        """Create R-D loss function."""
        return RDLoss(
            lambda_rd=0.01,
            distortion_metric="mixed",
            ms_ssim_weight=0.84,
        )

    @pytest.fixture
    def sample_data(self) -> tuple[Tensor, Tensor, Tensor]:
        """Create sample data for testing."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        rate = torch.tensor([1000.0, 1500.0])  # Bits
        return pred, target, rate

    def test_forward_returns_loss_output(
        self, loss_fn: RDLoss, sample_data: tuple[Tensor, Tensor, Tensor]
    ) -> None:
        """Test that forward returns LossOutput."""
        pred, target, rate = sample_data
        output = loss_fn(pred, target, rate)

        assert isinstance(output, LossOutput)
        assert output.total is not None
        assert output.rate is not None
        assert output.distortion is not None
        assert output.mse is not None
        assert output.psnr is not None

    def test_lambda_rd_scaling(self) -> None:
        """Test that lambda_rd scales rate contribution."""
        pred = torch.rand(1, 3, 64, 64)
        target = pred.clone()  # Same image -> low distortion
        rate = torch.tensor([4096.0])

        loss_low = RDLoss(lambda_rd=0.001)
        loss_high = RDLoss(lambda_rd=0.1)

        output_low = loss_low(pred, target, rate)
        output_high = loss_high(pred, target, rate)

        # Higher lambda should produce higher total loss (due to rate)
        assert output_high.total > output_low.total

    def test_rate_bpp_computation(
        self, loss_fn: RDLoss, sample_data: tuple[Tensor, Tensor, Tensor]
    ) -> None:
        """Test rate is converted to bits per pixel."""
        pred, target, rate = sample_data
        output = loss_fn(pred, target, rate)

        # Rate should be in reasonable BPP range
        # Original rate is in bits, h*w=64*64=4096
        expected_bpp = rate.mean() / 4096
        torch.testing.assert_close(output.rate, expected_bpp)

    def test_psnr_computation(
        self, loss_fn: RDLoss, sample_data: tuple[Tensor, Tensor, Tensor]
    ) -> None:
        """Test PSNR is computed."""
        pred, target, rate = sample_data
        output = loss_fn(pred, target, rate)

        # PSNR should be positive and finite
        assert output.psnr.item() > 0
        assert not torch.isinf(output.psnr)

    def test_gradient_flow(
        self, loss_fn: RDLoss, sample_data: tuple[Tensor, Tensor, Tensor]
    ) -> None:
        """Test gradient flow through loss."""
        pred, target, rate = sample_data
        pred = pred.clone().requires_grad_(True)
        rate = rate.clone().requires_grad_(True)

        output = loss_fn(pred, target, rate)
        output.total.backward()

        assert pred.grad is not None
        assert rate.grad is not None

    def test_ms_ssim_included(
        self, loss_fn: RDLoss, sample_data: tuple[Tensor, Tensor, Tensor]
    ) -> None:
        """Test MS-SSIM is included in output."""
        pred, target, rate = sample_data
        output = loss_fn(pred, target, rate)

        assert output.ms_ssim is not None


class TestCompressionLoss:
    """Tests for CompressionLoss module."""

    @pytest.fixture
    def sample_data(self) -> tuple[Tensor, Tensor, Tensor]:
        """Create sample data for testing."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        rate = torch.tensor([1000.0, 1500.0])
        return pred, target, rate

    def test_forward_returns_dict(self, sample_data: tuple[Tensor, Tensor, Tensor]) -> None:
        """Test that forward returns dictionary."""
        pred, target, rate = sample_data
        loss = CompressionLoss(use_perceptual=False)

        output = loss(pred, target, rate)

        assert isinstance(output, dict)
        assert "total" in output
        assert "rate" in output
        assert "distortion" in output
        assert "mse" in output
        assert "psnr" in output

    def test_perceptual_loss_enabled(self, sample_data: tuple[Tensor, Tensor, Tensor]) -> None:
        """Test with perceptual loss enabled."""
        pytest.importorskip("torchvision", reason="torchvision not installed")
        pred, target, rate = sample_data
        loss = CompressionLoss(use_perceptual=True, perceptual_weight=0.1)

        output = loss(pred, target, rate)

        assert "perceptual" in output
        assert output["perceptual"].item() >= 0

    def test_perceptual_loss_disabled(self, sample_data: tuple[Tensor, Tensor, Tensor]) -> None:
        """Test with perceptual loss disabled."""
        pred, target, rate = sample_data
        loss = CompressionLoss(use_perceptual=False)

        output = loss(pred, target, rate)

        assert "perceptual" not in output

    def test_ms_ssim_in_output(self, sample_data: tuple[Tensor, Tensor, Tensor]) -> None:
        """Test MS-SSIM loss is included."""
        pred, target, rate = sample_data
        loss = CompressionLoss(distortion_metric="mixed", use_perceptual=False)

        output = loss(pred, target, rate)

        assert "ms_ssim_loss" in output

    def test_total_combines_all_losses(self, sample_data: tuple[Tensor, Tensor, Tensor]) -> None:
        """Test that total includes all components."""
        pytest.importorskip("torchvision", reason="torchvision not installed")
        pred, target, rate = sample_data

        # Without perceptual
        loss_no_perc = CompressionLoss(use_perceptual=False)
        output_no_perc = loss_no_perc(pred, target, rate)

        # With perceptual
        loss_with_perc = CompressionLoss(use_perceptual=True, perceptual_weight=0.1)
        output_with_perc = loss_with_perc(pred, target, rate)

        # Total with perceptual should be higher
        assert output_with_perc["total"] >= output_no_perc["total"]

    def test_gradient_flow(self, sample_data: tuple[Tensor, Tensor, Tensor]) -> None:
        """Test gradient flow through complete loss."""
        pred, target, rate = sample_data
        pred = pred.clone().requires_grad_(True)
        rate = rate.clone().requires_grad_(True)

        loss = CompressionLoss(use_perceptual=False)
        output = loss(pred, target, rate)
        output["total"].backward()

        assert pred.grad is not None
        assert rate.grad is not None

    def test_different_lambda_values(self, sample_data: tuple[Tensor, Tensor, Tensor]) -> None:
        """Test different lambda_rd values."""
        pred, target, rate = sample_data

        for lambda_rd in [0.001, 0.01, 0.1, 1.0]:
            loss = CompressionLoss(lambda_rd=lambda_rd, use_perceptual=False)
            output = loss(pred, target, rate)

            assert output["total"].item() > 0
            assert not torch.isnan(output["total"])


class TestLossOutput:
    """Tests for LossOutput named tuple."""

    def test_fields(self) -> None:
        """Test LossOutput has all expected fields."""
        output = LossOutput(
            total=torch.tensor(1.0),
            rate=torch.tensor(0.5),
            distortion=torch.tensor(0.5),
            mse=torch.tensor(0.1),
            psnr=torch.tensor(30.0),
            ms_ssim=torch.tensor(0.9),
        )

        assert output.total.item() == pytest.approx(1.0)
        assert output.rate.item() == pytest.approx(0.5)
        assert output.distortion.item() == pytest.approx(0.5)
        assert output.mse.item() == pytest.approx(0.1)
        assert output.psnr.item() == pytest.approx(30.0)
        assert output.ms_ssim.item() == pytest.approx(0.9)

    def test_optional_ms_ssim(self) -> None:
        """Test ms_ssim can be None."""
        output = LossOutput(
            total=torch.tensor(1.0),
            rate=torch.tensor(0.5),
            distortion=torch.tensor(0.5),
            mse=torch.tensor(0.1),
            psnr=torch.tensor(30.0),
            ms_ssim=None,
        )

        assert output.ms_ssim is None
