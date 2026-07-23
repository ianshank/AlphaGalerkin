"""Unit tests for video quality metrics (PSNR, SSIM).

Validates quality metric computation with known reference pairs
to ensure accurate quality reporting in the E2E workflow.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def identical_frames() -> tuple[torch.Tensor, torch.Tensor]:
    """Create identical frame pair (should give perfect scores)."""
    frame = torch.rand(1, 3, 64, 64)
    return frame, frame.clone()


@pytest.fixture
def different_frames() -> tuple[torch.Tensor, torch.Tensor]:
    """Create different frame pair (should give lower scores)."""
    frame1 = torch.zeros(1, 3, 64, 64)
    frame2 = torch.ones(1, 3, 64, 64)
    return frame1, frame2


@pytest.fixture
def noisy_frames() -> tuple[torch.Tensor, torch.Tensor]:
    """Create frame with additive noise (known distortion)."""
    torch.manual_seed(42)
    clean = torch.rand(1, 3, 64, 64)
    noise_std = 0.1  # 10% noise
    noisy = clean + torch.randn_like(clean) * noise_std
    noisy = noisy.clamp(0, 1)
    return clean, noisy


# ============================================================================
# PSNR Tests
# ============================================================================


class TestPSNR:
    """Tests for PSNR (Peak Signal-to-Noise Ratio) computation."""

    def test_psnr_import(self) -> None:
        """Verify PSNR function can be imported."""
        from src.video_compression.metrics.quality import compute_psnr

        assert callable(compute_psnr)

    def test_psnr_identical_images(
        self, identical_frames: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """PSNR of identical images should be infinite (or very high)."""
        from src.video_compression.metrics.quality import compute_psnr

        frame1, frame2 = identical_frames
        psnr = compute_psnr(frame1, frame2)

        # PSNR should be very high (or inf) for identical images
        assert psnr > 40 or math.isinf(psnr)

    def test_psnr_different_images(
        self, different_frames: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """PSNR of completely different images should be low."""
        from src.video_compression.metrics.quality import compute_psnr

        frame1, frame2 = different_frames
        psnr = compute_psnr(frame1, frame2)

        # Black vs white gives MSE = 1, PSNR = 0 dB
        assert psnr < 5

    def test_psnr_noisy_images(self, noisy_frames: tuple[torch.Tensor, torch.Tensor]) -> None:
        """PSNR with 10% noise should be around 20 dB."""
        from src.video_compression.metrics.quality import compute_psnr

        clean, noisy = noisy_frames
        psnr = compute_psnr(clean, noisy)

        # 10% noise (std=0.1) gives roughly 20dB PSNR
        # PSNR = 20 * log10(1/0.1) = 20 dB
        assert 15 < psnr < 25, f"Unexpected PSNR: {psnr}"

    def test_psnr_symmetry(self, noisy_frames: tuple[torch.Tensor, torch.Tensor]) -> None:
        """PSNR should be symmetric: PSNR(a,b) == PSNR(b,a)."""
        from src.video_compression.metrics.quality import compute_psnr

        frame1, frame2 = noisy_frames
        psnr_ab = compute_psnr(frame1, frame2)
        psnr_ba = compute_psnr(frame2, frame1)

        assert abs(psnr_ab - psnr_ba) < 0.001

    def test_psnr_batch_support(self) -> None:
        """PSNR should support batched inputs."""
        from src.video_compression.metrics.quality import compute_psnr

        batch = torch.rand(4, 3, 64, 64)
        noisy = batch + torch.randn_like(batch) * 0.1
        noisy = noisy.clamp(0, 1)

        psnr = compute_psnr(batch, noisy)
        assert isinstance(psnr, (float, torch.Tensor))


# ============================================================================
# SSIM Tests
# ============================================================================


class TestSSIM:
    """Tests for SSIM (Structural Similarity Index) computation."""

    def test_ssim_import(self) -> None:
        """Verify SSIM function can be imported."""
        from src.video_compression.metrics.quality import compute_ssim

        assert callable(compute_ssim)

    def test_ssim_identical_images(
        self, identical_frames: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """SSIM of identical images should be 1.0."""
        from src.video_compression.metrics.quality import compute_ssim

        frame1, frame2 = identical_frames
        ssim = compute_ssim(frame1, frame2)

        assert abs(ssim - 1.0) < 0.001

    def test_ssim_different_images(
        self, different_frames: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """SSIM of completely different images should be low."""
        from src.video_compression.metrics.quality import compute_ssim

        frame1, frame2 = different_frames
        ssim = compute_ssim(frame1, frame2)

        # Very different images should have low SSIM
        assert ssim < 0.5

    def test_ssim_noisy_images(self, noisy_frames: tuple[torch.Tensor, torch.Tensor]) -> None:
        """SSIM with moderate noise should be moderate."""
        from src.video_compression.metrics.quality import compute_ssim

        clean, noisy = noisy_frames
        ssim = compute_ssim(clean, noisy)

        # 10% noise should give SSIM around 0.6-0.9
        assert 0.5 < ssim < 0.95, f"Unexpected SSIM: {ssim}"

    def test_ssim_range(self) -> None:
        """SSIM should always be in [-1, 1] range."""
        from src.video_compression.metrics.quality import compute_ssim

        for _ in range(10):
            frame1 = torch.rand(1, 3, 64, 64)
            frame2 = torch.rand(1, 3, 64, 64)
            ssim = compute_ssim(frame1, frame2)
            assert -1 <= ssim <= 1


# ============================================================================
# MS-SSIM Tests
# ============================================================================


class TestMSSSIM:
    """Tests for MS-SSIM computation."""

    def test_ms_ssim_nan_stability(self) -> None:
        """Test MS-SSIM handles identical/uniform patches without producing NaNs."""
        from src.video_compression.metrics.quality import compute_ms_ssim

        # Create two identical completely uniform frames.
        # This causes variance to be 0, which can result in slightly negative
        # values due to floating-point truncation, which causes NaNs when raised to a fractional power.
        frame = torch.zeros(1, 3, 256, 256)

        score = compute_ms_ssim(frame, frame)
        assert not torch.isnan(score).any(), "MS-SSIM produced NaN on uniform images"


# ============================================================================
# Integration Tests
# ============================================================================


class TestQualityMetricsIntegration:
    """Integration tests for quality metrics with video frames."""

    def test_metrics_with_opencv_frame(self) -> None:
        """Test metrics work with OpenCV-style frames (HWC format)."""
        from src.video_compression.metrics.quality import compute_psnr, compute_ssim

        # Simulate OpenCV frame (H, W, C) converted to tensor
        cv_frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        frame_tensor = torch.from_numpy(cv_frame).float() / 255.0

        # Convert to (B, C, H, W) format
        frame_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0)

        # Add noise
        noisy = frame_bchw + torch.randn_like(frame_bchw) * 0.05
        noisy = noisy.clamp(0, 1)

        # Compute metrics
        psnr = compute_psnr(frame_bchw, noisy)
        ssim = compute_ssim(frame_bchw, noisy)

        assert psnr > 20  # 5% noise should give ~26dB
        assert ssim > 0.8

    def test_metrics_match_expected_values(self) -> None:
        """Test that computed metrics match expected analytical values."""
        from src.video_compression.metrics.quality import compute_psnr

        # Create images with known MSE
        # If MSE = 0.01, PSNR = 10*log10(1/0.01) = 20 dB
        frame1 = torch.zeros(1, 3, 64, 64)
        mse_target = 0.01
        frame2 = torch.full((1, 3, 64, 64), math.sqrt(mse_target))

        psnr = compute_psnr(frame1, frame2)
        expected_psnr = 10 * math.log10(1.0 / mse_target)

        # Should be close (within 1dB)
        assert abs(psnr - expected_psnr) < 1.0
