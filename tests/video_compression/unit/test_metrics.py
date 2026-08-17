"""Tests for video compression quality metrics."""

import torch

from src.video_compression.metrics.quality import (
    MSSSIM,
    PSNR,
    SSIM,
    compute_ms_ssim,
    compute_psnr,
    compute_ssim,
)
from src.video_compression.metrics.rd_curves import (
    RDCurve,
    RDPoint,
    compute_bd_psnr,
    compute_bd_rate,
)


class TestPSNR:
    """Tests for PSNR computation."""

    def test_identical_images(self) -> None:
        """Test PSNR of identical images is very high."""
        x = torch.rand(2, 3, 64, 64)

        psnr = compute_psnr(x, x)

        # Should be very high (theoretically infinite)
        assert psnr > 50.0

    def test_different_images(self) -> None:
        """Test PSNR of different images is finite."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        psnr = compute_psnr(x, y)

        # Should be reasonable finite value
        assert 0.0 < psnr < 50.0

    def test_psnr_module(self) -> None:
        """Test PSNR module."""
        psnr_fn = PSNR()
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        psnr = psnr_fn(x, y)

        assert psnr.shape == ()  # Scalar
        assert 0.0 < psnr.item() < 50.0

    def test_max_val_scaling(self) -> None:
        """Test PSNR with different max values."""
        x = torch.randint(0, 256, (2, 3, 64, 64)).float()
        y = x + torch.randn_like(x) * 10

        psnr_255 = compute_psnr(x, y, max_val=255.0)
        psnr_1 = compute_psnr(x / 255, y / 255, max_val=1.0)

        # Should be approximately equal
        assert abs(psnr_255.item() - psnr_1.item()) < 0.1


class TestSSIM:
    """Tests for SSIM computation."""

    def test_identical_images(self) -> None:
        """Test SSIM of identical images is 1.0."""
        x = torch.rand(2, 3, 64, 64)

        ssim = compute_ssim(x, x)

        assert ssim > 0.99

    def test_different_images(self) -> None:
        """Test SSIM of different images is less than 1.0."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        ssim = compute_ssim(x, y)

        assert 0.0 < ssim < 1.0

    def test_ssim_range(self) -> None:
        """Test SSIM is in valid range."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        ssim = compute_ssim(x, y)

        # SSIM should be in [-1, 1], typically [0, 1] for real images
        assert -1.0 <= ssim <= 1.0

    def test_ssim_module(self) -> None:
        """Test SSIM module."""
        ssim_fn = SSIM()
        x = torch.rand(2, 3, 64, 64)

        ssim = ssim_fn(x, x)

        assert ssim > 0.99

    def test_ssim_as_loss(self) -> None:
        """Test SSIM as loss (1 - SSIM)."""
        ssim_loss = SSIM(as_loss=True)
        x = torch.rand(2, 3, 64, 64)

        loss = ssim_loss(x, x)

        # Loss should be near 0 for identical images
        assert loss < 0.01

    def test_ssim_gradient(self) -> None:
        """Test SSIM has valid gradients."""
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)

        ssim = compute_ssim(x, y)
        ssim.backward()

        assert x.grad is not None


class TestMSSSIM:
    """Tests for MS-SSIM computation."""

    def test_identical_images(self) -> None:
        """Test MS-SSIM of identical images is near 1.0."""
        x = torch.rand(2, 3, 128, 128)  # Need larger image for multiple scales

        ms_ssim = compute_ms_ssim(x, x)

        assert ms_ssim > 0.99

    def test_ms_ssim_module(self) -> None:
        """Test MS-SSIM module."""
        ms_ssim_fn = MSSSIM()
        x = torch.rand(2, 3, 128, 128)

        ms_ssim = ms_ssim_fn(x, x)

        assert ms_ssim > 0.99

    def test_ms_ssim_as_loss(self) -> None:
        """Test MS-SSIM as loss."""
        ms_ssim_loss = MSSSIM(as_loss=True)
        x = torch.rand(2, 3, 128, 128)

        loss = ms_ssim_loss(x, x)

        assert loss < 0.01

    def test_handles_small_images(self) -> None:
        """Test MS-SSIM handles small images gracefully."""
        x = torch.rand(2, 3, 32, 32)

        # Should not raise, may use fewer scales
        ms_ssim = compute_ms_ssim(x, x)

        assert 0.0 <= ms_ssim <= 1.0


class TestRDCurve:
    """Tests for R-D curve representation."""

    def test_add_point(self) -> None:
        """Test adding points to curve."""
        curve = RDCurve(name="test")

        curve.add_point(RDPoint(rate=0.5, distortion=0.01, psnr=35.0))
        curve.add_point(RDPoint(rate=0.2, distortion=0.02, psnr=30.0))

        assert len(curve.points) == 2
        # Should be sorted by rate
        assert curve.points[0].rate < curve.points[1].rate

    def test_monotonicity_check(self) -> None:
        """Test monotonicity checking."""
        curve = RDCurve(name="test")

        # Monotonic curve
        curve.add_point(RDPoint(rate=0.2, distortion=0.02, psnr=30.0))
        curve.add_point(RDPoint(rate=0.5, distortion=0.01, psnr=35.0))
        curve.add_point(RDPoint(rate=1.0, distortion=0.005, psnr=40.0))

        assert curve.is_monotonic()

    def test_interpolation(self) -> None:
        """Test R-D curve interpolation."""
        curve = RDCurve(name="test")
        curve.add_point(RDPoint(rate=0.2, distortion=0.02, psnr=30.0))
        curve.add_point(RDPoint(rate=0.5, distortion=0.01, psnr=35.0))
        curve.add_point(RDPoint(rate=1.0, distortion=0.005, psnr=40.0))

        # Interpolate at middle point
        psnr_at_0_35 = curve.interpolate(0.35, metric="psnr")

        assert 30.0 < psnr_at_0_35 < 35.0


class TestBDRate:
    """Tests for BD-rate computation."""

    def test_identical_curves(self) -> None:
        """Test BD-rate of identical curves is 0."""
        curve = RDCurve(name="test")
        for rate, psnr in [(0.1, 30), (0.2, 33), (0.4, 36), (0.8, 39)]:
            curve.add_point(RDPoint(rate=rate, distortion=0.01, psnr=psnr))

        bd_rate = compute_bd_rate(curve, curve)

        assert abs(bd_rate) < 1.0  # Should be ~0

    def test_better_curve_negative_bd_rate(self) -> None:
        """Test better curve has negative BD-rate."""
        anchor = RDCurve(name="anchor")
        test = RDCurve(name="test")

        # Anchor curve
        for rate, psnr in [(0.1, 30), (0.2, 33), (0.4, 36), (0.8, 39)]:
            anchor.add_point(RDPoint(rate=rate, distortion=0.01, psnr=psnr))

        # Test curve: same PSNR at half the rate (better)
        for rate, psnr in [(0.05, 30), (0.1, 33), (0.2, 36), (0.4, 39)]:
            test.add_point(RDPoint(rate=rate, distortion=0.01, psnr=psnr))

        bd_rate = compute_bd_rate(anchor, test)

        # Should be significantly negative (better compression)
        assert bd_rate < -30.0

    def test_worse_curve_positive_bd_rate(self) -> None:
        """Test worse curve has positive BD-rate."""
        anchor = RDCurve(name="anchor")
        test = RDCurve(name="test")

        # Anchor curve
        for rate, psnr in [(0.1, 30), (0.2, 33), (0.4, 36), (0.8, 39)]:
            anchor.add_point(RDPoint(rate=rate, distortion=0.01, psnr=psnr))

        # Test curve: same PSNR at double the rate (worse)
        for rate, psnr in [(0.2, 30), (0.4, 33), (0.8, 36), (1.6, 39)]:
            test.add_point(RDPoint(rate=rate, distortion=0.01, psnr=psnr))

        bd_rate = compute_bd_rate(anchor, test)

        # Should be positive (worse compression)
        assert bd_rate > 50.0


class TestBDPSNR:
    """Tests for BD-PSNR computation."""

    def test_better_curve_positive_bd_psnr(self) -> None:
        """Test better curve has positive BD-PSNR."""
        anchor = RDCurve(name="anchor")
        test = RDCurve(name="test")

        # Anchor curve
        for rate, psnr in [(0.1, 30), (0.2, 33), (0.4, 36), (0.8, 39)]:
            anchor.add_point(RDPoint(rate=rate, distortion=0.01, psnr=psnr))

        # Test curve: higher PSNR at same rate (better)
        for rate, psnr in [(0.1, 32), (0.2, 35), (0.4, 38), (0.8, 41)]:
            test.add_point(RDPoint(rate=rate, distortion=0.01, psnr=psnr))

        bd_psnr = compute_bd_psnr(anchor, test)

        # Should be positive (higher quality)
        assert bd_psnr > 1.5
