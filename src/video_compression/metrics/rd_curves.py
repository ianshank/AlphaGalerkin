"""Rate-distortion curve analysis and BD-rate computation.

Provides tools for:
- R-D curve representation
- BD-rate calculation (Bjøntegaard Delta rate)
- BD-PSNR calculation (Bjøntegaard Delta PSNR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Compatibility for numpy.trapz (renamed to trapezoid in numpy 2.0)
try:
    _trapezoid = np.trapezoid  # numpy >= 2.0
except AttributeError:
    _trapezoid = np.trapz  # numpy < 2.0


@dataclass
class RDPoint:
    """A single rate-distortion point."""

    rate: float  # Bits per pixel
    distortion: float  # MSE or similar
    psnr: float | None = None  # PSNR in dB
    ssim: float | None = None  # SSIM value
    vmaf: float | None = None  # VMAF score
    qp: int | None = None  # QP value used
    lambda_rd: float | None = None  # Lambda value used

    @property
    def rate_bpp(self) -> float:
        """Rate in bits per pixel."""
        return self.rate

    def __repr__(self) -> str:
        """String representation."""
        psnr_str = f", PSNR={self.psnr:.2f}dB" if self.psnr else ""
        return f"RDPoint(rate={self.rate:.4f}bpp{psnr_str})"


@dataclass
class RDCurve:
    """Collection of rate-distortion points forming a curve."""

    name: str
    points: list[RDPoint] = field(default_factory=list)

    def add_point(self, point: RDPoint) -> None:
        """Add a point to the curve.

        Args:
            point: RD point to add.

        """
        self.points.append(point)
        # Keep sorted by rate
        self.points.sort(key=lambda p: p.rate)

    @property
    def rates(self) -> np.ndarray:
        """Get rates as numpy array."""
        return np.array([p.rate for p in self.points])

    @property
    def psnrs(self) -> np.ndarray:
        """Get PSNRs as numpy array."""
        return np.array([p.psnr for p in self.points if p.psnr is not None])

    @property
    def ssims(self) -> np.ndarray:
        """Get SSIMs as numpy array."""
        return np.array([p.ssim for p in self.points if p.ssim is not None])

    def is_monotonic(self) -> bool:
        """Check if curve is monotonically increasing.

        Higher rate should give higher quality (PSNR).

        Returns:
            True if curve is monotonic.

        """
        psnrs = self.psnrs
        if len(psnrs) < 2:
            return True
        return all(psnrs[i] <= psnrs[i + 1] for i in range(len(psnrs) - 1))

    def interpolate(
        self,
        rate: float,
        metric: Literal["psnr", "ssim"] = "psnr",
    ) -> float:
        """Interpolate quality at given rate.

        Args:
            rate: Rate value to interpolate at.
            metric: Quality metric to interpolate.

        Returns:
            Interpolated quality value.

        """
        rates = self.rates
        qualities = self.psnrs if metric == "psnr" else self.ssims

        if len(rates) < 2:
            return qualities[0] if len(qualities) > 0 else 0.0

        # Clamp to curve range
        rate = np.clip(rate, rates[0], rates[-1])

        # Linear interpolation
        return float(np.interp(rate, rates, qualities))


def compute_bd_rate(
    anchor: RDCurve,
    test: RDCurve,
    metric: Literal["psnr", "ssim"] = "psnr",
) -> float:
    """Compute Bjøntegaard Delta Rate.

    BD-rate measures the average bitrate difference between two
    codecs at the same quality level.

    Negative BD-rate means test codec is better (lower rate at same quality).

    Args:
        anchor: Reference R-D curve.
        test: Test R-D curve to compare.
        metric: Quality metric to use.

    Returns:
        BD-rate as percentage (e.g., -5.0 means 5% bitrate savings).

    """
    # Get rates and qualities
    anchor_rates = anchor.rates
    anchor_qualities = anchor.psnrs if metric == "psnr" else anchor.ssims
    test_rates = test.rates
    test_qualities = test.psnrs if metric == "psnr" else test.ssims

    # Need at least 4 points for cubic interpolation
    if len(anchor_rates) < 4 or len(test_rates) < 4:
        return _bd_rate_linear(anchor_rates, anchor_qualities, test_rates, test_qualities)

    return _bd_rate_cubic(anchor_rates, anchor_qualities, test_rates, test_qualities)


def _bd_rate_linear(
    anchor_rates: np.ndarray,
    anchor_qualities: np.ndarray,
    test_rates: np.ndarray,
    test_qualities: np.ndarray,
) -> float:
    """Compute BD-rate using linear interpolation.

    Args:
        anchor_rates: Anchor rates.
        anchor_qualities: Anchor qualities.
        test_rates: Test rates.
        test_qualities: Test qualities.

    Returns:
        BD-rate percentage.

    """
    # Find overlapping quality range
    min_quality = max(anchor_qualities.min(), test_qualities.min())
    max_quality = min(anchor_qualities.max(), test_qualities.max())

    if min_quality >= max_quality:
        return 0.0  # No overlap

    # Sample quality points
    num_samples = 100
    quality_samples = np.linspace(min_quality, max_quality, num_samples)

    # Interpolate rates at each quality level
    anchor_rates_interp = np.interp(quality_samples, anchor_qualities, anchor_rates)
    test_rates_interp = np.interp(quality_samples, test_qualities, test_rates)

    # Compute average rate difference (in log domain)
    log_anchor = np.log10(anchor_rates_interp + 1e-10)
    log_test = np.log10(test_rates_interp + 1e-10)

    avg_log_diff = np.mean(log_test - log_anchor)

    # Convert to percentage
    bd_rate = (10**avg_log_diff - 1) * 100

    return float(bd_rate)


def _bd_rate_cubic(
    anchor_rates: np.ndarray,
    anchor_qualities: np.ndarray,
    test_rates: np.ndarray,
    test_qualities: np.ndarray,
) -> float:
    """Compute BD-rate using cubic polynomial fitting.

    Reference: G. Bjøntegaard, "Calculation of average PSNR differences
    between RD-curves," VCEG-M33, 2001.

    Args:
        anchor_rates: Anchor rates.
        anchor_qualities: Anchor qualities.
        test_rates: Test rates.
        test_qualities: Test qualities.

    Returns:
        BD-rate percentage.

    """
    # Work in log-rate domain
    log_anchor_rates = np.log10(anchor_rates + 1e-10)
    log_test_rates = np.log10(test_rates + 1e-10)

    # Fit cubic polynomials: quality = f(log_rate)
    np.polyfit(log_anchor_rates, anchor_qualities, 3)
    np.polyfit(log_test_rates, test_qualities, 3)

    # Find overlapping quality range
    min_quality = max(anchor_qualities.min(), test_qualities.min())
    max_quality = min(anchor_qualities.max(), test_qualities.max())

    if min_quality >= max_quality:
        return 0.0

    # Integrate area under each curve
    # For polynomial p(x), integral = sum(coef[i] * x^(n-i+1) / (n-i+1))
    def integrate_poly(poly_coeffs: np.ndarray, a: float, b: float) -> float:
        """Integrate polynomial from a to b."""
        # Integrate: p(x) -> P(x) where P'(x) = p(x)
        n = len(poly_coeffs)
        integral_coeffs = np.zeros(n + 1)
        for i, c in enumerate(poly_coeffs):
            integral_coeffs[i] = c / (n - i)
        integral_coeffs[-1] = 0  # Constant of integration

        # Evaluate at bounds
        P_b = np.polyval(integral_coeffs, b)
        P_a = np.polyval(integral_coeffs, a)

        return P_b - P_a

    # We need to integrate rate w.r.t. quality
    # Since we have quality = f(log_rate), we need to invert
    # This is complex, so we use numerical integration instead

    num_samples = 100
    quality_samples = np.linspace(min_quality, max_quality, num_samples)

    # Find log_rate for each quality by inverse interpolation
    anchor_log_rates_interp = np.interp(quality_samples, anchor_qualities, log_anchor_rates)
    test_log_rates_interp = np.interp(quality_samples, test_qualities, log_test_rates)

    # Integrate using trapezoidal rule
    dq = (max_quality - min_quality) / (num_samples - 1)
    anchor_area = _trapezoid(anchor_log_rates_interp, dx=dq)
    test_area = _trapezoid(test_log_rates_interp, dx=dq)

    # Compute average log rate difference
    avg_log_diff = (test_area - anchor_area) / (max_quality - min_quality)

    # Convert to percentage
    bd_rate = (10**avg_log_diff - 1) * 100

    return float(bd_rate)


def compute_bd_psnr(
    anchor: RDCurve,
    test: RDCurve,
) -> float:
    """Compute Bjøntegaard Delta PSNR.

    BD-PSNR measures the average PSNR difference between two
    codecs at the same bitrate.

    Positive BD-PSNR means test codec is better (higher PSNR at same rate).

    Args:
        anchor: Reference R-D curve.
        test: Test R-D curve to compare.

    Returns:
        BD-PSNR in dB.

    """
    anchor_rates = anchor.rates
    anchor_psnrs = anchor.psnrs
    test_rates = test.rates
    test_psnrs = test.psnrs

    # Work in log-rate domain
    log_anchor_rates = np.log10(anchor_rates + 1e-10)
    log_test_rates = np.log10(test_rates + 1e-10)

    # Find overlapping rate range
    min_log_rate = max(log_anchor_rates.min(), log_test_rates.min())
    max_log_rate = min(log_anchor_rates.max(), log_test_rates.max())

    if min_log_rate >= max_log_rate:
        return 0.0

    # Sample rate points
    num_samples = 100
    log_rate_samples = np.linspace(min_log_rate, max_log_rate, num_samples)

    # Interpolate PSNR at each rate
    anchor_psnrs_interp = np.interp(log_rate_samples, log_anchor_rates, anchor_psnrs)
    test_psnrs_interp = np.interp(log_rate_samples, log_test_rates, test_psnrs)

    # Compute average PSNR difference
    bd_psnr = float(np.mean(test_psnrs_interp - anchor_psnrs_interp))

    return bd_psnr
