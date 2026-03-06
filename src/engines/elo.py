"""Elo rating estimation from match results.

Implements standard Elo formulas for estimating rating differences
from win/loss/draw counts, with confidence intervals via normal
approximation.

Reference: https://en.wikipedia.org/wiki/Elo_rating_system
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

from src.engines.config import EloConfig

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class EloEstimate:
    """Result of Elo difference estimation.

    Attributes:
        elo_difference: Estimated Elo difference (positive = player stronger).
        confidence_interval: (lower, upper) bounds at configured confidence.
        likelihood_of_superiority: Probability that player is stronger.
        win_rate: Observed win rate (W + 0.5*D) / N.

    """

    elo_difference: float
    confidence_interval: tuple[float, float]
    likelihood_of_superiority: float
    win_rate: float


class EloCalculator:
    """Elo rating estimation from match results.

    Uses the standard Elo expected score formula:
        E = 1 / (1 + base^((R_opp - R_player) / divisor))

    With configurable base (default 10) and divisor (default 400).

    Args:
        config: Elo calculation configuration.

    """

    def __init__(self, config: EloConfig) -> None:
        self._config = config

    @property
    def config(self) -> EloConfig:
        """Get Elo configuration."""
        return self._config

    def expected_score(self, elo_diff: float) -> float:
        """Calculate expected score given Elo difference.

        Args:
            elo_diff: Rating difference (player - opponent).

        Returns:
            Expected score in [0, 1].

        """
        return 1.0 / (1.0 + self._config.elo_base ** (-elo_diff / self._config.elo_divisor))

    def elo_diff_from_score(self, score: float) -> float:
        """Inverse: estimate Elo difference from observed score.

        Args:
            score: Observed score in (0, 1).

        Returns:
            Estimated Elo difference.

        Raises:
            ValueError: If score is not in (0, 1).

        """
        if score <= 0.0 or score >= 1.0:
            raise ValueError(f"Score must be in (0, 1), got {score}")

        return -self._config.elo_divisor * math.log10(1.0 / score - 1.0)

    def update_rating(
        self,
        player_elo: float,
        opponent_elo: float,
        actual_score: float,
    ) -> float:
        """Update a player's rating after a single result.

        Args:
            player_elo: Player's current rating.
            opponent_elo: Opponent's current rating.
            actual_score: Actual score (1.0=win, 0.5=draw, 0.0=loss).

        Returns:
            Updated player rating.

        """
        expected = self.expected_score(player_elo - opponent_elo)
        return player_elo + self._config.k_factor * (actual_score - expected)

    def estimate_elo_difference(
        self,
        wins: int,
        losses: int,
        draws: int,
    ) -> EloEstimate:
        """Estimate Elo difference from match results.

        Uses the observed score to estimate Elo difference,
        with confidence intervals via normal approximation
        of the score variance.

        Args:
            wins: Number of wins.
            losses: Number of losses.
            draws: Number of draws.

        Returns:
            EloEstimate with difference, confidence interval, and LOS.

        Raises:
            ValueError: If total games is zero.

        """
        total = wins + losses + draws
        if total == 0:
            raise ValueError("Cannot estimate Elo from zero games")

        # Observed score
        score = (wins + 0.5 * draws) / total

        # Clamp score to avoid infinities in elo_diff_from_score
        # Use Laplace-style regularization for extreme cases
        clamped_score = max(0.5 / total, min(1.0 - 0.5 / total, score))

        # Point estimate
        elo_diff = self.elo_diff_from_score(clamped_score)

        # Variance of score (binomial approximation)
        # Var(S) = (W*(1-s)^2 + D*(0.5-s)^2 + L*(0-s)^2) / (N*(N-1))
        # Simplified: Var(S) ≈ s*(1-s)/N for large N
        score_var = clamped_score * (1.0 - clamped_score) / total

        # Standard error of the score
        score_se = math.sqrt(score_var) if score_var > 0 else 0.0

        # Z-score for confidence level
        z = _z_score(self._config.confidence_level)

        # Confidence interval on score
        score_low = max(0.01, clamped_score - z * score_se)
        score_high = min(0.99, clamped_score + z * score_se)

        # Map score CI to Elo CI
        elo_low = self.elo_diff_from_score(score_low)
        elo_high = self.elo_diff_from_score(score_high)

        # Likelihood of superiority (LOS)
        # P(player > opponent) based on score distribution
        if score_se > 0:
            los = _normal_cdf((clamped_score - 0.5) / score_se)
        else:
            los = 1.0 if score > 0.5 else (0.5 if score == 0.5 else 0.0)

        result = EloEstimate(
            elo_difference=elo_diff,
            confidence_interval=(elo_low, elo_high),
            likelihood_of_superiority=los,
            win_rate=score,
        )

        logger.info(
            "elo_estimated",
            wins=wins,
            losses=losses,
            draws=draws,
            elo_diff=f"{elo_diff:.1f}",
            ci_low=f"{elo_low:.1f}",
            ci_high=f"{elo_high:.1f}",
            los=f"{los:.3f}",
        )

        return result


def _z_score(confidence: float) -> float:
    """Approximate z-score for common confidence levels.

    Uses a lookup for standard values and falls back to
    the Abramowitz and Stegun approximation.

    Args:
        confidence: Confidence level in (0, 1).

    Returns:
        Z-score for the given confidence level.

    """
    # Common confidence levels
    _z_table: dict[float, float] = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }

    # Round to avoid float comparison issues
    rounded = round(confidence, 2)
    if rounded in _z_table:
        return _z_table[rounded]

    # Fallback: inverse normal via rational approximation
    p = (1.0 + confidence) / 2.0
    return _probit(p)


def _probit(p: float) -> float:
    """Approximate inverse normal CDF (probit function).

    Uses the Abramowitz and Stegun rational approximation (error < 4.5e-4).

    Args:
        p: Probability in (0, 1).

    Returns:
        Approximate z-score.

    """
    if p <= 0 or p >= 1:
        return 0.0

    if p < 0.5:
        return -_probit(1.0 - p)

    t = math.sqrt(-2.0 * math.log(1.0 - p))

    # Rational approximation coefficients
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t**3)


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function.

    Args:
        x: Z-score.

    Returns:
        P(Z <= x) for standard normal Z.

    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
