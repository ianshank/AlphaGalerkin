"""Tests for Elo rating estimation.

Tests known Elo calculations, edge cases, and confidence intervals.
"""

from __future__ import annotations

import pytest

from src.engines.config import EloConfig
from src.engines.elo import EloCalculator, EloEstimate


class TestExpectedScore:
    """Tests for Elo expected score calculation."""

    @pytest.fixture
    def calculator(self) -> EloCalculator:
        return EloCalculator(EloConfig(name="test"))

    def test_equal_rating(self, calculator: EloCalculator) -> None:
        assert calculator.expected_score(0.0) == pytest.approx(0.5)

    def test_200_elo_advantage(self, calculator: EloCalculator) -> None:
        # 200 Elo advantage ≈ 75% expected score
        score = calculator.expected_score(200.0)
        assert score == pytest.approx(0.7597, abs=0.01)

    def test_negative_elo_diff(self, calculator: EloCalculator) -> None:
        # Symmetric: E(+d) + E(-d) = 1
        pos = calculator.expected_score(200.0)
        neg = calculator.expected_score(-200.0)
        assert pos + neg == pytest.approx(1.0)

    def test_large_advantage(self, calculator: EloCalculator) -> None:
        score = calculator.expected_score(800.0)
        assert score > 0.99

    def test_large_disadvantage(self, calculator: EloCalculator) -> None:
        score = calculator.expected_score(-800.0)
        assert score < 0.01


class TestEloDiffFromScore:
    """Tests for inverse Elo calculation."""

    @pytest.fixture
    def calculator(self) -> EloCalculator:
        return EloCalculator(EloConfig(name="test"))

    def test_50_percent_is_zero(self, calculator: EloCalculator) -> None:
        assert calculator.elo_diff_from_score(0.5) == pytest.approx(0.0, abs=0.01)

    def test_75_percent_approx_200(self, calculator: EloCalculator) -> None:
        diff = calculator.elo_diff_from_score(0.76)
        assert 190 < diff < 210

    def test_roundtrip(self, calculator: EloCalculator) -> None:
        for elo_diff in [-300, -100, 0, 100, 300]:
            score = calculator.expected_score(float(elo_diff))
            recovered = calculator.elo_diff_from_score(score)
            assert recovered == pytest.approx(elo_diff, abs=0.1)

    def test_invalid_score_zero(self, calculator: EloCalculator) -> None:
        with pytest.raises(ValueError, match="must be in"):
            calculator.elo_diff_from_score(0.0)

    def test_invalid_score_one(self, calculator: EloCalculator) -> None:
        with pytest.raises(ValueError, match="must be in"):
            calculator.elo_diff_from_score(1.0)


class TestUpdateRating:
    """Tests for single-game rating update."""

    @pytest.fixture
    def calculator(self) -> EloCalculator:
        return EloCalculator(EloConfig(name="test"))

    def test_win_increases_rating(self, calculator: EloCalculator) -> None:
        new = calculator.update_rating(1500.0, 1500.0, 1.0)
        assert new > 1500.0

    def test_loss_decreases_rating(self, calculator: EloCalculator) -> None:
        new = calculator.update_rating(1500.0, 1500.0, 0.0)
        assert new < 1500.0

    def test_draw_no_change_equal_rating(self, calculator: EloCalculator) -> None:
        new = calculator.update_rating(1500.0, 1500.0, 0.5)
        assert new == pytest.approx(1500.0)

    def test_upset_win_big_gain(self, calculator: EloCalculator) -> None:
        # Beating a much stronger opponent gives big gain
        gain = calculator.update_rating(1200.0, 1800.0, 1.0) - 1200.0
        normal_gain = calculator.update_rating(1500.0, 1500.0, 1.0) - 1500.0
        assert gain > normal_gain


class TestEstimateEloDifference:
    """Tests for Elo difference estimation from W/L/D."""

    @pytest.fixture
    def calculator(self) -> EloCalculator:
        return EloCalculator(EloConfig(name="test"))

    def test_equal_results(self, calculator: EloCalculator) -> None:
        result = calculator.estimate_elo_difference(5, 5, 0)
        assert result.elo_difference == pytest.approx(0.0, abs=10)
        assert result.win_rate == pytest.approx(0.5)

    def test_all_wins(self, calculator: EloCalculator) -> None:
        result = calculator.estimate_elo_difference(10, 0, 0)
        assert result.elo_difference > 0
        assert result.likelihood_of_superiority > 0.95

    def test_all_losses(self, calculator: EloCalculator) -> None:
        result = calculator.estimate_elo_difference(0, 10, 0)
        assert result.elo_difference < 0
        assert result.likelihood_of_superiority < 0.05

    def test_all_draws(self, calculator: EloCalculator) -> None:
        result = calculator.estimate_elo_difference(0, 0, 10)
        assert result.elo_difference == pytest.approx(0.0, abs=10)
        assert result.win_rate == pytest.approx(0.5)

    def test_confidence_interval_contains_point(self, calculator: EloCalculator) -> None:
        result = calculator.estimate_elo_difference(7, 3, 0)
        low, high = result.confidence_interval
        assert low <= result.elo_difference <= high

    def test_more_games_tighter_ci(self, calculator: EloCalculator) -> None:
        few = calculator.estimate_elo_difference(7, 3, 0)
        many = calculator.estimate_elo_difference(70, 30, 0)
        few_width = few.confidence_interval[1] - few.confidence_interval[0]
        many_width = many.confidence_interval[1] - many.confidence_interval[0]
        assert many_width < few_width

    def test_zero_games_raises(self, calculator: EloCalculator) -> None:
        with pytest.raises(ValueError, match="zero games"):
            calculator.estimate_elo_difference(0, 0, 0)

    def test_los_strong_advantage(self, calculator: EloCalculator) -> None:
        result = calculator.estimate_elo_difference(9, 1, 0)
        assert result.likelihood_of_superiority > 0.9

    def test_win_rate_correct(self, calculator: EloCalculator) -> None:
        result = calculator.estimate_elo_difference(6, 2, 4)
        # win_rate = (6 + 0.5*4) / 12 = 8/12 = 0.667
        assert result.win_rate == pytest.approx(8.0 / 12.0)

    def test_elo_estimate_is_frozen(self, calculator: EloCalculator) -> None:
        result = calculator.estimate_elo_difference(5, 5, 0)
        assert isinstance(result, EloEstimate)
        # frozen=True means we can't modify it
        with pytest.raises(AttributeError):
            result.elo_difference = 999.0  # type: ignore[misc]
