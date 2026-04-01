"""Tests for evaluation pipeline.

Covers the EvaluationResult dataclass and its methods.
The Evaluator class requires complex dependencies (MCTS, models)
so we focus on the data structures and lightweight functionality.
"""

from __future__ import annotations

from src.training.evaluation import EvaluationResult

# --- EvaluationResult Tests ---


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation_required_fields(self) -> None:
        """Can be created with only required fields."""
        result = EvaluationResult(
            win_rate=0.75,
            n_games=100,
            wins=75,
            losses=20,
            draws=5,
            avg_game_length=150.0,
        )
        assert result.win_rate == 0.75
        assert result.n_games == 100
        assert result.wins == 75
        assert result.losses == 20
        assert result.draws == 5
        assert result.avg_game_length == 150.0

    def test_default_optional_fields(self) -> None:
        """Optional fields have correct defaults."""
        result = EvaluationResult(
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=100.0,
        )
        assert result.avg_value_error == 0.0
        assert result.policy_agreement == 0.0
        assert result.metadata == {}

    def test_custom_optional_fields(self) -> None:
        """Optional fields can be set."""
        result = EvaluationResult(
            win_rate=0.8,
            n_games=50,
            wins=40,
            losses=8,
            draws=2,
            avg_game_length=120.0,
            avg_value_error=0.15,
            policy_agreement=0.85,
            metadata={"board_size": 9, "opponent": "random"},
        )
        assert result.avg_value_error == 0.15
        assert result.policy_agreement == 0.85
        assert result.metadata["board_size"] == 9

    def test_to_dict_required_fields(self) -> None:
        """to_dict includes all required fields."""
        result = EvaluationResult(
            win_rate=0.6,
            n_games=20,
            wins=12,
            losses=6,
            draws=2,
            avg_game_length=90.0,
        )
        d = result.to_dict()
        assert d["win_rate"] == 0.6
        assert d["n_games"] == 20
        assert d["wins"] == 12
        assert d["losses"] == 6
        assert d["draws"] == 2
        assert d["avg_game_length"] == 90.0
        assert d["avg_value_error"] == 0.0
        assert d["policy_agreement"] == 0.0

    def test_to_dict_includes_metadata(self) -> None:
        """to_dict merges metadata into the dictionary."""
        result = EvaluationResult(
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=100.0,
            metadata={"board_size": 19, "device": "cuda"},
        )
        d = result.to_dict()
        assert d["board_size"] == 19
        assert d["device"] == "cuda"

    def test_to_dict_returns_new_dict(self) -> None:
        """to_dict returns a new dictionary each time."""
        result = EvaluationResult(
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=100.0,
        )
        d1 = result.to_dict()
        d2 = result.to_dict()
        assert d1 == d2
        assert d1 is not d2

    def test_wins_losses_draws_sum_to_n_games(self) -> None:
        """Wins + losses + draws should equal n_games."""
        result = EvaluationResult(
            win_rate=0.6,
            n_games=100,
            wins=60,
            losses=30,
            draws=10,
            avg_game_length=120.0,
        )
        assert result.wins + result.losses + result.draws == result.n_games

    def test_zero_games(self) -> None:
        """Can represent zero-game evaluation."""
        result = EvaluationResult(
            win_rate=0.0,
            n_games=0,
            wins=0,
            losses=0,
            draws=0,
            avg_game_length=0.0,
        )
        assert result.n_games == 0
        d = result.to_dict()
        assert d["n_games"] == 0

    def test_perfect_win_rate(self) -> None:
        """Can represent 100% win rate."""
        result = EvaluationResult(
            win_rate=1.0,
            n_games=50,
            wins=50,
            losses=0,
            draws=0,
            avg_game_length=80.0,
        )
        assert result.win_rate == 1.0
        assert result.losses == 0

    def test_metadata_not_shared_between_instances(self) -> None:
        """Default metadata dict is not shared between instances."""
        result1 = EvaluationResult(
            win_rate=0.5, n_games=10, wins=5, losses=5, draws=0, avg_game_length=100.0,
        )
        result2 = EvaluationResult(
            win_rate=0.5, n_games=10, wins=5, losses=5, draws=0, avg_game_length=100.0,
        )
        result1.metadata["key"] = "value"
        assert "key" not in result2.metadata
