"""Coverage tests for evaluation pipeline.

Tests cover:
- EvaluationResult: Dataclass and serialization
- Evaluator: Initialization and game play
- quick_evaluate: Quick evaluation helper
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.training.evaluation import EvaluationResult, Evaluator

SEED = 42


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_basic_creation(self) -> None:
        result = EvaluationResult(
            win_rate=0.75,
            n_games=20,
            wins=15,
            losses=3,
            draws=2,
            avg_game_length=100.0,
        )
        assert result.win_rate == 0.75
        assert result.n_games == 20
        assert result.wins == 15
        assert result.losses == 3
        assert result.draws == 2

    def test_default_values(self) -> None:
        result = EvaluationResult(
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=50.0,
        )
        assert result.avg_value_error == 0.0
        assert result.policy_agreement == 0.0
        assert result.metadata == {}

    def test_to_dict(self) -> None:
        result = EvaluationResult(
            win_rate=0.6,
            n_games=10,
            wins=6,
            losses=3,
            draws=1,
            avg_game_length=80.0,
            avg_value_error=0.15,
            policy_agreement=0.85,
            metadata={"opponent": "random", "board_size": 9},
        )
        d = result.to_dict()
        assert d["win_rate"] == 0.6
        assert d["n_games"] == 10
        assert d["wins"] == 6
        assert d["losses"] == 3
        assert d["draws"] == 1
        assert d["avg_game_length"] == 80.0
        assert d["avg_value_error"] == 0.15
        assert d["policy_agreement"] == 0.85
        assert d["opponent"] == "random"
        assert d["board_size"] == 9

    def test_to_dict_empty_metadata(self) -> None:
        result = EvaluationResult(
            win_rate=0.0,
            n_games=0,
            wins=0,
            losses=0,
            draws=0,
            avg_game_length=0.0,
        )
        d = result.to_dict()
        assert "win_rate" in d
        assert len(d) == 8  # 8 core fields


class TestEvaluatorInitialization:
    """Tests for Evaluator initialization."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.eval = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        return model

    def test_default_initialization(self, mock_model) -> None:
        evaluator = Evaluator(model=mock_model, device="cpu")
        assert evaluator.device == torch.device("cpu")
        assert evaluator.board_sizes == [9, 13, 19]
        assert evaluator.game is None

    def test_custom_board_sizes(self, mock_model) -> None:
        evaluator = Evaluator(
            model=mock_model,
            device="cpu",
            board_sizes=[5, 9],
        )
        assert evaluator.board_sizes == [5, 9]

    def test_with_mcts_config(self, mock_model) -> None:
        mcts_config = MagicMock()
        mcts_config.n_simulations = 10
        mcts_config.c_puct = 1.5
        mcts_config.dirichlet_alpha = 0.3
        mcts_config.dirichlet_epsilon = 0.25

        evaluator = Evaluator(
            model=mock_model,
            mcts_config=mcts_config,
            device="cpu",
        )
        assert evaluator._mcts_kwargs["n_simulations"] == 10
        assert evaluator._mcts_kwargs["c_puct"] == 1.5

    def test_with_game_interface(self, mock_model) -> None:
        game = MagicMock()
        evaluator = Evaluator(
            model=mock_model,
            device="cpu",
            game=game,
        )
        assert evaluator.game is game

    def test_evaluate_vs_engine_no_game_raises(self, mock_model) -> None:
        evaluator = Evaluator(model=mock_model, device="cpu")
        with pytest.raises(ValueError, match="evaluate_vs_engine requires"):
            evaluator.evaluate_vs_engine(
                engine_config=MagicMock(),
                match_config=MagicMock(),
            )
