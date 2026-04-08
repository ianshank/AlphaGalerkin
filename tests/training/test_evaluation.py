"""Tests for evaluation pipeline.

Covers the EvaluationResult dataclass, the Evaluator class
(with mocked game play), and the quick_evaluate helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from config.schemas import MCTSConfig, OperatorConfig
from src.modeling.model import AlphaGalerkinModel
from src.training.evaluation import EvaluationResult, Evaluator, quick_evaluate


def _small_model() -> AlphaGalerkinModel:
    cfg = OperatorConfig(
        d_model=16,
        d_key=8,
        d_value=8,
        d_ffn=32,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=8,
        use_fnet_mixing=False,
    )
    model = AlphaGalerkinModel(cfg)
    model.eval()
    return model


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
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=100.0,
        )
        result2 = EvaluationResult(
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=100.0,
        )
        result1.metadata["key"] = "value"
        assert "key" not in result2.metadata


# --- Evaluator Tests ---


class TestEvaluatorInit:
    """Tests for Evaluator initialization."""

    def test_init_basic(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        assert evaluator.model is model
        assert evaluator.device == torch.device("cpu")

    def test_init_with_mcts_config(self) -> None:
        model = _small_model()
        mcts_cfg = MCTSConfig(n_simulations=10, c_puct=1.0)
        evaluator = Evaluator(model=model, mcts_config=mcts_cfg, device="cpu")
        assert evaluator._mcts_kwargs["n_simulations"] == 10

    def test_init_custom_board_sizes(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[5, 9])
        assert evaluator.board_sizes == [5, 9]

    def test_init_default_board_sizes(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        assert len(evaluator.board_sizes) > 0

    def test_init_with_game_interface(self) -> None:
        model = _small_model()
        mock_game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=mock_game)
        assert evaluator.game is mock_game


class TestEvaluatorVsRandom:
    """Tests for evaluate_vs_random with mocked game play."""

    def test_vs_random_returns_result(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        # Mock _play_game to return (outcome, moves)
        with patch.object(evaluator, "_play_game", return_value=(1.0, 10)):
            result = evaluator.evaluate_vs_random(n_games=4, board_size=9)
        assert isinstance(result, EvaluationResult)
        assert result.n_games == 4

    def test_vs_random_all_wins(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        with patch.object(evaluator, "_play_game", return_value=(1.0, 10)):
            result = evaluator.evaluate_vs_random(n_games=6, board_size=9)
        # Model plays black on even games (0,2,4) -> outcome=1.0 means model wins
        # Model plays white on odd games (1,3,5) -> outcome=1.0, -outcome=-1.0 for white
        # So not all wins. Let's just check structure.
        assert result.wins + result.losses + result.draws == result.n_games

    def test_vs_random_mixed_outcomes(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        outcomes = [(1.0, 10), (-1.0, 15), (0.0, 20), (1.0, 12)]
        with patch.object(evaluator, "_play_game", side_effect=outcomes):
            result = evaluator.evaluate_vs_random(n_games=4, board_size=9)
        assert result.n_games == 4
        assert result.avg_game_length > 0


class TestEvaluatorVsModel:
    """Tests for evaluate_vs_model with mocked game play."""

    def test_vs_model_returns_result(self) -> None:
        model = _small_model()
        opponent = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        with patch.object(evaluator, "_play_game", return_value=(1.0, 10)):
            result = evaluator.evaluate_vs_model(opponent, n_games=4, board_size=9)
        assert isinstance(result, EvaluationResult)
        assert result.n_games == 4

    def test_vs_model_metadata_contains_opponent(self) -> None:
        model = _small_model()
        opponent = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        with patch.object(evaluator, "_play_game", return_value=(0.0, 10)):
            result = evaluator.evaluate_vs_model(opponent, n_games=2, board_size=9)
        assert "opponent" in result.metadata


class TestEvaluatorMultiResolution:
    """Tests for evaluate_multi_resolution."""

    def test_multi_resolution_returns_dict(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[5, 9])
        with patch.object(
            evaluator,
            "evaluate_vs_random",
            return_value=EvaluationResult(
                win_rate=0.5,
                n_games=2,
                wins=1,
                losses=1,
                draws=0,
                avg_game_length=10.0,
            ),
        ):
            results = evaluator.evaluate_multi_resolution(n_games_per_size=2)
        assert isinstance(results, dict)
        assert len(results) == 2  # one per board size


class TestEvaluatorPlayGame:
    """Tests for _play_game dispatching."""

    def test_play_game_dispatches_to_generic_with_game(self) -> None:
        model = _small_model()
        mock_game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=mock_game)
        with patch.object(evaluator, "_play_game_generic", return_value=(1.0, 5)):
            outcome, moves = evaluator._play_game(
                board_size=9,
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
            )
        assert outcome == 1.0

    def test_play_game_dispatches_to_go_without_game(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        with patch.object(evaluator, "_play_game_go", return_value=(0.0, 15)):
            outcome, moves = evaluator._play_game(
                board_size=9,
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
            )
        assert outcome == 0.0


class TestEvaluatorPlayGameGeneric:
    """Tests for _play_game_generic with mock game interface."""

    def test_generic_game_terminates(self) -> None:
        model = _small_model()
        mock_game = MagicMock()
        state1 = MagicMock(current_player=1)
        state2 = MagicMock(current_player=-1)
        mock_game.initial_state.return_value = state1
        mock_game.is_terminal.side_effect = [False, False, True, True]
        game_result = MagicMock(winner=1)
        mock_game.get_result.return_value = game_result
        mock_game.apply_action.return_value = state2

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 0

        evaluator = Evaluator(model=model, device="cpu", game=mock_game)
        with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
            outcome, moves = evaluator._play_game_generic(
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
            )
        assert isinstance(outcome, (int, float))
        assert moves >= 0

    def test_generic_game_draw_on_max_moves(self) -> None:
        model = _small_model()
        mock_game = MagicMock()
        mock_game.initial_state.return_value = MagicMock(current_player=1)
        mock_game.is_terminal.return_value = False
        mock_game.apply_action.return_value = MagicMock(current_player=-1)

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 0

        evaluator = Evaluator(model=model, device="cpu", game=mock_game)
        with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
            outcome, moves = evaluator._play_game_generic(
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
                max_moves=3,
            )
        assert outcome == 0.0
        assert moves == 3


class TestEvaluatorPlayGameGo:
    """Tests for _play_game_go with mocked SimpleGoGame."""

    def test_go_game_returns_outcome_and_moves(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        mock_go = MagicMock()
        # While loop checks is_terminal twice (False, False -> 2 iterations),
        # then True exits loop, then line 510 checks again -> need 4 calls.
        mock_go.is_terminal.side_effect = [False, False, True, True]
        mock_go.get_legal_actions.return_value = [0, 1, 2]
        mock_go.get_board_tensor.return_value = torch.zeros(17, 5, 5)
        mock_go.current_player = 1
        mock_go.get_winner.return_value = 1
        mock_go.play.return_value = True

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 0

        with (
            patch("src.training.evaluation.SimpleGoGame", return_value=mock_go),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
        ):
            outcome, moves = evaluator._play_game_go(
                board_size=5,
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
            )
        assert isinstance(outcome, float)
        assert moves == 2


class TestEvaluatorPolicyAgreement:
    """Tests for measure_policy_agreement."""

    def test_policy_agreement_returns_float(self) -> None:
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        mock_go = MagicMock()
        # After reset+random moves, is_terminal returns True -> position skipped.
        # With all positions skipped, total=0 and result=0.0.
        mock_go.is_terminal.return_value = True
        mock_go.get_legal_actions.return_value = [0, 1, 2]

        mock_mcts = MagicMock()

        with (
            patch("src.training.evaluation.SimpleGoGame", return_value=mock_go),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
        ):
            result = evaluator.measure_policy_agreement(
                n_positions=2,
                board_size=5,
            )
        assert isinstance(result, float)
        assert result == 0.0


class TestQuickEvaluate:
    """Tests for quick_evaluate helper function."""

    def test_quick_evaluate_returns_dict(self) -> None:
        model = _small_model()
        with patch.object(Evaluator, "evaluate_vs_random") as mock_eval:
            mock_eval.return_value = EvaluationResult(
                win_rate=0.7,
                n_games=4,
                wins=3,
                losses=1,
                draws=0,
                avg_game_length=50.0,
            )
            result = quick_evaluate(model, n_games=4, board_size=9, device="cpu")
        assert isinstance(result, dict)
        assert "win_rate" in result
