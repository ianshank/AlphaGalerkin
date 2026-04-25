"""Additional tests for evaluation.py to improve coverage.

Targets missed lines: 289-301, 316-339, 498, 503, 513, 516,
540-593, 627->639, 630-637, 643-654.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel
from src.tools.gtp import SimpleGoGame
from src.training.evaluation import EvaluationResult, Evaluator


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


def _mock_simple_go_game(board_size: int = 5) -> MagicMock:
    """Create a mock SimpleGoGame that preserves the real BLACK/WHITE constants."""
    mock_cls = MagicMock()
    mock_cls.BLACK = SimpleGoGame.BLACK  # 1
    mock_cls.WHITE = SimpleGoGame.WHITE  # 2
    return mock_cls


# ---------------------------------------------------------------------------
# evaluate_vs_checkpoint  (lines 289-301)
# ---------------------------------------------------------------------------


class TestEvaluateVsCheckpoint:
    """Tests for evaluate_vs_checkpoint method."""

    def test_vs_checkpoint_delegates_to_vs_model(self) -> None:
        """evaluate_vs_checkpoint loads model then delegates to evaluate_vs_model."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        fake_opponent = _small_model()

        with (
            patch.object(
                evaluator,
                "_load_model_from_checkpoint",
                return_value=fake_opponent,
            ) as mock_load,
            patch.object(
                evaluator,
                "evaluate_vs_model",
                return_value=EvaluationResult(
                    win_rate=0.6,
                    n_games=10,
                    wins=6,
                    losses=4,
                    draws=0,
                    avg_game_length=80.0,
                    metadata={"opponent": "model"},
                ),
            ) as mock_eval,
        ):
            result = evaluator.evaluate_vs_checkpoint(
                checkpoint_path="/tmp/fake_ckpt.pt",
                n_games=10,
                board_size=9,
            )

        mock_load.assert_called_once_with("/tmp/fake_ckpt.pt")
        mock_eval.assert_called_once_with(
            opponent_model=fake_opponent,
            n_games=10,
            board_size=9,
        )
        # Metadata should be overwritten with checkpoint info
        assert result.metadata["opponent"] == "checkpoint"
        assert result.metadata["checkpoint_path"] == "/tmp/fake_ckpt.pt"

    def test_vs_checkpoint_with_none_board_size(self) -> None:
        """evaluate_vs_checkpoint passes None board_size through."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        with (
            patch.object(
                evaluator,
                "_load_model_from_checkpoint",
                return_value=_small_model(),
            ),
            patch.object(
                evaluator,
                "evaluate_vs_model",
                return_value=EvaluationResult(
                    win_rate=0.5,
                    n_games=4,
                    wins=2,
                    losses=2,
                    draws=0,
                    avg_game_length=50.0,
                    metadata={"opponent": "model"},
                ),
            ) as mock_eval,
        ):
            evaluator.evaluate_vs_checkpoint(
                checkpoint_path="/tmp/ckpt.pt",
                n_games=4,
            )

        _, kwargs = mock_eval.call_args
        assert kwargs["board_size"] is None


# ---------------------------------------------------------------------------
# _load_model_from_checkpoint  (lines 316-339)
# ---------------------------------------------------------------------------


class TestLoadModelFromCheckpoint:
    """Tests for _load_model_from_checkpoint."""

    def test_load_with_dict_config(self) -> None:
        """When checkpoint contains a dict 'config', uses AlphaGalerkinConfig."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        fake_state_dict = model.state_dict()

        mock_ag_config = MagicMock()
        mock_ag_config.operator = model.config

        checkpoint_data = {
            "config": {"some_key": "some_value"},  # a dict triggers the branch
            "model_state_dict": fake_state_dict,
        }

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock(return_value=mock_model_instance)

        with (
            patch("torch.load", return_value=checkpoint_data),
            patch(
                "src.modeling.model.AlphaGalerkinModel",
                mock_model_cls,
            ),
            patch(
                "config.schemas.AlphaGalerkinConfig",
                return_value=mock_ag_config,
            ),
        ):
            loaded = evaluator._load_model_from_checkpoint("/tmp/ckpt.pt")

        mock_model_cls.assert_called_once_with(mock_ag_config.operator)
        mock_model_instance.load_state_dict.assert_called_once_with(fake_state_dict)
        mock_model_instance.to.assert_called_once_with(evaluator.device)
        mock_model_instance.eval.assert_called_once()
        assert loaded is mock_model_instance

    def test_load_with_non_dict_config_fallback(self) -> None:
        """When checkpoint config is not a dict, falls back to self.model.config."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        fake_state_dict = model.state_dict()
        checkpoint_data = {
            "config": None,  # not a dict -> fallback branch
            "model_state_dict": fake_state_dict,
        }

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock(return_value=mock_model_instance)

        with (
            patch("torch.load", return_value=checkpoint_data),
            patch(
                "src.modeling.model.AlphaGalerkinModel",
                mock_model_cls,
            ),
        ):
            loaded = evaluator._load_model_from_checkpoint("/tmp/ckpt.pt")

        mock_model_cls.assert_called_once_with(model.config)
        mock_model_instance.load_state_dict.assert_called_once_with(fake_state_dict)
        assert loaded is mock_model_instance

    def test_load_with_missing_config_key(self) -> None:
        """When checkpoint has no 'config' key, cfg is None -> fallback."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
        }

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock(return_value=mock_model_instance)

        with (
            patch("torch.load", return_value=checkpoint_data),
            patch(
                "src.modeling.model.AlphaGalerkinModel",
                mock_model_cls,
            ),
        ):
            evaluator._load_model_from_checkpoint("/tmp/ckpt.pt")

        mock_model_cls.assert_called_once_with(model.config)


# ---------------------------------------------------------------------------
# _play_game_go additional branches  (lines 497-498, 502-503, 512-513, 516)
# ---------------------------------------------------------------------------


class TestPlayGameGoBranches:
    """Cover uncovered branches in _play_game_go."""

    def _make_evaluator(self) -> Evaluator:
        model = _small_model()
        return Evaluator(model=model, device="cpu")

    def test_pass_action_branch(self) -> None:
        """When MCTS returns pass action (board_size**2), game.play_pass is called."""
        evaluator = self._make_evaluator()
        board_size = 5

        mock_go = MagicMock()
        mock_go.is_terminal.side_effect = [False, True, True]
        mock_go.current_player = SimpleGoGame.BLACK
        mock_go.get_winner.return_value = 1

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = board_size**2

        mock_cls = _mock_simple_go_game()
        mock_cls.return_value = mock_go

        with (
            patch("src.training.evaluation.SimpleGoGame", mock_cls),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
        ):
            outcome, moves = evaluator._play_game_go(
                board_size=board_size,
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
            )

        mock_go.play_pass.assert_called_once()
        assert moves == 1

    def test_illegal_move_falls_back_to_pass(self) -> None:
        """When game.play() returns False, should fall back to play_pass."""
        evaluator = self._make_evaluator()
        board_size = 5

        mock_go = MagicMock()
        mock_go.is_terminal.side_effect = [False, True, True]
        mock_go.current_player = SimpleGoGame.BLACK
        mock_go.get_winner.return_value = 1
        mock_go.play.return_value = False

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 0

        mock_cls = _mock_simple_go_game()
        mock_cls.return_value = mock_go

        with (
            patch("src.training.evaluation.SimpleGoGame", mock_cls),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
        ):
            outcome, moves = evaluator._play_game_go(
                board_size=board_size,
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
            )

        mock_go.play.assert_called_once_with(0, 0)
        mock_go.play_pass.assert_called_once()

    def test_winner_when_current_player_is_black(self) -> None:
        """When game ends with current_player==BLACK, return -float(winner)."""
        evaluator = self._make_evaluator()
        board_size = 5

        mock_go = MagicMock()
        mock_go.is_terminal.side_effect = [False, True, True]
        mock_go.current_player = SimpleGoGame.BLACK  # 1
        mock_go.get_winner.return_value = 1
        mock_go.play.return_value = True

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 0

        mock_cls = _mock_simple_go_game()
        mock_cls.return_value = mock_go

        with (
            patch("src.training.evaluation.SimpleGoGame", mock_cls),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
        ):
            outcome, moves = evaluator._play_game_go(
                board_size=board_size,
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
            )

        # current_player==BLACK -> return -float(winner) = -1.0
        assert outcome == -1.0

    def test_winner_when_current_player_is_white(self) -> None:
        """When game ends with current_player==WHITE, return float(winner)."""
        evaluator = self._make_evaluator()
        board_size = 5

        mock_go = MagicMock()
        mock_go.is_terminal.side_effect = [False, True, True]
        mock_go.current_player = SimpleGoGame.WHITE  # 2
        mock_go.get_winner.return_value = 1
        mock_go.play.return_value = True

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 0

        mock_cls = _mock_simple_go_game()
        mock_cls.return_value = mock_go

        with (
            patch("src.training.evaluation.SimpleGoGame", mock_cls),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
        ):
            outcome, moves = evaluator._play_game_go(
                board_size=board_size,
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
            )

        # current_player==WHITE -> return float(winner) = 1.0
        assert outcome == 1.0

    def test_draw_on_max_moves(self) -> None:
        """When max_moves reached without terminal, returns 0.0."""
        evaluator = self._make_evaluator()
        board_size = 5

        mock_go = MagicMock()
        mock_go.is_terminal.return_value = False
        mock_go.current_player = SimpleGoGame.BLACK
        mock_go.play.return_value = True

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 0

        mock_cls = _mock_simple_go_game()
        mock_cls.return_value = mock_go

        with (
            patch("src.training.evaluation.SimpleGoGame", mock_cls),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
        ):
            outcome, moves = evaluator._play_game_go(
                board_size=board_size,
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
                max_moves=3,
            )

        assert outcome == 0.0
        assert moves == 3

    def test_white_player_uses_white_mcts(self) -> None:
        """When current_player is WHITE, white_mcts is used."""
        evaluator = self._make_evaluator()
        board_size = 5

        mock_go = MagicMock()
        mock_go.is_terminal.side_effect = [False, False, True, True]
        mock_go.current_player = SimpleGoGame.WHITE  # always WHITE
        mock_go.get_winner.return_value = -1
        mock_go.play.return_value = True

        black_eval = MagicMock()
        white_eval = MagicMock()

        mock_black_mcts = MagicMock()
        mock_black_mcts.get_action.return_value = 0
        mock_white_mcts = MagicMock()
        mock_white_mcts.get_action.return_value = 1

        mock_cls = _mock_simple_go_game()
        mock_cls.return_value = mock_go

        with (
            patch("src.training.evaluation.SimpleGoGame", mock_cls),
            patch(
                "src.training.evaluation.MCTS",
                side_effect=[mock_black_mcts, mock_white_mcts],
            ),
        ):
            outcome, moves = evaluator._play_game_go(
                board_size=board_size,
                black_evaluator=black_eval,
                white_evaluator=white_eval,
            )

        # Since current_player is always WHITE, white_mcts should be used
        assert mock_white_mcts.get_action.call_count == 2


# ---------------------------------------------------------------------------
# evaluate_vs_engine  (lines 540-593)
# ---------------------------------------------------------------------------


class TestEvaluateVsEngine:
    """Tests for evaluate_vs_engine method."""

    def test_raises_when_no_game_interface(self) -> None:
        """Should raise ValueError when self.game is None."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        assert evaluator.game is None

        mock_engine_config = MagicMock()
        mock_match_config = MagicMock()

        with pytest.raises(ValueError, match="evaluate_vs_engine requires a GameInterface"):
            evaluator.evaluate_vs_engine(mock_engine_config, mock_match_config)

    def test_vs_engine_with_elo_estimate(self) -> None:
        """Full path with Elo estimation present in match_result."""
        model = _small_model()
        mock_game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=mock_game)

        game_record_1 = MagicMock(move_count=40)
        game_record_2 = MagicMock(move_count=60)
        mock_match_result = MagicMock()
        mock_match_result.win_rate = 0.75
        mock_match_result.total_games = 2
        mock_match_result.wins = 1
        mock_match_result.losses = 0
        mock_match_result.draws = 1
        mock_match_result.games = [game_record_1, game_record_2]

        mock_elo = MagicMock()
        mock_elo.elo_difference = 150.0
        mock_elo.confidence_interval = (80.0, 220.0)
        mock_elo.likelihood_of_superiority = 0.95
        mock_match_result.elo_estimate = mock_elo

        mock_engine_match_instance = MagicMock()
        mock_engine_match_instance.play_match.return_value = mock_match_result

        mock_engine_config = MagicMock()
        mock_match_config = MagicMock()

        with patch(
            "src.engines.match.EngineMatch",
            return_value=mock_engine_match_instance,
        ):
            result = evaluator.evaluate_vs_engine(
                engine_config=mock_engine_config,
                match_config=mock_match_config,
            )

        assert isinstance(result, EvaluationResult)
        assert result.win_rate == 0.75
        assert result.n_games == 2
        assert result.wins == 1
        assert result.losses == 0
        assert result.draws == 1
        assert result.avg_game_length == 50.0  # (40+60)/2
        assert result.metadata["opponent"] == "engine"
        assert result.metadata["elo_difference"] == 150.0
        assert result.metadata["elo_ci"] == (80.0, 220.0)
        assert result.metadata["los"] == 0.95

    def test_vs_engine_without_elo_estimate(self) -> None:
        """Path where elo_estimate is None."""
        model = _small_model()
        mock_game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=mock_game)

        game_record = MagicMock(move_count=30)
        mock_match_result = MagicMock()
        mock_match_result.win_rate = 0.0
        mock_match_result.total_games = 1
        mock_match_result.wins = 0
        mock_match_result.losses = 1
        mock_match_result.draws = 0
        mock_match_result.games = [game_record]
        mock_match_result.elo_estimate = None

        mock_engine_match_instance = MagicMock()
        mock_engine_match_instance.play_match.return_value = mock_match_result

        with patch(
            "src.engines.match.EngineMatch",
            return_value=mock_engine_match_instance,
        ):
            result = evaluator.evaluate_vs_engine(
                engine_config=MagicMock(),
                match_config=MagicMock(),
            )

        assert result.metadata["opponent"] == "engine"
        assert "elo_difference" not in result.metadata
        assert "elo_ci" not in result.metadata
        assert "los" not in result.metadata
        assert result.avg_game_length == 30.0

    def test_vs_engine_zero_games(self) -> None:
        """Path where total_games is 0 -> avg_length is 0.0."""
        model = _small_model()
        mock_game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=mock_game)

        mock_match_result = MagicMock()
        mock_match_result.win_rate = 0.0
        mock_match_result.total_games = 0
        mock_match_result.wins = 0
        mock_match_result.losses = 0
        mock_match_result.draws = 0
        mock_match_result.games = []
        mock_match_result.elo_estimate = None

        mock_engine_match_instance = MagicMock()
        mock_engine_match_instance.play_match.return_value = mock_match_result

        with patch(
            "src.engines.match.EngineMatch",
            return_value=mock_engine_match_instance,
        ):
            result = evaluator.evaluate_vs_engine(
                engine_config=MagicMock(),
                match_config=MagicMock(),
            )

        assert result.avg_game_length == 0.0
        assert result.n_games == 0

    def test_vs_engine_with_custom_mcts_config(self) -> None:
        """mcts_config_dict override is passed to EngineMatch."""
        model = _small_model()
        mock_game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=mock_game)

        mock_match_result = MagicMock()
        mock_match_result.win_rate = 0.5
        mock_match_result.total_games = 2
        mock_match_result.wins = 1
        mock_match_result.losses = 1
        mock_match_result.draws = 0
        mock_match_result.games = [MagicMock(move_count=20), MagicMock(move_count=30)]
        mock_match_result.elo_estimate = None

        mock_engine_match_instance = MagicMock()
        mock_engine_match_instance.play_match.return_value = mock_match_result

        custom_mcts = {"n_simulations": 200, "c_puct": 2.0}

        with patch(
            "src.engines.match.EngineMatch",
            return_value=mock_engine_match_instance,
        ) as mock_cls:
            evaluator.evaluate_vs_engine(
                engine_config=MagicMock(),
                match_config=MagicMock(),
                mcts_config_dict=custom_mcts,
            )

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["mcts_config"] == custom_mcts


# ---------------------------------------------------------------------------
# measure_policy_agreement with non-terminal positions (lines 627-654)
# ---------------------------------------------------------------------------


class TestMeasurePolicyAgreementCoverage:
    """Cover the non-terminal branch of measure_policy_agreement."""

    def test_agreement_with_matching_actions(self) -> None:
        """When raw policy argmax matches MCTS best action, agreement increments."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        mock_go = MagicMock()
        # After reset: is_terminal returns False for position evaluation,
        # then True to end.
        # n_positions=1, so we need:
        #   - After random moves loop: is_terminal -> False (so position is evaluated)
        mock_go.is_terminal.return_value = False
        mock_go.get_legal_actions.return_value = [0, 1, 2]
        mock_go.get_state.return_value = np.zeros((17, 5, 5), dtype=np.float32)

        # Mock the neural evaluator
        mock_eval_result = MagicMock()
        mock_eval_result.policy = np.array([0.7, 0.2, 0.1])  # argmax = 0

        mock_neural_eval = MagicMock()
        mock_neural_eval.evaluate.return_value = mock_eval_result

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = {0: 100, 1: 20, 2: 5}  # max at 0

        with (
            patch("src.training.evaluation.SimpleGoGame", return_value=mock_go),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
            patch("src.training.evaluation.random") as mock_random,
        ):
            # Force n_random_moves=0 so we skip the inner moves loop
            mock_random.randint.return_value = 0
            mock_random.choice.return_value = 0

            evaluator.neural_evaluator = mock_neural_eval
            result = evaluator.measure_policy_agreement(
                n_positions=1,
                board_size=5,
            )

        assert result == 1.0

    def test_agreement_with_disagreeing_actions(self) -> None:
        """When raw policy argmax differs from MCTS best, no agreement."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        mock_go = MagicMock()
        mock_go.is_terminal.return_value = False
        mock_go.get_legal_actions.return_value = [0, 1, 2]
        mock_go.get_state.return_value = np.zeros((17, 5, 5), dtype=np.float32)

        mock_eval_result = MagicMock()
        mock_eval_result.policy = np.array([0.1, 0.7, 0.2])  # argmax = 1

        mock_neural_eval = MagicMock()
        mock_neural_eval.evaluate.return_value = mock_eval_result

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = {0: 100, 1: 20, 2: 5}  # max at 0

        with (
            patch("src.training.evaluation.SimpleGoGame", return_value=mock_go),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
            patch("src.training.evaluation.random") as mock_random,
        ):
            mock_random.randint.return_value = 0
            evaluator.neural_evaluator = mock_neural_eval
            result = evaluator.measure_policy_agreement(
                n_positions=1,
                board_size=5,
            )

        assert result == 0.0

    def test_agreement_with_random_moves_played(self) -> None:
        """Cover the inner loop that plays random moves before measuring."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        board_size = 5

        mock_go = MagicMock()
        # is_terminal: False throughout random moves, False for evaluation check
        mock_go.is_terminal.return_value = False
        mock_go.get_legal_actions.return_value = [0, 1, 2, board_size**2]
        mock_go.get_state.return_value = np.zeros((17, 5, 5), dtype=np.float32)
        mock_go.play.return_value = True

        mock_eval_result = MagicMock()
        mock_eval_result.policy = np.array([0.8, 0.1, 0.05, 0.05])

        mock_neural_eval = MagicMock()
        mock_neural_eval.evaluate.return_value = mock_eval_result

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = {0: 50, 1: 10, 2: 5, 25: 1}

        with (
            patch("src.training.evaluation.SimpleGoGame", return_value=mock_go),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
            patch("src.training.evaluation.random") as mock_random,
        ):
            # Play 2 random moves before measuring
            mock_random.randint.return_value = 2
            # Return action 1 (not a pass), then action 0 (not a pass)
            mock_random.choice.side_effect = [1, 0]

            evaluator.neural_evaluator = mock_neural_eval
            result = evaluator.measure_policy_agreement(
                n_positions=1,
                board_size=board_size,
            )

        # play() should have been called for the random moves (not pass actions)
        assert mock_go.play.call_count == 2
        assert isinstance(result, float)

    def test_agreement_with_pass_in_random_moves(self) -> None:
        """Cover pass action branch inside random moves loop."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")
        board_size = 5

        mock_go = MagicMock()
        mock_go.is_terminal.return_value = False
        mock_go.get_legal_actions.return_value = [0, 1, board_size**2]
        mock_go.get_state.return_value = np.zeros((17, 5, 5), dtype=np.float32)

        mock_eval_result = MagicMock()
        mock_eval_result.policy = np.array([0.5, 0.3, 0.2])

        mock_neural_eval = MagicMock()
        mock_neural_eval.evaluate.return_value = mock_eval_result

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = {0: 50, 1: 10, 25: 1}

        with (
            patch("src.training.evaluation.SimpleGoGame", return_value=mock_go),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
            patch("src.training.evaluation.random") as mock_random,
        ):
            # 1 random move, and it is a pass
            mock_random.randint.return_value = 1
            mock_random.choice.return_value = board_size**2  # pass action

            evaluator.neural_evaluator = mock_neural_eval
            result = evaluator.measure_policy_agreement(
                n_positions=1,
                board_size=board_size,
            )

        mock_go.play_pass.assert_called_once()
        assert isinstance(result, float)

    def test_agreement_skips_terminal_positions(self) -> None:
        """Positions that become terminal during random moves are skipped."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        mock_go = MagicMock()
        # Position 1: becomes terminal after random moves -> skip
        # Position 2: not terminal -> evaluated
        call_count = [0]

        def is_terminal_side_effect() -> bool:
            call_count[0] += 1
            # First position: terminal after random moves (calls 1-3 in loop, 4 at check)
            # We'll make it terminal on first check inside the random moves loop
            if call_count[0] <= 2:
                # First position: is_terminal returns True inside loop
                return True
            # Second position: not terminal
            return False

        mock_go.is_terminal.side_effect = is_terminal_side_effect
        mock_go.get_legal_actions.return_value = [0, 1, 2]
        mock_go.get_state.return_value = np.zeros((17, 5, 5), dtype=np.float32)

        mock_eval_result = MagicMock()
        mock_eval_result.policy = np.array([0.7, 0.2, 0.1])

        mock_neural_eval = MagicMock()
        mock_neural_eval.evaluate.return_value = mock_eval_result

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = {0: 100, 1: 20, 2: 5}

        with (
            patch("src.training.evaluation.SimpleGoGame", return_value=mock_go),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
            patch("src.training.evaluation.random") as mock_random,
        ):
            mock_random.randint.return_value = 1
            mock_random.choice.return_value = 0

            evaluator.neural_evaluator = mock_neural_eval
            result = evaluator.measure_policy_agreement(
                n_positions=2,
                board_size=5,
            )

        # Only 1 out of 2 positions was evaluated (the other was terminal)
        assert isinstance(result, float)

    def test_agreement_multiple_positions_partial_match(self) -> None:
        """2 positions, only 1 matches -> 0.5 agreement."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        mock_go = MagicMock()
        mock_go.is_terminal.return_value = False
        mock_go.get_legal_actions.return_value = [0, 1, 2]
        mock_go.get_state.return_value = np.zeros((17, 5, 5), dtype=np.float32)

        # First position: argmax=0, second: argmax=1
        eval_results = [
            MagicMock(policy=np.array([0.7, 0.2, 0.1])),  # argmax=0
            MagicMock(policy=np.array([0.1, 0.7, 0.2])),  # argmax=1
        ]

        mock_neural_eval = MagicMock()
        mock_neural_eval.evaluate.side_effect = eval_results

        mock_mcts = MagicMock()
        # MCTS always picks action 0
        mock_mcts.search.return_value = {0: 100, 1: 20, 2: 5}

        with (
            patch("src.training.evaluation.SimpleGoGame", return_value=mock_go),
            patch("src.training.evaluation.MCTS", return_value=mock_mcts),
            patch("src.training.evaluation.random") as mock_random,
        ):
            mock_random.randint.return_value = 0  # no random moves
            evaluator.neural_evaluator = mock_neural_eval
            result = evaluator.measure_policy_agreement(
                n_positions=2,
                board_size=5,
            )

        assert result == 0.5


# ---------------------------------------------------------------------------
# evaluate_vs_random / evaluate_vs_model additional branches
# ---------------------------------------------------------------------------


class TestEvaluateVsRandomAdditional:
    """Additional branch coverage for evaluate_vs_random."""

    def test_vs_random_with_no_board_size_uses_random_choice(self) -> None:
        """When board_size=None, random.choice picks from self.board_sizes."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[5, 9])

        with patch.object(evaluator, "_play_game", return_value=(1.0, 10)):
            result = evaluator.evaluate_vs_random(n_games=2, board_size=None)

        assert result.n_games == 2
        assert result.metadata["board_size"] is None

    def test_vs_random_model_loss_and_draw(self) -> None:
        """Cover loss and draw counting for model player."""
        model = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        # game 0: model is black, outcome=-1.0 -> model_outcome=-1.0 (loss)
        # game 1: model is white, outcome=1.0 -> model_outcome=-1.0 (loss)
        # game 2: model is black, outcome=0.0 -> model_outcome=0.0 (draw)
        # game 3: model is white, outcome=0.0 -> model_outcome=0.0 (draw)
        outcomes = [(-1.0, 10), (1.0, 10), (0.0, 10), (0.0, 10)]

        with patch.object(evaluator, "_play_game", side_effect=outcomes):
            result = evaluator.evaluate_vs_random(n_games=4, board_size=9)

        assert result.losses == 2
        assert result.draws == 2
        assert result.wins == 0

    def test_vs_random_with_game_interface(self) -> None:
        """evaluate_vs_random uses game.action_space_size when game is set."""
        model = _small_model()
        mock_game = MagicMock()
        mock_game.action_space_size = 100
        evaluator = Evaluator(model=model, device="cpu", game=mock_game)

        with patch.object(evaluator, "_play_game", return_value=(1.0, 10)):
            result = evaluator.evaluate_vs_random(n_games=1, board_size=9)

        assert result.n_games == 1


class TestEvaluateVsModelAdditional:
    """Additional branch coverage for evaluate_vs_model."""

    def test_vs_model_with_no_board_size(self) -> None:
        """board_size=None causes random selection from self.board_sizes."""
        model = _small_model()
        opponent = _small_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[5])

        with patch.object(evaluator, "_play_game", return_value=(0.0, 10)):
            result = evaluator.evaluate_vs_model(opponent, n_games=2, board_size=None)

        assert result.n_games == 2

    def test_vs_model_all_draws(self) -> None:
        """All games ending in draws."""
        model = _small_model()
        opponent = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        with patch.object(evaluator, "_play_game", return_value=(0.0, 20)):
            result = evaluator.evaluate_vs_model(opponent, n_games=4, board_size=9)

        assert result.draws == 4
        assert result.wins == 0
        assert result.losses == 0
        assert result.win_rate == 0.0

    def test_vs_model_loss_counted_correctly(self) -> None:
        """Model losses are counted when outcome is negative for model."""
        model = _small_model()
        opponent = _small_model()
        evaluator = Evaluator(model=model, device="cpu")

        # game 0: model=black, outcome=-1 -> model_outcome=-1 (loss)
        # game 1: model=white, outcome=-1 -> model_outcome=1 (win, since -(-1)=1)
        outcomes = [(-1.0, 10), (-1.0, 10)]

        with patch.object(evaluator, "_play_game", side_effect=outcomes):
            result = evaluator.evaluate_vs_model(opponent, n_games=2, board_size=9)

        assert result.losses == 1
        assert result.wins == 1
