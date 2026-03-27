"""Coverage tests for evaluation pipeline.

Tests cover:
- EvaluationResult: Dataclass and serialization
- Evaluator: Initialization and game play
- quick_evaluate: Quick evaluation helper
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.training.evaluation import EvaluationResult, Evaluator, quick_evaluate

SEED = 42


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_model() -> MagicMock:
    """Return a minimal mock that satisfies FNetEvaluator's needs."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    # forward / forward_fast return an object with policy_logits and value
    output = MagicMock()
    output.policy_logits = torch.zeros(1, 82)  # batch=1, 9*9+1
    output.value = torch.zeros(1, 1)
    model.return_value = output
    model.forward_fast = MagicMock(return_value=output)
    return model


def _make_mcts_config(n_simulations: int = 2) -> MagicMock:
    cfg = MagicMock()
    cfg.n_simulations = n_simulations
    cfg.c_puct = 1.0
    cfg.dirichlet_alpha = 0.3
    cfg.dirichlet_epsilon = 0.25
    return cfg


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

    def test_mcts_kwargs_empty_when_no_config(self, mock_model) -> None:
        evaluator = Evaluator(model=mock_model, device="cpu")
        assert evaluator._mcts_kwargs == {}

    def test_device_as_torch_device(self, mock_model) -> None:
        evaluator = Evaluator(model=mock_model, device=torch.device("cpu"))
        assert evaluator.device == torch.device("cpu")

    @pytest.mark.parametrize("board_sizes", [None, [5], [9, 13, 19], [19]])
    def test_board_sizes_parametrize(self, mock_model, board_sizes) -> None:
        evaluator = Evaluator(model=mock_model, device="cpu", board_sizes=board_sizes)
        expected = board_sizes if board_sizes is not None else [9, 13, 19]
        assert evaluator.board_sizes == expected


# ---------------------------------------------------------------------------
# Helpers that fake out heavy _play_game machinery
# ---------------------------------------------------------------------------


def _make_evaluator_with_mocked_play(
    n_games: int,
    outcomes: list[tuple[float, int]],
    board_sizes: list[int] | None = None,
    game: object = None,
) -> Evaluator:
    """Build an Evaluator whose _play_game is patched to return ``outcomes``."""
    model = _make_mock_model()
    evaluator = Evaluator(
        model=model,
        device="cpu",
        board_sizes=board_sizes or [9],
        game=game,
    )
    # Cycle through provided outcomes
    call_count = [0]

    def fake_play_game(**kwargs):  # noqa: ARG001
        idx = call_count[0] % len(outcomes)
        call_count[0] += 1
        return outcomes[idx]

    evaluator._play_game = fake_play_game  # type: ignore[method-assign]
    return evaluator


class TestEvaluateVsRandom:
    """Tests for Evaluator.evaluate_vs_random.

    Color alternation: game_idx even → model is black (outcome used directly),
    game_idx odd → model is white (outcome negated).
    """

    @pytest.mark.parametrize(
        ("outcomes", "expected_wins", "expected_losses", "expected_draws"),
        [
            # All draws (0.0 always maps to draw regardless of color)
            ([(0.0, 8)] * 4, 0, 0, 4),
            # game 0 (black, +1→win), game 1 (white, -(-1)=+1→win),
            # game 2 (black, +1→win), game 3 (white, -(-1)=+1→win)
            # outcomes: odd games white plays, outcome=-1 → model_outcome=+1 → win
            ([(1.0, 5), (-1.0, 5)], 4, 0, 0),
            # outcome alternates -1/+1: black loses, white flips to loss
            ([(- 1.0, 5), (1.0, 5)], 0, 4, 0),
        ],
    )
    def test_win_loss_draw_counts(
        self,
        outcomes,
        expected_wins,
        expected_losses,
        expected_draws,
    ) -> None:
        n_games = 4
        evaluator = _make_evaluator_with_mocked_play(n_games, outcomes)
        result = evaluator.evaluate_vs_random(n_games=n_games, board_size=9)

        assert result.n_games == n_games
        assert result.wins == expected_wins
        assert result.losses == expected_losses
        assert result.draws == expected_draws

    def test_win_rate_calculation(self) -> None:
        # outcomes cycle: (1.0,10),(-1.0,5),(1.0,10),(-1.0,5)
        # game 0 (black): +1 → win; game 1 (white): -(-1)=+1 → win
        # game 2 (black): +1 → win; game 3 (white): -(-1)=+1 → win
        # 4 wins → win_rate = 1.0
        outcomes = [(1.0, 10), (-1.0, 5)]
        evaluator = _make_evaluator_with_mocked_play(4, outcomes)
        result = evaluator.evaluate_vs_random(n_games=4, board_size=9)
        assert abs(result.win_rate - 1.0) < 1e-9

    def test_avg_game_length(self) -> None:
        # move counts: 10, 20, 30, 40 → avg = 25
        outcomes = [(1.0, 10), (1.0, 20), (1.0, 30), (1.0, 40)]
        evaluator = _make_evaluator_with_mocked_play(4, outcomes)
        result = evaluator.evaluate_vs_random(n_games=4, board_size=9)
        assert abs(result.avg_game_length - 25.0) < 1e-9

    def test_zero_games_returns_zero_win_rate(self) -> None:
        evaluator = _make_evaluator_with_mocked_play(0, [(1.0, 5)])
        result = evaluator.evaluate_vs_random(n_games=0, board_size=9)
        assert result.win_rate == 0.0
        assert result.avg_game_length == 0.0

    def test_metadata_opponent_and_board_size(self) -> None:
        evaluator = _make_evaluator_with_mocked_play(2, [(1.0, 5)])
        result = evaluator.evaluate_vs_random(n_games=2, board_size=13)
        assert result.metadata["opponent"] == "random"
        assert result.metadata["board_size"] == 13

    def test_random_board_size_when_none(self) -> None:
        """When board_size=None, the evaluator picks from self.board_sizes."""
        model = _make_mock_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[9, 13])
        # Just patch _play_game; the board size selection itself is tested by
        # ensuring the method completes without error.
        evaluator._play_game = MagicMock(return_value=(1.0, 5))  # type: ignore[method-assign]
        result = evaluator.evaluate_vs_random(n_games=4, board_size=None)
        assert result.n_games == 4

    def test_model_color_alternation(self) -> None:
        """Even game_idx: model is black; odd game_idx: model is white.

        Both games return outcome=1.0 (black wins).
        game 0: model is black → model_outcome=+1 → win.
        game 1: model is white → model_outcome=-1 → loss.
        """
        outcomes = [(1.0, 5)]  # cycled for both games
        evaluator = _make_evaluator_with_mocked_play(2, outcomes)
        result = evaluator.evaluate_vs_random(n_games=2, board_size=9)
        assert result.wins == 1
        assert result.losses == 1

    def test_game_interface_action_space_used(self) -> None:
        """When self.game is set, n_actions comes from game.action_space_size."""
        game = MagicMock()
        game.action_space_size = 100
        model = _make_mock_model()
        evaluator = Evaluator(model=model, device="cpu", game=game, board_sizes=[9])
        evaluator._play_game = MagicMock(return_value=(1.0, 5))  # type: ignore[method-assign]
        result = evaluator.evaluate_vs_random(n_games=2, board_size=9)
        assert result.n_games == 2


class TestEvaluateVsModel:
    """Tests for Evaluator.evaluate_vs_model."""

    def test_basic_vs_model(self) -> None:
        # Cycle outcomes: (1,10),(-1,10),(1,10),(-1,10)
        # game 0 black +1→win, game 1 white -(-1)=+1→win, etc. → 4 wins
        outcomes = [(1.0, 10), (-1.0, 10)]
        evaluator = _make_evaluator_with_mocked_play(4, outcomes)
        opponent = _make_mock_model()
        result = evaluator.evaluate_vs_model(opponent_model=opponent, n_games=4, board_size=9)
        assert result.n_games == 4
        assert result.wins == 4
        assert result.metadata["opponent"] == "model"

    def test_zero_games(self) -> None:
        evaluator = _make_evaluator_with_mocked_play(0, [(0.0, 1)])
        opponent = _make_mock_model()
        result = evaluator.evaluate_vs_model(opponent_model=opponent, n_games=0)
        assert result.win_rate == 0.0
        assert result.avg_game_length == 0.0

    def test_color_alternation(self) -> None:
        # Both games return -1.0 (black loses):
        # game 0 (model black): model_outcome = -1 → loss
        # game 1 (model white): model_outcome = -(-1) = +1 → win
        outcomes = [(-1.0, 5)]
        evaluator = _make_evaluator_with_mocked_play(2, outcomes)
        opponent = _make_mock_model()
        result = evaluator.evaluate_vs_model(opponent_model=opponent, n_games=2, board_size=9)
        assert result.wins == 1
        assert result.losses == 1

    def test_random_board_size(self) -> None:
        evaluator = _make_evaluator_with_mocked_play(3, [(0.0, 5)])
        opponent = _make_mock_model()
        result = evaluator.evaluate_vs_model(
            opponent_model=opponent, n_games=3, board_size=None
        )
        assert result.n_games == 3


class TestEvaluateVsCheckpoint:
    """Tests for Evaluator.evaluate_vs_checkpoint."""

    def test_checkpoint_calls_load_and_vs_model(self) -> None:
        """evaluate_vs_checkpoint should load a model and delegate to evaluate_vs_model."""
        model = _make_mock_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[9])

        loaded_model = _make_mock_model()

        with patch.object(evaluator, "_load_model_from_checkpoint", return_value=loaded_model):
            with patch.object(
                evaluator, "evaluate_vs_model", return_value=EvaluationResult(
                    win_rate=0.6, n_games=4, wins=3, losses=1, draws=0,
                    avg_game_length=10.0, metadata={"opponent": "model"},
                )
            ) as mock_vsm:
                result = evaluator.evaluate_vs_checkpoint(
                    checkpoint_path="/fake/checkpoint.pt",
                    n_games=4,
                    board_size=9,
                )

        mock_vsm.assert_called_once_with(
            opponent_model=loaded_model, n_games=4, board_size=9
        )
        assert result.metadata["opponent"] == "checkpoint"
        assert result.metadata["checkpoint_path"] == "/fake/checkpoint.pt"

    def test_checkpoint_path_as_path_object(self) -> None:
        model = _make_mock_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[9])
        loaded_model = _make_mock_model()

        with patch.object(evaluator, "_load_model_from_checkpoint", return_value=loaded_model):
            with patch.object(
                evaluator, "evaluate_vs_model", return_value=EvaluationResult(
                    win_rate=0.0, n_games=2, wins=0, losses=2, draws=0,
                    avg_game_length=5.0, metadata={"opponent": "model"},
                )
            ):
                result = evaluator.evaluate_vs_checkpoint(
                    checkpoint_path=Path("/fake/model.pt"),
                    n_games=2,
                )

        assert result.metadata["checkpoint_path"] == "/fake/model.pt"


class TestLoadModelFromCheckpoint:
    """Tests for Evaluator._load_model_from_checkpoint.

    AlphaGalerkinModel and AlphaGalerkinConfig are imported locally inside
    _load_model_from_checkpoint, so we patch them at their source modules.
    """

    def test_load_with_dict_config(self) -> None:
        model = _make_mock_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[9])

        checkpoint = {"config": {"some_field": 1}, "model_state_dict": {}}

        mock_loaded_model = MagicMock()
        mock_loaded_model.to = MagicMock(return_value=mock_loaded_model)
        mock_loaded_model.eval = MagicMock(return_value=mock_loaded_model)
        mock_loaded_model.load_state_dict = MagicMock()

        with patch("torch.load", return_value=checkpoint):
            with patch("src.modeling.model.AlphaGalerkinModel") as MockModel:
                with patch("config.schemas.AlphaGalerkinConfig") as MockConfig:
                    mock_cfg_instance = MagicMock()
                    MockConfig.return_value = mock_cfg_instance
                    MockModel.return_value = mock_loaded_model
                    # Also patch the local imports inside the function
                    with patch("src.training.evaluation.torch.load", return_value=checkpoint):
                        with patch(
                            "src.training.evaluation.AlphaGalerkinModel",
                            MockModel,
                            create=True,
                        ):
                            with patch(
                                "src.training.evaluation.AlphaGalerkinConfig",
                                MockConfig,
                                create=True,
                            ):
                                result = evaluator._load_model_from_checkpoint(
                                    "/fake/ckpt.pt"
                                )

        assert result is mock_loaded_model

    def test_load_without_dict_config_uses_self_model_config(self) -> None:
        """When checkpoint config is not a dict, fall back to self.model.config."""
        model = _make_mock_model()
        model.config = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[9])

        checkpoint = {"config": object(), "model_state_dict": {}}

        mock_loaded_model = MagicMock()
        mock_loaded_model.to = MagicMock(return_value=mock_loaded_model)
        mock_loaded_model.eval = MagicMock(return_value=mock_loaded_model)
        mock_loaded_model.load_state_dict = MagicMock()

        with patch("src.training.evaluation.torch.load", return_value=checkpoint):
            with patch(
                "src.modeling.model.AlphaGalerkinModel",
                return_value=mock_loaded_model,
            ) as MockModel:
                result = evaluator._load_model_from_checkpoint("/fake/ckpt.pt")

        MockModel.assert_called_once_with(model.config)
        assert result is mock_loaded_model

    def test_load_missing_config_key_uses_self_model_config(self) -> None:
        """When checkpoint has no 'config' key, fall back to self.model.config."""
        model = _make_mock_model()
        model.config = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[9])

        checkpoint = {"model_state_dict": {}}

        mock_loaded_model = MagicMock()
        mock_loaded_model.to = MagicMock(return_value=mock_loaded_model)
        mock_loaded_model.eval = MagicMock(return_value=mock_loaded_model)
        mock_loaded_model.load_state_dict = MagicMock()

        with patch("src.training.evaluation.torch.load", return_value=checkpoint):
            with patch(
                "src.modeling.model.AlphaGalerkinModel",
                return_value=mock_loaded_model,
            ) as MockModel:
                result = evaluator._load_model_from_checkpoint(Path("/fake/ckpt.pt"))

        MockModel.assert_called_once_with(model.config)
        assert result is mock_loaded_model


class TestEvaluateMultiResolution:
    """Tests for Evaluator.evaluate_multi_resolution."""

    def test_returns_result_per_board_size(self) -> None:
        sizes = [9, 13]
        model = _make_mock_model()
        evaluator = Evaluator(model=model, device="cpu", board_sizes=sizes)
        fake_result = EvaluationResult(
            win_rate=0.5, n_games=4, wins=2, losses=2, draws=0, avg_game_length=20.0
        )
        with patch.object(evaluator, "evaluate_vs_random", return_value=fake_result) as mock_evr:
            results = evaluator.evaluate_multi_resolution(n_games_per_size=4)

        assert set(results.keys()) == {9, 13}
        assert mock_evr.call_count == 2
        # Check each call used the correct board size and n_games
        calls_kwargs = [c.kwargs for c in mock_evr.call_args_list]
        called_sizes = {kw["board_size"] for kw in calls_kwargs}
        assert called_sizes == {9, 13}

    def test_empty_board_sizes(self) -> None:
        model = _make_mock_model()
        # board_sizes=[] but we can't pass it directly (default is [9,13,19])
        # We set it manually after construction to avoid FNetEvaluator issues
        evaluator = Evaluator(model=model, device="cpu", board_sizes=[9])
        evaluator.board_sizes = []
        results = evaluator.evaluate_multi_resolution(n_games_per_size=2)
        assert results == {}


class TestPlayGameDispatch:
    """Tests for Evaluator._play_game routing logic."""

    def test_delegates_to_go_when_game_is_none(self) -> None:
        model = _make_mock_model()
        evaluator = Evaluator(model=model, device="cpu")
        black_ev = MagicMock()
        white_ev = MagicMock()

        with patch.object(evaluator, "_play_game_go", return_value=(1.0, 5)) as mock_go:
            with patch.object(
                evaluator, "_play_game_generic", return_value=(0.0, 3)
            ) as mock_gen:
                outcome, moves = evaluator._play_game(
                    board_size=9,
                    black_evaluator=black_ev,
                    white_evaluator=white_ev,
                )

        mock_go.assert_called_once()
        mock_gen.assert_not_called()
        assert outcome == 1.0
        assert moves == 5

    def test_delegates_to_generic_when_game_set(self) -> None:
        model = _make_mock_model()
        game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=game)
        black_ev = MagicMock()
        white_ev = MagicMock()

        with patch.object(
            evaluator, "_play_game_generic", return_value=(-1.0, 8)
        ) as mock_gen:
            with patch.object(evaluator, "_play_game_go", return_value=(0.0, 1)) as mock_go:
                outcome, moves = evaluator._play_game(
                    board_size=9,
                    black_evaluator=black_ev,
                    white_evaluator=white_ev,
                )

        mock_gen.assert_called_once()
        mock_go.assert_not_called()
        assert outcome == -1.0
        assert moves == 8


class TestPlayGameGeneric:
    """Tests for Evaluator._play_game_generic."""

    def _make_generic_evaluator(self) -> tuple[Evaluator, MagicMock]:
        model = _make_mock_model()
        game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=game)
        return evaluator, game

    def test_terminal_from_start_returns_winner(self) -> None:
        evaluator, game = self._make_generic_evaluator()
        state = MagicMock()
        state.current_player = 1
        game.initial_state.return_value = state
        game.is_terminal.return_value = True

        game_result = MagicMock()
        game_result.winner = 1
        game.get_result.return_value = game_result

        with patch("src.training.evaluation.MCTS") as MockMCTS:
            outcome, moves = evaluator._play_game_generic(
                black_evaluator=MagicMock(), white_evaluator=MagicMock()
            )

        assert outcome == 1.0
        assert moves == 0
        MockMCTS.assert_called()

    def test_max_moves_causes_draw(self) -> None:
        evaluator, game = self._make_generic_evaluator()
        state = MagicMock()
        state.current_player = 1
        game.initial_state.return_value = state
        # Never terminal → hits max_moves
        game.is_terminal.return_value = False
        game.apply_action.return_value = state

        mock_mcts_instance = MagicMock()
        mock_mcts_instance.get_action.return_value = 0
        mock_mcts_instance.advance = MagicMock()

        with patch("src.training.evaluation.MCTS", return_value=mock_mcts_instance):
            outcome, moves = evaluator._play_game_generic(
                black_evaluator=MagicMock(),
                white_evaluator=MagicMock(),
                max_moves=3,
            )

        assert outcome == 0.0
        assert moves == 3

    def test_player_switching(self) -> None:
        """Verify p1_mcts is used for player 1 and p2_mcts for player -1."""
        evaluator, game = self._make_generic_evaluator()

        # Two moves then terminal
        state_p1 = MagicMock()
        state_p1.current_player = 1
        state_p2 = MagicMock()
        state_p2.current_player = -1

        game.initial_state.return_value = state_p1
        # is_terminal: False (loop iteration 1), False (loop iteration 2), True (after)
        game.is_terminal.side_effect = [False, False, True, True]
        game.apply_action.side_effect = [state_p2, state_p1]

        result_obj = MagicMock()
        result_obj.winner = 1
        game.get_result.return_value = result_obj

        p1_mcts = MagicMock()
        p1_mcts.get_action.return_value = 0
        p1_mcts.advance = MagicMock()
        p2_mcts = MagicMock()
        p2_mcts.get_action.return_value = 1
        p2_mcts.advance = MagicMock()

        call_count = [0]

        def mcts_factory(**kwargs):  # noqa: ARG001
            call_count[0] += 1
            return p1_mcts if call_count[0] == 1 else p2_mcts

        with patch("src.training.evaluation.MCTS", side_effect=mcts_factory):
            outcome, moves = evaluator._play_game_generic(
                black_evaluator=MagicMock(), white_evaluator=MagicMock()
            )

        assert outcome == 1.0
        assert moves == 2


class TestPlayGameGo:
    """Tests for Evaluator._play_game_go."""

    def _make_go_evaluator(self) -> Evaluator:
        model = _make_mock_model()
        return Evaluator(model=model, device="cpu")

    def test_immediate_terminal_returns_winner_black_perspective(self) -> None:
        """When is_terminal on entry and current_player==BLACK, return -float(winner)."""
        evaluator = self._make_go_evaluator()

        mock_game = MagicMock()
        mock_game.is_terminal.return_value = True
        mock_game.get_winner.return_value = 1

        mock_mcts = MagicMock()

        # The production code does: game.current_player == SimpleGoGame.BLACK
        # Since we patch SimpleGoGame entirely, SimpleGoGame.BLACK becomes mock_cls.BLACK.
        # We make current_player equal to that mock attribute to trigger the branch.
        sentinel_black = object()

        with patch("src.training.evaluation.SimpleGoGame") as MockSimpleGoGame:
            MockSimpleGoGame.return_value = mock_game
            MockSimpleGoGame.BLACK = sentinel_black
            mock_game.current_player = sentinel_black  # will compare equal

            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                outcome, moves = evaluator._play_game_go(
                    board_size=9,
                    black_evaluator=MagicMock(),
                    white_evaluator=MagicMock(),
                )

        # current_player == BLACK → return -float(winner) = -1.0
        assert outcome == -1.0
        assert moves == 0

    def test_terminal_white_player_perspective(self) -> None:
        """When current_player != BLACK, winner is returned as-is."""
        evaluator = self._make_go_evaluator()

        mock_game = MagicMock()
        mock_game.is_terminal.return_value = True
        mock_game.get_winner.return_value = 1

        sentinel_black = object()
        # Use a different sentinel for WHITE so current_player != BLACK
        sentinel_white = object()

        mock_mcts = MagicMock()

        with patch("src.training.evaluation.SimpleGoGame") as MockSimpleGoGame:
            MockSimpleGoGame.return_value = mock_game
            MockSimpleGoGame.BLACK = sentinel_black
            mock_game.current_player = sentinel_white  # != BLACK

            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                outcome, moves = evaluator._play_game_go(
                    board_size=9,
                    black_evaluator=MagicMock(),
                    white_evaluator=MagicMock(),
                )

        assert outcome == 1.0
        assert moves == 0

    def test_max_moves_draw(self) -> None:
        evaluator = self._make_go_evaluator()
        board_size = 9

        mock_game = MagicMock()
        mock_game.is_terminal.return_value = False
        sentinel_black = object()

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = board_size**2  # pass action
        mock_mcts.advance = MagicMock()

        with patch("src.training.evaluation.SimpleGoGame") as MockSimpleGoGame:
            MockSimpleGoGame.return_value = mock_game
            MockSimpleGoGame.BLACK = sentinel_black
            mock_game.current_player = sentinel_black

            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                outcome, moves = evaluator._play_game_go(
                    board_size=board_size,
                    black_evaluator=MagicMock(),
                    white_evaluator=MagicMock(),
                    max_moves=3,
                )

        assert outcome == 0.0
        assert moves == 3

    def test_invalid_move_causes_pass(self) -> None:
        """play() returning False should trigger play_pass."""
        evaluator = self._make_go_evaluator()
        board_size = 9

        mock_game = MagicMock()
        mock_game.is_terminal.side_effect = [False, True, True]
        mock_game.play.return_value = False
        mock_game.get_winner.return_value = 0

        sentinel_black = object()

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 5  # board position (not pass)
        mock_mcts.advance = MagicMock()

        with patch("src.training.evaluation.SimpleGoGame") as MockSimpleGoGame:
            MockSimpleGoGame.return_value = mock_game
            MockSimpleGoGame.BLACK = sentinel_black
            mock_game.current_player = sentinel_black

            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                evaluator._play_game_go(
                    board_size=board_size,
                    black_evaluator=MagicMock(),
                    white_evaluator=MagicMock(),
                )

        mock_game.play_pass.assert_called()

    def test_valid_move_played(self) -> None:
        """play() returning True should not trigger play_pass."""
        evaluator = self._make_go_evaluator()
        board_size = 9

        mock_game = MagicMock()
        mock_game.is_terminal.side_effect = [False, True, True]
        mock_game.play.return_value = True
        mock_game.get_winner.return_value = 1

        sentinel_black = object()
        sentinel_white = object()

        mock_mcts = MagicMock()
        mock_mcts.get_action.return_value = 10  # valid board position
        mock_mcts.advance = MagicMock()

        with patch("src.training.evaluation.SimpleGoGame") as MockSimpleGoGame:
            MockSimpleGoGame.return_value = mock_game
            MockSimpleGoGame.BLACK = sentinel_black
            mock_game.current_player = sentinel_white  # != BLACK

            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                outcome, _ = evaluator._play_game_go(
                    board_size=board_size,
                    black_evaluator=MagicMock(),
                    white_evaluator=MagicMock(),
                )

        mock_game.play_pass.assert_not_called()
        assert outcome == 1.0


class TestEvaluateVsEngine:
    """Tests for Evaluator.evaluate_vs_engine."""

    def test_requires_game_set(self) -> None:
        model = _make_mock_model()
        evaluator = Evaluator(model=model, device="cpu")
        with pytest.raises(ValueError, match="evaluate_vs_engine requires"):
            evaluator.evaluate_vs_engine(
                engine_config=MagicMock(),
                match_config=MagicMock(),
            )

    def _make_match_result(
        self,
        win_rate: float = 0.6,
        total_games: int = 10,
        wins: int = 6,
        losses: int = 3,
        draws: int = 1,
        elo_estimate=None,
        move_count: int = 20,
    ) -> MagicMock:
        match_result = MagicMock()
        match_result.win_rate = win_rate
        match_result.total_games = total_games
        match_result.wins = wins
        match_result.losses = losses
        match_result.draws = draws
        match_result.elo_estimate = elo_estimate
        game_mock = MagicMock()
        game_mock.move_count = move_count
        match_result.games = [game_mock] * max(total_games, 1) if total_games > 0 else []
        return match_result

    def test_with_elo_estimate(self) -> None:
        model = _make_mock_model()
        game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=game)

        elo_estimate = MagicMock()
        elo_estimate.elo_difference = 50.0
        elo_estimate.confidence_interval = (-10.0, 110.0)
        elo_estimate.likelihood_of_superiority = 0.85

        match_result = self._make_match_result(elo_estimate=elo_estimate)
        mock_match_instance = MagicMock()
        mock_match_instance.play_match.return_value = match_result

        em_patch = patch(
            "src.engines.match.EngineMatch",
            return_value=mock_match_instance,
            create=True,
        )
        eval_patch = patch(
            "src.training.evaluation.EngineMatch",
            return_value=mock_match_instance,
            create=True,
        )
        with em_patch, eval_patch:
            result = evaluator.evaluate_vs_engine(
                engine_config=MagicMock(),
                match_config=MagicMock(),
            )

        assert result.win_rate == 0.6
        assert result.n_games == 10
        assert result.wins == 6
        assert result.metadata["opponent"] == "engine"
        assert result.metadata["elo_difference"] == 50.0

    def test_without_elo_estimate(self) -> None:
        model = _make_mock_model()
        game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=game)

        match_result = self._make_match_result(
            win_rate=0.4, total_games=0, wins=0, losses=0, draws=0,
            elo_estimate=None,
        )

        mock_match_instance = MagicMock()
        mock_match_instance.play_match.return_value = match_result

        em_patch = patch(
            "src.engines.match.EngineMatch",
            return_value=mock_match_instance,
            create=True,
        )
        eval_patch = patch(
            "src.training.evaluation.EngineMatch",
            return_value=mock_match_instance,
            create=True,
        )
        with em_patch, eval_patch:
            result = evaluator.evaluate_vs_engine(
                engine_config=MagicMock(),
                match_config=MagicMock(),
            )

        assert "elo_difference" not in result.metadata
        assert result.avg_game_length == 0.0

    def test_mcts_config_dict_override_passed_to_match(self) -> None:
        model = _make_mock_model()
        game = MagicMock()
        evaluator = Evaluator(model=model, device="cpu", game=game)

        match_result = self._make_match_result(
            win_rate=0.5, total_games=2, wins=1, losses=1, draws=0,
            elo_estimate=None, move_count=5,
        )
        custom_mcts = {"n_simulations": 50}

        mock_match_instance = MagicMock()
        mock_match_instance.play_match.return_value = match_result

        with patch(
            "src.engines.match.EngineMatch",
            return_value=mock_match_instance,
            create=True,
        ) as MockEM:
            evaluator.evaluate_vs_engine(
                engine_config=MagicMock(),
                match_config=MagicMock(),
                mcts_config_dict=custom_mcts,
            )

        mock_match_instance.play_match.assert_called_once()
        call_kwargs = MockEM.call_args.kwargs
        assert call_kwargs["mcts_config"] == custom_mcts


class TestMeasurePolicyAgreement:
    """Tests for Evaluator.measure_policy_agreement."""

    def _make_evaluator(self) -> Evaluator:
        model = _make_mock_model()
        return Evaluator(model=model, device="cpu", board_sizes=[9])

    def test_returns_float_between_0_and_1(self) -> None:
        evaluator = self._make_evaluator()

        n_actions = 82
        policy = np.ones(n_actions, dtype=np.float32) / n_actions
        from src.mcts.evaluator import EvaluationResult as MCTSEvalResult

        eval_result = MCTSEvalResult(policy=policy, value=0.0)

        mock_game = MagicMock()
        mock_game.is_terminal.return_value = False
        mock_game.get_legal_actions.return_value = list(range(10))
        mock_game.get_state.return_value = np.zeros((17, 9, 9), dtype=np.float32)

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = {0: 0.5, 1: 0.3, 2: 0.2}

        evaluator.neural_evaluator.evaluate = MagicMock(return_value=eval_result)

        with patch("src.training.evaluation.SimpleGoGame", return_value=mock_game):
            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                rate = evaluator.measure_policy_agreement(n_positions=3, board_size=9)

        assert 0.0 <= rate <= 1.0

    def test_terminal_position_is_skipped(self) -> None:
        """Positions where game is terminal should be skipped (not counted)."""
        evaluator = self._make_evaluator()

        mock_game = MagicMock()
        # All positions are terminal → total=0 → agreement_rate=0.0
        mock_game.is_terminal.return_value = True
        mock_game.get_legal_actions.return_value = [0, 1, 2]

        mock_mcts = MagicMock()

        with patch("src.training.evaluation.SimpleGoGame", return_value=mock_game):
            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                rate = evaluator.measure_policy_agreement(n_positions=5, board_size=9)

        assert rate == 0.0

    def test_full_agreement_when_argmax_matches(self) -> None:
        evaluator = self._make_evaluator()
        n_actions = 82

        # Policy peaks at action 0
        policy = np.zeros(n_actions, dtype=np.float32)
        policy[0] = 1.0
        from src.mcts.evaluator import EvaluationResult as MCTSEvalResult
        eval_result = MCTSEvalResult(policy=policy, value=0.0)

        mock_game = MagicMock()
        mock_game.is_terminal.return_value = False
        mock_game.get_legal_actions.return_value = [0, 1, 2]
        mock_game.get_state.return_value = np.zeros((17, 9, 9), dtype=np.float32)

        # MCTS also picks action 0
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = {0: 0.9, 1: 0.05, 2: 0.05}

        evaluator.neural_evaluator.evaluate = MagicMock(return_value=eval_result)

        with patch("src.training.evaluation.SimpleGoGame", return_value=mock_game):
            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                rate = evaluator.measure_policy_agreement(n_positions=3, board_size=9)

        assert rate == 1.0

    def test_zero_positions_returns_zero(self) -> None:
        evaluator = self._make_evaluator()

        mock_game = MagicMock()
        mock_game.is_terminal.return_value = True  # always skipped
        mock_game.get_legal_actions.return_value = []
        mock_mcts = MagicMock()

        with patch("src.training.evaluation.SimpleGoGame", return_value=mock_game):
            with patch("src.training.evaluation.MCTS", return_value=mock_mcts):
                rate = evaluator.measure_policy_agreement(n_positions=0, board_size=9)

        assert rate == 0.0


class TestQuickEvaluate:
    """Tests for quick_evaluate helper function."""

    def test_returns_dict_with_expected_keys(self) -> None:
        model = _make_mock_model()
        fake_result = EvaluationResult(
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=30.0,
            metadata={"opponent": "random", "board_size": 9},
        )

        with patch("src.training.evaluation.Evaluator") as MockEvaluator:
            mock_ev = MagicMock()
            mock_ev.evaluate_vs_random.return_value = fake_result
            MockEvaluator.return_value = mock_ev

            result = quick_evaluate(model=model, n_games=10, board_size=9, device="cpu")

        assert isinstance(result, dict)
        assert "win_rate" in result
        assert "n_games" in result
        assert result["win_rate"] == 0.5

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_uses_correct_board_size(self, board_size) -> None:
        model = _make_mock_model()
        fake_result = EvaluationResult(
            win_rate=0.4,
            n_games=5,
            wins=2,
            losses=3,
            draws=0,
            avg_game_length=15.0,
            metadata={},
        )

        with patch("src.training.evaluation.Evaluator") as MockEvaluator:
            mock_ev = MagicMock()
            mock_ev.evaluate_vs_random.return_value = fake_result
            MockEvaluator.return_value = mock_ev

            quick_evaluate(model=model, n_games=5, board_size=board_size, device="cpu")

        init_kwargs = MockEvaluator.call_args.kwargs
        assert board_size in init_kwargs["board_sizes"]
        mock_ev.evaluate_vs_random.assert_called_once_with(
            n_games=5, board_size=board_size
        )

    def test_constructs_evaluator_with_correct_device(self) -> None:
        model = _make_mock_model()
        fake_result = EvaluationResult(
            win_rate=0.0, n_games=2, wins=0, losses=2, draws=0,
            avg_game_length=5.0, metadata={},
        )

        with patch("src.training.evaluation.Evaluator") as MockEvaluator:
            mock_ev = MagicMock()
            mock_ev.evaluate_vs_random.return_value = fake_result
            MockEvaluator.return_value = mock_ev

            quick_evaluate(model=model, n_games=2, board_size=9, device="cuda")

        init_kwargs = MockEvaluator.call_args.kwargs
        assert init_kwargs["device"] == "cuda"


class TestEvaluationResultEdgeCases:
    """Additional edge-case tests for EvaluationResult."""

    def test_to_dict_metadata_overrides_not_possible(self) -> None:
        """Core keys are always present; metadata adds new keys."""
        result = EvaluationResult(
            win_rate=1.0,
            n_games=1,
            wins=1,
            losses=0,
            draws=0,
            avg_game_length=5.0,
            metadata={"extra_key": "extra_value"},
        )
        d = result.to_dict()
        assert d["extra_key"] == "extra_value"
        assert d["win_rate"] == 1.0

    @pytest.mark.parametrize(
        ("win_rate", "policy_agreement", "avg_value_error"),
        [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
        ],
    )
    def test_optional_fields_stored_correctly(
        self, win_rate, policy_agreement, avg_value_error
    ) -> None:
        result = EvaluationResult(
            win_rate=win_rate,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=20.0,
            avg_value_error=avg_value_error,
            policy_agreement=policy_agreement,
        )
        assert result.policy_agreement == policy_agreement
        assert result.avg_value_error == avg_value_error
