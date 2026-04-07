"""End-to-end chess training smoke test.

Validates the full pipeline: model -> self-play -> replay buffer -> training step.
Uses minimal configuration to run fast while exercising the entire code path.

The self-play tests mock MCTS search to return uniform random policies instantly,
since these tests validate the pipeline plumbing (not MCTS search quality).
Without mocking, chess self-play with 4672 actions is far too slow for CI.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from config.schemas import MCTSConfig, OperatorConfig, TrainingConfig
from src.games.chess import ChessGame
from src.modeling.model import AlphaGalerkinModel
from src.training.replay_buffer import UniformReplayBuffer
from src.training.self_play import SelfPlayWorker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_move_counter: int = 0


def _fast_mcts_search(self, game, add_noise=True):  # noqa: ANN001, ANN202, ARG001
    """Return a uniform policy over legal actions instantly (no tree search).

    This replaces ``MCTS.search`` so that self-play games complete in
    milliseconds instead of minutes.
    """
    legal = game.get_legal_actions()
    n = len(legal)
    if n == 0:
        return {}
    prob = 1.0 / n
    return dict.fromkeys(legal, prob)


def _make_early_termination_wrapper(game_instance, max_moves=4):  # noqa: ANN001, ANN202
    """Create a wrapper that forces early termination of chess games.

    Forces termination after *max_moves* calls that returned False.
    ``game_instance`` is the concrete :class:`ChessGame` so we can call the
    *original* ``is_terminal`` via the unbound class method.  The wrapper is
    used as a ``side_effect`` on a :class:`~unittest.mock.MagicMock` that
    replaces the class-level attribute; because mocks are not descriptors the
    wrapper receives only ``(state,)`` -- *not* ``(self, state)``.
    """
    real_is_terminal = type(game_instance).is_terminal
    call_count = 0

    def _wrapper(state):  # noqa: ANN001, ANN202
        nonlocal call_count
        if real_is_terminal(game_instance, state):
            return True
        call_count += 1
        return call_count > max_moves

    return _wrapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chess_config() -> tuple[OperatorConfig, MCTSConfig, TrainingConfig]:
    """Minimal chess training configuration for smoke test."""
    op = OperatorConfig(
        d_model=32,
        d_key=8,
        d_value=8,
        d_ffn=64,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=16,
        input_channels=119,
        game_type="chess",
        action_space_size=4672,
    )
    mcts = MCTSConfig(n_simulations=4, c_puct=2.5, dirichlet_alpha=0.3)
    training = TrainingConfig(
        total_steps=2,
        batch_size=8,
        n_self_play_games=2,
        replay_buffer_size=1000,
        learning_rate=0.001,
    )
    return op, mcts, training


class TestChessTrainingE2E:
    """End-to-end chess training smoke tests."""

    @pytest.mark.slow
    def test_self_play_to_training_step(self, chess_config: tuple) -> None:
        """Full pipeline: self-play -> buffer -> training step.

        MCTS search is mocked to return uniform random policies so the game
        finishes quickly.  ``is_terminal`` is patched to cap games at 4 moves.
        """
        op_config, mcts_config, train_config = chess_config
        game = ChessGame()

        # 1. Create model
        model = AlphaGalerkinModel(op_config)
        model.eval()

        # 2. Create self-play worker
        worker = SelfPlayWorker(
            model=model,
            mcts_config=mcts_config,
            device=torch.device("cpu"),
            game=game,
        )

        # Mock MCTS.search and cap game length so each game takes ~ms
        with patch("src.mcts.search.MCTS.search", _fast_mcts_search):
            wrapper = _make_early_termination_wrapper(game, max_moves=4)
            with patch.object(ChessGame, "is_terminal", side_effect=wrapper):
                # 3. Play 2 games
                records = worker.generate_games(n_games=2, board_size=8)
                assert len(records) == 2

            # 4. Convert to experiences (plays 2 more games, need fresh wrapper)
            wrapper2 = _make_early_termination_wrapper(game, max_moves=4)
            with patch.object(ChessGame, "is_terminal", side_effect=wrapper2):
                experiences = worker.generate_experiences(n_games=2, board_size=8)
                assert len(experiences) > 0

        for exp in experiences:
            assert exp.board_state.shape == (119, 8, 8)
            assert exp.target_policy.shape == (4672,)
            assert -1.0 <= exp.target_value <= 1.0
            assert exp.board_size == 8

        # 5. Add to replay buffer
        buffer = UniformReplayBuffer(capacity=train_config.replay_buffer_size)
        for exp in experiences:
            buffer.add(exp)

        assert len(buffer) == len(experiences)

        # 6. Sample a batch
        batch_exps = buffer.sample(min(8, len(experiences)))
        assert len(batch_exps) <= 8

    def test_model_gradient_flow_chess(self, chess_config: tuple) -> None:
        """Verify gradients flow through the chess model end-to-end."""
        op_config, _, _ = chess_config
        model = AlphaGalerkinModel(op_config)
        model.train()

        # Forward pass with chess input
        x = torch.randn(2, 119, 8, 8)
        policy, value, lbb = model(x)

        assert policy.shape == (2, 4672)
        assert value.shape == (2, 1)

        # Backward pass
        loss = policy.sum() + value.sum()
        if lbb is not None:
            loss = loss + lbb
        loss.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.slow
    def test_self_play_game_terminates(self, chess_config: tuple) -> None:
        """Verify self-play games terminate (no infinite loops).

        MCTS search is mocked; ``is_terminal`` is patched to cap at 6 moves.
        """
        op_config, mcts_config, _ = chess_config
        game = ChessGame()
        model = AlphaGalerkinModel(op_config)
        model.eval()

        worker = SelfPlayWorker(
            model=model,
            mcts_config=mcts_config,
            device=torch.device("cpu"),
            game=game,
        )

        with patch("src.mcts.search.MCTS.search", _fast_mcts_search):
            wrapper = _make_early_termination_wrapper(game, max_moves=6)
            with patch.object(ChessGame, "is_terminal", side_effect=wrapper):
                records = worker.generate_games(n_games=1, board_size=8)

        assert len(records) == 1
        assert len(records[0].states) > 0
        assert len(records[0].policies) > 0
