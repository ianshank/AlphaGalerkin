"""Tests for chess self-play with generalized SelfPlayWorker.

Validates that the SelfPlayWorker can generate games and
experiences using the ChessGame interface via StatefulGameWrapper.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config.schemas import MCTSConfig, OperatorConfig
from src.games.chess import ACTION_SPACE_SIZE, ChessGame
from src.modeling.model import AlphaGalerkinModel
from src.training.self_play import GameRecord, SelfPlayWorker


@pytest.fixture
def chess_model() -> AlphaGalerkinModel:
    """Create small chess model for fast testing."""
    config = OperatorConfig(
        input_channels=119,
        action_space_size=ACTION_SPACE_SIZE,
        game_type="chess",
        d_model=32,
        d_key=8,
        d_value=8,
        d_ffn=64,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=16,
        use_fnet_mixing=False,
    )
    model = AlphaGalerkinModel(config)
    model.eval()
    return model


@pytest.fixture
def fast_mcts_config() -> MCTSConfig:
    """Create fast MCTS config for testing."""
    return MCTSConfig(
        n_simulations=4,  # Very few for speed
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
        temperature_drop_move=30,
    )


class TestChessSelfPlayWorker:
    """Tests for SelfPlayWorker with chess game."""

    def test_worker_initialization(
        self,
        chess_model: AlphaGalerkinModel,
        fast_mcts_config: MCTSConfig,
    ) -> None:
        """Test worker initializes with chess game."""
        game = ChessGame()
        worker = SelfPlayWorker(
            model=chess_model,
            mcts_config=fast_mcts_config,
            device="cpu",
            game=game,
        )
        assert worker.game is not None
        assert worker._games_played == 0

    def test_play_single_chess_game(
        self,
        chess_model: AlphaGalerkinModel,
        fast_mcts_config: MCTSConfig,
    ) -> None:
        """Test playing a single chess game (short, truncated)."""
        game = ChessGame()
        worker = SelfPlayWorker(
            model=chess_model,
            mcts_config=fast_mcts_config,
            device="cpu",
            game=game,
        )

        # Play with very short max_moves to keep test fast
        record = worker.play_game(max_moves=4)

        assert isinstance(record, GameRecord)
        assert len(record) > 0
        assert len(record.states) == len(record.policies)
        assert len(record.actions) <= 4

    def test_chess_policy_shape(
        self,
        chess_model: AlphaGalerkinModel,
        fast_mcts_config: MCTSConfig,
    ) -> None:
        """Test that chess policies have correct shape."""
        game = ChessGame()
        worker = SelfPlayWorker(
            model=chess_model,
            mcts_config=fast_mcts_config,
            device="cpu",
            game=game,
        )

        record = worker.play_game(max_moves=2)

        for policy in record.policies:
            assert policy.shape == (ACTION_SPACE_SIZE,)
            # Policy should be a valid distribution
            assert np.all(policy >= 0)
            # Sum might not be exactly 1 due to zero-padding, but non-zero entries should sum ≈ 1
            nonzero_sum = policy[policy > 0].sum()
            assert abs(nonzero_sum - 1.0) < 1e-4, f"Policy sum: {nonzero_sum}"

    def test_chess_state_shape(
        self,
        chess_model: AlphaGalerkinModel,
        fast_mcts_config: MCTSConfig,
    ) -> None:
        """Test that chess states have correct shape."""
        game = ChessGame()
        worker = SelfPlayWorker(
            model=chess_model,
            mcts_config=fast_mcts_config,
            device="cpu",
            game=game,
        )

        record = worker.play_game(max_moves=2)

        for state in record.states:
            assert state.shape == (119, 8, 8)


class TestChessGameRecordConversion:
    """Tests for converting chess game records to experiences."""

    def test_to_experiences(
        self,
        chess_model: AlphaGalerkinModel,
        fast_mcts_config: MCTSConfig,
    ) -> None:
        """Test converting chess game record to experiences."""
        game = ChessGame()
        worker = SelfPlayWorker(
            model=chess_model,
            mcts_config=fast_mcts_config,
            device="cpu",
            game=game,
        )

        record = worker.play_game(max_moves=4)
        experiences = record.to_experiences()

        assert len(experiences) == len(record.states)
        for exp in experiences:
            assert exp.board_state.shape == torch.Size([119, 8, 8])
            assert exp.target_policy.shape == torch.Size([ACTION_SPACE_SIZE])


class TestChessSelfPlayStats:
    """Tests for self-play statistics tracking."""

    def test_stats_after_game(
        self,
        chess_model: AlphaGalerkinModel,
        fast_mcts_config: MCTSConfig,
    ) -> None:
        """Test that stats are updated after playing a game."""
        game = ChessGame()
        worker = SelfPlayWorker(
            model=chess_model,
            mcts_config=fast_mcts_config,
            device="cpu",
            game=game,
        )

        worker.play_game(max_moves=2)
        stats = worker.get_stats()

        assert stats["games_played"] == 1
        assert stats["total_moves"] > 0


class TestGoSelfPlayBackwardsCompat:
    """Ensure Go self-play still works with game=None."""

    def test_go_self_play_unchanged(self) -> None:
        """Test that Go self-play path is unchanged when game=None."""
        go_config = OperatorConfig(
            d_model=32,
            d_key=8,
            d_value=8,
            d_ffn=64,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=16,
            use_fnet_mixing=False,
        )
        model = AlphaGalerkinModel(go_config)
        model.eval()

        mcts_config = MCTSConfig(n_simulations=2)
        worker = SelfPlayWorker(
            model=model,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
            game=None,  # Explicitly None = Go path
        )

        record = worker.play_game(max_moves=4, board_size=9)
        assert record.board_size == 9
        assert len(record) > 0
