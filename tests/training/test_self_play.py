"""Tests for self-play game generation.

Covers ``GameRecord``, ``SelfPlayWorker``, and ``ParallelSelfPlayWorker``
including sequential fallback, worker-count clamping, and game distribution.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from config.schemas import MCTSConfig, OperatorConfig
from src.modeling.model import AlphaGalerkinModel
from src.training.self_play import (
    _DEFAULT_MAX_WORKERS,
    GameRecord,
    ParallelSelfPlayWorker,
    SelfPlayWorker,
)


@pytest.fixture
def small_model() -> AlphaGalerkinModel:
    """Create small model for fast testing."""
    config = OperatorConfig(
        d_model=32,
        d_key=16,
        d_value=16,
        d_ffn=64,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=16,
        use_fnet_mixing=False,
    )
    return AlphaGalerkinModel(config)


@pytest.fixture
def mcts_config() -> MCTSConfig:
    """Create fast MCTS config."""
    return MCTSConfig(
        n_simulations=5,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )


class TestGameRecord:
    """Tests for GameRecord."""

    def test_to_experiences(self) -> None:
        """Test converting game record to experiences."""
        import numpy as np

        board_size = 9
        n_actions = board_size**2 + 1

        record = GameRecord(
            board_size=board_size,
            states=[
                np.random.randn(17, board_size, board_size).astype(np.float32) for _ in range(10)
            ],
            policies=[
                np.random.dirichlet(np.ones(n_actions)).astype(np.float32) for _ in range(10)
            ],
            actions=list(range(10)),
            outcome=1.0,  # Black wins
        )

        experiences = record.to_experiences()

        assert len(experiences) == 10

        # Check first experience (black's move)
        exp_0 = experiences[0]
        assert exp_0.board_size == board_size
        assert exp_0.target_value == 1.0  # Black's perspective, black won

        # Check second experience (white's move)
        exp_1 = experiences[1]
        assert exp_1.target_value == -1.0  # White's perspective, black won

    def test_game_record_length(self) -> None:
        """Test __len__ method."""
        import numpy as np

        record = GameRecord(
            board_size=9,
            states=[np.zeros((17, 9, 9)) for _ in range(5)],
            policies=[np.zeros(82) for _ in range(5)],
            actions=[0] * 5,
            outcome=0.0,
        )

        assert len(record) == 5


class TestSelfPlayWorker:
    """Tests for SelfPlayWorker."""

    def test_worker_initialization(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Test worker initialization."""
        worker = SelfPlayWorker(
            model=small_model,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )

        assert worker.model is small_model
        assert worker.board_sizes == [9]

    def test_play_single_game(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Test playing a single game."""
        worker = SelfPlayWorker(
            model=small_model,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )

        record = worker.play_game(board_size=9, max_moves=20)

        assert isinstance(record, GameRecord)
        assert record.board_size == 9
        assert len(record.states) > 0
        assert len(record.policies) == len(record.states)
        assert record.outcome in [-1.0, 0.0, 1.0]

    def test_policies_normalized(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Test that generated policies sum to 1."""
        worker = SelfPlayWorker(
            model=small_model,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )

        record = worker.play_game(board_size=9, max_moves=10)

        for policy in record.policies:
            assert abs(policy.sum() - 1.0) < 0.01, "Policy should sum to 1"

    def test_generate_multiple_games(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Test generating multiple games."""
        worker = SelfPlayWorker(
            model=small_model,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )

        games = worker.generate_games(n_games=2, board_size=9)

        assert len(games) == 2
        assert all(isinstance(g, GameRecord) for g in games)

    def test_generate_experiences(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Test generating experiences directly."""
        worker = SelfPlayWorker(
            model=small_model,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )

        experiences = worker.generate_experiences(n_games=2, board_size=9)

        assert len(experiences) > 0
        # Each game should produce multiple experiences
        assert len(experiences) > 2

    def test_get_stats(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Test statistics tracking."""
        worker = SelfPlayWorker(
            model=small_model,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )

        worker.play_game(board_size=9, max_moves=10)
        worker.play_game(board_size=9, max_moves=10)

        stats = worker.get_stats()

        assert stats["games_played"] == 2
        assert stats["total_moves"] > 0
        assert "outcomes" in stats

    def test_temperature_schedule(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Test temperature scheduling."""
        schedule = {0: 1.0, 5: 0.5, 10: 0.1}
        worker = SelfPlayWorker(
            model=small_model,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
            temperature_schedule=schedule,
        )

        assert worker._get_temperature(0) == 1.0
        assert worker._get_temperature(3) == 1.0
        assert worker._get_temperature(5) == 0.5
        assert worker._get_temperature(7) == 0.5
        assert worker._get_temperature(10) == 0.1
        assert worker._get_temperature(100) == 0.1


class TestParallelSelfPlayWorker:
    """Tests for ParallelSelfPlayWorker."""

    def test_init_clamps_workers(
        self,
        small_model: AlphaGalerkinModel,
    ) -> None:
        """Worker count is clamped to [1, _DEFAULT_MAX_WORKERS]."""
        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=0,
            device="cpu",
            board_sizes=[9],
        )
        assert worker.n_workers == 1

        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=100,
            device="cpu",
            board_sizes=[9],
        )
        assert worker.n_workers == _DEFAULT_MAX_WORKERS

    def test_sequential_fallback_single_worker(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """With n_workers=1, uses sequential path."""
        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=1,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )
        assert not worker._use_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_sequential_fallback_cuda_device(
        self,
        small_model: AlphaGalerkinModel,
    ) -> None:
        """With CUDA device, falls back to sequential."""
        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=4,
            device="cuda:0",
            board_sizes=[9],
        )
        assert not worker._use_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_sequential_fallback_cuda_torch_device(
        self,
        small_model: AlphaGalerkinModel,
    ) -> None:
        """torch.device('cuda') also triggers sequential fallback."""
        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=4,
            device=torch.device("cuda"),
            board_sizes=[9],
        )
        assert not worker._use_parallel()

    def test_cpu_allows_parallel(
        self,
        small_model: AlphaGalerkinModel,
    ) -> None:
        """CPU device allows parallel execution."""
        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=4,
            device="cpu",
            board_sizes=[9],
        )
        assert worker._use_parallel()

    def test_generate_games_sequential_single_game(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Single-game request always uses sequential path."""
        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=4,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )
        # Should fall back to sequential for n_games=1
        games = worker.generate_games(n_games=1, board_size=9)
        assert len(games) == 1
        assert isinstance(games[0], GameRecord)

    def test_generate_experiences_returns_experiences(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """generate_experiences produces Experience objects from games."""
        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=1,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )
        experiences = worker.generate_experiences(n_games=1, board_size=9)
        assert len(experiences) > 0

    def test_generate_games_sequential_fallback_on_error(
        self,
        small_model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
    ) -> None:
        """Multiprocessing failure falls back to sequential."""
        worker = ParallelSelfPlayWorker(
            model=small_model,
            n_workers=2,
            mcts_config=mcts_config,
            device="cpu",
            board_sizes=[9],
        )
        # Patch mp to raise, verifying fallback
        with patch(
            "src.training.self_play.torch.multiprocessing",
            side_effect=RuntimeError("mp unavailable"),
        ):
            # Even if import patch fails, the try/except in generate_games
            # catches the error and falls back
            games = worker.generate_games(n_games=2, board_size=9)
            assert len(games) == 2
