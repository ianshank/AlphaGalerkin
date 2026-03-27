"""Tests for distributed self-play worker coordination."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.distributed.config import SelfPlayDistributedConfig
from src.distributed.worker import (
    CoordinatorState,
    SelfPlayCoordinator,
    SelfPlayWorker,
    WorkerStats,
)

# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42


@dataclass
class FakeExperience:
    """Minimal experience stub for testing."""

    board_size: int
    policy: list[float]
    value: float


class FakeConfig:
    """Minimal config stub matching AlphaGalerkinModel.config interface."""

    def __init__(self, width: int = 8) -> None:
        self.width = width


class FakeModel(nn.Module):
    """Lightweight model that mimics AlphaGalerkinModel's interface.

    Avoids importing the real model which pulls in heavy dependencies.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__()
        self.config = config or FakeConfig()
        self.linear = nn.Linear(self.config.width, self.config.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FakeMCTSConfig:
    """Minimal MCTS config stub."""

    def __init__(self, n_simulations: int = 8) -> None:
        self.n_simulations = n_simulations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seed() -> int:
    return DEFAULT_SEED


@pytest.fixture(autouse=True)
def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture
def fake_model() -> FakeModel:
    """Create a lightweight fake model."""
    return FakeModel()


@pytest.fixture
def mcts_config() -> FakeMCTSConfig:
    return FakeMCTSConfig()


@pytest.fixture
def selfplay_config() -> SelfPlayDistributedConfig:
    return SelfPlayDistributedConfig(
        workers_per_node=2,
        games_per_worker=5,
        experience_sharing="local",
        cpu_workers=True,
    )


@pytest.fixture
def board_sizes() -> list[int]:
    return [9, 13]


def _make_fake_experiences(n: int, board_size: int = 9) -> list[FakeExperience]:
    """Create n fake experience objects."""
    return [
        FakeExperience(
            board_size=board_size,
            policy=[1.0 / board_size**2] * board_size**2,
            value=0.5,
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# WorkerStats / CoordinatorState dataclass tests
# ---------------------------------------------------------------------------


class TestWorkerStats:
    """Tests for the WorkerStats dataclass."""

    def test_default_values(self) -> None:
        stats = WorkerStats(worker_id=0)
        assert stats.games_completed == 0
        assert stats.experiences_generated == 0
        assert stats.model_version == 0

    @pytest.mark.parametrize("worker_id", [0, 1, 7])
    def test_worker_id_stored(self, worker_id: int) -> None:
        stats = WorkerStats(worker_id=worker_id)
        assert stats.worker_id == worker_id


class TestCoordinatorState:
    """Tests for the CoordinatorState dataclass."""

    def test_default_values(self) -> None:
        state = CoordinatorState()
        assert state.total_games == 0
        assert state.total_experiences == 0
        assert state.workers_active == 0


# ---------------------------------------------------------------------------
# SelfPlayWorker
# ---------------------------------------------------------------------------


class TestSelfPlayWorkerInit:
    """Tests for SelfPlayWorker initialization."""

    def test_init_stores_attributes(
        self,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
    ) -> None:
        worker = SelfPlayWorker(
            worker_id=0,
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            device="cpu",
        )
        assert worker.worker_id == 0
        assert worker.device == torch.device("cpu")
        assert worker.model_version == 0

    @pytest.mark.parametrize("device_str", ["cpu"])
    def test_device_from_string(
        self,
        device_str: str,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
    ) -> None:
        worker = SelfPlayWorker(
            worker_id=0,
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            device=device_str,
        )
        assert worker.device == torch.device(device_str)


class TestSelfPlayWorkerGenerateBatch:
    """Tests for generate_batch with mocked self-play."""

    @patch("src.training.self_play.SelfPlayWorker")
    def test_generate_batch_returns_experiences(
        self,
        mock_spw_cls: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """generate_batch collects experiences from the inner SelfPlayWorker."""
        # Set up mock: each call to generate_experiences returns 2 experiences
        fake_exp = _make_fake_experiences(2)
        mock_instance = MagicMock()
        mock_instance.generate_experiences.return_value = fake_exp
        mock_spw_cls.return_value = mock_instance

        worker = SelfPlayWorker(
            worker_id=0,
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            device="cpu",
        )

        n_games = 3
        experiences = worker.generate_batch(n_games=n_games, board_sizes=board_sizes)

        assert len(experiences) == n_games * len(fake_exp)
        assert mock_instance.generate_experiences.call_count == n_games

    @patch("src.training.self_play.SelfPlayWorker")
    def test_generate_batch_updates_stats(
        self,
        mock_spw_cls: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """Stats are updated after batch generation."""
        mock_instance = MagicMock()
        mock_instance.generate_experiences.return_value = _make_fake_experiences(1)
        mock_spw_cls.return_value = mock_instance

        worker = SelfPlayWorker(
            worker_id=0,
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            device="cpu",
        )

        n_games = 4
        worker.generate_batch(n_games=n_games, board_sizes=board_sizes)
        stats = worker.get_stats()

        assert stats.games_completed == n_games
        assert stats.experiences_generated == n_games  # 1 exp per game

    @patch("src.training.self_play.SelfPlayWorker")
    def test_stop_interrupts_batch(
        self,
        mock_spw_cls: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """Setting stop event mid-batch causes early termination."""
        call_count = 0

        def _slow_generate(n: int) -> list[FakeExperience]:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                worker.stop()
            return _make_fake_experiences(1)

        mock_instance = MagicMock()
        mock_instance.generate_experiences.side_effect = _slow_generate
        mock_spw_cls.return_value = mock_instance

        worker = SelfPlayWorker(
            worker_id=0,
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            device="cpu",
        )

        experiences = worker.generate_batch(n_games=100, board_sizes=board_sizes)
        # Should have stopped early
        assert len(experiences) < 100


class TestSelfPlayWorkerUpdateModel:
    """Tests for model update on worker."""

    def test_update_model_changes_version(
        self,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
    ) -> None:
        worker = SelfPlayWorker(
            worker_id=0,
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            device="cpu",
        )
        new_state = fake_model.state_dict()
        worker.update_model(new_state, version=5)

        assert worker.model_version == 5
        assert worker.get_stats().model_version == 5


# ---------------------------------------------------------------------------
# SelfPlayCoordinator
# ---------------------------------------------------------------------------


class TestSelfPlayCoordinatorInit:
    """Tests for coordinator initialization."""

    @patch("src.distributed.worker.from_environment", return_value=(0, 0, 1))
    def test_creates_workers(
        self,
        _mock_env: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """Coordinator initializes the configured number of workers."""
        coord = SelfPlayCoordinator(
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            board_sizes=board_sizes,
        )

        assert len(coord.workers) == selfplay_config.workers_per_node

    @patch("src.distributed.worker.from_environment", return_value=(0, 0, 1))
    def test_workers_have_unique_ids(
        self,
        _mock_env: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """Each worker has a unique ID."""
        coord = SelfPlayCoordinator(
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            board_sizes=board_sizes,
        )
        ids = [w.worker_id for w in coord.workers]
        assert len(set(ids)) == len(ids)


class TestSelfPlayCoordinatorExperiences:
    """Tests for experience generation and synchronization."""

    @patch("src.distributed.worker.from_environment", return_value=(0, 0, 1))
    @patch("src.training.self_play.SelfPlayWorker")
    def test_generate_experiences_distributes_games(
        self,
        mock_spw_cls: MagicMock,
        _mock_env: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """generate_experiences distributes games across workers."""
        mock_instance = MagicMock()
        mock_instance.generate_experiences.return_value = _make_fake_experiences(1)
        mock_spw_cls.return_value = mock_instance

        coord = SelfPlayCoordinator(
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            board_sizes=board_sizes,
        )

        total_games = 6
        experiences = coord.generate_experiences(total_games=total_games)

        assert len(experiences) >= total_games  # At least 1 exp per game
        state = coord.get_state()
        assert state.total_games == total_games

    @patch("src.distributed.worker.from_environment", return_value=(0, 0, 1))
    @patch("src.training.self_play.SelfPlayWorker")
    def test_synchronize_local_returns_local_experiences(
        self,
        mock_spw_cls: MagicMock,
        _mock_env: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        board_sizes: list[int],
    ) -> None:
        """With 'local' sharing, synchronize returns local experiences only."""
        config = SelfPlayDistributedConfig(
            workers_per_node=1,
            experience_sharing="local",
            cpu_workers=True,
        )
        mock_instance = MagicMock()
        mock_instance.generate_experiences.return_value = _make_fake_experiences(2)
        mock_spw_cls.return_value = mock_instance

        coord = SelfPlayCoordinator(
            model=fake_model,
            mcts_config=mcts_config,
            config=config,
            board_sizes=board_sizes,
        )
        coord.generate_experiences(total_games=2)
        synced = coord.synchronize_experiences()

        assert len(synced) > 0

    @patch("src.distributed.worker.from_environment", return_value=(0, 0, 1))
    @patch("src.training.self_play.SelfPlayWorker")
    def test_clear_local_experiences(
        self,
        mock_spw_cls: MagicMock,
        _mock_env: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """clear_local_experiences empties the buffer."""
        mock_instance = MagicMock()
        mock_instance.generate_experiences.return_value = _make_fake_experiences(1)
        mock_spw_cls.return_value = mock_instance

        coord = SelfPlayCoordinator(
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            board_sizes=board_sizes,
        )
        coord.generate_experiences(total_games=2)
        coord.clear_local_experiences()

        assert coord.get_state().buffer_size == 0


class TestSelfPlayCoordinatorBroadcast:
    """Tests for model broadcast to workers."""

    @patch("src.distributed.worker.from_environment", return_value=(0, 0, 1))
    def test_broadcast_updates_all_workers(
        self,
        _mock_env: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """broadcast_model propagates state_dict and version to all workers."""
        coord = SelfPlayCoordinator(
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            board_sizes=board_sizes,
        )

        new_state = fake_model.state_dict()
        coord.broadcast_model(new_state, version=42)

        for worker in coord.workers:
            assert worker.model_version == 42


class TestSelfPlayCoordinatorShutdown:
    """Tests for coordinator shutdown."""

    @patch("src.distributed.worker.from_environment", return_value=(0, 0, 1))
    def test_shutdown_stops_workers(
        self,
        _mock_env: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """Shutdown sets the stop event on all workers."""
        coord = SelfPlayCoordinator(
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            board_sizes=board_sizes,
        )
        coord.shutdown()

        for worker in coord.workers:
            assert worker._should_stop.is_set()

    @patch("src.distributed.worker.from_environment", return_value=(0, 0, 1))
    def test_get_worker_stats(
        self,
        _mock_env: MagicMock,
        fake_model: FakeModel,
        mcts_config: FakeMCTSConfig,
        selfplay_config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """get_worker_stats returns one entry per worker."""
        coord = SelfPlayCoordinator(
            model=fake_model,
            mcts_config=mcts_config,
            config=selfplay_config,
            board_sizes=board_sizes,
        )
        stats = coord.get_worker_stats()
        assert len(stats) == selfplay_config.workers_per_node
        for s in stats:
            assert isinstance(s, WorkerStats)
