"""Tests for ``src.distributed.worker`` (``SelfPlayWorker`` / ``SelfPlayCoordinator``).

NAMING TRAP: ``src.distributed.worker.SelfPlayWorker`` is a *different, unrelated*
class from ``src.training.self_play.SelfPlayWorker`` (already covered by
``tests/training/test_self_play.py``). Every import in this file targets
``src.distributed.worker`` explicitly -- see the coverage-gate section of the
SQE report for this wave for a run confirming this file actually exercises
``src/distributed/worker.py`` and not the training module of the same class name.

``SelfPlayWorker.generate_batch`` (in ``src/distributed/worker.py``) does
``from src.training.self_play import SelfPlayWorker as SPW`` *inside its own
method body* on every call, then drives real MCTS self-play through ``SPW``.
That inner pipeline is already covered by ``tests/training/test_self_play.py``
and is out of scope here (do not touch it). To unit-test
``src/distributed/worker.py``'s own logic (thread fan-out across workers,
per-worker stats bookkeeping, experience aggregation) in isolation, tests
below monkeypatch the ``src.training.self_play.SelfPlayWorker`` module
attribute with a lightweight double before calling into
``generate_batch``/``generate_experiences``. Because the import is
late-binding (inside the method, not at module load time), patching the
attribute before the call is picked up correctly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.distributed.config import SelfPlayDistributedConfig
from src.distributed.worker import (
    CoordinatorState,
    SelfPlayCoordinator,
    SelfPlayWorker,
    WorkerStats,
    create_self_play_coordinator,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _TinyConfig:
    """Placeholder "model config" object; identity is all that matters here."""


class _TinyModel(torch.nn.Module):
    """Minimal ``nn.Module`` double satisfying the coordinator's clone-on-init contract.

    ``SelfPlayCoordinator._initialize_workers`` calls
    ``type(self.model)(self.model.config)`` to build one fresh model instance
    per worker, so the double must accept a single positional ``config``
    argument and expose it back as ``self.config`` -- mirroring
    ``AlphaGalerkinModel``'s ``(config)`` constructor shape without pulling in
    the real (heavier) network.
    """

    def __init__(self, config: object | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else _TinyConfig()
        self.linear = torch.nn.Linear(2, 2)


@pytest.fixture(autouse=True)
def _clear_distributed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force single-process rank/world-size defaults regardless of ambient env.

    ``SelfPlayCoordinator.__init__`` reads ``RANK``/``LOCAL_RANK``/``WORLD_SIZE``
    via ``_get_env_rank_info()``; clearing them guarantees the
    ``(rank, local_rank, world_size) == (0, 0, 1)`` single-process defaults
    this whole file's assumptions rely on.
    """
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


@pytest.fixture
def tiny_model() -> _TinyModel:
    """A tiny real ``nn.Module`` standing in for ``AlphaGalerkinModel``."""
    return _TinyModel(_TinyConfig())


@pytest.fixture
def mcts_config_stub() -> SimpleNamespace:
    """A stand-in for ``config.schemas.MCTSConfig``.

    Nothing in ``src/distributed/worker.py`` itself introspects this object
    (it is only stored and forwarded); the real dataclass is never imported
    at runtime (``TYPE_CHECKING``-only in the module under test).
    """
    return SimpleNamespace(n_simulations=4, c_puct=1.0)


@pytest.fixture
def make_coordinator(tiny_model: _TinyModel, mcts_config_stub: SimpleNamespace):
    """Factory fixture building a ``SelfPlayCoordinator`` with sane fast defaults."""

    def _make(**config_kwargs: object) -> SelfPlayCoordinator:
        merged = {"num_workers": 1, **config_kwargs}
        config = SelfPlayDistributedConfig(**merged)
        return SelfPlayCoordinator(
            model=tiny_model,
            mcts_config=mcts_config_stub,
            config=config,
            board_sizes=[9],
        )

    return _make


@pytest.fixture
def fake_inner_spw(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a lightweight double for ``src.training.self_play.SelfPlayWorker``.

    Returns the double's class object so tests can inspect
    ``call_log`` (the sequence of ``n_games`` arguments passed to
    ``generate_experiences``).
    """

    class _FakeSPW:
        call_log: ClassVar[list[int]] = []

        def __init__(
            self, model: object, mcts_config: object, device: object, board_sizes: list[int]
        ) -> None:
            self.model = model
            self.mcts_config = mcts_config
            self.device = device
            self.board_sizes = board_sizes

        def generate_experiences(self, n_games: int) -> list[dict]:
            _FakeSPW.call_log.append(n_games)
            return [{"board_size": self.board_sizes[0]} for _ in range(n_games)]

    monkeypatch.setattr("src.training.self_play.SelfPlayWorker", _FakeSPW)
    return _FakeSPW


# ---------------------------------------------------------------------------
# SelfPlayWorker construction
# ---------------------------------------------------------------------------


class TestSelfPlayWorkerConstruction:
    """Construction of ``SelfPlayWorker`` with a small mock model."""

    def test_construction_sets_basic_attributes(
        self, tiny_model: _TinyModel, mcts_config_stub: SimpleNamespace
    ) -> None:
        """Constructor stores every argument and initializes zeroed stats."""
        config = SelfPlayDistributedConfig(num_workers=1)

        worker = SelfPlayWorker(
            worker_id=3,
            model=tiny_model,
            mcts_config=mcts_config_stub,
            config=config,
            device="cpu",
        )

        assert worker.worker_id == 3
        assert worker.model is tiny_model
        assert worker.mcts_config is mcts_config_stub
        assert worker.config is config
        assert worker.device == torch.device("cpu")
        assert worker.model_version == 0

        stats = worker.get_stats()
        assert isinstance(stats, WorkerStats)
        assert stats.worker_id == 3
        assert stats.games_completed == 0
        assert stats.experiences_generated == 0

    def test_construction_accepts_torch_device_object(
        self, tiny_model: _TinyModel, mcts_config_stub: SimpleNamespace
    ) -> None:
        """A ``torch.device`` (not just a string) is accepted directly."""
        config = SelfPlayDistributedConfig(num_workers=1)

        worker = SelfPlayWorker(
            worker_id=0,
            model=tiny_model,
            mcts_config=mcts_config_stub,
            config=config,
            device=torch.device("cpu"),
        )

        assert worker.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# SelfPlayWorker.generate_batch
# ---------------------------------------------------------------------------


class TestSelfPlayWorkerGenerateBatch:
    """Tests for ``SelfPlayWorker.generate_batch`` with the inner SPW mocked out."""

    def test_generate_batch_returns_experiences_and_updates_stats(
        self,
        tiny_model: _TinyModel,
        mcts_config_stub: SimpleNamespace,
        fake_inner_spw: type,
    ) -> None:
        """One ``generate_experiences(1)`` call per requested game, stats updated."""
        config = SelfPlayDistributedConfig(num_workers=1)
        worker = SelfPlayWorker(0, tiny_model, mcts_config_stub, config, device="cpu")

        experiences = worker.generate_batch(n_games=3, board_sizes=[9, 13])

        assert len(experiences) == 3
        assert fake_inner_spw.call_log == [1, 1, 1]
        stats = worker.get_stats()
        assert stats.games_completed == 3
        assert stats.experiences_generated == 3
        assert stats.average_time_per_game_ms >= 0.0

    def test_generate_batch_zero_games_leaves_average_time_at_zero(
        self,
        tiny_model: _TinyModel,
        mcts_config_stub: SimpleNamespace,
        fake_inner_spw: type,
    ) -> None:
        """``n_games=0`` exercises the ``else 0`` branch of the average-time expression."""
        config = SelfPlayDistributedConfig(num_workers=1)
        worker = SelfPlayWorker(0, tiny_model, mcts_config_stub, config, device="cpu")

        experiences = worker.generate_batch(n_games=0, board_sizes=[9])

        assert experiences == []
        assert fake_inner_spw.call_log == []
        stats = worker.get_stats()
        assert stats.average_time_per_game_ms == 0.0
        assert stats.games_completed == 0

    def test_stop_before_generate_batch_records_zero_games_completed(
        self,
        tiny_model: _TinyModel,
        mcts_config_stub: SimpleNamespace,
        fake_inner_spw: type,
    ) -> None:
        """``games_completed`` counts games that ran, not games that were requested.

        ``generate_batch``'s loop breaks out on the ``_should_stop`` check
        before doing any work, so a worker stopped ahead of the call produces
        no experiences *and* must record no completed games. Both stats are
        derived from what the loop actually did, so they agree: a request for
        five games that runs none leaves the counter at zero.
        """
        config = SelfPlayDistributedConfig(num_workers=1)
        worker = SelfPlayWorker(0, tiny_model, mcts_config_stub, config, device="cpu")
        worker.stop()

        experiences = worker.generate_batch(n_games=5, board_sizes=[9])

        assert experiences == []
        assert fake_inner_spw.call_log == []  # loop body never ran
        stats = worker.get_stats()
        assert stats.experiences_generated == 0
        assert stats.games_completed == 0  # not the 5 requested

    def test_stop_midway_counts_only_the_games_that_actually_ran(
        self,
        tiny_model: _TinyModel,
        mcts_config_stub: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A partially-completed batch reports its iteration count.

        This is the case that separates "count loop iterations" from the
        weaker "report zero when stopped, else the full request": the worker
        is signalled to stop from *inside* its second game, so exactly two of
        the five requested games run to completion.
        """
        config = SelfPlayDistributedConfig(num_workers=1)
        worker = SelfPlayWorker(0, tiny_model, mcts_config_stub, config, device="cpu")

        class _StopAfterSecondGame:
            """Inner-SPW double that trips the worker's stop event mid-batch."""

            n_calls: ClassVar[int] = 0

            def __init__(
                self, model: object, mcts_config: object, device: object, board_sizes: list[int]
            ) -> None:
                self.board_sizes = board_sizes

            def generate_experiences(self, n_games: int) -> list[dict]:
                _StopAfterSecondGame.n_calls += 1
                if _StopAfterSecondGame.n_calls == 2:
                    worker.stop()
                return [{"board_size": self.board_sizes[0]} for _ in range(n_games)]

        monkeypatch.setattr("src.training.self_play.SelfPlayWorker", _StopAfterSecondGame)

        experiences = worker.generate_batch(n_games=5, board_sizes=[9])

        # Games 1 and 2 run (the stop event is only checked at the top of the
        # loop); game 3 hits the break.
        assert _StopAfterSecondGame.n_calls == 2
        assert len(experiences) == 2
        stats = worker.get_stats()
        assert stats.games_completed == 2  # not the 5 requested
        assert stats.experiences_generated == 2
        assert stats.average_time_per_game_ms > 0.0


# ---------------------------------------------------------------------------
# SelfPlayWorker.update_model / stop
# ---------------------------------------------------------------------------


class TestSelfPlayWorkerUpdateModelAndStop:
    """Tests for the remaining two ``SelfPlayWorker`` public methods."""

    def test_update_model_updates_state_dict_and_version(
        self, tiny_model: _TinyModel, mcts_config_stub: SimpleNamespace
    ) -> None:
        """``update_model`` loads the new state dict and bumps the version."""
        config = SelfPlayDistributedConfig(num_workers=1)
        worker = SelfPlayWorker(0, tiny_model, mcts_config_stub, config, device="cpu")
        new_state = {k: v.clone() for k, v in tiny_model.state_dict().items()}

        worker.update_model(new_state, version=9)

        assert worker.model_version == 9
        assert worker.get_stats().model_version == 9

    def test_stop_sets_should_stop_event(
        self, tiny_model: _TinyModel, mcts_config_stub: SimpleNamespace
    ) -> None:
        """``stop()`` sets the internal stop-event flag."""
        config = SelfPlayDistributedConfig(num_workers=1)
        worker = SelfPlayWorker(0, tiny_model, mcts_config_stub, config, device="cpu")

        assert not worker._should_stop.is_set()
        worker.stop()
        assert worker._should_stop.is_set()


# ---------------------------------------------------------------------------
# SelfPlayCoordinator construction
# ---------------------------------------------------------------------------


class TestSelfPlayCoordinatorConstruction:
    """Construction of ``SelfPlayCoordinator`` (single-process, ``cpu_workers=True``)."""

    def test_single_process_defaults_rank_zero_world_size_one(self, make_coordinator) -> None:
        """With no distributed env vars set, rank/local_rank/world_size default to (0, 0, 1)."""
        coordinator = make_coordinator(num_workers=1)

        assert coordinator.rank == 0
        assert coordinator.local_rank == 0
        assert coordinator.world_size == 1

    def test_creates_num_workers_worker_instances_with_sequential_ids(
        self, make_coordinator
    ) -> None:
        """``_initialize_workers`` builds exactly ``num_workers`` workers, IDs 0..N-1 at rank 0."""
        coordinator = make_coordinator(num_workers=3)

        assert len(coordinator.workers) == 3
        assert [w.worker_id for w in coordinator.workers] == [0, 1, 2]
        assert all(isinstance(w, SelfPlayWorker) for w in coordinator.workers)

    def test_cpu_workers_true_places_all_workers_on_cpu(self, make_coordinator) -> None:
        """``cpu_workers=True`` (the default) places every worker's device on CPU."""
        coordinator = make_coordinator(num_workers=2, cpu_workers=True)

        assert all(w.device == torch.device("cpu") for w in coordinator.workers)

    def test_clones_a_distinct_model_instance_per_worker(self, make_coordinator) -> None:
        """Each worker gets its own model clone, not a shared reference."""
        coordinator = make_coordinator(num_workers=2)

        assert coordinator.workers[0].model is not coordinator.model
        assert coordinator.workers[0].model is not coordinator.workers[1].model

    def test_initial_coordinator_state_is_empty(self, make_coordinator) -> None:
        """A freshly constructed coordinator reports zeroed state."""
        coordinator = make_coordinator(num_workers=1)

        state = coordinator.get_state()
        assert state.total_games == 0
        assert state.total_experiences == 0
        assert state.buffer_size == 0
        assert state.workers_active == 1


@pytest.mark.gpu_required
class TestSelfPlayCoordinatorGpuDeviceSelection:
    """``cpu_workers=False`` selects a CUDA device keyed off ``local_rank``."""

    def test_non_cpu_workers_uses_local_rank_cuda_device(self, make_coordinator) -> None:
        """Auto-skips on CPU CI via the root conftest.py ``gpu_required`` hook."""
        coordinator = make_coordinator(num_workers=1, cpu_workers=False)

        assert coordinator.workers[0].device == torch.device("cuda:0")


# ---------------------------------------------------------------------------
# SelfPlayCoordinator.generate_experiences
# ---------------------------------------------------------------------------


class TestSelfPlayCoordinatorGenerateExperiences:
    """``generate_experiences`` in single-process mode (inner SPW mocked out)."""

    def test_generate_experiences_single_worker(
        self, make_coordinator, fake_inner_spw: type
    ) -> None:
        """All requested games run on the lone worker; state/buffer updated."""
        coordinator = make_coordinator(num_workers=1)

        experiences = coordinator.generate_experiences(total_games=4)

        assert len(experiences) == 4
        assert fake_inner_spw.call_log.count(1) == 4
        assert coordinator._local_experiences == experiences

        state = coordinator.get_state()
        assert state.total_games == 4
        assert state.total_experiences == 4

    def test_generate_experiences_distributes_remainder_across_workers(
        self, make_coordinator, fake_inner_spw: type
    ) -> None:
        """7 games / 3 workers -> games_per_worker=2, remainder=1 -> [3, 2, 2]."""
        coordinator = make_coordinator(num_workers=3)

        experiences = coordinator.generate_experiences(total_games=7)

        assert len(experiences) == 7
        per_worker = sorted(w.get_stats().games_completed for w in coordinator.workers)
        assert per_worker == [2, 2, 3]

    def test_stopped_workers_do_not_inflate_total_games(
        self, make_coordinator, fake_inner_spw: type
    ) -> None:
        """``CoordinatorState.total_games`` counts completions, not the request.

        The coordinator accumulates ``total_games`` on its own line rather
        than reading it back off the workers, so it needs its own guard: with
        every worker already shut down, each batch loop breaks immediately and
        the requested six games must not be booked as run.
        """
        coordinator = make_coordinator(num_workers=2)
        coordinator.shutdown()

        experiences = coordinator.generate_experiences(total_games=6)

        assert experiences == []
        assert fake_inner_spw.call_log == []
        assert all(w.get_stats().games_completed == 0 for w in coordinator.workers)
        state = coordinator.get_state()
        assert state.total_games == 0  # not the 6 requested
        assert state.total_experiences == 0

    def test_total_games_sums_actual_per_worker_completions(
        self, make_coordinator, fake_inner_spw: type
    ) -> None:
        """With one of two workers stopped, only the live worker's games count.

        Pins the *summation* semantics rather than an all-or-nothing rule: 6
        games split 3/3, one worker stopped, so the coordinator must report 3
        -- matching what ``get_worker_stats()`` shows for the same run.
        """
        coordinator = make_coordinator(num_workers=2)
        coordinator.workers[0].stop()

        experiences = coordinator.generate_experiences(total_games=6)

        assert coordinator.workers[0].get_stats().games_completed == 0
        assert coordinator.workers[1].get_stats().games_completed == 3
        assert len(experiences) == 3

        state = coordinator.get_state()
        assert state.total_games == 3  # not the 6 requested
        assert state.total_games == sum(s.games_completed for s in coordinator.get_worker_stats())


# ---------------------------------------------------------------------------
# SelfPlayCoordinator.synchronize_experiences -- guarded early return
# ---------------------------------------------------------------------------


class TestSynchronizeExperiencesGuardedReturn:
    """The ``dist.is_initialized() == False`` guarded early-return path."""

    def test_local_sharing_returns_local_experiences_without_checking_dist(
        self, make_coordinator
    ) -> None:
        """``experience_sharing="local"`` short-circuits before touching ``dist`` at all."""
        coordinator = make_coordinator(num_workers=1, experience_sharing="local")
        coordinator._local_experiences.extend([{"x": 1}])

        with patch("torch.distributed.is_initialized") as mock_is_init:
            result = coordinator.synchronize_experiences()

        mock_is_init.assert_not_called()
        assert result == coordinator._local_experiences

    def test_global_sharing_guarded_early_return_when_dist_not_initialized(
        self, make_coordinator
    ) -> None:
        """``experience_sharing="global"`` + no process group returns local experiences."""
        coordinator = make_coordinator(num_workers=1, experience_sharing="global")
        coordinator._local_experiences.extend([{"x": 1}, {"x": 2}])

        with patch("torch.distributed.is_initialized", return_value=False) as mock_is_init:
            result = coordinator.synchronize_experiences()

        mock_is_init.assert_called_once()
        assert result is coordinator._local_experiences

    def test_hierarchical_sharing_guarded_early_return_when_dist_not_initialized(
        self, make_coordinator
    ) -> None:
        """Same guard applies to ``experience_sharing="hierarchical"``."""
        coordinator = make_coordinator(num_workers=1, experience_sharing="hierarchical")
        coordinator._local_experiences.extend([{"x": 1}])

        with patch("torch.distributed.is_initialized", return_value=False):
            result = coordinator.synchronize_experiences()

        assert result is coordinator._local_experiences


# ---------------------------------------------------------------------------
# _serialize_experiences / _deserialize_experiences
# ---------------------------------------------------------------------------


class TestSerializeDeserializeExperiences:
    """Plain ``pickle.dumps``/``pickle.loads`` round trips -- no distributed dependency."""

    def test_round_trip_preserves_data(self, make_coordinator) -> None:
        """A round trip through serialize -> deserialize reproduces the input list."""
        coordinator = make_coordinator(num_workers=1)
        experiences = [{"board_size": 9, "value": 1.0}, {"board_size": 13, "value": -0.5}]

        data = coordinator._serialize_experiences(experiences)
        restored = coordinator._deserialize_experiences(data)

        assert isinstance(data, bytes)
        assert restored == experiences

    def test_round_trip_empty_list(self, make_coordinator) -> None:
        """An empty experience list round-trips to an empty list."""
        coordinator = make_coordinator(num_workers=1)

        data = coordinator._serialize_experiences([])
        restored = coordinator._deserialize_experiences(data)

        assert restored == []


# ---------------------------------------------------------------------------
# Remaining SelfPlayCoordinator public methods
# ---------------------------------------------------------------------------


class TestSelfPlayCoordinatorModelAndStateManagement:
    """Tests for ``broadcast_model``, ``clear_local_experiences``, ``get_state``, etc."""

    def test_broadcast_model_updates_all_workers(
        self, make_coordinator, tiny_model: _TinyModel
    ) -> None:
        """Broadcasting propagates the state dict and version to every worker."""
        coordinator = make_coordinator(num_workers=2)
        new_state = tiny_model.state_dict()

        coordinator.broadcast_model(new_state, version=7)

        assert coordinator._model_version == 7
        for w in coordinator.workers:
            assert w.model_version == 7
            assert w.get_stats().model_version == 7

    def test_clear_local_experiences(self, make_coordinator) -> None:
        """Clearing empties the local experience buffer."""
        coordinator = make_coordinator(num_workers=1)
        coordinator._local_experiences.extend([{"a": 1}])

        coordinator.clear_local_experiences()

        assert coordinator._local_experiences == []

    def test_get_state_reports_workers_active_and_buffer_size(self, make_coordinator) -> None:
        """``get_state`` reflects the live worker count and buffer size."""
        coordinator = make_coordinator(num_workers=2)
        coordinator._local_experiences.extend([{"a": 1}, {"a": 2}])

        state = coordinator.get_state()

        assert isinstance(state, CoordinatorState)
        assert state.workers_active == 2
        assert state.buffer_size == 2

    def test_get_worker_stats_returns_one_entry_per_worker(self, make_coordinator) -> None:
        """One ``WorkerStats`` per worker, with distinct worker IDs."""
        coordinator = make_coordinator(num_workers=3)

        stats = coordinator.get_worker_stats()

        assert len(stats) == 3
        assert all(isinstance(s, WorkerStats) for s in stats)
        assert {s.worker_id for s in stats} == {0, 1, 2}

    def test_shutdown_stops_all_workers(self, make_coordinator) -> None:
        """``shutdown()`` signals every worker's stop event."""
        coordinator = make_coordinator(num_workers=2)

        coordinator.shutdown()

        assert all(w._should_stop.is_set() for w in coordinator.workers)


# ---------------------------------------------------------------------------
# create_self_play_coordinator factory
# ---------------------------------------------------------------------------


class TestCreateSelfPlayCoordinatorFactory:
    """Tests for the ``create_self_play_coordinator`` factory function."""

    def test_factory_builds_a_coordinator(
        self, tiny_model: _TinyModel, mcts_config_stub: SimpleNamespace
    ) -> None:
        """The factory returns a working ``SelfPlayCoordinator`` sized by ``workers_per_node``."""
        coordinator = create_self_play_coordinator(
            model=tiny_model,
            mcts_config=mcts_config_stub,
            board_sizes=[9],
            workers_per_node=1,
        )

        assert isinstance(coordinator, SelfPlayCoordinator)
        assert coordinator.config.num_workers == 1
        assert len(coordinator.workers) == 1

    def test_factory_forwards_extra_kwargs_to_config(
        self, tiny_model: _TinyModel, mcts_config_stub: SimpleNamespace
    ) -> None:
        """Extra keyword arguments reach the underlying ``SelfPlayDistributedConfig``."""
        coordinator = create_self_play_coordinator(
            model=tiny_model,
            mcts_config=mcts_config_stub,
            board_sizes=[9],
            workers_per_node=1,
            experience_sharing="local",
        )

        assert coordinator.config.experience_sharing == "local"


# ---------------------------------------------------------------------------
# _all_gather_experiences -- mocked torch.distributed process group
# ---------------------------------------------------------------------------


class TestAllGatherExperiencesMocked:
    """Exercises the ``dist.all_gather``-dependent inner path with a mocked process group.

    Single-process construction defaults ``world_size=1`` (see
    ``_get_env_rank_info``), so a correctly behaving fake ``all_gather`` need
    only echo the local tensor back into the (length-1) output list --
    exactly what a real single-rank ``all_gather`` would produce -- while
    every other line in ``_all_gather_experiences`` (byte<->tensor packing,
    size negotiation, padding, pickling) runs for real, unmocked.
    """

    @staticmethod
    def _fake_all_gather(
        output_list: list[torch.Tensor],
        input_tensor: torch.Tensor,
        group: object = None,
        async_op: bool = False,
    ) -> None:
        for i in range(len(output_list)):
            output_list[i] = input_tensor.clone()

    def test_global_sharing_round_trips_through_all_gather(self, make_coordinator) -> None:
        """An initialized process group with "global" sharing calls ``all_gather`` twice."""
        coordinator = make_coordinator(num_workers=1, experience_sharing="global")
        coordinator._local_experiences.extend([{"idx": 0}, {"idx": 1}, {"idx": 2}])

        mock_gather = MagicMock(side_effect=self._fake_all_gather)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_gather", mock_gather),
        ):
            result = coordinator.synchronize_experiences()

        assert result == coordinator._local_experiences
        assert mock_gather.call_count == 2  # sizes gather + payload gather

    def test_hierarchical_sharing_delegates_to_all_gather(self, make_coordinator) -> None:
        """``experience_sharing="hierarchical"`` currently delegates straight to the global path."""
        coordinator = make_coordinator(num_workers=1, experience_sharing="hierarchical")
        coordinator._local_experiences.extend([{"idx": 0}])

        mock_gather = MagicMock(side_effect=self._fake_all_gather)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_gather", mock_gather),
        ):
            result = coordinator.synchronize_experiences()

        assert result == coordinator._local_experiences
        assert mock_gather.call_count == 2
