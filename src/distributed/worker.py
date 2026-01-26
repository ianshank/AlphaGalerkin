"""Distributed self-play worker coordination.

This module provides utilities for coordinating self-play workers
across multiple nodes for efficient experience generation.

Features:
    - Distributed experience generation
    - Experience buffer synchronization
    - Model broadcast to workers
    - Load balancing across nodes
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
import torch
import torch.distributed as dist

from src.distributed.config import SelfPlayDistributedConfig, from_environment

if TYPE_CHECKING:
    from config.schemas import MCTSConfig
    from src.modeling.model import AlphaGalerkinModel
    from src.training.self_play import Experience

logger = structlog.get_logger(__name__)


@dataclass
class WorkerStats:
    """Statistics from a self-play worker."""

    worker_id: int
    games_completed: int = 0
    experiences_generated: int = 0
    average_game_length: float = 0.0
    average_time_per_game_ms: float = 0.0
    model_version: int = 0


@dataclass
class CoordinatorState:
    """State of the self-play coordinator."""

    total_games: int = 0
    total_experiences: int = 0
    model_version: int = 0
    workers_active: int = 0
    buffer_size: int = 0


class SelfPlayWorker:
    """Self-play worker for distributed experience generation.

    Runs on a single process and generates experiences via MCTS self-play.
    Can be configured to use CPU for inference to free GPU for training.

    Attributes:
        worker_id: Unique worker identifier.
        model: The model used for inference.
        config: Self-play configuration.
        device: Device for inference.

    """

    def __init__(
        self,
        worker_id: int,
        model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
        config: SelfPlayDistributedConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize self-play worker.

        Args:
            worker_id: Unique worker identifier.
            model: Model for inference.
            mcts_config: MCTS configuration.
            config: Self-play distributed configuration.
            device: Device for inference.

        """
        self.worker_id = worker_id
        self.model = model
        self.mcts_config = mcts_config
        self.config = config

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.model_version = 0
        self._should_stop = threading.Event()
        self._experience_queue: queue.Queue[Experience] = queue.Queue()

        self._stats = WorkerStats(worker_id=worker_id)

        self._logger = structlog.get_logger(__name__).bind(
            worker_id=worker_id,
            device=str(device),
        )

    def generate_batch(
        self,
        n_games: int,
        board_sizes: list[int],
    ) -> list[Experience]:
        """Generate a batch of self-play games.

        Args:
            n_games: Number of games to generate.
            board_sizes: Available board sizes to sample from.

        Returns:
            List of generated experiences.

        """
        from random import choice

        from src.training.self_play import SelfPlayWorker as SPW

        experiences: list[Experience] = []
        game_times = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(n_games):
                if self._should_stop.is_set():
                    break

                start_time = time.perf_counter()

                # Select random board size
                board_size = choice(board_sizes)

                # Create internal self-play worker for this game
                spw = SPW(
                    model=self.model,
                    mcts_config=self.mcts_config,
                    device=self.device,
                    board_sizes=[board_size],
                )

                # Generate experiences from one game
                game_experiences = spw.generate_experiences(1)
                experiences.extend(game_experiences)

                game_time = (time.perf_counter() - start_time) * 1000
                game_times.append(game_time)

        # Update stats
        self._stats.games_completed += n_games
        self._stats.experiences_generated += len(experiences)
        self._stats.average_time_per_game_ms = sum(game_times) / len(game_times) if game_times else 0

        self._logger.debug(
            "batch_generated",
            n_games=n_games,
            n_experiences=len(experiences),
            avg_time_ms=f"{self._stats.average_time_per_game_ms:.2f}",
        )

        return experiences

    def update_model(self, state_dict: dict[str, Any], version: int) -> None:
        """Update the worker's model.

        Args:
            state_dict: New model state dict.
            version: Model version number.

        """
        self.model.load_state_dict(state_dict)
        self.model_version = version
        self._stats.model_version = version

        self._logger.debug("model_updated", version=version)

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._should_stop.set()

    def get_stats(self) -> WorkerStats:
        """Get worker statistics.

        Returns:
            Current worker statistics.

        """
        return self._stats


class SelfPlayCoordinator:
    """Coordinates distributed self-play across multiple workers.

    Manages multiple SelfPlayWorker instances, handles experience
    collection, and synchronizes experiences across nodes.

    Attributes:
        config: Self-play distributed configuration.
        workers: List of managed workers.
        model: Shared model for workers.

    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        mcts_config: MCTSConfig,
        config: SelfPlayDistributedConfig,
        board_sizes: list[int],
    ) -> None:
        """Initialize coordinator.

        Args:
            model: Model for self-play inference.
            mcts_config: MCTS configuration.
            config: Self-play distributed configuration.
            board_sizes: Available board sizes.

        """
        self.model = model.cpu() if config.cpu_workers else model
        self.mcts_config = mcts_config
        self.config = config
        self.board_sizes = board_sizes

        # Distributed info
        self.rank, self.local_rank, self.world_size = from_environment()

        # Workers
        self.workers: list[SelfPlayWorker] = []

        # Experience buffer
        self._local_experiences: list[Experience] = []
        self._experience_lock = threading.Lock()

        # State
        self._state = CoordinatorState()
        self._model_version = 0

        self._logger = structlog.get_logger(__name__).bind(
            rank=self.rank,
            world_size=self.world_size,
            n_workers=config.workers_per_node,
        )

        # Initialize workers
        self._initialize_workers()

    def _initialize_workers(self) -> None:
        """Initialize self-play workers."""
        device = torch.device("cpu") if self.config.cpu_workers else torch.device("cuda")

        for i in range(self.config.workers_per_node):
            worker_id = self.rank * self.config.workers_per_node + i

            # Clone model for each worker to avoid state sharing
            worker_model = type(self.model)(self.model.config)
            worker_model.load_state_dict(self.model.state_dict())
            worker_model = worker_model.to(device)

            worker = SelfPlayWorker(
                worker_id=worker_id,
                model=worker_model,
                mcts_config=self.mcts_config,
                config=self.config,
                device=device,
            )
            self.workers.append(worker)

        self._logger.info("workers_initialized", count=len(self.workers))

    def generate_experiences(
        self,
        total_games: int,
    ) -> list[Experience]:
        """Generate experiences using all workers.

        Args:
            total_games: Total games to generate across all workers.

        Returns:
            List of generated experiences.

        """
        games_per_worker = total_games // len(self.workers)
        remainder = total_games % len(self.workers)

        all_experiences: list[Experience] = []

        # Generate in parallel using threads
        threads: list[threading.Thread] = []
        results: list[list[Experience]] = [[] for _ in self.workers]

        def worker_task(worker: SelfPlayWorker, n_games: int, result_idx: int) -> None:
            experiences = worker.generate_batch(n_games, self.board_sizes)
            results[result_idx] = experiences

        for i, worker in enumerate(self.workers):
            n_games = games_per_worker + (1 if i < remainder else 0)
            thread = threading.Thread(target=worker_task, args=(worker, n_games, i))
            threads.append(thread)
            thread.start()

        # Wait for all workers
        for thread in threads:
            thread.join()

        # Collect results
        for result in results:
            all_experiences.extend(result)

        # Update state
        self._state.total_games += total_games
        self._state.total_experiences += len(all_experiences)

        with self._experience_lock:
            self._local_experiences.extend(all_experiences)

        self._logger.info(
            "experiences_generated",
            n_games=total_games,
            n_experiences=len(all_experiences),
        )

        return all_experiences

    def synchronize_experiences(self) -> list[Experience]:
        """Synchronize experiences across all nodes.

        Gathers experiences from all nodes based on the sharing strategy.

        Returns:
            Combined experiences from all nodes.

        """
        if self.config.experience_sharing == "local":
            return self._local_experiences

        if not dist.is_initialized():
            return self._local_experiences

        # Serialize local experiences
        with self._experience_lock:
            local_data = self._serialize_experiences(self._local_experiences)

        if self.config.experience_sharing == "global":
            # All-gather experiences from all nodes
            gathered = self._all_gather_experiences(local_data)
        else:  # hierarchical
            # First gather within node, then across nodes
            gathered = self._hierarchical_gather_experiences(local_data)

        # Deserialize
        all_experiences = self._deserialize_experiences(gathered)

        self._logger.debug(
            "experiences_synchronized",
            local_count=len(self._local_experiences),
            total_count=len(all_experiences),
        )

        return all_experiences

    def _serialize_experiences(self, experiences: list[Experience]) -> bytes:
        """Serialize experiences to bytes.

        Args:
            experiences: List of experiences.

        Returns:
            Serialized bytes.

        """
        import pickle

        return pickle.dumps(experiences)

    def _deserialize_experiences(self, data: bytes) -> list[Experience]:
        """Deserialize experiences from bytes.

        Args:
            data: Serialized bytes.

        Returns:
            List of experiences.

        """
        import pickle

        return pickle.loads(data)

    def _all_gather_experiences(self, local_data: bytes) -> bytes:
        """Gather experiences from all nodes.

        Args:
            local_data: Serialized local experiences.

        Returns:
            Combined serialized experiences.

        """
        import pickle

        # Create tensor from bytes
        local_tensor = torch.ByteTensor(list(local_data))
        local_size = torch.tensor([len(local_data)], dtype=torch.long)

        # Gather sizes
        sizes = [torch.tensor([0], dtype=torch.long) for _ in range(self.world_size)]
        dist.all_gather(sizes, local_size)

        # Pad to max size
        max_size = max(s.item() for s in sizes)
        padded_tensor = torch.zeros(max_size, dtype=torch.uint8)
        padded_tensor[: len(local_data)] = local_tensor

        # Gather tensors
        gathered = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded_tensor)

        # Combine experiences
        all_experiences: list[Experience] = []
        for i, (tensor, size) in enumerate(zip(gathered, sizes)):
            data = bytes(tensor[: size.item()].tolist())
            experiences = pickle.loads(data)
            all_experiences.extend(experiences)

        return pickle.dumps(all_experiences)

    def _hierarchical_gather_experiences(self, local_data: bytes) -> bytes:
        """Gather experiences hierarchically (intra-node then inter-node).

        Args:
            local_data: Serialized local experiences.

        Returns:
            Combined serialized experiences.

        """
        # For simplicity, use global gather
        # A full implementation would create intra-node and inter-node process groups
        return self._all_gather_experiences(local_data)

    def broadcast_model(self, state_dict: dict[str, Any], version: int) -> None:
        """Broadcast updated model to all workers.

        Args:
            state_dict: New model state dict.
            version: Model version number.

        """
        self._model_version = version

        for worker in self.workers:
            worker.update_model(state_dict, version)

        self._logger.debug("model_broadcasted_to_workers", version=version)

    def clear_local_experiences(self) -> None:
        """Clear the local experience buffer."""
        with self._experience_lock:
            self._local_experiences.clear()

    def get_state(self) -> CoordinatorState:
        """Get coordinator state.

        Returns:
            Current coordinator state.

        """
        self._state.workers_active = len(self.workers)
        self._state.model_version = self._model_version
        with self._experience_lock:
            self._state.buffer_size = len(self._local_experiences)
        return self._state

    def get_worker_stats(self) -> list[WorkerStats]:
        """Get statistics from all workers.

        Returns:
            List of worker statistics.

        """
        return [worker.get_stats() for worker in self.workers]

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self.workers:
            worker.stop()
        self._logger.info("coordinator_shutdown")


def create_self_play_coordinator(
    model: AlphaGalerkinModel,
    mcts_config: MCTSConfig,
    board_sizes: list[int],
    workers_per_node: int = 2,
    **kwargs: Any,
) -> SelfPlayCoordinator:
    """Factory function to create self-play coordinator.

    Args:
        model: Model for self-play.
        mcts_config: MCTS configuration.
        board_sizes: Available board sizes.
        workers_per_node: Number of workers per node.
        **kwargs: Additional configuration options.

    Returns:
        Configured SelfPlayCoordinator instance.

    """
    config = SelfPlayDistributedConfig(
        workers_per_node=workers_per_node,
        **kwargs,
    )

    return SelfPlayCoordinator(
        model=model,
        mcts_config=mcts_config,
        config=config,
        board_sizes=board_sizes,
    )
