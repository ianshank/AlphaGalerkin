"""Self-play game generation for AlphaGalerkin training.

Generates training data through self-play using MCTS-guided game play.
Supports multiple board sizes for resolution-independent training.
Includes parallel generation via ``ParallelSelfPlayWorker`` using
``torch.multiprocessing`` for safe model sharing across workers.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch

from src.constants import DEFAULT_BOARD_SIZES, DEFAULT_MAX_MOVES, DEFAULT_TEMPERATURE_SCHEDULE
from src.mcts.evaluator import FNetEvaluator
from src.mcts.search import MCTS, BatchMCTS
from src.tools.gtp import SimpleGoGame
from src.training.replay_buffer import Experience

if TYPE_CHECKING:
    from config.schemas import MCTSConfig
    from src.games.interface import GameInterface
    from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)

# Default worker count derived from environment; capped for safety.
_DEFAULT_MAX_WORKERS = 8


@dataclass
class GameRecord:
    """Record of a complete self-play game.

    Stores the full trajectory for training data extraction.
    """

    board_size: int
    states: list[np.ndarray] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    outcome: float = 0.0  # 1.0 = first player wins, -1.0 = second player wins
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Get number of moves in the game."""
        return len(self.states)

    def to_experiences(self) -> list[Experience]:
        """Convert game record to training experiences.

        Each position is converted to an experience with the game outcome
        adjusted for the current player's perspective.

        Returns:
            List of Experience objects.

        """
        experiences = []
        n_positions = len(self.states)

        for i, (state, policy) in enumerate(zip(self.states, self.policies, strict=False)):
            # Determine value from current player's perspective
            # Odd moves are second player, even moves are first player
            perspective = 1.0 if i % 2 == 0 else -1.0
            value_for_player = self.outcome * perspective

            # Create experience
            exp = Experience(
                board_state=torch.from_numpy(state),
                board_size=self.board_size,
                target_policy=torch.from_numpy(policy),
                target_value=value_for_player,
                metadata={
                    "move_number": i,
                    "game_length": n_positions,
                    "action_taken": self.actions[i] if i < len(self.actions) else None,
                    **self.metadata,
                },
            )
            experiences.append(exp)

        return experiences


class SelfPlayWorker:
    """Worker for generating self-play games.

    Uses MCTS with neural network guidance to play games against itself,
    generating training data in the process.
    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        mcts_config: MCTSConfig | None = None,
        device: torch.device | str = "cpu",
        board_sizes: list[int] | None = None,
        use_batch_mcts: bool = False,
        temperature_schedule: dict[int, float] | None = None,
        game: GameInterface | None = None,
    ) -> None:
        """Initialize self-play worker.

        Args:
            model: AlphaGalerkin model for evaluation.
            mcts_config: MCTS configuration.
            device: Device for inference.
            board_sizes: Board sizes to play on (sampled uniformly).
            use_batch_mcts: Whether to use batch MCTS.
            temperature_schedule: Move number -> temperature mapping.
            game: Optional GameInterface for non-Go games (e.g. chess).
                  When provided, uses StatefulGameWrapper for MCTS.
                  When None, falls back to SimpleGoGame.

        """
        self.model = model
        self.device = torch.device(device)
        self.board_sizes = board_sizes or list(DEFAULT_BOARD_SIZES)
        self.game = game

        # Default temperature schedule: high exploration early, low late
        self.temperature_schedule = temperature_schedule or dict(DEFAULT_TEMPERATURE_SCHEDULE)

        # Create evaluator
        self.evaluator = FNetEvaluator(
            model,
            device=self.device,
            use_fast_path=True,
        )

        # MCTS configuration
        self._mcts_kwargs: dict[str, Any] = {}
        if mcts_config is not None:
            self._mcts_kwargs = {
                "n_simulations": mcts_config.n_simulations,
                "c_puct": mcts_config.c_puct,
                "dirichlet_alpha": mcts_config.dirichlet_alpha,
                "dirichlet_epsilon": mcts_config.dirichlet_epsilon,
                "virtual_loss": mcts_config.virtual_loss,
            }
            if use_batch_mcts:
                self._mcts_kwargs["batch_size"] = mcts_config.batch_size

        self.use_batch_mcts = use_batch_mcts

        # Statistics
        self._games_played = 0
        self._total_moves = 0
        self._outcomes = {"black": 0, "white": 0, "draw": 0}

    def _create_mcts(self) -> MCTS:
        """Create MCTS instance."""
        if self.use_batch_mcts:
            return BatchMCTS(evaluator=self.evaluator, **self._mcts_kwargs)
        return MCTS(evaluator=self.evaluator, **self._mcts_kwargs)

    def _get_temperature(self, move_number: int) -> float:
        """Get temperature for given move number.

        Args:
            move_number: Current move number.

        Returns:
            Temperature value.

        """
        # Find the applicable temperature threshold
        applicable_threshold = 0
        for threshold in sorted(self.temperature_schedule.keys()):
            if move_number >= threshold:
                applicable_threshold = threshold
            else:
                break
        return self.temperature_schedule[applicable_threshold]

    def play_game(
        self,
        board_size: int | None = None,
        max_moves: int = DEFAULT_MAX_MOVES,
        add_noise: bool = True,
    ) -> GameRecord:
        """Play a complete self-play game.

        Args:
            board_size: Board size (random if None, ignored for chess).
            max_moves: Maximum moves before termination.
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            GameRecord with complete game trajectory.

        """
        if self.game is not None:
            return self._play_game_generic(max_moves, add_noise)
        return self._play_game_go(board_size, max_moves, add_noise)

    def _play_game_generic(
        self,
        max_moves: int = DEFAULT_MAX_MOVES,
        add_noise: bool = True,
    ) -> GameRecord:
        """Play a self-play game using a GameInterface (chess, etc.).

        Uses StatefulGameWrapper to bridge the stateless GameInterface
        to the MCTS protocol.
        """
        from src.games.wrapper import StatefulGameWrapper

        assert self.game is not None
        game = self.game
        state = game.initial_state()
        board_size = state.board.shape[0] if state.board.ndim >= 2 else 8
        n_actions = game.action_space_size
        mcts = self._create_mcts()

        record = GameRecord(
            board_size=board_size,
            metadata={"game_id": self._games_played, "game_type": "generic"},
        )

        move_number = 0
        while not game.is_terminal(state) and move_number < max_moves:
            # Get tensor representation as numpy
            state_tensor = game.to_tensor(state).cpu().numpy()

            # Wrap for MCTS
            game_wrapper = StatefulGameWrapper(game, state)
            policy_dist = mcts.search(game_wrapper, add_noise=add_noise and move_number == 0)

            # Convert visit distribution to full policy vector
            policy = np.zeros(n_actions, dtype=np.float32)
            for action, prob in policy_dist.items():
                policy[action] = prob

            # Store state and policy
            record.states.append(state_tensor)
            record.policies.append(policy)

            # Select action with temperature
            temperature = self._get_temperature(move_number)
            if temperature == 0:
                action = max(policy_dist.keys(), key=lambda a: policy_dist[a])
            else:
                actions = list(policy_dist.keys())
                probs = np.array([policy_dist[a] for a in actions])
                probs = probs ** (1.0 / temperature)
                probs = probs / probs.sum()
                action = int(np.random.choice(actions, p=probs))

            record.actions.append(action)

            # Apply action
            state = game.apply_action(state, action)

            # Advance MCTS tree
            mcts.advance(action)
            move_number += 1

        # Determine outcome
        winner = game.get_winner(state)
        if winner is not None:
            record.outcome = float(winner)
        else:
            record.outcome = 0.0

        # Update statistics
        self._games_played += 1
        self._total_moves += move_number
        if record.outcome > 0:
            self._outcomes["black"] += 1
        elif record.outcome < 0:
            self._outcomes["white"] += 1
        else:
            self._outcomes["draw"] += 1

        logger.debug(
            "game_completed",
            board_size=board_size,
            moves=move_number,
            outcome=record.outcome,
            game_type="generic",
        )

        return record

    def _play_game_go(
        self,
        board_size: int | None = None,
        max_moves: int = DEFAULT_MAX_MOVES,
        add_noise: bool = True,
    ) -> GameRecord:
        """Play a self-play game using SimpleGoGame (original Go path)."""
        # Select board size
        if board_size is None:
            board_size = random.choice(self.board_sizes)

        # Initialize game and MCTS
        game = SimpleGoGame(board_size)
        mcts = self._create_mcts()

        # Initialize record
        record = GameRecord(
            board_size=board_size,
            metadata={"game_id": self._games_played},
        )

        move_number = 0
        while not game.is_terminal() and move_number < max_moves:
            # Get current state
            state = game.get_state()

            # Run MCTS search
            policy_dist = mcts.search(game, add_noise=add_noise and move_number == 0)

            # Convert visit distribution to full policy vector
            n_actions = board_size**2 + 1  # +1 for pass
            policy = np.zeros(n_actions, dtype=np.float32)
            for action, prob in policy_dist.items():
                policy[action] = prob

            # Store state and policy
            record.states.append(state)
            record.policies.append(policy)

            # Select action with temperature
            temperature = self._get_temperature(move_number)
            if temperature == 0:
                action = max(policy_dist.keys(), key=lambda a: policy_dist[a])
            else:
                actions = list(policy_dist.keys())
                probs = np.array([policy_dist[a] for a in actions])

                # Apply temperature
                probs = probs ** (1.0 / temperature)
                probs = probs / probs.sum()

                action = int(np.random.choice(actions, p=probs))

            record.actions.append(action)

            # Apply action to game
            if action == board_size**2:
                game.play_pass()
            else:
                row = action // board_size
                col = action % board_size
                if not game.play(row, col):
                    # Illegal move - should not happen with proper MCTS
                    logger.warning(
                        "illegal_move_in_self_play",
                        action=action,
                        move_number=move_number,
                    )
                    game.play_pass()

            # Advance MCTS tree
            mcts.advance(action)
            move_number += 1

        # Determine outcome
        if game.is_terminal():
            winner = game.get_winner()
            # Winner from first player's perspective
            if game.current_player == SimpleGoGame.BLACK:
                # Black to play means White made last move
                record.outcome = -float(winner)
            else:
                record.outcome = float(winner)
        else:
            # Game didn't terminate - score as draw
            record.outcome = 0.0

        # Update statistics
        self._games_played += 1
        self._total_moves += move_number
        if record.outcome > 0:
            self._outcomes["black"] += 1
        elif record.outcome < 0:
            self._outcomes["white"] += 1
        else:
            self._outcomes["draw"] += 1

        logger.debug(
            "game_completed",
            board_size=board_size,
            moves=move_number,
            outcome=record.outcome,
        )

        return record

    def generate_games(
        self,
        n_games: int,
        board_size: int | None = None,
    ) -> list[GameRecord]:
        """Generate multiple self-play games.

        Args:
            n_games: Number of games to generate.
            board_size: Fixed board size (random per game if None).

        Returns:
            List of GameRecords.

        """
        games = []
        for i in range(n_games):
            record = self.play_game(board_size=board_size)
            games.append(record)

            if (i + 1) % 10 == 0:
                logger.info(
                    "self_play_progress",
                    completed=i + 1,
                    total=n_games,
                    avg_length=self._total_moves / max(1, self._games_played),
                )

        return games

    def generate_experiences(
        self,
        n_games: int,
        board_size: int | None = None,
    ) -> list[Experience]:
        """Generate training experiences from self-play.

        Convenience method that combines game generation and experience extraction.

        Args:
            n_games: Number of games to generate.
            board_size: Fixed board size (random per game if None).

        Returns:
            List of Experience objects.

        """
        games = self.generate_games(n_games, board_size)

        experiences = []
        for game in games:
            experiences.extend(game.to_experiences())

        logger.info(
            "experiences_generated",
            n_games=n_games,
            n_experiences=len(experiences),
        )

        return experiences

    def get_stats(self) -> dict[str, Any]:
        """Get self-play statistics.

        Returns:
            Dictionary with statistics.

        """
        return {
            "games_played": self._games_played,
            "total_moves": self._total_moves,
            "avg_game_length": (
                self._total_moves / self._games_played if self._games_played > 0 else 0.0
            ),
            "outcomes": self._outcomes.copy(),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._games_played = 0
        self._total_moves = 0
        self._outcomes = {"black": 0, "white": 0, "draw": 0}


def _worker_generate_games(args: tuple[Any, ...]) -> list[GameRecord]:
    """Worker function for parallel game generation.

    Runs in a subprocess spawned by ``torch.multiprocessing``.
    Each worker creates its own ``SelfPlayWorker`` around the shared
    model to avoid any mutable-state contention.

    Args:
        args: Tuple of (model, n_games, board_size, worker_kwargs, worker_id).

    Returns:
        List of GameRecords produced by this worker.

    """
    model, n_games, board_size, worker_kwargs, worker_id = args

    # Re-seed per-worker to avoid identical game trajectories.
    seed = int.from_bytes(os.urandom(4), "little") + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed)

    worker_logger = structlog.get_logger(__name__)
    worker_logger.debug(
        "parallel_worker_started",
        worker_id=worker_id,
        n_games=n_games,
    )

    worker = SelfPlayWorker(model, **worker_kwargs)
    records = worker.generate_games(n_games, board_size)

    worker_logger.debug(
        "parallel_worker_finished",
        worker_id=worker_id,
        games_generated=len(records),
    )
    return records


class ParallelSelfPlayWorker:
    """Parallel self-play using multiple worker processes.

    Distributes game generation across processes using
    ``torch.multiprocessing`` for safe model sharing.  When
    ``n_workers=1`` (or on platforms where ``fork`` is unavailable),
    falls back to sequential generation for maximum compatibility.
    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        n_workers: int = 4,
        **worker_kwargs: Any,
    ) -> None:
        """Initialize parallel self-play.

        Args:
            model: AlphaGalerkin model (shared across workers via
                ``torch.multiprocessing`` memory sharing).
            n_workers: Number of worker processes.  Clamped to
                ``[1, _DEFAULT_MAX_WORKERS]``.  When ``1``, uses
                sequential fallback.
            **worker_kwargs: Arguments forwarded to each
                ``SelfPlayWorker`` (e.g. ``mcts_config``, ``device``,
                ``board_sizes``, ``temperature_schedule``).

        """
        self.model = model
        self.n_workers = max(1, min(n_workers, _DEFAULT_MAX_WORKERS))
        self.worker_kwargs = worker_kwargs

        # Keep a sequential fallback for n_workers == 1 or error recovery
        self._sequential_worker = SelfPlayWorker(model, **worker_kwargs)

        logger.info(
            "parallel_self_play_initialized",
            n_workers=self.n_workers,
        )

    def _use_parallel(self) -> bool:
        """Determine whether to use multiprocessing.

        Falls back to sequential when only one worker is requested,
        when CUDA is in use (forking with CUDA is unsafe), or when the
        start method cannot be set to ``spawn``.
        """
        if self.n_workers <= 1:
            return False

        device = self.worker_kwargs.get("device", "cpu")
        if isinstance(device, torch.device):
            device = device.type
        if str(device).startswith("cuda"):
            # CUDA tensors cannot be shared across fork; spawn is
            # required but adds overhead.  For safety, fall back.
            logger.debug(
                "parallel_self_play_cuda_fallback",
                reason="CUDA device detected; using sequential",
            )
            return False

        return True

    def generate_games(
        self,
        n_games: int,
        board_size: int | None = None,
    ) -> list[GameRecord]:
        """Generate games across multiple worker processes.

        Splits ``n_games`` evenly across workers, collects results,
        and returns the merged list.  Falls back to sequential on
        error or when parallel mode is unavailable.

        Args:
            n_games: Total number of games to generate.
            board_size: Fixed board size (random per game if ``None``).

        Returns:
            List of GameRecords from all workers.

        """
        if not self._use_parallel() or n_games <= 1:
            return self._sequential_worker.generate_games(n_games, board_size)

        # Distribute games as evenly as possible across workers
        n_workers = min(self.n_workers, n_games)
        base_count = n_games // n_workers
        remainder = n_games % n_workers
        per_worker = [base_count + (1 if i < remainder else 0) for i in range(n_workers)]

        logger.info(
            "parallel_self_play_starting",
            n_games=n_games,
            n_workers=n_workers,
            per_worker=per_worker,
        )

        self.model.cpu()
        self.model.train(False)

        worker_args = [
            (self.model, count, board_size, self.worker_kwargs, worker_id)
            for worker_id, count in enumerate(per_worker)
        ]

        try:
            import torch.multiprocessing as mp

            # Ensure model is in shared memory for subprocess access.
            # Placed inside the try block so that platforms that do not
            # support POSIX shared memory (e.g. Windows without /dev/shm)
            # fall back to sequential generation rather than raising before
            # the except handler.
            self.model.share_memory()

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                results = pool.map(_worker_generate_games, worker_args)

            all_games: list[GameRecord] = []
            for worker_result in results:
                all_games.extend(worker_result)

            logger.info(
                "parallel_self_play_completed",
                total_games=len(all_games),
                n_workers=n_workers,
            )
            return all_games

        except Exception as exc:
            logger.warning(
                "parallel_self_play_failed_falling_back",
                error=str(exc),
                n_games=n_games,
            )
            return self._sequential_worker.generate_games(n_games, board_size)

    def generate_experiences(
        self,
        n_games: int,
        board_size: int | None = None,
    ) -> list[Experience]:
        """Generate training experiences from parallel self-play.

        Args:
            n_games: Number of games to generate.
            board_size: Fixed board size (random per game if ``None``).

        Returns:
            Flat list of experiences extracted from all games.

        """
        games = self.generate_games(n_games, board_size)
        experiences: list[Experience] = []
        for game in games:
            experiences.extend(game.to_experiences())
        return experiences
