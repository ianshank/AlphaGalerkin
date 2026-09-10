"""Replay-buffer fill loop extracted from :class:`src.training.trainer.Trainer`.

The production trainer's ``__init__`` does not call ``super().__init__()``
(see that class's docstring). These helpers are called from Trainer methods
so tests that patch ``Trainer._fill_buffer`` keep working.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class BufferFillError(RuntimeError):
    """Raised when the replay buffer cannot be filled within the call budget.

    Raised by :meth:`Trainer._fill_buffer` when it cannot reach the target
    replay-buffer size within ``TrainingConfig.max_buffer_fill_iterations``
    calls to ``SelfPlayWorker.generate_experiences``.

    This guards the self-play buffer-fill loop against hot-looping
    indefinitely when self-play stops yielding usable experiences (e.g. a
    game-length or self-play configuration bug): a buffer that never fills
    is a real configuration/environment problem the caller needs to know
    about, not something to silently give up on and continue training with
    an under-filled buffer.
    """


class SupportsBufferFill(Protocol):
    """Attributes ``fill_replay_buffer`` reads and mutates."""

    buffer: Any
    training_config: Any
    model: Any
    curriculum: Any
    global_step: int
    self_play_worker: Any
    tracker: Any
    total_games_generated: int


def fill_replay_buffer(trainer: SupportsBufferFill, min_size: int) -> None:
    """Fill replay buffer to minimum size.

    Bounded by ``training_config.max_buffer_fill_iterations`` self-play
    generation calls. Without this bound, a self-play call that nets
    zero (or too few) usable experiences -- e.g. from a game-length or
    configuration bug -- would make this loop re-invoke full MCTS
    self-play generation forever.

    Args:
        trainer: Trainer facade holding buffer, self-play worker, and config.
        min_size: Minimum number of experiences needed.

    Raises:
        BufferFillError: If the buffer still has not reached
            ``min_size`` after ``max_buffer_fill_iterations`` self-play
            generation calls.

    """
    fill_start = time.time()
    initial_size = len(trainer.buffer)
    max_iterations = trainer.training_config.max_buffer_fill_iterations
    iterations = 0

    while len(trainer.buffer) < min_size:
        if iterations >= max_iterations:
            elapsed = time.time() - fill_start
            raise BufferFillError(
                f"_fill_buffer did not reach the minimum buffer size of "
                f"{min_size} experiences after {iterations} self-play "
                f"generation call(s) ({elapsed:.1f}s elapsed): buffer "
                f"holds {len(trainer.buffer)} experiences (started at "
                f"{initial_size}). generate_experiences() is likely "
                "yielding zero or too few usable experiences per call "
                "-- check the self-play/game configuration (board size, "
                "game-length limits, curriculum settings) or the "
                "self-play worker for a bug before retrying. Raise "
                "TrainingConfig.max_buffer_fill_iterations (currently "
                f"{max_iterations}) if more self-play iterations are "
                "genuinely expected to be needed."
            )
        iterations += 1
        n_games = trainer.training_config.n_self_play_games
        logger.info(
            "generating_self_play_games",
            n_games=n_games,
            buffer_size=len(trainer.buffer),
            target_size=min_size,
            iteration=iterations,
            max_iterations=max_iterations,
        )

        # Generate games (use curriculum board size if enabled)
        trainer.model.eval()
        board_size = None
        if trainer.curriculum is not None:
            board_size = trainer.curriculum.sample_board_size(trainer.global_step)
        experiences = trainer.self_play_worker.generate_experiences(n_games, board_size=board_size)
        trainer.model.train()

        # Add to buffer
        trainer.buffer.add_batch(experiences)
        trainer.total_games_generated += n_games

        # Log self-play progress to W&B
        if trainer.tracker is not None:
            stats = trainer.self_play_worker.get_stats()
            trainer.tracker.log_metrics(
                {
                    "self_play/games_completed": stats["games_played"],
                    "self_play/avg_game_length": stats["avg_game_length"],
                    "self_play/buffer_size": len(trainer.buffer),
                    "self_play/black_wins": stats["outcomes"]["black"],
                    "self_play/white_wins": stats["outcomes"]["white"],
                    "self_play/draws": stats["outcomes"]["draw"],
                },
                step=trainer.global_step,
            )

    # Log buffer fill statistics
    fill_time = time.time() - fill_start
    experiences_added = len(trainer.buffer) - initial_size
    fill_rate = experiences_added / max(fill_time, 0.001)
    logger.info(
        "buffer_filled",
        initial_size=initial_size,
        final_size=len(trainer.buffer),
        target_size=min_size,
        experiences_added=experiences_added,
        fill_time_seconds=round(fill_time, 2),
        fill_rate_per_second=round(fill_rate, 1),
        iterations=iterations,
    )

    # Log buffer fill summary to W&B
    if trainer.tracker is not None:
        trainer.tracker.log_metrics(
            {
                "self_play/fill_time_seconds": round(fill_time, 2),
                "self_play/experiences_added": experiences_added,
                "self_play/fill_rate_per_second": round(fill_rate, 1),
                "self_play/total_games_generated": trainer.total_games_generated,
            },
            step=trainer.global_step,
        )
