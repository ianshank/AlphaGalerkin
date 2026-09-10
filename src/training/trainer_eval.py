"""Evaluation / tournament / engine-eval helpers for :class:`Trainer`.

Extracted so ``src/training/trainer.py`` stays a facade. Methods on Trainer
remain the public call sites (tests patch them); these functions implement
the bodies. ``__init__`` still does not call ``super().__init__()``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import structlog

from src.constants import (
    WIN_RATE_ACCEPT_THRESHOLD,
    WIN_RATE_REJECT_THRESHOLD,
)
from src.engines.config import MatchConfig, UCIConfig

logger = structlog.get_logger(__name__)


class SupportsTrainerEval(Protocol):
    """Attributes the evaluation helpers read, plus methods tests patch."""

    model: Any
    training_config: Any
    evaluator: Any
    tracker: Any
    elo_tracker: Any
    config: Any
    checkpoint_manager: Any

    def _run_checkpoint_tournament(self, step: int, n_games: int) -> None: ...

    def _run_engine_evaluation(self, step: int) -> None: ...

    def _extract_step_from_checkpoint(self, checkpoint_path: Path) -> int: ...


def run_evaluation(trainer: SupportsTrainerEval, step: int) -> float:
    """Run evaluation and log results.

    Args:
        trainer: Trainer facade.
        step: Current training step.

    Returns:
        Average win rate across board sizes (for early stopping).

    """
    logger.info("evaluation_starting", step=step)
    trainer.model.eval()

    n_games = trainer.training_config.eval_games
    use_multi_res = trainer.training_config.multi_resolution_eval

    win_rates: list[float] = []

    if use_multi_res and hasattr(trainer.evaluator, "evaluate_multi_resolution"):
        # Use multi-resolution evaluation
        results = trainer.evaluator.evaluate_multi_resolution(n_games_per_size=n_games)
        for board_size, result in results.items():
            win_rates.append(result.win_rate)
            if trainer.tracker is not None:
                trainer.tracker.log_evaluation(
                    result=result,
                    prefix=f"eval/{board_size}x{board_size}",
                    step=step,
                )
    else:
        # Evaluate on each board size individually
        for board_size in trainer.config.board_sizes:
            result = trainer.evaluator.evaluate_vs_random(
                n_games=n_games,
                board_size=board_size,
            )
            win_rates.append(result.win_rate)
            if trainer.tracker is not None:
                trainer.tracker.log_evaluation(
                    result=result,
                    prefix=f"eval/{board_size}x{board_size}",
                    step=step,
                )

    # Checkpoint tournament evaluation (Elo tracking)
    if trainer.elo_tracker is not None:
        trainer._run_checkpoint_tournament(step, n_games)

    # Engine evaluation (Stockfish benchmark)
    if (
        trainer.training_config.engine_eval_enabled
        and trainer.training_config.engine_eval_path is not None
        and trainer.evaluator.game is not None
    ):
        trainer._run_engine_evaluation(step)

    # Measure policy agreement
    policy_agreement = trainer.evaluator.measure_policy_agreement(
        n_positions=100,
        board_size=9,
    )

    if trainer.tracker is not None:
        trainer.tracker.log_metrics(
            {"eval/policy_agreement": policy_agreement},
            step=step,
        )

    trainer.model.train()
    logger.info("evaluation_completed", step=step)

    # Return average win rate for early stopping
    return sum(win_rates) / len(win_rates) if win_rates else 0.0


def run_checkpoint_tournament(trainer: SupportsTrainerEval, step: int, n_games: int) -> None:
    """Run tournament against previous checkpoints for Elo tracking.

    Args:
        trainer: Trainer facade.
        step: Current training step.
        n_games: Number of games per opponent.

    """
    if trainer.elo_tracker is None:
        return

    # Get list of available checkpoints
    checkpoint_paths = trainer.checkpoint_manager.get_all_checkpoints()
    n_opponents = min(
        len(checkpoint_paths),
        trainer.training_config.n_tournament_opponents,
    )

    if n_opponents == 0:
        return

    logger.info(
        "checkpoint_tournament_starting",
        step=step,
        n_opponents=n_opponents,
    )

    # Select recent checkpoints as opponents
    opponent_paths = checkpoint_paths[-n_opponents:]

    for opponent_path in opponent_paths:
        try:
            result = trainer.evaluator.evaluate_vs_checkpoint(
                checkpoint_path=opponent_path,
                n_games=n_games,
            )

            # Extract opponent step from checkpoint filename
            opponent_step = trainer._extract_step_from_checkpoint(opponent_path)

            # Determine score: 1.0=win, 0.5=draw, 0.0=loss
            if result.win_rate > WIN_RATE_ACCEPT_THRESHOLD:
                score = 1.0
            elif result.win_rate < WIN_RATE_REJECT_THRESHOLD:
                score = 0.0
            else:
                score = 0.5

            # Update Elo ratings
            trainer.elo_tracker.update_ratings(step, opponent_step, score)

            # Log to W&B
            if trainer.tracker is not None:
                current_rating = trainer.elo_tracker.get_rating(step)
                trainer.tracker.log_metrics(
                    {
                        f"elo/vs_step_{opponent_step}": result.win_rate,
                        "elo/current_rating": current_rating,
                    },
                    step=step,
                )

            logger.debug(
                "checkpoint_match_completed",
                opponent_step=opponent_step,
                win_rate=result.win_rate,
                score=score,
            )

        except Exception as e:
            logger.warning(
                "checkpoint_match_failed",
                opponent_path=str(opponent_path),
                error=str(e),
            )


def run_engine_evaluation(trainer: SupportsTrainerEval, step: int) -> None:
    """Run evaluation against external UCI engine (e.g., Stockfish).

    Creates engine and match configs from training config values,
    plays games, and logs Elo metrics to W&B.

    Args:
        trainer: Trainer facade.
        step: Current training step.

    """
    if trainer.training_config.engine_eval_path is None:
        return

    engine_path = trainer.training_config.engine_eval_path
    depth = trainer.training_config.engine_eval_depth
    n_games = trainer.training_config.engine_eval_games
    movetime = trainer.training_config.engine_eval_movetime_ms

    try:
        engine_config = UCIConfig(
            name="stockfish_eval",
            engine_path=Path(engine_path),
            depth_limit=depth if movetime is None else None,
            movetime_ms=movetime,
        )
        match_config = MatchConfig(
            name="engine_eval_match",
            n_games=n_games,
        )

        logger.info(
            "engine_evaluation_starting",
            step=step,
            engine_path=engine_path,
            depth=depth,
            n_games=n_games,
        )

        result = trainer.evaluator.evaluate_vs_engine(
            engine_config=engine_config,
            match_config=match_config,
        )

        # Log Elo metrics to W&B
        elo_metrics: dict[str, float | int] = {
            "eval/engine/win_rate": result.win_rate,
            "eval/engine/wins": result.wins,
            "eval/engine/losses": result.losses,
            "eval/engine/draws": result.draws,
            "eval/engine/n_games": result.n_games,
            "eval/engine/avg_game_length": result.avg_game_length,
        }

        # Extract Elo estimate from metadata if available
        if "elo_difference" in result.metadata:
            elo_metrics["eval/engine/elo_diff"] = result.metadata["elo_difference"]
        if "los" in result.metadata:
            elo_metrics["eval/engine/los"] = result.metadata["los"]

        if trainer.tracker is not None:
            trainer.tracker.log_metrics(
                elo_metrics,
                step=step,
            )

        logger.info(
            "engine_evaluation_completed",
            step=step,
            win_rate=f"{result.win_rate:.2%}",
            elo_diff=result.metadata.get("elo_difference", "N/A"),
        )

    except Exception as e:
        logger.warning(
            "engine_evaluation_failed",
            step=step,
            error=str(e),
        )


def extract_step_from_checkpoint(checkpoint_path: Path) -> int:
    """Extract training step from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Training step number.

    """
    # Filename format: checkpoint_00010000.pt
    match = re.search(r"checkpoint_(\d+)", checkpoint_path.stem)
    if match:
        return int(match.group(1))
    return 0
