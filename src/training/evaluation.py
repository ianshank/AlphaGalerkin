"""Evaluation pipeline for tracking training progress.

Provides metrics for assessing model quality:
- Win rate against baseline (random) player
- Win rate against previous model version
- Policy agreement with MCTS
- Value prediction accuracy
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch

from src.mcts.evaluator import FNetEvaluator, RandomEvaluator
from src.mcts.search import MCTS
from src.tools.gtp import SimpleGoGame

if TYPE_CHECKING:
    from config.schemas import MCTSConfig
    from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluation run."""

    win_rate: float
    n_games: int
    wins: int
    losses: int
    draws: int
    avg_game_length: float
    avg_value_error: float = 0.0
    policy_agreement: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "win_rate": self.win_rate,
            "n_games": self.n_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "avg_game_length": self.avg_game_length,
            "avg_value_error": self.avg_value_error,
            "policy_agreement": self.policy_agreement,
            **self.metadata,
        }


class Evaluator:
    """Evaluates model quality through game play and metrics.

    Supports:
    - Playing against random baseline
    - Playing against another model
    - Measuring policy agreement with MCTS
    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        mcts_config: MCTSConfig | None = None,
        device: torch.device | str = "cpu",
        board_sizes: list[int] | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            mcts_config: MCTS configuration.
            device: Evaluation device.
            board_sizes: Board sizes to evaluate on.

        """
        self.model = model
        self.device = torch.device(device)
        self.board_sizes = board_sizes or [9, 13, 19]

        # MCTS configuration
        self._mcts_kwargs = {}
        if mcts_config is not None:
            self._mcts_kwargs = {
                "n_simulations": mcts_config.n_simulations,
                "c_puct": mcts_config.c_puct,
                "dirichlet_alpha": mcts_config.dirichlet_alpha,
                "dirichlet_epsilon": mcts_config.dirichlet_epsilon,
            }

        # Neural evaluator
        self.neural_evaluator = FNetEvaluator(
            model,
            device=self.device,
            use_fast_path=True,
        )

    def evaluate_vs_random(
        self,
        n_games: int = 50,
        board_size: int | None = None,
    ) -> EvaluationResult:
        """Evaluate model against random player.

        Args:
            n_games: Number of games to play.
            board_size: Board size (random if None).

        Returns:
            Evaluation results.

        """
        wins = 0
        losses = 0
        draws = 0
        total_moves = 0

        self.model.eval()

        for game_idx in range(n_games):
            # Select board size
            size = board_size or random.choice(self.board_sizes)

            # Create players
            n_actions = size ** 2 + 1
            random_evaluator = RandomEvaluator(n_actions)

            # Alternate colors
            model_is_black = game_idx % 2 == 0

            # Play game
            outcome, moves = self._play_game(
                board_size=size,
                black_evaluator=self.neural_evaluator if model_is_black else random_evaluator,
                white_evaluator=random_evaluator if model_is_black else self.neural_evaluator,
            )

            total_moves += moves

            # Determine result from model's perspective
            if model_is_black:
                model_outcome = outcome
            else:
                model_outcome = -outcome

            if model_outcome > 0:
                wins += 1
            elif model_outcome < 0:
                losses += 1
            else:
                draws += 1

        win_rate = wins / n_games if n_games > 0 else 0.0
        avg_length = total_moves / n_games if n_games > 0 else 0.0

        result = EvaluationResult(
            win_rate=win_rate,
            n_games=n_games,
            wins=wins,
            losses=losses,
            draws=draws,
            avg_game_length=avg_length,
            metadata={"opponent": "random", "board_size": board_size},
        )

        logger.info(
            "evaluation_vs_random_complete",
            win_rate=f"{win_rate:.2%}",
            n_games=n_games,
            wins=wins,
            losses=losses,
            draws=draws,
        )

        return result

    def evaluate_vs_model(
        self,
        opponent_model: AlphaGalerkinModel,
        n_games: int = 50,
        board_size: int | None = None,
    ) -> EvaluationResult:
        """Evaluate model against another model.

        Args:
            opponent_model: Model to play against.
            n_games: Number of games to play.
            board_size: Board size (random if None).

        Returns:
            Evaluation results.

        """
        wins = 0
        losses = 0
        draws = 0
        total_moves = 0

        self.model.eval()
        opponent_model.eval()

        # Create opponent evaluator
        opponent_evaluator = FNetEvaluator(
            opponent_model,
            device=self.device,
            use_fast_path=True,
        )

        for game_idx in range(n_games):
            size = board_size or random.choice(self.board_sizes)

            # Alternate colors
            model_is_black = game_idx % 2 == 0

            outcome, moves = self._play_game(
                board_size=size,
                black_evaluator=self.neural_evaluator if model_is_black else opponent_evaluator,
                white_evaluator=opponent_evaluator if model_is_black else self.neural_evaluator,
            )

            total_moves += moves

            if model_is_black:
                model_outcome = outcome
            else:
                model_outcome = -outcome

            if model_outcome > 0:
                wins += 1
            elif model_outcome < 0:
                losses += 1
            else:
                draws += 1

        win_rate = wins / n_games if n_games > 0 else 0.0
        avg_length = total_moves / n_games if n_games > 0 else 0.0

        result = EvaluationResult(
            win_rate=win_rate,
            n_games=n_games,
            wins=wins,
            losses=losses,
            draws=draws,
            avg_game_length=avg_length,
            metadata={"opponent": "model"},
        )

        logger.info(
            "evaluation_vs_model_complete",
            win_rate=f"{win_rate:.2%}",
            n_games=n_games,
        )

        return result

    def _play_game(
        self,
        board_size: int,
        black_evaluator: Any,
        white_evaluator: Any,
        max_moves: int = 500,
    ) -> tuple[float, int]:
        """Play a single game between two evaluators.

        Args:
            board_size: Board size.
            black_evaluator: Evaluator for black player.
            white_evaluator: Evaluator for white player.
            max_moves: Maximum moves before draw.

        Returns:
            Tuple of (outcome, num_moves).
            Outcome: 1.0 = black wins, -1.0 = white wins, 0.0 = draw.

        """
        game = SimpleGoGame(board_size)

        black_mcts = MCTS(evaluator=black_evaluator, **self._mcts_kwargs)
        white_mcts = MCTS(evaluator=white_evaluator, **self._mcts_kwargs)

        move_count = 0

        while not game.is_terminal() and move_count < max_moves:
            # Select MCTS based on current player
            is_black = game.current_player == SimpleGoGame.BLACK
            mcts = black_mcts if is_black else white_mcts

            # Get action
            action = mcts.get_action(
                game,
                temperature=0.0,  # Deterministic play
                add_noise=False,
            )

            # Apply action
            if action == board_size ** 2:
                game.play_pass()
            else:
                row = action // board_size
                col = action % board_size
                if not game.play(row, col):
                    game.play_pass()

            # Advance MCTS trees
            black_mcts.advance(action)
            white_mcts.advance(action)

            move_count += 1

        # Determine outcome
        if game.is_terminal():
            winner = game.get_winner()
            # Winner from black's perspective
            if game.current_player == SimpleGoGame.BLACK:
                return -float(winner), move_count
            return float(winner), move_count

        # Max moves reached - score as draw
        return 0.0, move_count

    def measure_policy_agreement(
        self,
        n_positions: int = 100,
        board_size: int = 9,
    ) -> float:
        """Measure agreement between raw policy and MCTS policy.

        Args:
            n_positions: Number of positions to evaluate.
            board_size: Board size.

        Returns:
            Agreement rate (0.0 to 1.0).

        """
        self.model.eval()

        game = SimpleGoGame(board_size)
        mcts = MCTS(evaluator=self.neural_evaluator, **self._mcts_kwargs)

        agreements = 0
        total = 0

        # Generate random positions and check agreement
        for _ in range(n_positions):
            # Play random moves to get to a position
            game.reset()
            mcts.reset()

            n_random_moves = random.randint(0, board_size * board_size // 4)
            legal_actions = game.get_legal_actions()

            for _ in range(n_random_moves):
                if not legal_actions or game.is_terminal():
                    break
                action = random.choice(legal_actions)
                if action == board_size ** 2:
                    game.play_pass()
                else:
                    row = action // board_size
                    col = action % board_size
                    game.play(row, col)
                legal_actions = game.get_legal_actions()

            if game.is_terminal():
                continue

            # Get raw policy
            state = game.get_state()
            legal_actions = game.get_legal_actions()
            raw_result = self.neural_evaluator.evaluate(state, legal_actions)
            raw_action = int(np.argmax(raw_result.policy))

            # Get MCTS policy
            mcts_dist = mcts.search(game, add_noise=False)
            mcts_action = max(mcts_dist.keys(), key=lambda a: mcts_dist[a])

            if raw_action == mcts_action:
                agreements += 1
            total += 1

        agreement_rate = agreements / total if total > 0 else 0.0

        logger.info(
            "policy_agreement_measured",
            agreement_rate=f"{agreement_rate:.2%}",
            n_positions=total,
        )

        return agreement_rate


def quick_evaluate(
    model: AlphaGalerkinModel,
    n_games: int = 10,
    board_size: int = 9,
    device: str = "cpu",
) -> dict[str, Any]:
    """Quick evaluation for training loop.

    Args:
        model: Model to evaluate.
        n_games: Number of games.
        board_size: Board size.
        device: Device.

    Returns:
        Evaluation metrics dictionary.

    """
    evaluator = Evaluator(
        model=model,
        device=device,
        board_sizes=[board_size],
    )

    result = evaluator.evaluate_vs_random(n_games=n_games, board_size=board_size)
    return result.to_dict()
