"""Engine evaluator adapter for MCTS integration.

Bridges external chess engines to the MCTS Evaluator protocol,
enabling engines like Stockfish to serve as opponents in matches
and evaluation pipelines.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from src.engines.config import UCIConfig
from src.engines.protocol import BaseEngine, EngineInfo
from src.games.fen import state_to_fen

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.games.interface import GameInterface
    from src.games.state import GameState
    from src.mcts.evaluator import EvaluationResult

logger = structlog.get_logger(__name__)

# Default scale for converting centipawn scores to [-1, 1] value
# tanh(score_cp / CP_SCALE) gives values in [-1, 1]
# 300 cp ≈ tanh(1) ≈ 0.76, 100 cp ≈ 0.32
DEFAULT_CP_SCALE: float = 300.0

# Value assigned for forced mate
MATE_VALUE: float = 0.999


class EngineEvaluator:
    """Wraps a BaseEngine as an MCTS-compatible evaluator.

    Translates between AlphaGalerkin's internal state representation
    and UCI engine communication, producing policy and value estimates
    compatible with the MCTS Evaluator protocol.

    The policy is a one-hot vector over the best move (engines don't
    provide a full probability distribution). The value is derived
    from the engine's centipawn score via tanh scaling.

    Args:
        engine: Started engine instance.
        game: Game implementation for move translation.
        config: UCI configuration for search parameters.
        cp_scale: Scale factor for centipawn-to-value conversion.

    """

    def __init__(
        self,
        engine: BaseEngine,
        game: GameInterface,
        config: UCIConfig,
        cp_scale: float = DEFAULT_CP_SCALE,
    ) -> None:
        self._engine = engine
        self._game = game
        self._config = config
        self._cp_scale = cp_scale
        self._current_state: GameState | None = None

    def set_state(self, state: GameState) -> None:
        """Set the current game state for evaluation.

        Must be called before evaluate() when the adapter
        needs to know the full GameState (not just the tensor).

        Args:
            state: Current chess game state.

        """
        self._current_state = state

    def evaluate(
        self,
        state: NDArray[np.float32],
        legal_actions: list[int],
    ) -> EvaluationResult:
        """Evaluate position using the external engine.

        Requires set_state() to have been called with the matching
        GameState, since the numpy tensor alone cannot be converted
        back to FEN.

        Args:
            state: Game state tensor (unused, kept for protocol compat).
            legal_actions: List of legal action indices.

        Returns:
            EvaluationResult with one-hot policy and engine value.

        """
        from src.mcts.evaluator import EvaluationResult as MCTSEvalResult

        if self._current_state is None:
            # Fallback: uniform policy, neutral value
            logger.warning("engine_eval_no_state")
            n_actions = self._game.action_space_size
            policy = np.zeros(n_actions, dtype=np.float32)
            if legal_actions:
                uniform_prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    policy[action] = uniform_prob
            return MCTSEvalResult(policy=policy, value=0.0)

        # Convert state to FEN
        fen = state_to_fen(self._current_state)

        # Query engine
        self._engine.set_position(fen)

        go_kwargs: dict[str, Any] = {}
        if self._config.depth_limit is not None:
            go_kwargs["depth"] = self._config.depth_limit
        if self._config.nodes_limit is not None:
            go_kwargs["nodes"] = self._config.nodes_limit
        if self._config.movetime_ms is not None:
            go_kwargs["movetime"] = self._config.movetime_ms

        best_move_uci, info = self._engine.go(**go_kwargs)

        # Convert best move to action index
        best_action = self._game.string_to_action(best_move_uci, self._current_state)

        # Build one-hot policy
        n_actions = self._game.action_space_size
        policy = np.zeros(n_actions, dtype=np.float32)

        if best_action is not None and best_action in legal_actions:
            policy[best_action] = 1.0
        elif legal_actions:
            # Fallback if engine move not in legal moves
            logger.warning(
                "engine_move_not_legal",
                move=best_move_uci,
                action=best_action,
            )
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                policy[action] = uniform_prob

        # Convert engine score to value in [-1, 1]
        value = self._score_to_value(info)

        return MCTSEvalResult(policy=policy, value=value)

    def evaluate_batch(
        self,
        states: list[NDArray[np.float32]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        """Evaluate a batch of states sequentially.

        UCI engines are inherently single-threaded for search,
        so batch evaluation is sequential.

        Args:
            states: List of state tensors.
            legal_actions_batch: Legal actions per state.

        Returns:
            List of evaluation results.

        """
        return [self.evaluate(s, la) for s, la in zip(states, legal_actions_batch, strict=True)]

    def _score_to_value(self, info: EngineInfo) -> float:
        """Convert engine score to value in [-1, 1].

        Uses tanh(score_cp / cp_scale) for centipawn scores,
        and ±MATE_VALUE for forced mates.

        Args:
            info: Engine search info.

        Returns:
            Value estimate in [-1, 1].

        """
        if "score_mate" in info:
            mate_dist = info["score_mate"]
            # Positive = winning mate, negative = losing mate
            return MATE_VALUE if mate_dist > 0 else -MATE_VALUE

        if "score_cp" in info:
            cp = info["score_cp"]
            return math.tanh(cp / self._cp_scale)

        # No score available — neutral
        return 0.0
