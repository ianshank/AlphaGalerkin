"""Position evaluation for game analysis.

Provides:
- Single position evaluation
- Batch evaluation
- Move ranking and policy extraction
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from src.analysis.config import AnalysisConfig, MoveClassification

if TYPE_CHECKING:
    pass


@dataclass
class EvaluationResult:
    """Result of position evaluation.

    Attributes:
        win_rate: Expected win rate (0.0 to 1.0).
        best_moves: List of (move, probability) tuples, sorted by probability.
        policy: Full policy distribution over moves.
        value: Raw value output from network.
        mcts_visits: Visit counts from MCTS if available.
        depth: Search depth achieved.

    """

    win_rate: float
    best_moves: list[tuple[tuple[int, int], float]] = field(default_factory=list)
    policy: dict[tuple[int, int], float] = field(default_factory=dict)
    value: float = 0.0
    mcts_visits: dict[tuple[int, int], int] = field(default_factory=dict)
    depth: int = 0
    confidence: float = 1.0

    @property
    def best_move(self) -> tuple[int, int] | None:
        """Get the single best move."""
        if self.best_moves:
            return self.best_moves[0][0]
        return None

    @property
    def best_move_probability(self) -> float:
        """Get probability of best move."""
        if self.best_moves:
            return self.best_moves[0][1]
        return 0.0

    def get_move_probability(self, move: tuple[int, int]) -> float:
        """Get probability for a specific move.

        Args:
            move: Move coordinates (x, y).

        Returns:
            Move probability or 0.0 if not found.

        """
        return self.policy.get(move, 0.0)

    def get_move_rank(self, move: tuple[int, int]) -> int | None:
        """Get rank of a move in best moves list.

        Args:
            move: Move coordinates (x, y).

        Returns:
            Rank (0-indexed) or None if not in top moves.

        """
        for i, (m, _) in enumerate(self.best_moves):
            if m == move:
                return i
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "win_rate": self.win_rate,
            "best_moves": [{"move": list(m), "probability": p} for m, p in self.best_moves],
            "value": self.value,
            "depth": self.depth,
            "confidence": self.confidence,
        }


class LRUCache:
    """Simple LRU cache for position evaluations."""

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum number of entries.

        """
        self._cache: OrderedDict[str, EvaluationResult] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> EvaluationResult | None:
        """Get cached result.

        Args:
            key: Cache key.

        Returns:
            Cached result or None.

        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: EvaluationResult) -> None:
        """Store result in cache.

        Args:
            key: Cache key.
            value: Result to cache.

        """
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class PositionEvaluator:
    """Evaluates Go positions using the model.

    Provides:
    - Single position evaluation
    - Batch evaluation for efficiency
    - Move ranking with policy probabilities
    - Optional caching for repeated positions
    """

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        model_evaluator: Callable[..., Any] | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize position evaluator.

        Args:
            config: Analysis configuration.
            model_evaluator: Optional model evaluation function.
            logger: Optional structured logger.

        """
        self.config = config or AnalysisConfig()
        self._model_evaluator = model_evaluator
        self._logger = logger or structlog.get_logger(__name__)

        # Initialize cache if enabled
        self._cache: LRUCache | None = None
        if self.config.cache_evaluations:
            self._cache = LRUCache(self.config.max_cache_size)

        self._evaluation_count = 0

    def set_model_evaluator(
        self,
        evaluator: Callable[..., Any],
    ) -> None:
        """Set the model evaluation function.

        Args:
            evaluator: Function that takes board state and returns (value, policy).

        """
        self._model_evaluator = evaluator

    def evaluate(
        self,
        board_state: Any,
        board_size: int = 19,
        legal_moves: list[tuple[int, int]] | None = None,
        use_cache: bool = True,
    ) -> EvaluationResult:
        """Evaluate a single position.

        Args:
            board_state: Board state representation.
            board_size: Size of the board.
            legal_moves: Optional list of legal moves.
            use_cache: Whether to use cache.

        Returns:
            EvaluationResult with win rate and best moves.

        """
        # Check cache
        cache_key = None
        if use_cache and self._cache is not None:
            cache_key = self._compute_cache_key(board_state)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Evaluate position
        result = self._evaluate_position(board_state, board_size, legal_moves)
        self._evaluation_count += 1

        # Store in cache
        if cache_key and self._cache is not None:
            self._cache.put(cache_key, result)

        return result

    def evaluate_batch(
        self,
        board_states: list[Any],
        board_sizes: list[int] | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple positions in batch.

        Args:
            board_states: List of board states.
            board_sizes: Optional list of board sizes.

        Returns:
            List of EvaluationResults.

        """
        if board_sizes is None:
            board_sizes = [19] * len(board_states)

        results = []
        for state, size in zip(board_states, board_sizes, strict=False):
            results.append(self.evaluate(state, size))

        return results

    def _evaluate_position(
        self,
        board_state: Any,
        board_size: int,
        legal_moves: list[tuple[int, int]] | None,
    ) -> EvaluationResult:
        """Internal position evaluation.

        Args:
            board_state: Board state representation.
            board_size: Size of the board.
            legal_moves: Optional list of legal moves.

        Returns:
            EvaluationResult.

        """
        if self._model_evaluator is None:
            # Return dummy result if no model
            return self._create_dummy_result(board_size, legal_moves)

        try:
            # Call model evaluator
            value, policy = self._model_evaluator(board_state)

            # Convert to evaluation result
            return self._process_model_output(value, policy, board_size, legal_moves)
        except Exception as e:
            self._logger.warning("evaluation_failed", error=str(e))
            return self._create_dummy_result(board_size, legal_moves)

    def _process_model_output(
        self,
        value: float | Any,
        policy: Any,
        board_size: int,
        legal_moves: list[tuple[int, int]] | None,
    ) -> EvaluationResult:
        """Process model output into EvaluationResult.

        Args:
            value: Model value output.
            policy: Model policy output.
            board_size: Board size.
            legal_moves: Legal moves.

        Returns:
            EvaluationResult.

        """
        # Convert value to float
        if hasattr(value, "item"):
            value = value.item()

        # Convert to win rate (value is typically in [-1, 1])
        win_rate = (float(value) + 1) / 2

        # Process policy
        policy_dict: dict[tuple[int, int], float] = {}
        best_moves: list[tuple[tuple[int, int], float]] = []

        # Convert policy tensor/array to dictionary
        if hasattr(policy, "numpy"):
            policy_np = policy.numpy()
        elif hasattr(policy, "cpu"):
            policy_np = policy.cpu().numpy()
        else:
            policy_np = policy

        # Flatten if needed
        policy_flat = policy_np.flatten() if hasattr(policy_np, "flatten") else policy_np

        # Build policy dictionary
        for i, prob in enumerate(policy_flat):
            if i < board_size * board_size:
                x = i % board_size
                y = i // board_size
                move = (x, y)

                # Filter by legal moves if provided
                if legal_moves is None or move in legal_moves:
                    policy_dict[move] = float(prob)

        # Sort by probability
        sorted_moves = sorted(
            policy_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        best_moves = list(sorted_moves[: self.config.max_variations + 1])

        return EvaluationResult(
            win_rate=win_rate,
            best_moves=best_moves,
            policy=policy_dict,
            value=float(value),
            depth=1,
        )

    def _create_dummy_result(
        self,
        board_size: int,
        legal_moves: list[tuple[int, int]] | None,
    ) -> EvaluationResult:
        """Create a dummy result when model is not available.

        Args:
            board_size: Board size.
            legal_moves: Legal moves.

        Returns:
            Dummy EvaluationResult.

        """
        # Uniform distribution over legal moves
        if legal_moves:
            n_moves = len(legal_moves)
            prob = 1.0 / n_moves if n_moves > 0 else 0.0
            policy = dict.fromkeys(legal_moves, prob)
            best_moves = [(m, prob) for m in legal_moves[:3]]
        else:
            policy = {}
            best_moves = []

        return EvaluationResult(
            win_rate=0.5,
            best_moves=best_moves,
            policy=policy,
            value=0.0,
            depth=0,
            confidence=0.0,
        )

    def _compute_cache_key(self, board_state: Any) -> str:
        """Compute cache key for board state.

        Args:
            board_state: Board state.

        Returns:
            Cache key string.

        """
        import hashlib

        # Convert to hashable representation
        if hasattr(board_state, "tobytes"):
            return board_state.tobytes().hex()[:32]
        elif isinstance(board_state, list):
            # Convert nested list to string and hash it
            flat = str(board_state)
            return hashlib.md5(flat.encode()).hexdigest()[:32]
        else:
            try:
                return str(hash(board_state))
            except TypeError:
                return str(id(board_state))

    def compare_moves(
        self,
        evaluation: EvaluationResult,
        played_move: tuple[int, int],
    ) -> tuple[MoveClassification, float]:
        """Compare played move to best move.

        Args:
            evaluation: Position evaluation.
            played_move: Move that was played.

        Returns:
            Tuple of (classification, win rate loss).

        """
        best_win_rate = evaluation.win_rate
        played_prob = evaluation.get_move_probability(played_move)

        # Estimate win rate of played move
        # If it's the best move, no loss
        if evaluation.best_move == played_move:
            return MoveClassification.EXCELLENT, 0.0

        # Otherwise, estimate loss based on probability ranking
        played_rank = evaluation.get_move_rank(played_move)

        if played_rank is not None and played_rank < len(evaluation.best_moves):
            # Use probability ratio as win rate loss estimate
            best_prob = evaluation.best_move_probability
            if best_prob > 0:
                ratio = played_prob / best_prob
                win_rate_loss = (1 - ratio) * 0.1  # Scale down
            else:
                win_rate_loss = 0.05
        else:
            # Move not in top moves, use higher penalty
            win_rate_loss = 0.15

        classification = self.config.classify_move(
            best_win_rate - win_rate_loss,
            best_win_rate,
        )

        return classification, win_rate_loss

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache) if self._cache else 0

    @property
    def evaluation_count(self) -> int:
        """Get total evaluation count."""
        return self._evaluation_count


def create_position_evaluator(
    mode: str = "standard",
    model_evaluator: Callable[..., Any] | None = None,
    **config_kwargs: Any,
) -> PositionEvaluator:
    """Factory function to create position evaluator.

    Args:
        mode: Analysis mode.
        model_evaluator: Optional model evaluation function.
        **config_kwargs: Additional config options.

    Returns:
        Configured PositionEvaluator.

    """
    from src.analysis.config import create_analysis_config

    config = create_analysis_config(mode=mode, **config_kwargs)
    return PositionEvaluator(config=config, model_evaluator=model_evaluator)
