"""Game review and analysis.

Provides:
- Move-by-move analysis
- Game quality assessment
- Turning point detection
- SGF annotation generation
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.analysis.config import (
    AnalysisConfig,
    AnnotationLevel,
    MoveClassification,
)
from src.analysis.evaluator import EvaluationResult, PositionEvaluator
from src.analysis.go_adapter import reconstruct_board
from src.games.go import GoGame

if TYPE_CHECKING:
    from src.games.sgf.node import SGFGameTree


@dataclass
class MoveAnalysis:
    """Analysis of a single move.

    Attributes:
        move_number: Move number in game.
        color: Move color ("B" or "W").
        move: Move coordinates (x, y) or None for pass.
        evaluation_before: Position evaluation before the move.
        evaluation_after: Position evaluation after the move.
        classification: Move quality classification.
        win_rate_change: Change in win rate from this move.
        best_move: Best move according to analysis.
        is_best: Whether the played move is the best.
        alternatives: Alternative moves with analysis.
        comment: Generated comment for this move.

    """

    move_number: int
    color: str
    move: tuple[int, int] | None
    evaluation_before: EvaluationResult | None = None
    evaluation_after: EvaluationResult | None = None
    classification: MoveClassification = MoveClassification.NEUTRAL
    win_rate_change: float = 0.0
    best_move: tuple[int, int] | None = None
    is_best: bool = False
    alternatives: list[tuple[tuple[int, int], float, str]] = field(default_factory=list)
    comment: str = ""

    @property
    def is_pass(self) -> bool:
        """Check if this is a pass move."""
        return self.move is None

    @property
    def is_mistake(self) -> bool:
        """Check if this move is a mistake or worse."""
        return self.classification in (
            MoveClassification.MISTAKE,
            MoveClassification.BLUNDER,
        )

    @property
    def is_inaccuracy(self) -> bool:
        """Check if this is an inaccuracy."""
        return self.classification == MoveClassification.INACCURACY

    @property
    def is_good(self) -> bool:
        """Check if this is a good or excellent move."""
        return self.classification in (
            MoveClassification.GOOD,
            MoveClassification.EXCELLENT,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "move_number": self.move_number,
            "color": self.color,
            "move": list(self.move) if self.move else None,
            "classification": self.classification.value,
            "win_rate_change": self.win_rate_change,
            "best_move": list(self.best_move) if self.best_move else None,
            "is_best": self.is_best,
            "comment": self.comment,
        }


@dataclass
class GameAnalysis:
    """Complete analysis of a game.

    Attributes:
        move_analyses: Analysis for each move.
        black_stats: Statistics for black player.
        white_stats: Statistics for white player.
        turning_points: Key moments in the game.
        opening_quality: Opening phase assessment.
        endgame_quality: Endgame phase assessment.
        timestamp: When analysis was performed.
        config_hash: Configuration hash for reproducibility.

    """

    move_analyses: list[MoveAnalysis] = field(default_factory=list)
    black_stats: dict[str, Any] = field(default_factory=dict)
    white_stats: dict[str, Any] = field(default_factory=dict)
    turning_points: list[int] = field(default_factory=list)
    opening_quality: float = 0.0
    endgame_quality: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str = ""

    @property
    def total_moves(self) -> int:
        """Total number of moves analyzed."""
        return len(self.move_analyses)

    @property
    def black_mistakes(self) -> list[MoveAnalysis]:
        """Get black's mistakes."""
        return [m for m in self.move_analyses if m.color == "B" and m.is_mistake]

    @property
    def white_mistakes(self) -> list[MoveAnalysis]:
        """Get white's mistakes."""
        return [m for m in self.move_analyses if m.color == "W" and m.is_mistake]

    def get_move_at(self, move_number: int) -> MoveAnalysis | None:
        """Get analysis for a specific move number.

        Args:
            move_number: Move number to get.

        Returns:
            MoveAnalysis or None if not found.

        """
        for analysis in self.move_analyses:
            if analysis.move_number == move_number:
                return analysis
        return None

    def get_moves_by_classification(
        self,
        classification: MoveClassification,
    ) -> list[MoveAnalysis]:
        """Get moves with specific classification.

        Args:
            classification: Classification to filter by.

        Returns:
            List of matching move analyses.

        """
        return [m for m in self.move_analyses if m.classification == classification]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_moves": self.total_moves,
            "move_analyses": [m.to_dict() for m in self.move_analyses],
            "black_stats": self.black_stats,
            "white_stats": self.white_stats,
            "turning_points": self.turning_points,
            "opening_quality": self.opening_quality,
            "endgame_quality": self.endgame_quality,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
        }


class GameReviewer:
    """Reviews games and provides move-by-move analysis.

    Features:
    - Full game analysis
    - Turning point detection
    - Quality statistics
    - SGF annotation generation
    """

    def __init__(
        self,
        evaluator: PositionEvaluator | None = None,
        config: AnalysisConfig | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
        model_evaluator: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize game reviewer.

        Args:
            evaluator: Position evaluator instance.
            config: Analysis configuration.
            logger: Optional structured logger.
            model_evaluator: Optional ``board_state -> (value, policy)`` callable.
                When provided it is attached to the evaluator, giving real model-backed
                evaluations instead of the uniform dummy fallback.

        """
        self.config = config or AnalysisConfig()
        self._logger = logger or structlog.get_logger(__name__)
        self._evaluator = evaluator or PositionEvaluator(config=self.config)
        self._game = GoGame()

        # Resolve a model evaluator: explicit argument wins, otherwise build one
        # from the configured checkpoint. If neither is available we keep the
        # uniform dummy fallback but log it so the no-signal case is not silent.
        if model_evaluator is not None:
            self._evaluator.set_model_evaluator(model_evaluator)
        elif self._evaluator._model_evaluator is None and self.config.model_checkpoint_path:
            self._wire_checkpoint_evaluator(self.config.model_checkpoint_path)

        if self._evaluator._model_evaluator is None:
            self._logger.warning(
                "review_no_model_evaluator",
                detail="no model wired; evaluations use uniform dummy (win_rate=0.5)",
            )

    def _wire_checkpoint_evaluator(self, checkpoint_path: str) -> None:
        """Attach a checkpoint-backed model evaluator, falling back on failure."""
        from src.analysis.go_adapter import build_checkpoint_model_evaluator

        try:
            evaluator_fn = build_checkpoint_model_evaluator(
                checkpoint_path,
                device=self.config.device,
                temperature=self.config.evaluator_temperature,
            )
        except Exception as exc:
            self._logger.warning(
                "review_checkpoint_load_failed",
                checkpoint_path=checkpoint_path,
                error=str(exc),
            )
            return
        self._evaluator.set_model_evaluator(evaluator_fn)
        self._logger.info("review_model_wired", checkpoint_path=checkpoint_path)

    def review_game(
        self,
        moves: list[tuple[str, int, int]],
        board_size: int = 19,
        game_tree: SGFGameTree | None = None,
    ) -> GameAnalysis:
        """Review a complete game.

        Args:
            moves: List of (color, x, y) tuples.
            board_size: Board size.
            game_tree: Optional SGF game tree for context.

        Returns:
            Complete GameAnalysis.

        """
        self._logger.info(
            "review_started",
            n_moves=len(moves),
            board_size=board_size,
        )

        analysis = GameAnalysis(config_hash=self.config.compute_hash())

        # Track game state
        black_total_loss = 0.0
        white_total_loss = 0.0
        black_mistake_count = 0
        white_mistake_count = 0
        black_excellent_count = 0
        white_excellent_count = 0

        # Analyze each move
        for i, (color, x, y) in enumerate(moves):
            move_number = i + 1
            move = (x, y) if x >= 0 and y >= 0 else None

            # Analyze this move
            move_analysis = self._analyze_move(
                move_number=move_number,
                color=color,
                move=move,
                moves_so_far=moves[:i],
                board_size=board_size,
            )

            analysis.move_analyses.append(move_analysis)

            # Track statistics
            if color == "B":
                black_total_loss += abs(move_analysis.win_rate_change)
                if move_analysis.is_mistake:
                    black_mistake_count += 1
                if move_analysis.classification == MoveClassification.EXCELLENT:
                    black_excellent_count += 1
            else:
                white_total_loss += abs(move_analysis.win_rate_change)
                if move_analysis.is_mistake:
                    white_mistake_count += 1
                if move_analysis.classification == MoveClassification.EXCELLENT:
                    white_excellent_count += 1

            # Detect turning points
            if abs(move_analysis.win_rate_change) > self.config.mistake_threshold:
                analysis.turning_points.append(move_number)

        # Compile statistics
        n_black_moves = sum(1 for m in analysis.move_analyses if m.color == "B")
        n_white_moves = sum(1 for m in analysis.move_analyses if m.color == "W")

        analysis.black_stats = {
            "total_moves": n_black_moves,
            "average_loss": black_total_loss / n_black_moves if n_black_moves > 0 else 0,
            "mistakes": black_mistake_count,
            "excellent_moves": black_excellent_count,
            "accuracy": 1 - (black_total_loss / n_black_moves) if n_black_moves > 0 else 1,
        }

        analysis.white_stats = {
            "total_moves": n_white_moves,
            "average_loss": white_total_loss / n_white_moves if n_white_moves > 0 else 0,
            "mistakes": white_mistake_count,
            "excellent_moves": white_excellent_count,
            "accuracy": 1 - (white_total_loss / n_white_moves) if n_white_moves > 0 else 1,
        }

        # Assess game phases
        analysis.opening_quality = self._assess_opening(analysis.move_analyses[:30])
        analysis.endgame_quality = self._assess_endgame(analysis.move_analyses[-30:])

        self._logger.info(
            "review_completed",
            n_moves=len(moves),
            turning_points=len(analysis.turning_points),
        )

        return analysis

    def _analyze_move(
        self,
        move_number: int,
        color: str,
        move: tuple[int, int] | None,
        moves_so_far: list[tuple[str, int, int]],
        board_size: int,
    ) -> MoveAnalysis:
        """Analyze a single move.

        Args:
            move_number: Move number.
            color: Move color.
            move: Move coordinates or None for pass.
            moves_so_far: Moves played before this one.
            board_size: Board size.

        Returns:
            MoveAnalysis for this move.

        """
        # Reconstruct a capture-correct board state for evaluation.
        board_state = self._create_board_state(moves_so_far, board_size)

        # Evaluate position before move
        eval_before = self._evaluator.evaluate(
            board_state,
            board_size=board_size,
        )

        # Default values
        classification = MoveClassification.NEUTRAL
        win_rate_change = 0.0
        is_best = False
        best_move = eval_before.best_move

        # Compare played move to best
        if move is not None and not self._is_pass(move):
            classification, loss = self._evaluator.compare_moves(eval_before, move)
            win_rate_change = -loss if color == "B" else loss
            is_best = move == best_move

        # Generate alternatives
        alternatives = []
        if self.config.include_variations and not is_best:
            for alt_move, prob in eval_before.best_moves[: self.config.max_variations]:
                if alt_move != move:
                    alt_desc = f"Better: {self._format_move(alt_move, board_size)}"
                    alternatives.append((alt_move, prob, alt_desc))

        # Generate comment
        comment = self._generate_comment(
            move_number, color, move, classification, best_move, board_size
        )

        return MoveAnalysis(
            move_number=move_number,
            color=color,
            move=move,
            evaluation_before=eval_before,
            classification=classification,
            win_rate_change=win_rate_change,
            best_move=best_move,
            is_best=is_best,
            alternatives=alternatives,
            comment=comment,
        )

    def _create_board_state(
        self,
        moves: list[tuple[str, int, int]],
        board_size: int,
    ) -> list[list[int]]:
        """Create a capture-correct board state from a move list.

        Delegates to :func:`src.analysis.go_adapter.reconstruct_board`, which replays
        the moves through the real Go engine so captured stones are removed and ko /
        superko are respected (the previous naive replay left captured stones on the
        board). The returned marks (``1`` black, ``2`` white, ``0`` empty, indexed
        ``board[y][x]``) preserve the existing downstream contract.

        Args:
            moves: List of ``(color, x, y)`` moves played so far.
            board_size: Board size.

        Returns:
            2D list representing the board state.

        """
        return reconstruct_board(
            moves,
            board_size,
            game=self._game,
            logger=self._logger,
        )

    def _is_pass(self, move: tuple[int, int]) -> bool:
        """Check if move is a pass."""
        return move[0] < 0 or move[1] < 0

    def _format_move(self, move: tuple[int, int], board_size: int) -> str:
        """Format move as human-readable string.

        Args:
            move: Move coordinates.
            board_size: Board size.

        Returns:
            Formatted move string (e.g., "D4").

        """
        x, y = move
        col = chr(ord("A") + x + (1 if x >= 8 else 0))  # Skip 'I'
        row = board_size - y
        return f"{col}{row}"

    def _generate_comment(
        self,
        move_number: int,
        color: str,
        move: tuple[int, int] | None,
        classification: MoveClassification,
        best_move: tuple[int, int] | None,
        board_size: int,
    ) -> str:
        """Generate comment for move.

        Args:
            move_number: Move number.
            color: Move color.
            move: Move coordinates.
            classification: Move classification.
            best_move: Best move.
            board_size: Board size.

        Returns:
            Generated comment string.

        """
        level = self.config.annotation_level

        # Minimal level: only critical moves
        if level == AnnotationLevel.MINIMAL:
            if classification == MoveClassification.BLUNDER:
                return f"Blunder! Better was {self._format_move(best_move, board_size)}"
            return ""

        # Normal level: notable moves
        if level == AnnotationLevel.NORMAL:
            if classification == MoveClassification.BLUNDER:
                return f"Blunder! Better was {self._format_move(best_move, board_size)}"
            elif classification == MoveClassification.MISTAKE:
                return f"Mistake. Consider {self._format_move(best_move, board_size)}"
            elif classification == MoveClassification.EXCELLENT:
                return "Excellent move!"
            return ""

        # Detailed level: all moves
        best_move_str = self._format_move(best_move, board_size) if best_move else None
        comments = {
            MoveClassification.EXCELLENT: "Excellent move!",
            MoveClassification.GOOD: "Good move.",
            MoveClassification.NEUTRAL: "",
            MoveClassification.INACCURACY: (
                f"Inaccuracy. Better: {best_move_str}" if best_move else "Inaccuracy."
            ),
            MoveClassification.MISTAKE: (
                f"Mistake. Better: {best_move_str}" if best_move else "Mistake."
            ),
            MoveClassification.BLUNDER: (
                f"Blunder! Better: {best_move_str}" if best_move else "Blunder!"
            ),
        }

        return comments.get(classification, "")

    def _assess_opening(self, moves: list[MoveAnalysis]) -> float:
        """Assess opening quality.

        Args:
            moves: Opening moves to assess.

        Returns:
            Quality score (0.0 to 1.0).

        """
        if not moves:
            return 0.5

        excellent_count = sum(1 for m in moves if m.classification == MoveClassification.EXCELLENT)
        good_count = sum(1 for m in moves if m.classification == MoveClassification.GOOD)
        mistake_count = sum(1 for m in moves if m.is_mistake)

        score = 0.5
        score += (excellent_count / len(moves)) * 0.3
        score += (good_count / len(moves)) * 0.2
        score -= (mistake_count / len(moves)) * 0.4

        return max(0.0, min(1.0, score))

    def _assess_endgame(self, moves: list[MoveAnalysis]) -> float:
        """Assess endgame quality.

        Args:
            moves: Endgame moves to assess.

        Returns:
            Quality score (0.0 to 1.0).

        """
        return self._assess_opening(moves)  # Same logic for now

    def annotate_sgf(
        self,
        game_tree: SGFGameTree,
        analysis: GameAnalysis,
    ) -> None:
        """Add analysis annotations to SGF game tree.

        Args:
            game_tree: SGF game tree to annotate.
            analysis: Game analysis to add.

        """
        from src.games.sgf.converter import SGFConverter

        converter = SGFConverter()

        # Iterate through moves and add comments
        for node in game_tree.mainline():
            if node.has_move:
                move_number = node.move_number
                move_analysis = analysis.get_move_at(move_number)

                if move_analysis and move_analysis.comment:
                    existing = node.get_comment()
                    if existing:
                        node.set_comment(f"{existing}\n\n{move_analysis.comment}")
                    else:
                        node.set_comment(move_analysis.comment)

                    # Add markers for alternatives
                    if move_analysis.alternatives and self.config.include_variations:
                        for alt_move, _prob, _ in move_analysis.alternatives[:3]:
                            converter.add_triangle(node, alt_move[0], alt_move[1])


def create_game_reviewer(
    mode: str = "standard",
    model_evaluator: Callable[..., Any] | None = None,
    **config_kwargs: Any,
) -> GameReviewer:
    """Factory function to create game reviewer.

    Args:
        mode: Analysis mode.
        model_evaluator: Optional ``board_state -> (value, policy)`` callable wired
            into the evaluator for real model-backed evaluations. When omitted, a
            checkpoint-backed evaluator is built if ``model_checkpoint_path`` is set
            in the config; otherwise the uniform dummy fallback is used.
        **config_kwargs: Additional config options.

    Returns:
        Configured GameReviewer.

    """
    from src.analysis.config import create_analysis_config
    from src.analysis.evaluator import PositionEvaluator

    config = create_analysis_config(mode=mode, **config_kwargs)
    evaluator = PositionEvaluator(config=config)

    return GameReviewer(
        evaluator=evaluator,
        config=config,
        model_evaluator=model_evaluator,
    )
