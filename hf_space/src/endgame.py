"""Endgame detection for Go.

Provides utilities to detect when the AI should pass to end the game,
particularly when the human player passes in an endgame situation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from config.board import EndgameConfig

    from src.tools.gtp import SimpleGoGame

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class EndgameAnalysis:
    """Result of endgame analysis.

    Attributes:
        fill_ratio: Percentage of board that is filled (0.0 to 1.0).
        empty_count: Number of empty intersections.
        total_positions: Total number of board positions.
        human_just_passed: Whether the human player just passed.
        current_passes: Number of consecutive passes in game state.
        should_ai_pass: Whether the AI should pass.
        reason: Human-readable reason for the decision.

    """

    fill_ratio: float
    empty_count: int
    total_positions: int
    human_just_passed: bool
    current_passes: int
    should_ai_pass: bool
    reason: str


class EndgameDetector:
    """Detects endgame scenarios for Go.

    Uses configurable heuristics to determine when the AI should
    automatically pass to end the game properly.

    Example:
        >>> from config.board import EndgameConfig
        >>> detector = EndgameDetector(EndgameConfig())
        >>> analysis = detector.analyze(game, human_just_passed=True)
        >>> if analysis.should_ai_pass:
        ...     game.play_pass()

    """

    def __init__(self, config: EndgameConfig) -> None:
        """Initialize detector with configuration.

        Args:
            config: Endgame detection configuration.

        """
        self.config = config
        self._logger = structlog.get_logger(__name__)

    def analyze(
        self,
        game: SimpleGoGame,
        human_just_passed: bool,
    ) -> EndgameAnalysis:
        """Analyze game state to determine if AI should pass.

        Args:
            game: Current game state.
            human_just_passed: Whether human just played a pass move.

        Returns:
            Analysis result with pass decision and reasoning.

        """
        # Calculate board fill metrics
        empty_count = int((game.board == game.EMPTY).sum())
        total = game.board_size**2
        fill_ratio = 1.0 - (empty_count / total) if total > 0 else 0.0

        should_pass = False
        reason = "game_continues"

        if not self.config.enable_auto_pass:
            reason = "auto_pass_disabled"
        elif not human_just_passed:
            reason = "human_did_not_pass"
        elif self.config.pass_on_consecutive and game.passes >= 1:
            # Human just passed and there's already a pass pending
            should_pass = True
            reason = "consecutive_pass_detection"
        elif fill_ratio >= self.config.fill_threshold:
            # Board is sufficiently full
            should_pass = True
            reason = "board_fill_threshold_reached"

        analysis = EndgameAnalysis(
            fill_ratio=fill_ratio,
            empty_count=empty_count,
            total_positions=total,
            human_just_passed=human_just_passed,
            current_passes=game.passes,
            should_ai_pass=should_pass,
            reason=reason,
        )

        self._logger.debug(
            "endgame_analysis",
            fill_ratio=round(fill_ratio, 3),
            empty_count=empty_count,
            passes=game.passes,
            should_pass=should_pass,
            reason=reason,
        )

        return analysis

    def should_override_to_pass(
        self,
        game: SimpleGoGame,
        mcts_action: int,
        human_just_passed: bool,
    ) -> bool:
        """Check if MCTS action should be overridden to pass.

        This is the primary entry point for determining whether
        the AI's chosen action should be replaced with a pass.

        Args:
            game: Current game state.
            mcts_action: Action chosen by MCTS.
            human_just_passed: Whether human just passed.

        Returns:
            True if mcts_action should be overridden to pass.

        """
        pass_action = game.board_size**2

        # If MCTS already chose pass, no override needed
        if mcts_action == pass_action:
            return False

        analysis = self.analyze(game, human_just_passed)

        if analysis.should_ai_pass:
            self._logger.info(
                "overriding_mcts_to_pass",
                mcts_action=mcts_action,
                mcts_row=mcts_action // game.board_size,
                mcts_col=mcts_action % game.board_size,
                reason=analysis.reason,
                fill_ratio=round(analysis.fill_ratio, 3),
            )

        return analysis.should_ai_pass

    def get_pass_action(self, board_size: int) -> int:
        """Get the pass action index for a given board size.

        Args:
            board_size: Board dimension.

        Returns:
            Action index representing pass.

        """
        return board_size**2
