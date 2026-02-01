"""Game state management for variable board sizes.

This module provides centralized game state management for the HuggingFace Space:
- Game creation with configurable board sizes
- Komi values per board size
- Game history and state tracking
- Score calculation and display formatting
- Zero-shot transfer information
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

# Ensure config imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.board import SpaceConfig, get_default_space_config

from src.tools.gtp import SimpleGoGame

if TYPE_CHECKING:
    from src.mcts.evaluator import FNetEvaluator

logger = structlog.get_logger(__name__)


@dataclass
class GameSession:
    """Represents an active game session.

    Tracks game state, history, and metadata for a single game.

    Attributes:
        game: The Go game instance.
        board_size: Board size for this session.
        komi: Komi value for this game.
        move_history: List of moves played (coordinates or "PASS").
        is_human_vs_ai: Whether this is a human vs AI game.
        training_board_size: Board size the model was trained on.

    """

    game: SimpleGoGame
    board_size: int
    komi: float
    move_history: list[tuple[int, int] | str] = field(default_factory=list)
    is_human_vs_ai: bool = True
    training_board_size: int = 9

    @property
    def move_count(self) -> int:
        """Get number of moves played."""
        return len(self.move_history)

    @property
    def current_player_name(self) -> str:
        """Get current player as human-readable string."""
        if self.game.current_player == SimpleGoGame.BLACK:
            return "Black"
        return "White"

    @property
    def is_terminal(self) -> bool:
        """Check if game has ended."""
        return self.game.is_terminal()

    @property
    def is_zero_shot(self) -> bool:
        """Check if this game uses zero-shot transfer (not training size)."""
        return self.board_size != self.training_board_size


class GameManager:
    """Manages game sessions with variable board sizes.

    Provides centralized game state management including:
    - Creating new games with appropriate komi
    - Replaying move history
    - Score calculation and formatting
    - Zero-shot transfer labels

    Attributes:
        config: Space configuration.
        evaluator: Neural network evaluator for AI moves (optional).
        mcts_kwargs: MCTS configuration parameters.

    Example:
        >>> manager = GameManager()
        >>> session = manager.create_game(board_size=13)
        >>> display = manager.get_score_display(session)

    """

    def __init__(
        self,
        config: SpaceConfig | None = None,
        evaluator: FNetEvaluator | None = None,
        mcts_kwargs: dict | None = None,
    ) -> None:
        """Initialize game manager.

        Args:
            config: Space configuration.
            evaluator: Neural network evaluator for AI moves.
            mcts_kwargs: MCTS configuration overrides.

        """
        self.config = config or get_default_space_config()
        self.evaluator = evaluator
        self.mcts_kwargs = mcts_kwargs or {
            "n_simulations": self.config.mcts_simulations,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.03,
            "dirichlet_epsilon": 0.0,
        }
        self._logger = logger.bind(component="GameManager")

    def create_game(
        self,
        board_size: int | None = None,
        is_human_vs_ai: bool = True,
    ) -> GameSession:
        """Create a new game session.

        Args:
            board_size: Board size (uses default if None).
            is_human_vs_ai: Whether this is human vs AI.

        Returns:
            New game session.

        Raises:
            ValueError: If board size is not supported.

        """
        board_size = board_size or self.config.default_board_size

        if board_size not in self.config.supported_sizes:
            raise ValueError(
                f"Board size {board_size} not in supported sizes: "
                f"{self.config.supported_sizes}"
            )

        komi = self.config.get_komi(board_size)
        game = SimpleGoGame(board_size)
        game.komi = komi

        session = GameSession(
            game=game,
            board_size=board_size,
            komi=komi,
            is_human_vs_ai=is_human_vs_ai,
            training_board_size=self.config.training_board_size,
        )

        self._logger.info(
            "game_created",
            board_size=board_size,
            komi=komi,
            is_human_vs_ai=is_human_vs_ai,
            is_zero_shot=session.is_zero_shot,
        )

        return session

    def replay_history(
        self,
        history: list[tuple[int, int] | str],
        board_size: int,
    ) -> SimpleGoGame:
        """Reconstruct game state from move history.

        Args:
            history: List of moves (coordinates or "PASS").
            board_size: Board size.

        Returns:
            Reconstructed game state.

        """
        game = SimpleGoGame(board_size)
        game.komi = self.config.get_komi(board_size)

        for move in history:
            if move == "PASS":
                game.play_pass()
            else:
                r, c = move
                game.play(r, c)

        return game

    def get_score_display(self, session: GameSession) -> str:
        """Get formatted score string for live display.

        Args:
            session: Active game session.

        Returns:
            Formatted score string with captures, moves, and komi.

        """
        game = session.game
        black_captures = game.captures.get(SimpleGoGame.BLACK, 0)
        white_captures = game.captures.get(SimpleGoGame.WHITE, 0)

        transfer_tag = " [Zero-shot]" if session.is_zero_shot else ""

        return (
            f"Black captures: {black_captures} | "
            f"White captures: {white_captures} | "
            f"Move: {session.move_count} | "
            f"{session.current_player_name} to play | "
            f"Komi: {session.komi}{transfer_tag}"
        )

    def calculate_final_score(self, session: GameSession) -> str:
        """Calculate and format end-game score.

        Uses simplified Chinese scoring (stones + captures).

        Args:
            session: Completed game session.

        Returns:
            Formatted final score string.

        """
        game = session.game
        komi = session.komi

        # Count stones + captures (simplified Chinese scoring)
        black_stones = int((game.board == SimpleGoGame.BLACK).sum())
        white_stones = int((game.board == SimpleGoGame.WHITE).sum())

        black_captures = game.captures.get(SimpleGoGame.BLACK, 0)
        white_captures = game.captures.get(SimpleGoGame.WHITE, 0)

        black_score = float(black_stones + black_captures)
        white_score = float(white_stones + white_captures + komi)

        if black_score > white_score:
            margin = black_score - white_score
            return (
                f"Black wins by {margin:.1f} points "
                f"(B: {black_score:.1f}, W: {white_score:.1f})"
            )
        elif white_score > black_score:
            margin = white_score - black_score
            return (
                f"White wins by {margin:.1f} points "
                f"(B: {black_score:.1f}, W: {white_score:.1f})"
            )
        else:
            return f"Draw (B: {black_score:.1f}, W: {white_score:.1f})"

    def get_board_size_label(self, size: int) -> str:
        """Get descriptive label for board size.

        Indicates whether the size is the training size or uses zero-shot transfer.

        Args:
            size: Board size.

        Returns:
            Descriptive label with training/zero-shot info.

        """
        if size == self.config.training_board_size:
            return f"{size}×{size} (Training size)"
        return f"{size}×{size} (Zero-shot transfer)"

    def get_board_size_choices(self) -> list[tuple[str, int]]:
        """Get board size choices for UI dropdown.

        Returns:
            List of (label, value) tuples for dropdown.

        """
        return [
            (self.get_board_size_label(size), size)
            for size in self.config.supported_sizes
        ]

    def format_move(self, row: int, col: int, board_size: int) -> str:
        """Format a move for display.

        Args:
            row: Row index.
            col: Column index.
            board_size: Board size.

        Returns:
            Formatted move string (e.g., "D4" or "4,4").

        """
        from config.board import get_column_letter

        letter = get_column_letter(col, skip_i=True)
        gtp_row = board_size - row
        return f"{letter}{gtp_row}"

    def parse_move(self, input_text: str, board_size: int) -> tuple[int, int] | str:
        """Parse move input from user.

        Accepts formats:
        - GTP format: "D4", "A1", "J10" (letter + number, skipping I)
        - Numeric format: "row,col" or "row col" (0-indexed)
        - Pass: "PASS", "pass", "P", "p"

        Args:
            input_text: User input text.
            board_size: Current board size.

        Returns:
            Tuple (row, col) or "PASS".

        Raises:
            ValueError: If input format is invalid.

        """
        from src.tools.gtp import gtp_to_coord

        text = input_text.strip().upper()

        # Handle pass moves
        if text in ("PASS", "P"):
            return "PASS"

        # Try GTP format first: letter + optional space + digits (e.g., A4, D10, J 5)
        # GTP letters are A-H, J-T (skipping I)
        if len(text) >= 2 and text[0].isalpha():
            # Extract letter and number parts
            letter = text[0]
            rest = text[1:].strip()
            if rest.isdigit():
                try:
                    gtp_coord = f"{letter}{rest}"
                    row, col = gtp_to_coord(gtp_coord, board_size)
                    if 0 <= row < board_size and 0 <= col < board_size:
                        logger.debug(
                            "parsed_gtp_move",
                            input=input_text,
                            gtp=gtp_coord,
                            row=row,
                            col=col,
                        )
                        return (row, col)
                except (ValueError, IndexError):
                    pass  # Fall through to numeric parsing

        # Try numeric format: row,col or row col (0-indexed)
        parts = text.replace(",", " ").split()
        if len(parts) == 2:
            try:
                row, col = int(parts[0]), int(parts[1])
                if 0 <= row < board_size and 0 <= col < board_size:
                    logger.debug(
                        "parsed_numeric_move",
                        input=input_text,
                        row=row,
                        col=col,
                    )
                    return (row, col)
                raise ValueError(
                    f"Position ({row},{col}) is outside the "
                    f"{board_size}×{board_size} board (valid: 0-{board_size-1})"
                )
            except ValueError:
                pass

        # If nothing worked, provide helpful error
        raise ValueError(
            f"Invalid format. Use GTP (e.g., D4, A1) or "
            f"numeric row,col (e.g., 3,3). Range: 0-{board_size-1}"
        )
