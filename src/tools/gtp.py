"""Go Text Protocol (GTP) interface for AlphaGalerkin.

Implements GTP version 2 for compatibility with Go GUIs and engines.
Reference: https://www.lysator.liu.se/~gunnar/gtp/
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

import numpy as np
import structlog

from src.mcts.evaluator import FNetEvaluator, RandomEvaluator
from src.mcts.search import MCTS
from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)

# GTP column letters (A-H, J-T, skipping I to avoid confusion with 1)
GTP_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def coord_to_gtp(row: int, col: int, board_size: int) -> str:
    """Convert internal coordinates to GTP format.

    GTP uses letters A-T (excluding I) for columns and 1-19 for rows.
    Row 0 is at the bottom (row 1 in GTP).

    Args:
        row: Internal row (0 = top).
        col: Internal column (0 = left).
        board_size: Size of the board.

    Returns:
        GTP coordinate string (e.g., "D4").

    """
    col_letter = GTP_LETTERS[col]

    # GTP rows: 1 at bottom, board_size at top
    gtp_row = board_size - row

    return f"{col_letter}{gtp_row}"


def gtp_to_coord(gtp: str, board_size: int) -> tuple[int, int]:
    """Convert GTP coordinate to internal format.

    Args:
        gtp: GTP coordinate string (e.g., "D4").
        board_size: Size of the board.

    Returns:
        Tuple of (row, col) in internal format.

    """
    col_letter = gtp[0].upper()
    gtp_row = int(gtp[1:])

    col = GTP_LETTERS.index(col_letter)
    row = board_size - gtp_row

    return row, col


def action_to_gtp(action: int, board_size: int) -> str:
    """Convert action index to GTP move.

    Args:
        action: Action index (0 to board_size^2 for positions, board_size^2 for pass).
        board_size: Size of the board.

    Returns:
        GTP move string.

    """
    if action == board_size ** 2:
        return "pass"

    row = action // board_size
    col = action % board_size

    return coord_to_gtp(row, col, board_size)


def gtp_to_action(gtp: str, board_size: int) -> int:
    """Convert GTP move to action index.

    Args:
        gtp: GTP move string.
        board_size: Size of the board.

    Returns:
        Action index.

    """
    gtp = gtp.strip().lower()

    if gtp == "pass":
        return board_size ** 2

    if gtp == "resign":
        return -1  # Special value for resign

    row, col = gtp_to_coord(gtp, board_size)
    return row * board_size + col


class SimpleGoGame:
    """Simple Go game state for GTP interface.

    Implements basic Go rules without full rule enforcement.
    For production, use a proper Go library like gym-go.
    """

    BLACK = 1
    WHITE = 2
    EMPTY = 0

    def __init__(self, board_size: int = 19) -> None:
        """Initialize game.

        Args:
            board_size: Size of the board.

        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = self.BLACK
        self.move_history: list[tuple[int, int] | None] = []
        self.captures = {self.BLACK: 0, self.WHITE: 0}
        self.komi = 6.5
        self.passes = 0

    def reset(self) -> None:
        """Reset the game."""
        self.board.fill(0)
        self.current_player = self.BLACK
        self.move_history.clear()
        self.captures = {self.BLACK: 0, self.WHITE: 0}
        self.passes = 0

    def play(self, row: int, col: int) -> bool:
        """Play a stone at the given position.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if move was legal, False otherwise.

        """
        if row < 0 or row >= self.board_size:
            return False
        if col < 0 or col >= self.board_size:
            return False
        if self.board[row, col] != self.EMPTY:
            return False

        # Place stone
        self.board[row, col] = self.current_player

        # Remove captured stones (simplified - doesn't handle all cases)
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == opponent:
                    if not self._has_liberty(nr, nc):
                        captured = self._remove_group(nr, nc)
                        self.captures[self.current_player] += captured

        # Check if own group has liberty (suicide rule - simplified)
        if not self._has_liberty(row, col):
            # Undo move
            self.board[row, col] = self.EMPTY
            return False

        self.move_history.append((row, col))
        self.current_player = opponent
        self.passes = 0

        return True

    def play_pass(self) -> None:
        """Play a pass move."""
        self.move_history.append(None)
        self.current_player = (
            self.WHITE if self.current_player == self.BLACK else self.BLACK
        )
        self.passes += 1

    def _has_liberty(self, row: int, col: int) -> bool:
        """Check if the group at (row, col) has any liberty."""
        color = self.board[row, col]
        if color == self.EMPTY:
            return True

        visited = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == self.EMPTY:
                        return True
                    if self.board[nr, nc] == color and (nr, nc) not in visited:
                        stack.append((nr, nc))

        return False

    def _remove_group(self, row: int, col: int) -> int:
        """Remove the group at (row, col) and return count."""
        color = self.board[row, col]
        if color == self.EMPTY:
            return 0

        count = 0
        stack = [(row, col)]
        visited = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if self.board[r, c] != color:
                continue

            visited.add((r, c))
            self.board[r, c] = self.EMPTY
            count += 1

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    stack.append((nr, nc))

        return count

    def is_game_over(self) -> bool:
        """Check if the game is over (two consecutive passes)."""
        return self.passes >= 2

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices.

        A move is legal if:
        1. The position is empty
        2. The move doesn't result in suicide (group has liberty after captures)
        """
        legal = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self._is_legal_move(row, col):
                    legal.append(row * self.board_size + col)

        # Pass is always legal
        legal.append(self.board_size ** 2)

        return legal

    def _is_legal_move(self, row: int, col: int) -> bool:
        """Check if a move at (row, col) is legal.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if the move is legal, False otherwise.

        """
        # Must be empty
        if self.board[row, col] != self.EMPTY:
            return False

        # Simulate placing the stone
        self.board[row, col] = self.current_player
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK

        # Check if any adjacent opponent group would be captured
        captures_any = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == opponent:
                    if not self._has_liberty(nr, nc):
                        captures_any = True
                        break

        # Check if our stone/group has liberty (or captures give us liberty)
        has_liberty = self._has_liberty(row, col) or captures_any

        # Undo the simulation
        self.board[row, col] = self.EMPTY

        return has_liberty

    def get_state(self) -> np.ndarray:
        """Get state tensor for neural network.

        Returns:
            State tensor of shape (17, board_size, board_size).

        """
        # Simplified feature planes:
        # Planes 0-7: Current player's stones (history)
        # Planes 8-15: Opponent's stones (history)
        # Plane 16: Current player indicator

        state = np.zeros((17, self.board_size, self.board_size), dtype=np.float32)

        # Current position
        current_color = self.current_player
        opponent_color = self.WHITE if current_color == self.BLACK else self.BLACK

        state[0] = (self.board == current_color).astype(np.float32)
        state[8] = (self.board == opponent_color).astype(np.float32)

        # Player indicator
        state[16] = 1.0 if current_color == self.BLACK else 0.0

        return state

    def clone(self) -> SimpleGoGame:
        """Create a deep copy of the game."""
        game = SimpleGoGame(self.board_size)
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.move_history = self.move_history.copy()
        game.captures = self.captures.copy()
        game.komi = self.komi
        game.passes = self.passes
        return game

    def get_winner(self) -> int:
        """Get winner (simplified scoring)."""
        # Count territory (simplified - just count stones + captures)
        black_score = (self.board == self.BLACK).sum() + self.captures[self.BLACK]
        white_score = (
            (self.board == self.WHITE).sum() + self.captures[self.WHITE] + self.komi
        )

        if black_score > white_score:
            return 1 if self.current_player == self.BLACK else -1
        else:
            return 1 if self.current_player == self.WHITE else -1

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.is_game_over()

    def apply_action(self, action: int) -> None:
        """Apply action to game state.

        Args:
            action: Action index (0 to board_size^2-1 for moves, board_size^2 for pass).

        """
        pass_action = self.board_size ** 2
        if action == pass_action:
            self.play_pass()
        else:
            row = action // self.board_size
            col = action % self.board_size
            self.play(row, col)


class GTPEngine:
    """GTP engine for AlphaGalerkin."""

    def __init__(
        self,
        model: AlphaGalerkinModel | None = None,
        board_size: int = 19,
        device: str = "cpu",
    ) -> None:
        """Initialize GTP engine.

        Args:
            model: AlphaGalerkin model (or None for random play).
            board_size: Default board size.
            device: Device for inference.

        """
        self.model = model
        self.board_size = board_size
        self.device = device
        self.game = SimpleGoGame(board_size)

        # Set up evaluator and MCTS
        if model is not None:
            self.evaluator = FNetEvaluator(model, device=device)
        else:
            n_actions = board_size ** 2 + 1
            self.evaluator = RandomEvaluator(n_actions)

        self.mcts = MCTS(
            evaluator=self.evaluator,
            n_simulations=100,  # Reduced for responsiveness
        )

        # GTP command handlers
        self.commands: dict[str, Callable[..., str]] = {
            "protocol_version": self._protocol_version,
            "name": self._name,
            "version": self._version,
            "known_command": self._known_command,
            "list_commands": self._list_commands,
            "quit": self._quit,
            "boardsize": self._boardsize,
            "clear_board": self._clear_board,
            "komi": self._komi,
            "play": self._play,
            "genmove": self._genmove,
            "showboard": self._showboard,
        }

        self._quit_flag = False

    def run(
        self,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> None:
        """Run the GTP engine.

        Args:
            input_stream: Input stream for commands.
            output_stream: Output stream for responses.

        """
        while not self._quit_flag:
            try:
                line = input_stream.readline()
                if not line:
                    break

                response = self.process_command(line)
                output_stream.write(response)
                output_stream.flush()

            except KeyboardInterrupt:
                break

    def process_command(self, line: str) -> str:
        """Process a GTP command and return response.

        Args:
            line: Command line.

        Returns:
            GTP response string.

        """
        line = line.strip()
        if not line or line.startswith("#"):
            return ""

        # Parse command ID if present
        parts = line.split()
        cmd_id = None
        if parts[0].isdigit():
            cmd_id = parts[0]
            parts = parts[1:]

        if not parts:
            return ""

        cmd = parts[0].lower()
        args = parts[1:]

        # Execute command
        if cmd in self.commands:
            try:
                result = self.commands[cmd](*args)
                return self._success_response(cmd_id, result)
            except Exception as e:
                return self._error_response(cmd_id, str(e))
        else:
            return self._error_response(cmd_id, f"unknown command: {cmd}")

    def _success_response(self, cmd_id: str | None, result: str) -> str:
        """Format success response."""
        prefix = f"={cmd_id}" if cmd_id else "="
        if result:
            return f"{prefix} {result}\n\n"
        return f"{prefix}\n\n"

    def _error_response(self, cmd_id: str | None, message: str) -> str:
        """Format error response."""
        prefix = f"?{cmd_id}" if cmd_id else "?"
        return f"{prefix} {message}\n\n"

    def _protocol_version(self) -> str:
        return "2"

    def _name(self) -> str:
        return "AlphaGalerkin"

    def _version(self) -> str:
        return "0.1.0"

    def _known_command(self, cmd: str) -> str:
        return "true" if cmd.lower() in self.commands else "false"

    def _list_commands(self) -> str:
        return "\n".join(sorted(self.commands.keys()))

    def _quit(self) -> str:
        self._quit_flag = True
        return ""

    def _boardsize(self, size: str) -> str:
        size_int = int(size)
        if size_int < 2 or size_int > 25:
            raise ValueError("invalid board size")

        self.board_size = size_int
        self.game = SimpleGoGame(size_int)

        # Update evaluator if using random
        if isinstance(self.evaluator, RandomEvaluator):
            self.evaluator = RandomEvaluator(size_int ** 2 + 1)

        self.mcts.reset()
        return ""

    def _clear_board(self) -> str:
        self.game.reset()
        self.mcts.reset()
        return ""

    def _komi(self, komi: str) -> str:
        self.game.komi = float(komi)
        return ""

    def _play(self, color: str, vertex: str) -> str:
        vertex = vertex.lower()

        if vertex == "pass":
            self.game.play_pass()
        elif vertex == "resign":
            pass  # Game over
        else:
            row, col = gtp_to_coord(vertex, self.board_size)
            if not self.game.play(row, col):
                raise ValueError("illegal move")

        # Advance MCTS tree
        action = gtp_to_action(vertex, self.board_size)
        if action >= 0:
            self.mcts.advance(action)

        return ""

    def _genmove(self, color: str) -> str:
        """Generate a move for the specified color.

        Args:
            color: Color to play ("black", "b", "white", or "w").

        Returns:
            GTP move string (e.g., "D4" or "pass").
        """
        # Determine expected player from GTP color
        if color.lower() in ("b", "black"):
            pass
        else:
            pass

        # Validate that game state matches expected player
        if self.game.current_player != expected_player:
            logger.warning(
                "genmove_player_mismatch",
                expected=expected_player,
                actual=self.game.current_player,
            )

        # Generate move using MCTS
        action = self.mcts.get_action(
            self.game,
            temperature=0.0,  # Deterministic for play
            add_noise=False,
        )

        # Convert to GTP format
        move = action_to_gtp(action, self.board_size)

        # Apply move
        if action == self.board_size ** 2:
            self.game.play_pass()
        else:
            row = action // self.board_size
            col = action % self.board_size
            self.game.play(row, col)

        self.mcts.advance(action)

        return move

    def _showboard(self) -> str:
        """Show current board state."""
        lines = []

        # Column labels
        col_labels = "   " + " ".join(GTP_LETTERS[: self.board_size])
        lines.append(col_labels)

        for row in range(self.board_size):
            row_num = self.board_size - row
            row_str = f"{row_num:2d} "

            for col in range(self.board_size):
                stone = self.game.board[row, col]
                if stone == SimpleGoGame.BLACK:
                    row_str += "X "
                elif stone == SimpleGoGame.WHITE:
                    row_str += "O "
                else:
                    row_str += ". "

            row_str += f"{row_num:2d}"
            lines.append(row_str)

        lines.append(col_labels)

        return "\n" + "\n".join(lines)


def main() -> None:
    """Run GTP engine.

    Loads a trained AlphaGalerkin model and runs a GTP interface for
    communication with Go GUIs and engines.

    Usage:
        python -m src.tools.gtp --model checkpoints/best.pt --board-size 9
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="AlphaGalerkin GTP Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--board-size", type=int, default=19, help="Board size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")

    args = parser.parse_args()

    # Load model if provided
    model = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error("model_not_found", path=str(model_path))
            sys.exit(1)

        logger.info("loading_model", path=str(model_path), device=args.device)

        try:
            from src.training.checkpoint import create_model_from_checkpoint

            model, config_dict = create_model_from_checkpoint(
                path=model_path,
                device=args.device,
            )
            if config_dict:
                logger.info("model_config_loaded_from_checkpoint")
            else:
                logger.info("using_default_model_config")
        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            sys.exit(1)

    # Create and run engine
    logger.info("starting_gtp_engine", board_size=args.board_size)
    engine = GTPEngine(model=model, board_size=args.board_size, device=args.device)
    engine.run()


if __name__ == "__main__":
    main()
