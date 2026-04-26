"""Converter between SGF and AlphaGalerkin game state.

Provides bidirectional conversion for integration with the game engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import structlog

from src.games.sgf.config import SGFConfig
from src.games.sgf.node import SGFGameTree, SGFMove, SGFNode
from src.games.sgf.parser import SGFParser
from src.games.sgf.writer import SGFWriter

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class SGFConverter:
    """Converter between SGF game trees and AlphaGalerkin game states.

    Features:
    - Convert SGF to sequence of game states
    - Convert game state history to SGF
    - Add analysis annotations (policy, value, etc.)
    - Support for evaluation data in comments

    Example:
        converter = SGFConverter()

        # SGF to game states
        game = converter.load_game("game.sgf")
        for state in converter.iter_states(game):
            policy, value = model(state)

        # Game states to SGF
        sgf_text = converter.from_states(states, game_info={...})

    """

    def __init__(self, config: SGFConfig | None = None) -> None:
        """Initialize converter.

        Args:
            config: Configuration for parsing/writing

        """
        self.config = config or SGFConfig(name="sgf_converter")
        self._parser = SGFParser(config)
        self._writer = SGFWriter(config)

    def load_game(self, path: str) -> SGFGameTree:
        """Load an SGF file.

        Args:
            path: Path to SGF file

        Returns:
            Parsed game tree

        """
        return self._parser.parse_file(path)

    def parse_sgf(self, sgf_text: str) -> SGFGameTree:
        """Parse SGF text.

        Args:
            sgf_text: SGF format string

        Returns:
            Parsed game tree

        """
        return self._parser.parse(sgf_text)

    def write_sgf(self, tree: SGFGameTree) -> str:
        """Write game tree to SGF string.

        Args:
            tree: Game tree to write

        Returns:
            SGF format string

        """
        return self._writer.write(tree)

    def iter_positions(
        self,
        tree: SGFGameTree,
    ) -> Iterator[tuple[SGFNode, list[SGFMove]]]:
        """Iterate through positions on the main line.

        Yields:
            Tuples of (node, move_history) for each position

        """
        moves: list[SGFMove] = []
        for node in tree.mainline():
            yield node, list(moves)
            if node.move:
                moves.append(node.move)

    def to_move_sequence(self, tree: SGFGameTree) -> list[tuple[str, int, int]]:
        """Extract move sequence from game tree.

        Args:
            tree: Game tree

        Returns:
            List of (color, x, y) tuples

        """
        moves = []
        for move in tree.mainline_moves():
            if not move.is_pass:
                moves.append((move.color, move.x, move.y))
            else:
                moves.append((move.color, -1, -1))
        return moves

    def from_move_sequence(
        self,
        moves: list[tuple[str, int, int]],
        board_size: int = 19,
        game_info: dict[str, str] | None = None,
    ) -> SGFGameTree:
        """Create game tree from move sequence.

        Args:
            moves: List of (color, x, y) tuples
            board_size: Board size
            game_info: Optional game info to include

        Returns:
            New game tree

        """
        tree = SGFGameTree()
        tree.board_size = board_size

        # Set standard properties
        tree.root.set_property("FF", "4")
        tree.root.set_property("GM", "1")
        tree.root.set_property("SZ", str(board_size))

        # Set game info
        if game_info:
            tree.set_game_info(**game_info)

        # Add moves
        current = tree.root
        for color, x, y in moves:
            child = current.new_child()
            move = SGFMove(color=color, x=x, y=y)
            child.move = move
            current = child

        return tree

    def add_analysis(
        self,
        node: SGFNode,
        policy: dict[tuple[int, int], float] | None = None,
        value: float | None = None,
        principal_variation: list[tuple[int, int]] | None = None,
        comment: str | None = None,
    ) -> None:
        """Add analysis data to a node.

        Args:
            node: Node to annotate
            policy: Move probabilities as {(x, y): probability}
            value: Position evaluation (-1 to 1, positive = black winning)
            principal_variation: Best move sequence
            comment: Text comment to add

        """
        parts = []

        # Add value
        if value is not None:
            win_rate = (value + 1) / 2 * 100
            parts.append(f"Win rate: {win_rate:.1f}%")
            node.set_property("V", f"{value:.4f}")

        # Add top policy moves
        if policy:
            sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)
            top_moves = sorted_moves[:5]
            move_strs = []
            for (x, y), prob in top_moves:
                coord = SGFMove(color="", x=x, y=y).to_gtp(node._board_size)
                move_strs.append(f"{coord}:{prob * 100:.1f}%")
            parts.append(f"Top moves: {', '.join(move_strs)}")

        # Add principal variation
        if principal_variation:
            pv_strs = []
            for x, y in principal_variation[:5]:
                coord = SGFMove(color="", x=x, y=y).to_gtp(node._board_size)
                pv_strs.append(coord)
            parts.append(f"PV: {' '.join(pv_strs)}")

        # Add custom comment
        if comment:
            parts.append(comment)

        # Set combined comment
        if parts:
            existing = node.get_comment()
            if existing:
                parts.insert(0, existing)
            node.set_comment("\n".join(parts))

    def mark_move_quality(
        self,
        node: SGFNode,
        quality: str,
    ) -> None:
        """Mark the quality of a move.

        Args:
            node: Node with the move
            quality: One of "good", "bad", "doubtful", "interesting", "tesuji"

        """
        quality_map = {
            "good": None,  # No special marking for good moves
            "bad": "BM",
            "doubtful": "DO",
            "interesting": "IT",
            "tesuji": "TE",
        }

        if quality_map.get(quality):
            node.set_property(quality_map[quality], "1")

    def add_label(
        self,
        node: SGFNode,
        x: int,
        y: int,
        label: str,
    ) -> None:
        """Add a text label to a position.

        Args:
            node: Node to add label to
            x: X coordinate
            y: Y coordinate
            label: Label text

        """
        coord = SGFMove(color="", x=x, y=y).to_sgf(node._board_size)
        node.add_property_value("LB", f"{coord}:{label}")

    def add_triangle(self, node: SGFNode, x: int, y: int) -> None:
        """Add a triangle marker."""
        coord = SGFMove(color="", x=x, y=y).to_sgf(node._board_size)
        node.add_property_value("TR", coord)

    def add_circle(self, node: SGFNode, x: int, y: int) -> None:
        """Add a circle marker."""
        coord = SGFMove(color="", x=x, y=y).to_sgf(node._board_size)
        node.add_property_value("CR", coord)

    def add_square(self, node: SGFNode, x: int, y: int) -> None:
        """Add a square marker."""
        coord = SGFMove(color="", x=x, y=y).to_sgf(node._board_size)
        node.add_property_value("SQ", coord)

    def add_mark(self, node: SGFNode, x: int, y: int) -> None:
        """Add an X marker."""
        coord = SGFMove(color="", x=x, y=y).to_sgf(node._board_size)
        node.add_property_value("MA", coord)

    def create_analysis_tree(
        self,
        original: SGFGameTree,
        evaluations: list[dict[str, float | dict]],
    ) -> SGFGameTree:
        """Create a new tree with analysis annotations.

        Args:
            original: Original game tree
            evaluations: List of evaluation dicts for each move, containing:
                - "value": Position evaluation
                - "policy": Move probabilities
                - "pv": Principal variation

        Returns:
            New game tree with analysis

        """
        # Create a copy by writing and parsing
        sgf_text = self._writer.write(original)
        analyzed = self._parser.parse(sgf_text)

        # Add analysis to each node
        nodes = list(analyzed.mainline())
        for i, node in enumerate(nodes):
            if i < len(evaluations):
                eval_data = evaluations[i]
                self.add_analysis(
                    node,
                    policy=eval_data.get("policy"),
                    value=eval_data.get("value"),
                    principal_variation=eval_data.get("pv"),
                )

        # Update game name to indicate analysis
        game_name = analyzed.root.get_property("GN", "")
        analyzed.root.set_property("GN", f"{game_name} (analyzed)")

        return analyzed
