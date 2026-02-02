"""SGF node and game tree data structures.

Provides:
- SGFNode: Individual node in the game tree
- SGFGameTree: Complete game tree with traversal methods
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field

from src.games.sgf.config import SGF_PROPERTIES


@dataclass
class SGFMove:
    """Represents a single move in the game.

    Attributes:
        color: "B" for black, "W" for white
        x: Column coordinate (0-indexed, -1 for pass)
        y: Row coordinate (0-indexed, -1 for pass)
        sgf_coord: Original SGF coordinate string

    """

    color: str
    x: int
    y: int
    sgf_coord: str = ""

    @property
    def is_pass(self) -> bool:
        """Check if this is a pass move."""
        # Pass if both coordinates are negative (explicitly pass)
        if self.x < 0 and self.y < 0:
            return True
        # Pass if sgf_coord is explicitly set to "" or "tt" (parsed as pass)
        if self.sgf_coord == "tt":
            return True
        # Valid coordinates mean not a pass
        return False

    @classmethod
    def from_sgf(cls, color: str, coord: str, board_size: int = 19) -> SGFMove:
        """Create move from SGF coordinate.

        Args:
            color: "B" or "W"
            coord: SGF coordinate (e.g., "pd", "dd", "" for pass)
            board_size: Board size for validation

        Returns:
            SGFMove instance

        """
        if not coord or coord == "tt" or len(coord) != 2:
            return cls(color=color, x=-1, y=-1, sgf_coord=coord)

        # SGF uses lowercase letters a-s for 19x19
        # 'a' = 0, 'b' = 1, etc.
        x = ord(coord[0].lower()) - ord("a")
        y = ord(coord[1].lower()) - ord("a")

        # Handle boards larger than 19x19 (uses uppercase)
        if coord[0].isupper():
            x = ord(coord[0]) - ord("A") + 26
        if coord[1].isupper():
            y = ord(coord[1]) - ord("A") + 26

        return cls(color=color, x=x, y=y, sgf_coord=coord)

    def to_sgf(self, board_size: int = 19) -> str:
        """Convert to SGF coordinate string.

        Args:
            board_size: Board size

        Returns:
            SGF coordinate string

        """
        if self.is_pass:
            return ""

        # Standard SGF coordinates
        if self.x < 26 and self.y < 26:
            return chr(ord("a") + self.x) + chr(ord("a") + self.y)

        # Extended coordinates for larger boards
        x_chr = chr(ord("A") + self.x - 26) if self.x >= 26 else chr(ord("a") + self.x)
        y_chr = chr(ord("A") + self.y - 26) if self.y >= 26 else chr(ord("a") + self.y)
        return x_chr + y_chr

    def to_gtp(self, board_size: int = 19) -> str:
        """Convert to GTP coordinate string (e.g., "D4").

        Args:
            board_size: Board size

        Returns:
            GTP coordinate string

        """
        if self.is_pass:
            return "pass"

        # GTP uses letters A-T (skipping I) for columns
        col_letter = chr(ord("A") + self.x + (1 if self.x >= 8 else 0))
        # GTP uses 1-indexed rows from bottom
        row_number = board_size - self.y

        return f"{col_letter}{row_number}"

    def __str__(self) -> str:
        if self.is_pass:
            return f"{self.color}[pass]"
        return f"{self.color}[{self.sgf_coord}]"


@dataclass
class SGFNode:
    """A node in the SGF game tree.

    Each node can have:
    - Properties (key-value pairs)
    - A move (optional)
    - Children (variations)
    - A parent (except root)
    """

    properties: dict[str, list[str]] = field(default_factory=dict)
    children: list[SGFNode] = field(default_factory=list)
    parent: SGFNode | None = field(default=None, repr=False)

    # Cached values
    _move: SGFMove | None = field(default=None, repr=False)
    _board_size: int = field(default=19, repr=False)

    def get_property(self, key: str, default: str | None = None) -> str | None:
        """Get a property value.

        Args:
            key: Property key (e.g., "PB", "B", "C")
            default: Default value if not found

        Returns:
            First property value or default

        """
        values = self.properties.get(key, [])
        return values[0] if values else default

    def get_property_list(self, key: str) -> list[str]:
        """Get all values for a property.

        Args:
            key: Property key

        Returns:
            List of values (empty if not found)

        """
        return self.properties.get(key, [])

    def set_property(self, key: str, value: str | list[str]) -> None:
        """Set a property value.

        Args:
            key: Property key
            value: Value or list of values

        """
        if isinstance(value, list):
            self.properties[key] = value
        else:
            self.properties[key] = [value]

    def add_property_value(self, key: str, value: str) -> None:
        """Add a value to a property (for multi-value properties).

        Args:
            key: Property key
            value: Value to add

        """
        if key not in self.properties:
            self.properties[key] = []
        self.properties[key].append(value)

    def remove_property(self, key: str) -> bool:
        """Remove a property.

        Args:
            key: Property key

        Returns:
            True if property was removed

        """
        if key in self.properties:
            del self.properties[key]
            return True
        return False

    @property
    def move(self) -> SGFMove | None:
        """Get the move for this node (if any)."""
        if self._move is not None:
            return self._move

        # Check for black or white move
        black = self.get_property("B")
        white = self.get_property("W")

        if black is not None:
            self._move = SGFMove.from_sgf("B", black, self._board_size)
        elif white is not None:
            self._move = SGFMove.from_sgf("W", white, self._board_size)

        return self._move

    @move.setter
    def move(self, m: SGFMove | None) -> None:
        """Set the move for this node."""
        self._move = m
        if m is not None:
            coord = m.to_sgf(self._board_size)
            self.set_property(m.color, coord)
        else:
            self.remove_property("B")
            self.remove_property("W")

    @property
    def has_move(self) -> bool:
        """Check if this node has a move."""
        return self.move is not None

    @property
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        """Get the depth of this node in the tree."""
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    @property
    def move_number(self) -> int:
        """Get the move number (counting from 1)."""
        # Count moves from root to this node
        count = 0
        node: SGFNode | None = self
        while node is not None:
            if node.has_move:
                count += 1
            node = node.parent
        return count

    def add_child(self, child: SGFNode) -> SGFNode:
        """Add a child node.

        Args:
            child: Child node to add

        Returns:
            The added child

        """
        child.parent = self
        child._board_size = self._board_size
        self.children.append(child)
        return child

    def new_child(self) -> SGFNode:
        """Create and add a new child node.

        Returns:
            The new child node

        """
        child = SGFNode()
        return self.add_child(child)

    def remove_child(self, child: SGFNode) -> bool:
        """Remove a child node.

        Args:
            child: Child to remove

        Returns:
            True if child was removed

        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            return True
        return False

    def get_root(self) -> SGFNode:
        """Get the root node of the tree."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def get_path_to_root(self) -> list[SGFNode]:
        """Get the path from this node to the root.

        Returns:
            List of nodes from this node to root (inclusive)

        """
        path = []
        node: SGFNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path

    def get_comment(self) -> str:
        """Get the comment for this node."""
        return self.get_property("C", "")

    def set_comment(self, comment: str) -> None:
        """Set the comment for this node."""
        if comment:
            self.set_property("C", comment)
        else:
            self.remove_property("C")

    def get_annotations(self) -> dict[str, str]:
        """Get all annotation properties for this node."""
        annotations = {}
        for key, (name, category) in SGF_PROPERTIES.items():
            if category in ("annotation", "move_annotation"):
                value = self.get_property(key)
                if value is not None:
                    annotations[name] = value
        return annotations

    def __iter__(self) -> Iterator[SGFNode]:
        """Iterate through children."""
        return iter(self.children)

    def __len__(self) -> int:
        """Number of children."""
        return len(self.children)


@dataclass
class SGFGameTree:
    """Complete SGF game tree.

    Provides:
    - Root node access
    - Game info extraction
    - Mainline traversal
    - Move iteration
    """

    root: SGFNode = field(default_factory=SGFNode)
    _board_size: int = field(default=19)

    def __post_init__(self) -> None:
        """Initialize board size from root."""
        size_str = self.root.get_property("SZ")
        if size_str:
            with contextlib.suppress(ValueError):
                self._board_size = int(size_str)
        self.root._board_size = self._board_size

    @property
    def board_size(self) -> int:
        """Get the board size for this game."""
        return self._board_size

    @board_size.setter
    def board_size(self, size: int) -> None:
        """Set the board size."""
        self._board_size = size
        self.root.set_property("SZ", str(size))
        self._propagate_board_size(self.root)

    def _propagate_board_size(self, node: SGFNode) -> None:
        """Propagate board size to all nodes."""
        node._board_size = self._board_size
        for child in node.children:
            self._propagate_board_size(child)

    @property
    def game_info(self) -> dict[str, str]:
        """Get game info properties from root."""
        info = {}
        for key, (name, category) in SGF_PROPERTIES.items():
            if category in ("root", "game_info"):
                value = self.root.get_property(key)
                if value is not None:
                    info[name] = value
        return info

    def set_game_info(self, **kwargs: str) -> None:
        """Set game info properties.

        Args:
            **kwargs: Property values by name (e.g., player_black="Lee Sedol")

        """
        # Reverse lookup: name -> key
        name_to_key = {name: key for key, (name, _) in SGF_PROPERTIES.items()}

        for name, value in kwargs.items():
            key = name_to_key.get(name)
            if key:
                self.root.set_property(key, value)

    def mainline(self) -> Iterator[SGFNode]:
        """Iterate through the main line (first child at each branch).

        Yields:
            Nodes along the main line, starting from root

        """
        node: SGFNode | None = self.root
        while node is not None:
            yield node
            node = node.children[0] if node.children else None

    def mainline_moves(self) -> Iterator[SGFMove]:
        """Iterate through moves on the main line.

        Yields:
            Moves (excluding nodes without moves)

        """
        for node in self.mainline():
            if node.move is not None:
                yield node.move

    def all_nodes(self) -> Iterator[SGFNode]:
        """Iterate through all nodes in the tree (depth-first).

        Yields:
            All nodes in depth-first order

        """

        def traverse(node: SGFNode) -> Iterator[SGFNode]:
            yield node
            for child in node.children:
                yield from traverse(child)

        yield from traverse(self.root)

    def count_nodes(self) -> int:
        """Count total nodes in the tree."""
        return sum(1 for _ in self.all_nodes())

    def count_moves(self) -> int:
        """Count moves on the main line."""
        return sum(1 for node in self.mainline() if node.has_move)

    def get_node_at_move(self, move_number: int) -> SGFNode | None:
        """Get the node at a specific move number on the main line.

        Args:
            move_number: Move number (1-indexed, 0 for root)

        Returns:
            Node at that move, or None if not found

        """
        count = 0
        for node in self.mainline():
            if node.has_move:
                count += 1
            if count == move_number:
                return node
        return None

    def get_result(self) -> str:
        """Get the game result."""
        return self.root.get_property("RE", "")

    def set_result(self, result: str) -> None:
        """Set the game result.

        Args:
            result: Result string (e.g., "B+2.5", "W+R", "0")

        """
        self.root.set_property("RE", result)

    def get_komi(self) -> float:
        """Get the komi value."""
        komi_str = self.root.get_property("KM", "0")
        try:
            return float(komi_str)
        except ValueError:
            return 0.0

    def set_komi(self, komi: float) -> None:
        """Set the komi value."""
        self.root.set_property("KM", str(komi))

    def get_handicap(self) -> int:
        """Get the handicap stones."""
        ha_str = self.root.get_property("HA", "0")
        try:
            return int(ha_str)
        except ValueError:
            return 0

    def get_handicap_stones(self) -> list[tuple[int, int]]:
        """Get the handicap stone positions.

        Returns:
            List of (x, y) coordinates for handicap stones

        """
        stones = []
        ab_values = self.root.get_property_list("AB")
        for coord in ab_values:
            move = SGFMove.from_sgf("B", coord, self._board_size)
            if not move.is_pass:
                stones.append((move.x, move.y))
        return stones

    def __str__(self) -> str:
        """String representation."""
        info = self.game_info
        pb = info.get("player_black", "Unknown")
        pw = info.get("player_white", "Unknown")
        result = info.get("result", "?")
        moves = self.count_moves()
        return f"SGFGameTree({pb} vs {pw}, {moves} moves, result: {result})"
