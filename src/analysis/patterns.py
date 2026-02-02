"""Pattern recognition and library.

Provides:
- Pattern matching in positions
- Pattern library management
- Opening and joseki detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog


class PatternType(str, Enum):
    """Types of patterns."""

    OPENING = "opening"  # Opening patterns (fuseki)
    JOSEKI = "joseki"  # Corner sequences
    TESUJI = "tesuji"  # Tactical patterns
    SHAPE = "shape"  # Good/bad shapes
    LIFE_DEATH = "life_death"  # Life and death patterns
    ENDGAME = "endgame"  # Endgame patterns
    CUSTOM = "custom"  # User-defined patterns


@dataclass
class Pattern:
    """A recognizable pattern in Go.

    Attributes:
        name: Pattern name.
        pattern_type: Type of pattern.
        stones: Stone positions relative to anchor.
        liberties: Expected liberty positions.
        description: Human-readable description.
        variations: Named variations of this pattern.
        tags: Tags for categorization.

    """

    name: str
    pattern_type: PatternType
    stones: dict[tuple[int, int], str] = field(default_factory=dict)
    liberties: list[tuple[int, int]] = field(default_factory=list)
    description: str = ""
    variations: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    difficulty: int = 1  # 1-5 scale

    def matches(
        self,
        board: list[list[int]],
        x: int,
        y: int,
        rotation: int = 0,
        flip: bool = False,
    ) -> bool:
        """Check if pattern matches at position.

        Args:
            board: Board state (0=empty, 1=black, 2=white).
            x: X coordinate of anchor.
            y: Y coordinate of anchor.
            rotation: Rotation (0, 90, 180, 270).
            flip: Whether to flip horizontally.

        Returns:
            True if pattern matches.

        """
        board_size = len(board)

        for (dx, dy), color in self.stones.items():
            # Apply transformation
            tx, ty = self._transform(dx, dy, rotation, flip)
            nx, ny = x + tx, y + ty

            # Check bounds
            if not (0 <= nx < board_size and 0 <= ny < board_size):
                return False

            # Check color
            expected = 1 if color == "B" else 2 if color == "W" else 0
            if board[ny][nx] != expected:
                return False

        return True

    def _transform(
        self,
        dx: int,
        dy: int,
        rotation: int,
        flip: bool,
    ) -> tuple[int, int]:
        """Transform coordinates.

        Args:
            dx: Delta X.
            dy: Delta Y.
            rotation: Rotation degrees.
            flip: Horizontal flip.

        Returns:
            Transformed (dx, dy).

        """
        if flip:
            dx = -dx

        if rotation == 90:
            dx, dy = -dy, dx
        elif rotation == 180:
            dx, dy = -dx, -dy
        elif rotation == 270:
            dx, dy = dy, -dx

        return dx, dy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "pattern_type": self.pattern_type.value,
            "stones": {f"{x},{y}": c for (x, y), c in self.stones.items()},
            "description": self.description,
            "tags": self.tags,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pattern:
        """Create from dictionary."""
        stones = {}
        for key, color in data.get("stones", {}).items():
            x, y = map(int, key.split(","))
            stones[(x, y)] = color

        return cls(
            name=data["name"],
            pattern_type=PatternType(data.get("pattern_type", "custom")),
            stones=stones,
            description=data.get("description", ""),
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", 1),
        )


@dataclass
class PatternMatch:
    """A pattern match in a position.

    Attributes:
        pattern: The matched pattern.
        x: X coordinate of match.
        y: Y coordinate of match.
        rotation: Rotation applied.
        flipped: Whether horizontally flipped.
        confidence: Match confidence (0-1).

    """

    pattern: Pattern
    x: int
    y: int
    rotation: int = 0
    flipped: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern.name,
            "x": self.x,
            "y": self.y,
            "rotation": self.rotation,
            "flipped": self.flipped,
            "confidence": self.confidence,
        }


class PatternLibrary:
    """Library of Go patterns.

    Manages a collection of patterns with:
    - Registration and retrieval
    - Category management
    - Persistence
    """

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize pattern library.

        Args:
            logger: Optional structured logger.

        """
        self._logger = logger or structlog.get_logger(__name__)
        self._patterns: dict[str, Pattern] = {}
        self._by_type: dict[PatternType, list[str]] = {t: [] for t in PatternType}
        self._by_tag: dict[str, list[str]] = {}

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to the library.

        Args:
            pattern: Pattern to add.

        """
        self._patterns[pattern.name] = pattern
        self._by_type[pattern.pattern_type].append(pattern.name)

        for tag in pattern.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(pattern.name)

        self._logger.debug("pattern_added", name=pattern.name)

    def get_pattern(self, name: str) -> Pattern | None:
        """Get a pattern by name.

        Args:
            name: Pattern name.

        Returns:
            Pattern or None if not found.

        """
        return self._patterns.get(name)

    def get_by_type(self, pattern_type: PatternType) -> list[Pattern]:
        """Get patterns by type.

        Args:
            pattern_type: Pattern type.

        Returns:
            List of patterns.

        """
        return [self._patterns[name] for name in self._by_type.get(pattern_type, [])]

    def get_by_tag(self, tag: str) -> list[Pattern]:
        """Get patterns by tag.

        Args:
            tag: Tag to search.

        Returns:
            List of matching patterns.

        """
        return [self._patterns[name] for name in self._by_tag.get(tag, [])]

    def list_patterns(self) -> list[str]:
        """List all pattern names.

        Returns:
            List of pattern names.

        """
        return list(self._patterns.keys())

    def list_types(self) -> list[PatternType]:
        """List all pattern types with patterns.

        Returns:
            List of pattern types.

        """
        return [t for t in PatternType if self._by_type.get(t)]

    def list_tags(self) -> list[str]:
        """List all tags.

        Returns:
            List of tags.

        """
        return list(self._by_tag.keys())

    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern.

        Args:
            name: Pattern name to remove.

        Returns:
            True if pattern was removed.

        """
        if name not in self._patterns:
            return False

        pattern = self._patterns[name]
        del self._patterns[name]

        self._by_type[pattern.pattern_type].remove(name)
        for tag in pattern.tags:
            if tag in self._by_tag:
                self._by_tag[tag].remove(name)

        return True

    def __len__(self) -> int:
        return len(self._patterns)

    def to_dict(self) -> dict[str, Any]:
        """Export library to dictionary.

        Returns:
            Dictionary with all patterns.

        """
        return {
            "patterns": [p.to_dict() for p in self._patterns.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatternLibrary:
        """Create library from dictionary.

        Args:
            data: Dictionary with patterns.

        Returns:
            PatternLibrary instance.

        """
        library = cls()
        for pattern_data in data.get("patterns", []):
            pattern = Pattern.from_dict(pattern_data)
            library.add_pattern(pattern)
        return library


class PatternMatcher:
    """Matches patterns in positions.

    Features:
    - Multi-pattern matching
    - Rotation and flip handling
    - Performance optimization
    """

    def __init__(
        self,
        library: PatternLibrary | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize pattern matcher.

        Args:
            library: Pattern library to use.
            logger: Optional structured logger.

        """
        self._library = library or PatternLibrary()
        self._logger = logger or structlog.get_logger(__name__)

    @property
    def library(self) -> PatternLibrary:
        """Get the pattern library."""
        return self._library

    def find_matches(
        self,
        board: list[list[int]],
        pattern_types: list[PatternType] | None = None,
        tags: list[str] | None = None,
        check_rotations: bool = True,
        check_flips: bool = True,
    ) -> list[PatternMatch]:
        """Find all pattern matches in a position.

        Args:
            board: Board state.
            pattern_types: Optional filter by types.
            tags: Optional filter by tags.
            check_rotations: Whether to check rotations.
            check_flips: Whether to check flips.

        Returns:
            List of pattern matches.

        """
        matches: list[PatternMatch] = []

        # Get patterns to check
        patterns = self._get_patterns_to_check(pattern_types, tags)

        if not patterns:
            return matches

        board_size = len(board)
        rotations = [0, 90, 180, 270] if check_rotations else [0]
        flips = [False, True] if check_flips else [False]

        for pattern in patterns:
            for y in range(board_size):
                for x in range(board_size):
                    for rotation in rotations:
                        for flip in flips:
                            if pattern.matches(board, x, y, rotation, flip):
                                matches.append(
                                    PatternMatch(
                                        pattern=pattern,
                                        x=x,
                                        y=y,
                                        rotation=rotation,
                                        flipped=flip,
                                    )
                                )

        return matches

    def find_pattern(
        self,
        board: list[list[int]],
        pattern_name: str,
        check_rotations: bool = True,
        check_flips: bool = True,
    ) -> list[PatternMatch]:
        """Find specific pattern in position.

        Args:
            board: Board state.
            pattern_name: Pattern to find.
            check_rotations: Whether to check rotations.
            check_flips: Whether to check flips.

        Returns:
            List of matches for this pattern.

        """
        pattern = self._library.get_pattern(pattern_name)
        if pattern is None:
            return []

        matches: list[PatternMatch] = []
        board_size = len(board)
        rotations = [0, 90, 180, 270] if check_rotations else [0]
        flips = [False, True] if check_flips else [False]

        for y in range(board_size):
            for x in range(board_size):
                for rotation in rotations:
                    for flip in flips:
                        if pattern.matches(board, x, y, rotation, flip):
                            matches.append(
                                PatternMatch(
                                    pattern=pattern,
                                    x=x,
                                    y=y,
                                    rotation=rotation,
                                    flipped=flip,
                                )
                            )

        return matches

    def _get_patterns_to_check(
        self,
        pattern_types: list[PatternType] | None,
        tags: list[str] | None,
    ) -> list[Pattern]:
        """Get list of patterns to check.

        Args:
            pattern_types: Optional type filter.
            tags: Optional tag filter.

        Returns:
            List of patterns to check.

        """
        if pattern_types is None and tags is None:
            return list(self._library._patterns.values())

        result: set[str] = set()

        if pattern_types:
            for pt in pattern_types:
                result.update(self._library._by_type.get(pt, []))

        if tags:
            for tag in tags:
                result.update(self._library._by_tag.get(tag, []))

        return [self._library._patterns[name] for name in result]

    def check_position(
        self,
        board: list[list[int]],
        x: int,
        y: int,
    ) -> list[PatternMatch]:
        """Check for patterns at a specific position.

        Args:
            board: Board state.
            x: X coordinate.
            y: Y coordinate.

        Returns:
            List of patterns matching at this position.

        """
        matches: list[PatternMatch] = []

        for pattern in self._library._patterns.values():
            for rotation in [0, 90, 180, 270]:
                for flip in [False, True]:
                    if pattern.matches(board, x, y, rotation, flip):
                        matches.append(
                            PatternMatch(
                                pattern=pattern,
                                x=x,
                                y=y,
                                rotation=rotation,
                                flipped=flip,
                            )
                        )

        return matches


def create_basic_library() -> PatternLibrary:
    """Create a library with basic patterns.

    Returns:
        PatternLibrary with common patterns.

    """
    library = PatternLibrary()

    # Empty triangle (bad shape)
    library.add_pattern(
        Pattern(
            name="empty_triangle",
            pattern_type=PatternType.SHAPE,
            stones={
                (0, 0): "B",
                (1, 0): "B",
                (0, 1): "B",
            },
            description="Empty triangle - inefficient shape",
            tags=["bad_shape", "beginner"],
            difficulty=1,
        )
    )

    # Tiger's mouth
    library.add_pattern(
        Pattern(
            name="tigers_mouth",
            pattern_type=PatternType.SHAPE,
            stones={
                (0, 0): "B",
                (2, 0): "B",
                (1, 1): "B",
            },
            description="Tiger's mouth - flexible shape with eye potential",
            tags=["good_shape", "connection"],
            difficulty=2,
        )
    )

    # Star point opening
    library.add_pattern(
        Pattern(
            name="star_point",
            pattern_type=PatternType.OPENING,
            stones={
                (3, 3): "B",  # 4-4 point
            },
            description="Star point opening - influence-oriented",
            tags=["opening", "fuseki"],
            difficulty=1,
        )
    )

    # 3-4 point
    library.add_pattern(
        Pattern(
            name="komoku",
            pattern_type=PatternType.OPENING,
            stones={
                (2, 3): "B",  # 3-4 point
            },
            description="Komoku (3-4 point) - balanced territory and influence",
            tags=["opening", "fuseki"],
            difficulty=1,
        )
    )

    return library
