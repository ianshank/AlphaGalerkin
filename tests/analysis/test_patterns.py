"""Tests for pattern recognition."""

from __future__ import annotations

import pytest

from src.analysis.patterns import (
    Pattern,
    PatternLibrary,
    PatternMatch,
    PatternMatcher,
    PatternType,
    create_basic_library,
)


class TestPatternType:
    """Tests for PatternType enum."""

    def test_all_types_exist(self) -> None:
        """Test all pattern types exist."""
        assert PatternType.OPENING.value == "opening"
        assert PatternType.JOSEKI.value == "joseki"
        assert PatternType.TESUJI.value == "tesuji"
        assert PatternType.SHAPE.value == "shape"
        assert PatternType.LIFE_DEATH.value == "life_death"
        assert PatternType.ENDGAME.value == "endgame"
        assert PatternType.CUSTOM.value == "custom"


class TestPattern:
    """Tests for Pattern dataclass."""

    def test_initialization(self, sample_pattern: Pattern) -> None:
        """Test pattern initialization."""
        assert sample_pattern.name == "test_pattern"
        assert sample_pattern.pattern_type == PatternType.SHAPE
        assert len(sample_pattern.stones) == 3

    def test_matches_exact(self, sample_pattern: Pattern) -> None:
        """Test exact pattern matching."""
        # Create board with pattern
        board = [[0] * 9 for _ in range(9)]
        board[0][0] = 1  # B at (0,0)
        board[0][1] = 1  # B at (1,0)
        board[1][0] = 1  # B at (0,1)

        assert sample_pattern.matches(board, 0, 0, rotation=0, flip=False)

    def test_matches_fail(self, sample_pattern: Pattern) -> None:
        """Test pattern not matching."""
        board = [[0] * 9 for _ in range(9)]
        # Only two stones instead of three
        board[0][0] = 1
        board[0][1] = 1

        assert not sample_pattern.matches(board, 0, 0, rotation=0, flip=False)

    def test_matches_with_rotation(self) -> None:
        """Test pattern matching with rotation."""
        pattern = Pattern(
            name="test",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B", (1, 0): "B"},  # Horizontal pair
        )

        # Board with vertical pair
        board = [[0] * 9 for _ in range(9)]
        board[3][3] = 1
        board[4][3] = 1

        # Should match with 90 degree rotation
        assert pattern.matches(board, 3, 3, rotation=90, flip=False)

    def test_matches_with_flip(self) -> None:
        """Test pattern matching with flip."""
        pattern = Pattern(
            name="test",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B", (1, 0): "B", (1, 1): "B"},  # L shape
        )

        # Board with mirrored L
        board = [[0] * 9 for _ in range(9)]
        board[0][3] = 1
        board[0][2] = 1  # Flipped x
        board[1][2] = 1

        # Should match with flip
        assert pattern.matches(board, 3, 0, rotation=0, flip=True)

    def test_to_dict(self, sample_pattern: Pattern) -> None:
        """Test serialization to dict."""
        data = sample_pattern.to_dict()

        assert data["name"] == "test_pattern"
        assert data["pattern_type"] == "shape"
        assert len(data["stones"]) == 3

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "name": "loaded_pattern",
            "pattern_type": "joseki",
            "stones": {"0,0": "B", "1,0": "W"},
            "description": "Test pattern",
            "tags": ["test"],
            "difficulty": 3,
        }

        pattern = Pattern.from_dict(data)

        assert pattern.name == "loaded_pattern"
        assert pattern.pattern_type == PatternType.JOSEKI
        assert pattern.stones[(0, 0)] == "B"
        assert pattern.difficulty == 3


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_to_dict(self, sample_pattern: Pattern) -> None:
        """Test serialization to dict."""
        match = PatternMatch(
            pattern=sample_pattern,
            x=3,
            y=3,
            rotation=90,
            flipped=True,
        )

        data = match.to_dict()

        assert data["pattern_name"] == "test_pattern"
        assert data["x"] == 3
        assert data["y"] == 3
        assert data["rotation"] == 90
        assert data["flipped"] is True


class TestPatternLibrary:
    """Tests for PatternLibrary."""

    def test_initialization(
        self, pattern_library: PatternLibrary
    ) -> None:
        """Test library initialization."""
        assert len(pattern_library) == 0

    def test_add_pattern(
        self,
        pattern_library: PatternLibrary,
        sample_pattern: Pattern,
    ) -> None:
        """Test adding pattern."""
        pattern_library.add_pattern(sample_pattern)
        assert len(pattern_library) == 1

    def test_get_pattern(
        self,
        pattern_library: PatternLibrary,
        sample_pattern: Pattern,
    ) -> None:
        """Test getting pattern by name."""
        pattern_library.add_pattern(sample_pattern)

        found = pattern_library.get_pattern("test_pattern")
        assert found is not None
        assert found.name == "test_pattern"

        assert pattern_library.get_pattern("nonexistent") is None

    def test_get_by_type(
        self,
        pattern_library: PatternLibrary,
    ) -> None:
        """Test getting patterns by type."""
        shape_pattern = Pattern(
            name="shape1",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B"},
        )
        opening_pattern = Pattern(
            name="opening1",
            pattern_type=PatternType.OPENING,
            stones={(3, 3): "B"},
        )

        pattern_library.add_pattern(shape_pattern)
        pattern_library.add_pattern(opening_pattern)

        shapes = pattern_library.get_by_type(PatternType.SHAPE)
        assert len(shapes) == 1
        assert shapes[0].name == "shape1"

    def test_get_by_tag(
        self,
        pattern_library: PatternLibrary,
    ) -> None:
        """Test getting patterns by tag."""
        pattern1 = Pattern(
            name="pattern1",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B"},
            tags=["beginner", "shape"],
        )
        pattern2 = Pattern(
            name="pattern2",
            pattern_type=PatternType.SHAPE,
            stones={(1, 0): "B"},
            tags=["advanced", "shape"],
        )

        pattern_library.add_pattern(pattern1)
        pattern_library.add_pattern(pattern2)

        beginner = pattern_library.get_by_tag("beginner")
        assert len(beginner) == 1
        assert beginner[0].name == "pattern1"

        shapes = pattern_library.get_by_tag("shape")
        assert len(shapes) == 2

    def test_list_patterns(
        self,
        pattern_library: PatternLibrary,
        sample_pattern: Pattern,
    ) -> None:
        """Test listing all patterns."""
        pattern_library.add_pattern(sample_pattern)
        names = pattern_library.list_patterns()

        assert "test_pattern" in names

    def test_list_types(
        self,
        pattern_library: PatternLibrary,
    ) -> None:
        """Test listing pattern types."""
        pattern_library.add_pattern(Pattern(
            name="p1",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B"},
        ))

        types = pattern_library.list_types()
        assert PatternType.SHAPE in types

    def test_list_tags(
        self,
        pattern_library: PatternLibrary,
    ) -> None:
        """Test listing all tags."""
        pattern_library.add_pattern(Pattern(
            name="p1",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B"},
            tags=["tag1", "tag2"],
        ))

        tags = pattern_library.list_tags()
        assert "tag1" in tags
        assert "tag2" in tags

    def test_remove_pattern(
        self,
        pattern_library: PatternLibrary,
        sample_pattern: Pattern,
    ) -> None:
        """Test removing pattern."""
        pattern_library.add_pattern(sample_pattern)
        assert len(pattern_library) == 1

        result = pattern_library.remove_pattern("test_pattern")
        assert result is True
        assert len(pattern_library) == 0

        result = pattern_library.remove_pattern("nonexistent")
        assert result is False

    def test_to_dict_and_from_dict(
        self,
        pattern_library: PatternLibrary,
        sample_pattern: Pattern,
    ) -> None:
        """Test serialization roundtrip."""
        pattern_library.add_pattern(sample_pattern)

        data = pattern_library.to_dict()
        restored = PatternLibrary.from_dict(data)

        assert len(restored) == 1
        assert restored.get_pattern("test_pattern") is not None


class TestPatternMatcher:
    """Tests for PatternMatcher."""

    def test_initialization(
        self, pattern_matcher: PatternMatcher
    ) -> None:
        """Test matcher initialization."""
        assert pattern_matcher.library is not None

    def test_find_matches_empty(
        self,
        pattern_matcher: PatternMatcher,
        sample_board: list[list[int]],
    ) -> None:
        """Test finding matches with no patterns."""
        matches = pattern_matcher.find_matches(sample_board)
        assert len(matches) == 0

    def test_find_matches(
        self,
        pattern_matcher: PatternMatcher,
    ) -> None:
        """Test finding pattern matches."""
        # Add pattern to library
        pattern = Pattern(
            name="corner",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B"},
        )
        pattern_matcher.library.add_pattern(pattern)

        # Board with black stone in corner
        board = [[0] * 9 for _ in range(9)]
        board[0][0] = 1

        matches = pattern_matcher.find_matches(board)
        assert len(matches) >= 1

    def test_find_pattern_specific(
        self,
        pattern_matcher: PatternMatcher,
    ) -> None:
        """Test finding specific pattern."""
        pattern = Pattern(
            name="target",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B", (1, 0): "B"},
        )
        pattern_matcher.library.add_pattern(pattern)

        board = [[0] * 9 for _ in range(9)]
        board[3][3] = 1
        board[3][4] = 1

        matches = pattern_matcher.find_pattern(board, "target")
        assert len(matches) >= 1

    def test_find_pattern_nonexistent(
        self,
        pattern_matcher: PatternMatcher,
        sample_board: list[list[int]],
    ) -> None:
        """Test finding nonexistent pattern."""
        matches = pattern_matcher.find_pattern(sample_board, "nonexistent")
        assert len(matches) == 0

    def test_check_position(
        self,
        pattern_matcher: PatternMatcher,
    ) -> None:
        """Test checking patterns at specific position."""
        pattern = Pattern(
            name="corner",
            pattern_type=PatternType.SHAPE,
            stones={(0, 0): "B"},
        )
        pattern_matcher.library.add_pattern(pattern)

        board = [[0] * 9 for _ in range(9)]
        board[3][3] = 1

        matches = pattern_matcher.check_position(board, 3, 3)
        assert len(matches) >= 1


class TestCreateBasicLibrary:
    """Tests for create_basic_library factory."""

    def test_creates_library(self) -> None:
        """Test that basic library is created."""
        library = create_basic_library()

        assert len(library) > 0
        assert library.get_pattern("empty_triangle") is not None
        assert library.get_pattern("tigers_mouth") is not None

    def test_patterns_have_correct_types(self) -> None:
        """Test patterns have correct types."""
        library = create_basic_library()

        empty_triangle = library.get_pattern("empty_triangle")
        assert empty_triangle.pattern_type == PatternType.SHAPE

        star_point = library.get_pattern("star_point")
        assert star_point.pattern_type == PatternType.OPENING
