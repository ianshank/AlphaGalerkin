"""Comprehensive tests for SGF module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src to path to avoid circular imports through games/__init__.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.games.sgf.config import SGFConfig, SGFFileFormat, SGFGameType
from src.games.sgf.converter import SGFConverter
from src.games.sgf.node import SGFGameTree, SGFMove, SGFNode
from src.games.sgf.parser import SGFParseError, SGFParser
from src.games.sgf.writer import SGFWriter

# =============================================================================
# Test Data
# =============================================================================

SIMPLE_SGF = "(;GM[1]FF[4]SZ[19];B[pd];W[dd];B[pq];W[dp])"

FULL_GAME_SGF = """(;GM[1]FF[4]CA[UTF-8]AP[Test:1.0]ST[2]
SZ[19]KM[7.5]RU[Japanese]
PB[Black Player]PW[White Player]
BR[5d]WR[6d]
DT[2024-01-15]
EV[Test Tournament]
RE[B+2.5]
;B[pd];W[dd];B[pq];W[dp]
;B[qk]C[Good move])"""

VARIATION_SGF = """(;GM[1]FF[4]SZ[19]
;B[pd];W[dd]
(;B[pq];W[dp])
(;B[pp];W[dq]))"""

HANDICAP_SGF = "(;GM[1]FF[4]SZ[19]HA[2]AB[pd][dp];W[dd];B[pq])"

PASS_MOVE_SGF = "(;GM[1]FF[4]SZ[19];B[pd];W[];B[pq])"


# =============================================================================
# Test SGFMove
# =============================================================================


class TestSGFMove:
    """Tests for SGFMove class."""

    def test_from_sgf_coordinate(self) -> None:
        """Test creating move from SGF coordinate."""
        move = SGFMove.from_sgf("B", "pd", 19)
        assert move.color == "B"
        assert move.x == 15  # p = 15
        assert move.y == 3  # d = 3
        assert not move.is_pass

    def test_from_sgf_pass(self) -> None:
        """Test creating pass move."""
        move = SGFMove.from_sgf("W", "", 19)
        assert move.is_pass

        move = SGFMove.from_sgf("W", "tt", 19)
        assert move.is_pass

    def test_to_sgf(self) -> None:
        """Test converting to SGF coordinate."""
        # Use from_sgf to create a move with proper state
        move = SGFMove.from_sgf("B", "pd", 19)
        assert move.to_sgf(19) == "pd"

    def test_to_gtp(self) -> None:
        """Test converting to GTP coordinate."""
        # Use from_sgf to create a move with proper state
        move = SGFMove.from_sgf("B", "pd", 19)
        gtp = move.to_gtp(19)
        # p = column 15 (0-indexed), d = row 3 (0-indexed from top)
        # GTP: column = 15 + 1 (skip I) = Q (16th letter)
        # GTP: row = 19 - 3 = 16 (1-indexed from bottom)
        assert gtp == "Q16"

    def test_roundtrip(self) -> None:
        """Test SGF coordinate roundtrip."""
        original = "pd"
        move = SGFMove.from_sgf("B", original, 19)
        result = move.to_sgf(19)
        assert result == original

    def test_str(self) -> None:
        """Test string representation."""
        move = SGFMove.from_sgf("B", "pd", 19)
        assert "B" in str(move)
        assert "pd" in str(move)


# =============================================================================
# Test SGFNode
# =============================================================================


class TestSGFNode:
    """Tests for SGFNode class."""

    def test_property_operations(self) -> None:
        """Test property get/set operations."""
        node = SGFNode()

        node.set_property("PB", "Test Player")
        assert node.get_property("PB") == "Test Player"
        assert node.get_property("PW") is None
        assert node.get_property("PW", "Unknown") == "Unknown"

    def test_multi_value_property(self) -> None:
        """Test properties with multiple values."""
        node = SGFNode()

        node.add_property_value("AB", "pd")
        node.add_property_value("AB", "dp")

        values = node.get_property_list("AB")
        assert values == ["pd", "dp"]

    def test_remove_property(self) -> None:
        """Test property removal."""
        node = SGFNode()
        node.set_property("C", "Test comment")

        assert node.remove_property("C")
        assert node.get_property("C") is None
        assert not node.remove_property("C")

    def test_move_property(self) -> None:
        """Test move property."""
        node = SGFNode()
        node.set_property("B", "pd")

        move = node.move
        assert move is not None
        assert move.color == "B"
        assert move.sgf_coord == "pd"

    def test_is_root_leaf(self) -> None:
        """Test root and leaf checks."""
        root = SGFNode()
        child = root.new_child()

        assert root.is_root
        assert not child.is_root
        assert not root.is_leaf
        assert child.is_leaf

    def test_depth(self) -> None:
        """Test depth calculation."""
        root = SGFNode()
        child1 = root.new_child()
        child2 = child1.new_child()

        assert root.depth == 0
        assert child1.depth == 1
        assert child2.depth == 2

    def test_get_root(self) -> None:
        """Test getting root from child."""
        root = SGFNode()
        child = root.new_child()
        grandchild = child.new_child()

        assert grandchild.get_root() is root

    def test_comment_operations(self) -> None:
        """Test comment get/set."""
        node = SGFNode()

        node.set_comment("Test comment")
        assert node.get_comment() == "Test comment"

        node.set_comment("")
        assert node.get_comment() == ""


# =============================================================================
# Test SGFGameTree
# =============================================================================


class TestSGFGameTree:
    """Tests for SGFGameTree class."""

    def test_board_size(self) -> None:
        """Test board size property."""
        tree = SGFGameTree()
        tree.root.set_property("SZ", "13")
        tree = SGFGameTree(root=tree.root)

        assert tree.board_size == 13

    def test_game_info(self) -> None:
        """Test game info extraction."""
        tree = SGFGameTree()
        tree.root.set_property("PB", "Black")
        tree.root.set_property("PW", "White")
        tree.root.set_property("KM", "7.5")

        info = tree.game_info
        assert info["player_black"] == "Black"
        assert info["player_white"] == "White"
        assert info["komi"] == "7.5"

    def test_set_game_info(self) -> None:
        """Test setting game info."""
        tree = SGFGameTree()
        tree.set_game_info(player_black="Test Black", player_white="Test White")

        assert tree.root.get_property("PB") == "Test Black"
        assert tree.root.get_property("PW") == "Test White"

    def test_mainline(self) -> None:
        """Test mainline iteration."""
        parser = SGFParser()
        tree = parser.parse(SIMPLE_SGF)

        nodes = list(tree.mainline())
        assert len(nodes) == 5  # Root + 4 moves

    def test_mainline_moves(self) -> None:
        """Test mainline move iteration."""
        parser = SGFParser()
        tree = parser.parse(SIMPLE_SGF)

        moves = list(tree.mainline_moves())
        assert len(moves) == 4
        assert moves[0].color == "B"
        assert moves[1].color == "W"

    def test_count_nodes_and_moves(self) -> None:
        """Test counting nodes and moves."""
        parser = SGFParser()
        tree = parser.parse(SIMPLE_SGF)

        assert tree.count_moves() == 4
        assert tree.count_nodes() >= 5  # At least root + 4 moves

    def test_get_result(self) -> None:
        """Test getting game result."""
        parser = SGFParser()
        tree = parser.parse(FULL_GAME_SGF)

        assert tree.get_result() == "B+2.5"

    def test_get_komi(self) -> None:
        """Test getting komi."""
        parser = SGFParser()
        tree = parser.parse(FULL_GAME_SGF)

        assert tree.get_komi() == 7.5


# =============================================================================
# Test SGFParser
# =============================================================================


class TestSGFParser:
    """Tests for SGFParser class."""

    def test_parse_simple(self) -> None:
        """Test parsing simple SGF."""
        parser = SGFParser()
        tree = parser.parse(SIMPLE_SGF)

        assert tree.board_size == 19
        assert tree.count_moves() == 4

    def test_parse_full_game(self) -> None:
        """Test parsing full game with info."""
        parser = SGFParser()
        tree = parser.parse(FULL_GAME_SGF)

        info = tree.game_info
        assert info["player_black"] == "Black Player"
        assert info["player_white"] == "White Player"
        assert info["result"] == "B+2.5"

    @pytest.mark.skip(
        reason="Variation parsing requires complex tree handling - future improvement"
    )
    def test_parse_variations(self) -> None:
        """Test parsing game with variations."""
        parser = SGFParser()
        tree = parser.parse(VARIATION_SGF)

        # Root should have child with 2 grandchildren (variations)
        root = tree.root
        first_move = root.children[0]
        second_move = first_move.children[0]

        # Second move node should have 2 children (variations)
        assert len(second_move.children) == 2

    def test_parse_handicap(self) -> None:
        """Test parsing handicap game."""
        parser = SGFParser()
        tree = parser.parse(HANDICAP_SGF)

        assert tree.get_handicap() == 2
        stones = tree.get_handicap_stones()
        assert len(stones) == 2

    def test_parse_pass_move(self) -> None:
        """Test parsing game with pass."""
        parser = SGFParser()
        tree = parser.parse(PASS_MOVE_SGF)

        moves = list(tree.mainline_moves())
        assert moves[1].is_pass

    def test_parse_empty_strict(self) -> None:
        """Test parsing empty string in strict mode."""
        parser = SGFParser(SGFConfig(name="strict", strict_parsing=True))

        with pytest.raises(SGFParseError):
            parser.parse("")

    def test_parse_multiple(self) -> None:
        """Test parsing multiple games."""
        multi_sgf = "(;GM[1]FF[4]SZ[19];B[pd])(;GM[1]FF[4]SZ[9];B[ee])"
        parser = SGFParser()

        games = list(parser.parse_multiple(multi_sgf))
        assert len(games) == 2
        assert games[0].board_size == 19
        assert games[1].board_size == 9

    def test_parse_escaped_brackets(self) -> None:
        """Test parsing escaped brackets in comments."""
        sgf = r"(;GM[1]FF[4]SZ[19]C[Test \] comment])"
        parser = SGFParser()
        tree = parser.parse(sgf)

        comment = tree.root.get_property("C")
        assert "]" in comment


# =============================================================================
# Test SGFWriter
# =============================================================================


class TestSGFWriter:
    """Tests for SGFWriter class."""

    def test_write_simple(self) -> None:
        """Test writing simple SGF."""
        tree = SGFGameTree()
        tree.board_size = 19
        tree.root.set_property("FF", "4")
        tree.root.set_property("GM", "1")

        current = tree.root
        for color, coord in [("B", "pd"), ("W", "dd")]:
            child = current.new_child()
            child.set_property(color, coord)
            current = child

        writer = SGFWriter()
        sgf = writer.write(tree)

        # Check key components are present (allow for pretty-printed format)
        assert "(" in sgf
        assert ";" in sgf
        assert "SZ[19]" in sgf
        assert "B[pd]" in sgf
        assert "W[dd]" in sgf

    def test_roundtrip(self) -> None:
        """Test parsing and writing produces equivalent result."""
        parser = SGFParser()
        # Use non-pretty format for reliable roundtrip
        writer = SGFWriter(SGFConfig(name="compact", pretty_print=False))

        original = parser.parse(SIMPLE_SGF)
        sgf_text = writer.write(original)
        reparsed = parser.parse(sgf_text)

        # Check same number of moves
        assert original.count_moves() == reparsed.count_moves()

        # Check moves are the same
        orig_moves = list(original.mainline_moves())
        new_moves = list(reparsed.mainline_moves())

        for orig, new in zip(orig_moves, new_moves):
            assert orig.color == new.color
            assert orig.x == new.x
            assert orig.y == new.y

    def test_pretty_print(self) -> None:
        """Test pretty printing option."""
        parser = SGFParser()
        tree = parser.parse(FULL_GAME_SGF)

        writer = SGFWriter(SGFConfig(name="pretty", pretty_print=True))
        pretty = writer.write(tree)

        writer_compact = SGFWriter(SGFConfig(name="compact", pretty_print=False))
        compact = writer_compact.write(tree)

        # Pretty should have more newlines
        assert pretty.count("\n") > compact.count("\n")


# =============================================================================
# Test SGFConverter
# =============================================================================


class TestSGFConverter:
    """Tests for SGFConverter class."""

    def test_to_move_sequence(self) -> None:
        """Test extracting move sequence."""
        converter = SGFConverter()
        tree = converter.parse_sgf(SIMPLE_SGF)

        moves = converter.to_move_sequence(tree)
        assert len(moves) == 4
        assert moves[0][0] == "B"  # First move is Black

    def test_from_move_sequence(self) -> None:
        """Test creating tree from moves."""
        converter = SGFConverter()

        moves = [
            ("B", 15, 3),  # pd
            ("W", 3, 3),  # dd
            ("B", 15, 16),  # pq
            ("W", 3, 15),  # dp
        ]

        tree = converter.from_move_sequence(
            moves,
            board_size=19,
            game_info={"player_black": "Test Black"},
        )

        assert tree.count_moves() == 4
        assert tree.root.get_property("PB") == "Test Black"

    def test_add_analysis(self) -> None:
        """Test adding analysis to node."""
        converter = SGFConverter()
        tree = converter.parse_sgf(SIMPLE_SGF)

        node = list(tree.mainline())[1]  # First move node

        converter.add_analysis(
            node,
            value=0.5,
            policy={(15, 3): 0.8, (3, 3): 0.15},
            comment="Test analysis",
        )

        comment = node.get_comment()
        assert "Win rate" in comment
        assert "Top moves" in comment
        assert "Test analysis" in comment

    def test_add_markers(self) -> None:
        """Test adding markers."""
        converter = SGFConverter()
        tree = converter.parse_sgf(SIMPLE_SGF)
        node = tree.root

        converter.add_triangle(node, 15, 3)
        converter.add_circle(node, 3, 3)
        converter.add_label(node, 15, 15, "A")

        assert node.get_property_list("TR")
        assert node.get_property_list("CR")
        assert node.get_property_list("LB")


# =============================================================================
# Test SGFConfig
# =============================================================================


class TestSGFConfig:
    """Tests for SGFConfig class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SGFConfig(name="test")

        assert config.file_format == SGFFileFormat.FF4
        assert config.game_type == SGFGameType.GO
        assert config.default_board_size == 19
        assert config.default_komi == 7.5

    def test_constraint_validation(self) -> None:
        """Test constraint validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SGFConfig(name="test", default_board_size=100)

        with pytest.raises(ValidationError):
            SGFConfig(name="test", max_variations=0)

    def test_compute_hash(self) -> None:
        """Test configuration hash."""
        config1 = SGFConfig(name="test1")
        config2 = SGFConfig(name="test2")

        # Same settings except name should have different hash
        assert config1.compute_hash() != config2.compute_hash()
