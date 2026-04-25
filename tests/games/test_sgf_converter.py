"""Tests for SGF converter between SGF and AlphaGalerkin game state."""

from __future__ import annotations

import pytest

from src.games.sgf.config import SGFConfig
from src.games.sgf.converter import SGFConverter
from src.games.sgf.node import SGFGameTree, SGFMove, SGFNode


class TestSGFConverterInit:
    """Tests for SGFConverter initialization."""

    def test_create_default(self) -> None:
        converter = SGFConverter()
        assert converter.config is not None
        assert converter.config.name == "sgf_converter"

    def test_custom_config(self) -> None:
        config = SGFConfig(name="custom", default_board_size=9)
        converter = SGFConverter(config=config)
        assert converter.config.default_board_size == 9


class TestParseSGF:
    """Tests for parsing SGF text."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_parse_simple_sgf(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[ce])"
        tree = converter.parse_sgf(sgf)
        assert tree.board_size == 9
        moves = list(tree.mainline_moves())
        assert len(moves) == 2
        assert moves[0].color == "B"
        assert moves[1].color == "W"

    def test_parse_extracts_coordinates(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[ce])"
        tree = converter.parse_sgf(sgf)
        moves = list(tree.mainline_moves())
        assert moves[0].x == 4  # 'e' - 'a'
        assert moves[0].y == 4
        assert moves[1].x == 2  # 'c' - 'a'
        assert moves[1].y == 4

    def test_parse_game_info(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9]PB[Alice]PW[Bob]RE[B+2.5]KM[6.5];B[ee])"
        tree = converter.parse_sgf(sgf)
        info = tree.game_info
        assert info.get("player_black") == "Alice"
        assert info.get("player_white") == "Bob"
        assert info.get("result") == "B+2.5"

    def test_parse_empty_game(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9])"
        tree = converter.parse_sgf(sgf)
        assert tree.board_size == 9
        moves = list(tree.mainline_moves())
        assert len(moves) == 0


class TestWriteSGF:
    """Tests for writing SGF text."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_write_empty_tree(self, converter: SGFConverter) -> None:
        tree = SGFGameTree()
        tree.board_size = 9
        tree.root.set_property("GM", "1")
        tree.root.set_property("FF", "4")
        sgf_text = converter.write_sgf(tree)
        assert isinstance(sgf_text, str)
        assert "SZ[9]" in sgf_text

    def test_write_tree_with_moves(self, converter: SGFConverter) -> None:
        tree = SGFGameTree()
        tree.board_size = 9
        tree.root.set_property("FF", "4")
        tree.root.set_property("GM", "1")
        c1 = tree.root.new_child()
        c1.move = SGFMove(color="B", x=4, y=4)
        sgf_text = converter.write_sgf(tree)
        assert "B[ee]" in sgf_text

    def test_roundtrip_parse_write(self, converter: SGFConverter) -> None:
        original = "(;GM[1]FF[4]SZ[9];B[ee];W[ce])"
        tree = converter.parse_sgf(original)
        written = converter.write_sgf(tree)
        assert len(written) > 0
        # Verify the written SGF contains key properties
        assert "SZ[9]" in written
        assert "B[ee]" in written
        assert "W[ce]" in written
        # Verify original tree moves are intact
        moves = list(tree.mainline_moves())
        assert len(moves) == 2


class TestMoveSequenceConversion:
    """Tests for move sequence conversions."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_to_move_sequence(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[ce])"
        tree = converter.parse_sgf(sgf)
        moves = converter.to_move_sequence(tree)
        assert len(moves) == 2
        assert moves[0] == ("B", 4, 4)
        assert moves[1] == ("W", 2, 4)

    def test_to_move_sequence_empty(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9])"
        tree = converter.parse_sgf(sgf)
        moves = converter.to_move_sequence(tree)
        assert len(moves) == 0

    def test_to_move_sequence_with_pass(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[])"
        tree = converter.parse_sgf(sgf)
        moves = converter.to_move_sequence(tree)
        assert len(moves) == 2
        assert moves[1] == ("W", -1, -1)

    def test_from_move_sequence_basic(self, converter: SGFConverter) -> None:
        moves = [("B", 4, 4), ("W", 2, 4), ("B", 6, 6)]
        tree = converter.from_move_sequence(moves, board_size=9)
        assert tree.board_size == 9
        result_moves = list(tree.mainline_moves())
        assert len(result_moves) == 3
        assert result_moves[0].color == "B"
        assert result_moves[0].x == 4
        assert result_moves[0].y == 4

    def test_from_move_sequence_empty(self, converter: SGFConverter) -> None:
        tree = converter.from_move_sequence([], board_size=9)
        assert tree.board_size == 9
        assert tree.count_moves() == 0

    def test_from_move_sequence_with_pass(self, converter: SGFConverter) -> None:
        moves = [("B", 4, 4), ("W", -1, -1)]
        tree = converter.from_move_sequence(moves, board_size=9)
        result_moves = list(tree.mainline_moves())
        assert len(result_moves) == 2
        assert result_moves[1].is_pass is True

    def test_from_move_sequence_with_game_info(self, converter: SGFConverter) -> None:
        moves = [("B", 4, 4)]
        game_info = {"player_black": "Alice", "player_white": "Bob"}
        tree = converter.from_move_sequence(moves, board_size=9, game_info=game_info)
        assert tree.root.get_property("PB") == "Alice"
        assert tree.root.get_property("PW") == "Bob"

    def test_from_move_sequence_sets_standard_properties(self, converter: SGFConverter) -> None:
        moves = [("B", 0, 0)]
        tree = converter.from_move_sequence(moves, board_size=19)
        assert tree.root.get_property("FF") == "4"
        assert tree.root.get_property("GM") == "1"
        assert tree.root.get_property("SZ") == "19"

    def test_roundtrip_move_sequence(self, converter: SGFConverter) -> None:
        original = [("B", 4, 4), ("W", 2, 4), ("B", 6, 6)]
        tree = converter.from_move_sequence(original, board_size=9)
        recovered = converter.to_move_sequence(tree)
        assert recovered == original

    def test_roundtrip_move_sequence_with_pass(self, converter: SGFConverter) -> None:
        original = [("B", 4, 4), ("W", -1, -1), ("B", 2, 2)]
        tree = converter.from_move_sequence(original, board_size=9)
        recovered = converter.to_move_sequence(tree)
        assert recovered == original


class TestBoardSizeHandling:
    """Tests for different board sizes."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_9x9_board(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9];B[ee])"
        tree = converter.parse_sgf(sgf)
        assert tree.board_size == 9

    def test_13x13_board(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[13];B[gg])"
        tree = converter.parse_sgf(sgf)
        assert tree.board_size == 13

    def test_19x19_board(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[19];B[pd])"
        tree = converter.parse_sgf(sgf)
        assert tree.board_size == 19

    def test_from_move_sequence_sets_board_size(self, converter: SGFConverter) -> None:
        tree = converter.from_move_sequence([], board_size=13)
        assert tree.board_size == 13
        assert tree.root.get_property("SZ") == "13"

    def test_move_encoding_preserves_coords_9x9(self, converter: SGFConverter) -> None:
        moves = [("B", 4, 4)]
        tree = converter.from_move_sequence(moves, board_size=9)
        result = converter.to_move_sequence(tree)
        assert result[0] == ("B", 4, 4)

    def test_move_encoding_preserves_coords_19x19(self, converter: SGFConverter) -> None:
        moves = [("B", 15, 3)]  # "pd"
        tree = converter.from_move_sequence(moves, board_size=19)
        result = converter.to_move_sequence(tree)
        assert result[0] == ("B", 15, 3)


class TestIterPositions:
    """Tests for iterating positions."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_iter_positions_empty(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9])"
        tree = converter.parse_sgf(sgf)
        positions = list(converter.iter_positions(tree))
        assert len(positions) == 1
        _, history = positions[0]
        assert history == []

    def test_iter_positions_with_moves(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[ce])"
        tree = converter.parse_sgf(sgf)
        positions = list(converter.iter_positions(tree))
        assert len(positions) >= 1
        # First position has empty move history
        _, history = positions[0]
        assert len(history) == 0

    def test_iter_positions_accumulates_history(self, converter: SGFConverter) -> None:
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[ce];B[ec])"
        tree = converter.parse_sgf(sgf)
        positions = list(converter.iter_positions(tree))
        # Last position should have accumulated moves
        _, last_history = positions[-1]
        # History at the last node contains all previous moves
        assert len(last_history) >= 2


class TestAnalysis:
    """Tests for analysis annotations."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_add_analysis_value(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_analysis(node, value=0.75)
        comment = node.get_comment()
        assert "Win rate" in comment
        assert node.get_property("V") is not None

    def test_add_analysis_value_negative(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_analysis(node, value=-0.5)
        comment = node.get_comment()
        assert "Win rate" in comment
        # -0.5 -> 25% win rate
        assert "25.0%" in comment

    def test_add_analysis_policy(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        policy = {(4, 4): 0.5, (2, 2): 0.3, (6, 6): 0.2}
        converter.add_analysis(node, policy=policy)
        comment = node.get_comment()
        assert "Top moves" in comment

    def test_add_analysis_pv(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        pv = [(4, 4), (2, 2), (6, 6)]
        converter.add_analysis(node, principal_variation=pv)
        comment = node.get_comment()
        assert "PV" in comment

    def test_add_analysis_comment(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_analysis(node, comment="Test annotation")
        assert "Test annotation" in node.get_comment()

    def test_add_analysis_preserves_existing_comment(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        node.set_comment("Existing comment")
        converter.add_analysis(node, value=0.5)
        comment = node.get_comment()
        assert "Existing comment" in comment
        assert "Win rate" in comment

    def test_add_analysis_no_data_no_change(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_analysis(node)
        assert node.get_comment() == ""


class TestMoveQuality:
    """Tests for move quality marking."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_mark_bad_move(self, converter: SGFConverter) -> None:
        node = SGFNode()
        converter.mark_move_quality(node, "bad")
        assert node.get_property("BM") == "1"

    def test_mark_doubtful(self, converter: SGFConverter) -> None:
        node = SGFNode()
        converter.mark_move_quality(node, "doubtful")
        assert node.get_property("DO") == "1"

    def test_mark_interesting(self, converter: SGFConverter) -> None:
        node = SGFNode()
        converter.mark_move_quality(node, "interesting")
        assert node.get_property("IT") == "1"

    def test_mark_tesuji(self, converter: SGFConverter) -> None:
        node = SGFNode()
        converter.mark_move_quality(node, "tesuji")
        assert node.get_property("TE") == "1"

    def test_mark_good_no_property(self, converter: SGFConverter) -> None:
        node = SGFNode()
        converter.mark_move_quality(node, "good")
        # "good" has no special marking (None in map)
        assert node.get_property("BM") is None
        assert node.get_property("TE") is None


class TestMarkers:
    """Tests for adding markers to nodes."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_add_label(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_label(node, 4, 4, "A")
        labels = node.get_property_list("LB")
        assert len(labels) == 1
        assert "A" in labels[0]

    def test_add_triangle(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_triangle(node, 4, 4)
        markers = node.get_property_list("TR")
        assert len(markers) == 1

    def test_add_circle(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_circle(node, 4, 4)
        markers = node.get_property_list("CR")
        assert len(markers) == 1

    def test_add_square(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_square(node, 4, 4)
        markers = node.get_property_list("SQ")
        assert len(markers) == 1

    def test_add_mark(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_mark(node, 4, 4)
        markers = node.get_property_list("MA")
        assert len(markers) == 1

    def test_multiple_markers_same_type(self, converter: SGFConverter) -> None:
        node = SGFNode()
        node._board_size = 9
        converter.add_triangle(node, 0, 0)
        converter.add_triangle(node, 1, 1)
        markers = node.get_property_list("TR")
        assert len(markers) == 2


class TestCreateAnalysisTree:
    """Tests for creating analysis trees."""

    @pytest.fixture
    def converter(self) -> SGFConverter:
        return SGFConverter()

    def test_add_analysis_to_mainline(self, converter: SGFConverter) -> None:
        """Test adding analysis annotations directly to tree nodes."""
        tree = converter.from_move_sequence(
            [("B", 4, 4), ("W", 2, 4)],
            board_size=9,
        )
        evaluations = [{"value": 0.1}, {"value": 0.3}]
        # Add analysis to each mainline node directly
        for i, node in enumerate(tree.mainline()):
            if i < len(evaluations):
                converter.add_analysis(
                    node,
                    value=evaluations[i].get("value"),
                )
        # Verify annotations were added
        nodes = list(tree.mainline())
        assert nodes[0].get_property("V") is not None

    def test_analysis_preserves_move_count(self, converter: SGFConverter) -> None:
        tree = converter.from_move_sequence(
            [("B", 4, 4), ("W", 2, 4)],
            board_size=9,
        )
        original_count = tree.count_moves()
        # Add analysis to nodes
        for node in tree.mainline():
            converter.add_analysis(node, value=0.5)
        assert tree.count_moves() == original_count

    def test_analysis_on_empty_game(self, converter: SGFConverter) -> None:
        tree = converter.from_move_sequence([], board_size=9)
        converter.add_analysis(tree.root, value=0.0)
        assert tree.root.get_property("V") is not None
