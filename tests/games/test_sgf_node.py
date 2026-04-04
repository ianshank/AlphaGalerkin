"""Tests for SGF node and game tree data structures."""

from __future__ import annotations

from src.games.sgf.node import SGFGameTree, SGFMove, SGFNode


class TestSGFMove:
    """Tests for SGFMove."""

    def test_create_move(self) -> None:
        move = SGFMove(color="B", x=3, y=15)
        assert move.color == "B"
        assert move.x == 3
        assert move.y == 15

    def test_is_pass_negative_coords(self) -> None:
        move = SGFMove(color="B", x=-1, y=-1)
        assert move.is_pass is True

    def test_is_pass_tt(self) -> None:
        move = SGFMove(color="B", x=0, y=0, sgf_coord="tt")
        assert move.is_pass is True

    def test_not_pass(self) -> None:
        move = SGFMove(color="B", x=3, y=3, sgf_coord="dd")
        assert move.is_pass is False

    def test_not_pass_with_normal_coord(self) -> None:
        move = SGFMove(color="W", x=15, y=3, sgf_coord="pd")
        assert move.is_pass is False

    def test_from_sgf_normal(self) -> None:
        move = SGFMove.from_sgf("B", "pd", board_size=19)
        assert move.color == "B"
        assert move.x == 15  # 'p' - 'a' = 15
        assert move.y == 3   # 'd' - 'a' = 3
        assert move.sgf_coord == "pd"

    def test_from_sgf_pass_empty(self) -> None:
        move = SGFMove.from_sgf("B", "", board_size=19)
        assert move.is_pass is True
        assert move.x == -1
        assert move.y == -1

    def test_from_sgf_pass_tt(self) -> None:
        move = SGFMove.from_sgf("W", "tt", board_size=19)
        assert move.is_pass is True

    def test_from_sgf_top_left(self) -> None:
        move = SGFMove.from_sgf("B", "aa", board_size=19)
        assert move.x == 0
        assert move.y == 0

    def test_from_sgf_bottom_right_9x9(self) -> None:
        move = SGFMove.from_sgf("B", "ii", board_size=9)
        assert move.x == 8
        assert move.y == 8

    def test_from_sgf_center_19x19(self) -> None:
        move = SGFMove.from_sgf("B", "jj", board_size=19)
        assert move.x == 9
        assert move.y == 9

    def test_from_sgf_single_char_treated_as_pass(self) -> None:
        """Single character coords are invalid and treated as pass."""
        move = SGFMove.from_sgf("B", "a", board_size=19)
        assert move.is_pass is True

    def test_to_sgf_normal(self) -> None:
        move = SGFMove(color="B", x=15, y=3)
        sgf = move.to_sgf(board_size=19)
        assert sgf == "pd"

    def test_to_sgf_top_left(self) -> None:
        move = SGFMove(color="B", x=0, y=0)
        sgf = move.to_sgf(board_size=19)
        assert sgf == "aa"

    def test_to_sgf_pass(self) -> None:
        move = SGFMove(color="B", x=-1, y=-1)
        sgf = move.to_sgf(board_size=19)
        assert sgf == ""

    def test_roundtrip_sgf(self) -> None:
        """from_sgf -> to_sgf should be identity for various coords."""
        for coord in ["aa", "dd", "pd", "dp", "ss"]:
            move = SGFMove.from_sgf("B", coord, board_size=19)
            assert move.to_sgf(board_size=19) == coord

    def test_roundtrip_sgf_pass(self) -> None:
        """Pass roundtrip."""
        move = SGFMove.from_sgf("B", "", board_size=19)
        assert move.to_sgf(board_size=19) == ""

    def test_to_gtp_normal(self) -> None:
        # x=3 ('d'), y=0 (top row) on 19x19 -> D19
        move = SGFMove(color="B", x=3, y=0)
        gtp = move.to_gtp(board_size=19)
        assert gtp == "D19"

    def test_to_gtp_skip_i(self) -> None:
        # x=8 should map to 'J' (since I is skipped)
        move = SGFMove(color="B", x=8, y=0)
        gtp = move.to_gtp(board_size=19)
        assert gtp[0] == "J"

    def test_to_gtp_pass(self) -> None:
        move = SGFMove(color="B", x=-1, y=-1)
        gtp = move.to_gtp(board_size=19)
        assert gtp == "pass"

    def test_to_gtp_bottom_row(self) -> None:
        move = SGFMove(color="B", x=0, y=18)
        gtp = move.to_gtp(board_size=19)
        assert gtp == "A1"

    def test_to_gtp_9x9(self) -> None:
        move = SGFMove(color="B", x=0, y=8)
        gtp = move.to_gtp(board_size=9)
        assert gtp == "A1"

    def test_str_normal(self) -> None:
        move = SGFMove(color="B", x=3, y=3, sgf_coord="dd")
        s = str(move)
        assert "B" in s
        assert "dd" in s

    def test_str_pass(self) -> None:
        move = SGFMove(color="W", x=-1, y=-1)
        s = str(move)
        assert "pass" in s

    def test_extended_coords_large_board(self) -> None:
        """Test coordinates for boards larger than 26."""
        move = SGFMove(color="B", x=26, y=26)
        sgf = move.to_sgf(board_size=30)
        assert sgf[0].isupper()
        assert sgf[1].isupper()


class TestSGFNode:
    """Tests for SGFNode."""

    def test_create_empty_node(self) -> None:
        node = SGFNode()
        assert node.properties == {}
        assert node.children == []
        assert node.parent is None

    def test_set_get_property(self) -> None:
        node = SGFNode()
        node.set_property("PB", "Player Black")
        assert node.get_property("PB") == "Player Black"

    def test_get_property_default(self) -> None:
        node = SGFNode()
        assert node.get_property("PB") is None
        assert node.get_property("PB", "default") == "default"

    def test_set_property_list(self) -> None:
        node = SGFNode()
        node.set_property("AB", ["dd", "pd", "dp"])
        values = node.get_property_list("AB")
        assert len(values) == 3
        assert "dd" in values

    def test_set_property_scalar_wraps_in_list(self) -> None:
        node = SGFNode()
        node.set_property("PB", "Test")
        assert node.properties["PB"] == ["Test"]

    def test_get_property_list_empty(self) -> None:
        node = SGFNode()
        values = node.get_property_list("AB")
        assert values == []

    def test_add_property_value(self) -> None:
        node = SGFNode()
        node.add_property_value("AB", "dd")
        node.add_property_value("AB", "pd")
        values = node.get_property_list("AB")
        assert len(values) == 2
        assert values[0] == "dd"
        assert values[1] == "pd"

    def test_add_property_value_creates_key(self) -> None:
        node = SGFNode()
        node.add_property_value("LB", "dd:A")
        assert node.get_property("LB") == "dd:A"

    def test_remove_property(self) -> None:
        node = SGFNode()
        node.set_property("PB", "Test")
        assert node.remove_property("PB") is True
        assert node.get_property("PB") is None

    def test_remove_nonexistent_property(self) -> None:
        node = SGFNode()
        assert node.remove_property("PB") is False

    def test_move_black(self) -> None:
        node = SGFNode()
        node.set_property("B", "dd")
        move = node.move
        assert move is not None
        assert move.color == "B"
        assert move.x == 3
        assert move.y == 3

    def test_move_white(self) -> None:
        node = SGFNode()
        node.set_property("W", "pd")
        move = node.move
        assert move is not None
        assert move.color == "W"
        assert move.x == 15
        assert move.y == 3

    def test_no_move(self) -> None:
        node = SGFNode()
        assert node.move is None

    def test_set_move(self) -> None:
        node = SGFNode()
        move = SGFMove(color="B", x=3, y=3)
        node.move = move
        assert node.has_move is True
        assert node.get_property("B") is not None

    def test_set_move_updates_property(self) -> None:
        node = SGFNode()
        move = SGFMove(color="W", x=15, y=3)
        node.move = move
        coord = node.get_property("W")
        assert coord == "pd"

    def test_clear_move(self) -> None:
        node = SGFNode()
        node.move = SGFMove(color="B", x=3, y=3)
        node.move = None
        assert node.has_move is False
        assert node.get_property("B") is None
        assert node.get_property("W") is None

    def test_has_move(self) -> None:
        node = SGFNode()
        assert node.has_move is False
        node.set_property("B", "dd")
        assert node.has_move is True

    def test_is_root(self) -> None:
        root = SGFNode()
        assert root.is_root is True
        child = root.new_child()
        assert child.is_root is False

    def test_is_leaf(self) -> None:
        node = SGFNode()
        assert node.is_leaf is True
        node.new_child()
        assert node.is_leaf is False

    def test_add_child(self) -> None:
        parent = SGFNode()
        child = SGFNode()
        parent.add_child(child)
        assert len(parent.children) == 1
        assert child.parent is parent

    def test_add_child_returns_child(self) -> None:
        parent = SGFNode()
        child = SGFNode()
        result = parent.add_child(child)
        assert result is child

    def test_new_child(self) -> None:
        parent = SGFNode()
        child = parent.new_child()
        assert child.parent is parent
        assert len(parent.children) == 1

    def test_multiple_children(self) -> None:
        parent = SGFNode()
        c1 = parent.new_child()
        c2 = parent.new_child()
        c3 = parent.new_child()
        assert len(parent.children) == 3
        assert parent.children[0] is c1
        assert parent.children[2] is c3

    def test_remove_child(self) -> None:
        parent = SGFNode()
        child = parent.new_child()
        assert parent.remove_child(child) is True
        assert len(parent.children) == 0
        assert child.parent is None

    def test_remove_nonexistent_child(self) -> None:
        parent = SGFNode()
        other = SGFNode()
        assert parent.remove_child(other) is False

    def test_depth(self) -> None:
        root = SGFNode()
        assert root.depth == 0
        child = root.new_child()
        assert child.depth == 1
        grandchild = child.new_child()
        assert grandchild.depth == 2

    def test_move_number(self) -> None:
        root = SGFNode()
        child = root.new_child()
        child.set_property("B", "dd")
        grandchild = child.new_child()
        grandchild.set_property("W", "pd")
        assert grandchild.move_number == 2

    def test_move_number_root_has_no_move(self) -> None:
        root = SGFNode()
        assert root.move_number == 0

    def test_move_number_skips_non_move_nodes(self) -> None:
        root = SGFNode()
        setup = root.new_child()
        setup.set_property("C", "setup node")
        m1 = setup.new_child()
        m1.set_property("B", "dd")
        assert m1.move_number == 1

    def test_get_root(self) -> None:
        root = SGFNode()
        child = root.new_child()
        grandchild = child.new_child()
        assert grandchild.get_root() is root

    def test_get_root_from_root(self) -> None:
        root = SGFNode()
        assert root.get_root() is root

    def test_get_path_to_root(self) -> None:
        root = SGFNode()
        child = root.new_child()
        grandchild = child.new_child()
        path = grandchild.get_path_to_root()
        assert len(path) == 3
        assert path[0] is grandchild
        assert path[1] is child
        assert path[-1] is root

    def test_comment(self) -> None:
        node = SGFNode()
        node.set_comment("Test comment")
        assert node.get_comment() == "Test comment"

    def test_empty_comment(self) -> None:
        node = SGFNode()
        assert node.get_comment() == ""

    def test_clear_comment(self) -> None:
        node = SGFNode()
        node.set_comment("Test")
        node.set_comment("")
        assert node.get_comment() == ""

    def test_iter(self) -> None:
        parent = SGFNode()
        c1 = parent.new_child()
        c2 = parent.new_child()
        children = list(parent)
        assert len(children) == 2
        assert c1 in children
        assert c2 in children

    def test_len(self) -> None:
        parent = SGFNode()
        assert len(parent) == 0
        parent.new_child()
        parent.new_child()
        assert len(parent) == 2

    def test_board_size_propagation(self) -> None:
        parent = SGFNode()
        parent._board_size = 9
        child = parent.new_child()
        assert child._board_size == 9

    def test_board_size_propagation_via_add_child(self) -> None:
        parent = SGFNode()
        parent._board_size = 13
        child = SGFNode()
        parent.add_child(child)
        assert child._board_size == 13

    def test_default_board_size(self) -> None:
        node = SGFNode()
        assert node._board_size == 19


class TestSGFGameTree:
    """Tests for SGFGameTree."""

    def test_create_empty_tree(self) -> None:
        tree = SGFGameTree()
        assert tree.root is not None
        assert tree.board_size == 19

    def test_board_size_from_root(self) -> None:
        root = SGFNode()
        root.set_property("SZ", "9")
        tree = SGFGameTree(root=root)
        assert tree.board_size == 9

    def test_board_size_from_root_13(self) -> None:
        root = SGFNode()
        root.set_property("SZ", "13")
        tree = SGFGameTree(root=root)
        assert tree.board_size == 13

    def test_set_board_size(self) -> None:
        tree = SGFGameTree()
        tree.board_size = 13
        assert tree.board_size == 13
        assert tree.root.get_property("SZ") == "13"

    def test_set_board_size_propagates(self) -> None:
        tree = SGFGameTree()
        c1 = tree.root.new_child()
        c2 = c1.new_child()
        tree.board_size = 9
        assert c2._board_size == 9

    def test_game_info(self) -> None:
        tree = SGFGameTree()
        tree.root.set_property("PB", "Lee Sedol")
        tree.root.set_property("PW", "AlphaGo")
        info = tree.game_info
        assert info.get("player_black") == "Lee Sedol"
        assert info.get("player_white") == "AlphaGo"

    def test_set_game_info(self) -> None:
        tree = SGFGameTree()
        tree.set_game_info(player_black="Black", player_white="White")
        assert tree.root.get_property("PB") == "Black"
        assert tree.root.get_property("PW") == "White"

    def test_set_game_info_unknown_field(self) -> None:
        """Unknown field names are silently ignored."""
        tree = SGFGameTree()
        tree.set_game_info(nonexistent_field="value")
        # Should not raise and no property added
        assert tree.root.get_property("nonexistent_field") is None

    def test_mainline_empty(self) -> None:
        tree = SGFGameTree()
        nodes = list(tree.mainline())
        assert len(nodes) == 1  # root only

    def test_mainline_with_moves(self) -> None:
        tree = SGFGameTree()
        tree.board_size = 9
        c1 = tree.root.new_child()
        c1.set_property("B", "dd")
        c2 = c1.new_child()
        c2.set_property("W", "pd")
        nodes = list(tree.mainline())
        assert len(nodes) == 3

    def test_mainline_follows_first_child(self) -> None:
        tree = SGFGameTree()
        c1 = tree.root.new_child()
        c1.set_property("B", "dd")
        # Add variation (second child)
        c2 = tree.root.new_child()
        c2.set_property("B", "pp")
        # Mainline should follow first child only
        moves = list(tree.mainline_moves())
        assert len(moves) == 1
        assert moves[0].sgf_coord == "dd"

    def test_mainline_moves(self) -> None:
        tree = SGFGameTree()
        tree.board_size = 9
        c1 = tree.root.new_child()
        c1.set_property("B", "dd")
        c2 = c1.new_child()
        c2.set_property("W", "pd")
        moves = list(tree.mainline_moves())
        assert len(moves) == 2
        assert moves[0].color == "B"
        assert moves[1].color == "W"

    def test_all_nodes(self) -> None:
        tree = SGFGameTree()
        c1 = tree.root.new_child()
        c2 = tree.root.new_child()  # variation
        c1.new_child()
        all_nodes = list(tree.all_nodes())
        assert len(all_nodes) == 4

    def test_count_nodes(self) -> None:
        tree = SGFGameTree()
        tree.root.new_child()
        tree.root.new_child()
        assert tree.count_nodes() == 3

    def test_count_moves(self) -> None:
        tree = SGFGameTree()
        c1 = tree.root.new_child()
        c1.set_property("B", "dd")
        c2 = c1.new_child()
        c2.set_property("W", "pd")
        assert tree.count_moves() == 2

    def test_count_moves_empty(self) -> None:
        tree = SGFGameTree()
        assert tree.count_moves() == 0

    def test_get_node_at_move(self) -> None:
        tree = SGFGameTree()
        tree.board_size = 9
        c1 = tree.root.new_child()
        c1.set_property("B", "dd")
        c2 = c1.new_child()
        c2.set_property("W", "pd")
        node = tree.get_node_at_move(1)
        assert node is c1
        node2 = tree.get_node_at_move(2)
        assert node2 is c2

    def test_get_node_at_move_not_found(self) -> None:
        tree = SGFGameTree()
        assert tree.get_node_at_move(99) is None

    def test_result(self) -> None:
        tree = SGFGameTree()
        tree.set_result("B+2.5")
        assert tree.get_result() == "B+2.5"

    def test_result_default(self) -> None:
        tree = SGFGameTree()
        assert tree.get_result() == ""

    def test_result_white_resign(self) -> None:
        tree = SGFGameTree()
        tree.set_result("W+R")
        assert tree.get_result() == "W+R"

    def test_komi(self) -> None:
        tree = SGFGameTree()
        tree.set_komi(6.5)
        assert tree.get_komi() == 6.5

    def test_komi_default(self) -> None:
        tree = SGFGameTree()
        assert tree.get_komi() == 0.0

    def test_komi_integer(self) -> None:
        tree = SGFGameTree()
        tree.set_komi(7.0)
        assert tree.get_komi() == 7.0

    def test_handicap(self) -> None:
        tree = SGFGameTree()
        tree.root.set_property("HA", "4")
        assert tree.get_handicap() == 4

    def test_handicap_default(self) -> None:
        tree = SGFGameTree()
        assert tree.get_handicap() == 0

    def test_handicap_stones(self) -> None:
        tree = SGFGameTree()
        tree.board_size = 9
        tree.root.set_property("AB", ["cc", "gc", "cg", "gg"])
        stones = tree.get_handicap_stones()
        assert len(stones) == 4
        # Verify coordinates are tuples of (x, y)
        for x, y in stones:
            assert isinstance(x, int)
            assert isinstance(y, int)

    def test_str(self) -> None:
        tree = SGFGameTree()
        tree.root.set_property("PB", "Black")
        tree.root.set_property("PW", "White")
        s = str(tree)
        assert "Black" in s
        assert "White" in s

    def test_board_size_propagation(self) -> None:
        tree = SGFGameTree()
        tree.board_size = 9
        c1 = tree.root.new_child()
        c2 = c1.new_child()
        assert c2._board_size == 9

    def test_invalid_board_size_string(self) -> None:
        """Invalid SZ property falls back to default."""
        root = SGFNode()
        root.set_property("SZ", "invalid")
        tree = SGFGameTree(root=root)
        assert tree.board_size == 19  # Default
