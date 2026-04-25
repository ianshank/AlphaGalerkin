"""Tests for SGF writer."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from src.games.sgf.config import SGFConfig
from src.games.sgf.node import SGFGameTree, SGFNode
from src.games.sgf.writer import SGFWriter, write_game_tree


def _simple_tree() -> SGFGameTree:
    """Create a simple game tree for testing."""
    root = SGFNode()
    root.properties = {"GM": ["1"], "FF": ["4"], "SZ": ["9"]}
    child1 = SGFNode()
    child1.properties = {"B": ["pd"]}
    root.add_child(child1)
    child2 = SGFNode()
    child2.properties = {"W": ["dd"]}
    child1.add_child(child2)
    return SGFGameTree(root=root)


class TestSGFWriterBasic:
    """Basic writing tests."""

    def test_write_simple(self) -> None:
        writer = SGFWriter()
        tree = _simple_tree()
        text = writer.write(tree)
        assert "(" in text
        assert "GM[1]" in text
        assert "SZ[9]" in text

    def test_write_not_pretty(self) -> None:
        config = SGFConfig(name="compact", pretty_print=False)
        writer = SGFWriter(config)
        tree = _simple_tree()
        text = writer.write(tree)
        assert "\n" not in text.strip()

    def test_write_pretty(self) -> None:
        config = SGFConfig(name="pretty", pretty_print=True)
        writer = SGFWriter(config)
        tree = _simple_tree()
        text = writer.write(tree)
        assert "\n" in text

    def test_write_file(self) -> None:
        writer = SGFWriter()
        tree = _simple_tree()
        with tempfile.NamedTemporaryFile(suffix=".sgf", delete=False) as f:
            writer.write_file(tree, f.name)
            text = Path(f.name).read_text()
        assert "GM[1]" in text

    def test_write_to_stream(self) -> None:
        writer = SGFWriter()
        tree = _simple_tree()
        stream = io.StringIO()
        writer.write_to_stream(tree, stream)
        text = stream.getvalue()
        assert "GM[1]" in text

    def test_escape_brackets(self) -> None:
        writer = SGFWriter()
        root = SGFNode()
        root.properties = {"C": ["hello]world"]}
        tree = SGFGameTree(root=root)
        text = writer.write(tree)
        assert "hello\\]world" in text

    def test_escape_backslash(self) -> None:
        writer = SGFWriter()
        root = SGFNode()
        root.properties = {"C": ["path\\to"]}
        tree = SGFGameTree(root=root)
        text = writer.write(tree)
        assert "path\\\\to" in text

    def test_skip_comments(self) -> None:
        config = SGFConfig(name="no_comments", include_comments=False)
        writer = SGFWriter(config)
        root = SGFNode()
        root.properties = {"GM": ["1"], "C": ["skip me"]}
        tree = SGFGameTree(root=root)
        text = writer.write(tree)
        assert "skip me" not in text

    def test_skip_timing(self) -> None:
        config = SGFConfig(name="no_timing", include_timing=False)
        writer = SGFWriter(config)
        root = SGFNode()
        root.properties = {"B": ["pd"], "BL": ["300"]}
        tree = SGFGameTree(root=root)
        text = writer.write(tree)
        assert "BL" not in text

    def test_write_variations(self) -> None:
        writer = SGFWriter()
        root = SGFNode()
        root.properties = {"GM": ["1"]}
        var1 = SGFNode()
        var1.properties = {"B": ["pd"]}
        var2 = SGFNode()
        var2.properties = {"B": ["dd"]}
        root.add_child(var1)
        root.add_child(var2)
        tree = SGFGameTree(root=root)
        text = writer.write(tree)
        assert "B[pd]" in text
        assert "B[dd]" in text


class TestWriteGameTree:
    """Tests for write_game_tree convenience function."""

    def test_to_string(self) -> None:
        tree = _simple_tree()
        text = write_game_tree(tree)
        assert "GM[1]" in text

    def test_to_file(self) -> None:
        tree = _simple_tree()
        with tempfile.NamedTemporaryFile(suffix=".sgf", delete=False) as f:
            write_game_tree(tree, path=f.name)
            text = Path(f.name).read_text()
        assert "GM[1]" in text
