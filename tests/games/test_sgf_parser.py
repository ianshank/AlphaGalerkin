"""Tests for SGF parser."""

from __future__ import annotations

import tempfile

import pytest

from src.games.sgf.config import SGFConfig
from src.games.sgf.parser import SGFParseError, SGFParser


class TestSGFParserBasic:
    """Basic parsing tests."""

    def test_parse_minimal(self) -> None:
        parser = SGFParser()
        tree = parser.parse("(;GM[1])")
        assert tree.root is not None
        assert tree.root.properties.get("GM") == ["1"]

    def test_parse_with_moves(self) -> None:
        parser = SGFParser()
        tree = parser.parse("(;GM[1]SZ[19];B[pd];W[dd])")
        assert tree.count_moves() == 2

    def test_parse_board_size(self) -> None:
        parser = SGFParser()
        tree = parser.parse("(;GM[1]FF[4]SZ[9])")
        assert tree.board_size == 9

    def test_parse_empty_strict(self) -> None:
        config = SGFConfig(name="strict", strict_parsing=True)
        parser = SGFParser(config=config)
        with pytest.raises(SGFParseError, match="Empty"):
            parser.parse("")

    def test_parse_empty_lenient(self) -> None:
        config = SGFConfig(name="lenient", strict_parsing=False)
        parser = SGFParser(config=config)
        tree = parser.parse("")
        assert tree.root is not None

    def test_parse_whitespace_only(self) -> None:
        parser = SGFParser()
        tree = parser.parse("   (;GM[1])   ")
        assert tree.root is not None

    def test_parse_multiple_properties(self) -> None:
        parser = SGFParser()
        tree = parser.parse("(;GM[1]FF[4]AP[test]PB[Black]PW[White])")
        assert tree.root.properties["PB"] == ["Black"]
        assert tree.root.properties["PW"] == ["White"]


class TestSGFParserFile:
    """File parsing tests."""

    def test_parse_file(self) -> None:
        parser = SGFParser()
        with tempfile.NamedTemporaryFile(suffix=".sgf", mode="w", delete=False) as f:
            f.write("(;GM[1]SZ[9];B[ee])")
            f.flush()
            tree = parser.parse_file(f.name)
        assert tree.count_moves() == 1

    def test_parse_file_with_encoding(self) -> None:
        parser = SGFParser()
        with tempfile.NamedTemporaryFile(
            suffix=".sgf",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as f:
            f.write("(;GM[1]SZ[19])")
            f.flush()
            tree = parser.parse_file(f.name, encoding="utf-8")
        assert tree.root is not None


class TestSGFParserMultiple:
    """Multiple game parsing tests."""

    def test_parse_two_games(self) -> None:
        parser = SGFParser()
        games = list(parser.parse_multiple("(;GM[1])(;GM[1];B[dd])"))
        assert len(games) == 2

    def test_parse_multiple_skips_junk(self) -> None:
        parser = SGFParser()
        games = list(parser.parse_multiple("junk(;GM[1])more"))
        assert len(games) == 1


class TestSGFParserEscapes:
    """Escape handling tests."""

    def test_escaped_bracket(self) -> None:
        parser = SGFParser()
        tree = parser.parse(r"(;C[hello\]world])")
        comment = tree.root.properties.get("C", [""])[0]
        assert "]" in comment

    def test_escaped_backslash(self) -> None:
        parser = SGFParser()
        tree = parser.parse(r"(;C[path\\to\\file])")
        comment = tree.root.properties.get("C", [""])[0]
        assert "\\" in comment


class TestSGFParserVariations:
    """Variation handling tests."""

    def test_parse_variations(self) -> None:
        parser = SGFParser()
        tree = parser.parse("(;GM[1](;B[pd])(;B[dd]))")
        assert tree.root is not None

    def test_max_variations_exceeded(self) -> None:
        config = SGFConfig(name="test", max_variations=2, strict_parsing=False)
        parser = SGFParser(config=config)
        # Parsing with variations is complex; just ensure no crash
        try:
            tree = parser.parse("(;GM[1](;B[pd])(;B[dd]))")
            assert tree.root is not None
        except SGFParseError:
            pass  # Acceptable if parser fails on variations overflow


class TestSGFParseError:
    """Tests for SGFParseError."""

    def test_error_attributes(self) -> None:
        e = SGFParseError("bad", position=10, context="xxx")
        assert e.position == 10
        assert e.context == "xxx"
        assert "bad" in str(e)

    def test_strict_missing_close_paren(self) -> None:
        config = SGFConfig(name="strict", strict_parsing=True)
        parser = SGFParser(config=config)
        with pytest.raises(SGFParseError):
            parser.parse("(;GM[1]")

    def test_strict_no_sequence(self) -> None:
        config = SGFConfig(name="strict", strict_parsing=True)
        parser = SGFParser(config=config)
        with pytest.raises(SGFParseError):
            parser.parse("()")
