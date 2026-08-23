"""SGF parser for reading game records.

Implements full SGF FF[4] specification parsing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import structlog

from src.games.sgf.config import SGFConfig
from src.games.sgf.node import SGFGameTree, SGFNode

logger = structlog.get_logger(__name__)


class SGFParseError(Exception):
    """Error during SGF parsing."""

    def __init__(self, message: str, position: int = -1, context: str = "") -> None:
        self.position = position
        self.context = context
        super().__init__(f"{message} at position {position}: ...{context}...")


class SGFParser:
    """Parser for SGF (Smart Game Format) files.

    Supports:
    - Full FF[4] specification
    - Multiple games in one file
    - Variations and game trees
    - All standard properties
    - Configurable strictness

    Example:
        parser = SGFParser()
        game = parser.parse_file("game.sgf")

        # Or from string
        game = parser.parse("(;GM[1]FF[4]SZ[19];B[pd];W[dd])")

    """

    # Token patterns
    _WHITESPACE = re.compile(r"\s+")
    _PROPERTY_NAME = re.compile(r"[A-Z]+")
    _PROPERTY_VALUE = re.compile(r"\[((?:[^\]\\]|\\.)*)\]")

    def __init__(self, config: SGFConfig | None = None) -> None:
        """Initialize parser.

        Args:
            config: Parser configuration (uses defaults if None)

        """
        self.config = config or SGFConfig(name="sgf_parser")
        self._text = ""
        self._pos = 0
        self._variation_count = 0

    def parse(self, text: str) -> SGFGameTree:
        """Parse SGF text into a game tree.

        Args:
            text: SGF format string

        Returns:
            Parsed game tree

        Raises:
            SGFParseError: If parsing fails and strict mode is enabled

        """
        self._text = text
        self._pos = 0
        self._variation_count = 0

        # Skip leading whitespace
        self._skip_whitespace()

        if self._pos >= len(self._text):
            if self.config.strict_parsing:
                raise SGFParseError("Empty SGF text", 0)
            return SGFGameTree()

        # Parse the game tree
        root = self._parse_game_tree()

        # Create game tree and extract board size
        tree = SGFGameTree(root=root)

        logger.debug(
            "sgf_parsed",
            moves=tree.count_moves(),
            nodes=tree.count_nodes(),
            board_size=tree.board_size,
        )

        return tree

    def parse_file(self, path: str | Path, encoding: str | None = None) -> SGFGameTree:
        """Parse an SGF file.

        Args:
            path: Path to SGF file
            encoding: Character encoding (uses config default if None)

        Returns:
            Parsed game tree

        """
        path = Path(path)
        enc = encoding or self.config.encoding

        logger.debug("parsing_sgf_file", path=str(path), encoding=enc)

        text = path.read_text(encoding=enc)
        return self.parse(text)

    def parse_multiple(self, text: str) -> Iterator[SGFGameTree]:
        """Parse multiple games from one SGF string.

        Args:
            text: SGF text potentially containing multiple games

        Yields:
            Game trees for each game found

        """
        self._text = text
        self._pos = 0

        while self._pos < len(self._text):
            self._skip_whitespace()
            if self._pos >= len(self._text):
                break

            if self._peek() != "(":
                self._pos += 1
                continue

            try:
                root = self._parse_game_tree()
                yield SGFGameTree(root=root)
            except SGFParseError as e:
                if self.config.strict_parsing:
                    raise
                logger.warning("sgf_parse_error", error=str(e))
                break

    def _parse_game_tree(self) -> SGFNode:
        """Parse a complete game tree starting with '('."""
        if not self._expect("("):
            self._error("Expected '(' at start of game tree")

        root = self._parse_sequence()

        # Parse variations
        while self._peek() == "(":
            self._variation_count += 1
            if self._variation_count > self.config.max_variations:
                logger.warning(
                    "max_variations_exceeded",
                    max=self.config.max_variations,
                )
                self._skip_to_matching_paren()
                continue

            # Find the last node in the sequence to attach variation
            last_node = root
            while last_node.children:
                last_node = last_node.children[0]

            variation_root = self._parse_game_tree()
            last_node.parent.add_child(variation_root) if last_node.parent else None

        if not self._expect(")"):
            self._error("Expected ')' at end of game tree")

        return root

    def _parse_sequence(self) -> SGFNode:
        """Parse a sequence of nodes starting with ';'."""
        root: SGFNode | None = None
        current: SGFNode | None = None

        while self._peek() == ";":
            self._advance()  # Skip ';'
            node = self._parse_node()

            if root is None:
                root = node
                current = node
            else:
                current.add_child(node)
                current = node

        if root is None:
            if self.config.strict_parsing:
                self._error("Expected at least one node in sequence")
            root = SGFNode()

        return root

    def _parse_node(self) -> SGFNode:
        """Parse a single node (properties until next ';', '(', or ')')."""
        node = SGFNode()

        while True:
            self._skip_whitespace()
            c = self._peek()

            if c in (";", "(", ")", ""):
                break

            # Parse property
            prop_name = self._parse_property_name()
            if not prop_name:
                if self.config.strict_parsing:
                    self._error(f"Expected property name, got '{c}'")
                self._advance()
                continue

            values = self._parse_property_values()
            node.properties[prop_name] = values

        return node

    def _parse_property_name(self) -> str:
        """Parse a property name (uppercase letters)."""
        self._skip_whitespace()
        match = self._PROPERTY_NAME.match(self._text, self._pos)
        if match:
            self._pos = match.end()
            return match.group()
        return ""

    def _parse_property_values(self) -> list[str]:
        """Parse property values (one or more [...] sequences)."""
        values = []

        while True:
            self._skip_whitespace()
            if self._peek() != "[":
                break

            self._advance()  # Skip '['
            value = self._parse_value_content()
            values.append(value)

            if not self._expect("]"):
                self._error("Expected ']' at end of property value")

        return values

    def _parse_value_content(self) -> str:
        """Parse content inside [...], handling escapes."""
        result = []
        escape = False

        while self._pos < len(self._text):
            c = self._text[self._pos]

            if escape:
                # Handle escape sequences
                if c == "\n":
                    pass  # Escaped newline is removed
                elif c == "\r":
                    pass  # Escaped CR is removed
                elif c in ("\\", "]", ":"):
                    result.append(c)
                else:
                    result.append(c)
                escape = False
            elif c == "\\":
                escape = True
            elif c == "]":
                break
            else:
                # Convert soft line breaks to spaces in text values
                if c in ("\r", "\n"):
                    result.append(" ")
                else:
                    result.append(c)

            self._pos += 1

        return "".join(result)

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _skip_to_matching_paren(self) -> None:
        """Skip to matching closing parenthesis."""
        depth = 1
        while self._pos < len(self._text) and depth > 0:
            c = self._text[self._pos]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            self._pos += 1

    def _peek(self) -> str:
        """Look at current character without advancing."""
        if self._pos >= len(self._text):
            return ""
        return self._text[self._pos]

    def _advance(self) -> str:
        """Advance to next character and return current."""
        if self._pos >= len(self._text):
            return ""
        c = self._text[self._pos]
        self._pos += 1
        return c

    def _expect(self, expected: str) -> bool:
        """Expect and consume a specific character."""
        self._skip_whitespace()
        if self._peek() == expected:
            self._advance()
            return True
        return False

    def _error(self, message: str) -> None:
        """Raise a parse error."""
        context_start = max(0, self._pos - 20)
        context_end = min(len(self._text), self._pos + 20)
        context = self._text[context_start:context_end]
        raise SGFParseError(message, self._pos, context)
