"""SGF writer for creating game records.

Produces properly formatted SGF FF[4] output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

import structlog

from src.games.sgf.config import SGFConfig
from src.games.sgf.node import SGFGameTree, SGFNode

logger = structlog.get_logger(__name__)


class SGFWriter:
    """Writer for SGF (Smart Game Format) files.

    Features:
    - FF[4] compliant output
    - Pretty printing with line wrapping
    - Property ordering for readability
    - Proper escape sequence handling

    Example:
        writer = SGFWriter()
        sgf_text = writer.write(game_tree)

        # Or write to file
        writer.write_file(game_tree, "output.sgf")

    """

    # Property ordering for readability
    _ROOT_PROPS_ORDER = ["FF", "GM", "SZ", "CA", "AP", "ST"]
    _GAME_INFO_ORDER = [
        "GN", "EV", "RO", "DT", "PC", "PB", "BR", "PW", "WR",
        "BT", "WT", "RU", "TM", "OT", "KM", "HA", "RE", "GC",
        "AN", "US", "SO", "CP", "ON",
    ]
    _MOVE_ORDER = ["B", "W", "KO", "MN"]
    _SETUP_ORDER = ["AB", "AW", "AE", "PL"]
    _ANNOTATION_ORDER = ["C", "N", "GB", "GW", "DM", "UC", "HO", "V"]

    def __init__(self, config: SGFConfig | None = None) -> None:
        """Initialize writer.

        Args:
            config: Writer configuration (uses defaults if None)

        """
        self.config = config or SGFConfig(name="sgf_writer")

    def write(self, tree: SGFGameTree) -> str:
        """Write game tree to SGF string.

        Args:
            tree: Game tree to write

        Returns:
            SGF format string

        """
        lines: list[str] = []
        self._write_node(tree.root, lines, is_root=True)

        if self.config.pretty_print:
            return "\n".join(lines) + "\n"
        return "".join(lines)

    def write_file(
        self,
        tree: SGFGameTree,
        path: str | Path,
        encoding: str | None = None,
    ) -> None:
        """Write game tree to SGF file.

        Args:
            tree: Game tree to write
            path: Output file path
            encoding: Character encoding (uses config default if None)

        """
        path = Path(path)
        enc = encoding or self.config.encoding

        logger.debug("writing_sgf_file", path=str(path), encoding=enc)

        sgf_text = self.write(tree)
        path.write_text(sgf_text, encoding=enc)

    def write_to_stream(self, tree: SGFGameTree, stream: TextIO) -> None:
        """Write game tree to a text stream.

        Args:
            tree: Game tree to write
            stream: Text stream to write to

        """
        sgf_text = self.write(tree)
        stream.write(sgf_text)

    def _write_node(
        self,
        node: SGFNode,
        lines: list[str],
        is_root: bool = False,
        depth: int = 0,
    ) -> None:
        """Write a node and its children.

        Args:
            node: Node to write
            lines: Lines to append to
            is_root: Whether this is the root node
            depth: Current nesting depth

        """
        # Start game tree for root
        if is_root:
            lines.append("(")

        # Write node
        node_str = self._format_node(node, is_root=is_root)
        if self.config.pretty_print:
            lines.append(node_str)
        else:
            if lines:
                lines[-1] += node_str
            else:
                lines.append(node_str)

        # Handle children
        if len(node.children) == 0:
            pass  # Leaf node
        elif len(node.children) == 1:
            # Single child - continue sequence
            self._write_node(node.children[0], lines, is_root=False, depth=depth)
        else:
            # Multiple children - create variations
            for child in node.children:
                if self.config.pretty_print:
                    lines.append("(")
                else:
                    lines[-1] += "("
                self._write_node(child, lines, is_root=False, depth=depth + 1)
                if self.config.pretty_print:
                    lines.append(")")
                else:
                    lines[-1] += ")"

        # End game tree for root
        if is_root:
            lines.append(")")

    def _format_node(self, node: SGFNode, is_root: bool = False) -> str:
        """Format a single node as SGF string.

        Args:
            node: Node to format
            is_root: Whether this is the root node

        Returns:
            SGF string for this node

        """
        parts = [";"]

        # Order properties
        ordered_keys = self._order_properties(node.properties.keys(), is_root)

        for key in ordered_keys:
            values = node.properties.get(key, [])
            if not values:
                continue

            # Skip properties based on config
            if not self.config.include_comments and key == "C":
                continue
            if not self.config.include_timing and key in ("BL", "WL", "OB", "OW"):
                continue

            # Format property
            prop_str = self._format_property(key, values)
            parts.append(prop_str)

        return "".join(parts)

    def _format_property(self, key: str, values: list[str]) -> str:
        """Format a property as SGF string.

        Args:
            key: Property key
            values: Property values

        Returns:
            SGF property string

        """
        escaped_values = [self._escape_value(v) for v in values]
        return key + "".join(f"[{v}]" for v in escaped_values)

    def _escape_value(self, value: str) -> str:
        """Escape special characters in a property value.

        Args:
            value: Raw value

        Returns:
            Escaped value safe for SGF

        """
        # Escape backslashes first, then brackets
        value = value.replace("\\", "\\\\")
        value = value.replace("]", "\\]")
        return value

    def _order_properties(
        self,
        keys: set[str] | list[str],
        is_root: bool,
    ) -> list[str]:
        """Order properties for readability.

        Args:
            keys: Property keys to order
            is_root: Whether this is the root node

        Returns:
            Ordered list of keys

        """
        result = []
        remaining = set(keys)

        # Define ordering based on node type
        if is_root:
            orders = [
                self._ROOT_PROPS_ORDER,
                self._GAME_INFO_ORDER,
                self._SETUP_ORDER,
                self._MOVE_ORDER,
                self._ANNOTATION_ORDER,
            ]
        else:
            orders = [
                self._MOVE_ORDER,
                self._SETUP_ORDER,
                self._ANNOTATION_ORDER,
            ]

        # Add properties in order
        for order_list in orders:
            for key in order_list:
                if key in remaining:
                    result.append(key)
                    remaining.remove(key)

        # Add any remaining properties alphabetically
        result.extend(sorted(remaining))

        return result


def write_game_tree(
    tree: SGFGameTree,
    path: str | Path | None = None,
    pretty: bool = True,
) -> str:
    """Convenience function to write a game tree.

    Args:
        tree: Game tree to write
        path: Optional file path to write to
        pretty: Whether to pretty-print the output

    Returns:
        SGF string

    """
    config = SGFConfig(name="quick_writer", pretty_print=pretty)
    writer = SGFWriter(config)
    sgf_text = writer.write(tree)

    if path:
        Path(path).write_text(sgf_text)

    return sgf_text
