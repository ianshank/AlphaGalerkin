"""SGF (Smart Game Format) support for Go games.

This module provides:
- SGF parsing with full property support
- SGF writing with proper formatting
- Game tree traversal and manipulation
- Integration with AlphaGalerkin game state

Example:
    from src.games.sgf import SGFParser, SGFWriter, SGFNode

    # Parse an SGF file
    parser = SGFParser()
    game_tree = parser.parse_file("game.sgf")

    # Access game info
    print(game_tree.root.get_property("PB"))  # Black player
    print(game_tree.root.get_property("PW"))  # White player

    # Iterate through moves
    for node in game_tree.mainline():
        if node.move:
            print(f"Move: {node.move}")

    # Write to SGF
    writer = SGFWriter()
    sgf_text = writer.write(game_tree)
"""

from src.games.sgf.parser import SGFParser
from src.games.sgf.writer import SGFWriter
from src.games.sgf.node import SGFNode, SGFGameTree
from src.games.sgf.config import SGFConfig
from src.games.sgf.converter import SGFConverter

__all__ = [
    "SGFParser",
    "SGFWriter",
    "SGFNode",
    "SGFGameTree",
    "SGFConfig",
    "SGFConverter",
]
