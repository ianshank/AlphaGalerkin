"""Configuration for SGF module.

Uses the template-based configuration pattern for consistency.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field

from src.templates.config import BaseModuleConfig


class SGFFileFormat(str, Enum):
    """SGF file format versions."""

    FF1 = "1"
    FF3 = "3"
    FF4 = "4"  # Current standard


class SGFGameType(str, Enum):
    """Game types supported in SGF."""

    GO = "1"
    OTHELLO = "2"
    CHESS = "3"
    GOMOKU = "4"
    NINE_MENS_MORRIS = "5"
    BACKGAMMON = "6"
    CHINESE_CHESS = "7"
    SHOGI = "8"
    LINES_OF_ACTION = "9"
    ATAXX = "10"
    HEX = "11"
    JUNGLE = "12"
    NEUTRON = "13"
    PHILOSOPHERS_FOOTBALL = "14"
    QUADRATURE = "15"
    TRAX = "16"
    TANTRIX = "17"
    AMAZONS = "18"
    OCTI = "19"
    GESS = "20"


class SGFConfig(BaseModuleConfig):
    """Configuration for SGF parsing and writing.

    Attributes:
        file_format: SGF file format version to use.
        game_type: Game type identifier.
        default_board_size: Default board size if not specified.
        default_komi: Default komi if not specified.
        encoding: Character encoding for SGF files.
        strict_parsing: If True, raise errors on invalid SGF.
        preserve_unknown_properties: Keep properties not recognized.
        include_comments: Include comment properties in output.
        pretty_print: Format output for readability.
        max_variations: Maximum number of variations to parse.
    """

    name: str = Field(default="sgf", description="Configuration name")

    # Format settings
    file_format: SGFFileFormat = Field(
        default=SGFFileFormat.FF4,
        description="SGF file format version",
    )
    game_type: SGFGameType = Field(
        default=SGFGameType.GO,
        description="Game type identifier",
    )

    # Game defaults
    default_board_size: int = Field(
        default=19,
        ge=3,
        le=25,
        description="Default board size",
    )
    default_komi: float = Field(
        default=7.5,
        ge=-100.0,
        le=100.0,
        description="Default komi value",
    )
    default_handicap: int = Field(
        default=0,
        ge=0,
        le=9,
        description="Default handicap stones",
    )

    # Parsing settings
    encoding: Literal["utf-8", "latin-1", "ascii", "gb2312", "euc-kr", "shift-jis"] = Field(
        default="utf-8",
        description="Character encoding for SGF files",
    )
    strict_parsing: bool = Field(
        default=False,
        description="Raise errors on invalid SGF syntax",
    )
    preserve_unknown_properties: bool = Field(
        default=True,
        description="Keep unrecognized properties",
    )
    max_variations: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum variations to parse",
    )

    # Writing settings
    include_comments: bool = Field(
        default=True,
        description="Include comment properties",
    )
    pretty_print: bool = Field(
        default=True,
        description="Format output for readability",
    )
    line_width: int = Field(
        default=80,
        ge=40,
        le=200,
        description="Maximum line width for pretty printing",
    )
    include_timing: bool = Field(
        default=False,
        description="Include time-related properties",
    )


# Standard SGF property definitions
SGF_PROPERTIES = {
    # Root properties
    "FF": ("file_format", "root"),
    "GM": ("game_type", "root"),
    "SZ": ("board_size", "root"),
    "CA": ("charset", "root"),
    "AP": ("application", "root"),
    "ST": ("style", "root"),
    # Game info
    "PB": ("player_black", "game_info"),
    "PW": ("player_white", "game_info"),
    "BR": ("black_rank", "game_info"),
    "WR": ("white_rank", "game_info"),
    "BT": ("black_team", "game_info"),
    "WT": ("white_team", "game_info"),
    "DT": ("date", "game_info"),
    "EV": ("event", "game_info"),
    "GN": ("game_name", "game_info"),
    "GC": ("game_comment", "game_info"),
    "ON": ("opening", "game_info"),
    "PC": ("place", "game_info"),
    "RE": ("result", "game_info"),
    "RO": ("round", "game_info"),
    "RU": ("rules", "game_info"),
    "SO": ("source", "game_info"),
    "TM": ("time_limit", "game_info"),
    "OT": ("overtime", "game_info"),
    "KM": ("komi", "game_info"),
    "HA": ("handicap", "game_info"),
    "AN": ("annotator", "game_info"),
    "CP": ("copyright", "game_info"),
    "US": ("user", "game_info"),
    # Move properties
    "B": ("black_move", "move"),
    "W": ("white_move", "move"),
    "KO": ("ko", "move"),
    "MN": ("move_number", "move"),
    # Setup properties
    "AB": ("add_black", "setup"),
    "AW": ("add_white", "setup"),
    "AE": ("add_empty", "setup"),
    "PL": ("player_to_play", "setup"),
    # Node annotation
    "C": ("comment", "annotation"),
    "DM": ("even_position", "annotation"),
    "GB": ("good_for_black", "annotation"),
    "GW": ("good_for_white", "annotation"),
    "HO": ("hotspot", "annotation"),
    "N": ("node_name", "annotation"),
    "UC": ("unclear", "annotation"),
    "V": ("value", "annotation"),
    # Move annotation
    "BM": ("bad_move", "move_annotation"),
    "DO": ("doubtful", "move_annotation"),
    "IT": ("interesting", "move_annotation"),
    "TE": ("tesuji", "move_annotation"),
    # Markup
    "AR": ("arrow", "markup"),
    "CR": ("circle", "markup"),
    "DD": ("dim", "markup"),
    "LB": ("label", "markup"),
    "LN": ("line", "markup"),
    "MA": ("mark", "markup"),
    "SL": ("selected", "markup"),
    "SQ": ("square", "markup"),
    "TR": ("triangle", "markup"),
    # Timing
    "BL": ("black_time_left", "timing"),
    "WL": ("white_time_left", "timing"),
    "OB": ("black_stones_left", "timing"),
    "OW": ("white_stones_left", "timing"),
    # Miscellaneous
    "FG": ("figure", "misc"),
    "PM": ("print_mode", "misc"),
    "VW": ("view", "misc"),
}
