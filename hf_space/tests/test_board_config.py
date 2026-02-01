"""Tests for board configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.board import (
    BoardRenderConfig,
    BoardSize,
    CoordinateLabelConfig,
    GTP_LETTERS,
    KOMI_BY_SIZE,
    SpaceConfig,
    STAR_POINTS_BY_SIZE,
    get_column_letter,
    get_default_space_config,
)


class TestBoardSize:
    """Tests for BoardSize enum."""

    def test_board_size_values(self) -> None:
        """Test BoardSize enum has correct values."""
        assert BoardSize.SIZE_9 == 9
        assert BoardSize.SIZE_13 == 13
        assert BoardSize.SIZE_19 == 19

    def test_board_size_iteration(self) -> None:
        """Test BoardSize can be iterated."""
        sizes = list(BoardSize)
        assert len(sizes) == 3
        assert 9 in [s.value for s in sizes]


class TestKomiBySize:
    """Tests for komi values per board size."""

    def test_standard_komi_values(self) -> None:
        """Test standard komi values are defined."""
        assert KOMI_BY_SIZE[9] == 5.5
        assert KOMI_BY_SIZE[13] == 6.5
        assert KOMI_BY_SIZE[19] == 7.5

    def test_all_supported_sizes_have_komi(self) -> None:
        """Test all supported sizes have komi defined."""
        for size in [9, 13, 19]:
            assert size in KOMI_BY_SIZE
            assert isinstance(KOMI_BY_SIZE[size], float)

    def test_komi_ordering(self) -> None:
        """Test komi values increase with board size."""
        assert KOMI_BY_SIZE[9] < KOMI_BY_SIZE[13] < KOMI_BY_SIZE[19]


class TestStarPoints:
    """Tests for star point definitions."""

    def test_9x9_star_points(self) -> None:
        """Test 9x9 has 5 star points including tengen."""
        points = STAR_POINTS_BY_SIZE[9]
        assert len(points) == 5
        assert (4, 4) in points  # Tengen (center)

    def test_13x13_star_points(self) -> None:
        """Test 13x13 has 5 star points."""
        points = STAR_POINTS_BY_SIZE[13]
        assert len(points) == 5
        assert (6, 6) in points  # Tengen

    def test_19x19_star_points(self) -> None:
        """Test 19x19 has 9 star points."""
        points = STAR_POINTS_BY_SIZE[19]
        assert len(points) == 9
        assert (9, 9) in points  # Tengen

    def test_star_points_within_bounds(self) -> None:
        """Test all star points are within board bounds."""
        for size, points in STAR_POINTS_BY_SIZE.items():
            for row, col in points:
                assert 0 <= row < size
                assert 0 <= col < size


class TestGTPLetters:
    """Tests for GTP column letters."""

    def test_skips_i(self) -> None:
        """Test that 'I' is not in GTP letters."""
        assert "I" not in GTP_LETTERS

    def test_has_correct_length(self) -> None:
        """Test has enough letters for 25x25 board."""
        assert len(GTP_LETTERS) >= 25

    def test_starts_with_a(self) -> None:
        """Test starts with 'A'."""
        assert GTP_LETTERS[0] == "A"

    def test_consecutive_except_i(self) -> None:
        """Test letters are consecutive except for 'I'."""
        expected = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
        assert GTP_LETTERS == expected


class TestGetColumnLetter:
    """Tests for get_column_letter function."""

    def test_first_column(self) -> None:
        """Test first column is 'A'."""
        assert get_column_letter(0, skip_i=True) == "A"
        assert get_column_letter(0, skip_i=False) == "A"

    def test_skip_i_behavior(self) -> None:
        """Test behavior around 'I'."""
        # Column 8 should be 'J' when skipping 'I'
        assert get_column_letter(8, skip_i=True) == "J"
        # Column 8 should be 'I' when not skipping
        assert get_column_letter(8, skip_i=False) == "I"

    def test_all_columns_valid(self) -> None:
        """Test all column indices produce valid letters."""
        for col in range(19):
            letter = get_column_letter(col, skip_i=True)
            assert letter.isalpha()
            assert letter.isupper()


class TestCoordinateLabelConfig:
    """Tests for coordinate label configuration."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CoordinateLabelConfig()
        assert config.show_labels is True
        assert config.label_style == "letters"
        assert config.skip_i is True
        assert config.font_color == "black"

    def test_font_size_bounds(self) -> None:
        """Test font size must be within bounds."""
        with pytest.raises(ValidationError):
            CoordinateLabelConfig(font_size=5)  # Below minimum

        with pytest.raises(ValidationError):
            CoordinateLabelConfig(font_size=25)  # Above maximum

    def test_valid_font_sizes(self) -> None:
        """Test valid font sizes are accepted."""
        config = CoordinateLabelConfig(font_size=10)
        assert config.font_size == 10

    def test_label_style_options(self) -> None:
        """Test label style options."""
        config_letters = CoordinateLabelConfig(label_style="letters")
        assert config_letters.label_style == "letters"

        config_numbers = CoordinateLabelConfig(label_style="numbers")
        assert config_numbers.label_style == "numbers"

    def test_invalid_label_style(self) -> None:
        """Test invalid label style is rejected."""
        with pytest.raises(ValidationError):
            CoordinateLabelConfig(label_style="invalid")

    def test_padding_bounds(self) -> None:
        """Test padding must be within bounds."""
        with pytest.raises(ValidationError):
            CoordinateLabelConfig(padding=0.2)  # Below minimum

        with pytest.raises(ValidationError):
            CoordinateLabelConfig(padding=2.0)  # Above maximum


class TestBoardRenderConfig:
    """Tests for board render configuration."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BoardRenderConfig()
        assert config.board_color == "#e3c586"
        assert config.stone_radius == 0.45
        assert config.line_color == "black"

    def test_coordinate_labels_included(self) -> None:
        """Test coordinate labels config is included."""
        config = BoardRenderConfig()
        assert isinstance(config.coordinate_labels, CoordinateLabelConfig)

    def test_stone_radius_bounds(self) -> None:
        """Test stone radius bounds."""
        with pytest.raises(ValidationError):
            BoardRenderConfig(stone_radius=0.2)  # Below minimum

        with pytest.raises(ValidationError):
            BoardRenderConfig(stone_radius=0.6)  # Above maximum

    def test_line_width_bounds(self) -> None:
        """Test line width bounds."""
        with pytest.raises(ValidationError):
            BoardRenderConfig(line_width=0.3)  # Below minimum

    def test_figure_size_customization(self) -> None:
        """Test figure size can be customized."""
        config = BoardRenderConfig(figure_size=(8.0, 8.0))
        assert config.figure_size == (8.0, 8.0)


class TestSpaceConfig:
    """Tests for space configuration."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SpaceConfig()
        assert config.default_board_size == 9
        assert config.supported_sizes == [9, 13, 19]
        assert config.mcts_simulations == 60

    def test_get_komi(self) -> None:
        """Test komi retrieval."""
        config = SpaceConfig()
        assert config.get_komi(9) == 5.5
        assert config.get_komi(13) == 6.5
        assert config.get_komi(19) == 7.5

    def test_get_komi_unknown_size(self) -> None:
        """Test komi for unknown size returns default."""
        config = SpaceConfig()
        # Unknown sizes should return default 7.5
        assert config.get_komi(11) == 7.5

    def test_get_star_points(self) -> None:
        """Test star point retrieval."""
        config = SpaceConfig()
        assert len(config.get_star_points(9)) == 5
        assert len(config.get_star_points(13)) == 5
        assert len(config.get_star_points(19)) == 9

    def test_get_star_points_unknown_size(self) -> None:
        """Test star points for unknown size returns empty list."""
        config = SpaceConfig()
        assert config.get_star_points(7) == []

    def test_unsupported_size_rejected(self) -> None:
        """Test unsupported sizes are rejected."""
        with pytest.raises(ValidationError):
            SpaceConfig(supported_sizes=[7])

    def test_default_must_be_in_supported(self) -> None:
        """Test default size must be in supported sizes."""
        with pytest.raises(ValidationError):
            SpaceConfig(default_board_size=19, supported_sizes=[9, 13])

    def test_empty_supported_sizes_rejected(self) -> None:
        """Test empty supported sizes is rejected."""
        with pytest.raises(ValidationError):
            SpaceConfig(supported_sizes=[])

    def test_mcts_simulations_bounds(self) -> None:
        """Test MCTS simulations bounds."""
        with pytest.raises(ValidationError):
            SpaceConfig(mcts_simulations=5)  # Below minimum

        with pytest.raises(ValidationError):
            SpaceConfig(mcts_simulations=600)  # Above maximum

    def test_render_config_included(self) -> None:
        """Test render config is included."""
        config = SpaceConfig()
        assert isinstance(config.render, BoardRenderConfig)

    def test_supported_sizes_normalized(self) -> None:
        """Test supported sizes are sorted and deduplicated."""
        config = SpaceConfig(supported_sizes=[19, 9, 13, 9])
        assert config.supported_sizes == [9, 13, 19]


class TestGetDefaultSpaceConfig:
    """Tests for get_default_space_config function."""

    def test_returns_valid_config(self) -> None:
        """Test returns a valid SpaceConfig."""
        config = get_default_space_config()
        assert isinstance(config, SpaceConfig)
        assert config.default_board_size in config.supported_sizes

    def test_returns_new_instance(self) -> None:
        """Test returns new instance each call."""
        config1 = get_default_space_config()
        config2 = get_default_space_config()
        assert config1 is not config2
