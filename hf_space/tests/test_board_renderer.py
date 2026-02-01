"""Tests for board renderer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.board import BoardRenderConfig, CoordinateLabelConfig, GTP_LETTERS
from src.rendering.board_renderer import BoardRenderer, render_board
from src.tools.gtp import SimpleGoGame


class TestBoardRenderer:
    """Tests for BoardRenderer class."""

    def test_initialization_default(self) -> None:
        """Test initialization with default config."""
        renderer = BoardRenderer()
        assert renderer.config is not None

    def test_initialization_custom_config(
        self,
        render_config: BoardRenderConfig,
    ) -> None:
        """Test initialization with custom config."""
        renderer = BoardRenderer(render_config)
        assert renderer.config == render_config

    def test_render_empty_board_9x9(
        self,
        renderer: BoardRenderer,
        game_9x9: SimpleGoGame,
    ) -> None:
        """Test rendering empty 9x9 board produces valid image."""
        image = renderer.render(game_9x9)

        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3  # RGB

    def test_render_empty_board_13x13(
        self,
        renderer: BoardRenderer,
        game_13x13: SimpleGoGame,
    ) -> None:
        """Test rendering empty 13x13 board produces valid image."""
        image = renderer.render(game_13x13)

        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3

    def test_render_empty_board_19x19(
        self,
        renderer: BoardRenderer,
        game_19x19: SimpleGoGame,
    ) -> None:
        """Test rendering empty 19x19 board produces valid image."""
        image = renderer.render(game_19x19)

        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3

    def test_render_with_stones(
        self,
        renderer: BoardRenderer,
    ) -> None:
        """Test rendering board with stones."""
        game = SimpleGoGame(9)
        game.play(4, 4)  # Black at center
        game.play(3, 3)  # White at 3-3

        image = renderer.render(game)

        assert isinstance(image, np.ndarray)
        assert image.shape[0] > 0
        assert image.shape[1] > 0

    def test_render_with_last_move(
        self,
        renderer: BoardRenderer,
    ) -> None:
        """Test rendering with last move highlighted."""
        game = SimpleGoGame(9)
        game.play(4, 4)

        # Last move at center (row 4, col 4) = index 40 on 9x9
        image = renderer.render(game, last_move=40)

        assert isinstance(image, np.ndarray)

    def test_render_different_sizes(
        self,
        renderer: BoardRenderer,
        board_size: int,
    ) -> None:
        """Test rendering works for all standard sizes."""
        game = SimpleGoGame(board_size)
        image = renderer.render(game)

        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3

    def test_render_full_board(
        self,
        renderer: BoardRenderer,
    ) -> None:
        """Test rendering a board with many stones."""
        game = SimpleGoGame(9)

        # Play several moves
        moves = [(4, 4), (3, 3), (5, 5), (2, 2), (6, 6)]
        for i, (r, c) in enumerate(moves):
            game.play(r, c)

        last_move = moves[-1][0] * 9 + moves[-1][1]
        image = renderer.render(game, last_move=last_move)

        assert isinstance(image, np.ndarray)


class TestCoordinateLabelRendering:
    """Tests for coordinate label rendering."""

    def test_labels_enabled_default(self) -> None:
        """Test labels are enabled by default."""
        config = BoardRenderConfig()
        renderer = BoardRenderer(config)
        game = SimpleGoGame(9)

        image = renderer.render(game)

        # Should still produce valid image
        assert isinstance(image, np.ndarray)

    def test_labels_disabled(self) -> None:
        """Test rendering with labels disabled."""
        config = BoardRenderConfig(
            coordinate_labels=CoordinateLabelConfig(show_labels=False)
        )
        renderer = BoardRenderer(config)
        game = SimpleGoGame(9)

        image = renderer.render(game)

        # Should still produce valid image
        assert isinstance(image, np.ndarray)

    def test_numeric_labels(self) -> None:
        """Test rendering with numeric column labels."""
        config = BoardRenderConfig(
            coordinate_labels=CoordinateLabelConfig(label_style="numbers")
        )
        renderer = BoardRenderer(config)
        game = SimpleGoGame(9)

        image = renderer.render(game)

        assert isinstance(image, np.ndarray)

    def test_custom_label_font_size(self) -> None:
        """Test rendering with custom label font size."""
        config = BoardRenderConfig(
            coordinate_labels=CoordinateLabelConfig(font_size=14)
        )
        renderer = BoardRenderer(config)
        game = SimpleGoGame(9)

        image = renderer.render(game)

        assert isinstance(image, np.ndarray)


class TestRenderBoardFunction:
    """Tests for render_board convenience function."""

    def test_render_board_default(self) -> None:
        """Test render_board with defaults."""
        game = SimpleGoGame(9)
        image = render_board(game)

        assert isinstance(image, np.ndarray)
        assert image.ndim == 3

    def test_render_board_with_last_move(self) -> None:
        """Test render_board with last move."""
        game = SimpleGoGame(9)
        game.play(4, 4)

        image = render_board(game, last_move=40)

        assert isinstance(image, np.ndarray)

    def test_render_board_with_config(self) -> None:
        """Test render_board with custom config."""
        config = BoardRenderConfig(board_color="#cccccc")
        game = SimpleGoGame(9)

        image = render_board(game, config=config)

        assert isinstance(image, np.ndarray)


class TestGTPLettersInRenderer:
    """Tests for GTP letter usage in renderer."""

    def test_gtp_letters_available(self) -> None:
        """Test GTP_LETTERS constant is available."""
        assert GTP_LETTERS is not None
        assert len(GTP_LETTERS) >= 19

    def test_gtp_letters_skip_i(self) -> None:
        """Test GTP letters skip 'I'."""
        assert "I" not in GTP_LETTERS


class TestBoardRendererEdgeCases:
    """Tests for edge cases in board rendering."""

    def test_render_after_pass(
        self,
        renderer: BoardRenderer,
    ) -> None:
        """Test rendering after pass moves."""
        game = SimpleGoGame(9)
        game.play(4, 4)
        game.play_pass()

        image = renderer.render(game)

        assert isinstance(image, np.ndarray)

    def test_render_near_terminal(
        self,
        renderer: BoardRenderer,
    ) -> None:
        """Test rendering near game end."""
        game = SimpleGoGame(9)
        game.play_pass()
        game.play_pass()  # Game should be terminal

        image = renderer.render(game)

        assert isinstance(image, np.ndarray)

    def test_render_capture_scenario(
        self,
        renderer: BoardRenderer,
    ) -> None:
        """Test rendering after capture."""
        game = SimpleGoGame(9)
        # Set up a capture scenario
        game.play(0, 1)  # Black
        game.play(0, 0)  # White (will be captured)
        game.play(1, 0)  # Black captures white

        image = renderer.render(game, last_move=9)  # Last move at (1,0)

        assert isinstance(image, np.ndarray)
