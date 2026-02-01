"""Integration tests for HuggingFace Space."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.board import SpaceConfig, get_default_space_config
from src.game_manager import GameManager
from src.rendering.board_renderer import BoardRenderer
from src.tools.gtp import SimpleGoGame


class TestFullGameFlow:
    """Integration tests for complete game flows."""

    @pytest.fixture
    def setup(self) -> tuple[SpaceConfig, GameManager, BoardRenderer]:
        """Set up test fixtures."""
        config = get_default_space_config()
        manager = GameManager(config=config, evaluator=None)
        renderer = BoardRenderer(config.render)
        return config, manager, renderer

    def test_create_and_render_all_sizes(
        self,
        setup: tuple[SpaceConfig, GameManager, BoardRenderer],
    ) -> None:
        """Test creating and rendering games for all sizes."""
        config, manager, renderer = setup

        for size in config.supported_sizes:
            session = manager.create_game(board_size=size)
            image = renderer.render(session.game)

            assert session.board_size == size
            assert isinstance(image, np.ndarray)
            assert image.shape[2] == 3

    def test_play_game_flow(
        self,
        setup: tuple[SpaceConfig, GameManager, BoardRenderer],
    ) -> None:
        """Test complete game flow with moves."""
        config, manager, renderer = setup

        # Create game
        session = manager.create_game(board_size=9)

        # Play some moves
        moves = [(4, 4), (3, 3), (2, 2), (6, 6)]
        for r, c in moves:
            session.game.play(r, c)
            session.move_history.append((r, c))

        # Render with last move
        last_action = moves[-1][0] * 9 + moves[-1][1]
        image = renderer.render(session.game, last_move=last_action)

        assert isinstance(image, np.ndarray)
        assert session.move_count == 4

    def test_komi_values_per_size(
        self,
        setup: tuple[SpaceConfig, GameManager, BoardRenderer],
    ) -> None:
        """Test komi values are correct for each size."""
        config, manager, _ = setup

        expected_komi = {9: 5.5, 13: 6.5, 19: 7.5}

        for size, komi in expected_komi.items():
            session = manager.create_game(board_size=size)
            assert session.komi == komi
            assert session.game.komi == komi

    def test_game_end_detection(
        self,
        setup: tuple[SpaceConfig, GameManager, BoardRenderer],
    ) -> None:
        """Test game end is detected correctly."""
        _, manager, renderer = setup

        session = manager.create_game(board_size=9)

        # Play until game ends
        session.game.play_pass()
        session.move_history.append("PASS")
        session.game.play_pass()
        session.move_history.append("PASS")

        assert session.is_terminal
        assert session.game.is_terminal()

        # Should still render
        image = renderer.render(session.game)
        assert isinstance(image, np.ndarray)


class TestZeroShotTransferInfo:
    """Tests for zero-shot transfer information."""

    def test_labels_indicate_transfer(self) -> None:
        """Test that labels correctly indicate zero-shot transfer."""
        manager = GameManager()

        label_9 = manager.get_board_size_label(9)
        label_13 = manager.get_board_size_label(13)
        label_19 = manager.get_board_size_label(19)

        assert "Training" in label_9
        assert "Zero-shot" in label_13
        assert "Zero-shot" in label_19

    def test_session_tracks_zero_shot(self) -> None:
        """Test session tracks zero-shot status."""
        manager = GameManager()

        session_9 = manager.create_game(board_size=9)
        session_13 = manager.create_game(board_size=13)
        session_19 = manager.create_game(board_size=19)

        assert not session_9.is_zero_shot
        assert session_13.is_zero_shot
        assert session_19.is_zero_shot


class TestCoordinateLabelIntegration:
    """Tests for coordinate label integration."""

    def test_all_sizes_render_with_labels(self) -> None:
        """Test all board sizes render with coordinate labels."""
        config = get_default_space_config()
        renderer = BoardRenderer(config.render)

        for size in [9, 13, 19]:
            game = SimpleGoGame(size)
            image = renderer.render(game)

            # Should produce valid image with labels
            assert isinstance(image, np.ndarray)
            assert image.shape[2] == 3

    def test_labels_disabled_still_renders(self) -> None:
        """Test rendering works with labels disabled."""
        from config.board import BoardRenderConfig, CoordinateLabelConfig

        config = BoardRenderConfig(
            coordinate_labels=CoordinateLabelConfig(show_labels=False)
        )
        renderer = BoardRenderer(config)

        for size in [9, 13, 19]:
            game = SimpleGoGame(size)
            image = renderer.render(game)

            assert isinstance(image, np.ndarray)


class TestHistoryReplay:
    """Tests for game history replay."""

    def test_replay_preserves_board_state(self) -> None:
        """Test replaying history preserves board state."""
        manager = GameManager()

        # Create original game
        original = SimpleGoGame(9)
        history = []

        moves = [(4, 4), (3, 3), (5, 5), (2, 2)]
        for r, c in moves:
            original.play(r, c)
            history.append((r, c))

        # Replay
        replayed = manager.replay_history(history, 9)

        # Compare boards
        assert np.array_equal(original.board, replayed.board)

    def test_replay_with_passes(self) -> None:
        """Test replaying history with passes."""
        manager = GameManager()

        history = [(4, 4), "PASS", (3, 3), "PASS"]
        game = manager.replay_history(history, 9)

        assert game.board[4, 4] == SimpleGoGame.BLACK
        assert game.board[3, 3] == SimpleGoGame.BLACK  # After white pass

    def test_replay_different_sizes(self) -> None:
        """Test replaying on different board sizes."""
        manager = GameManager()

        for size in [9, 13, 19]:
            center = size // 2
            history = [(center, center)]
            game = manager.replay_history(history, size)

            assert game.board_size == size
            assert game.board[center, center] == SimpleGoGame.BLACK


class TestScoreCalculation:
    """Tests for score calculation."""

    def test_score_display_format(self) -> None:
        """Test score display format is consistent."""
        manager = GameManager()

        for size in [9, 13, 19]:
            session = manager.create_game(board_size=size)
            display = manager.get_score_display(session)

            # Should contain all expected elements
            assert "Black captures" in display
            assert "White captures" in display
            assert "Move:" in display
            assert "Komi:" in display

    def test_final_score_format(self) -> None:
        """Test final score format is consistent."""
        manager = GameManager()

        for size in [9, 13, 19]:
            session = manager.create_game(board_size=size)
            session.game.play_pass()
            session.game.play_pass()

            score = manager.calculate_final_score(session)

            # Should indicate winner or draw
            assert "wins" in score or "Draw" in score


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_board_rendering(self) -> None:
        """Test rendering empty board for all sizes."""
        renderer = BoardRenderer()

        for size in [9, 13, 19]:
            game = SimpleGoGame(size)
            image = renderer.render(game)

            assert isinstance(image, np.ndarray)
            assert image.shape[0] > 0
            assert image.shape[1] > 0

    def test_capture_scenario(self) -> None:
        """Test rendering after capture."""
        manager = GameManager()
        renderer = BoardRenderer()

        session = manager.create_game(board_size=9)

        # Set up capture
        session.game.play(0, 1)  # Black
        session.game.play(0, 0)  # White (corner)
        session.game.play(1, 0)  # Black captures

        # White stone should be captured
        assert session.game.board[0, 0] == 0  # Empty

        # Should render correctly
        image = renderer.render(session.game, last_move=9)
        assert isinstance(image, np.ndarray)

    def test_multiple_games_same_manager(self) -> None:
        """Test creating multiple games with same manager."""
        manager = GameManager()

        sessions = []
        for size in [9, 13, 19]:
            session = manager.create_game(board_size=size)
            sessions.append(session)

        # Each session should be independent
        assert len(sessions) == 3
        assert all(s.board_size != sessions[0].board_size for s in sessions[1:])
