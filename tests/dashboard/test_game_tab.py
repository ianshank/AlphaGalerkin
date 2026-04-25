"""Tests for dashboard/tabs/game_tab.py — Go Game tab."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import gradio as gr
import numpy as np
import pytest

from dashboard.config import GameConfig
from dashboard.tabs.game_tab import (
    _board_size_choices,
    _ensure_loaded,
    _fallback_board,
    ai_reset,
    ai_step,
    create_game_tab,
    human_move,
    human_reset,
)

# ---------------------------------------------------------------------------
# Helpers to reset module-level state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_game_module_state():
    """Reset the lazy-init globals between tests to prevent cross-test pollution."""
    import dashboard.tabs.game_tab as game_tab

    original_loaded = game_tab._loaded
    original_model = game_tab._model
    original_evaluator = game_tab._evaluator
    original_gm = game_tab._game_manager
    original_renderer = game_tab._renderer
    original_endgame = game_tab._endgame
    original_sc = game_tab._space_config
    original_err = game_tab._init_error

    yield

    game_tab._loaded = original_loaded
    game_tab._model = original_model
    game_tab._evaluator = original_evaluator
    game_tab._game_manager = original_gm
    game_tab._renderer = original_renderer
    game_tab._endgame = original_endgame
    game_tab._space_config = original_sc
    game_tab._init_error = original_err


# ---------------------------------------------------------------------------
# _fallback_board
# ---------------------------------------------------------------------------


class TestFallbackBoard:
    def test_returns_numpy_array(self, game_cfg):
        board = _fallback_board(9, cfg=game_cfg)
        assert isinstance(board, np.ndarray)

    def test_shape_is_square_rgb(self, game_cfg):
        board = _fallback_board(9, cfg=game_cfg)
        px = game_cfg.fallback_board_size_px
        assert board.shape == (px, px, 3)

    def test_dtype_uint8(self, game_cfg):
        board = _fallback_board(9, cfg=game_cfg)
        assert board.dtype == np.uint8

    def test_uses_default_config_when_none(self):
        board = _fallback_board(9)
        assert isinstance(board, np.ndarray)
        assert board.ndim == 3

    def test_custom_fallback_size(self):
        cfg = GameConfig(fallback_board_size_px=200)
        board = _fallback_board(9, cfg=cfg)
        assert board.shape == (200, 200, 3)


# ---------------------------------------------------------------------------
# _board_size_choices (when game manager is not loaded)
# ---------------------------------------------------------------------------


class TestBoardSizeChoices:
    def test_returns_list_of_tuples(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True  # prevent actual loading
        game_tab._game_manager = None

        choices = _board_size_choices(game_cfg)
        assert isinstance(choices, list)
        for item in choices:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_values_match_config_board_sizes(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        choices = _board_size_choices(game_cfg)
        sizes = [v for _, v in choices]
        assert sizes == game_cfg.board_sizes


# ---------------------------------------------------------------------------
# _ensure_loaded (with mocked infrastructure)
# ---------------------------------------------------------------------------


class TestEnsureLoaded:
    def test_returns_bool(self):
        """_ensure_loaded() always returns a bool and never raises.

        The function wraps all loading logic in try/except so it must return
        a bool regardless of missing imports or absent checkpoint files.
        Resetting _loaded and _model forces the full loading path to run;
        the result is False because either imports are unavailable or there is
        no checkpoint.pt in the test environment.
        """
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = False
        game_tab._model = None

        # _ensure_loaded() catches all exceptions internally and returns bool.
        result = _ensure_loaded()

        assert isinstance(result, bool)
        # In the test env there is no checkpoint → _model stays None → False.
        assert result is False

    def test_idempotent_after_first_call(self):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._model = None
        result = _ensure_loaded()
        assert result is False  # model is None, so returns False

    def test_idempotent_when_loaded_with_model(self):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._model = MagicMock()  # non-None
        result = _ensure_loaded()
        assert result is True


# ---------------------------------------------------------------------------
# human_reset / human_move (without real model)
# ---------------------------------------------------------------------------


class TestHumanReset:
    def test_returns_tuple_of_4(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        result = human_reset(9, cfg=game_cfg)
        assert len(result) == 4

    def test_history_is_empty_list(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        history, status, board, score = human_reset(9, cfg=game_cfg)
        assert history == []

    def test_status_is_string(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        _, status, _, _ = human_reset(9, cfg=game_cfg)
        assert isinstance(status, str)

    def test_board_is_numpy_array(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        _, _, board, _ = human_reset(9, cfg=game_cfg)
        assert isinstance(board, np.ndarray)

    def test_uses_default_config(self):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        result = human_reset(9)
        assert len(result) == 4

    def test_with_mock_game_manager(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        mock_session = MagicMock()
        mock_session.is_zero_shot = True
        mock_session.komi = 6.5
        mock_session.game = MagicMock()

        mock_gm = MagicMock()
        mock_gm.create_game.return_value = mock_session
        mock_gm.get_score_display.return_value = "B:0 W:0"

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

        game_tab._loaded = True
        game_tab._game_manager = mock_gm
        game_tab._renderer = mock_renderer

        history, status, board, score = human_reset(9, cfg=game_cfg)
        assert history == []
        assert "6.5" in status
        assert "Zero-shot" in status


class TestHumanMove:
    def test_returns_4_tuple_when_no_model(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        result = human_move([], 9, "4,4", cfg=game_cfg)
        assert len(result) == 4

    def test_invalid_move_text_returns_error(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        # Set up minimal mock game manager
        mock_gm = MagicMock()
        mock_gm.parse_move.side_effect = ValueError("invalid move")
        mock_gm.replay_history.return_value = MagicMock(is_terminal=lambda: False)
        mock_gm.get_score_display.return_value = ""

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

        mock_sc = MagicMock()
        mock_sc.get_komi.return_value = 5.5
        mock_sc.training_board_size = 9

        game_tab._loaded = True
        game_tab._game_manager = mock_gm
        game_tab._renderer = mock_renderer
        game_tab._space_config = mock_sc

        # _build_session imports src.game_manager which may not be available
        # in the test environment; patch it out to focus on the move parsing path
        with patch("dashboard.tabs.game_tab._build_session", return_value=MagicMock()):
            history, status, board, score = human_move([], 9, "XYZ", cfg=game_cfg)
        assert "Invalid" in status or "invalid" in status.lower() or "error" in status.lower()


# ---------------------------------------------------------------------------
# ai_reset / ai_step (without real model)
# ---------------------------------------------------------------------------


class TestAiReset:
    def test_returns_4_tuple_when_no_model(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        result = ai_reset(9, cfg=game_cfg)
        assert len(result) == 4

    def test_history_empty(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        history, *_ = ai_reset(9, cfg=game_cfg)
        assert history == []

    def test_with_mock_game_manager(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        mock_session = MagicMock()
        mock_session.is_zero_shot = False
        mock_session.game = MagicMock()

        mock_gm = MagicMock()
        mock_gm.create_game.return_value = mock_session
        mock_gm.get_score_display.return_value = "B:0 W:0"

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

        game_tab._loaded = True
        game_tab._game_manager = mock_gm
        game_tab._renderer = mock_renderer

        history, status, board, score = ai_reset(9, cfg=game_cfg)
        assert history == []
        assert "Training" in status or "Ready" in status


class TestAiStep:
    def test_returns_4_tuple_when_no_model(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        result = ai_step([], 9, cfg=game_cfg)
        assert len(result) == 4

    def test_game_over_when_terminal(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        mock_game = MagicMock()
        mock_game.is_terminal.return_value = True

        mock_gm = MagicMock()
        mock_gm.replay_history.return_value = mock_game
        mock_gm.calculate_final_score.return_value = "B+5.5"
        mock_gm.get_score_display.return_value = "Final"

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

        mock_sc = MagicMock()
        mock_sc.get_komi.return_value = 5.5
        mock_sc.training_board_size = 9

        game_tab._loaded = True
        game_tab._game_manager = mock_gm
        game_tab._renderer = mock_renderer
        game_tab._space_config = mock_sc

        # Patch _build_session so tests don't depend on src.game_manager availability
        with patch("dashboard.tabs.game_tab._build_session", return_value=MagicMock()):
            _, status, _, _ = ai_step([], 9, cfg=game_cfg)
        assert "over" in status.lower() or "B+" in status or "Final" in status

    def test_no_evaluator_returns_error(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        mock_game = MagicMock()
        mock_game.is_terminal.return_value = False

        mock_gm = MagicMock()
        mock_gm.replay_history.return_value = mock_game
        mock_gm.get_score_display.return_value = ""

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

        mock_sc = MagicMock()
        mock_sc.get_komi.return_value = 5.5
        mock_sc.training_board_size = 9

        game_tab._loaded = True
        game_tab._game_manager = mock_gm
        game_tab._renderer = mock_renderer
        game_tab._space_config = mock_sc
        game_tab._evaluator = None

        # Patch _build_session so tests don't depend on src.game_manager availability
        with patch("dashboard.tabs.game_tab._build_session", return_value=MagicMock()):
            _, status, _, _ = ai_step([], 9, cfg=game_cfg)
        assert "model" in status.lower() or "loaded" in status.lower()


# ---------------------------------------------------------------------------
# create_game_tab
# ---------------------------------------------------------------------------


class TestCreateGameTab:
    def test_creates_gradio_tab_with_default(self):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        with gr.Blocks():
            create_game_tab()

    def test_creates_gradio_tab_with_custom_config(self, game_cfg):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        with gr.Blocks():
            create_game_tab(game_cfg)

    def test_custom_board_sizes_in_tab(self):
        import dashboard.tabs.game_tab as game_tab

        game_tab._loaded = True
        game_tab._game_manager = None

        cfg = GameConfig(board_sizes=[5, 9], default_board_size=5)
        with gr.Blocks():
            create_game_tab(cfg)
