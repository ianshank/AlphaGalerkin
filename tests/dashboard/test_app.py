"""Tests for dashboard/app.py — main application entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from dashboard.app import _build_css, _parse_args, build_app
from dashboard.config import DEFAULT_CONFIG, AppConfig, DashboardConfig

# ---------------------------------------------------------------------------
# _build_css
# ---------------------------------------------------------------------------


class TestBuildCss:
    def test_returns_string(self):
        css = _build_css(AppConfig())
        assert isinstance(css, str)

    def test_contains_tab_nav_rule(self):
        css = _build_css(AppConfig())
        assert ".tab-nav" in css

    def test_contains_footer_rule(self):
        css = _build_css(AppConfig())
        assert "footer" in css

    def test_contains_font_size_from_config(self):
        cfg = AppConfig(css_tab_font_size="18px")
        css = _build_css(cfg)
        assert "18px" in css

    def test_contains_padding_from_config(self):
        cfg = AppConfig(css_tab_padding="8px 16px")
        css = _build_css(cfg)
        assert "8px 16px" in css

    def test_custom_config_reflected(self):
        cfg = AppConfig(css_tab_font_size="20px", css_tab_padding="12px 24px")
        css = _build_css(cfg)
        assert "20px" in css
        assert "12px 24px" in css


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.host == DEFAULT_CONFIG.app.host
        assert args.port == DEFAULT_CONFIG.app.port
        assert args.share is False
        assert args.debug is False

    def test_host_override(self):
        args = _parse_args(["--host", "127.0.0.1"])
        assert args.host == "127.0.0.1"

    def test_port_override(self):
        args = _parse_args(["--port", "9090"])
        assert args.port == 9090

    def test_share_flag(self):
        args = _parse_args(["--share"])
        assert args.share is True

    def test_debug_flag(self):
        args = _parse_args(["--debug"])
        assert args.debug is True

    def test_all_overrides(self):
        args = _parse_args(["--host", "0.0.0.0", "--port", "8080", "--share", "--debug"])
        assert args.host == "0.0.0.0"
        assert args.port == 8080
        assert args.share is True
        assert args.debug is True

    def test_invalid_port_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["--port", "not-a-number"])


# ---------------------------------------------------------------------------
# build_app
# ---------------------------------------------------------------------------


class TestBuildApp:
    def test_returns_gradio_blocks(self):
        app = build_app()
        assert isinstance(app, gr.Blocks)

    def test_accepts_none_cfg(self):
        app = build_app(None)
        assert isinstance(app, gr.Blocks)

    def test_accepts_custom_cfg(self, dashboard_cfg):
        app = build_app(dashboard_cfg)
        assert isinstance(app, gr.Blocks)

    def test_custom_app_config(self):
        cfg = DashboardConfig(app=AppConfig(css_tab_font_size="20px"))
        app = build_app(cfg)
        assert isinstance(app, gr.Blocks)

    def test_hf_demos_unavailable_path(self):
        """Ensure the fallback 'Demos (unavailable)' branch runs without error."""
        with patch("dashboard.app._HF_DEMOS_AVAILABLE", False):
            app = build_app()
        assert isinstance(app, gr.Blocks)

    def test_hf_demos_available_path(self):
        """Ensure the HF demo branch runs when mocked demos are available.

        Uses create=True on each patch so the test works regardless of whether
        the optional hf_space demo modules are importable in the test environment.
        """
        mock_tab_fn = MagicMock()
        with (
            patch("dashboard.app._HF_DEMOS_AVAILABLE", True),
            patch("dashboard.app.create_physics_demo_tab", mock_tab_fn, create=True),
            patch("dashboard.app.create_benchmark_demo_tab", mock_tab_fn, create=True),
            patch("dashboard.app.create_architecture_demo_tab", mock_tab_fn, create=True),
        ):
            app = build_app()
        assert isinstance(app, gr.Blocks)
        # Each demo tab creator should be called exactly once
        assert mock_tab_fn.call_count == 3

    def test_title_set(self):
        app = build_app()
        assert app.title == "AlphaGalerkin Dashboard"

    def test_no_css_deprecation_warning(self):
        """build_app() must not trigger the Gradio 6 'css moved to launch()' warning.

        Records all warnings and checks that none of them are the specific
        Gradio 6 CSS-location warning.  Other Gradio UserWarnings are ignored
        so the test remains forward-compatible across Gradio patch releases.
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_app()

        # The Gradio 6 CSS deprecation warning contains both "css" and "launch"
        css_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "css" in str(w.message).lower()
            and "launch" in str(w.message).lower()
        ]
        assert not css_warnings, (
            "build_app() must not emit the Gradio 6 CSS-moved-to-launch warning; "
            f"got: {[str(w.message) for w in css_warnings]}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_calls_launch(self):
        from dashboard.app import main

        mock_app = MagicMock(spec=gr.Blocks)
        with (
            patch("dashboard.app.build_app", return_value=mock_app),
        ):
            main(["--host", "127.0.0.1", "--port", "7861"])

        mock_app.launch.assert_called_once()
        _, kwargs = mock_app.launch.call_args
        assert kwargs["server_name"] == "127.0.0.1"
        assert kwargs["server_port"] == 7861
        assert kwargs["share"] is False
        assert kwargs["debug"] is False
        # Gradio 6: css is passed to launch(), not gr.Blocks()
        assert isinstance(kwargs.get("css"), str)

    def test_main_share_flag(self):
        from dashboard.app import main

        mock_app = MagicMock(spec=gr.Blocks)
        with patch("dashboard.app.build_app", return_value=mock_app):
            main(["--share"])

        _, kwargs = mock_app.launch.call_args
        assert kwargs["share"] is True

    def test_main_debug_flag(self):
        from dashboard.app import main

        mock_app = MagicMock(spec=gr.Blocks)
        with patch("dashboard.app.build_app", return_value=mock_app):
            main(["--debug"])

        _, kwargs = mock_app.launch.call_args
        assert kwargs["debug"] is True

    def test_main_default_args(self):
        from dashboard.app import main

        mock_app = MagicMock(spec=gr.Blocks)
        with patch("dashboard.app.build_app", return_value=mock_app):
            main([])

        mock_app.launch.assert_called_once()
        _, kwargs = mock_app.launch.call_args
        assert kwargs["server_name"] == DEFAULT_CONFIG.app.host
        assert kwargs["server_port"] == DEFAULT_CONFIG.app.port
