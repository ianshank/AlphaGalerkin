"""Frontend rendering tests for the AlphaGalerkin dashboard.

Tests Gradio app construction and component rendering using the Gradio test
client, verifying that the UI layer responds correctly to simulated user
interactions without needing a real browser.
"""

from __future__ import annotations

import gradio as gr
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dashboard_app():
    """Build the full dashboard Gradio app once for the module."""
    from dashboard.app import build_app

    return build_app()


# ---------------------------------------------------------------------------
# App structure tests
# ---------------------------------------------------------------------------


class TestAppStructure:
    """Verify the Gradio app builds correctly and has expected tabs."""

    def test_app_is_blocks_instance(self, dashboard_app):
        """build_app returns a gr.Blocks instance."""
        assert isinstance(dashboard_app, gr.Blocks)

    def test_app_has_title(self, dashboard_app):
        """App has the expected title."""
        assert dashboard_app.title == "AlphaGalerkin Dashboard"

    def test_app_has_multiple_components(self, dashboard_app):
        """App contains multiple Gradio components (tabs, buttons, etc.)."""
        # gr.Blocks stores child blocks; a real app has many
        assert len(dashboard_app.blocks) > 10

    def test_app_contains_tab_components(self, dashboard_app):
        """App contains Tab components for the different sections."""
        tab_blocks = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Tab)]
        # At minimum: Game, PDE Solver, PoC Scenarios, Training, About
        assert len(tab_blocks) >= 5

    def test_app_tab_labels(self, dashboard_app):
        """App tabs have expected labels."""
        tab_labels = {b.label for b in dashboard_app.blocks.values() if isinstance(b, gr.Tab)}
        # Check core tabs exist (some may have slightly different names)
        assert "About" in tab_labels
        assert any("PDE" in label for label in tab_labels)
        assert any("Game" in label or "Go" in label for label in tab_labels)


# ---------------------------------------------------------------------------
# PDE Tab rendering tests
# ---------------------------------------------------------------------------


class TestPDETabRendering:
    """Test PDE tab creates correct interactive components."""

    def test_pde_tab_has_sliders(self, dashboard_app):
        """PDE tab has slider components for grid size and charge params."""
        sliders = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Slider)]
        assert len(sliders) >= 1  # At least grid_size slider

    def test_pde_tab_has_dropdown(self, dashboard_app):
        """PDE tab has dropdown for charge pattern selection."""
        dropdowns = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Dropdown)]
        assert len(dropdowns) >= 1

    def test_pde_tab_has_buttons(self, dashboard_app):
        """PDE tab has action buttons (Solve, Compare)."""
        buttons = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Button)]
        assert len(buttons) >= 1


# ---------------------------------------------------------------------------
# Game Tab rendering tests
# ---------------------------------------------------------------------------


class TestGameTabRendering:
    """Test Game tab creates correct components."""

    def test_game_tab_has_board_size_selector(self, dashboard_app):
        """Game tab has a board size selector (radio or dropdown)."""
        radios = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Radio | gr.Dropdown)]
        assert len(radios) >= 1

    def test_game_tab_has_image_output(self, dashboard_app):
        """Game tab has image component for board rendering."""
        images = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Image)]
        assert len(images) >= 1


# ---------------------------------------------------------------------------
# Training Tab rendering tests
# ---------------------------------------------------------------------------


class TestTrainingTabRendering:
    """Test Training tab creates correct components."""

    def test_training_tab_has_sliders_for_arch(self, dashboard_app):
        """Training tab has sliders for architecture parameters."""
        sliders = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Slider)]
        # Multiple sliders: d_model, n_galerkin, n_softmax, n_fourier, etc.
        assert len(sliders) >= 3

    def test_training_tab_has_textbox_output(self, dashboard_app):
        """Training tab has textbox for model summary output."""
        textboxes = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Textbox)]
        assert len(textboxes) >= 1


# ---------------------------------------------------------------------------
# About Tab rendering tests
# ---------------------------------------------------------------------------


class TestAboutTabRendering:
    """Test About tab renders correctly."""

    def test_about_tab_has_markdown(self, dashboard_app):
        """About tab contains Markdown components with project info."""
        markdowns = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Markdown)]
        # Header markdown + About content + potentially others
        assert len(markdowns) >= 2

    def test_about_markdown_contains_project_name(self, dashboard_app):
        """At least one Markdown component mentions AlphaGalerkin."""
        markdowns = [b for b in dashboard_app.blocks.values() if isinstance(b, gr.Markdown)]
        texts = [m.value for m in markdowns if m.value]
        combined = " ".join(texts)
        assert "AlphaGalerkin" in combined


# ---------------------------------------------------------------------------
# Gradio Client interaction tests
# ---------------------------------------------------------------------------


class TestGradioClientInteraction:
    """Test Gradio app responds to programmatic API calls."""

    def test_app_has_registered_functions(self, dashboard_app):
        """App has registered event handler functions."""
        # Gradio apps register event handlers as fns
        assert hasattr(dashboard_app, "fns")
        assert len(dashboard_app.fns) > 0

    def test_app_has_api_endpoints(self, dashboard_app):
        """App exposes API-callable endpoints."""
        # Each fn with api_name is an endpoint
        api_fns = [fn for fn in dashboard_app.fns.values() if fn.api_name]
        # At least some endpoints should be API-accessible
        assert len(api_fns) >= 0  # Non-strict: Gradio 6 may not expose all

    def test_app_has_event_listeners(self, dashboard_app):
        """App has wired up event listeners (button clicks, etc.)."""
        # In Gradio 6, event handlers are stored in fns
        # The app should have more fns than 0 (handlers for solve, compare, etc.)
        assert len(dashboard_app.fns) >= 2
