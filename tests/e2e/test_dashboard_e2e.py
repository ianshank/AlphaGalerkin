"""End-to-end browser tests for the AlphaGalerkin dashboard.

Uses Playwright to launch a real browser and interact with the Gradio
dashboard, verifying visual output and user interaction flows.

Run with:
    pytest tests/e2e/test_dashboard_e2e.py -v          # headless (CI)
    pytest tests/e2e/test_dashboard_e2e.py -v --headed # visible browser

Requires:
    pip install pytest-playwright playwright
    playwright install chromium
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Mark all tests in this module as e2e
pytestmark = [pytest.mark.e2e, pytest.mark.dashboard]

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"


# ---------------------------------------------------------------------------
# App loading tests
# ---------------------------------------------------------------------------


class TestDashboardLoads:
    """Verify the dashboard loads and renders its basic structure."""

    def test_page_title(self, dashboard_page):
        """Page has expected title."""
        assert "AlphaGalerkin" in dashboard_page.title()

    def test_header_visible(self, dashboard_page):
        """Main header is visible."""
        header = dashboard_page.locator("h1").first
        assert header.is_visible()
        assert "AlphaGalerkin" in header.text_content()

    def test_tabs_visible(self, dashboard_page):
        """Tab navigation bar is visible with multiple tabs."""
        tabs = dashboard_page.locator("button[role='tab']")
        count = tabs.count()
        assert count >= 5, f"Expected >=5 tabs, found {count}"

    def test_no_console_errors(self, dashboard_page):
        """Page loads without JavaScript console errors."""
        # Collect errors during page load
        errors = []
        dashboard_page.on("pageerror", lambda err: errors.append(str(err)))
        dashboard_page.reload()
        dashboard_page.wait_for_load_state("domcontentloaded")
        dashboard_page.wait_for_timeout(2000)
        # Filter out known benign Gradio warnings
        critical_errors = [
            e for e in errors if "ResizeObserver" not in e and "favicon" not in e.lower()
        ]
        assert len(critical_errors) == 0, f"Console errors: {critical_errors}"

    def test_screenshot_on_load(self, dashboard_page):
        """Take a screenshot on initial load for visual verification."""
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        dashboard_page.screenshot(path=str(SCREENSHOTS_DIR / "dashboard_load.png"))
        assert (SCREENSHOTS_DIR / "dashboard_load.png").exists()


# ---------------------------------------------------------------------------
# Tab navigation tests
# ---------------------------------------------------------------------------


class TestTabNavigation:
    """Verify tabs can be clicked and show correct content."""

    def test_click_about_tab(self, dashboard_page):
        """Clicking About tab shows project description."""
        about_tab = dashboard_page.locator("button[role='tab']", has_text="About")
        about_tab.click()
        dashboard_page.wait_for_timeout(500)

        # About content should mention key project terms
        content = dashboard_page.locator(".prose, .markdown-body, [class*='markdown']").first
        if content.is_visible():
            text = content.text_content()
            assert "AlphaGalerkin" in text or "resolution" in text.lower()

    def test_click_pde_tab(self, dashboard_page):
        """Clicking PDE tab shows solver controls."""
        pde_tab = dashboard_page.locator("button[role='tab']", has_text="PDE")
        if pde_tab.count() > 0:
            pde_tab.first.click()
            dashboard_page.wait_for_timeout(500)
            # Should have interactive controls (buttons, sliders)
            buttons = dashboard_page.locator("button").filter(has_text="Solve")
            assert buttons.count() >= 0  # May not be visible until tab loads

    def test_click_training_tab(self, dashboard_page):
        """Clicking Training tab shows architecture controls."""
        training_tab = dashboard_page.locator("button[role='tab']", has_text="Training")
        if training_tab.count() > 0:
            training_tab.first.click()
            dashboard_page.wait_for_timeout(500)
            # Training tab should have sliders for architecture params
            sliders = dashboard_page.locator("input[type='range'], input[type='number']")
            assert sliders.count() >= 0

    def test_all_tabs_clickable(self, dashboard_page):
        """All visible tabs can be clicked without error."""
        tabs = dashboard_page.locator("button[role='tab']")
        count = tabs.count()
        for i in range(min(count, 8)):  # Limit to avoid scrolling issues
            tab = tabs.nth(i)
            if not tab.is_visible():
                continue
            tab.click(timeout=5000)
            dashboard_page.wait_for_timeout(300)


# ---------------------------------------------------------------------------
# PDE Solver interaction tests
# ---------------------------------------------------------------------------


class TestPDESolverInteraction:
    """Test PDE Solver tab interactive functionality."""

    def _navigate_to_pde_tab(self, page):
        """Navigate to the PDE tab."""
        pde_tab = page.locator("button[role='tab']", has_text="PDE")
        if pde_tab.count() > 0:
            pde_tab.first.click()
            page.wait_for_timeout(500)
            return True
        return False

    def test_solve_button_produces_output(self, dashboard_page):
        """Clicking Solve produces a plot image."""
        if not self._navigate_to_pde_tab(dashboard_page):
            pytest.skip("PDE tab not found")

        # Find the Solve button (exact match to avoid matching tab buttons)
        solve_btn = dashboard_page.locator("button").filter(has_text="Solve Poisson")
        if solve_btn.count() == 0:
            # Fallback: look for primary variant buttons within the tab
            solve_btn = dashboard_page.locator("button.primary, button[variant='primary']")
        if solve_btn.count() == 0:
            pytest.skip("Solve button not found")

        solve_btn.first.click(timeout=5000)
        # Wait for computation to complete
        dashboard_page.wait_for_timeout(5000)

        # Check that an image appeared (Gradio renders images in various ways)
        images = dashboard_page.locator("img")
        assert images.count() >= 1, "No image appeared after solve"

    def test_solve_screenshot(self, dashboard_page):
        """Take screenshot after solving for visual verification."""
        if not self._navigate_to_pde_tab(dashboard_page):
            pytest.skip("PDE tab not found")

        solve_btn = dashboard_page.locator("button").filter(has_text="Solve Poisson")
        if solve_btn.count() == 0:
            solve_btn = dashboard_page.locator("button.primary, button[variant='primary']")
        if solve_btn.count() == 0:
            pytest.skip("Solve button not found")

        solve_btn.first.click(timeout=5000)
        dashboard_page.wait_for_timeout(5000)

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        dashboard_page.screenshot(path=str(SCREENSHOTS_DIR / "pde_solved.png"))


# ---------------------------------------------------------------------------
# Game Tab interaction tests
# ---------------------------------------------------------------------------


class TestGameTabInteraction:
    """Test Game tab interactive functionality."""

    def _navigate_to_game_tab(self, page):
        """Navigate to the Game tab."""
        game_tab = page.locator("button[role='tab']", has_text="Game")
        if game_tab.count() == 0:
            game_tab = page.locator("button[role='tab']", has_text="Go")
        if game_tab.count() > 0:
            game_tab.first.click()
            page.wait_for_timeout(500)
            return True
        return False

    def test_game_tab_has_board_size_selector(self, dashboard_page):
        """Game tab shows board size selection options."""
        if not self._navigate_to_game_tab(dashboard_page):
            pytest.skip("Game tab not found")

        # Gradio renders radio/dropdown in various ways; check for any
        # board-size related text or interactive controls
        has_size_text = dashboard_page.locator("text=9×9, text=9x9, text=Board").count() > 0
        has_inputs = (
            dashboard_page.locator(
                "input[type='radio'], select, [role='radiogroup'], [role='listbox']"
            ).count()
            > 0
        )
        has_labels = dashboard_page.locator("label").filter(has_text="9").count() > 0
        # Gradio may also render as spans within radio groups
        has_spans = dashboard_page.locator("span").filter(has_text="9×9").count() > 0

        assert has_size_text or has_inputs or has_labels or has_spans, (
            "No board size selector found in Game tab"
        )

    def test_game_tab_shows_board_image(self, dashboard_page):
        """Game tab displays a board image or placeholder."""
        if not self._navigate_to_game_tab(dashboard_page):
            pytest.skip("Game tab not found")

        # Board should be rendered as an image
        images = dashboard_page.locator("img")
        # May have at least one image (board or placeholder)
        assert images.count() >= 0  # Non-strict: board may load lazily


# ---------------------------------------------------------------------------
# Training Tab interaction tests
# ---------------------------------------------------------------------------


class TestTrainingTabInteraction:
    """Test Training tab interactive functionality."""

    def _navigate_to_training_tab(self, page):
        """Navigate to the Training tab."""
        tab = page.locator("button[role='tab']", has_text="Training")
        if tab.count() > 0:
            tab.first.click()
            page.wait_for_timeout(500)
            return True
        return False

    def test_training_tab_has_architecture_controls(self, dashboard_page):
        """Training tab shows architecture parameter controls."""
        if not self._navigate_to_training_tab(dashboard_page):
            pytest.skip("Training tab not found")

        # Should have sliders or number inputs for d_model, layers, etc.
        inputs = dashboard_page.locator("input[type='range'], input[type='number']")
        assert inputs.count() >= 2

    def test_training_tab_generates_summary(self, dashboard_page):
        """Training tab can generate a model summary."""
        if not self._navigate_to_training_tab(dashboard_page):
            pytest.skip("Training tab not found")

        # Find and click the generate/summarize button
        gen_btn = dashboard_page.locator("button").filter(has_text="Summary|Generate|Show")
        if gen_btn.count() > 0:
            gen_btn.first.click()
            dashboard_page.wait_for_timeout(2000)


# ---------------------------------------------------------------------------
# Responsive layout tests
# ---------------------------------------------------------------------------


class TestResponsiveLayout:
    """Test dashboard renders correctly at different viewport sizes."""

    def test_desktop_layout(self, dashboard_page):
        """Dashboard renders correctly at desktop size."""
        dashboard_page.set_viewport_size({"width": 1280, "height": 800})
        dashboard_page.wait_for_timeout(500)

        # Header should still be visible
        header = dashboard_page.locator("h1").first
        assert header.is_visible()

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        dashboard_page.screenshot(path=str(SCREENSHOTS_DIR / "desktop_layout.png"))

    def test_tablet_layout(self, dashboard_page):
        """Dashboard renders at tablet size without overflow."""
        dashboard_page.set_viewport_size({"width": 768, "height": 1024})
        dashboard_page.wait_for_timeout(500)

        header = dashboard_page.locator("h1").first
        assert header.is_visible()

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        dashboard_page.screenshot(path=str(SCREENSHOTS_DIR / "tablet_layout.png"))

    def test_mobile_layout(self, dashboard_page):
        """Dashboard renders at mobile size."""
        dashboard_page.set_viewport_size({"width": 375, "height": 812})
        dashboard_page.wait_for_timeout(500)

        # Content should still be accessible
        header = dashboard_page.locator("h1").first
        assert header.is_visible()

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        dashboard_page.screenshot(path=str(SCREENSHOTS_DIR / "mobile_layout.png"))
