"""Backend handler tests for the AlphaGalerkin dashboard.

Tests the actual event handler functions in each tab directly (no Gradio mocking),
verifying correct inputs produce correct outputs — the "BE" layer of our test pyramid.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# PDE Tab handler tests
# ---------------------------------------------------------------------------


class TestPDETabHandlers:
    """Test PDE tab handler functions with real computation."""

    def test_make_charge_grid_point_charge(self, pde_cfg):
        """Point charge places a single nonzero value at the correct position."""
        from dashboard.tabs.pde_tab import _make_charge_grid

        grid = _make_charge_grid("Point Charge", 9, 0.5, 0.5, 1.0, pde_cfg)
        assert grid.shape == (9, 9)
        assert grid.dtype == np.float32
        assert np.count_nonzero(grid) == 1
        assert grid[4, 4] == pytest.approx(1.0)

    def test_make_charge_grid_dipole(self, pde_cfg):
        """Dipole places two opposite charges."""
        from dashboard.tabs.pde_tab import _make_charge_grid

        grid = _make_charge_grid("Dipole", 9, 0.5, 0.5, 2.0, pde_cfg)
        assert grid.shape == (9, 9)
        positive = grid[grid > 0].sum()
        negative = grid[grid < 0].sum()
        assert positive == pytest.approx(-negative, abs=1e-5)

    def test_make_charge_grid_quadrupole(self, pde_cfg):
        """Quadrupole places four charges with zero net charge."""
        from dashboard.tabs.pde_tab import _make_charge_grid

        grid = _make_charge_grid("Quadrupole", 9, 0.5, 0.5, 1.0, pde_cfg)
        assert grid.sum() == pytest.approx(0.0, abs=1e-5)

    def test_make_charge_grid_ring(self, pde_cfg):
        """Ring distributes charge across multiple points."""
        from dashboard.tabs.pde_tab import _make_charge_grid

        grid = _make_charge_grid("Ring", 13, 0.5, 0.5, 1.0, pde_cfg)
        assert grid.shape == (13, 13)
        assert np.count_nonzero(grid) >= 3

    def test_make_charge_grid_random(self, pde_cfg):
        """Random pattern is reproducible (seeded RNG)."""
        from dashboard.tabs.pde_tab import _make_charge_grid

        grid1 = _make_charge_grid("Random", 9, 0.5, 0.5, 1.0, pde_cfg)
        grid2 = _make_charge_grid("Random", 9, 0.5, 0.5, 1.0, pde_cfg)
        np.testing.assert_array_equal(grid1, grid2)

    def test_make_charge_grid_unknown_pattern(self, pde_cfg):
        """Unknown pattern returns all-zeros grid."""
        from dashboard.tabs.pde_tab import _make_charge_grid

        grid = _make_charge_grid("INVALID", 9, 0.5, 0.5, 1.0, pde_cfg)
        assert np.count_nonzero(grid) == 0

    def test_solve_and_visualize_returns_image_and_metrics(self, pde_cfg):
        """solve_and_visualize returns a PIL image and metrics string."""
        from dashboard.tabs.pde_tab import solve_and_visualize

        img, metrics = solve_and_visualize("Point Charge", 9, 0.5, 0.5, 1.0, pde_cfg)
        assert isinstance(img, PILImage.Image)
        assert img.mode == "RGB"
        assert img.size[0] > 0 and img.size[1] > 0
        assert "Grid: 9×9" in metrics
        assert "spectral DST-I" in metrics

    def test_solve_and_visualize_different_sizes(self, pde_cfg):
        """PDE solver works at multiple resolutions."""
        from dashboard.tabs.pde_tab import solve_and_visualize

        for size in [9, 13, 19]:
            img, metrics = solve_and_visualize("Point Charge", size, 0.5, 0.5, 1.0, pde_cfg)
            assert img is not None, f"Failed at size {size}"
            assert f"Grid: {size}×{size}" in metrics

    def test_compare_resolutions_returns_image(self, pde_cfg):
        """compare_resolutions returns comparison plot."""
        from dashboard.tabs.pde_tab import compare_resolutions

        img, summary = compare_resolutions("Point Charge", 1.0, pde_cfg)
        assert isinstance(img, PILImage.Image)
        assert "Resolution" in summary or "×" in summary


# ---------------------------------------------------------------------------
# Training Tab handler tests
# ---------------------------------------------------------------------------


class TestTrainingTabHandlers:
    """Test training tab handler functions."""

    def test_get_model_summary_returns_string(self):
        """get_model_summary returns a non-empty string."""
        from dashboard.tabs.training_tab import get_model_summary

        summary = get_model_summary(d_model=64, n_galerkin=2, n_softmax=1, n_fourier=32)
        assert isinstance(summary, str)
        assert len(summary) > 10

    def test_get_model_summary_includes_params(self):
        """Summary mentions parameter counts or architecture details."""
        from dashboard.tabs.training_tab import get_model_summary

        summary = get_model_summary(d_model=128, n_galerkin=4, n_softmax=2, n_fourier=64)
        # Should contain either parameter count or architectural description
        assert any(
            kw in summary.lower() for kw in ["param", "layer", "galerkin", "model", "attention"]
        )


# ---------------------------------------------------------------------------
# PoC Tab handler tests
# ---------------------------------------------------------------------------


class TestPoCTabHandlers:
    """Test PoC scenario tab helper functions."""

    def test_parse_int_list_valid(self):
        """_parse_int_list parses comma-separated ints."""
        from dashboard.tabs.poc_tab import _parse_int_list

        result = _parse_int_list("81, 169, 361", fallback=[100, 200])
        assert result == [81, 169, 361]

    def test_parse_int_list_fallback_on_invalid(self):
        """_parse_int_list falls back on invalid input."""
        from dashboard.tabs.poc_tab import _parse_int_list

        result = _parse_int_list("not,numbers", fallback=[100, 200])
        assert result == [100, 200]

    def test_parse_int_list_deduplicates_and_sorts(self):
        """_parse_int_list returns sorted unique values."""
        from dashboard.tabs.poc_tab import _parse_int_list

        result = _parse_int_list("361, 81, 169, 81", fallback=[])
        assert result == [81, 169, 361]

    def test_parse_int_list_fallback_on_too_few(self):
        """_parse_int_list falls back when fewer than min_count values."""
        from dashboard.tabs.poc_tab import _parse_int_list

        result = _parse_int_list("42", fallback=[100, 200], min_count=2)
        assert result == [100, 200]


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestDashboardUtils:
    """Test shared dashboard utility functions."""

    def test_fig_to_pil_returns_rgb_image(self):
        """fig_to_pil converts matplotlib figure to PIL RGB image."""
        import matplotlib.pyplot as plt

        from dashboard.utils import fig_to_pil

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot([0, 1], [0, 1])
        img = fig_to_pil(fig, dpi=72)
        assert isinstance(img, PILImage.Image)
        assert img.mode == "RGB"
        assert img.size[0] > 0

    def test_fig_to_pil_closes_figure(self):
        """fig_to_pil closes the matplotlib figure after conversion."""
        import matplotlib.pyplot as plt

        from dashboard.utils import fig_to_pil

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        num_before = len(plt.get_fignums())
        fig_to_pil(fig)
        num_after = len(plt.get_fignums())
        assert num_after < num_before

    def test_device_str_returns_valid_string(self):
        """device_str returns 'cpu' or 'cuda'."""
        from dashboard.utils import device_str

        dev = device_str()
        assert dev in ("cpu", "cuda")

    def test_format_exc_includes_type_and_message(self):
        """format_exc formats exception with type and message."""
        from dashboard.utils import format_exc

        msg = format_exc(ValueError("bad input"), prefix="Test error")
        assert "Test error" in msg
        assert "ValueError" in msg
        assert "bad input" in msg
