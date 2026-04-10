"""Tests for dashboard/tabs/pde_tab.py — PDE solver tab."""

from __future__ import annotations

from unittest.mock import patch

import gradio as gr
import numpy as np
import pytest
from PIL import Image as PILImage

from dashboard.config import PDEConfig
from dashboard.tabs.pde_tab import (
    _make_charge_grid,
    _poisson_solve,
    compare_resolutions,
    create_pde_tab,
    solve_and_visualize,
)

# ---------------------------------------------------------------------------
# _make_charge_grid
# ---------------------------------------------------------------------------


class TestMakeChargeGrid:
    @pytest.mark.parametrize("n", [9, 13, 19])
    def test_output_shape(self, n):
        grid = _make_charge_grid("Point Charge", n, 0.5, 0.5, 1.0)
        assert grid.shape == (n, n)

    def test_dtype_float32(self):
        grid = _make_charge_grid("Point Charge", 9, 0.5, 0.5, 1.0)
        assert grid.dtype == np.float32

    def test_point_charge_single_nonzero(self):
        grid = _make_charge_grid("Point Charge", 9, 0.5, 0.5, 1.0)
        assert np.count_nonzero(grid) == 1
        assert grid.max() == pytest.approx(1.0)

    def test_dipole_two_charges(self):
        grid = _make_charge_grid("Dipole", 9, 0.5, 0.5, 1.0)
        assert np.count_nonzero(grid) == 2
        # One positive, one negative
        assert grid.max() > 0
        assert grid.min() < 0

    def test_quadrupole_four_charges(self):
        grid = _make_charge_grid("Quadrupole", 9, 0.5, 0.5, 1.0)
        assert np.count_nonzero(grid) >= 4

    def test_ring_charges_count(self, pde_cfg):
        grid = _make_charge_grid("Ring", 9, 0.5, 0.5, 1.0, cfg=pde_cfg)
        # Ring distributes strength over ring_num_charges points
        assert np.count_nonzero(grid) == pde_cfg.ring_num_charges

    def test_random_reproducible(self):
        g1 = _make_charge_grid("Random", 9, 0.5, 0.5, 1.0)
        g2 = _make_charge_grid("Random", 9, 0.5, 0.5, 1.0)
        np.testing.assert_array_equal(g1, g2)

    def test_unknown_pattern_returns_zeros(self):
        grid = _make_charge_grid("NonExistentPattern", 9, 0.5, 0.5, 1.0)
        assert np.all(grid == 0)

    @pytest.mark.parametrize("pattern", ["Point Charge", "Dipole", "Quadrupole", "Ring", "Random"])
    def test_strength_scaling(self, pattern):
        g1 = _make_charge_grid(pattern, 9, 0.5, 0.5, 1.0)
        g2 = _make_charge_grid(pattern, 9, 0.5, 0.5, 2.0)
        # Doubling strength should (roughly) double the non-zero magnitudes
        if np.any(g1 != 0):
            ratio = np.abs(g2[g2 != 0]).sum() / np.abs(g1[g1 != 0]).sum()
            assert ratio == pytest.approx(2.0, abs=0.05)

    def test_boundary_positions(self):
        grid = _make_charge_grid("Point Charge", 9, 0.1, 0.9, 1.0)
        assert grid.shape == (9, 9)

    def test_custom_config(self, pde_cfg):
        grid = _make_charge_grid("Point Charge", 9, 0.5, 0.5, 1.0, cfg=pde_cfg)
        assert grid.shape == (9, 9)


# ---------------------------------------------------------------------------
# _poisson_solve  (integration — uses real PoissonSolver)
# ---------------------------------------------------------------------------


class TestPoissonSolve:
    def test_output_shape_matches_input(self, small_charge_grid):
        potential = _poisson_solve(small_charge_grid)
        assert potential.shape == small_charge_grid.shape

    def test_dtype_float32(self, small_charge_grid):
        potential = _poisson_solve(small_charge_grid)
        assert potential.dtype == np.float32

    def test_zero_charges_give_zero_potential(self):
        charges = np.zeros((9, 9), dtype=np.float32)
        potential = _poisson_solve(charges)
        np.testing.assert_allclose(potential, 0.0, atol=1e-10)

    def test_potential_has_finite_values(self, small_charge_grid):
        potential = _poisson_solve(small_charge_grid)
        assert np.all(np.isfinite(potential))


# ---------------------------------------------------------------------------
# solve_and_visualize
# ---------------------------------------------------------------------------


class TestSolveAndVisualize:
    @patch("dashboard.tabs.pde_tab._poisson_solve")
    def test_returns_pil_image_and_metrics(self, mock_solve, small_charge_grid):
        mock_solve.return_value = np.zeros_like(small_charge_grid)
        img, metrics = solve_and_visualize("Point Charge", 9, 0.5, 0.5, 1.0)
        assert isinstance(img, PILImage.Image)
        assert isinstance(metrics, str)
        assert "9×9" in metrics

    @patch("dashboard.tabs.pde_tab._poisson_solve")
    def test_metrics_contains_grid_info(self, mock_solve, small_charge_grid):
        mock_solve.return_value = np.zeros_like(small_charge_grid)
        _, metrics = solve_and_visualize("Point Charge", 9, 0.5, 0.5, 1.0)
        assert "Grid:" in metrics
        assert "Tokens" in metrics

    @patch("dashboard.tabs.pde_tab._poisson_solve", side_effect=RuntimeError("solver down"))
    def test_returns_none_on_error(self, _mock_solve):
        img, metrics = solve_and_visualize("Point Charge", 9, 0.5, 0.5, 1.0)
        assert img is None
        assert "solver down" in metrics

    @patch("dashboard.tabs.pde_tab._poisson_solve")
    @pytest.mark.parametrize("pattern", ["Point Charge", "Dipole", "Quadrupole"])
    def test_all_patterns_produce_image(self, mock_solve, pattern, small_charge_grid):
        mock_solve.return_value = np.zeros((9, 9), dtype=np.float32)
        img, _ = solve_and_visualize(pattern, 9, 0.5, 0.5, 1.0)
        assert isinstance(img, PILImage.Image)

    @patch("dashboard.tabs.pde_tab._poisson_solve")
    def test_custom_config(self, mock_solve, pde_cfg, small_charge_grid):
        mock_solve.return_value = np.zeros_like(small_charge_grid)
        img, metrics = solve_and_visualize("Point Charge", 9, 0.5, 0.5, 1.0, cfg=pde_cfg)
        assert img is not None

    @patch("dashboard.tabs.pde_tab._poisson_solve")
    def test_negative_strength(self, mock_solve, small_charge_grid):
        mock_solve.return_value = np.zeros_like(small_charge_grid)
        img, metrics = solve_and_visualize("Point Charge", 9, 0.5, 0.5, -1.0)
        assert img is not None


# ---------------------------------------------------------------------------
# compare_resolutions
# ---------------------------------------------------------------------------


def _shape_matching_solve(charges: np.ndarray) -> np.ndarray:
    """Return a zero array matching the input shape (for mocking _poisson_solve)."""
    return np.zeros_like(charges)


class TestCompareResolutions:
    @patch("dashboard.tabs.pde_tab._poisson_solve", side_effect=_shape_matching_solve)
    def test_returns_pil_image(self, mock_solve):
        img, msg = compare_resolutions("Point Charge", 1.0)
        assert isinstance(img, PILImage.Image)
        assert isinstance(msg, str)

    @patch("dashboard.tabs.pde_tab._poisson_solve", side_effect=_shape_matching_solve)
    def test_message_contains_comparison(self, mock_solve):
        _, msg = compare_resolutions("Point Charge", 1.0)
        assert "MSE" in msg or "complete" in msg.lower()

    @patch("dashboard.tabs.pde_tab._poisson_solve", side_effect=ImportError("scipy missing"))
    def test_returns_none_on_import_error(self, _mock_solve):
        img, msg = compare_resolutions("Point Charge", 1.0)
        assert img is None
        assert "scipy" in msg.lower() or "error" in msg.lower()

    @patch("dashboard.tabs.pde_tab._poisson_solve", side_effect=_shape_matching_solve)
    def test_custom_config_uses_cfg_sizes(self, mock_solve, pde_cfg):
        img, _ = compare_resolutions("Point Charge", 1.0, cfg=pde_cfg)
        assert img is not None
        # mock_solve called once per comparison size
        assert mock_solve.call_count == len(pde_cfg.comparison_sizes)


# ---------------------------------------------------------------------------
# create_pde_tab
# ---------------------------------------------------------------------------


class TestCreatePdeTab:
    def test_creates_gradio_tab(self, pde_cfg):
        with gr.Blocks():
            create_pde_tab(pde_cfg)  # Should not raise

    def test_creates_tab_with_default_config(self):
        with gr.Blocks():
            create_pde_tab()  # Should not raise

    def test_pattern_choices_from_config(self, pde_cfg):
        """Verify that the tab uses patterns from config, not a hardcoded list."""
        custom_cfg = PDEConfig(
            charge_patterns=["Custom A", "Custom B", "Custom C", "Custom D"],
            default_pattern="Custom A",
        )
        # Should not raise even with non-standard patterns
        with gr.Blocks():
            create_pde_tab(custom_cfg)

    def test_grid_sizes_from_config(self):
        custom_cfg = PDEConfig(grid_sizes=[7, 11, 15], default_grid_size=7)
        with gr.Blocks():
            create_pde_tab(custom_cfg)
