"""Tests for dashboard/tabs/reentry_tab.py — Reentry TPS analysis."""

from __future__ import annotations

import gradio as gr
import numpy as np
import pytest
from PIL import Image as PILImage

from dashboard.tabs.reentry_tab import (
    _apply_surface_boundary,
    _build_initial_temperature,
    _heat_diffusion_step,
    _simulate_reentry,
    compare_resolutions_reentry,
    create_reentry_tab,
    solve_and_visualize_reentry,
)

# ---------------------------------------------------------------------------
# _build_initial_temperature
# ---------------------------------------------------------------------------


class TestBuildInitialTemperature:
    def test_shape_matches_grid_size(self):
        temp = _build_initial_temperature(9, 300.0)
        assert temp.shape == (9, 9)

    def test_dtype_is_float32(self):
        temp = _build_initial_temperature(9, 300.0)
        assert temp.dtype == np.float32

    def test_uniform_at_interior_temp(self):
        temp = _build_initial_temperature(9, 300.0)
        np.testing.assert_array_equal(temp, 300.0)

    @pytest.mark.parametrize("n", [9, 13, 19])
    def test_different_sizes(self, n):
        temp = _build_initial_temperature(n, 300.0)
        assert temp.shape == (n, n)
        np.testing.assert_array_equal(temp, 300.0)


# ---------------------------------------------------------------------------
# _apply_surface_boundary
# ---------------------------------------------------------------------------


class TestApplySurfaceBoundary:
    def test_top_row_modified(self):
        temp = np.full((9, 9), 300.0, dtype=np.float32)
        surface_temp = 2500.0
        velocity = 7.5
        result = _apply_surface_boundary(temp, surface_temp, velocity)
        expected_top = surface_temp * (velocity / 7.5)
        np.testing.assert_allclose(result[-1, :], expected_top)

    def test_interior_unchanged(self):
        temp = np.full((9, 9), 300.0, dtype=np.float32)
        result = _apply_surface_boundary(temp, 2500.0, 7.5)
        # All rows except the top should remain at 300.0
        np.testing.assert_array_equal(result[:-1, :], 300.0)

    def test_returns_copy(self):
        temp = np.full((9, 9), 300.0, dtype=np.float32)
        result = _apply_surface_boundary(temp, 2500.0, 7.5)
        # Modifying the result should not change the input
        result[0, 0] = 9999.0
        assert temp[0, 0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# _heat_diffusion_step
# ---------------------------------------------------------------------------


class TestHeatDiffusionStep:
    def test_output_shape_matches_input(self):
        temp = np.full((9, 9), 300.0, dtype=np.float32)
        surface_row = np.full(9, 2500.0, dtype=np.float32)
        result = _heat_diffusion_step(
            temp,
            kappa=0.1,
            dt=0.001,
            dx=0.125,
            surface_temp_row=surface_row,
        )
        assert result.shape == temp.shape

    def test_boundary_preserved(self):
        temp = np.full((9, 9), 300.0, dtype=np.float32)
        surface_row = np.full(9, 2500.0, dtype=np.float32)
        result = _heat_diffusion_step(
            temp,
            kappa=0.1,
            dt=0.001,
            dx=0.125,
            surface_temp_row=surface_row,
        )
        # Top row (Dirichlet) should be the surface temperature
        np.testing.assert_allclose(result[-1, :], surface_row)

    def test_interior_evolves(self):
        n = 9
        temp = np.full((n, n), 300.0, dtype=np.float32)
        temp[-1, :] = 2500.0  # hot top row
        surface_row = np.full(n, 2500.0, dtype=np.float32)
        result = _heat_diffusion_step(
            temp,
            kappa=0.1,
            dt=0.001,
            dx=0.125,
            surface_temp_row=surface_row,
        )
        # The row just below the surface should have warmed up
        assert float(result[-2, n // 2]) > 300.0


# ---------------------------------------------------------------------------
# _simulate_reentry
# ---------------------------------------------------------------------------


class TestSimulateReentry:
    def test_returns_correct_number_of_snapshots(self):
        n_snapshots = 3
        snapshots, _times = _simulate_reentry(
            n=9,
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            total_time=1.0,
            n_snapshots=n_snapshots,
            interior_temp=300.0,
        )
        assert len(snapshots) == n_snapshots

    def test_snapshot_shapes_match_grid(self):
        snapshots, _times = _simulate_reentry(
            n=9,
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            total_time=0.1,
            n_snapshots=3,
            interior_temp=300.0,
        )
        for snap in snapshots:
            assert snap.shape == (9, 9)

    def test_times_array_length(self):
        n_snapshots = 3
        snapshots, times = _simulate_reentry(
            n=9,
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            total_time=1.0,
            n_snapshots=n_snapshots,
            interior_temp=300.0,
        )
        assert len(times) == len(snapshots)

    def test_bondline_heats_up(self):
        interior_temp = 300.0
        snapshots, _times = _simulate_reentry(
            n=9,
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            total_time=1.0,
            n_snapshots=3,
            interior_temp=interior_temp,
        )
        # Heat should have penetrated: bottom row of last snapshot > interior_temp
        last_bottom_row = snapshots[-1][0, :]
        assert float(last_bottom_row.max()) > interior_temp


# ---------------------------------------------------------------------------
# solve_and_visualize_reentry
# ---------------------------------------------------------------------------


class TestSolveAndVisualizeReentry:
    def test_returns_image_and_metrics(self):
        img, metrics = solve_and_visualize_reentry(
            grid_size=9,
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            total_time=0.1,
        )
        assert isinstance(img, PILImage.Image)
        assert isinstance(metrics, str)

    def test_metrics_contains_bondline(self):
        _img, metrics = solve_and_visualize_reentry(
            grid_size=9,
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            total_time=0.1,
        )
        assert "bondline" in metrics.lower() or "Bondline" in metrics

    def test_metrics_contains_grid_info(self):
        _img, metrics = solve_and_visualize_reentry(
            grid_size=9,
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            total_time=0.1,
        )
        assert "9" in metrics

    @pytest.mark.parametrize("n", [9, 13])
    def test_different_grid_sizes(self, n):
        img, metrics = solve_and_visualize_reentry(
            grid_size=n,
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            total_time=0.1,
        )
        assert isinstance(img, PILImage.Image)
        assert str(n) in metrics


# ---------------------------------------------------------------------------
# compare_resolutions_reentry
# ---------------------------------------------------------------------------


class TestCompareResolutionsReentry:
    def test_returns_image_and_summary(self):
        img, summary = compare_resolutions_reentry(
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
        )
        assert isinstance(img, PILImage.Image)
        assert isinstance(summary, str)

    def test_summary_contains_mse(self):
        _img, summary = compare_resolutions_reentry(
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
        )
        assert "MSE" in summary or "mse" in summary

    def test_uses_comparison_sizes(self, reentry_cfg):
        img, summary = compare_resolutions_reentry(
            kappa=0.1,
            surface_temp=2500.0,
            velocity=7.5,
            cfg=reentry_cfg,
        )
        assert img is not None
        assert summary


# ---------------------------------------------------------------------------
# create_reentry_tab
# ---------------------------------------------------------------------------


class TestCreateReentryTab:
    def test_creates_without_error(self):
        with gr.Blocks():
            create_reentry_tab()  # Should not raise

    def test_creates_with_custom_config(self, reentry_cfg):
        with gr.Blocks():
            create_reentry_tab(reentry_cfg)  # Should not raise
