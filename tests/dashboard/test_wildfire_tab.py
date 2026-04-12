"""Tests for dashboard/tabs/wildfire_tab.py -- Wildfire spread simulation."""

from __future__ import annotations

import gradio as gr
import numpy as np
import pytest
from PIL import Image as PILImage

from dashboard.tabs.wildfire_tab import (
    _advection_diffusion_step,
    _build_fuel_field,
    _build_ignition_field,
    _simulate_wildfire,
    compare_resolutions_wildfire,
    create_wildfire_tab,
    solve_and_visualize_wildfire,
)

# ---------------------------------------------------------------------------
# _build_ignition_field
# ---------------------------------------------------------------------------


class TestBuildIgnitionField:
    def test_center_pattern_shape(self):
        field = _build_ignition_field("Center", 9)
        assert field.shape == (9, 9)

    def test_center_has_hot_spot(self):
        field = _build_ignition_field("Center", 9)
        # The Gaussian centred at (4.5, 4.5) should produce values > 0
        # in the centre region.
        centre_region = field[3:6, 3:6]
        assert centre_region.max() > 0.0

    def test_edge_pattern(self):
        field = _build_ignition_field("Edge", 9)
        # "Edge" sets the entire top row to 1.0
        assert np.all(field[0, :] > 0.0)

    def test_random_is_reproducible(self):
        f1 = _build_ignition_field("Random", 9)
        f2 = _build_ignition_field("Random", 9)
        np.testing.assert_array_equal(f1, f2)

    def test_unknown_pattern_returns_zeros(self):
        field = _build_ignition_field("NonExistentPattern", 9)
        assert np.all(field == 0.0)


# ---------------------------------------------------------------------------
# _build_fuel_field
# ---------------------------------------------------------------------------


class TestBuildFuelField:
    def test_shape_and_dtype(self):
        fuel = _build_fuel_field(9, fuel_density=1.0)
        assert fuel.shape == (9, 9)
        assert fuel.dtype == np.float32

    def test_positive_values(self):
        fuel = _build_fuel_field(9, fuel_density=1.0)
        assert np.all(fuel > 0.0)

    def test_reproducible_with_seed(self):
        f1 = _build_fuel_field(9, fuel_density=1.0, seed=123)
        f2 = _build_fuel_field(9, fuel_density=1.0, seed=123)
        np.testing.assert_array_equal(f1, f2)

    def test_scales_with_density(self):
        low = _build_fuel_field(9, fuel_density=0.5, seed=42)
        high = _build_fuel_field(9, fuel_density=1.5, seed=42)
        assert high.mean() > low.mean()


# ---------------------------------------------------------------------------
# _advection_diffusion_step
# ---------------------------------------------------------------------------


class TestAdvectionDiffusionStep:
    def test_output_shapes(self):
        n = 9
        temp = np.random.default_rng(0).random((n, n)).astype(np.float32)
        fuel = np.ones((n, n), dtype=np.float32)
        new_temp, new_fuel = _advection_diffusion_step(
            temp,
            fuel,
            wind_vx=1.0,
            wind_vy=0.0,
            diffusion=0.1,
            dt=0.001,
            ignition_threshold=0.5,
        )
        assert new_temp.shape == (n, n)
        assert new_fuel.shape == (n, n)

    def test_fuel_decreases_near_ignition(self):
        n = 9
        # Temperature field that is entirely above threshold
        temp = np.full((n, n), 1.0, dtype=np.float32)
        fuel = np.ones((n, n), dtype=np.float32)
        _new_temp, new_fuel = _advection_diffusion_step(
            temp,
            fuel,
            wind_vx=0.0,
            wind_vy=0.0,
            diffusion=0.1,
            dt=0.01,
            ignition_threshold=0.5,
        )
        # Fuel should have been consumed where temp > threshold
        assert new_fuel.sum() < fuel.sum()

    def test_no_nans(self):
        n = 9
        temp = _build_ignition_field("Center", n)
        fuel = _build_fuel_field(n, fuel_density=1.0)
        new_temp, new_fuel = _advection_diffusion_step(
            temp,
            fuel,
            wind_vx=2.0,
            wind_vy=1.0,
            diffusion=0.1,
            dt=0.0001,
            ignition_threshold=0.5,
        )
        assert not np.any(np.isnan(new_temp))
        assert not np.any(np.isnan(new_fuel))


# ---------------------------------------------------------------------------
# _simulate_wildfire
# ---------------------------------------------------------------------------


class TestSimulateWildfire:
    def test_correct_snapshot_count(self):
        n_snapshots = 4
        temp_snaps, _fuel_snaps, _times = _simulate_wildfire(
            n=9,
            wind_speed=2.0,
            wind_direction=45.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern="Center",
            total_time=0.5,
            n_snapshots=n_snapshots,
        )
        assert len(temp_snaps) == n_snapshots

    def test_snapshot_shapes(self):
        temp_snaps, _fuel_snaps, _times = _simulate_wildfire(
            n=9,
            wind_speed=2.0,
            wind_direction=0.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern="Center",
            total_time=0.5,
            n_snapshots=3,
        )
        for snap in temp_snaps:
            assert snap.shape == (9, 9)

    def test_times_array(self):
        n_snapshots = 4
        _temp_snaps, _fuel_snaps, times = _simulate_wildfire(
            n=9,
            wind_speed=2.0,
            wind_direction=0.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern="Center",
            total_time=0.5,
            n_snapshots=n_snapshots,
        )
        assert len(times) == n_snapshots

    def test_fire_spreads(self):
        temp_snaps, fuel_snaps, _times = _simulate_wildfire(
            n=9,
            wind_speed=5.0,
            wind_direction=0.0,
            diffusion=0.2,
            fuel_density=1.0,
            ignition_pattern="Center",
            total_time=0.5,
            n_snapshots=3,
        )
        # Fire should consume fuel: the final fuel total should be less
        # than the initial fuel total because combustion burns fuel.
        initial_fuel_total = float(fuel_snaps[0].sum())
        final_fuel_total = float(fuel_snaps[-1].sum())
        assert final_fuel_total < initial_fuel_total


# ---------------------------------------------------------------------------
# solve_and_visualize_wildfire
# ---------------------------------------------------------------------------


class TestSolveAndVisualizeWildfire:
    def test_returns_image_and_metrics(self):
        img, metrics = solve_and_visualize_wildfire(
            grid_size=9,
            wind_speed=2.0,
            wind_direction=45.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern="Center",
            total_time=0.5,
        )
        assert isinstance(img, PILImage.Image)
        assert isinstance(metrics, str)

    def test_metrics_contains_burned(self):
        _img, metrics = solve_and_visualize_wildfire(
            grid_size=9,
            wind_speed=2.0,
            wind_direction=45.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern="Center",
            total_time=0.5,
        )
        assert "burned" in metrics.lower()

    def test_metrics_contains_grid_info(self):
        _img, metrics = solve_and_visualize_wildfire(
            grid_size=9,
            wind_speed=2.0,
            wind_direction=45.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern="Center",
            total_time=0.5,
        )
        assert "9x9" in metrics

    @pytest.mark.parametrize("pattern", ["Center", "Edge"])
    def test_different_patterns(self, pattern):
        img, metrics = solve_and_visualize_wildfire(
            grid_size=9,
            wind_speed=2.0,
            wind_direction=0.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern=pattern,
            total_time=0.5,
        )
        assert isinstance(img, PILImage.Image)
        assert isinstance(metrics, str)


# ---------------------------------------------------------------------------
# compare_resolutions_wildfire
# ---------------------------------------------------------------------------


class TestCompareResolutionsWildfire:
    def test_returns_image_and_summary(self):
        img, summary = compare_resolutions_wildfire(
            wind_speed=2.0,
            wind_direction=45.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern="Center",
        )
        assert isinstance(img, PILImage.Image)
        assert isinstance(summary, str)
        assert "MSE" in summary or "omplete" in summary

    def test_uses_default_config(self):
        img, summary = compare_resolutions_wildfire(
            wind_speed=2.0,
            wind_direction=0.0,
            diffusion=0.1,
            fuel_density=1.0,
            ignition_pattern="Center",
        )
        assert img is not None
        assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# create_wildfire_tab
# ---------------------------------------------------------------------------


class TestCreateWildfireTab:
    def test_creates_without_error(self):
        with gr.Blocks():
            create_wildfire_tab()

    def test_creates_with_custom_config(self, wildfire_cfg):
        with gr.Blocks():
            create_wildfire_tab(wildfire_cfg)
