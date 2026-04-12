"""Tests for dashboard/tabs/missile_defense_tab.py -- Missile defense intercept analysis."""

from __future__ import annotations

import gradio as gr
import numpy as np
import pytest
from PIL import Image as PILImage

from dashboard.tabs.missile_defense_tab import (
    _compute_closest_approach,
    _compute_interceptor_trajectory,
    _compute_potential_flow,
    _compute_threat_trajectory,
    _intercept_probability,
    compare_resolutions_intercept,
    create_missile_defense_tab,
    solve_and_visualize_intercept,
)

# ---------------------------------------------------------------------------
# Default trajectory parameters (small / fast)
# ---------------------------------------------------------------------------

_DEFAULT_ANGLE = 45.0
_DEFAULT_VEL = 3.0
_DEFAULT_GRAVITY = 9.81
_DEFAULT_DT = 0.05
_DEFAULT_MAX_TIME = 5.0
_DEFAULT_INT_X = 0.7
_DEFAULT_INT_Y = 0.0
_DEFAULT_INT_SPEED = 5.0


def _make_threat(**overrides):
    """Build a threat trajectory with sensible defaults."""
    kw = {
        "launch_angle": _DEFAULT_ANGLE,
        "velocity": _DEFAULT_VEL,
        "gravity": _DEFAULT_GRAVITY,
        "dt": _DEFAULT_DT,
        "max_time": _DEFAULT_MAX_TIME,
    }
    kw.update(overrides)
    return _compute_threat_trajectory(**kw)


def _make_interceptor(threat_traj, **overrides):
    """Build an interceptor trajectory with sensible defaults."""
    kw = {
        "start_x": _DEFAULT_INT_X,
        "start_y": _DEFAULT_INT_Y,
        "speed": _DEFAULT_INT_SPEED,
        "threat_positions": threat_traj,
        "dt": _DEFAULT_DT,
    }
    kw.update(overrides)
    return _compute_interceptor_trajectory(**kw)


# ---------------------------------------------------------------------------
# _compute_threat_trajectory
# ---------------------------------------------------------------------------


class TestComputeThreatTrajectory:
    def test_returns_2d_array(self):
        traj = _make_threat()
        assert traj.ndim == 2
        assert traj.shape[1] == 2
        assert traj.shape[0] > 0

    def test_starts_at_origin(self):
        traj = _make_threat()
        # After normalisation x starts at 0.  y starts at the normalised
        # value of 0 (which may be non-zero because the trajectory ends
        # below zero and normalisation shifts).  The key invariant is that
        # x[0] is the minimum x (i.e. 0 after normalisation).
        assert traj[0, 0] == pytest.approx(0.0, abs=1e-6)
        # y[0] should be within the [0, 1] normalised range.
        assert 0.0 <= traj[0, 1] <= 1.0

    def test_parabolic_shape(self):
        traj = _make_threat()
        y_vals = traj[:, 1]
        # y should go up and come back down: max not at first or last index.
        peak_idx = int(np.argmax(y_vals))
        assert 0 < peak_idx < len(y_vals) - 1

    def test_angle_affects_range(self):
        traj_45 = _make_threat(launch_angle=45.0)
        traj_20 = _make_threat(launch_angle=20.0)
        # 45 deg gives maximum range for a projectile; the raw (un-normalised)
        # horizontal distance should be longer than at 20 deg.  Because both
        # are normalised to [0,1] we instead check that 45-deg produces more
        # time steps (longer flight) than 20 deg.
        assert traj_45.shape[0] > traj_20.shape[0]


# ---------------------------------------------------------------------------
# _compute_interceptor_trajectory
# ---------------------------------------------------------------------------


class TestComputeInterceptorTrajectory:
    def test_returns_2d_array(self):
        threat = _make_threat()
        traj = _make_interceptor(threat)
        assert traj.ndim == 2
        assert traj.shape[1] == 2

    def test_starts_at_specified_position(self):
        threat = _make_threat()
        sx, sy = 0.8, 0.1
        traj = _make_interceptor(threat, start_x=sx, start_y=sy)
        np.testing.assert_allclose(traj[0], [sx, sy], atol=1e-10)

    def test_moves_toward_threat(self):
        threat = _make_threat()
        traj = _make_interceptor(threat)
        # Distance between interceptor and threat should decrease over the
        # first several steps (the interceptor is homing in).
        min_len = min(len(threat), len(traj))
        dists = np.linalg.norm(threat[:min_len] - traj[:min_len], axis=1)
        # The initial distance should be larger than the distance a few steps later.
        assert dists[0] > dists[min(5, min_len - 1)]


# ---------------------------------------------------------------------------
# _compute_closest_approach
# ---------------------------------------------------------------------------


class TestComputeClosestApproach:
    def test_returns_three_values(self):
        threat = _make_threat()
        intercept = _make_interceptor(threat)
        result = _compute_closest_approach(threat, intercept)
        assert len(result) == 3
        miss_dist, time_frac, idx = result
        assert isinstance(miss_dist, float)
        assert isinstance(time_frac, float)
        assert isinstance(idx, int)

    def test_miss_distance_non_negative(self):
        threat = _make_threat()
        intercept = _make_interceptor(threat)
        miss_dist, _, _ = _compute_closest_approach(threat, intercept)
        assert miss_dist >= 0.0

    def test_identical_trajectories_zero_miss(self):
        threat = _make_threat()
        miss_dist, _, _ = _compute_closest_approach(threat, threat)
        assert miss_dist == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# _compute_potential_flow
# ---------------------------------------------------------------------------


class TestComputePotentialFlow:
    def test_shape_matches_grid(self):
        n = 9
        field = _compute_potential_flow(n, 0.5, 0.5)
        assert field.shape == (n, n)

    def test_dtype_is_float32(self):
        field = _compute_potential_flow(9, 0.5, 0.5)
        assert field.dtype == np.float32

    def test_no_nans(self):
        field = _compute_potential_flow(9, 0.5, 0.5)
        assert not np.any(np.isnan(field))


# ---------------------------------------------------------------------------
# _intercept_probability
# ---------------------------------------------------------------------------


class TestInterceptProbability:
    def test_zero_miss_gives_one(self):
        p = _intercept_probability(0.0, 0.05)
        assert p == pytest.approx(1.0, abs=1e-8)

    def test_large_miss_gives_low_prob(self):
        p = _intercept_probability(1.0, 0.05)
        assert p < 0.01

    def test_bounded_zero_to_one(self):
        for miss in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            p = _intercept_probability(miss, 0.05)
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# solve_and_visualize_intercept
# ---------------------------------------------------------------------------


class TestSolveAndVisualizeIntercept:
    def test_returns_image_and_metrics(self):
        img, metrics = solve_and_visualize_intercept(
            grid_size=9,
            threat_angle=45.0,
            threat_velocity=3.0,
            interceptor_x=0.7,
            interceptor_y=0.0,
            interceptor_speed=5.0,
        )
        assert isinstance(img, PILImage.Image)
        assert isinstance(metrics, str)

    def test_metrics_contains_miss_distance(self):
        _, metrics = solve_and_visualize_intercept(
            grid_size=9,
            threat_angle=45.0,
            threat_velocity=3.0,
            interceptor_x=0.7,
            interceptor_y=0.0,
            interceptor_speed=5.0,
        )
        assert "miss" in metrics.lower()

    def test_metrics_contains_probability(self):
        _, metrics = solve_and_visualize_intercept(
            grid_size=9,
            threat_angle=45.0,
            threat_velocity=3.0,
            interceptor_x=0.7,
            interceptor_y=0.0,
            interceptor_speed=5.0,
        )
        assert "P_kill" in metrics or "probability" in metrics.lower()

    @pytest.mark.parametrize("grid_size", [9, 13])
    def test_different_grid_sizes(self, grid_size):
        img, metrics = solve_and_visualize_intercept(
            grid_size=grid_size,
            threat_angle=45.0,
            threat_velocity=3.0,
            interceptor_x=0.7,
            interceptor_y=0.0,
            interceptor_speed=5.0,
        )
        assert isinstance(img, PILImage.Image)
        assert isinstance(metrics, str)


# ---------------------------------------------------------------------------
# compare_resolutions_intercept
# ---------------------------------------------------------------------------


class TestCompareResolutionsIntercept:
    def test_returns_image_and_summary(self):
        img, summary = compare_resolutions_intercept(
            threat_angle=45.0,
            threat_velocity=3.0,
            interceptor_x=0.7,
            interceptor_y=0.0,
            interceptor_speed=5.0,
        )
        assert isinstance(img, PILImage.Image)
        assert isinstance(summary, str)

    def test_summary_contains_resolution_info(self):
        _, summary = compare_resolutions_intercept(
            threat_angle=45.0,
            threat_velocity=3.0,
            interceptor_x=0.7,
            interceptor_y=0.0,
            interceptor_speed=5.0,
        )
        assert "MSE" in summary or any(ch.isdigit() for ch in summary)


# ---------------------------------------------------------------------------
# create_missile_defense_tab
# ---------------------------------------------------------------------------


class TestCreateMissileDefenseTab:
    def test_creates_without_error(self):
        with gr.Blocks():
            create_missile_defense_tab()

    def test_creates_with_custom_config(self, missile_defense_cfg):
        with gr.Blocks():
            create_missile_defense_tab(missile_defense_cfg)
