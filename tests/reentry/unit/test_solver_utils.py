"""Tests for solver utilities: CFL, shock detector, residual monitor."""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.solver.cfl import CFLController
from src.reentry.solver.residual import ResidualMonitor
from src.reentry.solver.shock_detector import ShockDetector


class TestCFLController:
    def test_constant_cfl(self) -> None:
        ctrl = CFLController(cfl_target=0.5, adaptive=False)
        assert ctrl.current_cfl(0) == 0.5
        assert ctrl.current_cfl(1000) == 0.5

    def test_cfl_ramping(self) -> None:
        ctrl = CFLController(cfl_target=0.5, cfl_ramp_start=0.1, cfl_ramp_steps=100, adaptive=True)
        assert ctrl.current_cfl(0) == pytest.approx(0.1)
        assert ctrl.current_cfl(50) == pytest.approx(0.3)
        assert ctrl.current_cfl(100) == pytest.approx(0.5)
        assert ctrl.current_cfl(200) == pytest.approx(0.5)

    def test_compute_timestep(self) -> None:
        ctrl = CFLController(cfl_target=0.5, adaptive=False)
        ws = np.ones((5, 5)) * 340.0  # ~speed of sound
        dx = np.ones((5, 5)) * 0.01
        dy = np.ones((5, 5)) * 0.01
        dt = ctrl.compute_timestep(ws, dx, dy, step=0)
        expected = 0.5 * 0.01 / 340.0
        assert dt == pytest.approx(expected, rel=1e-6)

    def test_wave_speed(self) -> None:
        u = np.array([[100.0]])
        v = np.array([[50.0]])
        a = np.array([[340.0]])
        ws = CFLController.wave_speed(u, v, a)
        assert ws[0, 0] == pytest.approx(490.0)


class TestShockDetector:
    def test_smooth_field_no_shock(self) -> None:
        detector = ShockDetector(pressure_threshold=0.3)
        p = np.ones((10, 10)) * 101325.0
        sigma = detector.detect(p)
        assert np.all(sigma < 0.01)

    def test_shock_detected(self) -> None:
        detector = ShockDetector(pressure_threshold=0.3, enable_ducros=False)
        p = np.ones((20, 20)) * 101325.0
        p[:, 10:] = 10 * 101325.0  # Pressure jump
        sigma = detector.detect(p)
        # Near the jump, sigma should be high
        assert sigma[:, 9:11].max() > 0.5

    def test_ducros_sensor(self) -> None:
        u = np.ones((10, 10)) * 100.0
        v = np.zeros((10, 10))
        ducros = ShockDetector._ducros_sensor(u, v)
        assert ducros.shape == (10, 10)
        assert np.all(ducros >= 0)
        assert np.all(ducros <= 1)


class TestResidualMonitor:
    def test_convergence_detection(self) -> None:
        monitor = ResidualMonitor(log_interval=1000)
        rho = np.ones((10, 10))

        monitor.update(rho, step=0, dt=0.001)
        # Same density => zero residual
        l2 = monitor.update(rho, step=1, dt=0.001)
        assert l2 == pytest.approx(0.0)
        assert monitor.is_converged(tolerance=1e-8)

    def test_nonzero_residual(self) -> None:
        monitor = ResidualMonitor()
        rho1 = np.ones((10, 10))
        rho2 = rho1 * 1.01  # 1% change

        monitor.update(rho1, step=0, dt=0.001)
        l2 = monitor.update(rho2, step=1, dt=0.001)
        assert l2 > 0

    def test_history_tracking(self) -> None:
        monitor = ResidualMonitor()
        rho = np.ones((5, 5))
        for i in range(5):
            monitor.update(rho * (1 + 0.01 * i), step=i, dt=0.001)
        assert monitor.history.n_records == 5
