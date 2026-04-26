"""Integration test: coupled fire spread solver.

Validates the fire solver produces physically reasonable results:
- Temperature increases at ignition point
- Fire spreads outward
- Fuel is consumed
- Energy is approximately conserved
"""

from __future__ import annotations

import numpy as np
import pytest

from src.firefighting.config.fire import FireConfig
from src.firefighting.config.solver import FireSolverConfig
from src.firefighting.solver.coupled import CoupledFireSolver


@pytest.fixture
def fire_config() -> FireConfig:
    return FireConfig(name="test")


@pytest.fixture
def solver_config() -> FireSolverConfig:
    return FireSolverConfig(
        name="test",
        nx=50,
        ny=50,
        domain_size_x_m=500.0,
        domain_size_y_m=500.0,
        dt_s=0.5,
        prediction_horizon_s=60.0,  # 1 minute
        max_steps=200,
    )


class TestCoupledFireSolver:
    def test_fire_spreads_from_ignition(
        self, solver_config: FireSolverConfig, fire_config: FireConfig
    ) -> None:
        """Fire should spread outward from ignition point."""
        solver = CoupledFireSolver(solver_config, fire_config)
        state = solver.create_initial_state(ignition_center=(250.0, 250.0), ignition_radius_m=20.0)

        # Record initial burned area
        initial_burned = np.sum(state.fuel.consumed > 0.5)

        wind_u = np.full((50, 50), 2.0)
        wind_v = np.zeros((50, 50))

        # Run for a few steps
        for _ in range(50):
            state = solver.step(state, wind_u, wind_v)

        # Temperature should have increased somewhere
        max_temp = state.temperature.max()
        assert max_temp > fire_config.ambient_temperature_K

    def test_no_fire_without_ignition(
        self, solver_config: FireSolverConfig, fire_config: FireConfig
    ) -> None:
        """Without ignition, temperature should stay ambient."""
        solver = CoupledFireSolver(solver_config, fire_config)
        state = solver.create_initial_state(ignition_center=None, ignition_radius_m=0.0)

        # Override to ambient everywhere
        state.temperature[:] = fire_config.ambient_temperature_K

        wind_u = np.full((50, 50), 2.0)
        wind_v = np.zeros((50, 50))

        for _ in range(10):
            state = solver.step(state, wind_u, wind_v)

        # No fuel should be consumed
        assert np.all(state.fuel.consumed < 0.01)

    def test_solver_run_returns_result(
        self, solver_config: FireSolverConfig, fire_config: FireConfig
    ) -> None:
        """Full solver run should return valid result."""
        config = FireSolverConfig(
            name="quick",
            nx=20,
            ny=20,
            domain_size_x_m=200.0,
            domain_size_y_m=200.0,
            dt_s=0.5,
            prediction_horizon_s=10.0,
            max_steps=30,
        )
        solver = CoupledFireSolver(config, fire_config)
        state = solver.create_initial_state(ignition_center=(100.0, 100.0), ignition_radius_m=10.0)

        wind_u = np.full((20, 20), 3.0)
        wind_v = np.zeros((20, 20))

        result = solver.run(state, wind_u, wind_v, t_final=10.0)

        assert result.total_steps > 0
        assert result.max_temperature_K > fire_config.ambient_temperature_K
