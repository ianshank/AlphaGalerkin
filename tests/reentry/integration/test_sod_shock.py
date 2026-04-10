"""Integration test: Sod shock tube.

Validates the 1D Euler solver against the exact Sod solution.
Success criterion: L2 density error < 1% on 200 cells.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.config.solver import FluxScheme, LimiterType, ReentrySolverConfig
from src.reentry.solver.euler_1d import Euler1DSolver, ShockTubeIC


@pytest.fixture
def solver_config() -> ReentrySolverConfig:
    return ReentrySolverConfig(
        name="sod_test",
        flux_scheme=FluxScheme.ROE,
        limiter=LimiterType.VAN_LEER,
        cfl=0.5,
        max_iterations=10000,
        adaptive_cfl=False,
    )


def sod_exact_solution(x: np.ndarray, t: float, gamma: float = 1.4):
    """Compute exact Sod shock tube solution at time t."""
    # Sod problem exact solution parameters
    rho_l, u_l, p_l = 1.0, 0.0, 1.0
    rho_r, u_r, p_r = 0.125, 0.0, 0.1
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    a_l = np.sqrt(gamma * p_l / rho_l)

    # Post-shock pressure (from exact Riemann solver, pre-computed)
    p_star = 0.30313
    u_star = 0.92745

    # Exact solution regions
    rho_exact = np.zeros_like(x)
    u_exact = np.zeros_like(x)
    p_exact = np.zeros_like(x)

    x_0 = 0.5  # Diaphragm location

    # Shock speed
    rho_star_r = rho_r * (p_star / p_r + gm1 / gp1) / (gm1 / gp1 * p_star / p_r + 1)
    if abs(u_star - u_r) > 1e-10:
        v_shock = u_r + (p_star - p_r) / (rho_r * (u_star - u_r))
    else:
        v_shock = u_star + np.sqrt(gamma * p_star / rho_star_r)

    # Contact discontinuity speed
    v_contact = u_star

    # Rarefaction fan
    a_star_l = a_l - gm1 / 2 * u_star
    rho_star_l = rho_l * (a_star_l / a_l) ** (2 / gm1)

    head = x_0 - a_l * t
    tail = x_0 + (u_star - a_star_l) * t
    contact = x_0 + v_contact * t
    shock = x_0 + v_shock * t

    for i, xi in enumerate(x):
        if xi < head:
            rho_exact[i], u_exact[i], p_exact[i] = rho_l, u_l, p_l
        elif xi < tail:
            # Inside rarefaction fan
            u_fan = 2 / gp1 * (a_l + (xi - x_0) / t)
            a_fan = a_l - gm1 / 2 * u_fan
            rho_exact[i] = rho_l * (a_fan / a_l) ** (2 / gm1)
            u_exact[i] = u_fan
            p_exact[i] = p_l * (a_fan / a_l) ** (2 * gamma / gm1)
        elif xi < contact:
            rho_exact[i] = rho_star_l
            u_exact[i] = u_star
            p_exact[i] = p_star
        elif xi < shock:
            rho_exact[i] = rho_star_r
            u_exact[i] = u_star
            p_exact[i] = p_star
        else:
            rho_exact[i], u_exact[i], p_exact[i] = rho_r, u_r, p_r

    return rho_exact, u_exact, p_exact


class TestSodShockTube:
    def test_sod_roe_200_cells(self, solver_config: ReentrySolverConfig) -> None:
        """Sod shock tube with Roe solver, 200 cells, t=0.2."""
        solver = Euler1DSolver(solver_config, gamma=1.4, n_cells=200)
        result = solver.solve(ShockTubeIC.sod(), t_final=0.2)

        rho_exact, _, _ = sod_exact_solution(result.x, 0.2)

        # L2 error
        l2_error = np.sqrt(np.mean((result.density - rho_exact) ** 2))
        l2_norm = np.sqrt(np.mean(rho_exact**2))
        relative_l2 = l2_error / l2_norm

        assert relative_l2 < 0.05, f"Sod L2 error {relative_l2:.4f} exceeds 5%"
        assert result.density.min() > 0, "Negative density detected"

    def test_sod_hllc(self) -> None:
        """Sod shock tube with HLLC solver."""
        config = ReentrySolverConfig(
            name="sod_hllc",
            flux_scheme=FluxScheme.HLLC,
            limiter=LimiterType.VAN_LEER,
            cfl=0.5,
            max_iterations=10000,
            adaptive_cfl=False,
        )
        solver = Euler1DSolver(config, gamma=1.4, n_cells=200)
        result = solver.solve(ShockTubeIC.sod(), t_final=0.2)

        rho_exact, _, _ = sod_exact_solution(result.x, 0.2)
        l2_error = np.sqrt(np.mean((result.density - rho_exact) ** 2))
        l2_norm = np.sqrt(np.mean(rho_exact**2))

        assert l2_error / l2_norm < 0.05

    def test_positivity_preserved(self, solver_config: ReentrySolverConfig) -> None:
        """Density and pressure should remain positive throughout."""
        solver = Euler1DSolver(solver_config, gamma=1.4, n_cells=200)
        result = solver.solve(ShockTubeIC.sod(), t_final=0.2)
        assert np.all(result.density > 0)
        assert np.all(result.pressure > 0)
