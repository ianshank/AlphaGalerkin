"""Tests for boundary conditions."""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.config.freestream import FreestreamConfig
from src.reentry.config.wall import WallConfig
from src.reentry.solver.boundary import (
    BoundaryFace,
    FreestreamBC,
    SupersonicOutflowBC,
    SymmetryBC,
    WallBC,
)


@pytest.fixture
def q_field() -> np.ndarray:
    """Conservative variable field (10, 10, 4) with ghost=2."""
    q = np.zeros((10, 10, 4), dtype=np.float64)
    # Interior: uniform flow
    q[2:8, 2:8, 0] = 1.225  # density
    q[2:8, 2:8, 1] = 1.225 * 100.0  # rho*u
    q[2:8, 2:8, 2] = 0.0  # rho*v
    q[2:8, 2:8, 3] = 1.225 * (101325.0 / (0.4 * 1.225) + 0.5 * 100**2)
    return q


class TestFreestreamBC:
    def test_west_face(self, q_field: np.ndarray) -> None:
        config = FreestreamConfig(
            name="test",
            mach=10.0,
            velocity_m_s=3000.0,
            density_kg_m3=0.01,
            temperature_K=250.0,
        )
        bc = FreestreamBC(config)
        bc.apply(q_field, BoundaryFace.WEST, n_ghost=2, gamma=1.4)
        # Ghost cells should be set to freestream density
        assert q_field[5, 0, 0] > 0
        assert q_field[5, 1, 0] > 0


class TestWallBC:
    def test_no_slip(self, q_field: np.ndarray) -> None:
        config = WallConfig(name="test", wall_temperature_K=300.0)
        bc = WallBC(config)
        bc.apply(q_field, BoundaryFace.SOUTH, n_ghost=2, gamma=1.4)
        # Ghost cell momentum should be opposite (no-slip)
        rhou_interior = q_field[2, 5, 1]
        rhou_ghost = q_field[1, 5, 1]
        assert rhou_ghost == pytest.approx(-rhou_interior)


class TestSymmetryBC:
    def test_south_symmetry(self, q_field: np.ndarray) -> None:
        bc = SymmetryBC()
        bc.apply(q_field, BoundaryFace.SOUTH, n_ghost=2, gamma=1.4)
        # v-momentum should be reflected
        rhov_interior = q_field[2, 5, 2]
        rhov_ghost = q_field[1, 5, 2]
        assert rhov_ghost == pytest.approx(-rhov_interior)
        # Density should be copied
        assert q_field[1, 5, 0] == pytest.approx(q_field[2, 5, 0])


class TestSupersonicOutflowBC:
    def test_east_extrapolation(self, q_field: np.ndarray) -> None:
        bc = SupersonicOutflowBC()
        bc.apply(q_field, BoundaryFace.EAST, n_ghost=2, gamma=1.4)
        # Ghost cells should match last interior cell
        last_interior = q_field[5, 7, 0]  # nx=6 interior, ghost=2
        assert q_field[5, 8, 0] == pytest.approx(last_interior)
        assert q_field[5, 9, 0] == pytest.approx(last_interior)
