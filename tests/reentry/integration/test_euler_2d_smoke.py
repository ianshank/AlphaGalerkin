"""Integration test: 2D Euler solver on a uniform flow + oblique shock.

Verifies the full pipeline: mesh creation → IC setup → BC setup →
solver execution → result extraction. Uses a very coarse grid and
short run to keep the test fast.
"""

from __future__ import annotations

import numpy as np

from src.reentry.config.freestream import FreestreamConfig
from src.reentry.config.mesh import ReentryMeshConfig
from src.reentry.config.solver import FluxScheme, ReentrySolverConfig
from src.reentry.mesh.structured import StructuredMesh2D
from src.reentry.solver.boundary import (
    BoundaryFace,
    FreestreamBC,
    SupersonicOutflowBC,
)
from src.reentry.solver.euler_2d import Euler2DSolver


class TestEuler2DSmoke:
    """Smoke test: uniform supersonic flow through domain."""

    def test_uniform_flow_stays_uniform(self) -> None:
        """A uniform freestream should propagate unchanged."""
        mesh_config = ReentryMeshConfig(
            name="test",
            nx=20,
            ny=10,
            wall_clustering=False,
            x_min=0.0,
            x_max=2.0,
            y_min=0.0,
            y_max=1.0,
        )
        solver_config = ReentrySolverConfig(
            name="test",
            flux_scheme=FluxScheme.HLLC,
            cfl=0.3,
            max_iterations=50,
            residual_tolerance=1e-10,
        )
        mesh = StructuredMesh2D(mesh_config, n_ghost=2)

        solver = Euler2DSolver(solver_config, mesh, gamma=1.4)

        # Freestream conditions
        rho_inf = 1.225
        u_inf = 300.0
        p_inf = 101325.0

        # Set all BCs to freestream
        fs_config = FreestreamConfig(
            name="test",
            mach=0.87,
            velocity_m_s=u_inf,
            density_kg_m3=rho_inf,
            temperature_K=p_inf / (rho_inf * 287.058),
        )
        fs_bc = FreestreamBC(fs_config)
        for face in BoundaryFace:
            solver.set_bc(face, fs_bc)

        # Initialize
        q0 = solver.initialize_uniform(rho_inf, u_inf, 0.0, p_inf)

        # Short run
        result = solver.solve(q0, t_final=0.001)

        # Uniform flow should remain approximately uniform
        assert result.density.shape == (10, 20)
        np.testing.assert_allclose(result.density, rho_inf, rtol=0.05)
        np.testing.assert_allclose(result.velocity_x, u_inf, rtol=0.05)
        np.testing.assert_allclose(result.pressure, p_inf, rtol=0.05)

    def test_solver_completes_with_wall(self) -> None:
        """Solver runs without crashing with mixed BCs."""
        from src.reentry.config.wall import WallConfig
        from src.reentry.solver.boundary import SymmetryBC, WallBC

        mesh_config = ReentryMeshConfig(
            name="test",
            nx=10,
            ny=8,
            wall_clustering=False,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=0.5,
        )
        solver_config = ReentrySolverConfig(
            name="test",
            flux_scheme=FluxScheme.ROE,
            cfl=0.2,
            max_iterations=20,
        )
        mesh = StructuredMesh2D(mesh_config, n_ghost=2)
        solver = Euler2DSolver(solver_config, mesh, gamma=1.4)

        fs_config = FreestreamConfig(
            name="test",
            mach=3.0,
            velocity_m_s=1000.0,
            density_kg_m3=0.5,
            temperature_K=250.0,
        )
        wall_config = WallConfig(name="test", wall_temperature_K=300.0)

        solver.set_bc(BoundaryFace.WEST, FreestreamBC(fs_config))
        solver.set_bc(BoundaryFace.EAST, SupersonicOutflowBC())
        solver.set_bc(BoundaryFace.SOUTH, WallBC(wall_config))
        solver.set_bc(BoundaryFace.NORTH, SymmetryBC())

        q0 = solver.initialize_uniform(0.5, 1000.0, 0.0, 50000.0)
        result = solver.solve(q0, t_final=0.0005)

        # Should complete without NaN
        assert not np.any(np.isnan(result.density))
        assert not np.any(np.isnan(result.pressure))
        assert result.n_steps > 0
