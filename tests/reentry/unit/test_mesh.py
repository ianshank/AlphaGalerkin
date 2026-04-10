"""Tests for 2D structured mesh."""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.config.mesh import ReentryMeshConfig
from src.reentry.mesh.structured import StructuredMesh2D


class TestStructuredMesh2D:
    def test_uniform_mesh(self) -> None:
        config = ReentryMeshConfig(
            name="test",
            nx=10,
            ny=5,
            wall_clustering=False,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        )
        mesh = StructuredMesh2D(config, n_ghost=2)
        assert mesh.nx == 10
        assert mesh.ny == 5
        assert mesh.total_nx == 14
        assert mesh.total_ny == 9

    def test_cell_centers(self) -> None:
        config = ReentryMeshConfig(
            name="test",
            nx=10,
            ny=10,
            wall_clustering=False,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        )
        mesh = StructuredMesh2D(config, n_ghost=1)
        x, y = mesh.cell_centers()
        assert x.shape == (10, 10)
        np.testing.assert_allclose(x[0, 0], 0.05, atol=1e-10)
        np.testing.assert_allclose(x[0, -1], 0.95, atol=1e-10)

    def test_wall_clustering(self) -> None:
        config = ReentryMeshConfig(
            name="test",
            nx=10,
            ny=20,
            wall_clustering=True,
            wall_first_cell_height=0.001,
            wall_growth_rate=1.3,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=2.0,
        )
        mesh = StructuredMesh2D(config, n_ghost=2)
        # First cell should be much smaller than last
        dy = mesh.metrics.dy
        assert dy[0, 0] < dy[-1, 0]
        # First cell ~ wall_first_cell_height (scaled)
        assert dy[0, 0] < 0.1 * dy[-1, 0]

    def test_allocate_field(self) -> None:
        config = ReentryMeshConfig(
            name="test",
            nx=10,
            ny=5,
            wall_clustering=False,
        )
        mesh = StructuredMesh2D(config, n_ghost=2)
        field = mesh.allocate_field(4)
        assert field.shape == (9, 14, 4)
        np.testing.assert_allclose(field, 0.0)

    def test_min_cell_size(self) -> None:
        config = ReentryMeshConfig(
            name="test",
            nx=10,
            ny=10,
            wall_clustering=False,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        )
        mesh = StructuredMesh2D(config, n_ghost=1)
        assert mesh.min_cell_size() == pytest.approx(0.1, abs=1e-10)

    def test_interior_slice(self) -> None:
        config = ReentryMeshConfig(
            name="test",
            nx=10,
            ny=5,
            wall_clustering=False,
        )
        mesh = StructuredMesh2D(config, n_ghost=2)
        si, sj = mesh.interior_slice
        field = mesh.allocate_field()
        interior = field[si, sj]
        assert interior.shape == (5, 10)
