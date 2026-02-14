"""Tests for manufactured solutions (physics/manufactured.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.physics.base import ManufacturedSolution
from src.alphagalerkin.physics.manufactured import (
    MMS_CATALOG,
    poisson_polynomial,
    poisson_sinsin,
)


def _interior_points(n: int = 50) -> np.ndarray:
    """Generate n random interior points in [0, 1]^2."""
    rng = np.random.default_rng(seed=42)
    return rng.random((n, 2))


def _boundary_points(n: int = 20) -> np.ndarray:
    """Generate n points on the boundary of [0, 1]^2."""
    rng = np.random.default_rng(seed=42)
    points = []
    per_side = max(1, n // 4)
    for _ in range(per_side):
        t = rng.random()
        points.append([t, 0.0])  # bottom
        points.append([t, 1.0])  # top
        points.append([0.0, t])  # left
        points.append([1.0, t])  # right
    return np.array(points[:n])


class TestPoissonSinSin:
    """poisson_sinsin manufactured solution."""

    def test_returns_manufactured_solution(self) -> None:
        mms = poisson_sinsin()

        assert isinstance(mms, ManufacturedSolution)
        assert mms.name == "poisson_sinsin"

    def test_exact_solution_shape(self) -> None:
        mms = poisson_sinsin()
        pts = _interior_points(30)

        u = mms.exact_solution(pts)

        assert u.shape == (30,)

    def test_forcing_shape(self) -> None:
        mms = poisson_sinsin()
        pts = _interior_points(25)

        f = mms.forcing(pts)

        assert f.shape == (25,)

    def test_boundary_is_zero(self) -> None:
        mms = poisson_sinsin()
        pts = _boundary_points(20)

        bc = mms.boundary_data(pts)

        np.testing.assert_allclose(bc, 0.0, atol=1e-12)

    def test_exact_at_boundary_is_zero(self) -> None:
        """sin(pi*x)*sin(pi*y) vanishes on [0,1]^2 boundary."""
        mms = poisson_sinsin()
        pts = _boundary_points(20)

        u = mms.exact_solution(pts)

        np.testing.assert_allclose(u, 0.0, atol=1e-12)

    def test_forcing_consistent_with_laplacian(self) -> None:
        """For -laplacian(u) = f, f should equal 2*pi^2*sin(pi*x)*sin(pi*y)."""
        mms = poisson_sinsin()
        pts = _interior_points(50)

        f = mms.forcing(pts)
        expected = (
            2.0
            * np.pi ** 2
            * np.sin(np.pi * pts[:, 0])
            * np.sin(np.pi * pts[:, 1])
        )

        np.testing.assert_allclose(f, expected, rtol=1e-12)

    def test_convergence_order(self) -> None:
        mms = poisson_sinsin()

        assert mms.expected_convergence_order == pytest.approx(2.0)

    def test_compute_error_zero_for_exact(self) -> None:
        mms = poisson_sinsin()
        pts = _interior_points(50)
        exact = mms.exact_solution(pts)

        error = mms.compute_error(exact, pts)

        assert error == pytest.approx(0.0, abs=1e-15)


class TestPoissonPolynomial:
    """poisson_polynomial manufactured solution."""

    def test_returns_manufactured_solution(self) -> None:
        mms = poisson_polynomial()

        assert isinstance(mms, ManufacturedSolution)
        assert mms.name == "poisson_polynomial"

    def test_exact_solution_shape(self) -> None:
        mms = poisson_polynomial()
        pts = _interior_points(30)

        u = mms.exact_solution(pts)

        assert u.shape == (30,)

    def test_boundary_is_zero(self) -> None:
        mms = poisson_polynomial()
        pts = _boundary_points(20)

        bc = mms.boundary_data(pts)

        np.testing.assert_allclose(bc, 0.0, atol=1e-12)

    def test_exact_at_boundary_is_zero(self) -> None:
        """x*(1-x)*y*(1-y) vanishes on [0,1]^2 boundary."""
        mms = poisson_polynomial()
        pts = _boundary_points(20)

        u = mms.exact_solution(pts)

        np.testing.assert_allclose(u, 0.0, atol=1e-12)

    def test_forcing_consistent(self) -> None:
        """For u = x(1-x)y(1-y), -laplacian(u) = 2(x(1-x) + y(1-y))."""
        mms = poisson_polynomial()
        pts = _interior_points(50)
        x, y = pts[:, 0], pts[:, 1]

        f = mms.forcing(pts)
        expected = 2.0 * (x * (1 - x) + y * (1 - y))

        np.testing.assert_allclose(f, expected, rtol=1e-12)

    def test_convergence_order(self) -> None:
        mms = poisson_polynomial()

        assert mms.expected_convergence_order == pytest.approx(2.0)

    def test_compute_error_nonzero_for_perturbed(self) -> None:
        mms = poisson_polynomial()
        pts = _interior_points(50)
        exact = mms.exact_solution(pts)
        perturbed = exact + 0.1

        error = mms.compute_error(perturbed, pts)

        assert error > 0.0


class TestMMSCatalog:
    """MMS_CATALOG dictionary of manufactured solutions."""

    def test_contains_sinsin(self) -> None:
        assert "poisson_sinsin" in MMS_CATALOG

    def test_contains_polynomial(self) -> None:
        assert "poisson_polynomial" in MMS_CATALOG

    def test_catalog_values_are_manufactured_solutions(self) -> None:
        for name, mms in MMS_CATALOG.items():
            assert isinstance(mms, ManufacturedSolution), (
                f"MMS_CATALOG['{name}'] is not a ManufacturedSolution"
            )

    def test_catalog_length(self) -> None:
        assert len(MMS_CATALOG) >= 2
