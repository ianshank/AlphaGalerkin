"""Manufactured solution catalog for MMS testing."""
from __future__ import annotations

import numpy as np

from src.alphagalerkin.physics.base import ManufacturedSolution


def poisson_sinsin() -> ManufacturedSolution:
    """U = sin(pi*x)*sin(pi*y) on [0,1]^2."""
    def exact(points: np.ndarray) -> np.ndarray:
        return (
            np.sin(np.pi * points[:, 0])
            * np.sin(np.pi * points[:, 1])
        )

    def forcing(points: np.ndarray) -> np.ndarray:
        return (
            2.0
            * np.pi**2
            * np.sin(np.pi * points[:, 0])
            * np.sin(np.pi * points[:, 1])
        )

    def boundary(points: np.ndarray) -> np.ndarray:
        return np.zeros(len(points))

    return ManufacturedSolution(
        exact_solution=exact,
        forcing=forcing,
        boundary_data=boundary,
        expected_convergence_order=2.0,
        name="poisson_sinsin",
    )


def poisson_polynomial() -> ManufacturedSolution:
    """U = x*(1-x)*y*(1-y) on [0,1]^2.

    Exactly representable by p>=2 elements.
    """
    def exact(points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        result: np.ndarray = x * (1 - x) * y * (1 - y)
        return result

    def forcing(points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        result: np.ndarray = 2.0 * (x * (1 - x) + y * (1 - y))
        return result

    def boundary(points: np.ndarray) -> np.ndarray:
        return np.zeros(len(points))

    return ManufacturedSolution(
        exact_solution=exact,
        forcing=forcing,
        boundary_data=boundary,
        expected_convergence_order=2.0,
        name="poisson_polynomial",
    )


# Catalog of all manufactured solutions
MMS_CATALOG: dict[str, ManufacturedSolution] = {
    "poisson_sinsin": poisson_sinsin(),
    "poisson_polynomial": poisson_polynomial(),
}
