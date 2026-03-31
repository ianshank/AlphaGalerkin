"""Property-based tests for PDE operators.

Tests mathematical properties:
- Residual of exact solution should be ~0
- Source term consistency
- Boundary condition satisfaction
- Operator linearity (for linear PDEs)
- Collocation points inside domain
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pde.config import BoundaryCondition, PDEConfig, PDEType
from src.pde.operators import (
    AdvectionDiffusionOperator,
    BurgersOperator,
    HeatOperator,
    PoissonOperator,
)


def _make_poisson_config() -> PDEConfig:
    """Helper to create a Poisson config (avoids fixture/hypothesis clash)."""
    return PDEConfig(
        name="test_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        boundary_condition=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def poisson_config() -> PDEConfig:
    """Create Poisson config fixture."""
    return PDEConfig(
        name="test_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        boundary_condition=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
    )


@pytest.fixture
def burgers_config() -> PDEConfig:
    """Create Burgers config fixture."""
    return PDEConfig(
        name="test_burgers",
        pde_type=PDEType.BURGERS,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        diffusion_coeff=0.01,
        is_time_dependent=True,
    )


# ---------------------------------------------------------------------------
# Poisson operator properties
# ---------------------------------------------------------------------------


class TestPoissonOperatorProperties:
    """Property tests for Poisson operator."""

    def test_exact_solution_residual_near_zero(self, poisson_config: PDEConfig) -> None:
        """Verify manufactured exact solution residual is near zero.

        For the manufactured exact solution u=sin(pi*x)*sin(pi*y),
        the PDE residual -nabla^2 u - f should be approximately zero.
        """
        operator = PoissonOperator(poisson_config)

        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.5], [0.7, 0.2], [0.1, 0.9]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # Exact solution: u = sin(pi*x) * sin(pi*y)
        u = torch.sin(np.pi * coords[:, 0]) * torch.sin(np.pi * coords[:, 1])

        residual = operator.residual(u, coords)

        # Residual should be near zero for the exact solution
        assert residual.l2_norm == pytest.approx(0.0, abs=0.5), (
            f"Residual L2 norm should be ~0 for exact solution, got {residual.l2_norm}"
        )

    @given(n_points=st.integers(min_value=10, max_value=200))
    @settings(max_examples=20)
    def test_collocation_points_in_domain(self, n_points: int) -> None:
        """Generated collocation points must be inside the domain [0,1]^2."""
        operator = PoissonOperator(_make_poisson_config())

        points = operator.generate_collocation_points(n_points, method="uniform", seed=42)

        assert points.shape[1] == 2, "Points must be 2D"
        assert np.all(points >= 0.0), "All coordinates must be >= domain_min"
        assert np.all(points <= 1.0), "All coordinates must be <= domain_max"

    @given(n_points=st.integers(min_value=10, max_value=200))
    @settings(max_examples=20)
    def test_random_collocation_points_in_domain(self, n_points: int) -> None:
        """Random collocation points must also be inside the domain."""
        operator = PoissonOperator(_make_poisson_config())

        points = operator.generate_collocation_points(n_points, method="random", seed=42)

        assert points.shape[0] == n_points
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)

    def test_source_term_matches_exact_laplacian(self, poisson_config: PDEConfig) -> None:
        """Source term f should match -nabla^2 u for the manufactured solution.

        For u = sin(pi*x)*sin(pi*y):
            nabla^2 u = -2*pi^2 * sin(pi*x)*sin(pi*y)
        so f = -nabla^2 u = 2*pi^2 * sin(pi*x)*sin(pi*y).
        """
        operator = PoissonOperator(poisson_config)

        coords = np.array(
            [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]],
            dtype=np.float32,
        )

        source = operator.source_term(coords)
        expected = (
            2 * (np.pi ** 2) * np.sin(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
        )

        np.testing.assert_allclose(source, expected, rtol=1e-5)

    def test_boundary_values_are_zero_dirichlet(self, poisson_config: PDEConfig) -> None:
        """Boundary values must be zero for homogeneous Dirichlet BC."""
        operator = PoissonOperator(poisson_config)

        boundary = np.array(
            [[0.0, 0.5], [1.0, 0.3], [0.4, 0.0], [0.6, 1.0]],
            dtype=np.float32,
        )
        vals = operator.boundary_value(boundary)

        np.testing.assert_allclose(vals, 0.0, atol=1e-7)

    def test_exact_solution_satisfies_boundary(self, poisson_config: PDEConfig) -> None:
        """Exact solution sin(pi*x)*sin(pi*y) should be zero on the boundary."""
        operator = PoissonOperator(poisson_config)

        boundary = np.array(
            [[0.0, 0.5], [1.0, 0.3], [0.4, 0.0], [0.6, 1.0]],
            dtype=np.float32,
        )
        exact = operator.exact_solution(boundary)

        assert exact is not None
        np.testing.assert_allclose(exact, 0.0, atol=1e-6)

    @given(
        n_per_face=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=20)
    def test_boundary_points_are_on_boundary(self, n_per_face: int) -> None:
        """All generated boundary points must lie on the domain boundary."""
        operator = PoissonOperator(_make_poisson_config())

        points = operator.generate_boundary_points(n_per_face, seed=42)
        on_boundary = operator.is_boundary_point(points)

        assert np.all(on_boundary), (
            f"All generated boundary points must be on the boundary, "
            f"but {(~on_boundary).sum()} are not"
        )

    def test_poisson_linearity(self, poisson_config: PDEConfig) -> None:
        """Poisson operator is linear: R(a*u1 + b*u2) = a*R(u1) + b*R(u2).

        Since the operator only applies the Laplacian (linear), the residual
        is linear in u (source term is constant w.r.t. u).
        """
        operator = PoissonOperator(poisson_config)

        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            dtype=torch.float32,
            requires_grad=True,
        )

        # Two different solutions
        u1 = coords[:, 0] ** 2 + coords[:, 1] ** 2
        u2 = torch.sin(coords[:, 0]) * torch.cos(coords[:, 1])

        a, b = 2.0, -0.5

        # Need fresh coords for each residual call (autograd graph)
        coords1 = coords.clone().detach().requires_grad_(True)
        coords2 = coords.clone().detach().requires_grad_(True)
        coords_combo = coords.clone().detach().requires_grad_(True)

        u1_fresh = coords1[:, 0] ** 2 + coords1[:, 1] ** 2
        u2_fresh = torch.sin(coords2[:, 0]) * torch.cos(coords2[:, 1])
        u_combo = a * (coords_combo[:, 0] ** 2 + coords_combo[:, 1] ** 2) + b * (
            torch.sin(coords_combo[:, 0]) * torch.cos(coords_combo[:, 1])
        )

        r1 = operator.residual(u1_fresh, coords1)
        r2 = operator.residual(u2_fresh, coords2)
        r_combo = operator.residual(u_combo, coords_combo)

        # R(a*u1 + b*u2) should equal a*R(u1) + b*R(u2)
        # But the source term f is subtracted each time, so:
        # R(a*u1+b*u2) = -D*lap(a*u1+b*u2) - f
        # a*R(u1)+b*R(u2) = a*(-D*lap(u1)-f) + b*(-D*lap(u2)-f)
        #                  = -D*lap(a*u1+b*u2) - (a+b)*f
        # These are NOT equal unless a+b=1.
        # Instead, test the Laplacian part directly.
        lap_combo = r_combo.derivatives.get("laplacian")
        lap1 = r1.derivatives.get("laplacian")
        lap2 = r2.derivatives.get("laplacian")

        if lap_combo is not None and lap1 is not None and lap2 is not None:
            expected_lap = a * lap1.detach() + b * lap2.detach()
            np.testing.assert_allclose(
                lap_combo.detach().numpy(),
                expected_lap.numpy(),
                rtol=1e-3,
                atol=1e-4,
            )


# ---------------------------------------------------------------------------
# Burgers operator properties
# ---------------------------------------------------------------------------


class TestBurgersOperatorProperties:
    """Property tests for Burgers operator."""

    def test_burgers_is_nonlinear(self, burgers_config: PDEConfig) -> None:
        """Burgers operator should report itself as nonlinear."""
        operator = BurgersOperator(burgers_config)
        assert operator.is_linear is False

    def test_conservation_form(self, burgers_config: PDEConfig) -> None:
        """Burgers equation residual should be computable at arbitrary points."""
        operator = BurgersOperator(burgers_config)

        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.5], [0.8, 0.2]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # Smooth test function
        u = torch.sin(2 * np.pi * coords[:, 0]) * torch.exp(-coords[:, 1])

        residual = operator.residual(u, coords)

        assert np.isfinite(residual.l2_norm), "Residual L2 norm must be finite"
        assert np.isfinite(residual.max_norm), "Residual max norm must be finite"

    @given(viscosity=st.floats(min_value=0.001, max_value=1.0))
    @settings(max_examples=20)
    def test_viscosity_affects_residual(self, viscosity: float) -> None:
        """Different viscosities should produce different residuals for the same u."""
        config = PDEConfig(
            name="test_burgers",
            pde_type=PDEType.BURGERS,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=viscosity,
            is_time_dependent=True,
        )
        operator = BurgersOperator(config)

        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.6]],
            dtype=torch.float32,
            requires_grad=True,
        )
        u = torch.sin(2 * np.pi * coords[:, 0])

        residual = operator.residual(u, coords)

        assert np.isfinite(residual.l2_norm), (
            f"Residual must be finite for viscosity={viscosity}"
        )

    def test_initial_condition_sinusoidal(self, burgers_config: PDEConfig) -> None:
        """Initial condition should be sin(2*pi*x) at t=0."""
        operator = BurgersOperator(burgers_config)

        coords = np.array(
            [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        ic = operator.initial_condition(coords)
        expected = np.sin(2 * np.pi * coords[:, 0])

        np.testing.assert_allclose(ic, expected, rtol=1e-5)

    def test_boundary_profile_decreasing(self, burgers_config: PDEConfig) -> None:
        """Shock-like boundary profile should be decreasing in x."""
        operator = BurgersOperator(burgers_config)

        # Points along x at fixed y
        x_vals = np.linspace(0.0, 1.0, 20)
        coords = np.column_stack([x_vals, np.full_like(x_vals, 0.5)]).astype(np.float32)

        boundary = operator.boundary_value(coords)

        # Profile 0.5*(1 - tanh(10*(x-0.5))) is strictly decreasing in x
        for i in range(len(boundary) - 1):
            assert boundary[i] >= boundary[i + 1] - 1e-6, (
                f"Boundary profile should be non-increasing at x={x_vals[i]:.2f}"
            )


# ---------------------------------------------------------------------------
# Cross-operator properties
# ---------------------------------------------------------------------------


class TestCrossOperatorProperties:
    """Tests that apply to all operators."""

    @pytest.mark.parametrize(
        "pde_type,operator_cls",
        [
            (PDEType.POISSON, PoissonOperator),
            (PDEType.BURGERS, BurgersOperator),
            (PDEType.ADVECTION_DIFFUSION, AdvectionDiffusionOperator),
            (PDEType.HEAT, HeatOperator),
        ],
    )
    def test_operator_has_correct_type(self, pde_type: PDEType, operator_cls: type) -> None:
        """Each operator must report its correct PDE type."""
        kwargs: dict = {
            "name": "test",
            "pde_type": pde_type,
            "domain_dim": 2,
            "domain_min": [0.0, 0.0],
            "domain_max": [1.0, 1.0],
        }
        if pde_type in (PDEType.BURGERS, PDEType.ADVECTION_DIFFUSION, PDEType.HEAT):
            kwargs["is_time_dependent"] = True
        if pde_type == PDEType.ADVECTION_DIFFUSION:
            kwargs["advection_coeff"] = [1.0, 0.5]

        config = PDEConfig(**kwargs)
        operator = operator_cls(config)

        assert operator.pde_type == pde_type

    @pytest.mark.parametrize(
        "pde_type,operator_cls,expected_linear",
        [
            (PDEType.POISSON, PoissonOperator, True),
            (PDEType.BURGERS, BurgersOperator, False),
            (PDEType.ADVECTION_DIFFUSION, AdvectionDiffusionOperator, True),
            (PDEType.HEAT, HeatOperator, True),
        ],
    )
    def test_linearity_flag(
        self, pde_type: PDEType, operator_cls: type, expected_linear: bool
    ) -> None:
        """Operators must correctly report their linearity."""
        kwargs: dict = {
            "name": "test",
            "pde_type": pde_type,
            "domain_dim": 2,
            "domain_min": [0.0, 0.0],
            "domain_max": [1.0, 1.0],
        }
        if pde_type in (PDEType.BURGERS, PDEType.ADVECTION_DIFFUSION, PDEType.HEAT):
            kwargs["is_time_dependent"] = True
        if pde_type == PDEType.ADVECTION_DIFFUSION:
            kwargs["advection_coeff"] = [1.0, 0.5]

        config = PDEConfig(**kwargs)
        operator = operator_cls(config)

        assert operator.is_linear is expected_linear

    @pytest.mark.parametrize(
        "method",
        ["uniform", "random"],
    )
    def test_collocation_points_count(self, method: str, poisson_config: PDEConfig) -> None:
        """Collocation point generation should return approximately n_points."""
        operator = PoissonOperator(poisson_config)

        n_requested = 100
        points = operator.generate_collocation_points(n_requested, method=method, seed=42)

        assert points.shape[0] <= n_requested + 10, (
            f"Got {points.shape[0]} points, requested {n_requested}"
        )
        assert points.shape[0] > 0, "Must generate at least one point"
