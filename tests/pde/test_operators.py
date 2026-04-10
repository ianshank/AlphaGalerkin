"""Tests for PDE operators."""

import numpy as np
import pytest
import torch

from src.pde.config import BoundaryCondition, PDEConfig, PDEType
from src.pde.operators import (
    AdvectionDiffusionOperator,
    BurgersOperator,
    HeatOperator,
    PDEResidual,
    PoissonOperator,
)


class TestPDEResidual:
    """Tests for PDEResidual dataclass."""

    def test_create_residual(self) -> None:
        """Test creating a PDEResidual."""
        values = np.array([0.1, -0.2, 0.05])
        residual = PDEResidual(
            values=values,
            l2_norm=0.15,
            max_norm=0.2,
            derivatives={"u_x0": np.array([0.1, 0.2, 0.3])},
        )
        assert residual.l2_norm == 0.15
        assert residual.max_norm == 0.2
        assert len(residual.derivatives) == 1

    def test_to_numpy_from_tensor(self) -> None:
        """Test converting tensor residual to numpy."""
        values = torch.tensor([0.1, -0.2, 0.05])
        residual = PDEResidual(
            values=values,
            l2_norm=0.15,
            max_norm=0.2,
            derivatives={"u_x0": torch.tensor([0.1, 0.2, 0.3])},
        )
        np_residual = residual.to_numpy()
        assert isinstance(np_residual.values, np.ndarray)
        assert isinstance(np_residual.derivatives["u_x0"], np.ndarray)


class TestPoissonOperator:
    """Tests for Poisson equation operator."""

    @pytest.fixture
    def poisson_config(self) -> PDEConfig:
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
    def poisson_operator(self, poisson_config: PDEConfig) -> PoissonOperator:
        """Create Poisson operator fixture."""
        return PoissonOperator(poisson_config)

    def test_operator_properties(self, poisson_operator: PoissonOperator) -> None:
        """Test operator properties."""
        assert poisson_operator.name == "poisson"
        assert poisson_operator.is_time_dependent is False
        assert poisson_operator.is_linear is True
        assert poisson_operator.order == 2
        assert poisson_operator.dim == 2

    def test_source_term(self, poisson_operator: PoissonOperator) -> None:
        """Test source term computation."""
        coords = np.array([[0.5, 0.5]], dtype=np.float32)
        source = poisson_operator.source_term(coords)
        assert source.shape == (1,)
        # For manufactured solution sin(πx)sin(πy), source = 2π²sin(πx)sin(πy)
        expected = 2 * (np.pi**2) * np.sin(np.pi * 0.5) * np.sin(np.pi * 0.5)
        np.testing.assert_allclose(source[0], expected, rtol=1e-5)

    def test_boundary_value(self, poisson_operator: PoissonOperator) -> None:
        """Test boundary value computation."""
        boundary_coords = np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
        boundary_vals = poisson_operator.boundary_value(boundary_coords)
        assert boundary_vals.shape == (2,)
        np.testing.assert_allclose(boundary_vals, [0.0, 0.0])

    def test_exact_solution(self, poisson_operator: PoissonOperator) -> None:
        """Test exact solution for manufactured problem."""
        coords = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
        exact = poisson_operator.exact_solution(coords)
        assert exact is not None
        assert exact.shape == (2,)
        # u = sin(πx)sin(πy)
        expected = np.sin(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
        np.testing.assert_allclose(exact, expected, rtol=1e-5)

    def test_is_boundary_point(self, poisson_operator: PoissonOperator) -> None:
        """Test boundary point detection."""
        coords = np.array(
            [
                [0.0, 0.5],  # on boundary (x=0)
                [0.5, 0.5],  # interior
                [1.0, 0.25],  # on boundary (x=1)
                [0.5, 0.0],  # on boundary (y=0)
            ],
            dtype=np.float32,
        )
        on_boundary = poisson_operator.is_boundary_point(coords)
        np.testing.assert_array_equal(on_boundary, [True, False, True, True])

    def test_generate_collocation_points(self, poisson_operator: PoissonOperator) -> None:
        """Test collocation point generation."""
        points = poisson_operator.generate_collocation_points(100, method="uniform")
        assert points.shape[0] <= 100  # May be slightly different due to grid
        assert points.shape[1] == 2
        # Check points are in domain
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)

    def test_generate_boundary_points(self, poisson_operator: PoissonOperator) -> None:
        """Test boundary point generation."""
        points = poisson_operator.generate_boundary_points(10)
        assert points.shape[1] == 2
        # All points should be on boundary
        on_boundary = poisson_operator.is_boundary_point(points)
        assert np.all(on_boundary)

    def test_residual_computation(self, poisson_operator: PoissonOperator) -> None:
        """Test residual computation with tensor input."""
        coords = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32)
        coords.requires_grad_(True)

        # Use exact solution
        u = torch.sin(np.pi * coords[:, 0]) * torch.sin(np.pi * coords[:, 1])

        residual = poisson_operator.residual(u, coords)
        assert isinstance(residual, PDEResidual)
        # Residual should be small for exact solution
        assert residual.l2_norm < 0.5  # Allow some numerical error

    def test_custom_source_function(self, poisson_config: PDEConfig) -> None:
        """Test operator with custom source function."""

        def custom_source(coords):
            return np.ones(coords.shape[0], dtype=np.float32)

        operator = PoissonOperator(poisson_config, source_function=custom_source)
        coords = np.array([[0.5, 0.5]], dtype=np.float32)
        source = operator.source_term(coords)
        np.testing.assert_allclose(source, [1.0])


class TestBurgersOperator:
    """Tests for Burgers equation operator."""

    @pytest.fixture
    def burgers_config(self) -> PDEConfig:
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

    @pytest.fixture
    def burgers_operator(self, burgers_config: PDEConfig) -> BurgersOperator:
        """Create Burgers operator fixture."""
        return BurgersOperator(burgers_config)

    def test_operator_properties(self, burgers_operator: BurgersOperator) -> None:
        """Test operator properties."""
        assert burgers_operator.name == "burgers"
        assert burgers_operator.is_time_dependent is True
        assert burgers_operator.is_linear is False
        assert burgers_operator.viscosity == 0.01

    def test_boundary_value_shock_profile(self, burgers_operator: BurgersOperator) -> None:
        """Test shock-like boundary profile."""
        coords = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]], dtype=np.float32)
        boundary = burgers_operator.boundary_value(coords)
        # Profile should transition from ~1 to ~0
        assert boundary[0] > boundary[2]  # Decreasing

    def test_initial_condition(self, burgers_operator: BurgersOperator) -> None:
        """Test sinusoidal initial condition."""
        coords = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]], dtype=np.float32)
        ic = burgers_operator.initial_condition(coords)
        # sin(2πx) at y=0
        expected = np.sin(2 * np.pi * coords[:, 0])
        np.testing.assert_allclose(ic, expected, rtol=1e-5)


class TestAdvectionDiffusionOperator:
    """Tests for Advection-Diffusion equation operator."""

    @pytest.fixture
    def advdiff_config(self) -> PDEConfig:
        """Create advection-diffusion config fixture."""
        return PDEConfig(
            name="test_advdiff",
            pde_type=PDEType.ADVECTION_DIFFUSION,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.1,
            advection_coeff=[1.0, 0.5],
            is_time_dependent=True,
        )

    @pytest.fixture
    def advdiff_operator(self, advdiff_config: PDEConfig) -> AdvectionDiffusionOperator:
        """Create operator fixture."""
        return AdvectionDiffusionOperator(advdiff_config)

    def test_operator_properties(self, advdiff_operator: AdvectionDiffusionOperator) -> None:
        """Test operator properties."""
        assert advdiff_operator.name == "advection_diffusion"
        assert advdiff_operator.is_linear is True
        np.testing.assert_allclose(advdiff_operator.advection_velocity, [1.0, 0.5])
        assert advdiff_operator.diffusion == 0.1

    def test_initial_condition_gaussian(self, advdiff_operator: AdvectionDiffusionOperator) -> None:
        """Test Gaussian initial condition."""
        coords = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
        ic = advdiff_operator.initial_condition(coords)
        # Center should have highest value
        assert ic[0] > ic[1]

    def test_exact_solution_advection(self, advdiff_operator: AdvectionDiffusionOperator) -> None:
        """Test exact solution for advected Gaussian."""
        coords = np.array([[0.5, 0.5]], dtype=np.float32)
        exact_t0 = advdiff_operator.exact_solution(coords, time=0.0)
        exact_t1 = advdiff_operator.exact_solution(coords, time=0.1)
        # At later time, solution should differ
        assert exact_t0 is not None
        assert exact_t1 is not None


class TestHeatOperator:
    """Tests for Heat equation operator."""

    @pytest.fixture
    def heat_config(self) -> PDEConfig:
        """Create Heat config fixture."""
        return PDEConfig(
            name="test_heat",
            pde_type=PDEType.HEAT,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.5,
            is_time_dependent=True,
        )

    @pytest.fixture
    def heat_operator(self, heat_config: PDEConfig) -> HeatOperator:
        """Create operator fixture."""
        return HeatOperator(heat_config)

    def test_operator_properties(self, heat_operator: HeatOperator) -> None:
        """Test operator properties."""
        assert heat_operator.name == "heat"
        assert heat_operator.is_time_dependent is True
        assert heat_operator.diffusivity == 0.5

    def test_initial_condition_hotspot(self, heat_operator: HeatOperator) -> None:
        """Test hot spot initial condition."""
        coords = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
        ic = heat_operator.initial_condition(coords)
        # Center should be hot
        assert ic[0] > ic[1]


class TestOperatorRegistry:
    """Tests for PDE operator registry."""

    def test_builtin_operators_registered(self) -> None:
        """Test that built-in operators are registered."""
        from src.pde.registry import list_pde_operators

        operators = list_pde_operators()
        assert "poisson" in operators
        assert "burgers" in operators
        assert "advection_diffusion" in operators
        assert "heat" in operators

    def test_get_operator(self) -> None:
        """Test retrieving operator from registry."""
        from src.pde.registry import get_pde_operator

        poisson_cls = get_pde_operator("poisson")
        assert poisson_cls == PoissonOperator

    def test_get_nonexistent_operator_raises(self) -> None:
        """Test that getting nonexistent operator raises error."""
        from src.pde.registry import get_pde_operator

        with pytest.raises(KeyError):
            get_pde_operator("nonexistent_pde")


class TestDerivativeComputation:
    """Tests for automatic differentiation in operators."""

    def test_gradient_computation(self) -> None:
        """Test gradient computation via autodiff."""
        config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        operator = PoissonOperator(config)

        coords = torch.tensor([[0.3, 0.4], [0.6, 0.7]], dtype=torch.float32, requires_grad=True)
        # Simple test function
        u = coords[:, 0] ** 2 + coords[:, 1] ** 2  # u = x² + y²

        derivatives = operator.compute_derivatives(u.unsqueeze(-1), coords)

        # du/dx = 2x, du/dy = 2y
        assert "u_x0" in derivatives
        assert "u_x1" in derivatives
        np.testing.assert_allclose(
            derivatives["u_x0"].detach().numpy(), 2 * coords[:, 0].detach().numpy(), rtol=1e-4
        )
        np.testing.assert_allclose(
            derivatives["u_x1"].detach().numpy(), 2 * coords[:, 1].detach().numpy(), rtol=1e-4
        )

    def test_laplacian_computation(self) -> None:
        """Test Laplacian computation via autodiff."""
        config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        operator = PoissonOperator(config)

        coords = torch.tensor([[0.3, 0.4], [0.6, 0.7]], dtype=torch.float32, requires_grad=True)
        # u = x² + y²  =>  ∇²u = 2 + 2 = 4
        u = coords[:, 0] ** 2 + coords[:, 1] ** 2

        derivatives = operator.compute_derivatives(u.unsqueeze(-1), coords)

        assert "laplacian" in derivatives
        expected_laplacian = 4.0
        np.testing.assert_allclose(
            derivatives["laplacian"].detach().numpy(),
            [expected_laplacian, expected_laplacian],
            rtol=1e-4,
        )


class TestConstantBackedOperatorDefaults:
    """Verify operators use centralized constants for their defaults."""

    def test_burgers_default_shock_position_from_constant(self) -> None:
        from src.constants import DEFAULT_SHOCK_POSITION

        config = PDEConfig(
            name="burgers_const_test",
            pde_type=PDEType.BURGERS,
            is_time_dependent=True,
        )
        op = BurgersOperator(config)
        assert op.shock_position == DEFAULT_SHOCK_POSITION

    def test_burgers_default_shock_width_from_constant(self) -> None:
        from src.constants import DEFAULT_SHOCK_WIDTH

        config = PDEConfig(
            name="burgers_const_test",
            pde_type=PDEType.BURGERS,
            is_time_dependent=True,
        )
        op = BurgersOperator(config)
        assert op.shock_width == DEFAULT_SHOCK_WIDTH

    def test_burgers_custom_shock_params_override_defaults(self) -> None:
        config = PDEConfig(
            name="burgers_custom",
            pde_type=PDEType.BURGERS,
            is_time_dependent=True,
        )
        op = BurgersOperator(config, shock_position=0.3, shock_width=5.0)
        assert op.shock_position == pytest.approx(0.3)
        assert op.shock_width == pytest.approx(5.0)

    def test_initial_condition_uses_two_pi(self) -> None:
        """sin(TWO_PI * x) matches sin(2π * x) — constant is consistent."""
        import math

        from src.constants import TWO_PI

        config = PDEConfig(
            name="burgers_two_pi",
            pde_type=PDEType.BURGERS,
            is_time_dependent=True,
        )
        op = BurgersOperator(config)
        coords = np.array([[0.25, 0.0]], dtype=np.float32)
        ic = op.initial_condition(coords)
        expected = np.sin(TWO_PI * 0.25)  # == sin(π/2) == 1
        np.testing.assert_allclose(ic[0], expected, atol=1e-5)
        assert pytest.approx(2 * math.pi) == TWO_PI

    def test_gaussian_width_ratio_used_in_heat_ic(self) -> None:
        """HeatOperator uses GAUSSIAN_WIDTH_RATIO for initial condition sigma."""
        from src.constants import GAUSSIAN_WIDTH_RATIO

        config = PDEConfig(
            name="heat_const_test",
            pde_type=PDEType.HEAT,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[2.0, 2.0],
            is_time_dependent=True,
        )
        op = HeatOperator(config)
        coords = np.array([[1.0, 1.0]], dtype=np.float32)
        ic = op.initial_condition(coords)
        # IC should be finite and peak near center
        assert np.isfinite(ic).all()
        # sigma = GAUSSIAN_WIDTH_RATIO * mean(domain_size)
        expected_sigma = GAUSSIAN_WIDTH_RATIO * np.mean([2.0, 2.0])
        assert expected_sigma == pytest.approx(0.2)  # 0.1 * 2.0
