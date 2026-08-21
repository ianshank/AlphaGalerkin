"""Tests for PDE operators."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.special import ive

from src.pde.config import BoundaryCondition, PDEConfig, PDEType
from src.pde.operators import (
    COLE_HOPF_CLAMP_EPS,
    COLE_HOPF_MAX_TERMS,
    COLE_HOPF_MIN_RESOLVED_VISCOSITY,
    COLE_HOPF_N_TERMS,
    AdvectionDiffusionOperator,
    BurgersOperator,
    HeatOperator,
    LShapedPoissonOperator,
    NavierStokesOperator,
    PDEResidual,
    PoissonOperator,
    _cole_hopf_coefficients,
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

    def test_boundary_value_homogeneous_dirichlet(
        self,
        burgers_operator: BurgersOperator,
    ) -> None:
        """Test homogeneous Dirichlet boundary values u(0, t) = u(1, t) = 0.

        Previously asserted a decreasing tanh shock profile (u(0, t) = 1), which
        contradicted both the initial condition and the Cole-Hopf exact solution
        this operator implements.
        """
        coords = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]], dtype=np.float32)
        boundary = burgers_operator.boundary_value(coords)
        np.testing.assert_array_equal(boundary, np.zeros(3, dtype=np.float32))

    def test_initial_condition(self, burgers_operator: BurgersOperator) -> None:
        """Test the -sin(pi*x) Cole-Hopf benchmark initial condition.

        Previously asserted sin(2*pi*x), which is not the profile that
        ``exact_solution`` propagates and is incompatible with homogeneous
        Dirichlet data at x = 0.5.
        """
        coords = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]], dtype=np.float32)
        ic = burgers_operator.initial_condition(coords)
        # -sin(pi*x) at y=0
        expected = -np.sin(np.pi * coords[:, 0])
        np.testing.assert_allclose(ic, expected, atol=1e-6)


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


# ---------------------------------------------------------------------------
# Coverage sprint additions (Section 2.1 of docs/PLAN_2026-04-27.md).
# Targets the largest gaps in src/pde/operators.py identified by the audit:
# BurgersOperator.exact_solution Cole-Hopf branches (698-732),
# AdvectionDiffusionOperator.residual autodiff path (805-826),
# NavierStokesOperator.residual full body (1059-1133), and
# LShapedPoissonOperator residual + compute_error + boundary sampling
# (1410-1424, 1500-1537).
# ---------------------------------------------------------------------------


class TestBurgersOperatorColeHopf:
    """Cover BurgersOperator.exact_solution Cole-Hopf branches (lines 698-732)."""

    @pytest.fixture
    def time_dep_burgers(self) -> BurgersOperator:
        config = PDEConfig(
            name="burgers_td",
            pde_type=PDEType.BURGERS,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.01,
            is_time_dependent=True,
        )
        return BurgersOperator(config)

    def test_steady_returns_none(self) -> None:
        """Non-time-dependent Burgers has no closed-form Cole-Hopf solution."""
        config = PDEConfig(
            name="burgers_steady",
            pde_type=PDEType.BURGERS,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.01,
            is_time_dependent=False,
        )
        op = BurgersOperator(config)
        coords = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        assert op.exact_solution(coords, time=0.5) is None

    def test_default_config_keeps_class_default_time_dependent(self) -> None:
        """P0-1 regression: an unset ``is_time_dependent`` must keep the class default.

        Before the fix, ``BurgersOperator.__init__`` unconditionally assigned
        ``self.is_time_dependent = config.is_time_dependent``, and
        ``PDEConfig.is_time_dependent`` defaults to ``False`` -- so every caller
        that built a Burgers config without mentioning the flag (e.g.
        ``_centaur_common.build_pde_operator``, ``PDETrainer``) silently got a
        "steady" Burgers operator whose ``exact_solution()`` always returned
        ``None``, making the Burgers OOD reward structurally flat/zero
        (docs/CODE_HYGIENE_AUDIT.md P0-1). This is the previously-broken case.
        """
        config = PDEConfig(
            name="burgers_unset_time_flag",
            pde_type=PDEType.BURGERS,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.01,
            # is_time_dependent deliberately omitted: this is exactly the
            # construction pattern _centaur_common.build_pde_operator uses.
        )
        assert "is_time_dependent" not in config.model_fields_set

        op = BurgersOperator(config)

        # The class-level default (True) must survive, not config's False default.
        assert op.is_time_dependent is True

        coords = torch.tensor([[0.3, 0.0], [0.5, 0.0], [0.7, 0.0]], dtype=torch.float32)
        u = op.exact_solution(coords)
        assert u is not None
        assert isinstance(u, torch.Tensor)
        assert torch.isfinite(u).all()

    def test_cole_hopf_t0_magnitude_is_physically_sane(self) -> None:
        """FIXED DEFECT (was: magnitude ~1e10-1e13 at t=0). Inverted assertion.

        Fixing the P0-1 sentinel above (unset ``is_time_dependent`` keeping the
        class default) makes ``exact_solution()`` reachable at the implicit
        ``t=0.0`` default -- exactly the call ``BasisSelectionGame.__init__``
        makes, via ``_centaur_common.build_pde_operator``, for the *default* OOD
        arm of the shipped ``llm_prior_ablation`` scenario
        (``src/poc/scenarios/llm_prior_config.py``'s default ``ood_pde="burgers"``,
        ``config/scenarios/llm_prior_demo.yaml``).

        Historically every Fourier coefficient of the Cole-Hopf series was
        hardcoded to 1 -- the transform of a Dirac comb, not of a sinusoid -- so
        the truncated denominator ``phi`` went negative over roughly half the
        domain, the ``clamp(min=1e-10)`` guard floored it, and ``|u|`` reached
        ~1e13 at ``nu=1.0`` / ~5e11 at ``nu=0.01``: large but finite, so an
        ``isfinite``-only assertion passed on a meaningless value. This test used
        to *pin* that magnitude (``u.abs().max() > 1.0e6``) and its docstring
        authorised inverting it once the number became sane.

        The coefficients are now the correct ``2*(-1)^n*ive(n, R)`` and ``u`` is
        the true ``-sin(pi*x)`` profile, bounded by the maximum principle. See
        ``TestBurgersColeHopf`` below for the full correctness surface.
        """
        config = PDEConfig(
            name="burgers_unset_time_flag",
            pde_type=PDEType.BURGERS,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.01,
        )
        op = BurgersOperator(config)
        coords = torch.tensor([[0.3, 0.0], [0.5, 0.0], [0.7, 0.0]], dtype=torch.float32)

        u = op.exact_solution(coords)  # implicit t=0.0

        assert u is not None
        assert torch.isfinite(u).all()
        # The viscous Burgers maximum principle bounds |u(.,t)| by max|u(.,0)| = 1
        # for the -sin(pi*x) initial condition, for every t >= 0.
        assert u.abs().max() <= 1.0 + 1.0e-3, (
            f"Cole-Hopf t=0 magnitude {float(u.abs().max())} violates the maximum "
            "principle bound |u| <= max|u(.,0)| = 1 -- do not loosen this bound."
        )
        # This fixture uses nu=0.01, i.e. exactly COLE_HOPF_MIN_RESOLVED_VISCOSITY,
        # where the float64 series has ~3 significant digits left near x=0
        # (measured max error 1e-3 over the full grid, 5e-6 at these points).
        # TestBurgersColeHopf asserts the 1e-6 profile at well-conditioned nu.
        torch.testing.assert_close(
            u,
            -torch.sin(torch.pi * coords[:, 0]),
            atol=1.0e-4,
            rtol=1.0e-4,
        )

    def test_explicit_true_still_time_dependent(self) -> None:
        """An explicit ``is_time_dependent=True`` is honoured (unchanged by the fix)."""
        config = PDEConfig(
            name="burgers_explicit_true",
            pde_type=PDEType.BURGERS,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.01,
            is_time_dependent=True,
        )
        assert "is_time_dependent" in config.model_fields_set

        op = BurgersOperator(config)
        assert op.is_time_dependent is True

        coords = torch.tensor([[0.3, 0.0]], dtype=torch.float32)
        assert op.exact_solution(coords) is not None

    def test_tensor_path_returns_finite_tensor(self, time_dep_burgers: BurgersOperator) -> None:
        coords = torch.linspace(0.05, 0.95, 11).unsqueeze(-1)
        coords = torch.cat([coords, torch.zeros_like(coords)], dim=-1)
        u = time_dep_burgers.exact_solution(coords, time=0.1)
        assert isinstance(u, torch.Tensor)
        assert u.shape == (11,)
        assert torch.isfinite(u).all()

    def test_numpy_path_returns_finite_array(self, time_dep_burgers: BurgersOperator) -> None:
        coords = np.column_stack(
            [np.linspace(0.05, 0.95, 11), np.zeros(11)],
        ).astype(np.float32)
        u = time_dep_burgers.exact_solution(coords, time=0.1)
        assert isinstance(u, np.ndarray)
        assert u.dtype == np.float32
        assert u.shape == (11,)
        assert np.isfinite(u).all()

    def test_default_time_is_zero(self, time_dep_burgers: BurgersOperator) -> None:
        """Both paths default time=None to t=0.0."""
        coords_t = torch.tensor([[0.3, 0.0]], dtype=torch.float32)
        u_t = time_dep_burgers.exact_solution(coords_t, time=None)
        u_t0 = time_dep_burgers.exact_solution(coords_t, time=0.0)
        assert isinstance(u_t, torch.Tensor)
        assert isinstance(u_t0, torch.Tensor)
        assert torch.allclose(u_t, u_t0)


# ---------------------------------------------------------------------------
# Cole-Hopf correctness surface.
#
# The historical implementation hardcoded every Fourier coefficient to 1, which
# is the Cole-Hopf image of a Dirac comb rather than of a sinusoid, and made
# BurgersOperator describe three mutually inconsistent problems at once
# (initial_condition = sin(2*pi*x), boundary_value = a tanh shock,
# exact_solution's docstring = -sin(pi*x)). All three now describe the standard
# Basdevant / Cole-Hopf benchmark: u(x,0) = -sin(pi*x) on [0,1] with
# homogeneous Dirichlet data.
# ---------------------------------------------------------------------------

# Viscosities at/above COLE_HOPF_MIN_RESOLVED_VISCOSITY, where the float64
# series resolves u across the whole domain (1.0 is the shipped default from a
# bare PDEConfig; 0.01 is the value used by this file's other fixtures).
WELL_CONDITIONED_VISCOSITIES = [1.0, 0.5, 0.1, 0.05]
# The full acceptance sweep, including the deliberately under-resolved 0.001
# (R = 1/(2*pi*nu) = 159, where phi's dynamic range exp(-2R) = 5.8e-139 is far
# below float64 resolution near x = 0).
ACCEPTANCE_VISCOSITIES = [1.0, 0.01, 0.001]
TIME_SWEEP = [0.0, 0.05, 0.2, 1.0]


def _burgers(viscosity: float, dim: int = 2) -> BurgersOperator:
    """Time-dependent Burgers operator on the unit box at a given viscosity."""
    config = PDEConfig(
        name="burgers_cole_hopf",
        pde_type=PDEType.BURGERS,
        domain_dim=dim,
        domain_min=[0.0] * dim,
        domain_max=[1.0] * dim,
        advection_coeff=[0.0] * dim,
        diffusion_coeff=viscosity,
        is_time_dependent=True,
    )
    return BurgersOperator(config)


def _grid(n_points: int = 201, dim: int = 2) -> NDArray[np.float32]:
    """Uniform x-grid over [0, 1] padded with zeros in the trailing dimensions."""
    x = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    columns = [x] + [np.zeros_like(x) for _ in range(dim - 1)]
    return np.column_stack(columns).astype(np.float32)


class TestBurgersColeHopfCoefficients:
    """The Fourier-Bessel coefficients themselves (the site of the defect)."""

    @pytest.mark.parametrize("viscosity", [1.0, 0.1, 0.01])
    def test_coefficients_are_bessel_not_unity(self, viscosity: float) -> None:
        """a_n must be 2*(-1)^n*ive(n, R), not the historical constant 1."""
        c0, n, coefficients = _cole_hopf_coefficients(viscosity)
        r = 1.0 / (2.0 * np.pi * viscosity)

        assert c0 == pytest.approx(float(ive(0.0, r)))
        np.testing.assert_allclose(coefficients, 2.0 * ((-1.0) ** n) * ive(n, r), rtol=1e-12)
        # The defect: every coefficient equal to 1 (and no alternating sign).
        assert not np.allclose(np.abs(coefficients), 1.0)
        # The alternating sign is what selects -sin(pi*x) over +sin(pi*x).
        assert np.all(coefficients[::2] < 0.0), "odd n must carry a negative sign"
        assert np.all(coefficients[1::2] > 0.0), "even n must carry a positive sign"

    @pytest.mark.parametrize("viscosity", [1.0, 0.1, 0.01, 0.001])
    def test_scaled_coefficients_have_unit_l1_norm(self, viscosity: float) -> None:
        """``e^-R*(I_0(R) + 2*sum I_n(R)) == 1`` -- the property COLE_HOPF_CLAMP_EPS rests on.

        It is what makes an *absolute* denominator floor simultaneously a
        *relative* one, and it bounds ``phi`` above by 1.
        """
        c0, _, coefficients = _cole_hopf_coefficients(viscosity)
        assert c0 + float(np.abs(coefficients).sum()) == pytest.approx(1.0, abs=1e-12)

    def test_term_count_adapts_to_viscosity_and_is_capped(self) -> None:
        """Fewer terms suffice at large nu; the count grows as nu shrinks, bounded."""
        counts = [len(_cole_hopf_coefficients(nu)[1]) for nu in (1.0, 0.01, 0.001, 1e-5)]
        assert counts[0] == COLE_HOPF_N_TERMS  # floor honoured
        assert counts == sorted(counts), f"term count must be monotone in 1/nu, got {counts}"
        assert counts[-1] > COLE_HOPF_N_TERMS
        assert len(_cole_hopf_coefficients(1e-9)[1]) == COLE_HOPF_MAX_TERMS

    def test_coefficients_are_cached_per_viscosity(self) -> None:
        """Coefficients are time-independent, so they are computed once per nu."""
        first = _cole_hopf_coefficients(0.25)
        second = _cole_hopf_coefficients(0.25)
        assert first[1] is second[1]
        assert first[2] is second[2]

    @pytest.mark.parametrize("viscosity", [0.0, -1.0])
    def test_non_positive_viscosity_raises(self, viscosity: float) -> None:
        """R = 1/(2*pi*nu) is undefined for nu <= 0."""
        with pytest.raises(ValueError, match="strictly positive viscosity"):
            _cole_hopf_coefficients(viscosity)


class TestBurgersColeHopf:
    """Acceptance surface for the corrected Cole-Hopf exact solution."""

    @pytest.mark.parametrize("viscosity", WELL_CONDITIONED_VISCOSITIES)
    def test_exact_solution_at_t0_matches_initial_condition(self, viscosity: float) -> None:
        """LOAD-BEARING: exact_solution(t=0) == initial_condition, both branches.

        This is the self-consistency property the operator previously failed in
        three different ways at once. Mirrors the precedent at
        ``tests/pde/test_taylor_green_invariants.py::test_initial_condition_matches_exact_at_t0``.
        """
        operator = _burgers(viscosity)
        coords_np = _grid()
        coords_torch = torch.from_numpy(coords_np)

        np.testing.assert_allclose(
            np.asarray(operator.exact_solution(coords_np, time=0.0)),
            np.asarray(operator.initial_condition(coords_np)),
            atol=1e-6,
        )
        torch.testing.assert_close(
            operator.exact_solution(coords_torch, time=0.0),
            operator.initial_condition(coords_torch),
            atol=1e-6,
            rtol=1e-6,
        )
        # ...and both equal the analytic profile, so neither can drift together.
        np.testing.assert_allclose(
            np.asarray(operator.exact_solution(coords_np, time=0.0)),
            -np.sin(np.pi * coords_np[:, 0].astype(np.float64)),
            atol=1e-6,
        )

    @pytest.mark.parametrize("viscosity", ACCEPTANCE_VISCOSITIES)
    @pytest.mark.parametrize("time", TIME_SWEEP)
    def test_denominator_is_strictly_positive_across_viscosities(
        self,
        viscosity: float,
        time: float,
    ) -> None:
        """The divisor must be strictly positive at every t, including t=0.

        The historical denominator was negative over roughly half the domain at
        t=0 for *every* viscosity, which the clamp converted into a finite but
        meaningless ~1e13 solution. Two distinct claims are checked:

        1. the value actually divided by is strictly positive (always, by the
           :data:`COLE_HOPF_CLAMP_EPS` floor); and
        2. at or above :data:`COLE_HOPF_MIN_RESOLVED_VISCOSITY` the *raw* series
           is already strictly positive, i.e. the floor is never load-bearing
           in the regime where the representation is well conditioned.
        """
        operator = _burgers(viscosity)
        coords = _grid(401)

        phi, _ = operator.cole_hopf_potential(coords, time=time)
        divisor = np.maximum(phi, COLE_HOPF_CLAMP_EPS)

        assert np.all(divisor > 0.0)
        assert np.all(np.isfinite(np.asarray(operator.exact_solution(coords, time=time))))

        if viscosity >= COLE_HOPF_MIN_RESOLVED_VISCOSITY:
            assert phi.min() > 0.0, (
                f"raw Cole-Hopf denominator went non-positive ({phi.min():.3e}) at "
                f"nu={viscosity}, t={time}, where the series is well conditioned"
            )
            np.testing.assert_array_equal(divisor, phi)

    @pytest.mark.parametrize("viscosity", ACCEPTANCE_VISCOSITIES)
    @pytest.mark.parametrize("time", TIME_SWEEP)
    def test_float32_output_is_finite_and_order_one(
        self,
        viscosity: float,
        time: float,
    ) -> None:
        """Finite is not enough: the old failure mode was finite-but-meaningless.

        The viscous Burgers maximum principle gives ``max|u(.,t)| <= max|u(.,0)|``
        = 1 for the ``-sin(pi*x)`` datum, so a genuine solution is O(1). The old
        implementation returned finite float32 values of magnitude ~1e10-1e13,
        which no ``isfinite`` check catches.
        """
        operator = _burgers(viscosity)
        coords = _grid(401)

        u_np = np.asarray(operator.exact_solution(coords, time=time))
        u_torch = operator.exact_solution(torch.from_numpy(coords), time=time)

        assert u_np.dtype == np.float32
        assert np.isfinite(u_np).all()
        assert torch.isfinite(u_torch).all()
        # 1e-3 slack absorbs float64 round-off in the under-resolved regime.
        assert np.abs(u_np).max() <= 1.0 + 1e-3
        assert float(u_torch.abs().max()) <= 1.0 + 1e-3

    @pytest.mark.parametrize("viscosity", WELL_CONDITIONED_VISCOSITIES)
    @pytest.mark.parametrize("time", TIME_SWEEP)
    def test_numpy_and_torch_branches_agree(self, viscosity: float, time: float) -> None:
        """Numpy and torch implementations must produce identical fields.

        Mirrors ``tests/pde/test_taylor_green_invariants.py::
        test_numpy_and_torch_branches_agree`` (atol=rtol=1e-6).
        """
        operator = _burgers(viscosity)
        coords_np = _grid(97)

        out_np = np.asarray(operator.exact_solution(coords_np, time=time))
        out_torch = (
            operator.exact_solution(torch.from_numpy(coords_np), time=time).detach().cpu().numpy()
        )

        np.testing.assert_allclose(out_np, out_torch, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("viscosity", [1.0, 0.1, 0.01])
    def test_solution_decays_in_time(self, viscosity: float) -> None:
        """Viscous Burgers dissipates: the L2 norm decreases monotonically."""
        operator = _burgers(viscosity)
        coords = _grid(401)

        norms = [
            float(np.sqrt(np.mean(np.asarray(operator.exact_solution(coords, time=t)) ** 2)))
            for t in (0.0, 0.25, 0.5, 1.0, 2.0)
        ]

        assert norms == sorted(norms, reverse=True), f"L2 norm must decay, got {norms}"
        assert norms[0] == pytest.approx(np.sqrt(0.5), abs=1e-2)  # RMS of sin(pi*x)
        assert norms[-1] < norms[0]

    def test_under_resolved_viscosity_stays_bounded(self) -> None:
        """Below COLE_HOPF_MIN_RESOLVED_VISCOSITY: degraded, but never unbounded.

        Honest documentation of an intrinsic limit of the Fourier-Bessel
        representation (not of this implementation): ``phi`` spans
        ``[exp(-2R), 1]``, so once ``exp(-2R)`` falls under float64 round-off the
        series loses all significance near ``x = 0``. Accuracy survives in the
        far field, and the magnitude stays physical everywhere.
        """
        operator = _burgers(0.001)
        coords = _grid(401)
        x = coords[:, 0].astype(np.float64)

        u = np.asarray(operator.exact_solution(coords, time=0.0), dtype=np.float64)
        error = np.abs(u + np.sin(np.pi * x))

        assert np.abs(u).max() <= 1.0 + 1e-3, "must stay bounded even when under-resolved"
        assert error[x >= 0.9].max() < 1e-3, "far field must remain accurate"
        assert error.max() > 1e-2, (
            "nu=0.001 is expected to be under-resolved near x=0; if this now "
            "passes accurately, raise the documented COLE_HOPF_MIN_RESOLVED_VISCOSITY"
        )

    @pytest.mark.parametrize(
        "viscosity,time,n_points",
        [(1.0, 0.05, 101), (0.1, 0.1, 101)],
    )
    def test_matches_independent_finite_difference_march(
        self,
        viscosity: float,
        time: float,
        n_points: int,
    ) -> None:
        """Independent verification of the *time evolution*, not just t=0.

        Marches the nonlinear PDE ``u_t = -u*u_x + nu*u_xx`` forward from
        ``u(x,0) = -sin(pi*x)`` with homogeneous Dirichlet data using an explicit
        central-difference scheme -- a method that shares no code, no
        coefficients and no Bessel functions with the Cole-Hopf series -- and
        compares the result against ``exact_solution(t)``.

        This is what pins the whole derivation end to end: the alternating
        ``(-1)^n`` sign, the ``exp(-n^2*pi^2*nu*t)`` modal decay, and consistency
        with the *nonlinear* advection term. A t=0 identity alone could not
        detect a wrong decay rate. The tolerance is set by the scheme's own
        O(dx^2) truncation error (measured 1.3e-5 and 4.9e-5 for these two
        cases; a finer march at nx=401 converges to 8.8e-7 at nu=1.0).
        """
        x = np.linspace(0.0, 1.0, n_points)
        dx = float(x[1] - x[0])
        u = -np.sin(np.pi * x)
        # Explicit stability: diffusive dx^2/(2*nu) and advective dx/max|u|.
        dt = 0.2 * min(dx * dx / (2.0 * viscosity), dx / max(float(np.abs(u).max()), 1e-12))
        n_steps = int(np.ceil(time / dt))
        dt = time / n_steps

        for _ in range(n_steps):
            u_x = np.zeros_like(u)
            u_xx = np.zeros_like(u)
            u_x[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
            u_xx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
            u = u + dt * (-u * u_x + viscosity * u_xx)
            u[0] = 0.0
            u[-1] = 0.0

        operator = _burgers(viscosity, dim=1)
        exact = np.asarray(
            operator.exact_solution(x.reshape(-1, 1).astype(np.float32), time=time),
            dtype=np.float64,
        )

        np.testing.assert_allclose(u, exact, atol=1e-3)
        # The profile must actually have evolved, or this would be a t=0 retest.
        assert np.abs(exact).max() < 0.999

    def test_gradients_flow_through_coords(self) -> None:
        """float64 assembly must not sever autograd on the torch branch."""
        coords = torch.tensor(
            [[0.3, 0.0], [0.5, 0.0], [0.7, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        operator = _burgers(1.0)

        u = operator.exact_solution(coords)
        assert isinstance(u, torch.Tensor)
        u.sum().backward()

        assert coords.grad is not None
        # d/dx of -sin(pi*x) is -pi*cos(pi*x).
        expected = -np.pi * np.cos(np.pi * np.array([0.3, 0.5, 0.7]))
        np.testing.assert_allclose(coords.grad[:, 0].numpy(), expected, atol=1e-4)

    def test_initial_condition_is_negative_sine(self) -> None:
        """u(x, 0) = -sin(pi*x), matching exact_solution (was sin(2*pi*x))."""
        operator = _burgers(1.0, dim=1)
        coords = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)

        expected = -np.sin(np.pi * coords[:, 0].astype(np.float64))
        np.testing.assert_allclose(
            np.asarray(operator.initial_condition(coords)),
            expected,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            operator.initial_condition(torch.from_numpy(coords)).numpy(),
            expected,
            atol=1e-6,
        )

    def test_boundary_value_is_homogeneous_dirichlet(self) -> None:
        """u(0, t) = u(1, t) = 0 (was a tanh shock giving u(0, t) = 1)."""
        operator = _burgers(1.0, dim=1)
        coords = np.array([[0.0], [1.0]], dtype=np.float32)

        np.testing.assert_array_equal(
            np.asarray(operator.boundary_value(coords)),
            np.zeros(2, dtype=np.float32),
        )
        torch_out = operator.boundary_value(torch.from_numpy(coords))
        assert isinstance(torch_out, torch.Tensor)
        assert torch.equal(torch_out, torch.zeros(2))
        # ...and it agrees with the exact solution's own boundary trace.
        for time in TIME_SWEEP:
            np.testing.assert_allclose(
                np.asarray(operator.exact_solution(coords, time=time)),
                np.zeros(2),
                atol=1e-6,
            )


class TestAdvectionDiffusionOperatorResidual:
    """Cover AdvectionDiffusionOperator.residual autodiff body (lines 805-831)."""

    @pytest.fixture
    def operator(self) -> AdvectionDiffusionOperator:
        config = PDEConfig(
            name="advdiff_residual",
            pde_type=PDEType.ADVECTION_DIFFUSION,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.1,
            advection_coeff=[1.0, 0.5],
            is_time_dependent=False,
        )
        return AdvectionDiffusionOperator(config)

    def test_residual_finite_on_smooth_u(self, operator: AdvectionDiffusionOperator) -> None:
        coords = torch.tensor(
            [[0.3, 0.4], [0.6, 0.7], [0.5, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        u = (coords[:, 0] + coords[:, 1]).unsqueeze(-1)
        residual = operator.residual(u, coords)
        assert isinstance(residual, PDEResidual)
        assert residual.values.shape == (3,)
        assert torch.isfinite(residual.values).all()
        assert np.isfinite(residual.l2_norm)
        assert np.isfinite(residual.max_norm)

    def test_residual_drops_derivatives_on_request(
        self, operator: AdvectionDiffusionOperator
    ) -> None:
        coords = torch.tensor([[0.3, 0.4]], dtype=torch.float32, requires_grad=True)
        u = (coords[:, 0] ** 2).unsqueeze(-1)
        residual = operator.residual(u, coords, compute_derivatives=False)
        assert residual.derivatives == {}


class TestNavierStokesOperator:
    """Construction + residual coverage for NavierStokesOperator (lines 1003-1138)."""

    @pytest.fixture
    def ns_config(self) -> PDEConfig:
        return PDEConfig(
            name="ns_test",
            pde_type=PDEType.NAVIER_STOKES,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.1,
            is_time_dependent=True,
        )

    def test_viscosity_from_diffusion_coeff(self, ns_config: PDEConfig) -> None:
        op = NavierStokesOperator(ns_config)
        assert op.viscosity == pytest.approx(0.1)
        assert op.reynolds_number == pytest.approx(10.0)

    def test_viscosity_from_reynolds_number(self, ns_config: PDEConfig) -> None:
        op = NavierStokesOperator(ns_config, reynolds_number=100.0)
        assert op.reynolds_number == pytest.approx(100.0)
        assert op.viscosity == pytest.approx(0.01)

    def test_residual_taylor_green_continuity_is_zero(self, ns_config: PDEConfig) -> None:
        """Taylor-Green-shaped u is divergence-free analytically.

        For ``u_x = sin(x) cos(y)``, ``u_y = -cos(x) sin(y)``:
            div u = du_x/dx + du_y/dy
                  = cos(x) cos(y) + (-cos(x) cos(y))
                  = 0  exactly.

        This is a real analytical property that the autodiff residual
        should reproduce to machine precision. Strengthening the test
        from "didn't crash + finite values" to this exact assertion
        per PR #70 review feedback.
        """
        op = NavierStokesOperator(ns_config)
        # coords must be the grad-enabled tensor that u is computed from,
        # so torch.autograd.grad inside residual() can trace back through u.
        # The u must also depend non-linearly on coords so the SECOND-order
        # derivative inside residual (laplacian via d/dx of d ux/dx) has a
        # non-trivial grad_fn — a purely linear u produces a constant
        # first derivative and the second-order autograd.grad errors with
        # "element 0 of tensors does not require grad".
        coords = torch.tensor(
            [[0.3, 0.4], [0.6, 0.7], [0.5, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # Taylor-Green-shaped velocity (analytically divergence-free).
        u = torch.stack(
            [
                torch.sin(coords[:, 0]) * torch.cos(coords[:, 1]),
                -torch.cos(coords[:, 0]) * torch.sin(coords[:, 1]),
            ],
            dim=-1,
        )
        residual = op.residual(u, coords)
        assert isinstance(residual, PDEResidual)
        assert residual.values.shape == (3,)
        assert torch.isfinite(residual.values).all()
        # derivatives populated when compute_derivatives=True (default).
        for key in ("ux_x", "uy_y", "continuity", "momentum_x"):
            assert key in residual.derivatives, f"missing derivative key {key}"
        # Strong assertion: the analytically divergence-free TG vortex
        # must produce |continuity| < float32 noise (~1e-6) at every
        # collocation point. torch.testing.assert_close uses the
        # idiomatic close-comparison API recommended by code review.
        torch.testing.assert_close(
            residual.derivatives["continuity"],
            torch.zeros_like(residual.derivatives["continuity"]),
            atol=1e-5,
            rtol=0.0,
            msg="TG vortex must be divergence-free (du_x/dx + du_y/dy = 0)",
        )
        # Cross-check the individual partials against their closed-form:
        # du_x/dx = cos(x) cos(y), du_y/dy = -cos(x) cos(y).
        x, y = coords[:, 0], coords[:, 1]
        expected_dux_dx = torch.cos(x) * torch.cos(y)
        expected_duy_dy = -torch.cos(x) * torch.cos(y)
        torch.testing.assert_close(
            residual.derivatives["ux_x"],
            expected_dux_dx,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            residual.derivatives["uy_y"],
            expected_duy_dy,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_residual_drops_derivatives_when_disabled(self, ns_config: PDEConfig) -> None:
        op = NavierStokesOperator(ns_config)
        # Non-linear u in coords so the second-order autograd has a grad_fn.
        # A purely linear u would produce constant first derivatives and the
        # second-order grad call inside residual() would error.
        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        u = torch.stack(
            [torch.sin(coords[:, 0]) * coords[:, 1], -(coords[:, 0] ** 2)],
            dim=-1,
        )
        residual = op.residual(u, coords, compute_derivatives=False)
        assert residual.derivatives == {}


class TestLShapedPoissonOperatorCoverage:
    """Cover LShapedPoissonOperator gaps.

    Targets residual, compute_error, generate_boundary_points
    (lines 1410-1424, 1500-1537 in src/pde/operators.py).
    """

    @pytest.fixture
    def operator(self) -> LShapedPoissonOperator:
        from src.pde.geometry import GeometryConfig, GeometryType

        config = PDEConfig(
            name="lshaped_test",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-1.0, -1.0],
            domain_max=[1.0, 1.0],
            geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED),
        )
        return LShapedPoissonOperator(config)

    def test_residual_finite_on_smooth_u(self, operator: LShapedPoissonOperator) -> None:
        coords = torch.tensor(
            [[0.3, 0.4], [-0.2, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        u = (coords[:, 0] ** 2 + coords[:, 1] ** 2).unsqueeze(-1)
        residual = operator.residual(u, coords)
        assert isinstance(residual, PDEResidual)
        assert residual.values.shape == (2,)
        assert torch.isfinite(residual.values).all()

    def test_residual_drops_derivatives_when_disabled(
        self, operator: LShapedPoissonOperator
    ) -> None:
        coords = torch.tensor([[0.3, 0.4]], dtype=torch.float32, requires_grad=True)
        u = (coords[:, 0] ** 2).unsqueeze(-1)
        residual = operator.residual(u, coords, compute_derivatives=False)
        assert residual.derivatives == {}

    def test_compute_error_returns_finite_metrics(self, operator: LShapedPoissonOperator) -> None:
        coords = torch.tensor(
            [[0.3, 0.4], [-0.2, 0.5]],
            dtype=torch.float32,
        )
        # Use the exact solution itself as a prediction -> error must be ~0.
        u_pred = operator.exact_solution(coords)
        assert isinstance(u_pred, torch.Tensor)
        result = operator.compute_error(u_pred, coords)
        assert set(result.keys()) == {"l2_error", "linf_error", "mse"}
        for value in result.values():
            assert np.isfinite(value)
            # Predicting the exact solution should yield ~zero error.
            assert value < 1e-5

    def test_generate_boundary_points_shape(self, operator: LShapedPoissonOperator) -> None:
        # n_points_per_face=10 distributes 6*10=60 points across the L-shape's
        # boundary segments (per the operator's docstring).
        pts = operator.generate_boundary_points(n_points_per_face=10, seed=42)
        assert pts.shape == (60, 2)
        assert pts.dtype == np.float32

    def test_generate_boundary_points_seed_reproducible(
        self, operator: LShapedPoissonOperator
    ) -> None:
        pts_a = operator.generate_boundary_points(n_points_per_face=8, seed=11)
        pts_b = operator.generate_boundary_points(n_points_per_face=8, seed=11)
        np.testing.assert_array_equal(pts_a, pts_b)

    def test_singular_solution_zero_on_positive_x_axis(
        self, operator: LShapedPoissonOperator
    ) -> None:
        # The L-shaped exact solution u(r, theta) = r^(2/3) * sin(2*theta/3)
        # vanishes at theta = 0 (positive x-axis). r^(2/3) * sin(0) = 0.
        coords = torch.tensor([[0.5, 0.0], [0.7, 0.0]], dtype=torch.float32)
        u_exact = operator.exact_solution(coords)
        assert isinstance(u_exact, torch.Tensor)
        # Allow float-arithmetic noise around zero.
        assert float(u_exact.abs().max()) < 1e-5


# ---------------------------------------------------------------------------
# Section 2.1 follow-up: NavierStokes Taylor-Green vortex paths +
# 1D-u bug fix regression + LShapedPoissonOperator polar helpers.
# Targets the remaining gaps after PR #70 (operators.py 74% -> 80%+).
# ---------------------------------------------------------------------------


class TestNavierStokesOperatorTaylorGreen:
    """Cover NavierStokesOperator non-residual methods (lines 1157-1232).

    Validates the Taylor-Green vortex closed-form solution that the SBIR
    benchmarks rely on:

        u_x(x, y, t) = -cos(x) sin(y) exp(-2 nu t)
        u_y(x, y, t) =  sin(x) cos(y) exp(-2 nu t)
        p(x, y, t)   = -(cos(2x) + cos(2y)) exp(-4 nu t) / 4
    """

    @pytest.fixture
    def ns_config(self) -> PDEConfig:
        return PDEConfig(
            name="ns_tg",
            pde_type=PDEType.NAVIER_STOKES,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            diffusion_coeff=0.1,
            is_time_dependent=True,
        )

    @pytest.fixture
    def operator(self, ns_config: PDEConfig) -> NavierStokesOperator:
        return NavierStokesOperator(ns_config)

    def test_source_term_tensor_path_is_zero(self, operator: NavierStokesOperator) -> None:
        """Taylor-Green vortex has no explicit forcing — source is zero."""
        coords = torch.tensor([[0.3, 0.4], [0.6, 0.7]], dtype=torch.float32)
        source = operator.source_term(coords)
        assert isinstance(source, torch.Tensor)
        assert source.shape == (2,)
        assert torch.allclose(source, torch.zeros_like(source))

    def test_source_term_numpy_path_is_zero(self, operator: NavierStokesOperator) -> None:
        coords = np.array([[0.3, 0.4], [0.6, 0.7]], dtype=np.float32)
        source = operator.source_term(coords)
        assert isinstance(source, np.ndarray)
        assert source.shape == (2,)
        assert source.dtype == np.float32
        np.testing.assert_array_equal(source, np.zeros_like(source))

    def test_exact_solution_tensor_at_origin(self, operator: NavierStokesOperator) -> None:
        # u_x(0, 0, 0) = -cos(0) sin(0) = 0 ; u_y(0, 0, 0) = sin(0) cos(0) = 0
        coords = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        u = operator.exact_solution(coords, time=0.0)
        assert isinstance(u, torch.Tensor)
        assert u.shape == (1, 2)
        assert torch.allclose(u, torch.zeros_like(u), atol=1e-6)

    def test_exact_solution_numpy_path_finite(self, operator: NavierStokesOperator) -> None:
        coords = np.array([[0.5, 0.5], [np.pi / 4, np.pi / 4]], dtype=np.float32)
        u = operator.exact_solution(coords, time=0.1)
        assert isinstance(u, np.ndarray)
        assert u.dtype == np.float32
        assert u.shape == (2, 2)
        assert np.isfinite(u).all()

    def test_exact_solution_decays_with_time(self, operator: NavierStokesOperator) -> None:
        """exp(-2 nu t) factor strictly shrinks the magnitude over time."""
        coords = torch.tensor([[np.pi / 4, np.pi / 4]], dtype=torch.float32)
        u_t0 = operator.exact_solution(coords, time=0.0)
        u_t1 = operator.exact_solution(coords, time=1.0)
        assert isinstance(u_t0, torch.Tensor)
        assert isinstance(u_t1, torch.Tensor)
        # Both components should shrink by the decay factor.
        assert float(u_t1.abs().max()) < float(u_t0.abs().max())

    def test_exact_solution_default_time_is_zero(self, operator: NavierStokesOperator) -> None:
        coords = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        u_default = operator.exact_solution(coords, time=None)
        u_t0 = operator.exact_solution(coords, time=0.0)
        assert isinstance(u_default, torch.Tensor)
        assert isinstance(u_t0, torch.Tensor)
        assert torch.allclose(u_default, u_t0)

    def test_exact_pressure_tensor_path_finite(self, operator: NavierStokesOperator) -> None:
        coords = torch.tensor([[0.5, 0.5], [0.0, 0.0]], dtype=torch.float32)
        p = operator.exact_pressure(coords, time=0.0)
        assert isinstance(p, torch.Tensor)
        assert p.shape == (2,)
        assert torch.isfinite(p).all()
        # At (0, 0): p = -(cos(0)+cos(0))/4 = -0.5
        assert float(p[1].item()) == pytest.approx(-0.5, abs=1e-5)

    def test_exact_pressure_numpy_path_finite(self, operator: NavierStokesOperator) -> None:
        coords = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
        p = operator.exact_pressure(coords, time=0.0)
        assert isinstance(p, np.ndarray)
        assert p.dtype == np.float32
        assert p.shape == (2,)
        assert np.isfinite(p).all()

    def test_exact_pressure_default_time_is_zero(self, operator: NavierStokesOperator) -> None:
        coords = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        p_default = operator.exact_pressure(coords, time=None)
        p_t0 = operator.exact_pressure(coords, time=0.0)
        assert isinstance(p_default, torch.Tensor)
        assert isinstance(p_t0, torch.Tensor)
        assert torch.allclose(p_default, p_t0)

    def test_boundary_value_delegates_to_exact(self, operator: NavierStokesOperator) -> None:
        coords = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
        bv = operator.boundary_value(coords, time=0.5)
        u = operator.exact_solution(coords, time=0.5)
        assert isinstance(bv, torch.Tensor)
        assert isinstance(u, torch.Tensor)
        assert torch.allclose(bv, u)

    def test_initial_condition_is_exact_at_t_zero(self, operator: NavierStokesOperator) -> None:
        coords = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        ic = operator.initial_condition(coords)
        u_t0 = operator.exact_solution(coords, time=0.0)
        assert isinstance(ic, torch.Tensor)
        assert isinstance(u_t0, torch.Tensor)
        assert torch.allclose(ic, u_t0)


class TestNavierStokesOperator1DURegression:
    """Regression tests for the 1D-u residual fix.

    Pre-fix: ``residual()`` errored with "element 0 of tensors does not
    require grad" when called with a 1D ``u`` (shape ``(N,)``). The
    ``else`` branch at line 1067 set ``uy = torch.zeros_like(u)`` (a
    constant with no grad path), and the subsequent
    ``torch.autograd.grad(uy, coords, ...)`` raised.

    Post-fix: when ``uy`` lacks a grad path the operator skips that grad
    call entirely (``grad_uy = None``), which the downstream
    ``if grad_ux is not None and grad_uy is not None`` check at line 1089
    already handled by routing into the zero-residual fallback.
    """

    @pytest.fixture
    def operator(self) -> NavierStokesOperator:
        return NavierStokesOperator(
            PDEConfig(
                name="ns_1d_regression",
                pde_type=PDEType.NAVIER_STOKES,
                domain_dim=2,
                domain_min=[0.0, 0.0],
                domain_max=[1.0, 1.0],
                diffusion_coeff=0.1,
                is_time_dependent=True,
            )
        )

    def test_1d_u_does_not_raise(self, operator: NavierStokesOperator) -> None:
        """Pre-fix this would raise RuntimeError on autograd of zeros uy."""
        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # u depends on coords (so ux has grad), but is a single scalar field.
        u = torch.sin(coords[:, 0]) + torch.cos(coords[:, 1])
        residual = operator.residual(u, coords)
        assert isinstance(residual, PDEResidual)

    def test_1d_u_yields_zero_residual_via_fallback(self, operator: NavierStokesOperator) -> None:
        """grad_uy is None -> downstream zero-residual fallback at line 1128."""
        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        u = torch.sin(coords[:, 0]) + torch.cos(coords[:, 1])
        residual = operator.residual(u, coords)
        # The shape[-1] >= 2 branch is False so residual is the zero fallback.
        assert torch.allclose(residual.values, torch.zeros_like(residual.values))

    def test_1d_u_disabled_derivatives_does_not_raise(self, operator: NavierStokesOperator) -> None:
        coords = torch.tensor([[0.3, 0.4]], dtype=torch.float32, requires_grad=True)
        u = torch.sin(coords[:, 0])
        residual = operator.residual(u, coords, compute_derivatives=False)
        assert residual.derivatives == {}

    def test_detached_ux_does_not_raise(self, operator: NavierStokesOperator) -> None:
        """Symmetric guard: a detached ux (no grad path) must not crash either.

        Code-review feedback on PR #71 noted that the same defensive
        guard applied to ``uy`` should also protect ``ux`` against
        non-differentiable inputs (e.g., a numerical-stencil baseline
        being benchmarked against the autodiff residual). Pre-extension
        this would have raised "element 0 of tensors does not require
        grad" on the ``autograd.grad(ux, ...)`` call.
        """
        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # u_x is constructed without grad path (detached); u_y has one.
        ux_detached = torch.tensor([0.5, 0.7])
        uy_with_grad = torch.sin(coords[:, 0])
        u = torch.stack([ux_detached, uy_with_grad], dim=-1)
        residual = operator.residual(u, coords)
        # grad_ux is None -> downstream zero-residual fallback.
        assert torch.allclose(residual.values, torch.zeros_like(residual.values))

    def test_both_ux_and_uy_detached_is_safe_zero(self, operator: NavierStokesOperator) -> None:
        """Both components detached -> both grads None -> zero residual."""
        coords = torch.tensor(
            [[0.3, 0.4], [0.5, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        u = torch.tensor([[0.5, 0.7], [0.3, 0.2]])
        residual = operator.residual(u, coords)
        assert torch.allclose(residual.values, torch.zeros_like(residual.values))


class TestLShapedPoissonOperatorPolarHelpers:
    """Cover LShapedPoissonOperator polar conversion + numpy singular path.

    Targets lines 1297-1338, including the ``r > 0`` guard against
    ``0^(2/3) -> nan`` in ``_singular_solution_np``.
    """

    def test_polar_from_cartesian_positive_x_axis(self) -> None:
        # On the positive x-axis: r=|x|, theta=0.
        x = torch.tensor([0.5, 1.0])
        y = torch.tensor([0.0, 0.0])
        r, theta = LShapedPoissonOperator._polar_from_cartesian(x, y)
        torch.testing.assert_close(r, torch.tensor([0.5, 1.0]))
        torch.testing.assert_close(theta, torch.tensor([0.0, 0.0]))

    def test_polar_from_cartesian_wraps_negative_angles(self) -> None:
        """atan2 returns negative theta for y<0; the helper maps to [0, 2*pi)."""
        # Point at (0, -1) has atan2 = -pi/2; helper should remap to 3*pi/2.
        x = torch.tensor([0.0])
        y = torch.tensor([-1.0])
        _, theta = LShapedPoissonOperator._polar_from_cartesian(x, y)
        torch.testing.assert_close(theta, torch.tensor([3.0 * np.pi / 2.0]))

    def test_singular_solution_np_handles_origin(self) -> None:
        """Origin (r=0) must short-circuit to 0 to avoid 0**(2/3) -> nan."""
        x = np.array([0.0, 0.5], dtype=np.float32)
        y = np.array([0.0, 0.0], dtype=np.float32)
        u = LShapedPoissonOperator._singular_solution_np(x, y)
        assert u.dtype == np.float32
        assert np.isfinite(u).all()
        # u(0, 0) is the short-circuit zero.
        assert u[0] == 0.0

    def test_lshaped_exact_solution_numpy_path(self) -> None:
        """Exercise the numpy branch of exact_solution."""
        from src.pde.geometry import GeometryConfig, GeometryType

        config = PDEConfig(
            name="lshaped_np",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-1.0, -1.0],
            domain_max=[1.0, 1.0],
            geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED),
        )
        op = LShapedPoissonOperator(config)
        coords = np.array([[0.5, 0.0], [0.0, 0.0]], dtype=np.float32)
        u = op.exact_solution(coords)
        assert isinstance(u, np.ndarray)
        assert u.dtype == np.float32
        # u(0.5, 0) = 0.5^(2/3) * sin(0) = 0 ; u(0,0) = 0 (origin short-circuit).
        np.testing.assert_allclose(u, [0.0, 0.0], atol=1e-6)
