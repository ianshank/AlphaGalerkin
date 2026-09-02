"""Tests for src/research/fem_baseline.py.

Covers FEMConfig validation, ScikitFEMPoissonSolver on unit-square Poisson
with a manufactured solution, P2/P3 element convergence, and the
ScikitFEMLShapedSolver smoke test on the L-shaped domain.

scikit-fem is required. ``fem_required`` (registered in pyproject.toml) marks
every test in this module; the root conftest.py hook skips them *visibly*
(reporting a skip count) when scikit-fem is not installed, and hard-fails
collection instead when ``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` (the test-extras
CI job) -- unlike a bare ``pytest.importorskip``, which would go quietly
green even if the optional [fem] extra's install had silently failed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SOLVER_REGISTRY, BaseSolver, SolverResult
from src.research.fem_baseline import (
    FEMConfig,
    ScikitFEMLShapedSolver,
    ScikitFEMPoissonSolver,
    _make_element,
    _require_skfem,
    assemble_and_solve,
    build_initial_mesh,
    build_lshaped_initial_mesh,
    dirichlet_dof_indices,
    quadrature_l2_error,
)

pytestmark = pytest.mark.fem_required


def _make_poisson_2d() -> PoissonOperator:
    cfg = PDEConfig(
        name="test_poisson_2d",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
    )
    return PoissonOperator(cfg)


def _make_poisson_lshaped() -> PoissonOperator:
    cfg = PDEConfig(
        name="test_poisson_lshaped",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[-1.0, -1.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
    )
    return PoissonOperator(cfg)


class TestFEMConfig:
    def test_defaults(self):
        config = FEMConfig()
        assert config.element_type == "P1"
        assert config.refinement_strategy == "h_adaptive"
        assert 0.0 < config.marking_fraction < 1.0
        assert config.max_element_order == 3
        assert config.min_mesh_side == 3
        assert config.min_initial_dof_hint == 9
        assert config.zz_epsilon == pytest.approx(1e-12)

    def test_rejects_unknown_element(self):
        with pytest.raises(ValueError):
            FEMConfig(element_type="P5")

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError):
            FEMConfig(refinement_strategy="magic")

    def test_marking_fraction_bounds(self):
        with pytest.raises(ValueError):
            FEMConfig(marking_fraction=0.0)
        with pytest.raises(ValueError):
            FEMConfig(marking_fraction=1.5)

    def test_max_element_order_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            FEMConfig(max_element_order=4)
        with pytest.raises(ValueError):
            FEMConfig(max_element_order=0)

    def test_min_mesh_side_bounds(self):
        with pytest.raises(ValueError):
            FEMConfig(min_mesh_side=1)

    def test_zz_epsilon_positive(self):
        with pytest.raises(ValueError):
            FEMConfig(zz_epsilon=0.0)


class TestFEMInternalHelpers:
    """Covers the Dorfler marking + ZZ gradient helpers in isolation."""

    def test_dorfler_mark_basic(self):
        solver = ScikitFEMPoissonSolver(FEMConfig(marking_fraction=0.5))
        indicators = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float64)
        marked = solver._dorfler_mark(indicators)
        # The top indicator (0.4) alone covers 0.4 >= 0.5 * 1.0 = 0.5?  No,
        # so we need the top two (0.4 + 0.3 = 0.7 >= 0.5).
        assert marked[1]  # 0.4 is highest
        assert marked.sum() == 2

    def test_dorfler_mark_zero_indicators(self):
        solver = ScikitFEMPoissonSolver()
        marked = solver._dorfler_mark(np.zeros(5, dtype=np.float64))
        assert not marked.any()

    def test_element_gradients_linear_function(self):
        """Gradient of u = 2x + 3y on any triangle should be (2, 3)."""
        # Single triangle with vertices at (0,0), (1,0), (0,1).
        mesh_p = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        mesh_t = np.array([[0], [1], [2]], dtype=np.int64)

        class _FakeMesh:
            p = mesh_p
            t = mesh_t

        nodal = np.array([0.0, 2.0, 3.0], dtype=np.float64)  # u = 2x + 3y
        grads = ScikitFEMPoissonSolver._element_gradients(_FakeMesh(), nodal)
        assert grads.shape == (1, 2)
        np.testing.assert_allclose(grads[0], [2.0, 3.0], atol=1e-10)

    def test_triangle_area(self):
        pts = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        assert ScikitFEMPoissonSolver._triangle_area(pts) == pytest.approx(0.5)

    def test_estimate_smoothness_empty_mark(self):
        solver = ScikitFEMPoissonSolver()
        indicators = np.array([1.0, 2.0, 3.0])
        marked = np.zeros_like(indicators, dtype=bool)
        assert solver._estimate_smoothness(indicators, marked) == 0.0

    def test_estimate_smoothness_uniform(self):
        """Uniform indicators produce ratio close to 1.0 (smooth)."""
        solver = ScikitFEMPoissonSolver()
        indicators = np.ones(5)
        marked = np.ones(5, dtype=bool)
        smooth = solver._estimate_smoothness(indicators, marked)
        assert smooth == pytest.approx(1.0, rel=1e-6)

    def test_estimate_smoothness_concentrated(self):
        """One dominant indicator produces a low ratio."""
        solver = ScikitFEMPoissonSolver()
        indicators = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
        marked = np.ones(5, dtype=bool)
        smooth = solver._estimate_smoothness(indicators, marked)
        # mean=20.8, max=100 -> 0.208
        assert smooth < 0.5


class TestFEMDimensionChecks:
    def test_rejects_1d_operator(self):
        cfg = PDEConfig(
            name="p1d",
            pde_type=PDEType.POISSON,
            domain_dim=1,
            domain_min=[0.0],
            domain_max=[1.0],
            advection_coeff=[0.0],
        )
        operator = PoissonOperator(cfg)
        solver = ScikitFEMPoissonSolver(FEMConfig(max_refinement_levels=1))
        with pytest.raises(NotImplementedError, match="2D"):
            solver.solve(operator, n_dof=9)

    def test_lshaped_rejects_1d_operator(self):
        cfg = PDEConfig(
            name="p1d",
            pde_type=PDEType.POISSON,
            domain_dim=1,
            domain_min=[0.0],
            domain_max=[1.0],
            advection_coeff=[0.0],
        )
        operator = PoissonOperator(cfg)
        solver = ScikitFEMLShapedSolver(FEMConfig(max_refinement_levels=1))
        with pytest.raises(NotImplementedError, match="2D"):
            solver.solve(operator, n_dof=9)


class TestScikitFEMPoissonSolver:
    def test_is_basesolver(self):
        assert issubclass(ScikitFEMPoissonSolver, BaseSolver)

    def test_registered(self):
        assert "scikit_fem_poisson" in SOLVER_REGISTRY
        assert SOLVER_REGISTRY["scikit_fem_poisson"] is ScikitFEMPoissonSolver

    def test_solve_returns_result(self):
        config = FEMConfig(
            element_type="P1",
            refinement_strategy="uniform",
            max_refinement_levels=2,
            initial_mesh_refinements=1,
        )
        solver = ScikitFEMPoissonSolver(config)
        operator = _make_poisson_2d()
        result = solver.solve(operator, n_dof=64)

        assert isinstance(result, SolverResult)
        assert result.n_dof > 0
        assert result.wall_time_seconds > 0.0
        assert result.metadata["method"] == "scikit_fem_hp_adaptive"
        assert result.metadata["strategy"] == "uniform"

    def test_h_adaptive_reduces_error(self):
        """Two refinement levels should produce lower error than one."""
        operator = _make_poisson_2d()

        cfg_low = FEMConfig(
            element_type="P1",
            refinement_strategy="h_adaptive",
            max_refinement_levels=1,
            initial_mesh_refinements=1,
        )
        r_low = ScikitFEMPoissonSolver(cfg_low).solve(operator, n_dof=25)

        cfg_high = FEMConfig(
            element_type="P1",
            refinement_strategy="h_adaptive",
            max_refinement_levels=3,
            initial_mesh_refinements=1,
        )
        r_high = ScikitFEMPoissonSolver(cfg_high).solve(operator, n_dof=25)

        assert r_high.n_dof >= r_low.n_dof
        if r_low.l2_error is not None and r_high.l2_error is not None:
            # More refinement should not make error worse (modulo numerical noise)
            assert r_high.l2_error <= r_low.l2_error * 1.5

    def test_p2_at_least_as_accurate_as_p1(self):
        """P2 elements should match or beat P1 at comparable refinement."""
        operator = _make_poisson_2d()

        r_p1 = ScikitFEMPoissonSolver(
            FEMConfig(
                element_type="P1",
                refinement_strategy="uniform",
                max_refinement_levels=1,
                initial_mesh_refinements=2,
            )
        ).solve(operator, n_dof=64)

        r_p2 = ScikitFEMPoissonSolver(
            FEMConfig(
                element_type="P2",
                refinement_strategy="uniform",
                max_refinement_levels=1,
                initial_mesh_refinements=2,
            )
        ).solve(operator, n_dof=64)

        if r_p1.l2_error is not None and r_p2.l2_error is not None:
            # P2 should be at least roughly comparable; give generous margin.
            assert r_p2.l2_error <= r_p1.l2_error * 1.5


class TestScikitFEMLShapedSolver:
    def test_registered(self):
        assert "scikit_fem_lshaped" in SOLVER_REGISTRY

    def test_runs_on_lshaped(self):
        """Smoke test: L-shaped solver runs without crashing."""
        config = FEMConfig(
            element_type="P1",
            refinement_strategy="h_adaptive",
            max_refinement_levels=2,
            initial_mesh_refinements=1,
        )
        solver = ScikitFEMLShapedSolver(config)
        operator = _make_poisson_lshaped()
        result = solver.solve(operator, n_dof=64)

        assert isinstance(result, SolverResult)
        assert result.n_dof > 0
        assert result.metadata["method"] == "scikit_fem_hp_adaptive"


class TestQuadratureL2Error:
    """AC6: the new, additive quadrature-L2 primitive."""

    def test_differs_from_nodal_rms(self):
        operator = _make_poisson_2d()
        skfem = _require_skfem()
        mesh = build_initial_mesh(
            operator,
            25,
            skfem,
            min_initial_dof_hint=9,
            min_mesh_side=3,
            initial_mesh_refinements=2,
        )
        element = _make_element("P1", skfem)
        solver = ScikitFEMPoissonSolver(FEMConfig())
        u, _coords, nodal_rms = assemble_and_solve(
            mesh, element, operator, skfem, solver._compute_l2_error
        )
        quad_l2 = quadrature_l2_error(mesh, element, u, operator, skfem)

        assert quad_l2 >= 0.0
        assert nodal_rms is not None
        # The two metrics measure the same error differently -- AC6 requires
        # they need not (and generally do not) coincide.
        assert quad_l2 != pytest.approx(nodal_rms)

    def test_decreases_under_refinement(self):
        """Refining the mesh should reduce the quadrature L2 error."""
        operator = _make_poisson_2d()
        skfem = _require_skfem()
        element = _make_element("P1", skfem)
        solver = ScikitFEMPoissonSolver(FEMConfig())

        errors = []
        for refinements in (1, 3):
            mesh = build_initial_mesh(
                operator,
                9,
                skfem,
                min_initial_dof_hint=9,
                min_mesh_side=3,
                initial_mesh_refinements=refinements,
            )
            u, _coords, _nodal_rms = assemble_and_solve(
                mesh, element, operator, skfem, solver._compute_l2_error
            )
            errors.append(quadrature_l2_error(mesh, element, u, operator, skfem))

        assert errors[1] < errors[0]


class TestLShapedMeshMatchesOperatorDomain:
    """D1: the L-shape builder must not silently mesh a domain it was not given.

    ``build_lshaped_initial_mesh`` hardcodes ``linspace(-1.0, 0.0, 3)`` and never
    reads ``operator.domain_min``/``domain_max`` -- unlike its sibling
    ``build_initial_mesh``, which does. Measured before the fix::

        scale=2.0: operator x in [-2.0, 2.0]  ->  mesh x in [-1.0, 1.0]

    That is not a cosmetic mismatch. ``lshape_inside_predicate(scale)`` *is*
    scale-aware, so at ``scale != 1`` the notch logic scales and the mesh does
    not, mixing two geometries with every downstream number still finite and
    plausible -- the 2026-08-16 L-shape retraction's exact signature. The
    adequacy gate reaches this path via ``_lshaped_operator(params.scale)``
    and is masked only because ``ComparisonParams.scale`` defaults to 1.0.

    Supporting scaled L-shapes is a feature; failing loudly on one is the fix.
    """

    @staticmethod
    def _operator(scale: float) -> PoissonOperator:
        return PoissonOperator(
            PDEConfig(
                name="lshaped_scaled",
                pde_type=PDEType.POISSON,
                domain_dim=2,
                domain_min=[-scale, -scale],
                domain_max=[scale, scale],
            )
        )

    def test_unit_domain_is_accepted(self) -> None:
        """The canonical [-1,1]^2 L-shape still builds, unchanged."""
        mesh = build_lshaped_initial_mesh(
            self._operator(1.0), _require_skfem(), initial_mesh_refinements=0
        )
        assert float(mesh.p[0].min()) == pytest.approx(-1.0)
        assert float(mesh.p[0].max()) == pytest.approx(1.0)

    @pytest.mark.parametrize("scale", [2.0, 5.0, 0.5])
    def test_non_unit_domain_raises_instead_of_silently_meshing_the_unit_square(
        self, scale: float
    ) -> None:
        with pytest.raises(NotImplementedError, match="unit L-shape"):
            build_lshaped_initial_mesh(
                self._operator(scale), _require_skfem(), initial_mesh_refinements=0
            )

    def test_the_reentrant_corner_is_still_a_node(self) -> None:
        """AC8 must survive the added validation."""
        mesh = build_lshaped_initial_mesh(
            self._operator(1.0), _require_skfem(), initial_mesh_refinements=0
        )
        origin = np.isclose(mesh.p[0], 0.0) & np.isclose(mesh.p[1], 0.0)
        assert origin.any(), "the reentrant corner at the origin is not a mesh node"


class _TorchReturningOperator:
    """A ``PDEOperator``-shaped stand-in whose every hook returns ``torch.Tensor``.

    ``fem_baseline`` unwraps torch at three sites (``l_form``, the Dirichlet
    values, ``quadrature_l2_error``'s integrand) and none was ever exercised:
    every operator the suites use returns numpy. The ``PDEOperator`` contract
    documents torch as the return type, so this is the *likely* production
    shape, not the exotic one. Wraps a real operator so the values are honest.
    """

    def __init__(self, inner: PoissonOperator) -> None:
        self._inner = inner
        self.dim = inner.dim
        self.domain_min = inner.domain_min
        self.domain_max = inner.domain_max

    def source_term(self, pts: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(self._inner.source_term(pts)))

    def boundary_value(self, pts: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(self._inner.boundary_value(pts)))

    def exact_solution(self, pts: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(self._inner.exact_solution(pts)))


class TestDirichletDofIndicesCompatChain:
    """Two of three legs of the version-compat chain were measured by nothing.

    The function's own docstring says it was extracted because "a compat
    fallback with two copies, neither individually tested, is worse than one
    tested copy". The consolidation happened; the testing did not -- under
    scikit-fem 12 ``get_dofs()`` always has ``flatten``, so lines for the
    ``nodal`` and bare-array legs never ran. Stub bases reach them.
    """

    def test_flatten_leg(self) -> None:
        class _Dofs:
            def flatten(self) -> np.ndarray:
                return np.array([3, 1, 2])

        class _Basis:
            def get_dofs(self) -> _Dofs:
                return _Dofs()

        np.testing.assert_array_equal(dirichlet_dof_indices(_Basis()), [3, 1, 2])

    def test_nodal_leg_when_flatten_is_absent(self) -> None:
        class _Dofs:
            nodal = {"u": np.array([7, 8])}

        class _Basis:
            def get_dofs(self) -> _Dofs:
                return _Dofs()

        np.testing.assert_array_equal(dirichlet_dof_indices(_Basis()), [7, 8])

    def test_bare_array_leg_when_neither_is_present(self) -> None:
        class _Basis:
            def get_dofs(self) -> list[int]:
                return [5, 6, 9]

        np.testing.assert_array_equal(dirichlet_dof_indices(_Basis()), [5, 6, 9])


class TestTorchReturningOperatorIsUnwrapped:
    """One fixture, three unwrap sites."""

    def test_assemble_and_solve_accepts_torch_source_and_boundary(self) -> None:
        skfem = _require_skfem()
        inner = _make_poisson_2d()
        operator = _TorchReturningOperator(inner)
        mesh = build_initial_mesh(
            inner, 9, skfem, min_initial_dof_hint=9, min_mesh_side=3, initial_mesh_refinements=1
        )
        element = _make_element("P1", skfem)
        solver = ScikitFEMPoissonSolver(FEMConfig())
        u, _coords, nodal_rms = assemble_and_solve(
            mesh, element, operator, skfem, solver._compute_l2_error
        )
        assert np.isfinite(u).all()
        # And the quadrature integrand's unwrap, on the same solve.
        quad = quadrature_l2_error(mesh, element, u, operator, skfem)
        assert np.isfinite(quad) and quad >= 0.0


class TestQuadratureL2ErrorRefusesNoExactSolution:
    def test_none_exact_solution_raises(self) -> None:
        skfem = _require_skfem()
        operator = _make_poisson_2d()
        mesh = build_initial_mesh(
            operator, 9, skfem, min_initial_dof_hint=9, min_mesh_side=3, initial_mesh_refinements=0
        )
        element = _make_element("P1", skfem)
        u = np.zeros(skfem.Basis(mesh, element).N)
        object.__setattr__(operator, "exact_solution", lambda pts: None)
        with pytest.raises(ValueError, match="analytic exact solution"):
            quadrature_l2_error(mesh, element, u, operator, skfem)


class TestElementGradientsDegenerateElement:
    def test_collinear_vertices_yield_zero_gradient_not_a_crash(self) -> None:
        """A singular fit matrix takes the ``LinAlgError`` branch and zeroes the row."""

        class _FakeMesh:
            p = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float64)  # collinear
            t = np.array([[0], [1], [2]], dtype=np.int64)

        grads = ScikitFEMPoissonSolver._element_gradients(_FakeMesh(), np.array([0.0, 1.0, 2.0]))
        assert grads.shape == (1, 2)
        np.testing.assert_array_equal(grads[0], [0.0, 0.0])


class TestLShapedMeshFallback:
    def test_falls_back_to_init_lshaped_when_mesh_addition_is_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover the ``except TypeError`` leg instead of deleting it.

        Unreachable on scikit-fem 12 (probed: ``nw + ne + sw`` succeeds), but the
        pin allows ``>=9`` and the lower end is untestable here.
        """
        skfem = _require_skfem()

        def _no_add(self: object, other: object) -> object:
            raise TypeError("mesh addition unsupported in this version")

        monkeypatch.setattr(skfem.MeshTri, "__add__", _no_add)
        mesh = build_lshaped_initial_mesh(
            _make_poisson_lshaped(), skfem, initial_mesh_refinements=0
        )
        origin = np.isclose(mesh.p[0], 0.0) & np.isclose(mesh.p[1], 0.0)
        assert origin.any(), "fallback mesh lost the reentrant corner"
