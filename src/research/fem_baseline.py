"""scikit-fem hp-adaptive FEM baseline.

Provides a credible classical Galerkin reference that DOE ASCR applied-math
reviewers will accept as the baseline for AlphaGalerkin comparisons. Unlike
the hand-rolled FDM+Dorfler in ``baselines.py``, this uses true Lagrange
finite elements (P1/P2/P3) assembled via ``skfem`` with a
Zienkiewicz-Zhu recovered-gradient error estimator and Dorfler bulk-marking
adaptive refinement.

Supports three adaptation strategies:

* ``uniform`` — uniform global refinement
* ``h_adaptive`` — Dorfler marking on ZZ indicator, h-refinement only
* ``p_adaptive`` — p-enrichment only (increases polynomial degree)
* ``hp_adaptive`` — combined h/p refinement with smoothness-based arbitration
"""

from __future__ import annotations

import time
from typing import Any, Literal, cast

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from pydantic import Field

from src.pde.operators import PDEOperator
from src.research.baselines import (
    SOLVER_REGISTRY,
    BaseSolver,
    SolverConfig,
    SolverResult,
)
from src.research.marking import dorfler_mark

logger = structlog.get_logger(__name__)


class FEMConfig(SolverConfig):
    """Configuration for scikit-fem-based FEM solvers."""

    element_type: Literal["P1", "P2", "P3"] = Field(
        default="P1",
        description="Lagrange element type (P1=linear, P2=quadratic, P3=cubic)",
    )
    refinement_strategy: Literal["uniform", "h_adaptive", "p_adaptive", "hp_adaptive"] = Field(
        default="h_adaptive",
        description="Mesh adaptation strategy",
    )
    marking_fraction: float = Field(
        default=0.3,
        gt=0.0,
        lt=1.0,
        description="Dorfler bulk-marking fraction (theta)",
    )
    max_refinement_levels: int = Field(
        default=10,
        ge=1,
        description="Maximum adaptive refinement iterations",
    )
    target_tolerance: float = Field(
        default=1e-4,
        gt=0,
        description="Target L2 error tolerance for early termination",
    )
    initial_mesh_refinements: int = Field(
        default=2,
        ge=0,
        description="Number of uniform refinements applied to the initial mesh",
    )
    smoothness_threshold: float = Field(
        default=0.5,
        gt=0,
        description="hp-adaptive threshold: smoother elements (above this) get p-refined",
    )
    max_element_order: Literal[1, 2, 3] = Field(
        default=3,
        description="Upper bound on Lagrange element order (P3 is scikit-fem's max)",
    )
    min_mesh_side: int = Field(
        default=3,
        ge=2,
        description="Minimum grid points per side when sizing the initial tensor mesh",
    )
    min_initial_dof_hint: int = Field(
        default=9,
        ge=1,
        description="Floor on n_dof hint used to size the initial mesh",
    )
    zz_epsilon: float = Field(
        default=1e-12,
        gt=0,
        description="Numerical floor in the smoothness denominator and area guards",
    )


def _require_skfem() -> Any:
    """Import scikit-fem or raise a helpful error."""
    try:
        import skfem
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise ImportError(
            "ScikitFEM*Solver requires scikit-fem. "
            "Install via the project extra: pip install -e '.[fem]' "
            "(or ``pip install scikit-fem>=9.0`` if you only need the dependency)."
        ) from exc
    return skfem


def _make_element(element_type: str, skfem: Any) -> Any:
    """Return a scikit-fem Element instance for the requested Lagrange order."""
    mapping = {
        "P1": skfem.ElementTriP1,
        "P2": skfem.ElementTriP2,
        "P3": skfem.ElementTriP3,
    }
    cls = mapping.get(element_type)
    if cls is None:
        raise ValueError(f"Unknown element_type '{element_type}'. Expected P1/P2/P3.")
    return cls()


# ---------------------------------------------------------------------------
# Module-level primitives.
#
# Extracted from ScikitFEMPoissonSolver/ScikitFEMLShapedSolver so
# src.research.substrates.skfem_tri.SkfemTriSubstrate can reuse the exact
# same mesh/assembly/estimator code the classical baseline uses, instead of
# a second copy that could silently drift. The class methods below become
# thin delegates over these; SolverResult.l2_error's existing meaning (nodal
# RMS, asserted by all 23 pre-existing tests) is unchanged -- quadrature_l2_error
# is additive, per AC6, and is consumed only by SkfemTriSubstrate.
# ---------------------------------------------------------------------------


def build_initial_mesh(
    operator: PDEOperator,
    n_dof: int,
    skfem: Any,
    *,
    min_initial_dof_hint: int,
    min_mesh_side: int,
    initial_mesh_refinements: int,
) -> Any:
    """Construct an initial rectangular mesh sized to the hint n_dof."""
    if operator.dim != 2:
        raise NotImplementedError(
            f"ScikitFEMPoissonSolver currently supports 2D problems "
            f"(got operator.dim={operator.dim})"
        )

    xmin, ymin = float(operator.domain_min[0]), float(operator.domain_min[1])
    xmax, ymax = float(operator.domain_max[0]), float(operator.domain_max[1])
    dof_hint = max(n_dof, min_initial_dof_hint)
    side = max(min_mesh_side, int(np.ceil(np.sqrt(dof_hint))))

    xs = np.linspace(xmin, xmax, side)
    ys = np.linspace(ymin, ymax, side)
    mesh = skfem.MeshTri.init_tensor(xs, ys)

    for _ in range(initial_mesh_refinements):
        mesh = mesh.refined()
    return mesh


def build_lshaped_initial_mesh(
    operator: PDEOperator,
    skfem: Any,
    *,
    initial_mesh_refinements: int,
) -> Any:
    r"""Build an L-shaped mesh by assembling three unit squares.

    The L-shaped domain is defined as ``[-1,1]^2 \ [0,1]x[-1,0]``. The
    reentrant corner at the origin (AC8) is a mesh node on both the primary
    (sum) path and the ``init_lshaped()`` fallback.
    """
    if operator.dim != 2:
        raise NotImplementedError("L-shaped domain is inherently 2D")

    nw = skfem.MeshTri.init_tensor(np.linspace(-1.0, 0.0, 3), np.linspace(0.0, 1.0, 3))
    ne = skfem.MeshTri.init_tensor(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    sw = skfem.MeshTri.init_tensor(np.linspace(-1.0, 0.0, 3), np.linspace(-1.0, 0.0, 3))

    try:
        mesh = nw + ne + sw
    except TypeError:
        if hasattr(skfem.MeshTri, "init_lshaped"):
            mesh = skfem.MeshTri.init_lshaped()
        else:  # pragma: no cover
            raise

    for _ in range(initial_mesh_refinements):
        mesh = mesh.refined()
    return mesh


def assemble_and_solve(
    mesh: Any,
    element: Any,
    operator: PDEOperator,
    skfem: Any,
    l2_error_fn: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float | None]:
    """Assemble the Poisson system and solve it.

    Args:
        mesh: The skfem mesh to assemble on.
        element: The skfem Element (Lagrange order) to use.
        operator: PDE operator supplying ``source_term``/``boundary_value``.
        skfem: The imported ``skfem`` module.
        l2_error_fn: ``(solution, coords, operator) -> float | None`` --
            injected so this stays a pure function while
            ``SolverResult.l2_error``'s nodal-RMS meaning is preserved
            byte-for-byte (``BaseSolver._compute_l2_error``).

    """
    from scipy.sparse.linalg import spsolve
    from skfem import BilinearForm, LinearForm
    from skfem.helpers import dot, grad

    basis = skfem.Basis(mesh, element)

    @BilinearForm  # type: ignore[untyped-decorator]
    def a_form(u: Any, v: Any, _: Any) -> Any:
        return dot(grad(u), grad(v))

    @LinearForm  # type: ignore[untyped-decorator]
    def l_form(v: Any, w: Any) -> Any:
        x = w.x  # quadrature point coordinates (dim, n_q, n_elements)
        pts = np.stack([x[0].ravel(), x[1].ravel()], axis=-1).astype(np.float32)
        f_out = operator.source_term(pts)
        if isinstance(f_out, torch.Tensor):
            f_out = f_out.detach().cpu().numpy()
        f_vals = np.asarray(f_out, dtype=np.float64)
        f = f_vals.reshape(x[0].shape)
        return f * v

    A = a_form.assemble(basis)
    b = l_form.assemble(basis)

    # Dirichlet boundary conditions via operator.boundary_value,
    # evaluated at the actual DOF locations (covers P2/P3 edge dofs).
    dirichlet_dofs = basis.get_dofs()
    if hasattr(dirichlet_dofs, "flatten"):
        dof_indices = dirichlet_dofs.flatten()
    elif hasattr(dirichlet_dofs, "nodal"):
        dof_indices = dirichlet_dofs.nodal["u"]
    else:
        dof_indices = np.asarray(dirichlet_dofs)

    dof_locs = basis.doflocs  # (dim, n_dof)
    bc_pts = np.asarray(dof_locs[:, dof_indices].T, dtype=np.float32)
    bc_out = operator.boundary_value(bc_pts)
    if isinstance(bc_out, torch.Tensor):
        bc_out = bc_out.detach().cpu().numpy()
    bc_values = np.asarray(bc_out, dtype=np.float64).flatten()

    u = np.zeros(A.shape[0], dtype=np.float64)
    u[dof_indices] = bc_values

    # Solve the condensed system
    A_c, b_c, u_c, I = skfem.condense(A, b, D=dirichlet_dofs, x=u)
    u_c = spsolve(A_c, b_c)
    u[I] = u_c

    # Use basis.doflocs as the coordinate array so it has the same
    # length as u (critical for P2/P3 where u has edge-midpoint dofs).
    coords = basis.doflocs.T.astype(np.float64)
    l2_err = l2_error_fn(u, coords, operator)

    return u.astype(np.float64), coords, l2_err


def quadrature_l2_error(
    mesh: Any,
    element: Any,
    solution: NDArray[np.float64],
    operator: PDEOperator,
    skfem: Any,
) -> float:
    """Mesh-independent (quadrature) L2 error against the exact solution.

    New primitive (AC6): additive alongside the nodal RMS
    ``BaseSolver._compute_l2_error`` computes today, not a replacement.
    ``nodalRMS / quadratureL2`` drifts with mesh grading (spike-measured
    0.34->0.53 uniform vs 0.34->0.76 adaptive), so reporting only the nodal
    RMS would flatter whichever arm refines hardest.
    """
    basis = skfem.Basis(mesh, element)

    @skfem.Functional  # type: ignore[untyped-decorator]
    def squared_error(w: Any) -> Any:
        x = w.x
        pts = np.stack([x[0].ravel(), x[1].ravel()], axis=-1).astype(np.float32)
        exact_out = operator.exact_solution(pts)
        if isinstance(exact_out, torch.Tensor):
            exact_out = exact_out.detach().cpu().numpy()
        exact_vals = np.asarray(exact_out, dtype=np.float64).reshape(x[0].shape)
        return (w["uh"] - exact_vals) ** 2

    integral = squared_error.assemble(basis, uh=basis.interpolate(solution))
    return float(np.sqrt(integral))


def zz_indicator(mesh: Any, solution: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute per-element ZZ recovered-gradient error indicators.

    The ZZ estimator computes eta_K = ||grad(u_h) - G(grad(u_h))||_{L2(K)}
    where G is the patch-recovery operator implemented here as nodal
    averaging of element-wise gradients.
    """
    n_elements = mesh.t.shape[1]

    # Element-wise constant gradients via P1 approximation of u_h. For
    # simplicity we downsample to P1 gradient; P2/P3 refinements still get a
    # valid (if slightly conservative) indicator.
    nodal = solution[: mesh.p.shape[1]]
    grads = element_gradients(mesh, nodal)  # (n_elements, 2)

    # Nodal recovery via unweighted averaging over incident elements
    n_nodes = mesh.p.shape[1]
    node_sum = np.zeros((n_nodes, 2), dtype=np.float64)
    node_count = np.zeros(n_nodes, dtype=np.int64)
    for k in range(n_elements):
        for vid in mesh.t[:, k]:
            node_sum[vid] += grads[k]
            node_count[vid] += 1
    node_count = np.maximum(node_count, 1)
    recovered = node_sum / node_count[:, None]

    # Per-element indicator: area-weighted L2 gap between element grad
    # and averaged nodal-recovered grad.
    indicators = np.zeros(n_elements, dtype=np.float64)
    for k in range(n_elements):
        vids = mesh.t[:, k]
        rec_k = recovered[vids].mean(axis=0)
        diff = grads[k] - rec_k
        area = triangle_area(mesh.p[:, vids])
        indicators[k] = float(area * (diff @ diff))
    return indicators


def element_gradients(mesh: Any, nodal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute constant gradient per P1 triangle element."""
    t = mesh.t
    p = mesh.p
    n_elements = t.shape[1]
    out = np.zeros((n_elements, 2), dtype=np.float64)
    for k in range(n_elements):
        vids = t[:, k]
        pts = p[:, vids].T  # (3, 2)
        vals = nodal[vids]  # (3,)
        # Solve for linear fit u = a + b*x + c*y => grad = (b, c)
        M = np.column_stack([np.ones(3), pts])
        try:
            coef = np.linalg.solve(M, vals)
            out[k] = coef[1:]
        except np.linalg.LinAlgError:
            out[k] = 0.0
    return out


def triangle_area(pts: NDArray[np.float64]) -> float:
    """Area of a triangle given (2, 3) vertex coordinates."""
    x = pts[0]
    y = pts[1]
    return float(0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])))


def estimate_smoothness(
    indicators: NDArray[np.float64],
    marked: NDArray[np.bool_],
    zz_epsilon: float,
) -> float:
    """Rough smoothness proxy: ratio of marked-region mean to max.

    High values indicate a smooth solution (suitable for p-refinement);
    low values suggest a concentrated feature (suitable for h-refinement).
    """
    marked_vals = indicators[marked]
    if marked_vals.size == 0:
        return 0.0
    return float(np.mean(marked_vals) / (np.max(marked_vals) + zz_epsilon))


class ScikitFEMPoissonSolver(BaseSolver):
    """hp-adaptive FEM solver for Poisson-type problems on rectangular domains.

    Uses scikit-fem to assemble the stiffness matrix and load vector, solves
    via ``scipy.sparse.linalg.spsolve``, computes a Zienkiewicz-Zhu
    recovered-gradient error estimator per element, and iteratively refines
    using Dorfler bulk-marking.
    """

    name = "scikit_fem_poisson"
    description = "hp-adaptive FEM (scikit-fem) on rectangular domain"

    def __init__(self, config: FEMConfig | None = None) -> None:
        self.config = config or FEMConfig()

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Run adaptive FEM and return a ``SolverResult``.

        The ``n_dof`` argument is used only as a hint for the initial mesh
        size; the adaptive loop will terminate when either the target
        tolerance is met or ``max_refinement_levels`` is reached.
        """
        skfem = _require_skfem()
        log = logger.bind(
            solver=self.name,
            element_type=self.config.element_type,
            strategy=self.config.refinement_strategy,
            target_n_dof=n_dof,
        )
        log.info("fem_solve_start")
        t0 = time.perf_counter()

        mesh = self._build_initial_mesh(operator, n_dof, skfem)
        element_type = self.config.element_type
        element = _make_element(element_type, skfem)

        last_solution: NDArray[np.float64] | None = None
        last_coords: NDArray[np.float64] | None = None
        last_l2_err: float | None = None
        levels_used = 0

        for level in range(self.config.max_refinement_levels):
            solution, coords, l2_err = self._assemble_and_solve(mesh, element, operator, skfem)
            last_solution, last_coords, last_l2_err = solution, coords, l2_err
            levels_used = level + 1

            if l2_err is not None and l2_err < self.config.target_tolerance:
                log.info("fem_tolerance_reached", level=level, l2_error=l2_err)
                break

            if self.config.refinement_strategy == "uniform":
                mesh = mesh.refined()
                continue

            indicators = self._compute_zz_indicator(mesh, element, solution, skfem)
            marked = self._dorfler_mark(indicators)

            if not np.any(marked):
                log.debug("no_elements_marked", level=level)
                break

            max_p = self.config.max_element_order
            if self.config.refinement_strategy == "p_adaptive":
                current_p = _order_of(element_type)
                if current_p >= max_p:
                    log.info(
                        "p_adaptive_saturated",
                        level=level,
                        element_type=element_type,
                        max_element_order=max_p,
                    )
                    break
                element_type = cast("Literal['P1', 'P2', 'P3']", f"P{current_p + 1}")
                element = _make_element(element_type, skfem)
            elif self.config.refinement_strategy == "hp_adaptive":
                smooth = self._estimate_smoothness(indicators, marked)
                current_p = _order_of(element_type)
                if smooth > self.config.smoothness_threshold and current_p < max_p:
                    element_type = cast("Literal['P1', 'P2', 'P3']", f"P{current_p + 1}")
                    element = _make_element(element_type, skfem)
                else:
                    mesh = mesh.refined(np.where(marked)[0])
            else:  # h_adaptive
                mesh = mesh.refined(np.where(marked)[0])

        wall_time = time.perf_counter() - t0

        if last_solution is None or last_coords is None:
            raise RuntimeError("FEM solver produced no solution")

        log.info(
            "fem_solve_done",
            wall_time=wall_time,
            l2_error=last_l2_err,
            n_dof=len(last_solution),
            levels=levels_used,
        )

        return SolverResult(
            solution=last_solution,
            grid_points=last_coords,
            n_dof=len(last_solution),
            wall_time_seconds=wall_time,
            l2_error=last_l2_err,
            metadata={
                "method": "scikit_fem_hp_adaptive",
                "element_type": element_type,
                "strategy": self.config.refinement_strategy,
                "refinement_levels": levels_used,
                "marking_fraction": self.config.marking_fraction,
                "n_elements": int(mesh.t.shape[1]) if hasattr(mesh, "t") else None,
            },
        )

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------
    def _build_initial_mesh(self, operator: PDEOperator, n_dof: int, skfem: Any) -> Any:
        """Construct an initial rectangular mesh sized to the hint n_dof.

        Delegates to the module-level ``build_initial_mesh``, shared with
        ``SkfemTriSubstrate``.
        """
        return build_initial_mesh(
            operator,
            n_dof,
            skfem,
            min_initial_dof_hint=self.config.min_initial_dof_hint,
            min_mesh_side=self.config.min_mesh_side,
            initial_mesh_refinements=self.config.initial_mesh_refinements,
        )

    # ------------------------------------------------------------------
    # Assembly and solve
    # ------------------------------------------------------------------
    def _assemble_and_solve(
        self,
        mesh: Any,
        element: Any,
        operator: PDEOperator,
        skfem: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float | None]:
        """Assemble the Poisson system and solve it.

        Delegates to the module-level ``assemble_and_solve``, injecting
        ``self._compute_l2_error`` (inherited from ``BaseSolver``) so
        ``SolverResult.l2_error``'s nodal-RMS meaning is unchanged.
        """
        return assemble_and_solve(mesh, element, operator, skfem, self._compute_l2_error)

    # ------------------------------------------------------------------
    # Zienkiewicz-Zhu error estimator
    # ------------------------------------------------------------------
    def _compute_zz_indicator(
        self,
        mesh: Any,
        element: Any,
        solution: NDArray[np.float64],
        skfem: Any,
    ) -> NDArray[np.float64]:
        """Compute per-element ZZ recovered-gradient error indicators.

        Delegates to the module-level ``zz_indicator``. ``element``/``skfem``
        are accepted for backwards compatibility but unused, exactly as
        before this extraction.
        """
        del element, skfem
        return zz_indicator(mesh, solution)

    @staticmethod
    def _element_gradients(mesh: Any, nodal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute constant gradient per P1 triangle element."""
        return element_gradients(mesh, nodal)

    @staticmethod
    def _triangle_area(pts: NDArray[np.float64]) -> float:
        """Area of a triangle given (2, 3) vertex coordinates."""
        return triangle_area(pts)

    # ------------------------------------------------------------------
    # Dorfler marking
    # ------------------------------------------------------------------
    def _dorfler_mark(self, indicators: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Select the smallest subset whose indicators sum to theta * total.

        Delegates to ``src.research.marking.dorfler_mark`` (variant="linear"),
        the shared primitive that also backs ``DorflerAMRSolver._dorfler_mark``
        and any ``RefinementSubstrate``.
        """
        return dorfler_mark(indicators, self.config.marking_fraction, variant="linear")

    def _estimate_smoothness(
        self,
        indicators: NDArray[np.float64],
        marked: NDArray[np.bool_],
    ) -> float:
        """Rough smoothness proxy: ratio of marked-region mean to max.

        Delegates to the module-level ``estimate_smoothness``.
        """
        return estimate_smoothness(indicators, marked, self.config.zz_epsilon)


class ScikitFEMLShapedSolver(ScikitFEMPoissonSolver):
    """hp-adaptive FEM specialized for the L-shaped domain.

    The L-shaped geometry exposes a re-entrant corner at the origin with
    a ``r^(2/3) sin(2θ/3)`` singularity, which is the canonical stress
    test for adaptive finite-element methods. Uniform refinement converges
    at the reduced rate O(h^(2/3)); adaptive refinement recovers the
    optimal O(h^p) rate.
    """

    name = "scikit_fem_lshaped"
    description = "hp-adaptive FEM (scikit-fem) on L-shaped domain"

    def _build_initial_mesh(self, operator: PDEOperator, n_dof: int, skfem: Any) -> Any:
        r"""Build an L-shaped mesh by assembling three unit squares.

        The L-shaped domain is defined as ``[-1,1]^2 \ [0,1]x[-1,0]``.
        Delegates to the module-level ``build_lshaped_initial_mesh``, shared
        with ``SkfemTriSubstrate``. ``n_dof`` is accepted for the same
        signature as the base class's ``_build_initial_mesh`` but unused,
        exactly as before this extraction.
        """
        del n_dof
        return build_lshaped_initial_mesh(
            operator, skfem, initial_mesh_refinements=self.config.initial_mesh_refinements
        )


def _order_of(element_type: str) -> int:
    return int(element_type[1:])


# ---------------------------------------------------------------------------
# Registry injection
# ---------------------------------------------------------------------------
# Use setdefault to keep registration idempotent: a second import of this
# module (or a user-supplied override registered earlier) will not be
# silently overwritten.  Matches the pattern in src/alphagalerkin/solver.py.
SOLVER_REGISTRY.setdefault("scikit_fem_poisson", ScikitFEMPoissonSolver)
SOLVER_REGISTRY.setdefault("scikit_fem_lshaped", ScikitFEMLShapedSolver)
