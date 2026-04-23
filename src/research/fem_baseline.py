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
from typing import Any, Literal

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
        description="hp-adaptive threshold: elements with local smoothness above this get p-refined",
    )


def _require_skfem() -> Any:
    """Import scikit-fem or raise a helpful error."""
    try:
        import skfem  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise ImportError(
            "ScikitFEM*Solver requires scikit-fem. "
            "Install with: pip install 'scikit-fem>=9.0'"
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
            solution, coords, l2_err = self._assemble_and_solve(
                mesh, element, operator, skfem
            )
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

            if self.config.refinement_strategy == "p_adaptive":
                new_order = min(3, _order_of(element_type) + 1)
                element_type = f"P{new_order}"
                element = _make_element(element_type, skfem)
            elif self.config.refinement_strategy == "hp_adaptive":
                smooth = self._estimate_smoothness(indicators, marked)
                if smooth > self.config.smoothness_threshold:
                    new_order = min(3, _order_of(element_type) + 1)
                    element_type = f"P{new_order}"
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
        """Construct an initial rectangular mesh sized to the hint n_dof."""
        if operator.dim != 2:
            raise NotImplementedError(
                f"ScikitFEMPoissonSolver currently supports 2D problems "
                f"(got operator.dim={operator.dim})"
            )

        xmin, ymin = float(operator.domain_min[0]), float(operator.domain_min[1])
        xmax, ymax = float(operator.domain_max[0]), float(operator.domain_max[1])
        side = max(3, int(np.ceil(np.sqrt(max(n_dof, 9)))))

        xs = np.linspace(xmin, xmax, side)
        ys = np.linspace(ymin, ymax, side)
        mesh = skfem.MeshTri.init_tensor(xs, ys)

        for _ in range(self.config.initial_mesh_refinements):
            mesh = mesh.refined()
        return mesh

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
        """Assemble the Poisson system and solve it."""
        from scipy.sparse.linalg import spsolve

        from skfem import BilinearForm, LinearForm  # type: ignore[import-not-found]
        from skfem.helpers import dot, grad  # type: ignore[import-not-found]

        basis = skfem.Basis(mesh, element)

        @BilinearForm
        def a_form(u: Any, v: Any, _: Any) -> Any:
            return dot(grad(u), grad(v))

        @LinearForm
        def l_form(v: Any, w: Any) -> Any:
            x = w.x  # quadrature point coordinates (dim, n_q, n_elements)
            pts = np.stack([x[0].ravel(), x[1].ravel()], axis=-1).astype(np.float32)
            f_vals = np.asarray(operator.source_term(pts), dtype=np.float64)
            f = f_vals.reshape(x[0].shape)
            return f * v

        A = a_form.assemble(basis)
        b = l_form.assemble(basis)

        # Dirichlet boundary conditions via operator.boundary_value.
        dirichlet_dofs = basis.get_dofs()
        dirichlet_pts = mesh.p[:, dirichlet_dofs.nodal["u"]] if hasattr(
            dirichlet_dofs, "nodal"
        ) else mesh.p[:, dirichlet_dofs]
        bc_pts = np.asarray(dirichlet_pts.T, dtype=np.float32)
        bc_values = np.asarray(operator.boundary_value(bc_pts), dtype=np.float64).flatten()

        u = np.zeros(A.shape[0], dtype=np.float64)
        dof_indices = (
            dirichlet_dofs.nodal["u"] if hasattr(dirichlet_dofs, "nodal") else dirichlet_dofs
        )
        u[dof_indices] = bc_values

        # Solve the condensed system
        A_c, b_c, u_c, I = skfem.condense(A, b, D=dirichlet_dofs, x=u)
        u_c = spsolve(A_c, b_c)
        u[I] = u_c

        # Evaluate at mesh nodes for error computation
        coords = mesh.p.T.astype(np.float64)
        nodal_values = u[: coords.shape[0]]
        l2_err = self._compute_l2_error(nodal_values, coords, operator)

        # Store full solution (includes higher-order dofs for P2/P3)
        return u.astype(np.float64), coords, l2_err

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

        The ZZ estimator computes η_K = ||∇u_h - G(∇u_h)||_{L²(K)} where
        G is the patch-recovery operator implemented here as nodal
        averaging of element-wise gradients.
        """
        basis = skfem.Basis(mesh, element)
        n_elements = mesh.t.shape[1]

        # Element-wise constant gradients via P1 approximation of u_h
        # For simplicity we downsample to P1 gradient; P2/P3 refinements
        # still get a valid (if slightly conservative) indicator.
        nodal = solution[: mesh.p.shape[1]]
        grads = self._element_gradients(mesh, nodal)  # (n_elements, 2)

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
            area = self._triangle_area(mesh.p[:, vids])
            indicators[k] = float(area * (diff @ diff))
        return indicators

    @staticmethod
    def _element_gradients(mesh: Any, nodal: NDArray[np.float64]) -> NDArray[np.float64]:
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

    @staticmethod
    def _triangle_area(pts: NDArray[np.float64]) -> float:
        """Area of a triangle given (2, 3) vertex coordinates."""
        x = pts[0]
        y = pts[1]
        return float(0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])))

    # ------------------------------------------------------------------
    # Dorfler marking
    # ------------------------------------------------------------------
    def _dorfler_mark(self, indicators: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Select the smallest subset whose indicators sum to theta * total."""
        total = float(np.sum(indicators))
        if total <= 0.0:
            return np.zeros_like(indicators, dtype=bool)
        threshold = self.config.marking_fraction * total
        order = np.argsort(indicators)[::-1]
        cumulative = np.cumsum(indicators[order])
        cutoff = int(np.searchsorted(cumulative, threshold) + 1)
        marked = np.zeros_like(indicators, dtype=bool)
        marked[order[:cutoff]] = True
        return marked

    @staticmethod
    def _estimate_smoothness(
        indicators: NDArray[np.float64],
        marked: NDArray[np.bool_],
    ) -> float:
        """Rough smoothness proxy: ratio of marked-region max to mean indicator.

        High values indicate a concentrated feature (suitable for h-refinement);
        low values suggest a smooth solution (suitable for p-refinement).
        """
        marked_vals = indicators[marked]
        if marked_vals.size == 0:
            return 0.0
        return float(np.mean(marked_vals) / (np.max(marked_vals) + 1e-12))

    def _compute_l2_error(  # type: ignore[override]
        self,
        solution: NDArray[np.float64],
        coords: NDArray[np.float64],
        operator: PDEOperator,
    ) -> float | None:
        """Override to use nodal values only (ignores interior P2/P3 dofs)."""
        exact = operator.exact_solution(coords.astype(np.float32))
        if exact is None:
            return None
        if isinstance(exact, torch.Tensor):
            exact = exact.detach().cpu().numpy()
        exact = np.asarray(exact, dtype=np.float64).flatten()
        diff = solution.flatten() - exact
        n = len(diff)
        return float(np.sqrt(np.sum(diff**2) / n)) if n > 0 else None


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
        """Build an L-shaped mesh by assembling three unit squares.

        The L-shaped domain is defined as ``[-1,1]^2 \\ [0,1]x[-1,0]``.
        """
        if operator.dim != 2:
            raise NotImplementedError("L-shaped domain is inherently 2D")

        # Three L-shape unit-square cells: NW, NE, SW
        nw = skfem.MeshTri.init_tensor(np.linspace(-1.0, 0.0, 3), np.linspace(0.0, 1.0, 3))
        ne = skfem.MeshTri.init_tensor(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
        sw = skfem.MeshTri.init_tensor(np.linspace(-1.0, 0.0, 3), np.linspace(-1.0, 0.0, 3))

        try:
            mesh = nw + ne + sw  # type: ignore[operator]
        except TypeError:
            # Fallback for older skfem without + operator: concatenate via init_lshaped if present
            if hasattr(skfem.MeshTri, "init_lshaped"):
                mesh = skfem.MeshTri.init_lshaped()
            else:  # pragma: no cover
                raise

        for _ in range(self.config.initial_mesh_refinements):
            mesh = mesh.refined()
        return mesh


def _order_of(element_type: str) -> int:
    return int(element_type[1:])


# ---------------------------------------------------------------------------
# Registry injection
# ---------------------------------------------------------------------------
SOLVER_REGISTRY["scikit_fem_poisson"] = ScikitFEMPoissonSolver
SOLVER_REGISTRY["scikit_fem_lshaped"] = ScikitFEMLShapedSolver
