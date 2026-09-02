"""Tensor-product-grid ``RefinementSubstrate`` -- the back-compat control.

Wraps ``DorflerAMRSolver``'s existing static solve/refine primitives
(``src.research.baselines``) and the L-shape harness's area-weighted L2 norm
(``src.research.lshape_amr_compare``) behind the ``RefinementSubstrate``
interface, reproducing today's ``run_dorfler_arm`` trajectory byte-for-byte
(see ``tests/research/test_tensor_grid_substrate.py``). This is the substrate
whose tensor-product refinement inserts full grid lines per marked element --
the defect the ``SkfemTriSubstrate`` (element-local) substrate exists to fix --
kept exactly as-is here so the adequacy gate (Slice D) has something that must
fail its assertion, proving the gate discriminates.

``mark()``/``refine()`` split ``DorflerAMRSolver._dorfler_mark_2d``'s single
call (selection + x/y-axis projection) into two ``RefinementSubstrate``
primitives: ``mark()`` returns the shared, protocol-compliant flat element
selection via ``src.research.marking.dorfler_mark`` (variant="squared"),
which is the identical squared-bulk-chasing computation
``_dorfler_mark_2d`` performs internally before projecting; ``refine()``
does the axis projection (any marked element on a row/column marks that
axis) plus ``DorflerAMRSolver._refine_grid`` (verbatim, unmodified). This
composition is *not* asserted correct by inspection alone -- the golden test
drives the full ``initial_mesh -> solve -> mark -> refine`` loop and checks
its trajectory against a live ``run_dorfler_arm`` call, which is what
actually proves the split reproduces the fused legacy behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.refinement.substrate import SubstrateSolveResult
from src.research.baselines import DorflerAMRSolver
from src.research.lshape_amr_compare import _area_weighted_l2
from src.research.marking import dorfler_mark
from src.research.substrates.config import SubstrateConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from src.pde.operators import PDEOperator


@dataclass(frozen=True)
class TensorGridMesh:
    """A tensor-product grid: independent node coordinates per axis."""

    xs: NDArray[np.float64]
    ys: NDArray[np.float64]


class TensorGridSubstrate:
    """``RefinementSubstrate`` over a 2D tensor-product grid (the legacy control).

    Does not inherit from ``RefinementSubstrate`` -- Protocols are structural.
    """

    def __init__(
        self,
        operator: PDEOperator,
        inside: Callable[[NDArray[np.float64]], NDArray[np.bool_]] | None = None,
        config: SubstrateConfig | None = None,
    ) -> None:
        """Construct a substrate over ``operator``.

        Args:
            operator: The PDE operator to solve (must implement ``domain_min``/
                ``domain_max``/``source_term``/``boundary_value``/``exact_solution``).
            inside: Optional geometry predicate for a non-rectangular domain
                (e.g. the L-shape). ``None`` solves the full bounding box.
            config: ``SubstrateConfig``; ``initial_side`` drives the coarse grid.

        """
        try:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "TensorGridSubstrate requires scipy. Install with: pip install scipy"
            ) from exc

        self._operator = operator
        self._inside = inside
        self._config = config or SubstrateConfig(name="tensor_grid_substrate", kind="tensor_grid")
        self._sparse = sparse
        self._spsolve = spsolve

    def initial_mesh(self) -> TensorGridMesh:
        """Coarse grid matching ``run_dorfler_arm``'s own construction."""
        domain_min = np.asarray(self._operator.domain_min, dtype=np.float64)
        domain_max = np.asarray(self._operator.domain_max, dtype=np.float64)
        n = self._config.initial_side + 1
        return TensorGridMesh(
            xs=np.linspace(float(domain_min[0]), float(domain_max[0]), n, dtype=np.float64),
            ys=np.linspace(float(domain_min[1]), float(domain_max[1]), n, dtype=np.float64),
        )

    def solve(self, mesh: TensorGridMesh) -> SubstrateSolveResult:
        """Solve on ``mesh`` via the verbatim static primitives."""
        u_full, grid = DorflerAMRSolver._solve_on_grid_2d(
            mesh.xs, mesh.ys, self._operator, self._sparse, self._spsolve, inside=self._inside
        )
        indicators_2d = DorflerAMRSolver._compute_indicators_2d(
            mesh.xs, mesh.ys, u_full, self._operator, inside=self._inside
        )
        if self._inside is not None:
            in_mask = np.asarray(self._inside(grid), dtype=bool)
        else:
            in_mask = np.ones(len(grid), dtype=bool)

        exact = np.asarray(
            self._operator.exact_solution(grid.astype(np.float32)), dtype=np.float64
        ).ravel()
        diff = (u_full.ravel() - exact)[in_mask]
        l2_error = _area_weighted_l2(diff, mesh.xs, mesh.ys, in_mask)

        n_dof = int(in_mask.sum())
        n_dof_free = self._n_dof_free(mesh, in_mask)

        return SubstrateSolveResult(
            values=u_full,
            indicators=indicators_2d.ravel(),
            l2_error=l2_error,
            n_dof=n_dof,
            n_dof_free=n_dof_free,
            extra={},
        )

    @staticmethod
    def _n_dof_free(mesh: TensorGridMesh, in_mask: NDArray[np.bool_]) -> int:
        """In-domain *interior* nodes: the truly free unknowns (AC5: <= n_dof)."""
        nx_nodes, ny_nodes = len(mesh.xs), len(mesh.ys)
        is_boundary = np.zeros((nx_nodes, ny_nodes), dtype=bool)
        is_boundary[0, :] = True
        is_boundary[-1, :] = True
        is_boundary[:, 0] = True
        is_boundary[:, -1] = True
        interior_mask = ~is_boundary.ravel()
        return int((in_mask & interior_mask).sum())

    def mark(self, indicators: NDArray[np.float64], theta: float) -> NDArray[np.bool_]:
        """Flat element selection -- the shared primitive, squared-bulk variant."""
        return dorfler_mark(indicators, theta, variant="squared")

    def refine(self, mesh: TensorGridMesh, marked: NDArray[np.bool_]) -> TensorGridMesh:
        """Axis-project ``marked`` and insert grid lines via ``_refine_grid``."""
        nx = len(mesh.xs) - 1
        ny = len(mesh.ys) - 1
        marked_grid = marked.reshape(nx, ny)
        marked_x = np.asarray(marked_grid.any(axis=1), dtype=bool)
        marked_y = np.asarray(marked_grid.any(axis=0), dtype=bool)
        new_xs = DorflerAMRSolver._refine_grid(mesh.xs, marked_x)
        new_ys = DorflerAMRSolver._refine_grid(mesh.ys, marked_y)
        return TensorGridMesh(xs=new_xs, ys=new_ys)

    def n_units(self, mesh: TensorGridMesh) -> int:
        return (len(mesh.xs) - 1) * (len(mesh.ys) - 1)

    def refinable_mask(self, mesh: TensorGridMesh) -> NDArray[np.bool_]:
        """Every element is refinable (no geometry-aware exclusion at this layer)."""
        return np.ones(self.n_units(mesh), dtype=bool)

    def fingerprint(self, mesh: TensorGridMesh) -> bytes:
        return mesh.xs.tobytes() + b"|" + mesh.ys.tobytes()

    def describe(self) -> dict[str, str | int | float]:
        return {
            "kind": "tensor_grid",
            "dof_convention": "in_domain_grid_nodes",
            "initial_side": self._config.initial_side,
        }
