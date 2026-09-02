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
import structlog

from src.refinement.substrate import SubstrateSolveResult
from src.refinement.substrate_registry import register_refinement_substrate
from src.research.baselines import (
    DorflerAMRSolver,
    element_inside_mask,
    nodal_rms_l2_error,
    require_exact_solution,
    require_measurable_l2,
)
from src.research.lshape_amr_compare import area_weighted_l2
from src.research.marking import dorfler_mark
from src.research.substrates.config import (
    ERROR_METRIC_QUADRATURE,
    SUBSTRATE_AREA_WEIGHTED_L2_KEY,
    SUBSTRATE_KIND_TENSOR_GRID,
    SUBSTRATE_NODAL_RMS_L2_KEY,
    SUBSTRATE_PRIMARY_L2_KEY,
    SubstrateConfig,
    resolve_substrate_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TensorGridMesh:
    """A tensor-product grid: independent node coordinates per axis."""

    xs: NDArray[np.float64]
    ys: NDArray[np.float64]


@register_refinement_substrate(SUBSTRATE_KIND_TENSOR_GRID)
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
            inside: Optional **interior-unknown** predicate for a non-rectangular
                domain — "is this point an unknown the solver must solve for?"
                — **not** a closed-domain membership test. ``None`` solves the
                full bounding box.

                The distinction is not pedantry, and calling this a "geometry
                predicate" (as this docstring previously did) invites exactly
                the wrong one. The two answers differ precisely on the L-shape's
                reentrant edges: ``LShapedDomain.contains_point`` removes the
                *open* quadrant, so slit-edge points are members of the closed
                domain, while ``lshape_inside_predicate`` removes the *closed*
                quadrant so those nodes are pinned Dirichlet. Passing the
                membership flavour gives slit-edge nodes a full Laplacian row,
                their ``u = 0`` condition is never imposed, and the stencil
                couples across the slit — discretising a different, inconsistent
                problem on which the L2 error **grows** with DOF instead of
                converging (5.0e-2 at 65 DOF to 1.15e-1 at 12545). That is the
                defect behind the 2026-08-16 retraction, and it is silent: every
                downstream ratio stays a finite, plausible number.

                Use ``src.research.lshape_amr_compare.lshape_inside_predicate``.
                The contract is guarded by
                ``tests/research/test_lshape_convergence_gate.py::TestReentrantEdgesArePinned``.
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
        self._config = resolve_substrate_config(
            config, kind=SUBSTRATE_KIND_TENSOR_GRID, default_name="tensor_grid_substrate"
        )
        self._sparse = sparse
        self._spsolve = spsolve
        require_exact_solution(operator, "TensorGridSubstrate")
        self._log = logger.bind(**self.describe())
        self._log.info("substrate_initialised", has_geometry_predicate=inside is not None)

    def initial_mesh(self) -> TensorGridMesh:
        """Coarse grid matching ``run_dorfler_arm``'s own construction."""
        domain_min = np.asarray(self._operator.domain_min, dtype=np.float64)
        domain_max = np.asarray(self._operator.domain_max, dtype=np.float64)
        n = self._config.initial_side + 1
        mesh = TensorGridMesh(
            xs=np.linspace(float(domain_min[0]), float(domain_max[0]), n, dtype=np.float64),
            ys=np.linspace(float(domain_min[1]), float(domain_max[1]), n, dtype=np.float64),
        )
        self._log.debug("substrate_initial_mesh", n_units=self.n_units(mesh))
        return self._maybe_freeze(mesh)

    def _maybe_freeze(self, mesh: TensorGridMesh) -> TensorGridMesh:
        """Clear numpy write flags on the axis arrays (AC3), behind the config flag.

        ``TensorGridMesh`` is a frozen dataclass, but freezing a dataclass only
        stops rebinding its *fields* -- ``mesh.xs[0] = 3.0`` still mutates the
        array in place. This is the tensor-grid counterpart of
        ``SkfemTriSubstrate._maybe_freeze``; before it existed,
        ``enforce_immutable_meshes`` (default ``True``) was declared, validated,
        and read by exactly one of the two substrates.
        """
        if self._config.enforce_immutable_meshes:
            mesh.xs.flags.writeable = False
            mesh.ys.flags.writeable = False
        return mesh

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
        area_weighted = area_weighted_l2(diff, mesh.xs, mesh.ys, in_mask)
        # The unweighted counterpart, so `error_metric` is a real choice here.
        # Computed over in-domain nodes only, matching the area-weighted norm's
        # support. NOTE: this is NOT the same pair SkfemTriSubstrate reports --
        # its quadrature-metric key is `l2_error_quadrature`, ours is
        # `l2_error_area_weighted`, because an area-weighted nodal norm is not a
        # quadrature form. A comment here used to claim they matched; they never
        # did, which is why SUBSTRATE_PRIMARY_L2_KEY exists (D4).
        nodal_rms = nodal_rms_l2_error(u_full.ravel()[in_mask], grid[in_mask], self._operator)
        nodal_rms_value = require_measurable_l2(nodal_rms, "TensorGridSubstrate")

        l2_error = (
            area_weighted
            if self._config.error_metric == ERROR_METRIC_QUADRATURE
            else nodal_rms_value
        )
        n_dof = int(in_mask.sum())
        n_dof_free = self._n_dof_free(mesh, in_mask)

        self._log.info(
            "substrate_solve",
            n_dof=n_dof,
            n_dof_free=n_dof_free,
            n_units=self.n_units(mesh),
            l2_primary=l2_error,
            l2_area_weighted=area_weighted,
            l2_nodal_rms=nodal_rms_value,
        )
        return SubstrateSolveResult(
            values=u_full,
            indicators=indicators_2d.ravel(),
            l2_error=l2_error,
            n_dof=n_dof,
            n_dof_free=n_dof_free,
            extra={
                SUBSTRATE_PRIMARY_L2_KEY: l2_error,
                SUBSTRATE_AREA_WEIGHTED_L2_KEY: area_weighted,
                SUBSTRATE_NODAL_RMS_L2_KEY: nodal_rms_value,
            },
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
        """Flat element selection -- the shared primitive.

        The variant comes from the config (defaulting to ``"squared"``, the
        legacy behaviour the golden test pins) rather than being hardcoded,
        so ``marking_variant`` is a knob rather than decoration.
        """
        return dorfler_mark(indicators, theta, variant=self._config.marking_variant)

    def refine(self, mesh: TensorGridMesh, marked: NDArray[np.bool_]) -> TensorGridMesh:
        """Axis-project ``marked`` and insert grid lines via ``_refine_grid``.

        Warns on an empty selection for the same reason ``SkfemTriSubstrate``
        does: ``marking_variant="linear"`` returns all-False on an all-zero
        indicator array, and a DOF-growth loop would then spin silently. Both
        substrates read the same ``marking_variant`` field and are driven by the
        same ``run_refinement_sweep`` loop, but only the skfem side warned --
        so the identical failure was observable on one substrate and invisible
        on the other.
        """
        if not marked.any():
            self._log.warning("substrate_refine_noop", n_units=self.n_units(mesh))
        nx = len(mesh.xs) - 1
        ny = len(mesh.ys) - 1
        marked_grid = marked.reshape(nx, ny)
        marked_x = np.asarray(marked_grid.any(axis=1), dtype=bool)
        marked_y = np.asarray(marked_grid.any(axis=0), dtype=bool)
        new_xs = DorflerAMRSolver._refine_grid(mesh.xs, marked_x)
        new_ys = DorflerAMRSolver._refine_grid(mesh.ys, marked_y)
        refined = TensorGridMesh(xs=new_xs, ys=new_ys)
        # n_marked vs the resulting element growth is the tensor-product defect
        # made visible: marking a handful of elements inserts whole grid lines,
        # so n_units_after is driven by the axis projection, not by |M|.
        self._log.debug(
            "substrate_refine",
            n_marked=int(np.count_nonzero(marked)),
            n_axes_marked_x=int(np.count_nonzero(marked_x)),
            n_axes_marked_y=int(np.count_nonzero(marked_y)),
            n_units_before=self.n_units(mesh),
            n_units_after=self.n_units(refined),
        )
        return self._maybe_freeze(refined)

    def n_units(self, mesh: TensorGridMesh) -> int:
        return (len(mesh.xs) - 1) * (len(mesh.ys) - 1)

    def refinable_mask(self, mesh: TensorGridMesh) -> NDArray[np.bool_]:
        """Elements inside the physical domain; all of them when unmasked.

        Must agree with the estimator, not merely be permissive: with a
        geometry predicate (the L-shape notch), `_compute_indicators_2d`
        forces out-of-domain elements to a **zero** indicator, so they can
        never be marked. Reporting them as refinable was a claim the estimator
        contradicts — and it is read, not decorative: a uniform sweep marks
        exactly this mask (`src/research/substrates/sweep.py`).

        Shares `element_inside_mask` with the estimator rather than
        re-deriving the centre test, so the two cannot drift apart.
        """
        if self._inside is None:
            return np.ones(self.n_units(mesh), dtype=bool)
        return np.asarray(element_inside_mask(mesh.xs, mesh.ys, self._inside).ravel(), dtype=bool)

    def fingerprint(self, mesh: TensorGridMesh) -> bytes:
        return mesh.xs.tobytes() + b"|" + mesh.ys.tobytes()

    def describe(self) -> dict[str, str | int | float]:
        return {
            "kind": self._config.kind,
            "dof_convention": "in_domain_grid_nodes",
            "initial_side": self._config.initial_side,
        }
