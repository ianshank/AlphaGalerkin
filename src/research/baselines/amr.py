"""Dörfler-marking adaptive mesh refinement baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from src.pde.operators import PDEOperator
from src.research.baselines.base import BaseSolver
from src.research.baselines.predicates import InsidePredicate, element_inside_mask
from src.research.baselines.schemas import AMRConfig, SolverResult
from src.research.marking import dorfler_mark

logger = structlog.get_logger(__name__)


class DorflerAMRSolver(BaseSolver):
    """Dorfler marking adaptive mesh refinement.

    Uses residual-based error indicators to mark elements for
    refinement following the Dorfler bulk-chasing strategy.
    Implemented with scipy.sparse on simplex/rectangular meshes.
    """

    name = "dorfler_amr"
    description = "Dorfler marking adaptive mesh refinement"

    def __init__(
        self,
        marking_fraction: float | None = None,
        max_refinements: int | None = None,
        config: AMRConfig | None = None,
    ) -> None:
        self.config = config or AMRConfig()
        self.marking_fraction = (
            marking_fraction if marking_fraction is not None else self.config.marking_fraction
        )
        self.max_refinements = (
            max_refinements if max_refinements is not None else self.config.max_refinements
        )
        if not 0.0 < self.marking_fraction < 1.0:
            raise ValueError(f"marking_fraction must be in (0,1), got {self.marking_fraction}")

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve with adaptive refinement via Dorfler marking.

        Starts with a coarse uniform grid and refines cells where the
        residual-based error indicator is largest, until n_dof is reached
        or max_refinements is exhausted.
        """
        try:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
        except ImportError as exc:
            raise ImportError(
                "DorflerAMRSolver requires scipy. Install with: pip install scipy"
            ) from exc

        if operator.dim not in (1, 2):
            raise NotImplementedError(
                f"DorflerAMRSolver supports dim=1 and dim=2, got dim={operator.dim}"
            )

        log = logger.bind(solver=self.name, n_dof=n_dof, dim=operator.dim)
        log.info("amr_solve_start")
        t0 = time.perf_counter()

        if operator.dim == 1:
            result = self._solve_amr_1d(operator, n_dof, sparse, spsolve, log)
        else:
            result = self._solve_amr_2d(operator, n_dof, sparse, spsolve, log)

        wall_time = time.perf_counter() - t0
        solution, grid, n_refinements = result
        l2_err = self._compute_l2_error(solution, grid, operator)

        log.info("amr_solve_done", wall_time=wall_time, l2_error=l2_err, final_dof=len(solution))
        return SolverResult(
            solution=solution,
            grid_points=grid,
            n_dof=len(solution),
            wall_time_seconds=wall_time,
            l2_error=l2_err,
            metadata={
                "marking_fraction": self.marking_fraction,
                "n_refinements": n_refinements,
                "dim": operator.dim,
            },
        )

    def _solve_amr_1d(
        self,
        operator: PDEOperator,
        n_dof: int,
        sparse: Any,
        spsolve: Any,
        log: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Run 1D AMR loop. Returns (solution, grid, n_refinements)."""
        n_start = max(
            min(n_dof // self.config.initial_dof_divisor, self.config.max_initial_points_1d),
            self.config.min_initial_points,
        )
        x = np.linspace(
            float(operator.domain_min[0]),
            float(operator.domain_max[0]),
            n_start,
            dtype=np.float64,
        )

        step = 0
        for step in range(self.max_refinements):
            u, _ = self._solve_on_grid(x, operator, sparse, spsolve)
            if len(x) >= n_dof:
                break
            indicators = self._compute_indicators(x, u, operator)
            marked = self._dorfler_mark(indicators)
            x = self._refine_grid(x, marked)
            log.debug(
                "amr_step",
                step=step,
                n_points=len(x),
                max_indicator=float(np.max(indicators)),
            )

        u, _ = self._solve_on_grid(x, operator, sparse, spsolve)
        grid = x.reshape(-1, 1).astype(np.float64)
        return u, grid, step + 1

    def _solve_amr_2d(
        self,
        operator: PDEOperator,
        n_dof: int,
        sparse: Any,
        spsolve: Any,
        log: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Run 2D AMR loop with element-wise refinement.

        Uses a quadrilateral mesh with element-wise bisection.
        Returns (solution, grid, n_refinements).
        """
        # Start with coarse grid
        n_side = max(
            int(np.sqrt(n_dof)) // self.config.initial_side_divisor_2d,
            self.config.min_initial_side_2d,
        )
        x_lo, x_hi = float(operator.domain_min[0]), float(operator.domain_max[0])
        y_lo, y_hi = float(operator.domain_min[1]), float(operator.domain_max[1])
        xs = np.linspace(x_lo, x_hi, n_side + 1, dtype=np.float64)
        ys = np.linspace(y_lo, y_hi, n_side + 1, dtype=np.float64)

        step = 0
        for step in range(self.max_refinements):
            u, grid = self._solve_on_grid_2d(xs, ys, operator, sparse, spsolve)
            current_dof = len(xs) * len(ys)
            if current_dof >= n_dof:
                break

            indicators = self._compute_indicators_2d(xs, ys, u, operator)
            marked_x, marked_y = self._dorfler_mark_2d(indicators, xs, ys)

            xs = self._refine_grid(xs, marked_x)
            ys = self._refine_grid(ys, marked_y)

            log.debug(
                "amr_step_2d",
                step=step,
                n_x=len(xs),
                n_y=len(ys),
                n_dof=len(xs) * len(ys),
                max_indicator=float(np.max(indicators)),
            )

        u, grid = self._solve_on_grid_2d(xs, ys, operator, sparse, spsolve)
        return u, grid, step + 1

    # ------------------------------------------------------------------
    @staticmethod
    def _solve_on_grid(
        x: NDArray[np.float64],
        operator: PDEOperator,
        sparse: Any,
        spsolve: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve 1D Poisson on a (possibly non-uniform) grid."""
        n = len(x)
        interior = x[1:-1]
        h_left = np.diff(x[:-1])  # h_{i-1}
        h_right = np.diff(x[1:])  # h_i

        # Variable-coefficient FD stencil on non-uniform grid
        a_left = 2.0 / (h_left * (h_left + h_right))
        a_right = 2.0 / (h_right * (h_left + h_right))
        a_center = a_left + a_right

        ni = n - 2
        A = sparse.diags(
            [
                -a_left[1:],
                a_center,
                -a_right[:-1],
            ],
            [-1, 0, 1],
            shape=(ni, ni),
            format="csc",
        )

        coords = interior.reshape(-1, 1).astype(np.float32)
        f = np.asarray(operator.source_term(coords), dtype=np.float64).flatten()

        bc_l = float(
            np.asarray(operator.boundary_value(np.array([[x[0]]], dtype=np.float32))).flat[0]
        )
        bc_r = float(
            np.asarray(operator.boundary_value(np.array([[x[-1]]], dtype=np.float32))).flat[0]
        )

        rhs = f.copy()
        rhs[0] += a_left[0] * bc_l
        rhs[-1] += a_right[-1] * bc_r

        u_inner = spsolve(A, rhs)
        u_full = np.concatenate([[bc_l], u_inner, [bc_r]])
        return u_full.astype(np.float64), x

    def _compute_indicators(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        operator: PDEOperator,
    ) -> NDArray[np.float64]:
        """Compute residual-based error indicators per element."""
        n_elem = len(x) - 1
        indicators = np.zeros(n_elem, dtype=np.float64)

        for i in range(n_elem):
            h = x[i + 1] - x[i]
            mid = np.array([[(x[i] + x[i + 1]) / 2]], dtype=np.float32)
            f_mid = float(np.asarray(operator.source_term(mid)).flat[0])

            # Approximate second derivative at midpoint
            if 0 < i < n_elem - 1:
                u_xx = (u[i] - 2 * u[i + 1] + u[i + 2]) / (h**2) if h > 0 else 0.0
            else:
                u_xx = 0.0

            residual = abs(-u_xx - f_mid) if not np.isnan(u_xx) else 0.0
            indicators[i] = h * residual

        return indicators

    def _dorfler_mark(self, indicators: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Mark elements using Dorfler bulk-chasing strategy.

        Delegates to ``src.research.marking.dorfler_mark`` (variant="squared"),
        the shared primitive that also backs ``ScikitFEMPoissonSolver._dorfler_mark``
        and any ``RefinementSubstrate``.
        """
        return dorfler_mark(indicators, self.marking_fraction, variant="squared")

    @staticmethod
    def _refine_grid(
        x: NDArray[np.float64],
        marked: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Refine marked elements by bisection."""
        new_points = []
        for i, is_marked in enumerate(marked):
            if is_marked:
                new_points.append((x[i] + x[i + 1]) / 2)
        if new_points:
            x = np.sort(np.concatenate([x, new_points]))
        return x

    # ------------------------------------------------------------------
    # 2D AMR helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_on_grid_2d(
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        operator: PDEOperator,
        sparse: Any,
        spsolve: Any,
        inside: InsidePredicate | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve 2D Poisson-type PDE on a (possibly non-uniform) tensor-product grid.

        Args:
            xs: Grid node x-coordinates (monotone, includes both endpoints).
            ys: Grid node y-coordinates (monotone, includes both endpoints).
            operator: PDE operator supplying ``source_term`` and ``boundary_value``.
            sparse: The ``scipy.sparse`` module.
            spsolve: The ``scipy.sparse.linalg.spsolve`` callable.
            inside: Optional geometry predicate for non-rectangular (e.g.
                L-shaped) domains. When supplied, interior grid nodes whose
                coordinates fall *outside* the physical domain are pinned to
                their Dirichlet boundary value (identity row), which imposes
                the reentrant-edge boundary condition one grid layer in. When
                ``None`` (default) the full bounding box is solved — the
                historical behaviour, unchanged byte-for-byte.

        """
        nx = len(xs) - 2  # interior x points
        ny = len(ys) - 2  # interior y points
        if nx < 1 or ny < 1:
            # Grid too coarse — return zeros
            XX, YY = np.meshgrid(xs, ys, indexing="ij")
            grid = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float64)
            return np.zeros(len(grid), dtype=np.float64), grid

        hx = np.diff(xs)
        hy = np.diff(ys)

        xi = xs[1:-1]
        yi = ys[1:-1]
        XX, YY = np.meshgrid(xi, yi, indexing="ij")
        coords_flat = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)

        # Per-interior-node domain membership (i-major, idx = i*ny + j to match
        # the assembly loop below). All-True when no mask is supplied.
        if inside is not None:
            node_inside = np.asarray(inside(coords_flat.astype(np.float64)), dtype=bool).reshape(-1)
        else:
            node_inside = np.ones(nx * ny, dtype=bool)

        n_interior = nx * ny

        # Build sparse matrix with variable spacing
        # For each interior point (i,j), stencil:
        #   -u(i-1,j)/hx_l*hx_avg - u(i+1,j)/hx_r*hx_avg
        #   -u(i,j-1)/hy_l*hy_avg - u(i,j+1)/hy_r*hy_avg
        #   + u(i,j) * (1/hx_l + 1/hx_r)/hx_avg + (1/hy_l + 1/hy_r)/hy_avg)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        rhs = np.zeros(n_interior, dtype=np.float64)

        for i in range(nx):
            hx_l = hx[i]
            hx_r = hx[i + 1]
            hx_avg = (hx_l + hx_r) / 2.0
            for j in range(ny):
                hy_l = hy[j]
                hy_r = hy[j + 1]
                hy_avg = (hy_l + hy_r) / 2.0

                idx = i * ny + j

                # Masked (out-of-domain) node: pin to its Dirichlet value via an
                # identity row. Neighbouring in-domain equations still reference
                # this column, so the pinned value acts as the interior boundary
                # condition on the reentrant edge. The right-hand side entry is
                # set *after* the ``rhs += f`` step below so the source term does
                # not pollute it. No-op when ``inside is None``.
                if not node_inside[idx]:
                    rows.append(idx)
                    cols.append(idx)
                    vals.append(1.0)
                    continue

                cx = (1.0 / hx_l + 1.0 / hx_r) / hx_avg
                cy = (1.0 / hy_l + 1.0 / hy_r) / hy_avg
                rows.append(idx)
                cols.append(idx)
                vals.append(cx + cy)

                # x-neighbors
                if i > 0:
                    rows.append(idx)
                    cols.append((i - 1) * ny + j)
                    vals.append(-1.0 / (hx_l * hx_avg))
                else:
                    bc = float(
                        np.asarray(
                            operator.boundary_value(np.array([[xs[0], yi[j]]], dtype=np.float32))
                        ).flat[0]
                    )
                    rhs[idx] += bc / (hx_l * hx_avg)

                if i < nx - 1:
                    rows.append(idx)
                    cols.append((i + 1) * ny + j)
                    vals.append(-1.0 / (hx_r * hx_avg))
                else:
                    bc = float(
                        np.asarray(
                            operator.boundary_value(np.array([[xs[-1], yi[j]]], dtype=np.float32))
                        ).flat[0]
                    )
                    rhs[idx] += bc / (hx_r * hx_avg)

                # y-neighbors
                if j > 0:
                    rows.append(idx)
                    cols.append(i * ny + (j - 1))
                    vals.append(-1.0 / (hy_l * hy_avg))
                else:
                    bc = float(
                        np.asarray(
                            operator.boundary_value(np.array([[xi[i], ys[0]]], dtype=np.float32))
                        ).flat[0]
                    )
                    rhs[idx] += bc / (hy_l * hy_avg)

                if j < ny - 1:
                    rows.append(idx)
                    cols.append(i * ny + (j + 1))
                    vals.append(-1.0 / (hy_r * hy_avg))
                else:
                    bc = float(
                        np.asarray(
                            operator.boundary_value(np.array([[xi[i], ys[-1]]], dtype=np.float32))
                        ).flat[0]
                    )
                    rhs[idx] += bc / (hy_r * hy_avg)

        A = sparse.csc_matrix(
            (vals, (rows, cols)),
            shape=(n_interior, n_interior),
        )

        f = np.asarray(operator.source_term(coords_flat), dtype=np.float64).flatten()
        rhs += f

        # Overwrite masked-node right-hand sides with their pinned Dirichlet
        # value *after* the source term is applied, so neither ``f`` nor any
        # boundary contribution above pollutes the identity rows. Vectorised —
        # one ``boundary_value`` call for all masked nodes. No-op when unmasked.
        if not node_inside.all():
            masked = ~node_inside
            masked_coords = coords_flat[masked].astype(np.float32)
            masked_bvals = np.asarray(
                operator.boundary_value(masked_coords), dtype=np.float64
            ).reshape(-1)
            rhs[masked] = masked_bvals

        u_inner = spsolve(A, rhs)

        # Build full grid including boundary
        XX_full, YY_full = np.meshgrid(xs, ys, indexing="ij")
        grid_full = np.stack([XX_full.ravel(), YY_full.ravel()], axis=-1).astype(np.float64)

        u_full = np.zeros((len(xs), len(ys)), dtype=np.float64)
        u_full[1:-1, 1:-1] = u_inner.reshape(nx, ny)

        # Fill boundaries
        for i in range(len(xs)):
            for y_val in [ys[0], ys[-1]]:
                bc = float(
                    np.asarray(
                        operator.boundary_value(np.array([[xs[i], y_val]], dtype=np.float32))
                    ).flat[0]
                )
                j_idx = 0 if y_val == ys[0] else len(ys) - 1
                u_full[i, j_idx] = bc
        for j in range(len(ys)):
            for x_val in [xs[0], xs[-1]]:
                bc = float(
                    np.asarray(
                        operator.boundary_value(np.array([[x_val, ys[j]]], dtype=np.float32))
                    ).flat[0]
                )
                i_idx = 0 if x_val == xs[0] else len(xs) - 1
                u_full[i_idx, j] = bc

        return u_full.ravel().astype(np.float64), grid_full

    @staticmethod
    def _compute_indicators_2d(
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        u: NDArray[np.float64],
        operator: PDEOperator,
        inside: InsidePredicate | None = None,
    ) -> NDArray[np.float64]:
        """Compute element-wise residual error indicators on 2D grid.

        Returns a 2D array of shape (n_elem_x, n_elem_y) with residual indicators.

        Args:
            xs: Grid node x-coordinates.
            ys: Grid node y-coordinates.
            u: Flattened solution over the full ``(len(xs), len(ys))`` grid.
            operator: PDE operator supplying ``source_term``.
            inside: Optional geometry predicate. When supplied, elements whose
                centre falls outside the physical domain get a zero indicator so
                they are never marked for refinement. ``None`` (default) treats
                the full bounding box as the domain — historical behaviour.

        """
        nx = len(xs) - 1
        ny = len(ys) - 1
        u_grid = u.reshape(len(xs), len(ys))
        indicators = np.zeros((nx, ny), dtype=np.float64)

        # Evaluate the domain predicate for every element centre in a single
        # vectorised call (mirrors how _solve_on_grid_2d tests its nodes) instead
        # of once per element. The centres are built through the identical
        # float32 -> float64 round-trip used by the per-element ``mid`` below so
        # the predicate input is byte-for-byte the same.
        elem_inside: NDArray[np.bool_] | None = None
        if inside is not None:
            elem_inside = element_inside_mask(xs, ys, inside)

        for i in range(nx):
            hx = xs[i + 1] - xs[i]
            for j in range(ny):
                hy = ys[j + 1] - ys[j]
                # Element wholly outside the physical domain: never refine it.
                if elem_inside is not None and not elem_inside[i, j]:
                    continue
                mid = np.array(
                    [[(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2]],
                    dtype=np.float32,
                )
                f_mid = float(np.asarray(operator.source_term(mid)).flat[0])

                # Approximate Laplacian at element center using surrounding values
                ci = min(i + 1, len(xs) - 2)
                cj = min(j + 1, len(ys) - 2)
                ci = max(ci, 1)
                cj = max(cj, 1)

                u_xx = 0.0
                if 0 < ci < len(xs) - 1 and hx > 0:
                    u_xx = (u_grid[ci - 1, cj] - 2 * u_grid[ci, cj] + u_grid[ci + 1, cj]) / (hx**2)
                u_yy = 0.0
                if 0 < cj < len(ys) - 1 and hy > 0:
                    u_yy = (u_grid[ci, cj - 1] - 2 * u_grid[ci, cj] + u_grid[ci, cj + 1]) / (hy**2)

                laplacian = u_xx + u_yy
                has_nan = np.isnan(laplacian) or np.isnan(f_mid)
                residual = 0.0 if has_nan else abs(-laplacian - f_mid)
                h_elem = np.sqrt(hx * hy)
                indicators[i, j] = h_elem * residual

        return indicators

    def _dorfler_mark_2d(
        self,
        indicators: NDArray[np.float64],
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Dorfler marking on 2D grid. Returns marked arrays for x and y element edges."""
        flat = indicators.ravel()
        total = np.sum(flat**2)
        threshold = self.marking_fraction * total

        sorted_idx = np.argsort(flat)[::-1]
        cumsum = np.cumsum(flat[sorted_idx] ** 2)
        n_mark = int(np.searchsorted(cumsum, threshold)) + 1

        ny_elem = indicators.shape[1]
        marked_x = np.zeros(len(xs) - 1, dtype=bool)
        marked_y = np.zeros(len(ys) - 1, dtype=bool)

        for flat_idx in sorted_idx[:n_mark]:
            i = flat_idx // ny_elem
            j = flat_idx % ny_elem
            marked_x[i] = True
            marked_y[j] = True

        return marked_x, marked_y
