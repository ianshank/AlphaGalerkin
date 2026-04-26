"""Streamline-Upwind / Petrov-Galerkin (SUPG) FEM solver baseline.

The standard Galerkin / central-difference discretisation of the
advection-diffusion equation::

    a · ∇u  -  ν ∇² u  =  f

is unstable when the local Peclet number ``Pe = |a| h / (2 ν)``
exceeds 1.  SUPG adds a streamline-upwind weighting to the test
functions so the discrete operator remains coercive across the full
Peclet range.  This module implements a 1D SUPG-FEM solver against
which AlphaGalerkin is compared in
``config/proposals/doe_ascr_c59.yaml::advdiff_boundary_layer``.

The 1D restriction is intentional — every comparison benchmark in the
DOE ASCR proposal uses 1D advection-diffusion (the boundary-layer
problem is canonically 1D), and the linear-element formulation is
significantly clearer when written without tensor indices.

All hyperparameters (Peclet thresholds, stabilisation ``τ`` formula,
boundary-condition tolerance) are surfaced as Pydantic fields on
:class:`SUPGFEMConfig` — no hardcoded values.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
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


class SUPGFEMConfig(SolverConfig):
    """Configuration for :class:`SUPGFEMSolver`.

    The default values follow Brooks & Hughes (1982).  Every numerical
    knob is exposed as a Pydantic field so reproductions can override
    via Hydra YAML without editing source.
    """

    min_grid_points: int = Field(
        default=4,
        ge=4,
        description="Minimum interior nodes (must be >=4 for stable SUPG).",
    )
    velocity_floor: float = Field(
        default=1e-12,
        gt=0,
        description=(
            "Lower clamp on |a| to prevent τ -> ∞ when the advection "
            "velocity vanishes (degenerate diffusion-only problem)."
        ),
    )
    diffusion_floor: float = Field(
        default=1e-12,
        gt=0,
        description=(
            "Lower clamp on ν to prevent the Peclet number from "
            "exploding numerically when the input is exactly zero."
        ),
    )
    pe_low_threshold: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Below this Peclet number, SUPG reduces to standard "
            "Galerkin (τ = 0) — the diffusive regime is already stable."
        ),
    )


class SUPGFEMSolver(BaseSolver):
    """1D SUPG-stabilised linear-element FEM solver.

    Solves ``a u_x - ν u_xx = f`` on ``[x_min, x_max]`` with Dirichlet
    boundary data taken from
    :meth:`PDEOperator.boundary_value`.  Stabilisation parameter::

        τ = (h / (2 |a|)) · (coth(Pe) - 1/Pe),   Pe = |a| h / (2 ν)

    which is exact for piecewise-linear elements on a uniform mesh
    (Brooks & Hughes 1982, eq. 2.20).  When ``Pe < pe_low_threshold``
    the weighting collapses to standard Galerkin to avoid unnecessary
    numerical noise in the diffusive regime.
    """

    name = "supg_fem"
    description = "Streamline-Upwind / Petrov-Galerkin FEM (1D)"

    def __init__(self, config: SUPGFEMConfig | None = None) -> None:
        self.config = config or SUPGFEMConfig()

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve the advection-diffusion equation with SUPG-FEM.

        Args:
            operator: PDE operator (must be 1D advection-diffusion-like;
                attributes ``advection_velocity`` and ``diffusion`` are
                read with permissive fallbacks).
            n_dof: Target number of interior degrees of freedom.
            **kwargs: Reserved for forward compatibility with the
                :class:`BaseSolver` protocol; unused here.

        Returns:
            :class:`SolverResult` with the discrete solution sampled at
            grid nodes, the L2 error vs ``operator.exact_solution`` if
            available, and Peclet metadata for inspection.

        """
        try:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
        except ImportError as exc:
            raise ImportError(
                "SUPGFEMSolver requires scipy. Install with: pip install scipy"
            ) from exc

        if operator.dim != 1:
            raise NotImplementedError(
                f"SUPGFEMSolver currently supports 1D problems only "
                f"(operator.dim={operator.dim})."
            )

        log = logger.bind(solver=self.name, n_dof=n_dof)
        log.info("supg_solve_start")
        t0 = time.perf_counter()

        n = max(int(n_dof), self.config.min_grid_points)
        x = np.linspace(
            float(operator.domain_min[0]),
            float(operator.domain_max[0]),
            n + 2,
            dtype=np.float64,
        )
        h = float(x[1] - x[0])

        # Pull problem coefficients from the operator with safe fallbacks.
        a_vec = np.asarray(
            getattr(operator, "advection_velocity", np.array([1.0])),
            dtype=np.float64,
        ).flatten()
        a_scalar = float(a_vec[0]) if a_vec.size else 1.0
        nu = float(getattr(operator, "diffusion", 1.0))

        # Robust Peclet computation
        a_eff = max(abs(a_scalar), self.config.velocity_floor)
        nu_eff = max(nu, self.config.diffusion_floor)
        peclet = a_eff * h / (2.0 * nu_eff)
        if peclet < self.config.pe_low_threshold:
            tau = 0.0
        else:
            # τ = (h / (2 |a|)) * (coth(Pe) - 1/Pe)
            tau = (h / (2.0 * a_eff)) * (1.0 / np.tanh(peclet) - 1.0 / peclet)

        log.info("supg_peclet", peclet=peclet, tau=tau, h=h)
        log.debug(
            "supg_stencil",
            n_interior=n,
            advection_velocity=a_scalar,
            diffusion=nu,
            galerkin_only=tau == 0.0,
        )

        # Linear-element stiffness for diffusion: K_diff_ii = 2ν/h, K_diff_{i,i±1} = -ν/h
        diags_diff_main = np.full(n, 2.0 * nu / h)
        diags_diff_off = np.full(n - 1, -nu / h)

        # Centred advection: A_ii = 0, A_{i,i+1} = a/2, A_{i,i-1} = -a/2
        diags_adv_super = np.full(n - 1, a_scalar / 2.0)
        diags_adv_sub = np.full(n - 1, -a_scalar / 2.0)

        # SUPG correction: τ · (a u_x, a φ_x) — yields τ a² · K_diff/ν
        # because (φ_x, φ_x) has the same shape as the diffusion matrix.
        if tau > 0:
            supg_main = np.full(n, tau * (a_scalar**2) * 2.0 / h)
            supg_off = np.full(n - 1, tau * (a_scalar**2) * -1.0 / h)
        else:
            supg_main = np.zeros(n)
            supg_off = np.zeros(n - 1)

        # Combine diffusion (Galerkin), advection (centred), and SUPG
        # streamline-upwind contributions into one tridiagonal system.
        main = diags_diff_main + supg_main
        super_ = diags_diff_off + diags_adv_super + supg_off
        sub = diags_diff_off + diags_adv_sub + supg_off

        A = sparse.diags(
            [sub, main, super_],
            offsets=[-1, 0, 1],
            shape=(n, n),
            format="csc",
        )

        # Right-hand side: f(x_i) * h (lumped) plus SUPG correction τ·a·(f, φ_x)
        interior = x[1:-1]
        f_at_nodes = np.asarray(
            operator.source_term(interior.reshape(-1, 1).astype(np.float32)),
            dtype=np.float64,
        ).flatten()
        rhs = h * f_at_nodes
        if tau > 0:
            # τ a (f, φ_x) ≈ τ a (f_{i+1} - f_{i-1}) / 2 (centred)
            grad_f = np.zeros_like(f_at_nodes)
            grad_f[1:-1] = (f_at_nodes[2:] - f_at_nodes[:-2]) / 2.0
            rhs += tau * a_scalar * grad_f

        # Boundary contributions from Dirichlet data
        bc_left = float(
            np.asarray(
                operator.boundary_value(
                    np.array([[operator.domain_min[0]]], dtype=np.float32)
                )
            ).flat[0]
        )
        bc_right = float(
            np.asarray(
                operator.boundary_value(
                    np.array([[operator.domain_max[0]]], dtype=np.float32)
                )
            ).flat[0]
        )
        rhs[0] -= sub[0] * bc_left
        rhs[-1] -= super_[-1] * bc_right

        u_interior = spsolve(A, rhs)
        # Reattach Dirichlet end-points so consumers get a length-(n+2)
        # solution sampled on the full mesh.
        solution = np.concatenate([[bc_left], u_interior, [bc_right]]).astype(
            np.float64
        )

        wall_time = time.perf_counter() - t0
        l2_err = self._compute_l2_error(
            solution=solution,
            coords=x,
            operator=operator,
        )

        log.info(
            "supg_solve_done",
            wall_time=wall_time,
            l2_error=l2_err,
            peclet=peclet,
            tau=tau,
        )
        return SolverResult(
            solution=solution,
            grid_points=x,
            n_dof=int(n),
            wall_time_seconds=float(wall_time),
            l2_error=l2_err,
            metadata={
                "method": "supg_fem_linear_1d",
                "peclet": float(peclet),
                "tau": float(tau),
                "h": float(h),
                "diffusion": nu,
                "advection_velocity": float(a_scalar),
            },
        )

    def _compute_l2_error(
        self,
        solution: NDArray[np.float64],
        coords: NDArray[np.float64],
        operator: PDEOperator,
    ) -> float | None:
        """Override that calls the operator-level :meth:`exact_solution`.

        :class:`BaseSolver._compute_l2_error` expects 2D coords (N, dim);
        for our 1D problem we reshape so the call site lands in the
        same code path.
        """
        # The base helper expects (N, dim) float64 coords.  Reshape to
        # 2D and cast back from the float32 input we feed the operator
        # for compatibility with the abstract signature.
        coords_2d = coords.reshape(-1, 1).astype(np.float64)
        return super()._compute_l2_error(
            solution=solution,
            coords=coords_2d,
            operator=operator,
        )


# Module-level registration.  Idempotent via setdefault so re-imports
# during testing do not raise.
SOLVER_REGISTRY.setdefault("supg_fem", SUPGFEMSolver)
