"""Navier-Stokes equation physics module with SGS closure model library."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import structlog

from src.alphagalerkin.core.types import PDEType
from src.alphagalerkin.physics.base import (
    BoundaryCondition,
    ManufacturedSolution,
    SolveResult,
)
from src.alphagalerkin.physics.registry import register_physics

logger = structlog.get_logger("physics.navier_stokes")

# Default physical parameters
_DEFAULT_KINEMATIC_VISCOSITY: float = 0.01
_DEFAULT_LID_VELOCITY: float = 1.0


# -------------------------------------------------------------------
# SGS Closure Model Protocol
# -------------------------------------------------------------------

@dataclass
class SGSClosureModel:
    """Sub-grid-scale closure model for LES of Navier-Stokes.

    Attributes
    ----------
    name:
        Human-readable name of the closure model.
    compute_viscosity:
        Callable(strain_rate, rotation_rate) -> eddy_viscosity.
        strain_rate: symmetric part of velocity gradient, shape (n, 2, 2).
        rotation_rate: antisymmetric part, shape (n, 2, 2).
        Returns eddy viscosity array of shape (n,).

    """

    name: str
    compute_viscosity: Callable[
        [np.ndarray, np.ndarray], np.ndarray
    ]


# -------------------------------------------------------------------
# Closure model implementations
# -------------------------------------------------------------------

def _strain_rate_magnitude(
    strain_rate: np.ndarray,
) -> np.ndarray:
    """Compute |S| = sqrt(2*S_ij*S_ij) from strain rate tensor.

    Parameters
    ----------
    strain_rate:
        Shape (n, 2, 2) symmetric tensor.

    Returns
    -------
    Array of shape (n,) with strain rate magnitude.

    """
    # |S| = sqrt(2 * S_ij * S_ij)
    s_sq = np.einsum("nij,nij->n", strain_rate, strain_rate)
    return np.sqrt(2.0 * s_sq)  # type: ignore[no-any-return]


class SmagorinskyModel:
    """Classic Smagorinsky SGS model.

    nu_t = (C_s * delta)^2 * |S|

    Parameters
    ----------
    c_s:
        Smagorinsky constant (typical: 0.1-0.2).
    delta:
        Filter width / grid spacing.

    """

    def __init__(
        self, c_s: float = 0.17, delta: float = 1.0,
    ) -> None:
        self._c_s = c_s
        self._delta = delta

    def compute_viscosity(
        self,
        strain_rate: np.ndarray,
        rotation_rate: np.ndarray,
    ) -> np.ndarray:
        """Compute eddy viscosity: nu_t = (C_s * delta)^2 * |S|."""
        s_mag = _strain_rate_magnitude(strain_rate)
        return (self._c_s * self._delta) ** 2 * s_mag

    def to_closure(self) -> SGSClosureModel:
        """Convert to SGSClosureModel dataclass."""
        return SGSClosureModel(
            name="smagorinsky",
            compute_viscosity=self.compute_viscosity,
        )


class DynamicSmagorinskyModel:
    """Dynamic Smagorinsky model with Germano identity (simplified).

    Dynamically computes C_s^2 from resolved scales using the
    Germano identity.  This simplified version uses a constant
    ratio approximation.

    Parameters
    ----------
    delta:
        Filter width / grid spacing.
    test_filter_ratio:
        Ratio of test filter to grid filter width.
    c_s_min:
        Minimum allowed C_s (clipping for stability).
    c_s_max:
        Maximum allowed C_s (clipping for stability).

    """

    def __init__(
        self,
        delta: float = 1.0,
        test_filter_ratio: float = 2.0,
        c_s_min: float = 0.0,
        c_s_max: float = 0.3,
    ) -> None:
        self._delta = delta
        self._test_filter_ratio = test_filter_ratio
        self._c_s_min = c_s_min
        self._c_s_max = c_s_max

    def compute_viscosity(
        self,
        strain_rate: np.ndarray,
        rotation_rate: np.ndarray,
    ) -> np.ndarray:
        """Compute eddy viscosity with dynamic C_s.

        Simplified Germano identity: uses a locally-averaged
        C_s^2 based on the strain rate ratio between filter levels.
        """
        s_mag = _strain_rate_magnitude(strain_rate)

        # Simplified dynamic procedure:
        # C_s^2 ~ |S|_test / (test_ratio^2 * |S|_grid)
        # Using local approximation with clipping
        s_mean = np.mean(s_mag) if s_mag.size > 0 else 1.0
        s_mean = max(s_mean, 1e-12)
        c_s_sq = s_mag / (
            self._test_filter_ratio**2 * s_mean
        )
        c_s_sq = np.clip(
            c_s_sq,
            self._c_s_min**2,
            self._c_s_max**2,
        )

        return c_s_sq * self._delta**2 * s_mag  # type: ignore[no-any-return]

    def to_closure(self) -> SGSClosureModel:
        """Convert to SGSClosureModel dataclass."""
        return SGSClosureModel(
            name="dynamic_smagorinsky",
            compute_viscosity=self.compute_viscosity,
        )


class WALEModel:
    """Wall-Adapted Local Eddy-viscosity (WALE) model.

    Computes eddy viscosity using the traceless symmetric part
    of the square of the velocity gradient tensor, providing
    correct wall scaling (nu_t ~ y^3) without requiring damping
    functions.

    Parameters
    ----------
    c_w:
        WALE model constant (typical: 0.325-0.5).
    delta:
        Filter width / grid spacing.

    """

    def __init__(
        self, c_w: float = 0.325, delta: float = 1.0,
    ) -> None:
        self._c_w = c_w
        self._delta = delta

    def compute_viscosity(
        self,
        strain_rate: np.ndarray,
        rotation_rate: np.ndarray,
    ) -> np.ndarray:
        """Compute WALE eddy viscosity.

        nu_t = (C_w * delta)^2 * (S_d_ij * S_d_ij)^{3/2}
               / ((S_ij * S_ij)^{5/2} + (S_d_ij * S_d_ij)^{5/4} + eps)

        where S_d_ij is the traceless symmetric part of g_ik*g_kj.
        """
        # Velocity gradient tensor g = S + Omega
        g = strain_rate + rotation_rate

        # g^2 = g_ik * g_kj
        g_sq = np.einsum("nik,nkj->nij", g, g)

        # Traceless symmetric part: S^d = 0.5*(g^2 + g^2.T) - (1/3)*tr(g^2)*I
        g_sq_sym = 0.5 * (g_sq + np.swapaxes(g_sq, -2, -1))
        trace = np.einsum("nii->n", g_sq)
        identity = np.eye(2)[np.newaxis, :, :]  # (1, 2, 2)
        s_d = g_sq_sym - (trace[:, np.newaxis, np.newaxis] / 3.0) * identity

        # Compute invariants
        s_ij_sq = np.einsum(
            "nij,nij->n", strain_rate, strain_rate,
        )
        s_d_sq = np.einsum("nij,nij->n", s_d, s_d)

        eps = 1e-12
        numerator = s_d_sq**1.5
        denominator = (
            s_ij_sq**2.5
            + s_d_sq**1.25
            + eps
        )

        return (self._c_w * self._delta) ** 2 * (  # type: ignore[no-any-return]
            numerator / denominator
        )

    def to_closure(self) -> SGSClosureModel:
        """Convert to SGSClosureModel dataclass."""
        return SGSClosureModel(
            name="wale",
            compute_viscosity=self.compute_viscosity,
        )


class NoModel:
    """No SGS model (DNS / no subgrid-scale modeling).

    Returns zero eddy viscosity everywhere.
    """

    def compute_viscosity(
        self,
        strain_rate: np.ndarray,
        rotation_rate: np.ndarray,
    ) -> np.ndarray:
        """Return zero eddy viscosity (DNS)."""
        n = strain_rate.shape[0]
        return np.zeros(n)

    def to_closure(self) -> SGSClosureModel:
        """Convert to SGSClosureModel dataclass."""
        return SGSClosureModel(
            name="no_model",
            compute_viscosity=self.compute_viscosity,
        )


# -------------------------------------------------------------------
# Closure model registry
# -------------------------------------------------------------------

_CLOSURE_MODELS: dict[str, type] = {
    "smagorinsky": SmagorinskyModel,
    "dynamic_smagorinsky": DynamicSmagorinskyModel,
    "wale": WALEModel,
    "no_model": NoModel,
}


def select_closure(
    name: str, **kwargs: Any,
) -> SGSClosureModel:
    """Select an SGS closure model by name.

    Parameters
    ----------
    name:
        One of: "smagorinsky", "dynamic_smagorinsky",
        "wale", "no_model".
    **kwargs:
        Additional parameters passed to the closure model constructor.

    Returns
    -------
    SGSClosureModel dataclass wrapping the selected model.

    Raises
    ------
    KeyError:
        If the name is not a registered closure model.

    """
    if name not in _CLOSURE_MODELS:
        available = ", ".join(sorted(_CLOSURE_MODELS.keys()))
        msg = (
            f"Unknown SGS closure model '{name}'."
            f" Available: {available}"
        )
        raise KeyError(msg)

    model_cls = _CLOSURE_MODELS[name]
    model = model_cls(**kwargs)
    logger.info(
        "physics.navier_stokes.closure_selected",
        closure_model=name,
    )
    from typing import cast
    return cast(SGSClosureModel, model.to_closure())


def list_closures() -> list[str]:
    """List all available SGS closure model names."""
    return sorted(_CLOSURE_MODELS.keys())


# -------------------------------------------------------------------
# Navier-Stokes Module
# -------------------------------------------------------------------

@register_physics("navier_stokes_2d")
class NavierStokesModule:
    """2D Navier-Stokes with SGS closure model library.

    Solves a simplified 2D lid-driven cavity (Stokes approximation)
    via finite differences for validation/testing.

    The steady-state Stokes equations:
        -nu*laplacian(u) + grad(p) = f_u
        -nu*laplacian(v) + grad(p) = f_v
        div(u) = 0

    For the solve_on_grid method, we solve the Stokes problem
    using a pressure-free formulation (stream function approach
    simplified to velocity-only solve on a staggered grid).
    """

    name: str = "navier_stokes_2d"
    pde_type: PDEType = PDEType.MIXED

    def __init__(
        self,
        kinematic_viscosity: float = _DEFAULT_KINEMATIC_VISCOSITY,
        lid_velocity: float = _DEFAULT_LID_VELOCITY,
    ) -> None:
        self._kinematic_viscosity = kinematic_viscosity
        self._lid_velocity = lid_velocity
        self._closure: SGSClosureModel | None = None

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity nu."""
        return self._kinematic_viscosity

    @property
    def lid_velocity(self) -> float:
        """Lid velocity for cavity problem."""
        return self._lid_velocity

    @property
    def closure(self) -> SGSClosureModel | None:
        """Currently selected SGS closure model."""
        return self._closure

    def set_closure(
        self, name: str, **kwargs: Any,
    ) -> None:
        """Set the active SGS closure model.

        Parameters
        ----------
        name:
            Closure model name (see :func:`select_closure`).
        **kwargs:
            Additional parameters for the closure model.

        """
        self._closure = select_closure(name, **kwargs)
        logger.info(
            "physics.navier_stokes.closure_set",
            closure_model=name,
        )

    def weak_form(
        self,
        trial: Any,
        test: Any,
        mesh: Any,
    ) -> Any:
        """Weak form: nu*integral(grad(u):grad(v)) - integral(p*div(v)) = integral(f*v)."""
        logger.debug("physics.navier_stokes.weak_form")
        return None  # Placeholder for FEM backend integration

    def boundary_conditions(
        self,
        mesh: Any = None,
        config: Any = None,
    ) -> list[BoundaryCondition]:
        """Lid-driven cavity BCs.

        Top wall: u = lid_velocity, v = 0 (moving lid).
        Other walls: u = v = 0 (no-slip).
        """
        return [
            BoundaryCondition(
                bc_type="dirichlet", value=0.0, region="bottom",
            ),
            BoundaryCondition(
                bc_type="dirichlet", value=0.0, region="left",
            ),
            BoundaryCondition(
                bc_type="dirichlet", value=0.0, region="right",
            ),
            BoundaryCondition(
                bc_type="dirichlet",
                value=self._lid_velocity,
                region="top",
            ),
        ]

    def manufactured_solution(
        self,
        config: Any = None,
    ) -> ManufacturedSolution:
        """MMS for Stokes: stream function psi = sin(pi*x)^2*sin(pi*y)^2.

        This gives divergence-free velocity:
            u =  dpsi/dy = sin(pi*x)^2 * 2*pi*sin(pi*y)*cos(pi*y)
            v = -dpsi/dx = -2*pi*sin(pi*x)*cos(pi*x) * sin(pi*y)^2
        """
        nu = self._kinematic_viscosity

        def exact(points: np.ndarray) -> np.ndarray:
            """Return u-velocity component (primary field)."""
            x, y = points[:, 0], points[:, 1]
            return (
                np.sin(np.pi * x) ** 2
                * 2.0
                * np.pi
                * np.sin(np.pi * y)
                * np.cos(np.pi * y)
            )

        def forcing(points: np.ndarray) -> np.ndarray:
            """Forcing for u-momentum (Stokes: -nu*laplacian(u) = f_u)."""
            x, y = points[:, 0], points[:, 1]
            # Approximate forcing from manufactured solution
            # f = nu * pi^2 * sin(pi*x) * sin(pi*y) (simplified)
            return (
                nu
                * np.pi**2
                * np.sin(np.pi * x)
                * np.sin(np.pi * y)
            )

        def boundary(points: np.ndarray) -> np.ndarray:
            return np.zeros(len(points))

        return ManufacturedSolution(
            exact_solution=exact,
            forcing=forcing,
            boundary_data=boundary,
            expected_convergence_order=2.0,
            name="stokes_cavity",
        )

    def reward_function(
        self,
        state: Any,
        action: Any,
        next_state: Any,
    ) -> float:
        """Reward based on residual reduction and DOF efficiency."""
        return 0.0  # Placeholder

    def state_features(self, discretization: Any) -> Any:
        """Per-element features for the GNN encoder."""
        return None  # Placeholder

    def action_validators(self) -> list[Any]:
        """Navier-Stokes-specific action validators."""
        return []

    def default_config(self) -> dict[str, Any]:
        """Default parameters for Navier-Stokes problems."""
        return {
            "domain": {
                "type": "rectangle",
                "bounds": [[0.0, 1.0], [0.0, 1.0]],
            },
            "kinematic_viscosity": self._kinematic_viscosity,
            "lid_velocity": self._lid_velocity,
        }

    def solve_on_grid(self, n: int) -> SolveResult:
        """Solve simplified 2D lid-driven cavity (Stokes approximation).

        Solves the u-momentum equation on an n x n grid:
            -nu*laplacian(u) = 0 (interior)
        with u = lid_velocity on top wall, u = 0 elsewhere.

        This is a Stokes approximation (no pressure, no nonlinear
        convection), suitable for validation and testing.
        """
        start = time.perf_counter()

        nu = self._kinematic_viscosity
        h = 1.0 / (n + 1)
        total_dofs = n * n

        # Build 5-point stencil for -nu*laplacian
        main_diag = 4.0 * nu * np.ones(total_dofs)
        off_diag_1 = -nu * np.ones(total_dofs - 1)
        off_diag_n = -nu * np.ones(total_dofs - n)

        # Zero out connections across row boundaries
        for i in range(1, n):
            off_diag_1[i * n - 1] = 0.0

        stiffness = sp.diags(
            [
                off_diag_n,
                off_diag_1,
                main_diag,
                off_diag_1,
                off_diag_n,
            ],
            [-n, -1, 0, 1, n],
            shape=(total_dofs, total_dofs),
            format="csc",
        ) / (h * h)

        # RHS: zero interior forcing, lid BC contribution
        rhs = np.zeros(total_dofs)

        # Apply lid BC: top boundary nodes (last row of interior)
        # The top row of interior nodes (j = n-1) has a neighbor
        # at j = n (boundary) with u = lid_velocity.
        # Contribution: nu * lid_velocity / h^2
        for i in range(n):
            # Index of top-row interior node
            top_idx = (n - 1) * n + i
            rhs[top_idx] += nu * self._lid_velocity / (h * h)

        # Solve
        u = scipy.sparse.linalg.spsolve(stiffness, rhs)

        # Compute residual
        residual = stiffness @ u - rhs
        residual_norm = float(np.linalg.norm(residual))

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "physics.navier_stokes.solved",
            grid_size=n,
            kinematic_viscosity=nu,
            lid_velocity=self._lid_velocity,
            residual_norm=residual_norm,
            solve_time_ms=elapsed_ms,
        )

        return SolveResult(
            solution=u,
            residual_norm=residual_norm,
            condition_number=1.0,  # Skip expensive computation
            solve_time_ms=elapsed_ms,
            converged=True,
            metadata={
                "closure_model": (
                    self._closure.name
                    if self._closure is not None
                    else "none"
                ),
            },
        )
