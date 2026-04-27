"""Time-stepping methods for time-dependent PDEs.

Provides numerical time integration schemes for evolving PDE solutions,
enabling MCTS to plan across time steps for optimal adaptive refinement.

This is a key SBIR differentiator: MCTS plans not just spatial refinement
at a single time step, but multi-step temporal strategies where current
refinement decisions affect future solution quality.

Supported methods:
- Forward Euler (explicit, first-order)
- RK4 (explicit, fourth-order)
- Crank-Nicolson (implicit, second-order, A-stable)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import structlog
import torch
from pydantic import Field, model_validator
from torch import Tensor

from src.templates.config import BaseModuleConfig

logger = structlog.get_logger(__name__)


# Order-of-accuracy used by the step-doubling error estimator for each
# concrete time-stepping scheme.  These numbers are the *theoretical*
# orders the schemes are expected to attain on smooth solutions and are
# used only as exponents in the PI-controller, never as a substitute for
# a tunable parameter.
_SCHEME_ORDER: dict[str, int] = {
    "forward_euler": 1,
    "rk4": 4,
    "crank_nicolson": 2,
}


class TimeSteppingMethod(str, Enum):
    """Supported time-stepping methods."""

    FORWARD_EULER = "forward_euler"
    RK4 = "rk4"
    CRANK_NICOLSON = "crank_nicolson"


class TimeSteppingConfig(BaseModuleConfig):
    """Configuration for time-stepping integration."""

    method: TimeSteppingMethod = Field(
        default=TimeSteppingMethod.RK4,
        description="Time integration method.",
    )
    dt: float = Field(
        default=0.01,
        gt=0.0,
        description="Time step size.",
    )
    t_start: float = Field(
        default=0.0,
        ge=0.0,
        description="Start time.",
    )
    t_end: float = Field(
        default=1.0,
        gt=0.0,
        description="End time.",
    )
    adaptive_dt: bool = Field(
        default=False,
        description="Enable adaptive time step control.",
    )
    dt_min: float = Field(
        default=1e-6,
        gt=0.0,
        description="Minimum time step for adaptive control.",
    )
    dt_max: float = Field(
        default=0.1,
        gt=0.0,
        description="Maximum time step for adaptive control.",
    )
    error_tolerance: float = Field(
        default=1e-5,
        gt=0.0,
        description=(
            "Absolute error tolerance (atol) for adaptive time stepping. "
            "Used in the scaled error norm `err = ||u_full - u_two_half|| "
            "/ (atol + rtol * max(|u|))`."
        ),
    )
    relative_tolerance: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Relative tolerance (rtol) coefficient in the scaled error norm "
            "for adaptive time stepping.  Multiplied by max(|u_full|, |u_two_half|) "
            "and added to the absolute tolerance.  Default 1.0 reproduces "
            "the historical Track-C behaviour."
        ),
    )
    t_end_epsilon: float = Field(
        default=1e-12,
        gt=0.0,
        description=(
            "Floating-point slack used when comparing t against t_end and "
            "dt against dt_min during adaptive integration.  Prevents "
            "spurious extra steps caused by accumulated round-off."
        ),
    )
    max_steps: int = Field(
        default=100000,
        ge=1,
        description="Maximum number of time steps (safety limit).",
    )
    save_interval: int = Field(
        default=10,
        ge=1,
        description="Save solution every N steps.",
    )
    safety_factor: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="PI-controller safety factor applied to the proposed dt "
        "(0 < f <= 1).  Used only when adaptive_dt=True.",
    )
    pi_alpha: float = Field(
        default=0.7,
        gt=0.0,
        description="PI-controller proportional exponent on the current "
        "error ratio (tol/err)^alpha.  Used only when adaptive_dt=True.",
    )
    pi_beta: float = Field(
        default=0.4,
        ge=0.0,
        description="PI-controller integral exponent on the previous-step "
        "error ratio (err_prev/err)^beta.  beta=0 reduces to a pure "
        "I-controller.  Used only when adaptive_dt=True.",
    )

    @model_validator(mode="after")
    def _validate_dt_bounds(self) -> TimeSteppingConfig:
        """Cross-field validation for the dt range.

        Ensures dt_min < dt_max and that the requested initial dt sits
        inside [dt_min, dt_max] when adaptive control is active.  Without
        adaptive control the initial dt is honoured as-is.
        """
        if self.dt_min >= self.dt_max:
            msg = f"dt_min ({self.dt_min}) must be strictly less than dt_max " f"({self.dt_max})."
            raise ValueError(msg)
        if self.adaptive_dt and not (self.dt_min <= self.dt <= self.dt_max):
            msg = (
                f"Initial dt ({self.dt}) must lie within [dt_min, dt_max] = "
                f"[{self.dt_min}, {self.dt_max}] when adaptive_dt=True."
            )
            raise ValueError(msg)
        return self


class TimeStepper(ABC):
    """Abstract base class for time-stepping methods."""

    def __init__(self, config: TimeSteppingConfig) -> None:
        """Initialize time stepper.

        Args:
            config: Time-stepping configuration.

        """
        self.config = config
        self.dt = config.dt
        # Order p used by the step-doubling error estimator: the local
        # error of one full step minus two half-steps is O(dt^{p+1}); we
        # use p as the exponent in the dt update rule.
        self._scheme_order: int = _SCHEME_ORDER.get(config.method.value, 1)
        # PI-controller memory (previous-step error ratio).  Initialised
        # to 1 so the first adaptive step behaves as a pure I-controller.
        self._prev_err_ratio: float = 1.0

    @abstractmethod
    def step(
        self,
        u: Tensor,
        t: float,
        rhs_fn: Any,
    ) -> tuple[Tensor, float]:
        """Advance solution by one time step.

        Args:
            u: Current solution state.
            t: Current time.
            rhs_fn: Right-hand side function du/dt = rhs_fn(u, t).

        Returns:
            Tuple of (new solution, new time).

        """
        ...

    def integrate(
        self,
        u0: Tensor,
        rhs_fn: Any,
    ) -> list[tuple[Tensor, float]]:
        """Integrate from t_start to t_end.

        With ``adaptive_dt=False`` (the default) this is a fixed-dt
        loop and is byte-identical to the pre-adaptive implementation.
        With ``adaptive_dt=True`` a PI-controlled, step-doubling
        adaptive scheme drives ``self.dt`` between
        ``[config.dt_min, config.dt_max]``.

        Args:
            u0: Initial condition.
            rhs_fn: Right-hand side function du/dt = rhs_fn(u, t).

        Returns:
            List of (solution, time) snapshots at save intervals.

        """
        if self.config.adaptive_dt:
            return self._integrate_adaptive(u0, rhs_fn)
        return self._integrate_fixed(u0, rhs_fn)

    def _integrate_fixed(
        self,
        u0: Tensor,
        rhs_fn: Any,
    ) -> list[tuple[Tensor, float]]:
        """Fixed-dt integration loop (byte-identical to pre-adaptive)."""
        u = u0.clone()
        t = self.config.t_start
        snapshots: list[tuple[Tensor, float]] = [(u.clone(), t)]
        step_count = 0

        logger.info(
            "time_integration_start",
            method=self.__class__.__name__,
            t_start=self.config.t_start,
            t_end=self.config.t_end,
            dt=self.dt,
            adaptive=False,
        )

        while (
            t < self.config.t_end - self.config.t_end_epsilon and step_count < self.config.max_steps
        ):
            # Ensure we don't overshoot t_end
            dt_actual = min(self.dt, self.config.t_end - t)
            old_dt = self.dt
            self.dt = dt_actual

            u, t = self.step(u, t, rhs_fn)

            self.dt = old_dt
            step_count += 1

            if step_count % self.config.save_interval == 0:
                snapshots.append((u.clone(), t))

        # Always save final state
        if snapshots[-1][1] != t:
            snapshots.append((u.clone(), t))

        logger.info(
            "time_integration_complete",
            n_steps=step_count,
            t_final=t,
            n_snapshots=len(snapshots),
        )

        return snapshots

    # ------------------------------------------------------------------
    # Adaptive integration (PI controller + step doubling)
    # ------------------------------------------------------------------

    def _step_doubling_error(
        self,
        u: Tensor,
        t: float,
        rhs_fn: Any,
    ) -> tuple[Tensor, float]:
        """Local error estimate via step doubling.

        Returns the higher-order solution (two half-steps) and a scalar
        error norm comparing it to the single full-step solution.  The
        full step is taken at the current ``self.dt``.  This estimator
        works uniformly for any concrete :meth:`step` implementation,
        which keeps Forward-Euler / RK4 / Crank-Nicolson sharing the
        same adaptive-control surface.
        """
        full_dt = self.dt
        # One full step
        u_full, _ = self.step(u, t, rhs_fn)
        # Two half steps
        self.dt = full_dt / 2.0
        u_half, t_half = self.step(u, t, rhs_fn)
        u_two_half, _ = self.step(u_half, t_half, rhs_fn)
        self.dt = full_dt
        # Scaled error norm following the standard PI-controller recipe:
        # err = || u_full - u_two_half || / (atol + rtol * max(|u_full|, |u_two_half|))
        denom = self.config.error_tolerance + self.config.relative_tolerance * torch.maximum(
            torch.abs(u_full), torch.abs(u_two_half)
        )
        err_tensor = torch.linalg.vector_norm((u_full - u_two_half) / denom)
        # Use the higher-order (two half-step) solution as the accepted state.
        return u_two_half, float(err_tensor.item())

    def _propose_next_dt(self, err: float) -> float:
        """PI-controller dt update.

        ``dt_new = dt * safety * (tol/err)^(alpha/(p+1)) * (err_prev/err)^(beta/(p+1))``
        clamped to ``[dt_min, dt_max]``.  ``p`` is the scheme's order of
        accuracy.  ``err`` is the scaled error norm returned by
        :meth:`_step_doubling_error`.
        """
        cfg = self.config
        order = max(self._scheme_order, 1)
        # Guard against zero/near-zero error (perfectly smooth region):
        # treat as success at dt_max growth ceiling.
        if err <= 0.0:
            new_dt = cfg.dt_max
            self._prev_err_ratio = 1.0
        else:
            inv = 1.0 / float(order + 1)
            tol_ratio = cfg.error_tolerance / err
            growth = (tol_ratio ** (cfg.pi_alpha * inv)) * (
                self._prev_err_ratio ** (cfg.pi_beta * inv)
            )
            new_dt = cfg.safety_factor * self.dt * growth
            self._prev_err_ratio = tol_ratio
        # Clamp to configured bounds.
        return max(cfg.dt_min, min(cfg.dt_max, new_dt))

    def _integrate_adaptive(
        self,
        u0: Tensor,
        rhs_fn: Any,
    ) -> list[tuple[Tensor, float]]:
        """PI-controlled adaptive-dt integration loop."""
        cfg = self.config
        u = u0.clone()
        t = cfg.t_start
        snapshots: list[tuple[Tensor, float]] = [(u.clone(), t)]
        step_count = 0
        rejected = 0
        self._prev_err_ratio = 1.0

        logger.info(
            "time_integration_start",
            method=self.__class__.__name__,
            t_start=cfg.t_start,
            t_end=cfg.t_end,
            dt=self.dt,
            adaptive=True,
            dt_min=cfg.dt_min,
            dt_max=cfg.dt_max,
            error_tolerance=cfg.error_tolerance,
        )

        while t < cfg.t_end - cfg.t_end_epsilon and step_count < cfg.max_steps:
            # Don't overshoot t_end on the trial step
            self.dt = min(self.dt, cfg.t_end - t)
            u_trial, err = self._step_doubling_error(u, t, rhs_fn)

            if err <= cfg.error_tolerance or self.dt <= cfg.dt_min * (1.0 + cfg.t_end_epsilon):
                # Accept (always accept at dt_min to avoid infinite shrink)
                u = u_trial
                t = t + self.dt
                step_count += 1
                self.dt = self._propose_next_dt(err)
                if step_count % cfg.save_interval == 0:
                    snapshots.append((u.clone(), t))
            else:
                # Reject: shrink dt and retry
                rejected += 1
                self.dt = self._propose_next_dt(err)

        if snapshots[-1][1] != t:
            snapshots.append((u.clone(), t))

        logger.info(
            "time_integration_complete",
            n_steps=step_count,
            n_rejected=rejected,
            t_final=t,
            n_snapshots=len(snapshots),
            dt_final=self.dt,
        )

        return snapshots


class ForwardEuler(TimeStepper):
    """Forward Euler method (explicit, first-order).

    u^{n+1} = u^n + dt * f(u^n, t^n)

    Simple but requires small dt for stability (CFL condition).
    """

    def step(
        self,
        u: Tensor,
        t: float,
        rhs_fn: Any,
    ) -> tuple[Tensor, float]:
        """One forward Euler step."""
        dudt = rhs_fn(u, t)
        u_new = u + self.dt * dudt
        return u_new, t + self.dt


class RK4(TimeStepper):
    """Classical fourth-order Runge-Kutta method.

    Four-stage explicit method with O(dt^4) accuracy.
    Good balance of accuracy and computational cost for
    non-stiff problems.
    """

    def step(
        self,
        u: Tensor,
        t: float,
        rhs_fn: Any,
    ) -> tuple[Tensor, float]:
        """One RK4 step."""
        k1 = rhs_fn(u, t)
        k2 = rhs_fn(u + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = rhs_fn(u + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = rhs_fn(u + self.dt * k3, t + self.dt)

        u_new = u + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return u_new, t + self.dt


class CrankNicolson(TimeStepper):
    """Crank-Nicolson method (implicit, second-order, A-stable).

    u^{n+1} = u^n + dt/2 * (f(u^n, t^n) + f(u^{n+1}, t^{n+1}))

    Solved via fixed-point iteration. A-stable, making it suitable
    for stiff problems (e.g., diffusion-dominated PDEs).
    """

    def __init__(
        self,
        config: TimeSteppingConfig,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> None:
        """Initialize Crank-Nicolson stepper.

        Args:
            config: Time-stepping configuration.
            max_iterations: Maximum fixed-point iterations per step.
            tolerance: Convergence tolerance for fixed-point iteration.

        """
        super().__init__(config)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def step(
        self,
        u: Tensor,
        t: float,
        rhs_fn: Any,
    ) -> tuple[Tensor, float]:
        """One Crank-Nicolson step via fixed-point iteration."""
        f_n = rhs_fn(u, t)
        t_new = t + self.dt

        # Initial guess: forward Euler
        u_new = u + self.dt * f_n

        for _iteration in range(self.max_iterations):
            f_new = rhs_fn(u_new, t_new)
            u_next = u + 0.5 * self.dt * (f_n + f_new)

            # Check convergence
            residual = torch.norm(u_next - u_new).item()
            u_new = u_next

            if residual < self.tolerance:
                break
        else:
            logger.warning(
                "crank_nicolson_not_converged",
                max_iterations=self.max_iterations,
                final_residual=residual,
                t=t,
            )

        return u_new, t_new


def create_time_stepper(config: TimeSteppingConfig) -> TimeStepper:
    """Factory function for creating time steppers from config.

    Args:
        config: Time-stepping configuration.

    Returns:
        Configured TimeStepper instance.

    Raises:
        ValueError: If unknown method specified.

    """
    method_map: dict[TimeSteppingMethod, type[TimeStepper]] = {
        TimeSteppingMethod.FORWARD_EULER: ForwardEuler,
        TimeSteppingMethod.RK4: RK4,
        TimeSteppingMethod.CRANK_NICOLSON: CrankNicolson,
    }

    stepper_cls = method_map.get(config.method)
    if stepper_cls is None:
        msg = f"Unknown time-stepping method: {config.method}. Available: {list(method_map.keys())}"
        raise ValueError(msg)

    return stepper_cls(config)
