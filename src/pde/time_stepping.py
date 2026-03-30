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

import numpy as np
import structlog
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from src.templates.config import BaseModuleConfig

logger = structlog.get_logger(__name__)


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
        description="Error tolerance for adaptive time stepping.",
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


class TimeStepper(ABC):
    """Abstract base class for time-stepping methods."""

    def __init__(self, config: TimeSteppingConfig) -> None:
        """Initialize time stepper.

        Args:
            config: Time-stepping configuration.

        """
        self.config = config
        self.dt = config.dt

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

        Args:
            u0: Initial condition.
            rhs_fn: Right-hand side function du/dt = rhs_fn(u, t).

        Returns:
            List of (solution, time) snapshots at save intervals.

        """
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
        )

        while t < self.config.t_end - 1e-12 and step_count < self.config.max_steps:
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

        for iteration in range(self.max_iterations):
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
    method_map = {
        TimeSteppingMethod.FORWARD_EULER: ForwardEuler,
        TimeSteppingMethod.RK4: RK4,
        TimeSteppingMethod.CRANK_NICOLSON: CrankNicolson,
    }

    stepper_cls = method_map.get(config.method)
    if stepper_cls is None:
        msg = f"Unknown time-stepping method: {config.method}. Available: {list(method_map.keys())}"
        raise ValueError(msg)

    return stepper_cls(config)
