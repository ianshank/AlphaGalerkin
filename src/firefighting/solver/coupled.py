"""Coupled heat + advection-diffusion fire spread solver.

Solves the energy equation with fire physics source terms:
    dT/dt = alpha * nabla^2(T) - v . nabla(T) + Q_source / (rho * cp)

where:
- alpha: thermal diffusivity
- v: wind velocity
- Q_source: heat release from fuel combustion + radiation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.firefighting.config.fire import FireConfig
from src.firefighting.config.solver import FireSolverConfig
from src.firefighting.fire.convection import ConvectiveHeatTransfer
from src.firefighting.fire.fuel import FuelModel, FuelState
from src.firefighting.fire.radiation import RadiativeHeatTransfer

logger = structlog.get_logger(__name__)


@dataclass
class FireSolverState:
    """Complete state of the fire simulation.

    Attributes:
        temperature: Temperature field (ny, nx) in K.
        fuel: Current fuel state.
        time: Current simulation time in seconds.
        step: Current timestep number.
        total_energy: Cumulative energy (for conservation tracking).

    """

    temperature: NDArray[np.float64]
    fuel: FuelState
    time: float = 0.0
    step: int = 0
    total_energy: float = 0.0


@dataclass
class FireSolverResult:
    """Result of a fire simulation run."""

    final_state: FireSolverState
    burned_area_m2: float
    max_temperature_K: float  # noqa: N815
    total_steps: int
    energy_error: float  # Relative conservation error


class CoupledFireSolver:
    """Coupled heat equation + fire physics solver.

    Uses operator splitting:
    1. Diffusion step (thermal conduction)
    2. Advection step (wind-driven transport)
    3. Source step (combustion + radiation)
    4. Fuel update (consumption)
    """

    def __init__(
        self,
        solver_config: FireSolverConfig,
        fire_config: FireConfig,
    ) -> None:
        self.solver_config = solver_config
        self.fire_config = fire_config

        self.dx = solver_config.domain_size_x_m / solver_config.nx
        self.dy = solver_config.domain_size_y_m / solver_config.ny

        # Sub-models
        self.fuel_model = FuelModel(fire_config)
        self.radiation = RadiativeHeatTransfer(fire_config)
        self.convection = ConvectiveHeatTransfer(fire_config)

    def create_initial_state(
        self,
        ignition_center: tuple[float, float] | None = None,
        ignition_radius_m: float = 10.0,
    ) -> FireSolverState:
        """Create initial state with optional circular ignition source.

        Args:
            ignition_center: (x, y) in meters. Defaults to domain center.
            ignition_radius_m: Radius of initial fire in meters.

        Returns:
            Initial FireSolverState.

        """
        ny = self.solver_config.ny
        nx = self.solver_config.nx

        # Temperature field: ambient everywhere
        temperature = np.full((ny, nx), self.fire_config.ambient_temperature_K, dtype=np.float64)

        # Fuel: uniform loading
        fuel = FuelState(
            loading=np.full((ny, nx), self.fire_config.fuel_load_kg_m2, dtype=np.float64),
            moisture=np.full((ny, nx), self.fire_config.fuel_moisture_fraction, dtype=np.float64),
            consumed=np.zeros((ny, nx), dtype=np.float64),
        )

        # Ignition
        if ignition_center is not None:
            cx, cy = ignition_center
        else:
            cx = self.solver_config.domain_size_x_m / 2
            cy = self.solver_config.domain_size_y_m / 2

        x = np.linspace(self.dx / 2, self.solver_config.domain_size_x_m - self.dx / 2, nx)
        y = np.linspace(self.dy / 2, self.solver_config.domain_size_y_m - self.dy / 2, ny)
        xx, yy = np.meshgrid(x, y)

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ignition_mask = dist <= ignition_radius_m
        temperature[ignition_mask] = (
            self.fire_config.ignition_temperature_K * self.fire_config.ignition_multiplier
        )

        total_energy = np.sum(temperature) * self.dx * self.dy

        return FireSolverState(temperature=temperature, fuel=fuel, total_energy=total_energy)

    def step(
        self,
        state: FireSolverState,
        wind_u: NDArray[np.float64],
        wind_v: NDArray[np.float64],
    ) -> FireSolverState:
        """Advance the simulation by one timestep.

        Args:
            state: Current fire state.
            wind_u: x-component of wind (ny, nx) in m/s.
            wind_v: y-component of wind (ny, nx) in m/s.

        Returns:
            Updated FireSolverState.

        """
        dt = self.solver_config.dt_s
        alpha = self.solver_config.thermal_diffusivity_m2_s
        t = state.temperature.copy()

        # 1. Diffusion: dT/dt = alpha * nabla^2(T)
        t = self._diffusion_step(t, alpha, dt)

        # 2. Advection: dT/dt = -v . nabla(T)
        dt_conv = self.convection.compute(t, wind_u, wind_v, self.dx, self.dy)
        t += dt_conv * dt

        # 3. Source terms: combustion + radiation
        burning = self.fuel_model.ignition_mask(t, state.fuel.effective_loading)
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        q_source = self.fuel_model.heat_source(t, state.fuel.effective_loading, wind_speed)
        q_rad = self.radiation.compute(t, burning, self.dx, self.dy)

        rho_cp = self.fire_config.air_heat_capacity_J_m3_K
        t += (q_source + q_rad) / rho_cp * dt

        # 4. Fuel consumption
        consumption_rate = self.fuel_model.fuel_consumption_rate(t, state.fuel.effective_loading)
        consumed_delta = consumption_rate * dt / np.maximum(state.fuel.loading, 1e-10)
        new_consumed = np.clip(state.fuel.consumed + consumed_delta, 0.0, 1.0)

        # Clamp temperature
        t = np.clip(t, 0.0, self.fire_config.max_temperature_K)

        new_fuel = FuelState(
            loading=state.fuel.loading,
            moisture=state.fuel.moisture,
            consumed=new_consumed,
        )

        total_energy = np.sum(t) * self.dx * self.dy

        return FireSolverState(
            temperature=t,
            fuel=new_fuel,
            time=state.time + dt,
            step=state.step + 1,
            total_energy=total_energy,
        )

    def run(
        self,
        state: FireSolverState,
        wind_u: NDArray[np.float64],
        wind_v: NDArray[np.float64],
        t_final: float | None = None,
    ) -> FireSolverResult:
        """Run simulation until t_final or max_steps.

        Args:
            state: Initial state.
            wind_u: Wind x-component (ny, nx).
            wind_v: Wind y-component (ny, nx).
            t_final: Final time in seconds. Uses prediction_horizon if None.

        Returns:
            FireSolverResult with final state and metrics.

        """
        if t_final is None:
            t_final = self.solver_config.prediction_horizon_s

        initial_energy = state.total_energy

        while state.time < t_final and state.step < self.solver_config.max_steps:
            state = self.step(state, wind_u, wind_v)

            if state.step % 100 == 0:
                burned_frac = np.mean(state.fuel.consumed)
                logger.debug(
                    "fire_solver_step",
                    step=state.step,
                    time=state.time,
                    max_temp=float(state.temperature.max()),
                    burned_fraction=float(burned_frac),
                )

        # Compute metrics
        burned_mask = state.fuel.consumed > self.fire_config.burn_threshold_fraction
        burned_area = float(np.sum(burned_mask)) * self.dx * self.dy
        energy_error = abs(state.total_energy - initial_energy) / max(abs(initial_energy), 1e-30)

        return FireSolverResult(
            final_state=state,
            burned_area_m2=burned_area,
            max_temperature_K=float(state.temperature.max()),
            total_steps=state.step,
            energy_error=energy_error,
        )

    def _diffusion_step(
        self,
        t: NDArray[np.float64],
        alpha: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """Explicit diffusion using 5-point Laplacian stencil."""
        ny, nx = t.shape
        lap = np.zeros_like(t)

        if nx > 2:
            lap[:, 1:-1] += (t[:, 2:] - 2 * t[:, 1:-1] + t[:, :-2]) / self.dx**2
        if ny > 2:
            lap[1:-1, :] += (t[2:, :] - 2 * t[1:-1, :] + t[:-2, :]) / self.dy**2

        # Neumann BC (zero gradient at boundaries)
        return t + alpha * dt * lap
