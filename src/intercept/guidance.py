"""Interceptor guidance laws.

Provides guidance law implementations for computing acceleration
commands to steer an interceptor toward a threat.

Laws:
- ProportionalNavigation (PN): classic N' * Vc x LOS_rate
- AugmentedPN: PN + gravity compensation
- ZeroEffortMiss: ZEM-based optimal guidance

All registered via create_registry for plug-in extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import structlog
import torch
from torch import Tensor

from src.intercept.config import STANDARD_GRAVITY_MS2 as G0
from src.intercept.config import GuidanceConfig
from src.intercept.dynamics import RigidBodyState
from src.templates.registry import create_registry

logger = structlog.get_logger(__name__)


@dataclass
class GuidanceCommand:
    """Output of a guidance law computation.

    Attributes:
        acceleration: Commanded acceleration in NED (m/s^2) (3,).
        miss_distance: Estimated miss distance (m).
        time_to_go: Estimated time to intercept (s).
        is_terminal: Whether in terminal guidance phase.
        should_breakoff: Whether to break off engagement.

    """

    acceleration: Tensor
    miss_distance: float = float("inf")
    time_to_go: float = float("inf")
    is_terminal: bool = False
    should_breakoff: bool = False


class GuidanceLaw(ABC):
    """Abstract base class for guidance laws."""

    @abstractmethod
    def compute(
        self,
        own_state: RigidBodyState,
        threat_state: RigidBodyState,
        config: GuidanceConfig,
    ) -> GuidanceCommand:
        """Compute guidance acceleration command.

        Args:
            own_state: Interceptor state.
            threat_state: Threat (target) state.
            config: Guidance configuration.

        Returns:
            GuidanceCommand with acceleration and diagnostics.

        """
        ...


GuidanceLawRegistry, register_guidance_law = create_registry("GuidanceLaw", GuidanceLaw)


def _compute_los_geometry(
    own_state: RigidBodyState, threat_state: RigidBodyState
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute line-of-sight geometry.

    Returns:
        (relative_pos, relative_vel, los_unit, closing_vel, range_m).

    """
    rel_pos = threat_state.position - own_state.position
    rel_vel = threat_state.velocity - own_state.velocity
    range_m = torch.norm(rel_pos) + 1e-12
    los_unit = rel_pos / range_m

    # Closing velocity: negative of range rate
    closing_vel = -torch.dot(rel_vel, los_unit)

    return rel_pos, rel_vel, los_unit, closing_vel, range_m


def _compute_los_rate(rel_pos: Tensor, rel_vel: Tensor, range_m: Tensor) -> Tensor:
    """Compute line-of-sight angular rate vector.

    LOS_rate = (r x v) / |r|^2
    """
    los_cross = torch.cross(rel_pos, rel_vel, dim=-1)
    return los_cross / (range_m**2 + 1e-12)


def _compute_zero_effort_miss(rel_pos: Tensor, rel_vel: Tensor, tgo: Tensor) -> Tensor:
    """Compute zero-effort miss vector.

    ZEM = relative_pos + relative_vel * tgo
    (where miss would occur with no further guidance corrections)
    """
    return rel_pos + rel_vel * tgo


def _compute_time_to_go(range_m: Tensor, closing_vel: Tensor) -> Tensor:
    """Estimate time-to-go from range and closing velocity."""
    return range_m / (closing_vel.clamp(min=1.0))


@register_guidance_law("pn")
class ProportionalNavigation(GuidanceLaw):
    """True Proportional Navigation guidance law.

    a_cmd = N' * Vc * LOS_rate

    where:
    - N' is the navigation constant (typically 3-5)
    - Vc is the closing velocity
    - LOS_rate is the line-of-sight angular rate
    """

    def compute(
        self,
        own_state: RigidBodyState,
        threat_state: RigidBodyState,
        config: GuidanceConfig,
    ) -> GuidanceCommand:
        rel_pos, rel_vel, los_unit, closing_vel, range_m = _compute_los_geometry(
            own_state, threat_state
        )
        omega = _compute_los_rate(rel_pos, rel_vel, range_m)
        tgo = _compute_time_to_go(range_m, closing_vel)

        n_prime = config.navigation_constant
        is_terminal = range_m.item() < config.terminal_range_m
        if is_terminal:
            n_prime *= config.terminal_gain_multiplier

        # True PN in 3D: a = N' * Vc * (omega x los_unit)
        # This gives acceleration perpendicular to the LOS
        accel = n_prime * closing_vel * torch.cross(omega, los_unit, dim=-1)

        # Clamp to max acceleration
        max_accel = config.max_acceleration_g * G0
        accel_mag = torch.norm(accel)
        if accel_mag.item() > max_accel:
            accel = accel * (max_accel / accel_mag)

        # Miss distance estimate
        zem = _compute_zero_effort_miss(rel_pos, rel_vel, tgo)
        miss = torch.norm(zem).item()

        # Break-off decision
        should_breakoff = miss > config.breakoff_miss_m and tgo.item() < config.breakoff_tgo_s

        return GuidanceCommand(
            acceleration=accel,
            miss_distance=miss,
            time_to_go=tgo.item(),
            is_terminal=is_terminal,
            should_breakoff=should_breakoff,
        )


@register_guidance_law("apn")
class AugmentedPN(GuidanceLaw):
    """Augmented Proportional Navigation with gravity compensation.

    a_cmd = N' * Vc * LOS_rate + (N'/2) * a_target_normal

    Adds a term to compensate for target acceleration and gravity.
    """

    def compute(
        self,
        own_state: RigidBodyState,
        threat_state: RigidBodyState,
        config: GuidanceConfig,
    ) -> GuidanceCommand:
        rel_pos, rel_vel, los_unit, closing_vel, range_m = _compute_los_geometry(
            own_state, threat_state
        )
        los_rate = _compute_los_rate(rel_pos, rel_vel, range_m)
        tgo = _compute_time_to_go(range_m, closing_vel)

        n_prime = config.navigation_constant
        is_terminal = range_m.item() < config.terminal_range_m
        if is_terminal:
            n_prime *= config.terminal_gain_multiplier

        # True PN in 3D: a = N' * Vc * (omega x los_unit)
        accel = n_prime * closing_vel * torch.cross(los_rate, los_unit, dim=-1)

        # Gravity compensation: add g in up direction (negative NED down)
        gravity_comp = torch.zeros_like(accel)
        gravity_comp[2] = -G0 * 0.5 * n_prime  # compensate own gravity
        accel = accel + gravity_comp

        # Clamp
        max_accel = config.max_acceleration_g * G0
        accel_mag = torch.norm(accel)
        if accel_mag.item() > max_accel:
            accel = accel * (max_accel / accel_mag)

        zem = _compute_zero_effort_miss(rel_pos, rel_vel, tgo)
        miss = torch.norm(zem).item()
        should_breakoff = miss > config.breakoff_miss_m and tgo.item() < config.breakoff_tgo_s

        return GuidanceCommand(
            acceleration=accel,
            miss_distance=miss,
            time_to_go=tgo.item(),
            is_terminal=is_terminal,
            should_breakoff=should_breakoff,
        )


@register_guidance_law("zem_zev")
class ZeroEffortMissGuidance(GuidanceLaw):
    """Zero Effort Miss / Zero Effort Velocity guidance.

    a_cmd = -N * (ZEM / tgo^2 + ZEV / tgo)

    Optimal for constant-velocity targets. Converges to zero miss
    as tgo -> 0 when N >= 3.
    """

    def compute(
        self,
        own_state: RigidBodyState,
        threat_state: RigidBodyState,
        config: GuidanceConfig,
    ) -> GuidanceCommand:
        rel_pos, rel_vel, los_unit, closing_vel, range_m = _compute_los_geometry(
            own_state, threat_state
        )
        tgo = _compute_time_to_go(range_m, closing_vel)

        n = config.navigation_constant
        is_terminal = range_m.item() < config.terminal_range_m

        # ZEM and ZEV
        zem = _compute_zero_effort_miss(rel_pos, rel_vel, tgo)
        zev = rel_vel  # zero-effort velocity

        # Optimal guidance: a = -N * (ZEM/tgo^2 + ZEV/tgo)
        tgo_safe = tgo.clamp(min=0.1)  # prevent division by zero
        accel = -n * (zem / tgo_safe**2 + zev / tgo_safe)

        # Clamp
        max_accel = config.max_acceleration_g * G0
        accel_mag = torch.norm(accel)
        if accel_mag.item() > max_accel:
            accel = accel * (max_accel / accel_mag)

        miss = torch.norm(zem).item()
        should_breakoff = miss > config.breakoff_miss_m and tgo.item() < config.breakoff_tgo_s

        return GuidanceCommand(
            acceleration=accel,
            miss_distance=miss,
            time_to_go=tgo.item(),
            is_terminal=is_terminal,
            should_breakoff=should_breakoff,
        )


class EnergyTracker:
    """Tracks interceptor energy state for guidance decisions.

    Monitors kinetic + potential energy and remaining delta-V
    to determine if intercept is still achievable.
    """

    def __init__(
        self,
        initial_fuel_mass_kg: float = 10.0,
        specific_impulse_s: float = 250.0,
    ) -> None:
        self.initial_fuel = initial_fuel_mass_kg
        self.fuel_remaining = initial_fuel_mass_kg
        self.isp = specific_impulse_s
        self.total_delta_v_used = 0.0

    @property
    def fuel_fraction(self) -> float:
        """Fraction of fuel remaining [0, 1]."""
        return self.fuel_remaining / (self.initial_fuel + 1e-12)

    @property
    def delta_v_remaining(self) -> float:
        """Remaining delta-V in m/s (Tsiolkovsky)."""
        if self.fuel_remaining <= 0:
            return 0.0
        # Simplified: dV = Isp * g0 * ln(m0 / mf)
        # Here we just track linearly for simplicity
        return self.isp * G0 * self.fuel_fraction

    def update(self, accel_magnitude: float, dt: float, mass: float) -> None:
        """Update fuel consumption based on commanded acceleration.

        Args:
            accel_magnitude: Acceleration magnitude (m/s^2).
            dt: Time step (s).
            mass: Current vehicle mass (kg).

        """
        thrust = accel_magnitude * mass
        fuel_rate = thrust / (self.isp * G0 + 1e-12)
        self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_rate * dt)
        self.total_delta_v_used += accel_magnitude * dt

    def is_exhausted(self) -> bool:
        """Check if fuel is depleted."""
        return self.fuel_remaining <= 0.0
