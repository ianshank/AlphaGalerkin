"""Aerodynamic force and moment models.

Provides configurable aerodynamic models for computing forces and
torques on vehicles given their state and atmospheric conditions.

Models:
- SimpleAeroModel: Drag-only model for ballistic validation
- TabularAeroModel: Cd/Cl lookup tables with bilinear interpolation

All models registered via create_registry pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog
import torch
from torch import Tensor

from src.intercept.atmosphere import ISAAtmosphere
from src.intercept.config import ThreatConfig
from src.intercept.dynamics import RigidBodyState
from src.intercept.frames import FrameTransform
from src.templates.registry import create_registry

logger = structlog.get_logger(__name__)


class AeroModel(ABC):
    """Abstract base class for aerodynamic models.

    Computes aerodynamic forces and moments given vehicle state
    and atmospheric conditions.
    """

    @abstractmethod
    def compute_forces(
        self,
        state: RigidBodyState,
        atmosphere: ISAAtmosphere,
    ) -> tuple[Tensor, Tensor]:
        """Compute aerodynamic force and torque.

        Args:
            state: Current vehicle state.
            atmosphere: Atmosphere model for density/temperature.

        Returns:
            (force_ned, torque_body): Force in NED frame (..., 3)
                and torque in body frame (..., 3).

        """
        ...

    @abstractmethod
    def max_g_load(self, state: RigidBodyState) -> Tensor:
        """Compute maximum achievable g-load at current state.

        Args:
            state: Current vehicle state.

        Returns:
            Maximum g-load (scalar or ...).

        """
        ...


# Registry for aerodynamic models
AeroModelRegistry, register_aero_model = create_registry("AeroModel", AeroModel)


@register_aero_model("simple")
class SimpleAeroModel(AeroModel):
    """Drag-only aerodynamic model.

    Computes: F_drag = -0.5 * rho * V^2 * Cd * Sref * v_hat

    Suitable for ballistic trajectory validation and simple threats.
    No lift or moment computation.
    """

    def __init__(
        self,
        cd: float = 0.3,
        reference_area: float = 0.5,
        mass: float = 200.0,
        max_g: float = 5.0,
    ) -> None:
        self.cd = cd
        self.reference_area = reference_area
        self.mass = mass
        self._max_g = max_g

    @classmethod
    def from_config(cls, config: ThreatConfig) -> SimpleAeroModel:
        """Create from ThreatConfig."""
        return cls(
            cd=config.cd_0,
            reference_area=config.reference_area_m2,
            mass=config.mass_kg,
            max_g=config.max_g,
        )

    def compute_forces(
        self,
        state: RigidBodyState,
        atmosphere: ISAAtmosphere,
    ) -> tuple[Tensor, Tensor]:
        """Compute drag force in NED frame."""
        speed = state.speed
        altitude = state.altitude
        rho = atmosphere.density(altitude)
        q = 0.5 * rho * speed**2

        # Drag magnitude
        drag_mag = q * self.cd * self.reference_area

        # Drag direction: opposes velocity
        v_hat = state.velocity / (speed.unsqueeze(-1) + 1e-12)
        force_ned = -drag_mag.unsqueeze(-1) * v_hat

        # No aerodynamic torques in simple model
        torque_body = torch.zeros_like(state.angular_velocity)

        return force_ned, torque_body

    def max_g_load(self, state: RigidBodyState) -> Tensor:
        """Return configured maximum g-load."""
        return torch.tensor(self._max_g, device=state.position.device, dtype=state.position.dtype)


@register_aero_model("tabular")
class TabularAeroModel(AeroModel):
    """Tabular aerodynamic model with Cd/Cl lookup.

    Interpolates drag and lift coefficients from tables indexed by
    angle of attack (alpha) and Mach number. Computes lift and drag
    forces in the wind frame, then transforms to NED.
    """

    def __init__(
        self,
        alpha_table: Tensor | None = None,
        mach_table: Tensor | None = None,
        cd_table: Tensor | None = None,
        cl_table: Tensor | None = None,
        cd_0: float = 0.3,
        cl_alpha: float = 2.0,
        reference_area: float = 0.5,
        reference_length: float = 1.0,
        mass: float = 200.0,
        max_g: float = 5.0,
    ) -> None:
        """Initialize tabular aero model.

        If tables are not provided, uses linear approximations:
        - Cd = cd_0 + k * alpha^2 (drag polar)
        - Cl = cl_alpha * alpha (linear lift)

        Args:
            alpha_table: AoA breakpoints in radians (N_alpha,).
            mach_table: Mach breakpoints (N_mach,).
            cd_table: Drag coefficients (N_alpha, N_mach).
            cl_table: Lift coefficients (N_alpha, N_mach).
            cd_0: Zero-lift drag coefficient (fallback).
            cl_alpha: Lift curve slope (fallback).
            reference_area: Reference area in m^2.
            reference_length: Reference length in m.
            mass: Vehicle mass in kg.
            max_g: Maximum structural g-load.

        """
        self.alpha_table = alpha_table
        self.mach_table = mach_table
        self.cd_table = cd_table
        self.cl_table = cl_table
        self.cd_0 = cd_0
        self.cl_alpha = cl_alpha
        self.reference_area = reference_area
        self.reference_length = reference_length
        self.mass = mass
        self._max_g = max_g
        self._has_tables = (
            alpha_table is not None
            and mach_table is not None
            and cd_table is not None
            and cl_table is not None
        )

    @classmethod
    def from_config(cls, config: ThreatConfig) -> TabularAeroModel:
        """Create from ThreatConfig."""
        return cls(
            cd_0=config.cd_0,
            cl_alpha=config.cl_alpha,
            reference_area=config.reference_area_m2,
            reference_length=config.reference_length_m,
            mass=config.mass_kg,
            max_g=config.max_g,
        )

    def _compute_alpha(self, state: RigidBodyState) -> Tensor:
        """Compute angle of attack from velocity and orientation.

        AoA is the angle between the body x-axis (forward) and the
        velocity vector, projected onto the body xz-plane.
        """
        # Transform velocity to body frame
        v_body = FrameTransform.ned_to_body(state.velocity, state.quaternion)
        vx = v_body[..., 0]
        vz = v_body[..., 2]
        alpha = torch.atan2(-vz, vx + 1e-12)
        return alpha

    def _lookup_coefficients(self, alpha: Tensor, mach: Tensor) -> tuple[Tensor, Tensor]:
        """Look up Cd, Cl from tables or analytical fallback."""
        if self._has_tables:
            assert self.alpha_table is not None
            assert self.mach_table is not None
            assert self.cd_table is not None
            assert self.cl_table is not None

            alpha_table = self.alpha_table.to(device=alpha.device, dtype=alpha.dtype)
            mach_table = self.mach_table.to(device=alpha.device, dtype=alpha.dtype)
            cd_table = self.cd_table.to(device=alpha.device, dtype=alpha.dtype)
            cl_table = self.cl_table.to(device=alpha.device, dtype=alpha.dtype)

            # Bilinear interpolation
            a_idx = torch.searchsorted(
                alpha_table, alpha.clamp(alpha_table[0], alpha_table[-1])
            ).clamp(1, len(alpha_table) - 1)
            m_idx = torch.searchsorted(mach_table, mach.clamp(mach_table[0], mach_table[-1])).clamp(
                1, len(mach_table) - 1
            )

            a_frac = (alpha - alpha_table[a_idx - 1]) / (
                alpha_table[a_idx] - alpha_table[a_idx - 1] + 1e-12
            )
            m_frac = (mach - mach_table[m_idx - 1]) / (
                mach_table[m_idx] - mach_table[m_idx - 1] + 1e-12
            )
            a_frac = a_frac.clamp(0, 1)
            m_frac = m_frac.clamp(0, 1)

            # Bilinear interpolation on 2D table
            cd_00 = cd_table[a_idx - 1, m_idx - 1]
            cd_10 = cd_table[a_idx, m_idx - 1]
            cd_01 = cd_table[a_idx - 1, m_idx]
            cd_11 = cd_table[a_idx, m_idx]
            cd = (
                cd_00 * (1 - a_frac) * (1 - m_frac)
                + cd_10 * a_frac * (1 - m_frac)
                + cd_01 * (1 - a_frac) * m_frac
                + cd_11 * a_frac * m_frac
            )

            cl_00 = cl_table[a_idx - 1, m_idx - 1]
            cl_10 = cl_table[a_idx, m_idx - 1]
            cl_01 = cl_table[a_idx - 1, m_idx]
            cl_11 = cl_table[a_idx, m_idx]
            cl = (
                cl_00 * (1 - a_frac) * (1 - m_frac)
                + cl_10 * a_frac * (1 - m_frac)
                + cl_01 * (1 - a_frac) * m_frac
                + cl_11 * a_frac * m_frac
            )

            return cd, cl

        # Analytical fallback: drag polar + linear lift
        k = 1.0 / (torch.pi * 5.0 * 0.8)  # induced drag factor
        cd = self.cd_0 + k * alpha**2
        cl = self.cl_alpha * alpha
        return cd, cl

    def compute_forces(
        self,
        state: RigidBodyState,
        atmosphere: ISAAtmosphere,
    ) -> tuple[Tensor, Tensor]:
        """Compute lift and drag forces in NED frame."""
        speed = state.speed
        altitude = state.altitude
        alpha = self._compute_alpha(state)
        mach = atmosphere.mach_number(speed, altitude)
        rho = atmosphere.density(altitude)
        q = 0.5 * rho * speed**2

        cd, cl = self._lookup_coefficients(alpha, mach)

        drag_mag = q * cd * self.reference_area
        lift_mag = q * cl * self.reference_area

        # Drag: opposes velocity
        v_hat = state.velocity / (speed.unsqueeze(-1) + 1e-12)
        drag_ned = -drag_mag.unsqueeze(-1) * v_hat

        # Lift: perpendicular to velocity in vertical plane
        # Lift direction: cross(v_hat, [0,0,1]) x v_hat (simplified)
        down = torch.zeros_like(state.velocity)
        down[..., 2] = 1.0
        lift_dir = torch.cross(torch.cross(v_hat, down, dim=-1), v_hat, dim=-1)
        lift_dir_norm = torch.norm(lift_dir, dim=-1, keepdim=True) + 1e-12
        lift_dir = lift_dir / lift_dir_norm
        lift_ned = lift_mag.unsqueeze(-1) * lift_dir

        force_ned = drag_ned + lift_ned

        # Pitching moment (simplified)
        cm = 0.0  # zero for now -- extend with moment table
        moment = q * cm * self.reference_area * self.reference_length
        torque_body = torch.zeros_like(state.angular_velocity)
        torque_body[..., 1] = moment  # pitch axis

        return force_ned, torque_body

    def max_g_load(self, state: RigidBodyState) -> Tensor:
        """Return configured maximum g-load."""
        return torch.tensor(self._max_g, device=state.position.device, dtype=state.position.dtype)
