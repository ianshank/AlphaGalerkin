"""FIRE II capsule geometry and trajectory data.

The FIRE II (Flight Investigation of Reentry Environment)
experiment was launched in April 1965. The forebody is a
sphere-cone with:
- Nose radius: 0.9347 m
- Cone half-angle: 33 degrees

Reference: Cauchon, D.L. (1966) "Radiative heating results from
the FIRE II flight experiment at a reentry velocity of 11.4 km/s."
NASA TM X-1402.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.trajectory import TrajectoryConfig, TrajectoryPoint


def fire2_trajectory() -> TrajectoryConfig:
    """Standard FIRE II trajectory configuration with key time points."""
    return TrajectoryConfig(
        name="fire2",
        vehicle_name="FIRE_II",
        nose_radius_m=0.9347,
        cone_half_angle_deg=33.0,
        points=[
            TrajectoryPoint(
                name="t1634",
                time_s=1634.0,
                altitude_km=76.42,
                velocity_m_s=11360.0,
                density_kg_m3=3.72e-5,
                temperature_K=195.0,
                mach=38.9,
                expected_heat_flux_W_m2=1.0e6,
            ),
            TrajectoryPoint(
                name="t1636",
                time_s=1636.0,
                altitude_km=53.04,
                velocity_m_s=11360.0,
                density_kg_m3=4.855e-4,
                temperature_K=210.0,
                mach=35.7,
                expected_heat_flux_W_m2=1.1e7,
            ),
            TrajectoryPoint(
                name="t1637_5",
                time_s=1637.5,
                altitude_km=48.39,
                velocity_m_s=11280.0,
                density_kg_m3=1.08e-3,
                temperature_K=265.0,
                mach=31.9,
                expected_heat_flux_W_m2=1.3e7,
            ),
            TrajectoryPoint(
                name="t1643",
                time_s=1643.0,
                altitude_km=36.20,
                velocity_m_s=9890.0,
                density_kg_m3=6.55e-3,
                temperature_K=239.0,
                mach=29.0,
                expected_heat_flux_W_m2=1.2e7,
            ),
            TrajectoryPoint(
                name="t1645",
                time_s=1645.0,
                altitude_km=33.52,
                velocity_m_s=8430.0,
                density_kg_m3=1.02e-2,
                temperature_K=234.0,
                mach=25.0,
                expected_heat_flux_W_m2=8.0e6,
            ),
        ],
    )


@dataclass
class SphereConeGeometry:
    """Sphere-cone forebody geometry (axisymmetric).

    The body surface is a spherical nose cap joined tangentially
    to a conical afterbody.
    """

    nose_radius: float  # meters
    cone_half_angle_deg: float  # degrees

    @property
    def cone_half_angle_rad(self) -> float:
        return np.radians(self.cone_half_angle_deg)

    def surface_points(
        self, n_points: int = 100, max_s: float = 2.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate surface coordinates in (x, r) plane.

        Args:
            n_points: Number of surface points.
            max_s: Maximum distance along surface in nose radii.

        Returns:
            Tuple of (x, r) coordinates in meters.

        """
        s = np.linspace(0, max_s * self.nose_radius, n_points)
        theta_c = self.cone_half_angle_rad

        # Sphere-cone junction angle
        theta_junction = np.pi / 2 - theta_c

        # Points on spherical nose
        theta = s / self.nose_radius  # Arc angle
        nose_mask = theta <= theta_junction

        x = np.zeros_like(s)
        r = np.zeros_like(s)

        # Spherical nose
        x[nose_mask] = self.nose_radius * (1 - np.cos(theta[nose_mask]))
        r[nose_mask] = self.nose_radius * np.sin(theta[nose_mask])

        # Conical afterbody
        x_junction = self.nose_radius * (1 - np.sin(theta_c))
        r_junction = self.nose_radius * np.cos(theta_c)
        s_junction = self.nose_radius * theta_junction

        cone_mask = ~nose_mask
        ds = s[cone_mask] - s_junction
        x[cone_mask] = x_junction + ds * np.cos(theta_c)
        r[cone_mask] = r_junction + ds * np.sin(theta_c)

        return x, r

    def wall_normal(
        self, x: NDArray[np.float64], r: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute outward wall normal vectors.

        Args:
            x: Axial coordinates of surface points.
            r: Radial coordinates of surface points.

        Returns:
            Tuple of (nx, nr) normal components (unit vectors).

        """
        # Central differences for surface tangent
        n = len(x)
        tx = np.zeros(n)
        tr = np.zeros(n)

        tx[1:-1] = x[2:] - x[:-2]
        tr[1:-1] = r[2:] - r[:-2]
        tx[0] = x[1] - x[0]
        tr[0] = r[1] - r[0]
        tx[-1] = x[-1] - x[-2]
        tr[-1] = r[-1] - r[-2]

        # Outward normal (rotate tangent 90 degrees CCW)
        mag = np.sqrt(tx**2 + tr**2)
        mag = np.maximum(mag, 1e-10)

        nx = -tr / mag
        nr = tx / mag

        return nx, nr
