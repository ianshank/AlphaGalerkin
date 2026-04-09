"""Coordinate frame transforms and quaternion mathematics.

All operations use PyTorch tensors for GPU acceleration and
autodiff compatibility with physics-informed training losses.

Supported frames:
- NED (North-East-Down): local tangent plane, standard for guidance
- ENU (East-North-Up): common in mapping/GIS
- ECEF (Earth-Centered Earth-Fixed): global reference
- Body: vehicle-fixed frame aligned with body axes

Quaternion convention: [w, x, y, z] (scalar-first, Hamilton).
"""

from __future__ import annotations

import math

import structlog
import torch
from torch import Tensor

logger = structlog.get_logger(__name__)

# WGS84 ellipsoid constants
WGS84_A = 6_378_137.0  # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # first eccentricity squared


class QuaternionOps:
    """Pure-function quaternion operations on PyTorch tensors.

    Quaternion layout: [w, x, y, z] (scalar-first, Hamilton convention).
    All functions accept batched inputs with shape (..., 4).
    """

    @staticmethod
    def identity(
        batch_shape: tuple[int, ...] = (),
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Return identity quaternion(s) [1, 0, 0, 0]."""
        q = torch.zeros(*batch_shape, 4, device=device, dtype=dtype)
        q[..., 0] = 1.0
        return q

    @staticmethod
    def normalize(q: Tensor) -> Tensor:
        """Normalize quaternion to unit length."""
        return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def conjugate(q: Tensor) -> Tensor:
        """Quaternion conjugate (inverse for unit quaternions)."""
        conj = q.clone()
        conj[..., 1:] = -conj[..., 1:]
        return conj

    @staticmethod
    def multiply(q1: Tensor, q2: Tensor) -> Tensor:
        """Hamilton quaternion product q1 * q2.

        Args:
            q1: Left quaternion (..., 4).
            q2: Right quaternion (..., 4).

        Returns:
            Product quaternion (..., 4).

        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def rotate_vector(q: Tensor, v: Tensor) -> Tensor:
        """Rotate vector v by quaternion q: q * v * q_conj.

        Args:
            q: Unit quaternion (..., 4).
            v: Vector (..., 3).

        Returns:
            Rotated vector (..., 3).

        """
        # Expand v to quaternion [0, vx, vy, vz]
        v_quat = torch.zeros(*v.shape[:-1], 4, device=v.device, dtype=v.dtype)
        v_quat[..., 1:] = v

        q_conj = QuaternionOps.conjugate(q)
        result = QuaternionOps.multiply(QuaternionOps.multiply(q, v_quat), q_conj)
        return result[..., 1:]

    @staticmethod
    def from_euler(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
        """Convert Euler angles (ZYX convention) to quaternion.

        Args:
            roll: Roll angle in radians (...).
            pitch: Pitch angle in radians (...).
            yaw: Yaw angle in radians (...).

        Returns:
            Quaternion [w, x, y, z] (..., 4).

        """
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def to_euler(q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Convert quaternion to Euler angles (ZYX convention).

        Returns:
            (roll, pitch, yaw) each with shape (...).

        """
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        # Roll (x-axis rotation)
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr, cosr)

        # Pitch (y-axis rotation) -- clamp for numerical safety
        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        # Yaw (z-axis rotation)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny, cosy)

        return roll, pitch, yaw

    @staticmethod
    def to_rotation_matrix(q: Tensor) -> Tensor:
        """Convert quaternion to 3x3 rotation matrix.

        Args:
            q: Unit quaternion (..., 4).

        Returns:
            Rotation matrix (..., 3, 3).

        """
        q = QuaternionOps.normalize(q)
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y - w * z)
        r02 = 2.0 * (x * z + w * y)
        r10 = 2.0 * (x * y + w * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z - w * x)
        r20 = 2.0 * (x * z - w * y)
        r21 = 2.0 * (y * z + w * x)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        row0 = torch.stack([r00, r01, r02], dim=-1)
        row1 = torch.stack([r10, r11, r12], dim=-1)
        row2 = torch.stack([r20, r21, r22], dim=-1)

        return torch.stack([row0, row1, row2], dim=-2)

    @staticmethod
    def slerp(q0: Tensor, q1: Tensor, t: float) -> Tensor:
        """Spherical linear interpolation between quaternions.

        Args:
            q0: Start quaternion (..., 4).
            q1: End quaternion (..., 4).
            t: Interpolation parameter in [0, 1].

        Returns:
            Interpolated quaternion (..., 4).

        """
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
        # Ensure shortest path
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.abs(dot)

        # Clamp for numerical safety
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)

        # Fall back to lerp for small angles
        small_angle = (theta.abs() < 1e-6).squeeze(-1)
        sin_theta = torch.sin(theta)

        s0 = torch.sin((1.0 - t) * theta) / (sin_theta + 1e-12)
        s1 = torch.sin(t * theta) / (sin_theta + 1e-12)

        result = s0 * q0 + s1 * q1

        # Lerp fallback for near-zero angles
        if small_angle.any():
            lerp_result = (1.0 - t) * q0 + t * q1
            lerp_result = QuaternionOps.normalize(lerp_result)
            result = torch.where(small_angle.unsqueeze(-1), lerp_result, result)

        return QuaternionOps.normalize(result)


class FrameTransform:
    """Coordinate frame transformations.

    All transforms operate on PyTorch tensors and support batched inputs.
    """

    @staticmethod
    def ned_to_enu(v_ned: Tensor) -> Tensor:
        """Convert NED vector to ENU.

        NED(n,e,d) -> ENU(e,n,-d).
        """
        return torch.stack([v_ned[..., 1], v_ned[..., 0], -v_ned[..., 2]], dim=-1)

    @staticmethod
    def enu_to_ned(v_enu: Tensor) -> Tensor:
        """Convert ENU vector to NED.

        ENU(e,n,u) -> NED(n,e,-u).
        """
        return torch.stack([v_enu[..., 1], v_enu[..., 0], -v_enu[..., 2]], dim=-1)

    @staticmethod
    def body_to_ned(v_body: Tensor, q_body_to_ned: Tensor) -> Tensor:
        """Transform vector from body frame to NED frame.

        Args:
            v_body: Vector in body frame (..., 3).
            q_body_to_ned: Quaternion rotating body to NED (..., 4).

        Returns:
            Vector in NED frame (..., 3).

        """
        return QuaternionOps.rotate_vector(q_body_to_ned, v_body)

    @staticmethod
    def ned_to_body(v_ned: Tensor, q_body_to_ned: Tensor) -> Tensor:
        """Transform vector from NED frame to body frame.

        Args:
            v_ned: Vector in NED frame (..., 3).
            q_body_to_ned: Quaternion rotating body to NED (..., 4).

        Returns:
            Vector in body frame (..., 3).

        """
        q_ned_to_body = QuaternionOps.conjugate(q_body_to_ned)
        return QuaternionOps.rotate_vector(q_ned_to_body, v_ned)

    @staticmethod
    def geodetic_to_ecef(lat_rad: Tensor, lon_rad: Tensor, alt_m: Tensor) -> Tensor:
        """Convert geodetic (lat, lon, alt) to ECEF coordinates.

        Args:
            lat_rad: Latitude in radians (...).
            lon_rad: Longitude in radians (...).
            alt_m: Altitude above ellipsoid in meters (...).

        Returns:
            ECEF position (x, y, z) in meters (..., 3).

        """
        sin_lat = torch.sin(lat_rad)
        cos_lat = torch.cos(lat_rad)
        sin_lon = torch.sin(lon_rad)
        cos_lon = torch.cos(lon_rad)

        # Prime vertical radius of curvature
        n = WGS84_A / torch.sqrt(1.0 - WGS84_E2 * sin_lat**2)

        x = (n + alt_m) * cos_lat * cos_lon
        y = (n + alt_m) * cos_lat * sin_lon
        z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat

        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def ecef_to_geodetic(ecef: Tensor, n_iter: int = 5) -> tuple[Tensor, Tensor, Tensor]:
        """Convert ECEF to geodetic (lat, lon, alt).

        Uses iterative method (Bowring) for convergence.

        Args:
            ecef: ECEF position (..., 3) in meters.
            n_iter: Number of iterations for latitude convergence.

        Returns:
            (lat_rad, lon_rad, alt_m).

        """
        x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]

        lon = torch.atan2(y, x)
        p = torch.sqrt(x**2 + y**2)

        # Initial latitude estimate
        lat = torch.atan2(z, p * (1.0 - WGS84_E2))

        for _ in range(n_iter):
            sin_lat = torch.sin(lat)
            n = WGS84_A / torch.sqrt(1.0 - WGS84_E2 * sin_lat**2)
            lat = torch.atan2(z + WGS84_E2 * n * sin_lat, p)

        sin_lat = torch.sin(lat)
        cos_lat = torch.cos(lat)
        n = WGS84_A / torch.sqrt(1.0 - WGS84_E2 * sin_lat**2)

        alt = torch.where(
            cos_lat.abs() > 1e-10,
            p / cos_lat - n,
            z.abs() / sin_lat.abs() - n * (1.0 - WGS84_E2),
        )

        return lat, lon, alt

    @staticmethod
    def ned_to_ecef_rotation(lat_rad: Tensor, lon_rad: Tensor) -> Tensor:
        """Get 3x3 rotation matrix from NED to ECEF.

        Args:
            lat_rad: Latitude in radians (...).
            lon_rad: Longitude in radians (...).

        Returns:
            Rotation matrix (..., 3, 3).

        """
        sl = torch.sin(lat_rad)
        cl = torch.cos(lat_rad)
        slo = torch.sin(lon_rad)
        clo = torch.cos(lon_rad)

        # NED to ECEF rotation matrix columns
        r00 = -sl * clo
        r01 = -slo
        r02 = -cl * clo
        r10 = -sl * slo
        r11 = clo
        r12 = -cl * slo
        r20 = cl
        r21 = torch.zeros_like(cl)
        r22 = -sl

        row0 = torch.stack([r00, r01, r02], dim=-1)
        row1 = torch.stack([r10, r11, r12], dim=-1)
        row2 = torch.stack([r20, r21, r22], dim=-1)

        return torch.stack([row0, row1, row2], dim=-2)

    @staticmethod
    def relative_ned(
        pos_own: Tensor,
        pos_target: Tensor,
    ) -> Tensor:
        """Compute relative position in NED frame (flat Earth approximation).

        For short-range engagements (< 100 km), flat Earth is sufficient.

        Args:
            pos_own: Own position in NED (m) (..., 3).
            pos_target: Target position in NED (m) (..., 3).

        Returns:
            Relative position target - own in NED (..., 3).

        """
        return pos_target - pos_own

    @staticmethod
    def range_and_bearing(relative_ned: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute range, azimuth, and elevation from relative NED vector.

        Args:
            relative_ned: Relative position in NED (..., 3).

        Returns:
            (range_m, azimuth_rad, elevation_rad).
            Azimuth: clockwise from North [0, 2pi).
            Elevation: positive up from horizontal.

        """
        n, e, d = (
            relative_ned[..., 0],
            relative_ned[..., 1],
            relative_ned[..., 2],
        )
        horiz_range = torch.sqrt(n**2 + e**2)
        slant_range = torch.sqrt(n**2 + e**2 + d**2)
        azimuth = torch.atan2(e, n) % (2.0 * math.pi)
        elevation = torch.atan2(-d, horiz_range)  # positive up

        return slant_range, azimuth, elevation
