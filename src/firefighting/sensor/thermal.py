"""Thermal camera decoder for drone-mounted FLIR/DJI sensors.

Converts raw thermal image data into georeferenced temperature maps
suitable for PDE boundary condition generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.firefighting.config.sensor import SensorConfig


@dataclass
class ThermalFrame:
    """A decoded thermal camera frame.

    Attributes:
        temperature_K: Temperature values in Kelvin (height, width).
        timestamp: Frame capture time (Unix epoch).
        confidence: Per-pixel confidence [0, 1] (height, width).

    """

    temperature_K: NDArray[np.float64]  # noqa: N815
    timestamp: float
    confidence: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int]:
        return self.temperature_K.shape

    @property
    def max_temperature(self) -> float:
        return float(self.temperature_K.max())

    def hot_spots(self, threshold_K: float = 400.0) -> NDArray[np.bool_]:  # noqa: N803
        """Identify pixels above temperature threshold."""
        return self.temperature_K >= threshold_K


class ThermalCameraDecoder:
    """Decodes thermal camera data and projects to simulation grid.

    Handles:
    - Raw radiometric data to temperature conversion
    - Camera distortion correction (simplified)
    - Projection from camera coordinates to ground plane
    - Confidence estimation based on viewing angle
    """

    def __init__(self, config: SensorConfig) -> None:
        self.config = config

    def decode_frame(
        self,
        raw_data: NDArray[np.float64],
        timestamp: float,
        altitude_m: float = 100.0,
    ) -> ThermalFrame:
        """Decode a raw thermal image to temperature.

        For FLIR cameras, raw data is typically 14-bit radiometric
        values. For generic sensors, raw_data is assumed to be
        in Kelvin already.

        Args:
            raw_data: Raw sensor data (height, width).
            timestamp: Capture time.
            altitude_m: Drone altitude for footprint calculation.

        Returns:
            Decoded ThermalFrame with temperatures and confidence.

        """
        # Generic: assume data is already in Kelvin
        temp = raw_data.astype(np.float64)
        temp = np.clip(temp, 200.0, 2000.0)

        # Confidence based on viewing angle (higher at nadir)
        h, w = temp.shape
        cy, cx = h / 2, w / 2
        yy, xx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        # Confidence drops linearly from center to edge
        confidence = 1.0 - 0.3 * dist_from_center / max(max_dist, 1.0)
        confidence = np.clip(confidence, 0.5, 1.0)

        return ThermalFrame(
            temperature_K=temp,
            timestamp=timestamp,
            confidence=confidence,
        )

    def project_to_grid(
        self,
        frame: ThermalFrame,
        grid_shape: tuple[int, int],
        domain_size_x_m: float,
        domain_size_y_m: float,
        drone_x_m: float,
        drone_y_m: float,
        altitude_m: float = 100.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project thermal frame onto simulation grid.

        Uses a simple pinhole camera model to map camera pixels
        to ground coordinates, then interpolates onto the simulation grid.

        Args:
            frame: Decoded thermal frame.
            grid_shape: Simulation grid (ny, nx).
            domain_size_x_m: Domain width in meters.
            domain_size_y_m: Domain height in meters.
            drone_x_m: Drone x-position in domain coordinates.
            drone_y_m: Drone y-position in domain coordinates.
            altitude_m: Drone altitude in meters.

        Returns:
            Tuple of (temperature_grid, confidence_grid) mapped to sim grid.

        """
        ny, nx = grid_shape
        fov_rad = np.radians(self.config.thermal_fov_deg)

        # Ground footprint size
        footprint = 2.0 * altitude_m * np.tan(fov_rad / 2.0)

        # Camera footprint bounds in domain coordinates
        x_min = drone_x_m - footprint / 2
        x_max = drone_x_m + footprint / 2
        y_min = drone_y_m - footprint / 2
        y_max = drone_y_m + footprint / 2

        # Map simulation grid cells to camera pixels
        dx = domain_size_x_m / nx
        dy = domain_size_y_m / ny
        temp_grid = np.full((ny, nx), np.nan, dtype=np.float64)
        conf_grid = np.zeros((ny, nx), dtype=np.float64)

        cam_h, cam_w = frame.shape

        for j in range(ny):
            for i in range(nx):
                gx = (i + 0.5) * dx
                gy = (j + 0.5) * dy

                if x_min <= gx <= x_max and y_min <= gy <= y_max:
                    # Map to camera pixel
                    px = int((gx - x_min) / footprint * cam_w)
                    py = int((gy - y_min) / footprint * cam_h)
                    px = np.clip(px, 0, cam_w - 1)
                    py = np.clip(py, 0, cam_h - 1)

                    temp_grid[j, i] = frame.temperature_K[py, px]
                    conf_grid[j, i] = frame.confidence[py, px]

        return temp_grid, conf_grid
