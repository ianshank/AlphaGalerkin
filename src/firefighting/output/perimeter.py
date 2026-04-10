"""GeoJSON fire perimeter export.

Converts level-set fire boundary to GeoJSON format for
visualization on mapping software and transmission to
incident command ground stations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PerimeterExport:
    """Exported fire perimeter data."""

    geojson: dict
    burned_area_m2: float
    perimeter_length_m: float
    timestamp: float


class PerimeterExporter:
    """Exports fire perimeter as GeoJSON polygon.

    Extracts the zero contour of the level-set function and
    converts grid coordinates to geographic coordinates using
    a simple affine transformation.
    """

    def __init__(
        self,
        origin_lon: float = -117.0,
        origin_lat: float = 34.0,
        meters_per_degree_lon: float = 92000.0,
        meters_per_degree_lat: float = 111000.0,
    ) -> None:
        self.origin_lon = origin_lon
        self.origin_lat = origin_lat
        self.m_per_deg_lon = meters_per_degree_lon
        self.m_per_deg_lat = meters_per_degree_lat

    def export(
        self,
        burned_mask: NDArray[np.bool_],
        dx: float,
        dy: float,
        timestamp: float,
    ) -> PerimeterExport:
        """Export fire perimeter from a burned cell mask.

        Args:
            burned_mask: Boolean mask of burned cells (ny, nx).
            dx: Cell width in meters.
            dy: Cell height in meters.
            timestamp: Current simulation time.

        Returns:
            PerimeterExport with GeoJSON and metrics.

        """
        # Extract perimeter cells (burned cells adjacent to unburned)
        perimeter_mask = self._extract_perimeter(burned_mask)
        perimeter_coords = self._mask_to_coordinates(perimeter_mask, dx, dy)

        # Convert to geographic coordinates
        geo_coords = self._to_geographic(perimeter_coords)

        # Build GeoJSON
        geojson = self._build_geojson(geo_coords, timestamp)

        # Compute metrics
        burned_area = float(np.sum(burned_mask)) * dx * dy
        perimeter_length = float(len(perimeter_coords)) * max(dx, dy)

        return PerimeterExport(
            geojson=geojson,
            burned_area_m2=burned_area,
            perimeter_length_m=perimeter_length,
            timestamp=timestamp,
        )

    def to_json_string(self, export: PerimeterExport) -> str:
        """Serialize to JSON string for transmission."""
        return json.dumps(export.geojson, indent=2)

    @staticmethod
    def _extract_perimeter(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Find cells on the fire perimeter (burned adjacent to unburned)."""
        ny, nx = mask.shape
        perimeter = np.zeros_like(mask)

        for j in range(ny):
            for i in range(nx):
                if not mask[j, i]:
                    continue
                # Check if any neighbor is unburned
                if (
                    (j > 0 and not mask[j - 1, i])
                    or (j < ny - 1 and not mask[j + 1, i])
                    or (i > 0 and not mask[j, i - 1])
                    or (i < nx - 1 and not mask[j, i + 1])
                ):
                    perimeter[j, i] = True

        return perimeter

    @staticmethod
    def _mask_to_coordinates(
        mask: NDArray[np.bool_],
        dx: float,
        dy: float,
    ) -> list[tuple[float, float]]:
        """Convert mask to list of (x, y) coordinates in meters."""
        coords = []
        indices = np.argwhere(mask)
        for j, i in indices:
            coords.append(((i + 0.5) * dx, (j + 0.5) * dy))
        return coords

    def _to_geographic(
        self,
        coords: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Convert (x_m, y_m) to (longitude, latitude)."""
        geo = []
        for x_m, y_m in coords:
            lon = self.origin_lon + x_m / self.m_per_deg_lon
            lat = self.origin_lat + y_m / self.m_per_deg_lat
            geo.append((lon, lat))
        return geo

    @staticmethod
    def _build_geojson(
        coords: list[tuple[float, float]],
        timestamp: float,
    ) -> dict:
        """Build GeoJSON Feature with MultiPoint geometry."""
        if not coords:
            coordinates: list = []
        else:
            coordinates = [[lon, lat] for lon, lat in coords]

        return {
            "type": "Feature",
            "geometry": {
                "type": "MultiPoint",
                "coordinates": coordinates,
            },
            "properties": {
                "type": "fire_perimeter",
                "timestamp": timestamp,
                "n_points": len(coordinates),
            },
        }
