"""Radiative heat transfer for wildfire spread.

Implements simplified view-factor radiation model for
fire-to-fuel radiative heating using Stefan-Boltzmann law.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.firefighting.config.fire import FireConfig


class RadiativeHeatTransfer:
    """Radiative heat transfer between burning and unburned cells.

    Uses a simplified local radiation model where each burning cell
    radiates to its neighbors proportional to:
        q_rad = epsilon * sigma * (T_fire^4 - T_amb^4) * view_factor

    For computational efficiency on edge devices, only near-field
    radiation (adjacent cells) is computed.
    """

    def __init__(self, config: FireConfig) -> None:
        self.config = config

    def compute(
        self,
        temperature: NDArray[np.float64],
        burning_mask: NDArray[np.bool_],
        dx: float,
        dy: float,
    ) -> NDArray[np.float64]:
        """Compute radiative heat flux to each cell from burning neighbors.

        Args:
            temperature: Temperature field (ny, nx) in K.
            burning_mask: Boolean mask of burning cells.
            dx: Cell width in meters.
            dy: Cell height in meters.

        Returns:
            Radiative heat flux (ny, nx) in W/m^2.

        """
        sigma = self.config.stefan_boltzmann
        eps = self.config.emissivity
        frac = self.config.radiation_fraction
        t_amb = self.config.ambient_temperature_K

        # Emission power from burning cells
        emission = eps * sigma * (temperature**4 - t_amb**4) * frac
        emission *= burning_mask.astype(np.float64)

        # View factor approximation: fraction of hemisphere subtended
        # by adjacent cell at distance dx
        cell_area = dx * dy
        view_factor = cell_area / (2.0 * np.pi * max(dx, dy) ** 2)
        view_factor = min(view_factor, 0.5)  # Physical upper bound

        # Spread radiation to neighbors (5-point stencil)
        q_rad = np.zeros_like(temperature)
        ny, nx = temperature.shape

        # Shift and accumulate from all 4 neighbors
        if ny > 1:
            q_rad[1:, :] += emission[:-1, :]  # From above
            q_rad[:-1, :] += emission[1:, :]  # From below
        if nx > 1:
            q_rad[:, 1:] += emission[:, :-1]  # From left
            q_rad[:, :-1] += emission[:, 1:]  # From right

        return q_rad * view_factor
