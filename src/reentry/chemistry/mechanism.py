"""Chemical mechanism base class and protocol.

Defines the interface for finite-rate chemistry models
that compute species source terms from thermodynamic state.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ChemicalMechanism(Protocol):
    """Interface for finite-rate chemistry mechanisms."""

    @property
    def n_species(self) -> int:
        """Number of chemical species."""
        ...

    @property
    def n_reactions(self) -> int:
        """Number of reactions in the mechanism."""
        ...

    @property
    def species_names(self) -> list[str]:
        """Ordered species names."""
        ...

    def source_terms(
        self,
        density: NDArray[np.float64],
        temperature_tr: NDArray[np.float64],
        temperature_ve: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute chemical source terms for each species.

        Args:
            density: Mixture density (N,) in kg/m^3.
            temperature_tr: Translational-rotational temperature (N,) in K.
            temperature_ve: Vibrational-electronic temperature (N,) in K.
            mass_fractions: Species mass fractions (N, n_species).

        Returns:
            Species production rates (N, n_species) in kg/(m^3·s).

        """
        ...

    def energy_exchange_rate(
        self,
        density: NDArray[np.float64],
        temperature_tr: NDArray[np.float64],
        temperature_ve: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute vibrational-translational energy exchange rate.

        Args:
            density: Mixture density (N,).
            temperature_tr: Translational temperature (N,).
            temperature_ve: Vibrational temperature (N,).
            mass_fractions: Species mass fractions (N, n_species).

        Returns:
            Energy transfer rate Q_TV (N,) in W/m^3.
            Positive means energy transfer from translational to vibrational.

        """
        ...
