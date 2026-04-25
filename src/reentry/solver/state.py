"""Conservative and primitive state representations for compressible flow.

Conservative variables: [rho, rho*u, rho*v, rho*E, rho*Y1, ..., rho*Yn]
Primitive variables:    [rho, u, v, p, Y1, ..., Yn]

Conversion between the two requires the equation of state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.reentry.gas.eos import EquationOfState


@dataclass
class ConservativeState:
    """Conservative variable vector for compressible flow.

    For n_species species in 2D:
        Q = [rho, rho*u, rho*v, rho*E, rho*Y_1, ..., rho*Y_{ns-1}]

    The last species mass fraction is computed from Y_ns = 1 - sum(Y_i).

    Attributes:
        density: Density rho (N,).
        momentum_x: x-momentum rho*u (N,).
        momentum_y: y-momentum rho*v (N,).
        total_energy: Total energy rho*E (N,).
        species_density: Species densities rho*Y_i (N, ns-1).

    """

    density: NDArray[np.float64]
    momentum_x: NDArray[np.float64]
    momentum_y: NDArray[np.float64]
    total_energy: NDArray[np.float64]
    species_density: NDArray[np.float64] | None = None  # (N, ns-1)

    @property
    def n_cells(self) -> int:
        return self.density.shape[0]

    def velocity_x(self) -> NDArray[np.float64]:
        return self.momentum_x / np.maximum(self.density, 1e-30)

    def velocity_y(self) -> NDArray[np.float64]:
        return self.momentum_y / np.maximum(self.density, 1e-30)

    def specific_total_energy(self) -> NDArray[np.float64]:
        return self.total_energy / np.maximum(self.density, 1e-30)

    def kinetic_energy(self) -> NDArray[np.float64]:
        u = self.velocity_x()
        v = self.velocity_y()
        return 0.5 * (u**2 + v**2)

    def specific_internal_energy(self) -> NDArray[np.float64]:
        return self.specific_total_energy() - self.kinetic_energy()

    def mass_fractions(self, n_species: int) -> NDArray[np.float64]:
        """Extract mass fractions including the last species by constraint."""
        yi = np.zeros((self.n_cells, n_species), dtype=np.float64)
        if self.species_density is not None and n_species > 1:
            for i in range(n_species - 1):
                yi[:, i] = self.species_density[:, i] / np.maximum(self.density, 1e-30)
            yi[:, -1] = 1.0 - np.sum(yi[:, :-1], axis=1)
        else:
            yi[:, 0] = 1.0  # Single species
        return np.clip(yi, 0.0, 1.0)

    def to_array(self) -> NDArray[np.float64]:
        """Stack into a single array (N, n_vars)."""
        components = [
            self.density[:, np.newaxis],
            self.momentum_x[:, np.newaxis],
            self.momentum_y[:, np.newaxis],
            self.total_energy[:, np.newaxis],
        ]
        if self.species_density is not None:
            components.append(self.species_density)
        return np.concatenate(components, axis=1)

    @classmethod
    def from_array(cls, q: NDArray[np.float64], n_species: int = 1) -> ConservativeState:
        """Reconstruct from a stacked array."""
        species = q[:, 4:] if n_species > 1 and q.shape[1] > 4 else None
        return cls(
            density=q[:, 0],
            momentum_x=q[:, 1],
            momentum_y=q[:, 2],
            total_energy=q[:, 3],
            species_density=species,
        )


@dataclass
class PrimitiveState:
    """Primitive variable vector.

    W = [rho, u, v, p, Y_1, ..., Y_ns]
    """

    density: NDArray[np.float64]
    velocity_x: NDArray[np.float64]
    velocity_y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    mass_fractions: NDArray[np.float64]  # (N, ns)

    @property
    def n_cells(self) -> int:
        return self.density.shape[0]

    def to_conservative(self, eos: EquationOfState) -> ConservativeState:
        """Convert to conservative variables using EOS."""
        rho = self.density
        u = self.velocity_x
        v = self.velocity_y
        p = self.pressure
        e_int = eos.internal_energy(rho, p, self.mass_fractions)
        ke = 0.5 * (u**2 + v**2)
        rho_e = rho * (e_int + ke)

        species_density = None
        ns = self.mass_fractions.shape[1]
        if ns > 1:
            species_density = rho[:, np.newaxis] * self.mass_fractions[:, :-1]

        return ConservativeState(
            density=rho,
            momentum_x=rho * u,
            momentum_y=rho * v,
            total_energy=rho_e,
            species_density=species_density,
        )

    @classmethod
    def from_conservative(
        cls, q: ConservativeState, eos: EquationOfState, n_species: int = 1
    ) -> PrimitiveState:
        """Convert from conservative variables using EOS."""
        rho = q.density
        u = q.velocity_x()
        v = q.velocity_y()
        yi = q.mass_fractions(n_species)
        e_int = q.specific_internal_energy()
        p = eos.pressure(rho, e_int, yi)

        return cls(
            density=rho,
            velocity_x=u,
            velocity_y=v,
            pressure=p,
            mass_fractions=yi,
        )
