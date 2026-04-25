"""Park 1993 chemical kinetics for 5-species air.

Implements the Park (1993) mechanism with 17 reactions for:
N2, O2, NO, N, O

Reactions include:
- Dissociation: N2 + M -> 2N + M (5 third bodies)
- Dissociation: O2 + M -> 2O + M (5 third bodies)
- Dissociation: NO + M -> N + O + M (5 third bodies)
- Exchange: NO + O -> O2 + N
- Exchange: N2 + O -> NO + N

Rate constants from Park (1993):
    k_f = C_f * T_a^n_f * exp(-theta_d / T_a)
where T_a = T^s * Tv^(1-s) with s = 0.5 (Park's two-temperature model).

Reference: Park, C. (1993) "Review of chemical-kinetic problems of
future NASA missions, I: Earth entries." J. Thermophys. Heat Transfer 7:385-398.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.chemistry import ChemistryConfig


@dataclass(frozen=True)
class ReactionRate:
    """Arrhenius rate constant parameters: k = C * T^n * exp(-theta/T)."""

    C: float  # Pre-exponential factor  # noqa: N815
    n: float  # Temperature exponent
    theta: float  # Activation temperature (K)


# Park 1993 forward rate constants for dissociation reactions
# Reaction: AB + M -> A + B + M
# Third-body efficiencies vary by collision partner M
DISSOCIATION_N2 = {
    "N2": ReactionRate(C=7.0e21, n=-1.6, theta=113200.0),
    "O2": ReactionRate(C=7.0e21, n=-1.6, theta=113200.0),
    "NO": ReactionRate(C=7.0e21, n=-1.6, theta=113200.0),
    "N": ReactionRate(C=3.0e22, n=-1.6, theta=113200.0),  # Enhanced
    "O": ReactionRate(C=3.0e22, n=-1.6, theta=113200.0),  # Enhanced
}

DISSOCIATION_O2 = {
    "N2": ReactionRate(C=2.0e21, n=-1.5, theta=59500.0),
    "O2": ReactionRate(C=2.0e21, n=-1.5, theta=59500.0),
    "NO": ReactionRate(C=2.0e21, n=-1.5, theta=59500.0),
    "N": ReactionRate(C=1.0e22, n=-1.5, theta=59500.0),
    "O": ReactionRate(C=1.0e22, n=-1.5, theta=59500.0),
}

DISSOCIATION_NO = {
    "N2": ReactionRate(C=5.0e15, n=0.0, theta=75500.0),
    "O2": ReactionRate(C=5.0e15, n=0.0, theta=75500.0),
    "NO": ReactionRate(C=1.1e17, n=0.0, theta=75500.0),  # Enhanced
    "N": ReactionRate(C=1.1e17, n=0.0, theta=75500.0),
    "O": ReactionRate(C=1.1e17, n=0.0, theta=75500.0),
}

# Exchange reactions
EXCHANGE_NO_O = ReactionRate(C=8.4e12, n=0.0, theta=19450.0)  # NO + O -> O2 + N
EXCHANGE_N2_O = ReactionRate(C=6.4e17, n=-1.0, theta=38370.0)  # N2 + O -> NO + N


class Park1993Mechanism:
    """Park 1993 5-species air chemistry mechanism.

    Implements 17 reactions (5 dissociation * 3 species + 2 exchange).
    """

    SPECIES = ["N2", "O2", "NO", "N", "O"]

    def __init__(self, config: ChemistryConfig) -> None:
        self.config = config
        self._ttv_exponent = config.park_ttv_exponent

    @property
    def n_species(self) -> int:
        return 5

    @property
    def n_reactions(self) -> int:
        return 17

    @property
    def species_names(self) -> list[str]:
        return self.SPECIES

    def _controlling_temperature(
        self,
        t_tr: NDArray[np.float64],
        t_ve: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Park's controlling temperature T_a = T_tr^s * T_ve^(1-s)."""
        s = self._ttv_exponent
        return t_tr**s * t_ve ** (1.0 - s)

    def _forward_rate(
        self,
        rate: ReactionRate,
        t_a: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute forward rate constant k_f = C * T_a^n * exp(-theta/T_a)."""
        return rate.C * t_a**rate.n * np.exp(-rate.theta / np.maximum(t_a, 1.0))

    def _equilibrium_constant(
        self,
        theta_d: float,
        t_tr: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Simplified equilibrium constant from partition functions."""
        return np.exp(-theta_d / np.maximum(t_tr, 1.0))

    def source_terms(
        self,
        density: NDArray[np.float64],
        temperature_tr: NDArray[np.float64],
        temperature_ve: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute species production rates.

        Args:
            density: Mixture density (N,).
            temperature_tr: Translational temperature (N,).
            temperature_ve: Vibrational temperature (N,).
            mass_fractions: Species mass fractions (N, 5).

        Returns:
            Production rates (N, 5) in kg/(m^3·s).

        """
        n_pts = density.shape[0]
        omega = np.zeros((n_pts, 5), dtype=np.float64)

        t_a = self._controlling_temperature(temperature_tr, temperature_ve)

        # Species number densities (mol/m^3)
        from src.reentry.config.gas import DEFAULT_MOLECULAR_WEIGHTS as MW

        c = np.zeros((n_pts, 5), dtype=np.float64)
        for i, sp in enumerate(self.SPECIES):
            c[:, i] = density * mass_fractions[:, i] / MW[sp]

        # N2 dissociation: N2 + M -> 2N + M
        for j, m_sp in enumerate(self.SPECIES):
            kf = self._forward_rate(DISSOCIATION_N2[m_sp], t_a)
            keq = self._equilibrium_constant(113200.0, temperature_tr)
            kb = kf / np.maximum(keq, 1e-30)
            rate = kf * c[:, 0] * c[:, j] - kb * c[:, 3] ** 2 * c[:, j]
            omega[:, 0] -= rate * MW["N2"]  # N2 consumed
            omega[:, 3] += 2 * rate * MW["N"]  # N produced

        # O2 dissociation: O2 + M -> 2O + M
        for j, m_sp in enumerate(self.SPECIES):
            kf = self._forward_rate(DISSOCIATION_O2[m_sp], t_a)
            keq = self._equilibrium_constant(59500.0, temperature_tr)
            kb = kf / np.maximum(keq, 1e-30)
            rate = kf * c[:, 1] * c[:, j] - kb * c[:, 4] ** 2 * c[:, j]
            omega[:, 1] -= rate * MW["O2"]
            omega[:, 4] += 2 * rate * MW["O"]

        # NO dissociation: NO + M -> N + O + M
        for j, m_sp in enumerate(self.SPECIES):
            kf = self._forward_rate(DISSOCIATION_NO[m_sp], t_a)
            keq = self._equilibrium_constant(75500.0, temperature_tr)
            kb = kf / np.maximum(keq, 1e-30)
            rate = kf * c[:, 2] * c[:, j] - kb * c[:, 3] * c[:, 4] * c[:, j]
            omega[:, 2] -= rate * MW["NO"]
            omega[:, 3] += rate * MW["N"]
            omega[:, 4] += rate * MW["O"]

        # Exchange: NO + O -> O2 + N
        kf = self._forward_rate(EXCHANGE_NO_O, t_a)
        keq = self._equilibrium_constant(19450.0, temperature_tr)
        kb = kf / np.maximum(keq, 1e-30)
        rate = kf * c[:, 2] * c[:, 4] - kb * c[:, 1] * c[:, 3]
        omega[:, 2] -= rate * MW["NO"]
        omega[:, 4] -= rate * MW["O"]
        omega[:, 1] += rate * MW["O2"]
        omega[:, 3] += rate * MW["N"]

        # Exchange: N2 + O -> NO + N
        kf = self._forward_rate(EXCHANGE_N2_O, t_a)
        keq = self._equilibrium_constant(38370.0, temperature_tr)
        kb = kf / np.maximum(keq, 1e-30)
        rate = kf * c[:, 0] * c[:, 4] - kb * c[:, 2] * c[:, 3]
        omega[:, 0] -= rate * MW["N2"]
        omega[:, 4] -= rate * MW["O"]
        omega[:, 2] += rate * MW["NO"]
        omega[:, 3] += rate * MW["N"]

        return omega

    def energy_exchange_rate(
        self,
        density: NDArray[np.float64],
        temperature_tr: NDArray[np.float64],
        temperature_ve: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Landau-Teller vibrational-translational energy exchange.

        Q_TV = sum_s (rho_s * (e_v_eq(T_tr) - e_v(T_ve)) / tau_s)

        where tau_s is the Millikan-White relaxation time with
        Park's high-temperature correction.
        """
        from src.reentry.config.gas import DEFAULT_MOLECULAR_WEIGHTS as MW
        from src.reentry.gas.species import _DEFAULT_SPECIES

        n_pts = density.shape[0]
        q_tv = np.zeros(n_pts, dtype=np.float64)

        for i, sp_name in enumerate(self.SPECIES):
            sp = _DEFAULT_SPECIES.get(sp_name)
            if sp is None or sp.theta_v <= 0:
                continue

            rho_s = density * mass_fractions[:, i]
            e_v_eq = sp.e_vib(temperature_tr)
            e_v = sp.e_vib(temperature_ve)

            # Millikan-White relaxation time (simplified)
            # tau_MW = (1/p) * exp(A * (T^{-1/3} - B) - 18.42)
            a_mw = 1.16e-3 * np.sqrt(MW[sp_name] * 1000) * sp.theta_v ** (4.0 / 3.0)
            b_mw = 0.015 * (MW[sp_name] * 1000) ** 0.25
            p_atm = density * 287.0 * temperature_tr / 101325.0  # Approximate
            p_atm = np.maximum(p_atm, 1e-10)

            tau_mw = np.exp(a_mw * (temperature_tr ** (-1.0 / 3.0) - b_mw) - 18.42) / p_atm
            tau_mw = np.maximum(tau_mw, 1e-12)

            q_tv += rho_s * (e_v_eq - e_v) / tau_mw

        return q_tv
