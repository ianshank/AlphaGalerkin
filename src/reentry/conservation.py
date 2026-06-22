"""Conserved-quantity integrals and drift for the Euler-solver audit.

The 1D Euler solver result (:class:`~src.reentry.solver.euler_1d.Euler1DResult`)
exposes only primitive fields (density, velocity, pressure). A conservation audit
needs the *conserved* integrals — total mass ``∫ρ``, momentum ``∫ρu`` and energy
``∫ρE`` — reconstructed from those primitives via the ideal-gas relation
``ρE = p/(γ-1) + ½ρu²``.

Physical note on which quantities are conserved: a finite-volume scheme conserves
mass / momentum / energy by flux differencing *only when boundary fluxes vanish*.
For the classic shock-tube setups with transmissive boundaries, at the standard
evaluation time the waves are still interior, so the mass and energy fluxes at the
boundaries are zero (``u≈0`` there) and both are conserved to machine precision.
**Total momentum is not** — the net boundary pressure (``p_left ≠ p_right``)
exerts an impulse, so ``∫ρu`` legitimately changes. The audit therefore gates on
mass and energy and reports momentum drift as informational.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Denominator floor for relative-drift computation, so a near-zero baseline
# (e.g. total momentum of a symmetric initial state) does not divide by zero.
CONSERVATION_DRIFT_FLOOR: float = 1e-30


def conserved_integrals_1d(
    density: NDArray[np.float64],
    velocity: NDArray[np.float64],
    pressure: NDArray[np.float64],
    dx: float,
    gamma: float,
) -> dict[str, float]:
    """Compute total mass / momentum / energy from a 1D primitive field.

    Args:
        density: Cell densities ``ρ`` (N,).
        velocity: Cell velocities ``u`` (N,).
        pressure: Cell pressures ``p`` (N,).
        dx: Uniform cell width.
        gamma: Ratio of specific heats (``> 1``).

    Returns:
        ``{"mass", "momentum", "energy"}`` integrals (sum over cells × ``dx``).

    Raises:
        ValueError: If ``dx <= 0`` or ``gamma <= 1``.

    """
    if dx <= 0.0:
        raise ValueError(f"dx must be positive; got {dx}")
    if gamma <= 1.0:
        raise ValueError(f"gamma must exceed 1; got {gamma}")

    rho = np.asarray(density, dtype=np.float64)
    u = np.asarray(velocity, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)

    rho_u = rho * u
    rho_E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return {
        "mass": float(np.sum(rho) * dx),
        "momentum": float(np.sum(rho_u) * dx),
        "energy": float(np.sum(rho_E) * dx),
    }


def conservation_drift(
    initial: dict[str, float],
    final: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Absolute and relative drift between two integral dicts.

    Args:
        initial: Conserved integrals at ``t=0``.
        final: Conserved integrals at the final time. Must share keys with
            *initial*.

    Returns:
        Per-quantity ``{"absolute", "relative"}`` drift. ``relative`` divides by
        ``max(|initial|, CONSERVATION_DRIFT_FLOOR)``.

    Raises:
        KeyError: If *final* is missing a key present in *initial*.

    """
    out: dict[str, dict[str, float]] = {}
    for key, initial_value in initial.items():
        absolute = abs(final[key] - initial_value)
        relative = absolute / max(abs(initial_value), CONSERVATION_DRIFT_FLOOR)
        out[key] = {"absolute": absolute, "relative": relative}
    return out
