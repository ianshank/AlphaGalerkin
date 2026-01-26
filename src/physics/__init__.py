"""Synthetic Physics Data Generation for AlphaGalerkin.

This module provides ground truth influence field generation using
the Poisson equation (∇²φ = ρ) for validating zero-shot transfer
capabilities of the neural operator.

The key insight: by training on synthetic physics data at one resolution
(e.g., 9x9) and testing at another (e.g., 19x19), we can verify that
the model has learned a true resolution-independent operator.
"""

from __future__ import annotations

from src.physics.poisson import (
    PoissonDataset,
    PoissonSolver,
    create_poisson_dataloader,
    generate_influence_field,
    generate_random_charges,
)

__all__ = [
    "PoissonSolver",
    "PoissonDataset",
    "create_poisson_dataloader",
    "generate_influence_field",
    "generate_random_charges",
]
