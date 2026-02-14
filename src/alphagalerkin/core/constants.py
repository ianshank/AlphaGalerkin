"""Named constants for AlphaGalerkin -- no magic numbers.

Every literal that appears in more than one module (or that a
user might plausibly want to override) is defined here with a
docstring explaining its role.

Import example::

    from src.alphagalerkin.core.constants import (
        DEFAULT_SEED,
        EPSILON,
        MIN_SINGULAR_VALUE,
    )
"""

from __future__ import annotations

from typing import Final

# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

DEFAULT_SEED: Final[int] = 42
"""Default random seed for deterministic experiments."""


# -----------------------------------------------------------------------
# Numerical tolerances
# -----------------------------------------------------------------------

MIN_SINGULAR_VALUE: Final[float] = 1e-6
"""LBB / inf-sup stability threshold.

If the minimum singular value of the Key-to-Value projection
drops below this value, the Galerkin discretization is deemed
unstable and a ``StabilityViolationError`` is raised.
"""

EPSILON: Final[float] = 1e-8
"""General-purpose numerical epsilon for avoiding division by zero."""


# -----------------------------------------------------------------------
# MCTS defaults
# -----------------------------------------------------------------------

DEFAULT_C_PUCT: Final[float] = 2.5
"""Default PUCT exploration constant for MCTS child selection."""

DEFAULT_DIRICHLET_ALPHA: Final[float] = 0.3
"""Default Dirichlet noise alpha for root exploration."""

DEFAULT_DIRICHLET_EPSILON: Final[float] = 0.25
"""Fraction of root prior replaced by Dirichlet noise."""

DEFAULT_MAX_TREE_DEPTH: Final[int] = 200
"""Maximum depth of the MCTS search tree."""

DEFAULT_NUM_SIMULATIONS: Final[int] = 800
"""Default number of MCTS simulations per move."""


# -----------------------------------------------------------------------
# Network defaults
# -----------------------------------------------------------------------

DEFAULT_HIDDEN_DIM: Final[int] = 128
"""Default hidden dimension for GNN and MLP layers."""

DEFAULT_NUM_GNN_LAYERS: Final[int] = 6
"""Default number of GNN message-passing layers."""

DEFAULT_NUM_ATTENTION_HEADS: Final[int] = 8
"""Default number of attention heads in GAT layers."""


# -----------------------------------------------------------------------
# Training defaults
# -----------------------------------------------------------------------

DEFAULT_LEARNING_RATE: Final[float] = 1e-3
"""Default initial learning rate."""

DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4
"""Default L2 regularization coefficient."""

DEFAULT_BATCH_SIZE: Final[int] = 256
"""Default training mini-batch size."""

DEFAULT_REPLAY_CAPACITY: Final[int] = 500_000
"""Default replay buffer capacity (number of transitions)."""


# -----------------------------------------------------------------------
# Environment defaults
# -----------------------------------------------------------------------

DEFAULT_MAX_DOF: Final[int] = 50_000
"""Default maximum degrees of freedom budget per episode."""

DEFAULT_ERROR_TOLERANCE: Final[float] = 1e-4
"""Default target L2 error tolerance for episode termination."""

DEFAULT_MAX_STEPS: Final[int] = 200
"""Default maximum number of environment steps per episode."""
