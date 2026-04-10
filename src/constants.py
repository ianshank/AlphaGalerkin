"""Shared constants for AlphaGalerkin.

Centralizes magic numbers and default values that were previously
duplicated across multiple modules. Grouped by domain.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Board / Game defaults
# ---------------------------------------------------------------------------
DEFAULT_BOARD_SIZES: list[int] = [9, 13, 19]
"""Standard Go board sizes used for curriculum learning."""

DEFAULT_BOARD_SIZE: int = 19
"""Default single board size (full-size Go)."""

DEFAULT_MAX_MOVES: int = 500
"""Maximum moves per game before declaring a draw."""

# ---------------------------------------------------------------------------
# MCTS defaults
# ---------------------------------------------------------------------------
DEFAULT_MCTS_SIMULATIONS: int = 800
"""Default number of MCTS simulations per move."""

DEFAULT_PUCT_CONSTANT: float = 1.5
"""PUCT exploration constant for MCTS."""

DEFAULT_DIRICHLET_ALPHA: float = 0.03
"""Dirichlet noise alpha for root exploration."""

DEFAULT_DIRICHLET_EPSILON: float = 0.25
"""Fraction of Dirichlet noise mixed into root prior."""

DEFAULT_VIRTUAL_LOSS: float = 3.0
"""Virtual loss applied during parallel MCTS."""

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE_SCHEDULE: dict[int, float] = {
    0: 1.0,  # Moves 0-29: temperature 1.0
    30: 0.5,  # Moves 30-59: temperature 0.5
    60: 0.1,  # Moves 60+: temperature 0.1
}
"""Default temperature schedule for self-play move selection."""

DEFAULT_CURRICULUM_SCHEDULE: dict[int, list[int]] = {
    0: [9],
    10000: [9, 13],
    50000: [9, 13, 19],
}
"""Default board-size curriculum: step -> allowed sizes."""

DEFAULT_DROPOUT: float = 0.1
"""Default dropout rate for neural network layers."""

# ---------------------------------------------------------------------------
# Prioritized Experience Replay (PER) defaults
# ---------------------------------------------------------------------------
DEFAULT_PER_ALPHA: float = 0.6
"""Priority exponent for PER sampling."""

DEFAULT_PER_BETA: float = 0.4
"""Initial importance-sampling exponent for PER."""

DEFAULT_PER_BETA_INCREMENT: float = 0.001
"""Per-sample increment for PER beta annealing."""

# ---------------------------------------------------------------------------
# LBB stability / loss defaults
# ---------------------------------------------------------------------------
DEFAULT_LBB_WEIGHT: float = 0.01
"""Weight for LBB regularization term in loss."""

DEFAULT_LBB_THRESHOLD: float = 1e-6
"""Minimum singular value for LBB stability check."""

DEFAULT_LBB_TARGET: float = 0.1
"""Target beta value for LBB soft constraint."""

DEFAULT_LBB_EPS: float = 1e-8
"""Epsilon for numerical stability in LBB loss."""

# ---------------------------------------------------------------------------
# Win-rate thresholds
# ---------------------------------------------------------------------------
WIN_RATE_ACCEPT_THRESHOLD: float = 0.55
"""Win rate above which a new model is accepted."""

WIN_RATE_REJECT_THRESHOLD: float = 0.45
"""Win rate below which a new model is rejected."""

# ---------------------------------------------------------------------------
# Numeric stability
# ---------------------------------------------------------------------------
LAYER_NORM_EPSILON: float = 1e-5
"""Default epsilon for layer normalization."""

ATTENTION_EPSILON: float = 1e-8
"""Epsilon for attention normalization."""

NUMERIC_EPSILON: float = 1e-6
"""General-purpose numeric stability epsilon."""

DEFAULT_BOUNDARY_TOLERANCE: float = 1e-6
"""Tolerance for boundary point detection in PDE geometry."""

# ---------------------------------------------------------------------------
# Checkpoint naming
# ---------------------------------------------------------------------------
CHECKPOINT_BEST: str = "best.pt"
"""Filename for best-model checkpoint."""

CHECKPOINT_FINAL: str = "final.pt"
"""Filename for final checkpoint."""

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------
TWO_PI: float = 6.283185307179586
"""2π — used in Fourier/trigonometric calculations across PDE operators."""

# ---------------------------------------------------------------------------
# PDE operator defaults
# ---------------------------------------------------------------------------
BURGERS_FOURIER_TERMS: int = 50
"""Number of Fourier series terms for Cole-Hopf Burgers solution approximation.
Higher values improve accuracy near shocks at the cost of compute."""

DEFAULT_SHOCK_POSITION: float = 0.5
"""Default shock centre position (fraction of domain) for Burgers IC."""

DEFAULT_SHOCK_WIDTH: float = 10.0
"""Default shock width parameter (tanh sharpness) for Burgers IC."""

GAUSSIAN_WIDTH_RATIO: float = 0.1
"""Fraction of mean domain size used as Gaussian IC standard deviation.
Applies to AdvectionDiffusionOperator and HeatOperator initial conditions."""

# ---------------------------------------------------------------------------
# Loss / training numerical stability
# ---------------------------------------------------------------------------
LOG_PROB_MIN: float = -100.0
"""Floor for log-probability clamping.  Prevents log(0) → -inf from
poisoning policy gradient updates; equivalent to probability ~ 3.7e-44."""

DEFAULT_H1_GRADIENT_WEIGHT: float = 0.1
"""Default weight for the gradient term in H¹ Sobolev loss (H1Loss)."""

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
BYTES_PER_MB: int = 1024 * 1024
"""Bytes per megabyte — used when reporting memory usage."""
