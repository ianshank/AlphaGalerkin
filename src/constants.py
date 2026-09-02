"""Shared constants for AlphaGalerkin.

Centralizes magic numbers and default values that were previously
duplicated across multiple modules. Grouped by domain.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Board / Game defaults
# ---------------------------------------------------------------------------
DEFAULT_BOARD_SIZES: Final[list[int]] = [9, 13, 19]
"""Standard Go board sizes used for curriculum learning."""

DEFAULT_BOARD_SIZE: Final[int] = 19
"""Default single board size (full-size Go)."""

DEFAULT_MAX_MOVES: Final[int] = 500
"""Maximum moves per game before declaring a draw."""

# ---------------------------------------------------------------------------
# MCTS defaults
# ---------------------------------------------------------------------------
DEFAULT_MCTS_SIMULATIONS: Final[int] = 800
"""Default number of MCTS simulations per move."""

DEFAULT_PUCT_CONSTANT: Final[float] = 1.5
"""PUCT exploration constant for MCTS."""

DEFAULT_DIRICHLET_ALPHA: Final[float] = 0.03
"""Dirichlet noise alpha for root exploration."""

DEFAULT_DIRICHLET_EPSILON: Final[float] = 0.25
"""Fraction of Dirichlet noise mixed into root prior."""

DEFAULT_VIRTUAL_LOSS: Final[float] = 3.0
"""Virtual loss applied during parallel MCTS."""

DEFAULT_TEMPERATURE: Final[float] = 1.0
"""Default temperature for action selection."""

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE_SCHEDULE: Final[dict[int, float]] = {
    0: 1.0,  # Moves 0-29: temperature 1.0
    30: 0.5,  # Moves 30-59: temperature 0.5
    60: 0.1,  # Moves 60+: temperature 0.1
}
"""Default temperature schedule for self-play move selection."""

DEFAULT_CURRICULUM_SCHEDULE: Final[dict[int, list[int]]] = {
    0: [9],
    10000: [9, 13],
    50000: [9, 13, 19],
}
"""Default board-size curriculum: step -> allowed sizes."""

DEFAULT_DROPOUT: Final[float] = 0.1
"""Default dropout rate for neural network layers."""

# ---------------------------------------------------------------------------
# Prioritized Experience Replay (PER) defaults
# ---------------------------------------------------------------------------
DEFAULT_PER_ALPHA: Final[float] = 0.6
"""Priority exponent for PER sampling."""

DEFAULT_PER_BETA: Final[float] = 0.4
"""Initial importance-sampling exponent for PER."""

DEFAULT_PER_BETA_INCREMENT: Final[float] = 0.001
"""Per-sample increment for PER beta annealing."""

# ---------------------------------------------------------------------------
# LBB stability / loss defaults
# ---------------------------------------------------------------------------
DEFAULT_LBB_WEIGHT: Final[float] = 0.01
"""Weight for LBB regularization term in loss."""

DEFAULT_LBB_THRESHOLD: Final[float] = 1e-6
"""Minimum singular value for LBB stability check."""

DEFAULT_LBB_TARGET: Final[float] = 0.1
"""Target beta value for LBB soft constraint."""

DEFAULT_LBB_EPS: Final[float] = 1e-8
"""Epsilon for numerical stability in LBB loss."""

DEFAULT_BOUNDARY_TOLERANCE: Final[float] = 1e-6
"""Tolerance for boundary point detection in PDE geometry."""

DEFAULT_TRANSFER_RATIO_FLOOR: Final[float] = 1e-12
"""Floor used to avoid division by zero in transfer ratio calculation.

Numerically **different** from :data:`DEFAULT_RATIO_FLOOR` below (1e-12 vs
1e-15) and deliberately not unified with it: same knob, different live values
at different call sites, which ``surface-hardcoded-value`` Step 1 says to keep
apart rather than silently retune one of them.
"""

DEFAULT_RATIO_FLOOR: Final[float] = 1e-15
"""Floor used to avoid division by zero in benchmark ratio calculations.

The value **four** sites had each declared for themselves --
``research/lshape_amr_compare.py``'s ``RATIO_FLOOR``,
``research/transfer_baseline_compare.py``'s ``TRANSFER_RATIO_FLOOR``,
``research/substrates/config.py``'s ``RATIO_FLOOR``, and
``scripts/run_adaptive_vs_uniform.py``'s ``RATIO_FLOOR``. The third carried a
provenance comment that was wrong on every count: it cited a module path that
does not exist and claimed to mirror :data:`DEFAULT_TRANSFER_RATIO_FLOOR`,
which is a thousandfold larger. The fourth was found by a dead-code audit one
commit after this docstring first claimed "three" and promised a fourth could
not drift -- it already had. All four now source this one name.

**Not** interchangeable with :data:`DEFAULT_TRANSFER_RATIO_FLOOR`. Changing one
must not change the other.
"""

DEFAULT_POISSON_RHS: Final[float] = 1.0
"""Default right-hand side constant for Poisson test problems."""

# ---------------------------------------------------------------------------
# Win-rate thresholds
# ---------------------------------------------------------------------------
WIN_RATE_ACCEPT_THRESHOLD: Final[float] = 0.55
"""Win rate above which a new model is accepted."""

WIN_RATE_REJECT_THRESHOLD: Final[float] = 0.45
"""Win rate below which a new model is rejected."""

# ---------------------------------------------------------------------------
# Numeric stability
# ---------------------------------------------------------------------------
LAYER_NORM_EPSILON: Final[float] = 1e-5
"""Default epsilon for layer normalization."""

ATTENTION_EPSILON: Final[float] = 1e-8
"""Epsilon for attention normalization."""

NUMERIC_EPSILON: Final[float] = 1e-6
"""General-purpose numeric stability epsilon."""

DEFAULT_PICOGK_BOUNDARY_TOLERANCE: Final[float] = 1e-5
"""Boundary-classification band for SDF-based (PicoGK) operators.

Deliberately looser than ``DEFAULT_BOUNDARY_TOLERANCE``: this is a band on a
*signed-distance* value (|sdf(x)| < tol on a Newton-projected surface), a
semantically different test from the axis-aligned coordinate compare the
analytic operators use. Note the pre-existing intra-stack divergence: the
helical operators classify at 1e-5 while ``PicoGKDomain.is_boundary`` defaults
to ``DEFAULT_BOUNDARY_TOLERANCE`` (1e-6). Numerically equal to — but unrelated
to — ``DEFAULT_BOUNDARY_PROJECTION_TOL`` in ``src/pde/geometry_picogk.py``,
which is a Newton-projection *convergence* tolerance, not a classification
band. Do not merge them.
"""

# ---------------------------------------------------------------------------
# Checkpoint naming
# ---------------------------------------------------------------------------
CHECKPOINT_BEST: Final[str] = "best.pt"
"""Filename for best-model checkpoint."""

CHECKPOINT_FINAL: Final[str] = "final.pt"
"""Filename for final checkpoint."""

__all__ = [
    "ATTENTION_EPSILON",
    "CHECKPOINT_BEST",
    "CHECKPOINT_FINAL",
    "DEFAULT_BOARD_SIZE",
    "DEFAULT_BOARD_SIZES",
    "DEFAULT_BOUNDARY_TOLERANCE",
    "DEFAULT_CURRICULUM_SCHEDULE",
    "DEFAULT_DIRICHLET_ALPHA",
    "DEFAULT_DIRICHLET_EPSILON",
    "DEFAULT_DROPOUT",
    "DEFAULT_LBB_EPS",
    "DEFAULT_LBB_TARGET",
    "DEFAULT_LBB_THRESHOLD",
    "DEFAULT_LBB_WEIGHT",
    "DEFAULT_MAX_MOVES",
    "DEFAULT_MCTS_SIMULATIONS",
    "DEFAULT_PER_ALPHA",
    "DEFAULT_PER_BETA",
    "DEFAULT_PER_BETA_INCREMENT",
    "DEFAULT_PICOGK_BOUNDARY_TOLERANCE",
    "DEFAULT_POISSON_RHS",
    "DEFAULT_PUCT_CONSTANT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TEMPERATURE_SCHEDULE",
    "DEFAULT_RATIO_FLOOR",
    "DEFAULT_TRANSFER_RATIO_FLOOR",
    "DEFAULT_VIRTUAL_LOSS",
    "LAYER_NORM_EPSILON",
    "NUMERIC_EPSILON",
    "WIN_RATE_ACCEPT_THRESHOLD",
    "WIN_RATE_REJECT_THRESHOLD",
]
