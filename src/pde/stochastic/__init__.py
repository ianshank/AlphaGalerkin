"""Stochastic Galerkin operator-splitting layer (NKE).

Additive subpackage: Lagrangian Galerkin projection of a Kolmogorov-forward
generator ``L = A + D + J`` onto a Gaussian-mixture basis, with a Strang-
splitting parallel-in-time trainer. Import via this subpackage (not the
``src.pde`` root, whose ``__init__`` triggers game registration side effects).

Spec: specs/stochastic_galerkin_nke.spec.md.
"""

from src.pde.stochastic.analytic import (
    gaussian_density_on_grid,
    jump_ou_covariance,
    jump_ou_mean,
    ou_covariance,
    ou_mean,
)
from src.pde.stochastic.config import (
    DEFAULT_COV_JITTER,
    DEFAULT_KMEANS_MAX_ITERS,
    DEFAULT_KMEANS_TOL,
    DEFAULT_LOSS_RATIO_GATE,
    DEFAULT_MDN_MIN_SCALE,
    DEFAULT_MONOTONE_REL_TOL,
    DEFAULT_MONOTONE_WINDOW,
    DEFAULT_OU_MOMENT_TOL,
    DEFAULT_STRANG_SLOPE_MAX,
    DEFAULT_STRANG_SLOPE_MIN,
    DEFAULT_TRAINED_MDN_MOMENT_TOL,
    GaussianMixtureBasisConfig,
    JumpConfig,
    MDNJumpConfig,
    StochasticGeneratorConfig,
    StrangSplittingConfig,
    StrangTrainerConfig,
)
from src.pde.stochastic.errors import JumpModelMissingError, StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureBasis, GaussianMixtureState

__all__ = [
    "DEFAULT_COV_JITTER",
    "DEFAULT_KMEANS_MAX_ITERS",
    "DEFAULT_KMEANS_TOL",
    "DEFAULT_LOSS_RATIO_GATE",
    "DEFAULT_MDN_MIN_SCALE",
    "DEFAULT_MONOTONE_REL_TOL",
    "DEFAULT_MONOTONE_WINDOW",
    "DEFAULT_OU_MOMENT_TOL",
    "DEFAULT_STRANG_SLOPE_MAX",
    "DEFAULT_STRANG_SLOPE_MIN",
    "DEFAULT_TRAINED_MDN_MOMENT_TOL",
    "GaussianMixtureBasis",
    "GaussianMixtureBasisConfig",
    "GaussianMixtureState",
    "JumpConfig",
    "JumpModelMissingError",
    "MDNJumpConfig",
    "StochasticConfigurationError",
    "StochasticGeneratorConfig",
    "StrangSplittingConfig",
    "StrangTrainerConfig",
    "gaussian_density_on_grid",
    "jump_ou_covariance",
    "jump_ou_mean",
    "ou_covariance",
    "ou_mean",
]
