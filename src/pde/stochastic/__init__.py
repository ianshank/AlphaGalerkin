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
from src.pde.stochastic.gaussian_mixture import (
    GaussianMixtureBasis,
    GaussianMixtureState,
    pack_moments,
)
from src.pde.stochastic.generator import (
    DriftModel,
    JumpSemigroup,
    KolmogorovGenerator,
    LinearDrift,
)
from src.pde.stochastic.jump_mdn import (
    AnalyticCompoundPoissonMoments,
    MDNJumpSemigroup,
    batched_mixture_nll,
    pack_batch,
    unpack_batch,
)
from src.pde.stochastic.particles import (
    ParticleSimulationResult,
    TimeSliceClusters,
    cluster_time_slices,
    sample_gaussian,
    simulate_jump_diffusion,
)
from src.pde.stochastic.projection import GalerkinMomentProjection
from src.pde.stochastic.strang import StrangSplitStep

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
    "AnalyticCompoundPoissonMoments",
    "DriftModel",
    "GalerkinMomentProjection",
    "GaussianMixtureBasis",
    "GaussianMixtureBasisConfig",
    "GaussianMixtureState",
    "JumpConfig",
    "JumpModelMissingError",
    "JumpSemigroup",
    "KolmogorovGenerator",
    "LinearDrift",
    "MDNJumpConfig",
    "MDNJumpSemigroup",
    "ParticleSimulationResult",
    "StochasticConfigurationError",
    "StochasticGeneratorConfig",
    "StrangSplitStep",
    "StrangSplittingConfig",
    "StrangTrainerConfig",
    "TimeSliceClusters",
    "batched_mixture_nll",
    "cluster_time_slices",
    "gaussian_density_on_grid",
    "jump_ou_covariance",
    "jump_ou_mean",
    "ou_covariance",
    "ou_mean",
    "pack_batch",
    "pack_moments",
    "sample_gaussian",
    "simulate_jump_diffusion",
    "unpack_batch",
]
