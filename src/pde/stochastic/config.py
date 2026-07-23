"""Pydantic configuration contracts for the stochastic Galerkin layer.

Every tunable is a typed, bounded ``Field`` — no hardcoded values. Numerical-
stability literals are surfaced as named module-level constants (repo
convention; mirrors ``DEFAULT_TRANSFER_RATIO_FLOOR`` / ``EVAL_SEED_STRIDE``).

Spec: specs/stochastic_galerkin_nke.spec.md (Data Contract).
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig

# --- Numerical-stability and gate constants (spec: Data Contract) -----------

DEFAULT_COV_JITTER = 1e-8
"""Diagonal jitter added before Cholesky factorization of covariances."""

DEFAULT_MDN_MIN_SCALE = 1e-4
"""Floor on MDN output Cholesky diagonal (prevents covariance collapse)."""

DEFAULT_OU_MOMENT_TOL = 1e-3
"""AC1/AC3-oracle moment-error tolerance vs the closed-form OU solutions."""

DEFAULT_STRANG_SLOPE_MIN = 1.7
"""AC4 lower bound on the log2 error-halving slope (second-order splitting)."""

DEFAULT_STRANG_SLOPE_MAX = 2.3
"""AC4 upper bound on the log2 error-halving slope."""

DEFAULT_MONOTONE_WINDOW = 50
"""AC5 window length (steps) for windowed-mean loss monotonicity."""

DEFAULT_MONOTONE_REL_TOL = 1e-3
"""AC5 relative tolerance for non-increasing window means.

Calibrated on the pinned jump-OU run (seeds 42): observed max window-mean
increase 2.8e-6 — ~350× headroom.
"""

DEFAULT_LOSS_RATIO_GATE = 0.98
"""AC5 required ``final_loss / initial_loss`` ceiling.

Calibrated on the pinned jump-OU run (seeds 42): observed ratio 0.950 (the
initial 0.9 placeholder was unreachable — a dt-scaled residual MDN starts
near identity, so the closable NLL gap is inherently modest; the gap-closure
gate below is the sharper criterion).
"""

DEFAULT_LOSS_GAP_CLOSURE = 0.25
"""AC5 ceiling on the fraction of the (initial − oracle) loss gap left open.

The trainer can evaluate its own loss with the exact compound-Poisson moment
oracle substituted for the MDN — the achievable floor. Calibrated on the
pinned jump-OU run (seeds 42): observed closure fraction 0.000 (the trainer
reaches the oracle floor); 0.25 leaves generous headroom.
"""

DEFAULT_TRAINED_MDN_MOMENT_TOL = 5e-2
"""AC3 trained-MDN trajectory moment-error tolerance.

Placeholder until calibrated from the pinned jump-OU run (spec: Calibration
procedure); the calibrated value is recorded in the spec table.
"""

DEFAULT_KMEANS_MAX_ITERS = 100
"""Maximum Lloyd's iterations for per-slice particle clustering."""

DEFAULT_KMEANS_TOL = 1e-6
"""Relative centroid-shift convergence tolerance for Lloyd's iteration."""

DEFAULT_CLUSTER_COV_FLOOR = 1e-6
"""Diagonal floor on per-cluster empirical covariances (degenerate clusters)."""

_SYMMETRY_ATOL = 1e-9
"""Absolute tolerance when validating covariance symmetry in configs."""

_WEIGHT_SUM_ATOL = 1e-6
"""Absolute tolerance when validating that mixture weights sum to one."""

_MAX_DIM = 8
"""Upper bound on state dimension (v1 targets low-dimensional SDEs)."""

_MAX_COMPONENTS = 32
"""Upper bound on Gaussian-mixture size K."""


class GaussianMixtureBasisConfig(BaseModuleConfig):
    """Configuration of the Gaussian-mixture basis (the Galerkin trial space)."""

    name: str = Field(default="gaussian_mixture_basis", min_length=1)
    dim: int = Field(ge=1, le=_MAX_DIM, description="State dimension d.")
    n_components: int = Field(
        default=1,
        ge=1,
        le=_MAX_COMPONENTS,
        description="Mixture size K.",
    )
    dtype: Literal["float32", "float64"] = Field(
        default="float64",
        description="Dtype for moment state and flows (float64 recommended; AC1 tolerance).",
    )
    weight_dynamics: Literal["frozen"] = Field(
        default="frozen",
        description=(
            "v1 limitation surfaced as a forward-compatible knob: mixture weights are "
            "frozen during propagation (exact for linear drift; spec Out of Scope)."
        ),
    )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the configured dtype as a ``torch.dtype``."""
        return torch.float64 if self.dtype == "float64" else torch.float32


class JumpConfig(BaseModuleConfig):
    """Compound-Poisson jump term: rate λ and jump-size distribution ξ ~ N(μ_ξ, Σ_ξ)."""

    name: str = Field(default="jump", min_length=1)
    rate: float = Field(ge=0.0, description="Compound-Poisson intensity λ (0 ⇒ no jump term).")
    jump_mean: list[float] = Field(min_length=1, description="Jump-size mean μ_ξ (length d).")
    jump_cov: list[list[float]] = Field(
        min_length=1,
        description="Jump-size covariance Σ_ξ (d×d, symmetric).",
    )

    @model_validator(mode="after")
    def _validate_shapes(self) -> JumpConfig:
        d = len(self.jump_mean)
        if len(self.jump_cov) != d or any(len(row) != d for row in self.jump_cov):
            msg = f"jump_cov must be {d}x{d} to match jump_mean; got {self.jump_cov!r}"
            raise ValueError(msg)
        for i in range(d):
            for j in range(i + 1, d):
                if abs(self.jump_cov[i][j] - self.jump_cov[j][i]) > _SYMMETRY_ATOL:
                    msg = f"jump_cov must be symmetric; asymmetry at ({i},{j})"
                    raise ValueError(msg)
        return self


class StochasticGeneratorConfig(BaseModuleConfig):
    """Configuration of the Kolmogorov-forward generator L = A + D + J."""

    name: str = Field(default="stochastic_generator", min_length=1)
    dim: int = Field(ge=1, le=_MAX_DIM, description="State dimension d.")
    drift_matrix: list[list[float]] | None = Field(
        default=None,
        description="Linear drift matrix A (d×d) for f(x)=Ax+b; None ⇒ callable drift supplied.",
    )
    drift_bias: list[float] | None = Field(
        default=None,
        description="Linear drift bias b (length d); None with drift_matrix set ⇒ zero bias.",
    )
    diffusion: list[list[float]] = Field(
        min_length=1,
        description="Diffusion factor g (d×m); the generator uses Q = g gᵀ.",
    )
    jump: JumpConfig | None = Field(
        default=None,
        description="Optional compound-Poisson jump term.",
    )

    @model_validator(mode="after")
    def _validate_shapes(self) -> StochasticGeneratorConfig:
        d = self.dim
        if self.drift_matrix is not None and (
            len(self.drift_matrix) != d or any(len(row) != d for row in self.drift_matrix)
        ):
            msg = f"drift_matrix must be {d}x{d}; got {self.drift_matrix!r}"
            raise ValueError(msg)
        if self.drift_bias is not None and len(self.drift_bias) != d:
            msg = f"drift_bias must have length {d}; got {self.drift_bias!r}"
            raise ValueError(msg)
        if len(self.diffusion) != d:
            msg = f"diffusion must have {d} rows (one per state dim); got {len(self.diffusion)}"
            raise ValueError(msg)
        n_cols = len(self.diffusion[0])
        if n_cols < 1 or any(len(row) != n_cols for row in self.diffusion):
            msg = "diffusion rows must be non-empty and of equal length"
            raise ValueError(msg)
        if self.jump is not None and len(self.jump.jump_mean) != d:
            msg = f"jump term dimension {len(self.jump.jump_mean)} != generator dim {d}"
            raise ValueError(msg)
        return self

    def diffusion_tensor(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Return g as a (d, m) tensor."""
        return torch.tensor(self.diffusion, dtype=dtype)

    def drift_matrix_tensor(self, dtype: torch.dtype = torch.float64) -> torch.Tensor | None:
        """Return A as a (d, d) tensor, or None for callable drift."""
        if self.drift_matrix is None:
            return None
        return torch.tensor(self.drift_matrix, dtype=dtype)

    def drift_bias_tensor(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Return b as a (d,) tensor (zeros when unset)."""
        if self.drift_bias is None:
            return torch.zeros(self.dim, dtype=dtype)
        return torch.tensor(self.drift_bias, dtype=dtype)

    @property
    def has_jump(self) -> bool:
        """True iff a jump term with positive rate is configured."""
        return self.jump is not None and self.jump.rate > 0.0


class MDNJumpConfig(BaseModuleConfig):
    """Configuration of the MDN jump-semigroup network."""

    name: str = Field(default="mdn_jump", min_length=1)
    dim: int = Field(ge=1, le=_MAX_DIM, description="State dimension d.")
    n_components: int = Field(
        ge=1,
        le=_MAX_COMPONENTS,
        description="Mixture size K (must equal the basis K; enforced when advancing a state).",
    )
    hidden_dims: list[int] = Field(
        default_factory=lambda: [64, 64],
        min_length=1,
        description="MLP hidden-layer widths.",
    )
    dt_embed_dim: int = Field(
        default=8,
        ge=1,
        description="Width of the dt embedding concatenated to the packed mixture input.",
    )

    @model_validator(mode="after")
    def _validate_hidden_dims(self) -> MDNJumpConfig:
        if any(h < 1 for h in self.hidden_dims):
            msg = f"hidden_dims entries must be >= 1; got {self.hidden_dims!r}"
            raise ValueError(msg)
        return self


class StrangSplittingConfig(BaseModuleConfig):
    """Configuration of the Strang-splitting propagator."""

    name: str = Field(default="strang_splitting", min_length=1)
    dt: float = Field(gt=0.0, description="Coarse splitting step size.")
    t_end: float = Field(gt=0.0, description="Propagation horizon T.")
    ad_integrator: Literal["exact_expm", "rk4"] = Field(
        default="exact_expm",
        description=(
            "A+D flow method: exact matrix-exponential flows (linear drift only) or RK4 "
            "moment-ODE integration via src/pde/time_stepping.py (fixed dt)."
        ),
    )
    rk4_substeps: int = Field(
        default=4,
        ge=1,
        description="RK4 substeps per half-flow when ad_integrator='rk4'.",
    )

    @model_validator(mode="after")
    def _validate_horizon(self) -> StrangSplittingConfig:
        if self.dt > self.t_end:
            msg = f"dt ({self.dt}) must not exceed t_end ({self.t_end})"
            raise ValueError(msg)
        return self


class StrangTrainerConfig(BaseModuleConfig):
    """Configuration of the parallel-in-time Strang trainer."""

    name: str = Field(default="strang_trainer", min_length=1)
    n_particles: int = Field(
        ge=16,
        le=100_000,
        description="SDE particle count for the precomputed dataset.",
    )
    n_time_slices: int = Field(ge=3, description="Coarse time-grid points M.")
    sim_dt: float = Field(gt=0.0, description="Fine Euler–Maruyama simulation step.")
    max_steps: int = Field(default=500, ge=1, description="Optimizer step budget.")
    learning_rate: float = Field(default=1e-2, gt=0.0, description="AdamW learning rate.")
    full_batch: bool = Field(
        default=True,
        description="Deterministic AC mode: every interval loss is computed at every step.",
    )
    minibatch_slices: int = Field(
        default=4,
        ge=1,
        description="Intervals sampled per step when full_batch=False.",
    )
    monotone_window: int = Field(
        default=DEFAULT_MONOTONE_WINDOW,
        ge=1,
        description="Window length for the AC5 windowed-mean monotonicity check.",
    )
