"""Configuration schema for the Noyron HX zero-shot transfer scenario.

This config is intentionally a sibling to :mod:`src.poc.config` rather than
adding to that module: keeping Leap 71 / Noyron parameters out of the core
PoC config makes the integration optional and isolates churn to this single
file.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.poc.config import BaseScenarioConfig, ScenarioTier


class NoyronHXScenarioConfig(BaseScenarioConfig):
    """Zero-shot transfer scenario on Leap 71's helical heat exchanger.

    Train a 3D PINN-style ``PhysicsOperator`` at low collocation-point
    density on an SDF-bounded helical tube, evaluate at higher density,
    report MSE on the steady-state temperature field.

    Most fields mirror ``TransferScenarioConfig`` so consumers familiar
    with the existing ``transfer`` scenario can read this config without
    re-learning the surface.
    """

    name: str = Field(default="noyron_hx", description="Scenario identifier.")
    description: str = Field(
        default="Zero-shot 3D heat-equation transfer on Leap 71 helical HX.",
        description="Scenario description.",
    )
    tier: ScenarioTier = ScenarioTier.INTEGRATION

    # ----- helix geometry (matches GeometryConfig naming) -----
    helix_R_major: float = Field(  # noqa: N815 - mathematical convention
        default=0.05, gt=0.0, description="Helix radius."
    )
    helix_r_minor: float = Field(default=0.012, gt=0.0, description="Tube cross-section radius.")
    helix_pitch: float = Field(default=0.02, gt=0.0, description="Vertical rise per turn.")
    # Default of 5 matches both the YAML scenario (config/scenarios/noyron_hx.yaml)
    # and ``AnalyticalHelixSDF`` so callers that instantiate the config in
    # code see the same geometry as the headline run.
    helix_n_turns: int = Field(default=5, ge=1, description="Number of helical revolutions.")

    # ----- SDF backend selection -----
    use_picogk: bool = Field(
        default=False,
        description=(
            "If True, attempt to load a PicoGK voxel STL via the optional "
            "[picogk] extra; if False, use the closed-form analytical helix."
        ),
    )
    picogk_voxel_path: str | None = Field(
        default=None,
        description="Path to a PicoGK voxel STL (only when use_picogk=True).",
    )

    # ----- training data -----
    n_train_pts: int = Field(
        default=4096,
        ge=64,
        description="Collocation points sampled per training batch.",
    )
    n_train_boundary_pts: int = Field(
        default=512,
        ge=8,
        description="Boundary points sampled per training batch.",
    )
    n_eval_pts: int = Field(
        default=16384,
        ge=64,
        description=(
            "Collocation points used for zero-shot evaluation; should be "
            ">= n_train_pts to demonstrate resolution-independent transfer."
        ),
    )

    # ----- training settings -----
    n_epochs: int = Field(default=200, ge=1, description="Training epochs.")
    batch_size: int = Field(default=1, ge=1, description="Batch size.")
    learning_rate: float = Field(default=1e-3, gt=0, description="Adam learning rate.")
    diffusivity: float = Field(default=1.0, gt=0, description="Thermal diffusivity (kappa).")

    # ----- model -----
    d_model: int = Field(default=64, ge=16, description="Hidden dim.")
    n_heads: int = Field(default=4, ge=1, description="Attention heads.")
    n_layers: int = Field(default=3, ge=1, description="Galerkin layers.")
    n_fourier_features: int = Field(default=32, ge=8, description="Fourier feature count.")
    fourier_scale: float = Field(default=10.0, gt=0, description="Fourier feature scale.")
    use_fnet: bool = Field(default=True, description="Use FNet mixing.")

    # ----- reference solution -----
    ref_solver_kind: Literal["analytical_harmonic", "voxel_fdm"] = Field(
        default="analytical_harmonic",
        description=(
            "How to compute the ground-truth temperature field. "
            "'analytical_harmonic' picks BCs that admit a closed-form "
            "harmonic solution; 'voxel_fdm' uses the in-repo 64^3 FDM "
            "solver."
        ),
    )
    voxel_fdm_resolution: int = Field(
        default=48,
        ge=16,
        le=128,
        description="Cubic voxel grid resolution for the FDM reference.",
    )
    voxel_fdm_iterations: int = Field(
        default=1500,
        ge=10,
        description="Maximum Jacobi sweeps in the FDM reference solver.",
    )
    voxel_fdm_tolerance: float = Field(
        default=1e-5,
        gt=0,
        description="Convergence tolerance (max-norm update) for the FDM solver.",
    )
    harmonic_wave_number: float = Field(
        default=3.141592653589793,
        gt=0,
        description=(
            "Wave number ``k`` used by the analytical-harmonic reference "
            "field ``u(p) = sin(k*x) + sin(k*y) + sin(k*z)``. Default "
            "``k = pi`` (single full period across the unit cube) keeps "
            "the field within the surrogate's spectral capacity at the "
            "documented headline thresholds; raise to ``2*pi`` or ``4*pi`` "
            "to stress the Fourier-feature surrogate at higher frequencies "
            "and expect both ``mse_low`` and ``mse_high`` to grow accordingly."
        ),
    )

    # ----- success criteria -----
    # Defaults are calibrated to the achievable floor at the YAML-default
    # model size (4096 colloc, d_model=64, 32 Fourier features, 200 epochs,
    # k = pi), measured at ``mse_low ~= mse_high ~= 1.6e-2`` on a
    # Blackwell-class GPU. The transfer-ratio threshold (the headline
    # resolution-independence claim) is tight at ``1.5`` because the
    # measured ratio is ``1.00 +/- 0.01``. To reach tighter absolute MSEs
    # (e.g. 1e-3) increase ``d_model``, ``n_train_pts``, or ``n_epochs``
    # and tighten these thresholds in the per-scenario YAML accordingly.
    mse_threshold_low: float = Field(
        default=2e-2,
        gt=0,
        description="MSE threshold at training point density.",
    )
    mse_threshold_high: float = Field(
        default=2e-2,
        gt=0,
        description="MSE threshold at zero-shot evaluation point density.",
    )
    transfer_ratio_threshold: float = Field(
        default=1.5,
        gt=1.0,
        description=(
            "Maximum allowed ``mse_high / mse_low`` ratio. The headline "
            "resolution-independence claim: at ``k = pi`` and the YAML-"
            "default surrogate size the measured ratio is ``1.00 +/- 0.01``."
        ),
    )

    # ----- compute -----
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="cuda",
        description=(
            "Preferred training device. 'cuda' is the project default and "
            "fails loud if CUDA is unavailable; 'auto' silently falls back "
            "to CPU; 'cpu' forces CPU (used by the smoke test)."
        ),
    )
    requires_gpu: bool = Field(
        default=True,
        description=(
            "GPU strongly preferred for the headline run; the CI smoke "
            "test sets device='cpu' explicitly."
        ),
    )

    @model_validator(mode="after")
    def validate_self(self) -> NoyronHXScenarioConfig:
        if self.helix_r_minor >= self.helix_R_major:
            raise ValueError(
                f"helix_r_minor ({self.helix_r_minor}) must be < "
                f"helix_R_major ({self.helix_R_major}) to avoid "
                f"self-intersection."
            )
        if self.use_picogk and not self.picogk_voxel_path:
            raise ValueError("use_picogk=True requires picogk_voxel_path to be set.")
        if self.n_eval_pts < self.n_train_pts:
            raise ValueError(
                f"n_eval_pts ({self.n_eval_pts}) should be >= n_train_pts "
                f"({self.n_train_pts}) to make the transfer claim meaningful."
            )
        return self
