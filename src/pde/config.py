"""Configuration classes for PDE-based games.

This module provides Pydantic configurations for:
- PDE operator parameters
- Game settings (budget, tolerance, rewards)
- Basis selection parameters
- Mesh refinement parameters

All configurations follow AlphaGalerkin patterns with:
- Type-safe validation
- No hardcoded values
- Deterministic hashing for reproducibility
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.templates.config import BaseModuleConfig, MetricDefinition, ThresholdOperator


class PDEType(str, Enum):
    """Types of PDEs supported by the framework."""

    POISSON = "poisson"
    BURGERS = "burgers"
    ADVECTION_DIFFUSION = "advection_diffusion"
    NAVIER_STOKES = "navier_stokes"
    HEAT = "heat"
    WAVE = "wave"


class BoundaryCondition(str, Enum):
    """Types of boundary conditions."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    MIXED = "mixed"


class RefinementStrategy(str, Enum):
    """Mesh refinement strategies."""

    H_REFINEMENT = "h"  # Reduce element size
    P_REFINEMENT = "p"  # Increase polynomial degree
    HP_REFINEMENT = "hp"  # Combined


class ActionSpace(str, Enum):
    """Types of action spaces for PDE games."""

    DISCRETE = "discrete"  # Finite set of actions
    CONTINUOUS = "continuous"  # Continuous action space
    HYBRID = "hybrid"  # Mixed discrete/continuous


class PDEConfig(BaseModuleConfig):
    """Configuration for a specific PDE problem.

    Defines the mathematical properties of the PDE including:
    - Equation type and coefficients
    - Boundary conditions
    - Domain specification
    - Source terms
    """

    pde_type: PDEType = Field(
        ...,
        description="Type of PDE (poisson, burgers, etc.)",
    )

    # Domain specification
    domain_dim: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Spatial dimension of the domain",
    )
    domain_min: list[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="Minimum domain coordinates",
    )
    domain_max: list[float] = Field(
        default_factory=lambda: [1.0, 1.0],
        description="Maximum domain coordinates",
    )

    # Boundary conditions
    boundary_condition: BoundaryCondition = Field(
        default=BoundaryCondition.DIRICHLET,
        description="Type of boundary condition",
    )
    boundary_value: float = Field(
        default=0.0,
        description="Boundary value for Dirichlet BC",
    )

    # PDE coefficients (interpretation depends on PDE type)
    diffusion_coeff: float = Field(
        default=1.0,
        gt=0.0,
        description="Diffusion coefficient (nu, kappa, etc.)",
    )
    advection_coeff: list[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="Advection velocity vector",
    )
    reaction_coeff: float = Field(
        default=0.0,
        description="Reaction coefficient",
    )

    # Time-dependent settings
    is_time_dependent: bool = Field(
        default=False,
        description="Whether the PDE is time-dependent",
    )
    time_start: float = Field(
        default=0.0,
        ge=0.0,
        description="Start time for time-dependent PDEs",
    )
    time_end: float = Field(
        default=1.0,
        gt=0.0,
        description="End time for time-dependent PDEs",
    )

    @field_validator("domain_min", "domain_max", "advection_coeff")
    @classmethod
    def validate_list_length(cls, v: list[float], info) -> list[float]:
        """Ensure lists have consistent length."""
        # Note: Full validation requires cross-field check in model_validator
        return v

    @model_validator(mode="after")
    def validate_domain(self) -> PDEConfig:
        """Ensure domain is properly specified."""
        if len(self.domain_min) != self.domain_dim:
            raise ValueError(
                f"domain_min length ({len(self.domain_min)}) must match "
                f"domain_dim ({self.domain_dim})"
            )
        if len(self.domain_max) != self.domain_dim:
            raise ValueError(
                f"domain_max length ({len(self.domain_max)}) must match "
                f"domain_dim ({self.domain_dim})"
            )
        if len(self.advection_coeff) != self.domain_dim:
            raise ValueError(
                f"advection_coeff length ({len(self.advection_coeff)}) must match "
                f"domain_dim ({self.domain_dim})"
            )
        for i, (lo, hi) in enumerate(zip(self.domain_min, self.domain_max, strict=True)):
            if lo >= hi:
                raise ValueError(f"Domain dimension {i}: min ({lo}) >= max ({hi})")
        if self.is_time_dependent and self.time_start >= self.time_end:
            raise ValueError(
                f"time_start ({self.time_start}) >= time_end ({self.time_end})"
            )
        return self


class BasisSelectionConfig(BaseModuleConfig):
    """Configuration for basis function selection game.

    In this game mode:
    - State: Current approximation space
    - Actions: Add a new basis function
    - Reward: Error reduction per computational cost
    - Terminal: Error < tolerance or budget exhausted
    """

    # Basis function parameters
    basis_type: Literal["fourier", "polynomial", "rbf", "wavelet"] = Field(
        default="fourier",
        description="Type of basis functions",
    )
    max_basis_functions: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of basis functions",
    )
    initial_basis_count: int = Field(
        default=1,
        ge=1,
        description="Initial number of basis functions",
    )

    # Action space parameters
    n_candidate_bases: int = Field(
        default=32,
        ge=2,
        le=256,
        description="Number of candidate basis functions per step",
    )
    basis_scale_range: tuple[float, float] = Field(
        default=(0.1, 10.0),
        description="Range of basis function scales",
    )

    # Fourier-specific parameters
    max_frequency: int = Field(
        default=50,
        ge=1,
        description="Maximum frequency for Fourier basis",
    )
    include_dc_component: bool = Field(
        default=True,
        description="Include DC (constant) component",
    )

    # RBF-specific parameters
    rbf_kernel: Literal["gaussian", "multiquadric", "inverse", "thin_plate"] = Field(
        default="gaussian",
        description="RBF kernel type",
    )

    # Numerical parameters
    n_collocation_points: int = Field(
        default=500,
        ge=10,
        le=100000,
        description="Number of collocation points for residual evaluation",
    )
    n_boundary_points_per_face: int = Field(
        default=50,
        ge=5,
        le=1000,
        description="Number of boundary points per face",
    )

    # Random seed for reproducibility
    seed: int | None = Field(
        default=None,
        description="Random seed for basis initialization (None = random)",
    )

    @model_validator(mode="after")
    def validate_basis_config(self) -> BasisSelectionConfig:
        """Validate basis selection parameters."""
        if self.initial_basis_count > self.max_basis_functions:
            raise ValueError(
                f"initial_basis_count ({self.initial_basis_count}) > "
                f"max_basis_functions ({self.max_basis_functions})"
            )
        low, high = self.basis_scale_range
        if low >= high:
            raise ValueError(f"Invalid basis_scale_range: {low} >= {high}")
        return self


class MeshRefinementConfig(BaseModuleConfig):
    """Configuration for adaptive mesh refinement game.

    In this game mode:
    - State: Current mesh + DG solution
    - Actions: Refine specific elements (h or p)
    - Reward: Error reduction per DOF added
    - Terminal: Error < tolerance or DOF budget exhausted
    """

    # Mesh parameters
    initial_resolution: int = Field(
        default=8,
        ge=2,
        le=128,
        description="Initial mesh resolution per dimension",
    )
    max_resolution: int = Field(
        default=256,
        ge=4,
        le=4096,
        description="Maximum mesh resolution per dimension",
    )
    min_element_size: float = Field(
        default=1e-4,
        gt=0.0,
        lt=1.0,
        description="Minimum element size (fraction of domain)",
    )

    # Refinement parameters
    refinement_strategy: RefinementStrategy = Field(
        default=RefinementStrategy.H_REFINEMENT,
        description="Refinement strategy (h, p, or hp)",
    )
    max_refinement_level: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum refinement level per element",
    )

    # Polynomial degree (for p-refinement)
    initial_polynomial_degree: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Initial polynomial degree",
    )
    max_polynomial_degree: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum polynomial degree",
    )

    # Action space
    n_candidate_elements: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Number of elements to consider for refinement",
    )
    allow_coarsening: bool = Field(
        default=False,
        description="Allow mesh coarsening (de-refinement)",
    )

    @model_validator(mode="after")
    def validate_mesh_config(self) -> MeshRefinementConfig:
        """Validate mesh refinement parameters."""
        if self.initial_resolution > self.max_resolution:
            raise ValueError(
                f"initial_resolution ({self.initial_resolution}) > "
                f"max_resolution ({self.max_resolution})"
            )
        if self.initial_polynomial_degree > self.max_polynomial_degree:
            raise ValueError(
                f"initial_polynomial_degree ({self.initial_polynomial_degree}) > "
                f"max_polynomial_degree ({self.max_polynomial_degree})"
            )
        return self


class PDEGameConfig(BaseModuleConfig):
    """Configuration for a PDE game session.

    Combines PDE specification with game parameters including:
    - Computational budget
    - Error tolerances
    - Reward shaping
    - Terminal conditions
    """

    # PDE specification
    pde_config: PDEConfig = Field(
        ...,
        description="Configuration for the PDE problem",
    )

    # Game mode
    game_mode: Literal["basis_selection", "mesh_refinement", "collocation"] = Field(
        default="basis_selection",
        description="Type of PDE game",
    )
    basis_config: BasisSelectionConfig | None = Field(
        default=None,
        description="Configuration for basis selection (if game_mode='basis_selection')",
    )
    mesh_config: MeshRefinementConfig | None = Field(
        default=None,
        description="Configuration for mesh refinement (if game_mode='mesh_refinement')",
    )

    # Computational budget
    max_dof: int = Field(
        default=10000,
        ge=10,
        le=1000000,
        description="Maximum degrees of freedom",
    )
    max_steps: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum game steps",
    )
    computational_budget: float = Field(
        default=1e6,
        gt=0.0,
        description="Total computational budget (FLOPs)",
    )

    # Tolerance and accuracy
    error_tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        lt=1.0,
        description="Target error tolerance",
    )
    error_metric: Literal["l2", "h1", "linf", "residual"] = Field(
        default="l2",
        description="Error metric for evaluation",
    )

    # Reward shaping
    reward_per_error_reduction: float = Field(
        default=1.0,
        gt=0.0,
        description="Reward scaling for error reduction",
    )
    cost_per_dof: float = Field(
        default=0.01,
        ge=0.0,
        description="Cost penalty per DOF added",
    )
    terminal_bonus: float = Field(
        default=10.0,
        ge=0.0,
        description="Bonus for reaching tolerance",
    )

    # Success metrics
    success_metrics: list[MetricDefinition] = Field(
        default_factory=lambda: [
            MetricDefinition(
                name="final_error",
                description="Final approximation error",
                operator=ThresholdOperator.LESS_THAN,
                threshold=1e-4,
            ),
            MetricDefinition(
                name="efficiency",
                description="Error reduction per DOF",
                operator=ThresholdOperator.GREATER_THAN,
                threshold=0.1,
            ),
        ],
        description="Metrics for evaluating game success",
    )

    # Phase detection thresholds (for curriculum learning)
    early_phase_step_threshold: int = Field(
        default=5,
        ge=0,
        description="Step threshold for early phase detection",
    )
    exploration_error_threshold: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Error threshold for exploration vs refinement phase",
    )

    # Random seed for reproducibility
    seed: int | None = Field(
        default=None,
        description="Random seed for game initialization (None = random)",
    )

    @model_validator(mode="after")
    def validate_game_config(self) -> PDEGameConfig:
        """Validate game mode has required sub-config."""
        if self.game_mode == "basis_selection" and self.basis_config is None:
            # Create default basis config
            self.basis_config = BasisSelectionConfig(name=f"{self.name}_basis")
        if self.game_mode == "mesh_refinement" and self.mesh_config is None:
            # Create default mesh config
            self.mesh_config = MeshRefinementConfig(name=f"{self.name}_mesh")
        return self
