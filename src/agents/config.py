"""Configuration classes for the agent-physics integration module.

All agent configurations use Pydantic validation with no hardcoded values.
Every parameter is exposed via Field() with constraints and descriptions.

Example:
    from src.agents.config import SolverAgentConfig, AgentType

    config = SolverAgentConfig(
        name="poisson_solver",
        game_mode="basis_selection",
        n_simulations=200,
    )

"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from src.pde.config import PDEConfig
from src.templates.config import BaseModuleConfig


class AgentType(str, Enum):
    """Types of agents in the orchestration framework."""

    SOLVER = "solver"
    DECOMPOSITION = "decomposition"
    COUPLING = "coupling"
    META = "meta"


class DecompositionStrategy(str, Enum):
    """Strategies for decomposing coupled PDE systems."""

    OPERATOR_SPLITTING = "operator_splitting"
    DOMAIN_DECOMPOSITION = "domain_decomposition"
    DIMENSIONAL_REDUCTION = "dimensional_reduction"


class CouplingType(str, Enum):
    """Types of coupling conditions between subdomains."""

    DIRICHLET_NEUMANN = "dirichlet_neumann"
    ROBIN_ROBIN = "robin_robin"
    MORTAR = "mortar"


class CollocationStrategy(str, Enum):
    """Strategies for collocation point allocation."""

    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    ERROR_GUIDED = "error_guided"


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    STATE_UPDATE = "state_update"
    BOUNDARY_DATA = "boundary_data"
    CONVERGENCE_CHECK = "convergence_check"
    STRATEGY_CHANGE = "strategy_change"
    BUDGET_UPDATE = "budget_update"


class AgentConfig(BaseModuleConfig):
    """Base configuration for all agent types.

    Provides common parameters shared across all agents including
    step limits, budget, and timeout.
    """

    agent_type: AgentType = Field(
        ...,
        description="Type of agent",
    )
    max_steps: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum number of agent steps",
    )
    error_tolerance: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Error tolerance for convergence",
    )
    computational_budget: float = Field(
        default=1.0,
        gt=0.0,
        description="Total computational budget (normalized)",
    )
    stall_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Steps without improvement before declaring stall",
    )
    stall_tolerance: float = Field(
        default=1e-6,
        gt=0.0,
        lt=1.0,
        description="Relative error change below which a solver is considered stalled",
    )


class SolverAgentConfig(AgentConfig):
    """Configuration for a SolverAgent that wraps PDEGame + MCTS.

    Controls the MCTS search parameters and temperature schedule
    for action selection during PDE solving.
    """

    agent_type: AgentType = Field(
        default=AgentType.SOLVER,
        description="Agent type (always solver)",
    )
    game_mode: Literal["basis_selection", "mesh_refinement"] = Field(
        default="basis_selection",
        description="Which PDE game mode to use",
    )
    n_simulations: int = Field(
        default=200,
        ge=1,
        le=10000,
        description="Number of MCTS simulations per move",
    )
    temperature_start: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Initial temperature for action sampling",
    )
    temperature_end: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Final temperature for action sampling",
    )
    temperature_decay_steps: int = Field(
        default=100,
        ge=1,
        le=100000,
        description="Steps over which temperature decays linearly",
    )
    c_puct: float = Field(
        default=1.5,
        gt=0.0,
        le=10.0,
        description="PUCT exploration constant for MCTS",
    )
    dirichlet_alpha: float = Field(
        default=0.03,
        gt=0.0,
        le=10.0,
        description="Dirichlet noise alpha for MCTS root exploration",
    )
    dirichlet_epsilon: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Fraction of Dirichlet noise to mix at root",
    )
    budget_per_step: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Budget consumed per step",
    )

    @model_validator(mode="after")
    def validate_temperature_schedule(self) -> Self:
        """Ensure temperature_start >= temperature_end."""
        if self.temperature_start < self.temperature_end:
            msg = (
                f"temperature_start ({self.temperature_start}) must be >= "
                f"temperature_end ({self.temperature_end})"
            )
            raise ValueError(msg)
        return self


class DecompositionConfig(AgentConfig):
    """Configuration for a DecompositionAgent.

    Controls how coupled PDE systems are split into subproblems.
    """

    agent_type: AgentType = Field(
        default=AgentType.DECOMPOSITION,
        description="Agent type (always decomposition)",
    )
    strategy: DecompositionStrategy = Field(
        default=DecompositionStrategy.OPERATOR_SPLITTING,
        description="Decomposition strategy to use",
    )
    max_subproblems: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum number of subproblems to create",
    )
    overlap_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Overlap fraction for domain decomposition",
    )
    splitting_order: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Splitting order (1=Lie-Trotter, 2=Strang)",
    )
    dimensional_reduction_threshold: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Ratio below which a thin dimension is dropped during dimensional reduction",
    )


class CouplingConfig(AgentConfig):
    """Configuration for a CouplingAgent.

    Controls interface condition enforcement and convergence checking
    between coupled subdomains.
    """

    agent_type: AgentType = Field(
        default=AgentType.COUPLING,
        description="Agent type (always coupling)",
    )
    coupling_type: CouplingType = Field(
        default=CouplingType.DIRICHLET_NEUMANN,
        description="Type of coupling condition",
    )
    tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        lt=1.0,
        description="Interface residual tolerance for convergence",
    )
    max_iterations: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum Schwarz iterations",
    )
    relaxation_factor: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Relaxation factor for boundary updates (omega)",
    )
    convergence_window: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of consecutive steps below tolerance for convergence",
    )
    budget_per_step: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Budget consumed per coupling step",
    )


class CollocationConfig(BaseModuleConfig):
    """Configuration for collocation point allocation.

    Controls how collocation points are distributed across the domain,
    including adaptive strategies that concentrate points where error is high.
    """

    strategy: CollocationStrategy = Field(
        default=CollocationStrategy.UNIFORM,
        description="Allocation strategy",
    )
    n_points: int = Field(
        default=1000,
        ge=1,
        le=1000000,
        description="Target number of collocation points",
    )
    adaptation_rate: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Rate of adaptation toward high-error regions",
    )
    error_threshold: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Error threshold for triggering reallocation",
    )
    min_points: int = Field(
        default=10,
        ge=1,
        le=100000,
        description="Minimum number of collocation points",
    )
    max_points: int = Field(
        default=100000,
        ge=1,
        le=10000000,
        description="Maximum number of collocation points",
    )
    importance_exponent: float = Field(
        default=1.0,
        gt=0.0,
        le=5.0,
        description="Exponent p for importance weighting |residual|^p",
    )
    perturbation_fraction: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="Perturbation scale as fraction of domain extent",
    )
    refined_oversampling_factor: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Multiplier for refined points around high-error regions",
    )

    @model_validator(mode="after")
    def validate_point_bounds(self) -> Self:
        """Ensure min_points <= n_points <= max_points."""
        if self.min_points > self.n_points:
            msg = f"min_points ({self.min_points}) must be <= n_points ({self.n_points})"
            raise ValueError(msg)
        if self.n_points > self.max_points:
            msg = f"n_points ({self.n_points}) must be <= max_points ({self.max_points})"
            raise ValueError(msg)
        return self


class MessageBusConfig(BaseModuleConfig):
    """Configuration for the inter-agent message bus."""

    buffer_size: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum messages per agent queue",
    )
    enable_logging: bool = Field(
        default=False,
        description="Log all messages passing through the bus",
    )


class CouplingPairConfig(BaseModuleConfig):
    """Configuration for a single coupling pair between two physics."""

    physics_a: str = Field(
        ...,
        min_length=1,
        description="Name of the first physics",
    )
    physics_b: str = Field(
        ...,
        min_length=1,
        description="Name of the second physics",
    )
    interface_type: CouplingType = Field(
        default=CouplingType.DIRICHLET_NEUMANN,
        description="Type of coupling at the interface",
    )
    interface_region_min: list[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="Minimum coordinates of interface region",
    )
    interface_region_max: list[float] = Field(
        default_factory=lambda: [1.0, 1.0],
        description="Maximum coordinates of interface region",
    )


class MultiPhysicsConfig(BaseModuleConfig):
    """Configuration for a coupled multi-physics problem.

    Defines the individual physics, their coupling relationships,
    and global convergence criteria.
    """

    physics: list[PDEConfig] = Field(
        ...,
        min_length=1,
        description="List of PDE configurations for each physics",
    )
    coupling_pairs: list[CouplingPairConfig] = Field(
        default_factory=list,
        description="Coupling conditions between physics pairs",
    )
    global_tolerance: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Global convergence tolerance",
    )
    max_schwarz_iterations: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum outer Schwarz iterations",
    )
    budget_allocation: dict[str, float] = Field(
        default_factory=dict,
        description="Budget fraction per physics (must sum to 1.0)",
    )

    @model_validator(mode="after")
    def validate_coupling_references(self) -> Self:
        """Ensure coupling pairs reference valid physics names."""
        physics_names = {p.name for p in self.physics}
        for pair in self.coupling_pairs:
            if pair.physics_a not in physics_names:
                msg = (
                    f"Coupling pair references unknown physics '{pair.physics_a}'. "
                    f"Available: {sorted(physics_names)}"
                )
                raise ValueError(msg)
            if pair.physics_b not in physics_names:
                msg = (
                    f"Coupling pair references unknown physics '{pair.physics_b}'. "
                    f"Available: {sorted(physics_names)}"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_budget_allocation(self) -> Self:
        """Ensure budget allocation sums to 1.0 if provided."""
        if self.budget_allocation:
            total = sum(self.budget_allocation.values())
            if abs(total - 1.0) > 1e-6:
                msg = f"budget_allocation values must sum to 1.0, got {total:.6f}"
                raise ValueError(msg)
            physics_names = {p.name for p in self.physics}
            for name in self.budget_allocation:
                if name not in physics_names:
                    msg = (
                        f"budget_allocation references unknown physics '{name}'. "
                        f"Available: {sorted(physics_names)}"
                    )
                    raise ValueError(msg)
        return self


class OrchestratorConfig(BaseModuleConfig):
    """Top-level configuration for the AgentOrchestrator.

    Composes all sub-configs needed to run a multi-physics solve.
    A single YAML file with this config drives the entire pipeline.
    """

    multi_physics: MultiPhysicsConfig = Field(
        ...,
        description="Multi-physics problem specification",
    )
    decomposition: DecompositionConfig = Field(
        default_factory=lambda: DecompositionConfig(
            name="default_decomposition",
            agent_type=AgentType.DECOMPOSITION,
        ),
        description="Decomposition strategy configuration",
    )
    solver_defaults: SolverAgentConfig = Field(
        default_factory=lambda: SolverAgentConfig(
            name="default_solver",
            agent_type=AgentType.SOLVER,
        ),
        description="Default solver agent configuration",
    )
    coupling: CouplingConfig = Field(
        default_factory=lambda: CouplingConfig(
            name="default_coupling",
            agent_type=AgentType.COUPLING,
        ),
        description="Coupling agent configuration",
    )
    collocation: CollocationConfig = Field(
        default_factory=lambda: CollocationConfig(name="default_collocation"),
        description="Collocation point allocation configuration",
    )
    message_bus: MessageBusConfig = Field(
        default_factory=lambda: MessageBusConfig(name="default_bus"),
        description="Message bus configuration",
    )
    parallel_solvers: bool = Field(
        default=False,
        description="Run solver agents in parallel (requires thread safety)",
    )
