"""Pydantic v2 configuration system for AlphaGalerkin.

This module implements a hierarchical, validated configuration tree
that drives every tunable aspect of the framework.  Configuration
values come from three sources (highest priority first):

1. **Environment variables** -- prefixed ``AG_``
   (e.g. ``AG_MCTS__C_PUCT=3.0``).
2. **YAML file** -- loaded via ``AlphaGalerkinConfig.from_yaml(path)``.
3. **Defaults** -- defined inline on each Pydantic field.

All sub-configs are frozen after construction (``validate_assignment``
ensures mutations are validated).  The root ``AlphaGalerkinConfig``
exposes a ``from_yaml`` classmethod that handles the three-way merge.

Section reference (system prompt): Section 9 -- Configuration.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from src.alphagalerkin.core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_C_PUCT,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPSILON,
    DEFAULT_ERROR_TOLERANCE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DOF,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TREE_DEPTH,
    DEFAULT_NUM_ATTENTION_HEADS,
    DEFAULT_NUM_GNN_LAYERS,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
)
from src.alphagalerkin.core.exceptions import (
    ConfigError,
    ConfigValidationError,
)
from src.alphagalerkin.core.types import (
    BackupStrategy,
    Formulation,
    GNNArchitecture,
    NormalizationType,
    PDEType,
    PoolingType,
    RewardWeights,
    SelectionPolicy,
    TemperatureScheduleType,
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge *override* into *base* (non-destructive).

    Leaf values in *override* take precedence.  Both input dicts
    are left unmodified; a fresh dict is returned.

    >>> _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 99}})
    {'a': {'b': 99, 'c': 2}}
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides(prefix: str = "AG_") -> dict[str, Any]:
    """Collect environment-variable overrides into a nested dict.

    Variable names use double-underscore ``__`` as the nesting
    separator.  Example: ``AG_MCTS__C_PUCT=3.0`` maps to
    ``{"mcts": {"c_puct": "3.0"}}``.

    Returns
    -------
    dict[str, Any]
        A nested dict suitable for ``_deep_merge``.

    """
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        node = overrides
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


# -------------------------------------------------------------------
# Base config
# -------------------------------------------------------------------

class _Base(BaseModel):
    """Shared Pydantic v2 model config for every sub-config."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        arbitrary_types_allowed=False,
        str_strip_whitespace=True,
    )


# -------------------------------------------------------------------
# MCTS
# -------------------------------------------------------------------

class TemperatureSchedule(_Base):
    """Temperature annealing schedule for MCTS action sampling.

    Controls exploration-exploitation trade-off over the course
    of an episode (or training).
    """

    schedule_type: TemperatureScheduleType = Field(
        default=TemperatureScheduleType.STEP,
        description="Annealing schedule family.",
    )
    initial_temp: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature at the start of the schedule.",
    )
    final_temp: float = Field(
        default=0.1,
        gt=0.0,
        description="Temperature at the end of the schedule.",
    )
    step_threshold: int = Field(
        default=30,
        ge=0,
        description=(
            "Step at which STEP schedule transitions to "
            "final_temp."
        ),
    )
    decay_rate: float = Field(
        default=0.97,
        gt=0.0,
        le=1.0,
        description="Per-step decay for EXPONENTIAL schedule.",
    )

    @model_validator(mode="after")
    def _validate_temps(self) -> Self:
        if self.initial_temp < self.final_temp:
            raise ValueError(
                f"initial_temp ({self.initial_temp}) must be "
                f">= final_temp ({self.final_temp})"
            )
        return self


class MCTSConfig(_Base):
    """Monte Carlo Tree Search hyper-parameters."""

    num_simulations: int = Field(
        default=DEFAULT_NUM_SIMULATIONS,
        ge=1,
        description="Simulations per move.",
    )
    max_tree_depth: int = Field(
        default=DEFAULT_MAX_TREE_DEPTH,
        ge=1,
        description="Maximum depth of the search tree.",
    )
    c_puct: float = Field(
        default=DEFAULT_C_PUCT,
        gt=0.0,
        description="PUCT exploration constant.",
    )
    selection_policy: SelectionPolicy = Field(
        default=SelectionPolicy.PUCT,
        description="Child-selection policy.",
    )
    backup_strategy: BackupStrategy = Field(
        default=BackupStrategy.MEAN,
        description="Value backup strategy.",
    )
    dirichlet_alpha: float = Field(
        default=DEFAULT_DIRICHLET_ALPHA,
        gt=0.0,
        description="Dirichlet noise concentration.",
    )
    dirichlet_epsilon: float = Field(
        default=DEFAULT_DIRICHLET_EPSILON,
        ge=0.0,
        le=1.0,
        description="Root prior noise fraction.",
    )
    temperature_schedule: TemperatureSchedule = Field(
        default_factory=TemperatureSchedule,
        description="Action-sampling temperature schedule.",
    )
    action_topk: int = Field(
        default=50,
        ge=1,
        description="Max actions expanded per node.",
    )
    noise_at_root_only: bool = Field(
        default=True,
        description="Only add noise at root node.",
    )
    virtual_loss: float = Field(
        default=1.0,
        ge=0.0,
        description="Virtual loss for parallel tree search.",
    )
    value_delta_cutoff: float = Field(
        default=0.01,
        ge=0.0,
        description=(
            "Early-stop simulations when root value "
            "delta < cutoff."
        ),
    )


# -------------------------------------------------------------------
# Network
# -------------------------------------------------------------------

class GNNConfig(_Base):
    """Graph neural network (mesh encoder) configuration."""

    architecture: GNNArchitecture = Field(
        default=GNNArchitecture.GAT,
        description="GNN architecture family.",
    )
    hidden_dim: int = Field(
        default=DEFAULT_HIDDEN_DIM,
        ge=8,
        description="Hidden dimension per layer.",
    )
    num_layers: int = Field(
        default=DEFAULT_NUM_GNN_LAYERS,
        ge=1,
        description="Number of message-passing layers.",
    )
    attention_heads: int = Field(
        default=DEFAULT_NUM_ATTENTION_HEADS,
        ge=1,
        description="Attention heads per layer.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability.",
    )
    residual: bool = Field(
        default=True,
        description="Use residual connections.",
    )
    normalization: NormalizationType = Field(
        default=NormalizationType.LAYER,
        description="Normalization strategy.",
    )
    edge_feature_dim: int = Field(
        default=8,
        ge=0,
        description=(
            "Edge feature dimension (0 = no edge features)."
        ),
    )
    activation: str = Field(
        default="gelu",
        description="Activation function (gelu or relu).",
    )


class PolicyHeadConfig(_Base):
    """Configuration for the policy (action-distribution) head."""

    hidden_dim: int = Field(
        default=DEFAULT_HIDDEN_DIM,
        ge=8,
        description="Hidden dimension of the MLP.",
    )
    hidden_dims: list[int] = Field(
        default_factory=lambda: [128, 64],
        description="Hidden layer dimensions for the MLP.",
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        description="Number of MLP layers.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability.",
    )


class ValueHeadConfig(_Base):
    """Configuration for the value (state-evaluation) head."""

    hidden_dim: int = Field(
        default=DEFAULT_HIDDEN_DIM,
        ge=8,
        description="Hidden dimension of the MLP.",
    )
    hidden_dims: list[int] = Field(
        default_factory=lambda: [128, 64],
        description="Hidden layer dimensions for the MLP.",
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        description="Number of MLP layers.",
    )
    pooling: PoolingType = Field(
        default=PoolingType.ATTENTION,
        description="Graph pooling strategy.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability.",
    )


class NetworkConfig(_Base):
    """Top-level neural network configuration.

    Bundles the mesh encoder (GNN), policy head, and value head
    into a single validated unit.
    """

    gnn: GNNConfig = Field(
        default_factory=GNNConfig,
        description="Mesh encoder GNN config.",
    )
    policy_head: PolicyHeadConfig = Field(
        default_factory=PolicyHeadConfig,
        description="Policy head config.",
    )
    value_head: ValueHeadConfig = Field(
        default_factory=ValueHeadConfig,
        description="Value head config.",
    )
    input_features: int = Field(
        default=8,
        ge=1,
        description="Number of per-element input features.",
    )
    node_feature_dim: int = Field(
        default=32,
        ge=1,
        description="Per-element node feature dimension.",
    )
    global_feature_dim: int = Field(
        default=16,
        ge=0,
        description=(
            "Global (state-level) feature dimension "
            "(0 = no global features)."
        ),
    )


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

class OptimizerConfig(_Base):
    """Optimizer hyper-parameters."""

    name: Literal[
        "adam", "adamw", "sgd", "rmsprop",
    ] = Field(
        default="adamw",
        description="Optimizer algorithm.",
    )
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        gt=0.0,
        description="Initial learning rate.",
    )
    weight_decay: float = Field(
        default=DEFAULT_WEIGHT_DECAY,
        ge=0.0,
        description="L2 regularisation coefficient.",
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Adam / AdamW beta parameters.",
    )
    momentum: float = Field(
        default=0.9,
        ge=0.0,
        lt=1.0,
        description="SGD momentum.",
    )
    gradient_clip_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Max gradient norm for clipping.",
    )


class SchedulerConfig(_Base):
    """Learning-rate scheduler configuration."""

    name: Literal[
        "cosine",
        "step",
        "exponential",
        "reduce_on_plateau",
        "none",
    ] = Field(
        default="cosine",
        description="Scheduler algorithm.",
    )
    warmup_steps: int = Field(
        default=1000,
        ge=0,
        description="Linear warmup steps.",
    )
    min_lr: float = Field(
        default=1e-6,
        ge=0.0,
        description="Minimum learning rate.",
    )
    step_size: int = Field(
        default=10_000,
        ge=1,
        description="Step size for step scheduler.",
    )
    gamma: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="LR decay factor.",
    )
    patience: int = Field(
        default=10,
        ge=1,
        description="Patience for reduce_on_plateau.",
    )


class ReplayConfig(_Base):
    """Experience replay buffer configuration."""

    capacity: int = Field(
        default=DEFAULT_REPLAY_CAPACITY,
        ge=1000,
        description="Maximum stored transitions.",
    )
    prioritized: bool = Field(
        default=True,
        description="Use prioritized experience replay.",
    )
    priority_alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Priority exponent.",
    )
    priority_beta_start: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Initial importance-sampling exponent.",
    )
    priority_beta_end: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Final importance-sampling exponent.",
    )
    min_size_to_train: int = Field(
        default=1000,
        ge=1,
        description="Min buffer fill before training begins.",
    )

    @model_validator(mode="after")
    def _validate_beta_range(self) -> Self:
        if self.priority_beta_start > self.priority_beta_end:
            raise ValueError(
                f"priority_beta_start "
                f"({self.priority_beta_start}) must be <= "
                f"priority_beta_end ({self.priority_beta_end})"
            )
        return self


class CurriculumConfig(_Base):
    """Curriculum learning configuration.

    The curriculum progresses through a sequence of difficulty
    stages, each defined by a PDE problem or set of environment
    parameter overrides.
    """

    enabled: bool = Field(
        default=False,
        description="Enable curriculum learning.",
    )
    stages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Ordered list of curriculum stages. Each stage is "
            "a dict of environment-config overrides."
        ),
    )
    advance_threshold: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Win-rate threshold to advance.",
    )
    evaluation_window: int = Field(
        default=100,
        ge=10,
        description="Episodes to evaluate for advancement.",
    )


class TrainingConfig(_Base):
    """Top-level training loop configuration."""

    seed: int = Field(
        default=DEFAULT_SEED,
        description="Global random seed.",
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        ge=1,
        description="Mini-batch size.",
    )
    total_steps: int = Field(
        default=100_000,
        ge=1,
        description="Total training steps.",
    )
    self_play_games_per_step: int = Field(
        default=25,
        ge=1,
        description="Self-play games per training step.",
    )
    policy_loss_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for policy (cross-entropy) loss.",
    )
    value_loss_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for value (MSE) loss.",
    )
    lbb_loss_weight: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight for LBB regularisation loss.",
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer config.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    replay: ReplayConfig = Field(
        default_factory=ReplayConfig,
        description="Replay buffer config.",
    )
    curriculum: CurriculumConfig = Field(
        default_factory=CurriculumConfig,
        description="Curriculum learning config.",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Data-loader workers (0 = main process).",
    )
    mixed_precision: bool = Field(
        default=False,
        description="Enable AMP mixed-precision training.",
    )
    compile_model: bool = Field(
        default=False,
        description="Use torch.compile() for the network.",
    )


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

class EnvironmentConfig(_Base):
    """Discretization environment (MDP) configuration.

    Attributes:
        max_dof: Maximum DOF budget before the episode terminates
            with a budget-exceeded penalty.
        max_steps: Hard cap on actions per episode.
        error_tolerance: Target L2 error for termination.
        formulation: CG or DG Galerkin formulation.
        initial_mesh_resolution: Uniform-mesh elements per side
            at environment reset.
        initial_polynomial_order: Polynomial degree assigned to
            every element at reset.
        max_polynomial_order: Upper bound on polynomial degree
            reachable by p-refinement.
        min_element_size: Smallest allowed element diameter.
        default_basis_family: Basis family for new elements.
        reward_weights: Multi-objective reward weights.
        normalize_rewards: Apply running-mean normalisation.

    """

    max_dof: int = Field(
        default=DEFAULT_MAX_DOF,
        ge=10,
        description="Maximum DOF budget.",
    )
    max_steps: int = Field(
        default=DEFAULT_MAX_STEPS,
        ge=1,
        description="Maximum steps per episode.",
    )
    error_tolerance: float = Field(
        default=DEFAULT_ERROR_TOLERANCE,
        gt=0.0,
        description="Target L2 error tolerance.",
    )
    formulation: Formulation = Field(
        default=Formulation.CONTINUOUS,
        description="Galerkin formulation (CG / DG).",
    )
    initial_mesh_resolution: int = Field(
        default=4,
        ge=1,
        description="Initial uniform-mesh elements per side.",
    )
    initial_polynomial_order: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Initial polynomial order on each element.",
    )
    max_polynomial_order: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Max polynomial degree via p-refinement.",
    )
    min_element_size: float = Field(
        default=1e-4,
        gt=0.0,
        description=(
            "Minimum element diameter; h-refine is blocked "
            "when a child would be smaller."
        ),
    )
    default_basis_family: str = Field(
        default="lagrange",
        description="Basis family assigned to new elements.",
    )
    reward_weights: RewardWeights = Field(
        default_factory=lambda: {
            "accuracy": 1.0,
            "efficiency": 0.5,
            "stability": 0.3,
        },
        description="Multi-objective reward weights.",
    )
    normalize_rewards: bool = Field(
        default=True,
        description="Apply running-mean reward normalisation.",
    )

    @model_validator(mode="after")
    def _validate_polynomial_bounds(self) -> Self:
        if self.initial_polynomial_order > self.max_polynomial_order:
            raise ValueError(
                f"initial_polynomial_order "
                f"({self.initial_polynomial_order}) must be <= "
                f"max_polynomial_order "
                f"({self.max_polynomial_order})"
            )
        return self


# -------------------------------------------------------------------
# Physics
# -------------------------------------------------------------------

class PhysicsConfig(_Base):
    """PDE / physics problem specification.

    Defines the mathematical problem that the discretization
    environment must solve at each step.
    """

    pde_type: PDEType = Field(
        default=PDEType.ELLIPTIC,
        description="PDE classification.",
    )
    domain_dim: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Spatial dimension.",
    )
    domain_bounds: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 1.0), (0.0, 1.0)],
        description=(
            "Domain bounding box as (min, max) per dimension."
        ),
    )
    boundary_conditions: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "dirichlet",
            "value": 0.0,
        },
        description="Boundary condition specification.",
    )
    source_term: str = Field(
        default="sin(pi*x)*sin(pi*y)",
        description="Source / forcing term as a parseable expr.",
    )
    diffusion_coefficient: float = Field(
        default=1.0,
        gt=0.0,
        description="Diffusion coefficient.",
    )
    manufactured_solution: str | None = Field(
        default=None,
        description=(
            "Optional manufactured-solution expression for "
            "error computation."
        ),
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if len(self.domain_bounds) != self.domain_dim:
            raise ValueError(
                f"domain_bounds length "
                f"({len(self.domain_bounds)}) must match "
                f"domain_dim ({self.domain_dim})"
            )
        for i, (lo, hi) in enumerate(self.domain_bounds):
            if lo >= hi:
                raise ValueError(
                    f"domain_bounds[{i}]: "
                    f"min ({lo}) >= max ({hi})"
                )
        return self


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

class LoggingConfig(_Base):
    """Structured logging and metric tracking configuration."""

    level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
    ] = Field(
        default="INFO",
        description="Root log level.",
    )
    log_dir: str = Field(
        default="logs",
        description="Directory for log files.",
    )
    log_to_file: bool = Field(
        default=True,
        description="Write logs to a file in log_dir.",
    )
    log_to_console: bool = Field(
        default=True,
        description="Write logs to stderr.",
    )
    metrics_backend: Literal[
        "wandb", "tensorboard", "none",
    ] = Field(
        default="none",
        description="Experiment-tracking backend.",
    )
    wandb_project: str = Field(
        default="alphagalerkin",
        description="W&B project name.",
    )
    wandb_entity: str | None = Field(
        default=None,
        description="W&B entity (team or user).",
    )
    log_interval_steps: int = Field(
        default=100,
        ge=1,
        description="Steps between metric log flushes.",
    )


# -------------------------------------------------------------------
# Checkpoint
# -------------------------------------------------------------------

class CheckpointConfig(_Base):
    """Checkpoint saving / loading configuration."""

    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory for checkpoint files.",
    )
    save_interval_steps: int = Field(
        default=5000,
        ge=1,
        description="Steps between checkpoint saves.",
    )
    keep_last_n: int = Field(
        default=5,
        ge=1,
        description="Most-recent checkpoints to retain.",
    )
    save_best: bool = Field(
        default=True,
        description="Keep a copy of the best checkpoint.",
    )
    best_metric: str = Field(
        default="value_loss",
        description="Metric used to determine best checkpoint.",
    )
    best_metric_mode: Literal["min", "max"] = Field(
        default="min",
        description="Whether lower or higher is better.",
    )
    resume_from: str | None = Field(
        default=None,
        description=(
            "Path to checkpoint to resume from. "
            "None = start fresh."
        ),
    )


# -------------------------------------------------------------------
# Root configuration
# -------------------------------------------------------------------

class AlphaGalerkinConfig(_Base):
    """Root configuration for the AlphaGalerkin framework.

    Aggregates every sub-config into a single validated tree.
    Construct from a YAML file using the ``from_yaml`` classmethod,
    which automatically applies environment variable overrides.

    Example::

        cfg = AlphaGalerkinConfig.from_yaml("config/ag.yaml")
    """

    experiment_name: str = Field(
        default="alphagalerkin",
        min_length=1,
        max_length=128,
        description="Human-readable experiment name.",
    )
    seed: int = Field(
        default=DEFAULT_SEED,
        description=(
            "Global random seed (propagated to sub-configs)."
        ),
    )
    device: str = Field(
        default="cpu",
        description="PyTorch device string (e.g. 'cuda:0').",
    )
    mcts: MCTSConfig = Field(
        default_factory=MCTSConfig,
        description="MCTS hyper-parameters.",
    )
    network: NetworkConfig = Field(
        default_factory=NetworkConfig,
        description="Neural network architecture.",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training loop configuration.",
    )
    environment: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Discretization environment.",
    )
    physics: PhysicsConfig = Field(
        default_factory=PhysicsConfig,
        description="PDE / physics problem.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging and metrics.",
    )
    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
        description="Checkpoint management.",
    )

    # ---- Seed propagation ----

    @model_validator(mode="after")
    def _propagate_seed(self) -> Self:
        """Push the root seed into the training sub-config."""
        if self.training.seed == DEFAULT_SEED:
            object.__setattr__(
                self.training, "seed", self.seed,
            )
        return self

    # ---- Construction helpers ----

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> AlphaGalerkinConfig:
        """Load configuration from a YAML file.

        Merge order (highest-priority last):

        1. YAML file
        2. Explicit *overrides* dict
        3. Environment variables prefixed ``AG_``

        Parameters
        ----------
        path:
            Path to the YAML configuration file.
        overrides:
            Optional programmatic overrides.

        Returns
        -------
        AlphaGalerkinConfig
            Fully validated configuration object.

        Raises
        ------
        ConfigError
            If the YAML file cannot be read.
        ConfigValidationError
            If validation fails.

        """
        filepath = Path(path)
        if not filepath.exists():
            raise ConfigError(
                f"Configuration file not found: {filepath}",
                path=str(filepath),
            )

        try:
            with filepath.open("r") as fh:
                raw: dict[str, Any] = (
                    yaml.safe_load(fh) or {}
                )
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Failed to parse YAML: {exc}",
                path=str(filepath),
            ) from exc

        if overrides:
            raw = _deep_merge(raw, overrides)

        env_ovr = _env_overrides()
        if env_ovr:
            raw = _deep_merge(raw, env_ovr)

        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise ConfigValidationError(
                f"Configuration validation failed: {exc}",
                path=str(filepath),
            ) from exc

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> AlphaGalerkinConfig:
        """Construct from an in-memory dict (tests, notebooks).

        Environment-variable overrides are still applied.

        Parameters
        ----------
        data:
            Raw configuration dictionary.

        Returns
        -------
        AlphaGalerkinConfig
            Fully validated configuration object.

        """
        env_ovr = _env_overrides()
        if env_ovr:
            data = _deep_merge(data, env_ovr)

        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise ConfigValidationError(
                f"Configuration validation failed: {exc}",
            ) from exc
