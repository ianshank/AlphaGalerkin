"""Configuration schemas for AlphaGalerkin using Pydantic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainConfig(BaseModel):
    """Configuration for the physical domain Omega = [0,1]^2."""

    # Domain boundaries (normalized to unit square)
    x_min: float = Field(default=0.0, description="Minimum x coordinate")
    x_max: float = Field(default=1.0, description="Maximum x coordinate")
    y_min: float = Field(default=0.0, description="Minimum y coordinate")
    y_max: float = Field(default=1.0, description="Maximum y coordinate")

    @field_validator("x_max")
    @classmethod
    def x_max_greater_than_x_min(cls, v: float, info: object) -> float:
        """Ensure x_max > x_min."""
        # Access data through info.data in Pydantic v2
        data = getattr(info, "data", {})
        if "x_min" in data and v <= data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def y_max_greater_than_y_min(cls, v: float, info: object) -> float:
        """Ensure y_max > y_min."""
        data = getattr(info, "data", {})
        if "y_min" in data and v <= data["y_min"]:
            raise ValueError("y_max must be greater than y_min")
        return v


class OperatorConfig(BaseModel):
    """Configuration for the Neural Operator model.

    Note: No hard-coded board sizes! Resolution independence is achieved
    by parameterizing in terms of model dimensions, not sequence lengths.
    """

    # Model dimensions (resolution-independent)
    d_model: int = Field(default=256, description="Model embedding dimension")
    d_key: int = Field(default=64, description="Key dimension for attention")
    d_value: int = Field(default=64, description="Value dimension for attention")
    d_ffn: int = Field(default=1024, description="Feed-forward network dimension")

    # Attention configuration
    n_heads: int = Field(default=8, description="Number of attention heads")
    n_galerkin_layers: int = Field(default=6, description="Number of Galerkin attention layers")
    n_softmax_layers: int = Field(default=2, description="Number of softmax attention layers")

    # Fourier features configuration
    n_fourier_features: int = Field(
        default=128, description="Number of Fourier feature frequencies"
    )
    fourier_scale: float = Field(default=1.0, description="Scale for Fourier features")

    # FNet configuration
    use_fnet_mixing: bool = Field(default=True, description="Use FNet for fast mixing")
    fnet_dropout: float = Field(default=0.1, description="Dropout rate in FNet blocks")

    # LBB stability configuration
    lbb_beta_threshold: float = Field(
        default=1e-6, description="Minimum singular value threshold for LBB stability"
    )

    # Normalization
    norm_type: Literal["layernorm", "scalenorm", "galerkin"] = Field(
        default="layernorm", description="Normalization type"
    )

    # Input channels (stone colors + move history + etc.)
    input_channels: int = Field(default=17, description="Number of input feature planes")

    @field_validator("d_key")
    @classmethod
    def key_dim_constraint(cls, v: int, info: object) -> int:
        """Ensure key dimension satisfies LBB condition (dim(Key) >= dim(Query)).

        Since we use the same dimension for queries and keys by default,
        this is a placeholder for future flexibility.
        """
        if v < 1:
            raise ValueError("d_key must be positive")
        return v


class MCTSConfig(BaseModel):
    """Configuration for Monte Carlo Tree Search."""

    # Search parameters
    n_simulations: int = Field(default=800, description="Number of MCTS simulations per move")
    c_puct: float = Field(default=1.5, description="PUCT exploration constant")

    # Dirichlet noise for root exploration
    dirichlet_alpha: float = Field(default=0.03, description="Dirichlet noise alpha")
    dirichlet_epsilon: float = Field(
        default=0.25, description="Dirichlet noise mixing coefficient"
    )

    # Temperature for move selection
    temperature: float = Field(default=1.0, description="Temperature for move selection")
    temperature_drop_move: int = Field(
        default=30, description="Move number to drop temperature to 0"
    )

    # Batch inference
    batch_size: int = Field(default=8, description="Batch size for parallel leaf evaluation")

    # Virtual loss for parallel MCTS
    virtual_loss: float = Field(default=3.0, description="Virtual loss for parallel search")


class TrainingConfig(BaseModel):
    """Configuration for training loop."""

    # Optimization
    learning_rate: float = Field(default=2e-4, description="Initial learning rate")
    weight_decay: float = Field(default=1e-4, description="Weight decay for regularization")
    batch_size: int = Field(default=256, description="Training batch size")
    gradient_clip: float = Field(default=1.0, description="Gradient clipping norm")

    # Learning rate schedule
    lr_scheduler: Literal["cosine", "constant", "linear"] = Field(
        default="cosine", description="Learning rate scheduler type"
    )
    warmup_steps: int = Field(default=1000, description="Number of warmup steps")
    total_steps: int = Field(default=100000, description="Total training steps")

    # Self-play
    n_self_play_games: int = Field(default=100, description="Self-play games per iteration")
    replay_buffer_size: int = Field(default=500000, description="Replay buffer capacity")

    # Loss weights
    policy_loss_weight: float = Field(default=1.0, description="Weight for policy loss")
    value_loss_weight: float = Field(default=1.0, description="Weight for value loss")

    # Checkpointing
    checkpoint_interval: int = Field(default=1000, description="Steps between checkpoints")

    # Mixed precision
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")

    # Evaluation
    eval_interval: int = Field(default=5000, description="Steps between evaluations")
    eval_games: int = Field(default=20, description="Number of evaluation games")


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases logging.

    This is the single source of truth for W&B configuration.
    The WandbLogger accepts this configuration via model_dump().
    """

    # Core settings
    enabled: bool = Field(default=True, description="Enable W&B logging")
    project: str = Field(default="alphagalerkin", description="W&B project name")
    entity: str | None = Field(default=None, description="W&B team/user entity")
    name: str | None = Field(default=None, description="Run name (auto-generated if None)")
    tags: list[str] = Field(default_factory=list, description="Tags for the run")
    notes: str | None = Field(default=None, description="Notes for the run")
    group: str | None = Field(default=None, description="Group for organizing runs")
    job_type: str = Field(default="train", description="Job type (train, eval, etc.)")

    # Mode settings
    mode: Literal["online", "offline", "disabled"] = Field(
        default="online", description="W&B mode"
    )

    # Logging settings
    log_model: bool = Field(default=True, description="Log model checkpoints as artifacts")
    log_gradients: bool = Field(default=False, description="Log gradient histograms")
    log_code: bool = Field(default=True, description="Log source code")
    log_interval: int = Field(default=1, description="Steps between metric logging")

    # Model watching
    watch_model: bool = Field(default=False, description="Use wandb.watch() on model")
    watch_log_freq: int = Field(default=100, description="Frequency for gradient logging")

    # Resume configuration (for resuming W&B runs)
    resume_id: str | None = Field(default=None, description="W&B run ID to resume")
    resume_mode: str = Field(default="allow", description="Resume mode: 'allow', 'must', 'never'")


class AlphaGalerkinConfig(BaseModel):
    """Root configuration combining all sub-configurations."""

    # Sub-configurations
    domain: DomainConfig = Field(default_factory=DomainConfig)
    operator: OperatorConfig = Field(default_factory=OperatorConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # Experiment tracking
    experiment_name: str = Field(default="alphagalerkin", description="Experiment name")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_lbb_metrics: bool = Field(default=True, description="Log LBB stability metrics")

    # Runtime settings (previously unschematized)
    device: str = Field(default="auto", description="Training device: 'auto', 'cuda', 'cpu'")
    checkpoint_dir: str = Field(default="checkpoints", description="Directory for checkpoints")
    log_interval: int = Field(default=100, description="Steps between console logging")
    board_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Board sizes to train on (resolution-independent)",
    )

    # Resume configuration
    resume: str | None = Field(default=None, description="Path to checkpoint to resume from")

    # Allow extra fields for forward compatibility
    model_config = ConfigDict(extra="ignore")
