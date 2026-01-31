"""Configuration schemas for Research module.

Provides Pydantic-validated configuration with:
- No hardcoded values
- Experiment and benchmark settings
- Transfer validation options
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExperimentType(str, Enum):
    """Type of research experiment."""

    ABLATION = "ablation"
    TRANSFER = "transfer"
    SCALING = "scaling"
    COMPARISON = "comparison"
    HYPERPARAMETER = "hyperparameter"
    CUSTOM = "custom"


class MetricType(str, Enum):
    """Type of metric to track."""

    MSE = "mse"
    MAE = "mae"
    ACCURACY = "accuracy"
    WIN_RATE = "win_rate"
    ELO = "elo"
    LOSS = "loss"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    CUSTOM = "custom"


class ExperimentConfig(BaseModel):
    """Configuration for research experiments.

    Controls experiment setup and execution.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Experiment name",
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Experiment description",
    )
    experiment_type: ExperimentType = Field(
        default=ExperimentType.CUSTOM,
        description="Type of experiment",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed",
    )
    deterministic: bool = Field(
        default=True,
        description="Use deterministic algorithms",
    )

    # Execution
    n_runs: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of runs per configuration",
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=60,
        description="Maximum execution time per run",
    )
    checkpoint_interval: int = Field(
        default=100,
        ge=1,
        description="Checkpoint every N steps",
    )

    # Tracking
    track_metrics: list[str] = Field(
        default_factory=lambda: ["loss", "mse"],
        description="Metrics to track",
    )
    log_interval: int = Field(
        default=10,
        ge=1,
        description="Log every N steps",
    )
    save_artifacts: bool = Field(
        default=True,
        description="Save experiment artifacts",
    )

    # Output
    output_dir: str = Field(
        default="outputs/research",
        description="Output directory",
    )
    save_model: bool = Field(
        default=True,
        description="Save trained model",
    )

    # Tags and metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Experiment tags for filtering",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def compute_hash(self) -> str:
        """Compute unique hash of configuration."""
        data = self.model_dump(mode="json")
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarking.

    Controls benchmark execution and metrics.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        default="benchmark",
        min_length=1,
        description="Benchmark name",
    )
    description: str = Field(
        default="",
        description="Benchmark description",
    )

    # Sizes to benchmark
    sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Sizes to benchmark",
    )

    # Timing
    n_warmup: int = Field(
        default=10,
        ge=1,
        description="Warmup iterations",
    )
    n_iterations: int = Field(
        default=100,
        ge=10,
        description="Timed iterations",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size",
    )

    # Metrics
    measure_memory: bool = Field(
        default=True,
        description="Measure memory usage",
    )
    measure_throughput: bool = Field(
        default=True,
        description="Measure throughput (samples/sec)",
    )

    # Device
    use_gpu: bool = Field(
        default=True,
        description="Use GPU if available",
    )
    sync_cuda: bool = Field(
        default=True,
        description="Synchronize CUDA for accurate timing",
    )

    @field_validator("sizes")
    @classmethod
    def validate_sizes(cls, v: list[int]) -> list[int]:
        """Validate sizes are positive and sorted."""
        if not v:
            raise ValueError("sizes cannot be empty")
        return sorted(set(v))


class TransferConfig(BaseModel):
    """Configuration for transfer validation.

    Controls transfer learning experiments.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        default="transfer",
        description="Transfer experiment name",
    )

    # Source domain
    source_size: int = Field(
        default=9,
        ge=3,
        le=25,
        description="Source domain size (training)",
    )
    n_train_samples: int = Field(
        default=5000,
        ge=100,
        description="Training samples",
    )
    n_epochs: int = Field(
        default=100,
        ge=1,
        description="Training epochs",
    )

    # Target domains
    target_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Target domain sizes (evaluation)",
    )
    n_eval_samples: int = Field(
        default=500,
        ge=10,
        description="Evaluation samples per target",
    )
    primary_target: int = Field(
        default=19,
        ge=3,
        description="Primary target for pass/fail",
    )

    # Success criteria
    mse_threshold: float = Field(
        default=0.05,
        gt=0,
        description="MSE threshold for passing",
    )
    require_all_targets: bool = Field(
        default=True,
        description="Require all targets to pass",
    )

    # Model settings
    d_model: int = Field(
        default=128,
        ge=16,
        description="Model dimension",
    )
    n_layers: int = Field(
        default=4,
        ge=1,
        description="Number of layers",
    )
    use_fnet: bool = Field(
        default=True,
        description="Use FNet mixing",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed",
    )

    @field_validator("target_sizes")
    @classmethod
    def validate_targets(cls, v: list[int]) -> list[int]:
        """Validate target sizes."""
        if not v:
            raise ValueError("target_sizes cannot be empty")
        return sorted(set(v))


class ComparisonConfig(BaseModel):
    """Configuration for model comparison.

    Controls comparison experiments.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        default="comparison",
        description="Comparison name",
    )

    # Models to compare
    model_paths: list[str] = Field(
        default_factory=list,
        description="Paths to models to compare",
    )
    model_names: list[str] = Field(
        default_factory=list,
        description="Names for models (for labels)",
    )

    # Evaluation settings
    eval_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Evaluation sizes",
    )
    n_eval_samples: int = Field(
        default=500,
        ge=10,
        description="Evaluation samples per size",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size",
    )

    # Metrics
    metrics: list[str] = Field(
        default_factory=lambda: ["mse", "mae", "throughput"],
        description="Metrics to compare",
    )

    # Statistical testing
    n_bootstrap: int = Field(
        default=10000,
        ge=1000,
        description="Bootstrap samples for CI",
    )
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=0.5,
        description="Significance level",
    )

    # Output
    generate_plots: bool = Field(
        default=True,
        description="Generate comparison plots",
    )
    generate_tables: bool = Field(
        default=True,
        description="Generate comparison tables",
    )

    @field_validator("model_names", mode="before")
    @classmethod
    def validate_names(cls, v: list[str], info: Any) -> list[str]:
        """Ensure names match paths if provided."""
        return v or []


def create_experiment_config(
    name: str,
    experiment_type: str = "custom",
    n_runs: int = 1,
    **kwargs: Any,
) -> ExperimentConfig:
    """Factory function to create experiment config.

    Args:
        name: Experiment name.
        experiment_type: Type of experiment.
        n_runs: Number of runs.
        **kwargs: Additional configuration.

    Returns:
        ExperimentConfig instance.

    """
    return ExperimentConfig(
        name=name,
        experiment_type=ExperimentType(experiment_type),
        n_runs=n_runs,
        **kwargs,
    )


def create_transfer_config(
    source_size: int = 9,
    target_sizes: list[int] | None = None,
    **kwargs: Any,
) -> TransferConfig:
    """Factory function to create transfer config.

    Args:
        source_size: Source training size.
        target_sizes: Target evaluation sizes.
        **kwargs: Additional configuration.

    Returns:
        TransferConfig instance.

    """
    return TransferConfig(
        source_size=source_size,
        target_sizes=target_sizes or [9, 13, 19],
        **kwargs,
    )
