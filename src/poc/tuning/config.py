"""Configuration schemas for hyperparameter tuning.

This module defines Pydantic models for tuning configuration,
including search space definitions and trial settings.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchSpace(BaseModel):
    """Definition of a single hyperparameter search space.

    Attributes:
        type: Parameter type (float, int, categorical).
        low: Lower bound (for float/int).
        high: Upper bound (for float/int).
        choices: List of choices (for categorical).
        log_scale: Whether to sample in log scale.
        step: Step size for discretization (optional).

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    type: Literal["float", "int", "categorical"] = Field(
        ...,
        description="Parameter type",
    )
    low: float | None = Field(
        default=None,
        description="Lower bound (float/int)",
    )
    high: float | None = Field(
        default=None,
        description="Upper bound (float/int)",
    )
    choices: list[Any] | None = Field(
        default=None,
        description="Choices (categorical)",
    )
    log_scale: bool = Field(
        default=False,
        description="Sample in log scale",
    )
    step: float | None = Field(
        default=None,
        description="Step size for discretization",
    )
    default: Any | None = Field(
        default=None,
        description="Default value",
    )

    @field_validator("type")
    @classmethod
    def validate_bounds(cls, v: str, info: Any) -> str:
        """Validate bounds based on type."""
        return v

    def sample_random(self) -> Any:
        """Sample a random value from this space.

        Returns:
            Sampled value.

        """
        import random

        if self.type == "categorical":
            return random.choice(self.choices)

        if self.type == "float":
            if self.log_scale:
                import math

                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return math.exp(random.uniform(log_low, log_high))
            return random.uniform(self.low, self.high)

        if self.type == "int":
            if self.log_scale:
                import math

                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return int(round(math.exp(random.uniform(log_low, log_high))))
            return random.randint(int(self.low), int(self.high))

        raise ValueError(f"Unknown type: {self.type}")


class TuningConfig(BaseModel):
    """Configuration for hyperparameter tuning.

    Attributes:
        n_trials: Number of trials to run.
        sampler: Sampling strategy.
        pruner: Early stopping strategy.
        search_space: Parameter search spaces.
        objective_metric: Metric to optimize.
        direction: Optimization direction.
        seed: Random seed.
        timeout_per_trial: Maximum time per trial.
        parallel_trials: Number of parallel trials.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Trial settings
    n_trials: int = Field(
        default=100,
        ge=1,
        description="Number of trials to run",
    )
    timeout_per_trial: int | None = Field(
        default=3600,
        ge=1,
        description="Timeout per trial in seconds",
    )

    # Sampling strategy
    sampler: Literal["random", "grid", "tpe", "cmaes"] = Field(
        default="tpe",
        description="Sampling strategy",
    )

    # Pruning strategy
    pruner: Literal["none", "median", "hyperband", "percentile"] = Field(
        default="median",
        description="Early stopping strategy",
    )
    pruner_n_startup_trials: int = Field(
        default=5,
        ge=0,
        description="Trials before pruning starts",
    )
    pruner_percentile: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Percentile for pruning (percentile pruner)",
    )

    # Search space
    search_space: dict[str, SearchSpace] = Field(
        default_factory=dict,
        description="Hyperparameter search spaces",
    )

    # Objective
    objective_metric: str = Field(
        default="mse",
        description="Metric to optimize",
    )
    direction: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Optimization direction",
    )

    # Multi-objective
    additional_metrics: list[str] = Field(
        default_factory=list,
        description="Additional metrics to track",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        description="Random seed",
    )

    # Parallelization
    parallel_trials: int = Field(
        default=1,
        ge=1,
        description="Number of parallel trials",
    )

    # Storage
    study_name: str = Field(
        default="alphagalerkin_tuning",
        description="Name for the study",
    )
    storage_path: str | None = Field(
        default=None,
        description="Path for persistent storage",
    )

    @field_validator("search_space", mode="before")
    @classmethod
    def convert_search_space(cls, v: dict[str, Any]) -> dict[str, SearchSpace]:
        """Convert dict values to SearchSpace objects."""
        if not v:
            return {}

        result = {}
        for key, value in v.items():
            if isinstance(value, SearchSpace):
                result[key] = value
            elif isinstance(value, dict):
                result[key] = SearchSpace(**value)
            else:
                raise ValueError(f"Invalid search space for {key}")
        return result


def create_search_space_from_dict(data: dict[str, Any]) -> dict[str, SearchSpace]:
    """Create search space from dictionary.

    Args:
        data: Dictionary mapping param names to space definitions.

    Returns:
        Dictionary mapping param names to SearchSpace objects.

    """
    result = {}
    for name, spec in data.items():
        if isinstance(spec, SearchSpace):
            result[name] = spec
        else:
            result[name] = SearchSpace(**spec)
    return result
