"""Base configuration classes for AlphaGalerkin modules.

This module provides reusable Pydantic configuration patterns with:
- Type-safe configuration with validation
- Constraint enforcement (ge, le, gt, lt)
- Deterministic hashing for reproducibility
- Factory functions for common patterns

Example:
    from src.templates.config import BaseModuleConfig, MetricDefinition
    from pydantic import Field

    class MyConfig(BaseModuleConfig):
        learning_rate: float = Field(
            default=0.001,
            gt=0.0,
            lt=1.0,
            description="Learning rate for optimizer",
        )
        batch_size: int = Field(
            default=32,
            ge=1,
            le=1024,
            description="Training batch size",
        )

"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ThresholdOperator(str, Enum):
    """Operators for metric threshold evaluation."""

    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    EQUAL = "=="
    NOT_EQUAL = "!="


class MetricDefinition(BaseModel):
    """Definition of a metric with threshold for success/failure.

    Attributes:
        name: Unique identifier for the metric.
        description: Human-readable description.
        operator: Comparison operator for threshold.
        threshold: Value to compare against.
        unit: Optional unit of measurement.

    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,  # Immutable after creation
    )

    name: str = Field(..., min_length=1, description="Metric identifier")
    description: str = Field(default="", description="Human description")
    operator: ThresholdOperator = Field(
        default=ThresholdOperator.LESS_THAN,
        description="Comparison operator",
    )
    threshold: float = Field(..., description="Threshold value")
    unit: str = Field(default="", description="Unit of measurement")

    def evaluate(self, value: float, tolerance: float = 1e-9) -> bool:
        """Evaluate if value meets the threshold.

        Args:
            value: The measured value to evaluate.
            tolerance: Floating point comparison tolerance.

        Returns:
            True if the value satisfies the threshold condition.

        """
        match self.operator:
            case ThresholdOperator.LESS_THAN:
                return value < self.threshold
            case ThresholdOperator.LESS_EQUAL:
                return value <= self.threshold + tolerance
            case ThresholdOperator.GREATER_THAN:
                return value > self.threshold
            case ThresholdOperator.GREATER_EQUAL:
                return value >= self.threshold - tolerance
            case ThresholdOperator.EQUAL:
                return abs(value - self.threshold) <= tolerance
            case ThresholdOperator.NOT_EQUAL:
                return abs(value - self.threshold) > tolerance

    def format_result(self, value: float) -> str:
        """Format the evaluation result as a string.

        Args:
            value: The measured value.

        Returns:
            Human-readable result string.

        """
        passed = self.evaluate(value)
        status = "PASS" if passed else "FAIL"
        unit_str = f" {self.unit}" if self.unit else ""
        return (
            f"[{status}] {self.name}: {value:.6f}{unit_str} {self.operator.value} {self.threshold}"
        )


class BaseModuleConfig(BaseModel):
    """Base configuration class for all AlphaGalerkin modules.

    Features:
    - extra="forbid" catches configuration typos
    - validate_assignment=True re-validates on attribute changes
    - Deterministic hashing for reproducibility tracking
    - Created timestamp for audit trail

    Subclasses should:
    1. Add domain-specific fields with Field() and constraints
    2. Override compute_hash() if needed for custom hashing
    3. Add @field_validator for complex single-field validation
    4. Add @model_validator for cross-field validation
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    # Required fields (subclasses may override defaults)
    name: str = Field(..., min_length=1, description="Unique identifier for this configuration")
    description: str = Field(default="", description="Human-readable description")

    # Common optional fields
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=1,
        le=86400,  # Max 24 hours
        description="Maximum execution time in seconds",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging",
    )

    # Metadata (auto-populated)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Configuration creation timestamp",
    )

    def compute_hash(self) -> str:
        """Compute deterministic hash of configuration.

        Used for reproducibility tracking and cache keys.
        Excludes volatile fields like created_at.

        Returns:
            16-character hex hash string.

        """
        # Exclude volatile fields from hash
        hash_data = self.model_dump(exclude={"created_at"})
        config_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to YAML-friendly dictionary.

        Converts enums to values, datetimes to ISO strings.

        Returns:
            Dictionary suitable for YAML serialization.

        """
        data = self.model_dump()

        def convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        result = convert(data)
        assert isinstance(result, dict)
        return result

    def with_overrides(self, **overrides: Any) -> BaseModuleConfig:
        """Create a new config with specified overrides.

        Args:
            **overrides: Field values to override.

        Returns:
            New configuration instance with overrides applied.

        """
        current = self.model_dump()
        current.update(overrides)
        return self.__class__(**current)


class TrainableModuleConfig(BaseModuleConfig):
    """Extended configuration for modules that involve training.

    Adds common training-related parameters with sensible defaults.
    """

    # Learning parameters
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        lt=1.0,
        description="Learning rate for optimizer",
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        lt=1.0,
        description="Weight decay (L2 regularization)",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=4096,
        description="Training batch size",
    )
    gradient_clip: float = Field(
        default=1.0,
        gt=0.0,
        le=100.0,
        description="Gradient clipping threshold",
    )

    # Training schedule
    total_steps: int = Field(
        default=10000,
        ge=1,
        description="Total training steps",
    )
    warmup_steps: int = Field(
        default=1000,
        ge=0,
        description="Linear warmup steps",
    )
    eval_interval: int = Field(
        default=1000,
        ge=1,
        description="Steps between evaluations",
    )
    checkpoint_interval: int = Field(
        default=5000,
        ge=1,
        description="Steps between checkpoints",
    )

    # Hardware
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device for training",
    )
    use_amp: bool = Field(
        default=True,
        description="Use automatic mixed precision",
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_intervals(self) -> TrainableModuleConfig:
        """Ensure intervals don't exceed total steps."""
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < total_steps ({self.total_steps})"
            )
        return self


class BoardSizeConfig(BaseModel):
    """Configuration for board size parameters.

    Reusable component for any module that works with variable board sizes.
    """

    model_config = ConfigDict(extra="forbid")

    min_size: int = Field(
        default=5,
        ge=3,
        le=25,
        description="Minimum board size",
    )
    max_size: int = Field(
        default=19,
        ge=3,
        le=25,
        description="Maximum board size",
    )
    sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Specific board sizes to use",
    )

    @field_validator("sizes")  # type: ignore[untyped-decorator]
    @classmethod
    def validate_sizes(cls, v: list[int]) -> list[int]:
        """Validate and normalize board sizes."""
        if not v:
            raise ValueError("sizes cannot be empty")
        for size in v:
            if size < 3 or size > 25:
                raise ValueError(f"Board size {size} must be between 3 and 25")
        return sorted(set(v))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_size_range(self) -> BoardSizeConfig:
        """Ensure sizes are within min/max range."""
        for size in self.sizes:
            if size < self.min_size or size > self.max_size:
                raise ValueError(f"Size {size} outside range [{self.min_size}, {self.max_size}]")
        return self


T = TypeVar("T", bound=BaseModuleConfig)


def create_config_class(
    name: str,
    base: type[T] = BaseModuleConfig,
    **field_definitions: tuple[type, Any],
) -> type[T]:
    """Factory function to create configuration classes dynamically.

    Args:
        name: Name for the new config class.
        base: Base class to inherit from.
        **field_definitions: Field definitions as (type, Field) tuples.

    Returns:
        New configuration class.

    Example:
        MyConfig = create_config_class(
            "MyConfig",
            base=BaseModuleConfig,
            my_param=(int, Field(default=100, ge=1)),
            my_float=(float, Field(default=0.5, gt=0, lt=1)),
        )

    """
    # Create annotations dict
    annotations: dict[str, type] = {}
    namespace: dict[str, Any] = {}

    for field_name, (field_type, field_default) in field_definitions.items():
        annotations[field_name] = field_type
        namespace[field_name] = field_default

    namespace["__annotations__"] = annotations

    # Create and return the class
    return type(name, (base,), namespace)
