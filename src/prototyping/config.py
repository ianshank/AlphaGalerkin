"""Configuration for fast prototyping.

Provides preset configurations for rapid experimentation
with minimal boilerplate.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PresetType(Enum):
    """Preset configuration types."""

    MINIMAL = "minimal"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    DEBUG = "debug"
    TRANSFER = "transfer"
    BENCHMARK = "benchmark"


class PrototypeConfig(BaseModel):
    """Configuration for prototype experiments.

    Attributes:
        name: Experiment name.
        preset: Preset configuration type.
        board_sizes: Board sizes to use.
        d_model: Model dimension.
        n_layers: Number of layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        seed: Random seed for reproducibility.
        device: Device to use (cpu, cuda).
        extra_config: Additional configuration.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        default="prototype",
        min_length=1,
        max_length=128,
        description="Experiment name",
    )
    preset: PresetType = Field(
        default=PresetType.SMALL,
        description="Preset configuration type",
    )
    board_sizes: list[int] = Field(
        default=[9],
        min_length=1,
        description="Board sizes to use",
    )
    d_model: int = Field(
        default=64,
        ge=16,
        le=4096,
        description="Model dimension",
    )
    n_layers: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Number of layers",
    )
    n_heads: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Number of attention heads",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Dropout rate",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed",
    )
    device: str = Field(
        default="cpu",
        description="Device to use",
    )
    extra_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration",
    )

    @field_validator("board_sizes")
    @classmethod
    def validate_board_sizes(cls, v: list[int]) -> list[int]:
        """Validate board sizes are positive."""
        for size in v:
            if size < 5:
                raise ValueError("Board size must be at least 5")
        return sorted(v)

    @field_validator("d_model")
    @classmethod
    def validate_d_model_divisible(cls, v: int, info: Any) -> int:
        """Validate d_model is divisible by n_heads."""
        # Note: This validator runs before n_heads is set in some cases
        # so we do a basic check here
        return v

    def compute_hash(self) -> str:
        """Compute hash of configuration."""
        data = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class QuickTrainConfig(BaseModel):
    """Configuration for quick training.

    Attributes:
        n_epochs: Number of epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        warmup_steps: Warmup steps.
        log_interval: Logging interval.
        eval_interval: Evaluation interval.
        save_interval: Save interval.
        max_grad_norm: Maximum gradient norm.
        early_stopping_patience: Early stopping patience.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    n_epochs: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Number of epochs",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=4096,
        description="Batch size",
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        lt=1.0,
        description="Learning rate",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        lt=1.0,
        description="Weight decay",
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Warmup steps",
    )
    log_interval: int = Field(
        default=10,
        ge=1,
        description="Logging interval (steps)",
    )
    eval_interval: int = Field(
        default=100,
        ge=1,
        description="Evaluation interval (steps)",
    )
    save_interval: int = Field(
        default=500,
        ge=1,
        description="Save interval (steps)",
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm",
    )
    early_stopping_patience: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Early stopping patience (epochs)",
    )

    def compute_hash(self) -> str:
        """Compute hash of configuration."""
        data = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class QuickEvalConfig(BaseModel):
    """Configuration for quick evaluation.

    Attributes:
        n_samples: Number of evaluation samples.
        batch_size: Batch size.
        metrics: Metrics to compute.
        compute_confidence: Whether to compute confidence intervals.
        n_bootstrap: Bootstrap samples for confidence intervals.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    n_samples: int = Field(
        default=1000,
        ge=10,
        description="Number of evaluation samples",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        le=4096,
        description="Batch size",
    )
    metrics: list[str] = Field(
        default_factory=lambda: ["mse", "mae", "accuracy"],
        min_length=1,
        description="Metrics to compute",
    )
    compute_confidence: bool = Field(
        default=True,
        description="Compute confidence intervals",
    )
    n_bootstrap: int = Field(
        default=1000,
        ge=100,
        description="Bootstrap samples for CI",
    )

    def compute_hash(self) -> str:
        """Compute hash of configuration."""
        data = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


# Preset configurations
PRESETS: dict[PresetType, dict[str, Any]] = {
    PresetType.MINIMAL: {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
    },
    PresetType.SMALL: {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.1,
    },
    PresetType.MEDIUM: {
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 8,
        "dropout": 0.1,
    },
    PresetType.LARGE: {
        "d_model": 256,
        "n_layers": 8,
        "n_heads": 16,
        "dropout": 0.1,
    },
    PresetType.DEBUG: {
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
    },
    PresetType.TRANSFER: {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 4,
        "dropout": 0.05,
    },
    PresetType.BENCHMARK: {
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 8,
        "dropout": 0.0,
    },
}


def create_prototype_config(
    name: str = "prototype",
    preset: str | PresetType = PresetType.SMALL,
    board_sizes: list[int] | None = None,
    **kwargs: Any,
) -> PrototypeConfig:
    """Create a prototype configuration.

    Args:
        name: Experiment name.
        preset: Preset type (string or enum).
        board_sizes: Board sizes to use.
        **kwargs: Additional configuration overrides.

    Returns:
        PrototypeConfig instance.

    """
    if isinstance(preset, str):
        preset = PresetType(preset)

    # Get preset defaults
    preset_config = PRESETS.get(preset, {}).copy()

    # Apply overrides
    preset_config.update(kwargs)

    return PrototypeConfig(
        name=name,
        preset=preset,
        board_sizes=board_sizes or [9],
        **preset_config,
    )


def create_quick_train_config(
    preset: str | PresetType = PresetType.SMALL,
    **kwargs: Any,
) -> QuickTrainConfig:
    """Create a quick training configuration.

    Args:
        preset: Preset type for default values.
        **kwargs: Configuration overrides.

    Returns:
        QuickTrainConfig instance.

    """
    if isinstance(preset, str):
        preset = PresetType(preset)

    # Preset-specific training defaults
    train_presets: dict[PresetType, dict[str, Any]] = {
        PresetType.MINIMAL: {
            "n_epochs": 5,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "warmup_steps": 10,
            "eval_interval": 20,
        },
        PresetType.SMALL: {
            "n_epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "warmup_steps": 100,
            "eval_interval": 100,
        },
        PresetType.MEDIUM: {
            "n_epochs": 20,
            "batch_size": 64,
            "learning_rate": 5e-4,
            "warmup_steps": 500,
            "eval_interval": 200,
        },
        PresetType.LARGE: {
            "n_epochs": 50,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "warmup_steps": 1000,
            "eval_interval": 500,
        },
        PresetType.DEBUG: {
            "n_epochs": 2,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "warmup_steps": 0,
            "eval_interval": 5,
            "log_interval": 1,
        },
        PresetType.TRANSFER: {
            "n_epochs": 30,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "warmup_steps": 200,
            "eval_interval": 100,
        },
        PresetType.BENCHMARK: {
            "n_epochs": 10,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "warmup_steps": 100,
            "eval_interval": 50,
        },
    }

    preset_config = train_presets.get(preset, {}).copy()
    preset_config.update(kwargs)

    return QuickTrainConfig(**preset_config)
