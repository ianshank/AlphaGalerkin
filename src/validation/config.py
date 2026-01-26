"""Configuration schemas for validation framework.

This module defines Pydantic models for all validation configurations,
ensuring type safety, validation, and serialization.

Design Principles:
    - No hardcoded values: All constants are configurable with sensible defaults
    - Backwards compatible: New fields have defaults, removed fields are deprecated
    - Validated: Pydantic enforces types and constraints at runtime
    - Serializable: All configs can be saved/loaded from YAML/JSON
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ValidationStatus(str, Enum):
    """Status of a validation step."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ToleranceLevel(str, Enum):
    """Predefined tolerance levels for numerical comparisons."""

    STRICT = "strict"  # 1e-7 rtol, 1e-9 atol
    STANDARD = "standard"  # 1e-5 rtol, 1e-8 atol
    RELAXED = "relaxed"  # 1e-4 rtol, 1e-6 atol
    LOOSE = "loose"  # 1e-3 rtol, 1e-5 atol
    CUSTOM = "custom"  # User-defined tolerances


class ToleranceConfig(BaseModel):
    """Configuration for numerical tolerance in tests.

    Provides flexible tolerance configuration for addressing
    precision issues in mathematical tests.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Tolerance level presets
    level: ToleranceLevel = Field(
        default=ToleranceLevel.STANDARD,
        description="Predefined tolerance level",
    )

    # Custom tolerances (used when level=CUSTOM or to override presets)
    rtol: float | None = Field(
        default=None,
        ge=0,
        description="Relative tolerance (overrides level preset if set)",
    )
    atol: float | None = Field(
        default=None,
        ge=0,
        description="Absolute tolerance (overrides level preset if set)",
    )

    # Dtype-specific tolerances
    float32_rtol: float = Field(
        default=1e-5,
        ge=0,
        description="Relative tolerance for float32",
    )
    float32_atol: float = Field(
        default=1e-6,
        ge=0,
        description="Absolute tolerance for float32",
    )
    float64_rtol: float = Field(
        default=1e-7,
        ge=0,
        description="Relative tolerance for float64",
    )
    float64_atol: float = Field(
        default=1e-9,
        ge=0,
        description="Absolute tolerance for float64",
    )

    # Behavior options
    check_dtype: bool = Field(
        default=True,
        description="Whether to adjust tolerance based on tensor dtype",
    )
    allow_nan: bool = Field(
        default=False,
        description="Whether to allow NaN values in comparisons",
    )
    allow_inf: bool = Field(
        default=False,
        description="Whether to allow Inf values in comparisons",
    )

    # Debugging
    verbose: bool = Field(
        default=False,
        description="Log detailed comparison information",
    )
    max_failures_to_report: int = Field(
        default=10,
        ge=1,
        description="Maximum number of mismatches to report",
    )

    _PRESETS: dict[ToleranceLevel, tuple[float, float]] = {
        ToleranceLevel.STRICT: (1e-7, 1e-9),
        ToleranceLevel.STANDARD: (1e-5, 1e-8),
        ToleranceLevel.RELAXED: (1e-4, 1e-6),
        ToleranceLevel.LOOSE: (1e-3, 1e-5),
    }

    def get_tolerance(self) -> tuple[float, float]:
        """Get (rtol, atol) based on configuration.

        Returns:
            Tuple of (relative_tolerance, absolute_tolerance).
        """
        # Custom overrides
        if self.rtol is not None and self.atol is not None:
            return (self.rtol, self.atol)

        # Preset values
        if self.level in self._PRESETS:
            preset_rtol, preset_atol = self._PRESETS[self.level]
            return (
                self.rtol if self.rtol is not None else preset_rtol,
                self.atol if self.atol is not None else preset_atol,
            )

        # Default to standard if somehow CUSTOM with missing values
        return (1e-5, 1e-8)


class GPUTrainingConfig(BaseModel):
    """Configuration for GPU training validation.

    Validates training on GPU with larger models and full pipelines.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Identification
    name: str = Field(
        default="gpu_training",
        description="Validation identifier",
    )
    description: str = Field(
        default="Validate training pipeline on GPU with larger model",
        description="Human-readable description",
    )

    # Device settings
    device: str = Field(
        default="auto",
        description="Device for training: 'auto', 'cuda', 'cuda:0', etc.",
    )
    require_gpu: bool = Field(
        default=True,
        description="Fail if GPU is not available",
    )
    mixed_precision: bool = Field(
        default=True,
        description="Use automatic mixed precision (AMP)",
    )

    # Model configuration
    d_model: int = Field(
        default=256,
        ge=64,
        description="Model hidden dimension",
    )
    n_heads: int = Field(
        default=8,
        ge=1,
        description="Number of attention heads",
    )
    n_layers: int = Field(
        default=6,
        ge=1,
        description="Number of transformer layers",
    )

    # Training settings
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Training batch size",
    )
    n_steps: int = Field(
        default=1000,
        ge=10,
        description="Number of training steps",
    )
    learning_rate: float = Field(
        default=1e-4,
        gt=0,
        description="Learning rate",
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Learning rate warmup steps",
    )

    # Data settings
    board_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Board sizes to train on",
    )
    n_train_samples: int = Field(
        default=10000,
        ge=100,
        description="Number of training samples",
    )

    # Validation criteria
    max_loss_threshold: float = Field(
        default=5.0,
        gt=0,
        description="Maximum acceptable final loss",
    )
    loss_decrease_threshold: float = Field(
        default=0.5,
        ge=0,
        lt=1,
        description="Minimum loss decrease ratio (1 - final/initial)",
    )
    min_lbb_constant: float = Field(
        default=1e-6,
        gt=0,
        description="Minimum acceptable LBB constant",
    )
    max_gradient_norm: float = Field(
        default=100.0,
        gt=0,
        description="Maximum acceptable gradient norm",
    )

    # Resource limits
    timeout_seconds: int = Field(
        default=3600,
        ge=60,
        description="Maximum training time",
    )
    memory_limit_gb: float | None = Field(
        default=None,
        description="Maximum GPU memory to use (None for no limit)",
    )

    # Checkpointing
    checkpoint_interval: int = Field(
        default=200,
        ge=1,
        description="Steps between checkpoints",
    )
    save_best_model: bool = Field(
        default=True,
        description="Save model with lowest loss",
    )

    # Logging
    log_interval: int = Field(
        default=10,
        ge=1,
        description="Steps between log messages",
    )
    log_gradients: bool = Field(
        default=False,
        description="Log gradient statistics",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        description="Random seed",
    )

    @field_validator("board_sizes")
    @classmethod
    def validate_board_sizes(cls, v: list[int]) -> list[int]:
        """Ensure board sizes are valid Go board sizes."""
        for size in v:
            if size < 5 or size > 25:
                raise ValueError(f"Invalid board size {size}: must be in [5, 25]")
        return sorted(set(v))


class TransferValidationConfig(BaseModel):
    """Configuration for zero-shot transfer validation.

    Validates that a model trained on one resolution generalizes
    to different resolutions without retraining.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Identification
    name: str = Field(
        default="transfer_validation",
        description="Validation identifier",
    )
    description: str = Field(
        default="Validate zero-shot transfer from 9x9 to 19x19",
        description="Human-readable description",
    )

    # Resolution settings
    train_resolution: int = Field(
        default=9,
        ge=5,
        le=25,
        description="Grid size for training",
    )
    eval_resolutions: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Grid sizes for evaluation",
    )
    primary_eval_resolution: int = Field(
        default=19,
        ge=5,
        description="Primary resolution for pass/fail determination",
    )

    # Model source
    model_path: str | None = Field(
        default=None,
        description="Path to pre-trained model (None = train new)",
    )
    use_gpu_model: bool = Field(
        default=True,
        description="Use model from GPU training validation",
    )

    # Training settings (if training new model)
    n_epochs: int = Field(
        default=100,
        ge=1,
        description="Training epochs for new model",
    )
    n_train_samples: int = Field(
        default=5000,
        ge=100,
        description="Number of training samples",
    )
    n_eval_samples: int = Field(
        default=500,
        ge=10,
        description="Evaluation samples per resolution",
    )

    # Model architecture
    d_model: int = Field(
        default=128,
        ge=32,
        description="Model hidden dimension",
    )
    n_heads: int = Field(
        default=4,
        ge=1,
        description="Number of attention heads",
    )
    n_layers: int = Field(
        default=4,
        ge=1,
        description="Number of transformer layers",
    )

    # Evaluation settings
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Evaluation batch size",
    )
    n_charges: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of point charges for physics data",
    )

    # Success criteria
    mse_threshold: float = Field(
        default=0.05,
        gt=0,
        description="MSE threshold for transfer success",
    )
    relative_mse_threshold: float = Field(
        default=2.0,
        gt=1.0,
        description="Max ratio of transfer MSE to train MSE",
    )

    # Tolerance for numerical comparisons
    tolerance: ToleranceConfig = Field(
        default_factory=lambda: ToleranceConfig(level=ToleranceLevel.RELAXED),
        description="Tolerance for numerical comparisons",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        description="Random seed",
    )

    @field_validator("eval_resolutions")
    @classmethod
    def validate_eval_resolutions(cls, v: list[int]) -> list[int]:
        """Ensure eval resolutions are valid."""
        if not v:
            raise ValueError("eval_resolutions cannot be empty")
        for res in v:
            if res < 5 or res > 25:
                raise ValueError(f"Invalid resolution {res}: must be in [5, 25]")
        return sorted(set(v))

    @model_validator(mode="after")
    def validate_primary_in_eval(self) -> TransferValidationConfig:
        """Ensure primary eval resolution is in eval list."""
        if self.primary_eval_resolution not in self.eval_resolutions:
            self.eval_resolutions = sorted(
                set(self.eval_resolutions + [self.primary_eval_resolution])
            )
        return self


class TestFailureInfo(BaseModel):
    """Information about a failing test."""

    test_name: str = Field(..., description="Fully qualified test name")
    file_path: str = Field(..., description="Path to test file")
    line_number: int | None = Field(default=None, description="Line number of test")
    error_message: str = Field(..., description="Error message")
    failure_type: str = Field(
        default="assertion",
        description="Type of failure: assertion, tolerance, exception",
    )
    suggested_fix: str | None = Field(
        default=None,
        description="Suggested fix for the failure",
    )


class MergeReadinessConfig(BaseModel):
    """Configuration for PR merge readiness checking.

    Validates that all requirements are met for merging a PR.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Identification
    name: str = Field(
        default="merge_readiness",
        description="Validation identifier",
    )
    description: str = Field(
        default="Check PR #7 merge readiness",
        description="Human-readable description",
    )
    pr_number: int = Field(
        default=7,
        ge=1,
        description="Pull request number to check",
    )

    # Test requirements
    require_all_tests_pass: bool = Field(
        default=True,
        description="All tests must pass",
    )
    allowed_test_failures: list[str] = Field(
        default_factory=list,
        description="Test names allowed to fail (known issues)",
    )
    max_allowed_failures: int = Field(
        default=0,
        ge=0,
        description="Maximum number of allowed test failures",
    )

    # Tolerance requirements
    check_tolerance_tests: bool = Field(
        default=True,
        description="Check for tolerance-related test failures",
    )
    tolerance_config: ToleranceConfig = Field(
        default_factory=lambda: ToleranceConfig(level=ToleranceLevel.STANDARD),
        description="Tolerance configuration for precision tests",
    )

    # Code quality requirements
    require_lint_pass: bool = Field(
        default=True,
        description="Linting must pass (ruff check)",
    )
    require_type_check_pass: bool = Field(
        default=True,
        description="Type checking must pass (mypy)",
    )
    require_no_fixmes: bool = Field(
        default=False,
        description="No FIXME comments allowed",
    )
    require_no_todos: bool = Field(
        default=False,
        description="No TODO comments allowed",
    )

    # Documentation requirements
    require_docstrings: bool = Field(
        default=False,
        description="All public functions must have docstrings",
    )
    require_changelog: bool = Field(
        default=False,
        description="CHANGELOG.md must be updated",
    )

    # Dependency validation
    check_dependencies: bool = Field(
        default=True,
        description="Check for dependency conflicts",
    )

    # Command overrides (for flexibility)
    lint_command: str = Field(
        default="ruff check src/",
        description="Lint command to run",
    )
    type_check_command: str = Field(
        default="mypy src/ --strict",
        description="Type check command to run",
    )
    test_command: str = Field(
        default="pytest tests/ -v",
        description="Test command to run",
    )

    # Timeouts
    test_timeout_seconds: int = Field(
        default=600,
        ge=60,
        description="Maximum time for test suite",
    )
    lint_timeout_seconds: int = Field(
        default=60,
        ge=10,
        description="Maximum time for linting",
    )


class ValidationResult(BaseModel):
    """Result of a validation step.

    Captures all information needed for analysis and debugging.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    # Identification
    validation_name: str = Field(..., description="Validation identifier")
    config_hash: str = Field(..., description="Hash of configuration used")

    # Status
    status: ValidationStatus = Field(..., description="Execution status")
    passed: bool = Field(..., description="Whether validation passed")

    # Metrics and results
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Collected metrics",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result details",
    )

    # Test failures (for merge readiness)
    test_failures: list[TestFailureInfo] = Field(
        default_factory=list,
        description="List of test failures",
    )

    # Artifacts
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Paths to generated artifacts",
    )

    # Timing
    start_time: datetime = Field(..., description="Execution start time")
    end_time: datetime = Field(..., description="Execution end time")
    duration_seconds: float = Field(..., description="Total execution time")

    # Error information
    error_message: str | None = Field(
        default=None,
        description="Error message if any",
    )
    error_traceback: str | None = Field(
        default=None,
        description="Full traceback if error",
    )

    # Environment
    device: str = Field(default="cpu", description="Computation device used")
    python_version: str = Field(default="", description="Python version")
    gpu_info: str | None = Field(default=None, description="GPU information if used")

    def summary(self) -> str:
        """Generate human-readable summary."""
        status_indicator = {
            ValidationStatus.PASSED: "[PASS]",
            ValidationStatus.FAILED: "[FAIL]",
            ValidationStatus.ERROR: "[ERROR]",
            ValidationStatus.SKIPPED: "[SKIP]",
            ValidationStatus.PENDING: "[PENDING]",
            ValidationStatus.RUNNING: "[RUNNING]",
        }
        indicator = status_indicator.get(self.status, "[?]")

        lines = [
            f"{indicator} {self.validation_name}",
            f"   Status: {self.status.value}",
            f"   Duration: {self.duration_seconds:.2f}s",
            f"   Device: {self.device}",
        ]

        if self.metrics:
            lines.append("   Metrics:")
            for name, value in sorted(self.metrics.items()):
                lines.append(f"     {name}: {value:.6f}")

        if self.test_failures:
            lines.append(f"   Test Failures: {len(self.test_failures)}")
            for failure in self.test_failures[:5]:
                lines.append(f"     - {failure.test_name}: {failure.error_message[:50]}")
            if len(self.test_failures) > 5:
                lines.append(f"     ... and {len(self.test_failures) - 5} more")

        if self.error_message:
            lines.append(f"   Error: {self.error_message}")

        return "\n".join(lines)


class ValidationConfig(BaseModel):
    """Root configuration for the validation framework.

    Combines all validation step configurations.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
    )

    # Validation steps to run
    run_tolerance_fix: bool = Field(
        default=True,
        description="Run tolerance/precision test fixes",
    )
    run_gpu_training: bool = Field(
        default=True,
        description="Run GPU training validation",
    )
    run_transfer_validation: bool = Field(
        default=True,
        description="Run zero-shot transfer validation",
    )
    run_merge_readiness: bool = Field(
        default=True,
        description="Run merge readiness check",
    )

    # Step configurations
    tolerance: ToleranceConfig = Field(
        default_factory=ToleranceConfig,
        description="Tolerance configuration",
    )
    gpu_training: GPUTrainingConfig = Field(
        default_factory=GPUTrainingConfig,
        description="GPU training configuration",
    )
    transfer_validation: TransferValidationConfig = Field(
        default_factory=TransferValidationConfig,
        description="Transfer validation configuration",
    )
    merge_readiness: MergeReadinessConfig = Field(
        default_factory=MergeReadinessConfig,
        description="Merge readiness configuration",
    )

    # Execution settings
    stop_on_failure: bool = Field(
        default=False,
        description="Stop on first validation failure",
    )
    parallel: bool = Field(
        default=False,
        description="Run independent validations in parallel",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        description="Maximum parallel workers",
    )

    # Output settings
    output_dir: str = Field(
        default="outputs/validation",
        description="Directory for validation outputs",
    )
    save_results: bool = Field(
        default=True,
        description="Save validation results to JSON",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        description="Global random seed",
    )

    def compute_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        config_str = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
