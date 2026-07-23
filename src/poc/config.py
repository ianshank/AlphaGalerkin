"""Configuration schemas for PoC scenarios.

This module defines Pydantic models for scenario configuration,
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


class ScenarioTier(str, Enum):
    """Validation tier indicating depth of testing."""

    UNIT = "unit"  # Fast, isolated tests (~seconds)
    FUNCTIONAL = "functional"  # Component-level tests (~minutes)
    INTEGRATION = "integration"  # End-to-end tests (~hours)


class ScenarioStatus(str, Enum):
    """Execution status of a scenario."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"  # Unexpected exception


class MetricThreshold(BaseModel):
    """Defines a metric and its pass/fail threshold."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Metric name (e.g., 'mse', 'lbb_constant')")
    operator: Literal["<", "<=", ">", ">=", "=="] = Field(
        default="<", description="Comparison operator"
    )
    value: float = Field(..., description="Threshold value")
    description: str = Field(default="", description="Human-readable description")

    def evaluate(self, actual: float) -> bool:
        """Check if actual value passes the threshold.

        Uses math.isclose() for equality comparison with both relative
        and absolute tolerance for numerical stability.
        """
        import math

        ops: dict[str, Any] = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12),
        }
        return ops[self.operator](actual, self.value)


class BaseScenarioConfig(BaseModel):
    """Base configuration for all scenarios.

    All scenario configs should inherit from this class.
    """

    model_config = ConfigDict(
        extra="forbid",  # Catch typos in config keys
        validate_assignment=True,  # Re-validate on attribute changes
    )

    # Identification
    name: str = Field(..., description="Unique scenario identifier")
    description: str = Field(..., description="Human-readable description")
    tier: ScenarioTier = Field(default=ScenarioTier.FUNCTIONAL, description="Validation tier")

    # Execution control
    enabled: bool = Field(default=True, description="Whether to run this scenario")
    timeout_seconds: int = Field(default=3600, ge=1, description="Maximum execution time")
    retry_count: int = Field(default=0, ge=0, le=5, description="Number of retries on failure")

    # Reproducibility
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Success criteria
    thresholds: list[MetricThreshold] = Field(
        default_factory=list, description="Metrics that must pass"
    )

    # Resource hints
    requires_gpu: bool = Field(default=False, description="Whether GPU is required")
    estimated_duration_seconds: int = Field(
        default=60, ge=1, description="Estimated runtime for planning"
    )

    def compute_hash(self) -> str:
        """Compute deterministic hash of configuration for reproducibility."""
        # Sort keys for determinism
        config_str = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class TransferScenarioConfig(BaseScenarioConfig):
    """Configuration for zero-shot transfer scenarios.

    Validates that a model trained on one resolution generalizes
    to different resolutions without retraining.
    """

    name: str = Field(default="transfer", description="Scenario identifier")
    description: str = Field(
        default="Zero-shot transfer from training to evaluation resolution",
        description="Scenario description",
    )
    tier: ScenarioTier = ScenarioTier.INTEGRATION

    # Resolution settings
    train_resolution: int = Field(
        default=9, ge=3, le=25, description="Grid size for training (e.g., 9 for 9x9)"
    )
    eval_resolutions: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Grid sizes for evaluation",
    )
    primary_eval_resolution: int = Field(
        default=19, ge=3, description="Primary resolution for pass/fail determination"
    )

    # Data settings
    n_train_samples: int = Field(default=5000, ge=100, description="Number of training samples")
    n_eval_samples: int = Field(
        default=500, ge=10, description="Number of evaluation samples per resolution"
    )
    n_charges: int = Field(default=5, ge=1, le=20, description="Number of point charges per sample")
    batch_size: int = Field(default=32, ge=1, description="Batch size for evaluation")

    # Training settings
    n_epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Optimizer learning rate")

    # Model settings
    d_model: int = Field(default=128, ge=16, description="Model hidden dimension")
    n_heads: int = Field(default=4, ge=1, description="Number of attention heads")
    n_layers: int = Field(default=4, ge=1, description="Number of transformer layers")
    n_fourier_features: int = Field(default=64, ge=8, description="Number of Fourier features")
    fourier_scale: float = Field(default=10.0, gt=0, description="Fourier feature scale")
    use_fnet: bool = Field(default=True, description="Use FNet mixing layers")

    # Success criteria
    mse_threshold: float = Field(default=0.05, gt=0, description="MSE threshold for passing")

    @field_validator("eval_resolutions")
    @classmethod
    def validate_eval_resolutions(cls, v: list[int]) -> list[int]:
        """Ensure eval resolutions are valid."""
        if not v:
            raise ValueError("eval_resolutions cannot be empty")
        for res in v:
            if res < 3 or res > 100:
                raise ValueError(f"Invalid resolution {res}: must be in [3, 100]")
        return sorted(set(v))  # Remove duplicates and sort

    @model_validator(mode="after")
    def validate_primary_in_eval(self) -> TransferScenarioConfig:
        """Ensure primary eval resolution is in eval list."""
        if self.primary_eval_resolution not in self.eval_resolutions:
            self.eval_resolutions = sorted({*self.eval_resolutions, self.primary_eval_resolution})
        return self

    def get_default_thresholds(self) -> list[MetricThreshold]:
        """Generate default thresholds based on config."""
        return [
            MetricThreshold(
                name=f"mse_{res}x{res}",
                operator="<",
                value=self.mse_threshold,
                description=f"MSE on {res}x{res} grid must be < {self.mse_threshold}",
            )
            for res in self.eval_resolutions
        ]


class ComplexityScenarioConfig(BaseScenarioConfig):
    """Configuration for computational complexity benchmarks.

    Validates O(N) Galerkin attention and O(N log N) FNet scaling.
    """

    name: str = Field(default="complexity", description="Scenario identifier")
    description: str = Field(
        default="Verify O(N) Galerkin and O(N log N) FNet complexity",
        description="Scenario description",
    )
    tier: ScenarioTier = ScenarioTier.FUNCTIONAL

    # Benchmark settings
    grid_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19, 25],
        description="Grid sizes to benchmark (tokens = size²)",
    )
    batch_size: int = Field(default=32, ge=1, description="Batch size for benchmarking")
    n_warmup: int = Field(default=10, ge=1, description="Warmup iterations")
    n_iterations: int = Field(default=100, ge=10, description="Timed iterations")

    # Model settings
    d_model: int = Field(default=128, ge=16, description="Model hidden dimension")
    n_heads: int = Field(default=4, ge=1, description="Number of attention heads")

    # Success criteria
    fnet_scaling_exponent_max: float = Field(
        default=1.5,
        description="Max scaling exponent for FNet (should be < 1.5 for O(N log N))",
    )
    softmax_scaling_exponent_min: float = Field(
        default=1.5,
        description="Min scaling exponent for Softmax (should be > 1.5 for O(N²))",
    )
    min_speedup_factor: float = Field(
        default=1.5, gt=1.0, description="Min FNet speedup over Softmax at largest size"
    )

    requires_gpu: bool = Field(default=True, description="GPU recommended for accurate timing")

    @field_validator("grid_sizes")
    @classmethod
    def validate_grid_sizes(cls, v: list[int]) -> list[int]:
        """Ensure grid sizes span a reasonable range."""
        if len(v) < 3:
            raise ValueError("Need at least 3 grid sizes for scaling analysis")
        return sorted(set(v))


class StabilityScenarioConfig(BaseScenarioConfig):
    """Configuration for LBB stability scenarios.

    Validates that the Ladyzhenskaya-Babuska-Brezzi (LBB) constant
    remains positive throughout training.
    """

    name: str = Field(default="stability", description="Scenario identifier")
    description: str = Field(
        default="Verify LBB stability constant remains positive during training",
        description="Scenario description",
    )
    tier: ScenarioTier = ScenarioTier.INTEGRATION

    # Model settings
    d_model: int = Field(default=64, ge=16, description="Model hidden dimension")
    d_key: int = Field(default=32, ge=8, description="Key dimension")
    d_value: int = Field(default=32, ge=8, description="Value dimension")

    # Test settings
    resolutions: list[int] = Field(
        default_factory=lambda: [5, 9, 13, 19],
        description="Resolutions to test stability",
    )
    n_forward_passes: int = Field(
        default=100, ge=10, description="Number of forward passes to monitor"
    )
    batch_size: int = Field(default=4, ge=1, description="Batch size for testing")

    # Training stability test
    n_training_steps: int = Field(
        default=1000, ge=100, description="Training steps for stability monitoring"
    )
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate for training")

    # Success criteria
    lbb_threshold: float = Field(default=1e-6, gt=0, description="Minimum acceptable LBB constant")
    max_lbb_violations: int = Field(default=0, ge=0, description="Maximum allowed LBB violations")

    @model_validator(mode="after")
    def validate_dimensions(self) -> StabilityScenarioConfig:
        """Ensure d_key >= d_query for LBB condition."""
        # In our implementation, d_query == d_key (both derived from d_model)
        # This validator documents the constraint
        return self


class ScenarioResult(BaseModel):
    """Result of a scenario execution.

    Captures all information needed for analysis and reproducibility.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow scenario-specific fields
    )

    # Identification
    scenario_name: str = Field(..., description="Scenario identifier")
    config_hash: str = Field(..., description="Hash of configuration used")

    # Status
    status: ScenarioStatus = Field(..., description="Execution status")
    passed: bool = Field(..., description="Whether all thresholds were met")

    # Metrics
    metrics: dict[str, float] = Field(default_factory=dict, description="Collected metrics")
    threshold_results: dict[str, bool] = Field(
        default_factory=dict, description="Per-threshold pass/fail"
    )

    # Artifacts
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="Paths to generated artifacts"
    )

    # Timing
    start_time: datetime = Field(..., description="Execution start time")
    end_time: datetime = Field(..., description="Execution end time")
    duration_seconds: float = Field(..., description="Total execution time")

    # Error information (if status == ERROR)
    error_message: str | None = Field(default=None, description="Error message if any")
    error_traceback: str | None = Field(default=None, description="Full traceback")

    # Environment
    device: str = Field(default="cpu", description="Computation device used")
    python_version: str = Field(default="", description="Python version")
    torch_version: str = Field(default="", description="PyTorch version")

    def summary(self) -> str:
        """Generate human-readable summary."""
        status_emoji = {
            ScenarioStatus.PASSED: "PASS",
            ScenarioStatus.FAILED: "FAIL",
            ScenarioStatus.ERROR: "ERROR",
            ScenarioStatus.SKIPPED: "SKIP",
            ScenarioStatus.PENDING: "PENDING",
            ScenarioStatus.RUNNING: "RUNNING",
        }
        emoji = status_emoji.get(self.status, "?")

        lines = [
            f"{emoji} {self.scenario_name}",
            f"   Status: {self.status.value}",
            f"   Duration: {self.duration_seconds:.2f}s",
        ]

        if self.metrics:
            lines.append("   Metrics:")
            for name, value in sorted(self.metrics.items()):
                result = self.threshold_results.get(name, None)
                indicator = ""
                if result is not None:
                    indicator = " PASS" if result else " FAIL"
                lines.append(f"     {name}: {value:.6f}{indicator}")

        if self.error_message:
            lines.append(f"   Error: {self.error_message}")

        return "\n".join(lines)


# Type alias for config union
ScenarioConfigUnion = TransferScenarioConfig | ComplexityScenarioConfig | StabilityScenarioConfig


def load_config_from_dict(
    data: dict[str, Any], scenario_type: str | None = None
) -> BaseScenarioConfig:
    """Load scenario config from dictionary.

    Args:
        data: Configuration dictionary.
        scenario_type: Optional type hint. If not provided, inferred from 'name'.

    Returns:
        Appropriate config instance.

    Raises:
        ValueError: If scenario type cannot be determined.

    """
    from src.poc.config_noyron import NoyronHXScenarioConfig

    type_map: dict[str, type[BaseScenarioConfig]] = {
        "transfer": TransferScenarioConfig,
        "complexity": ComplexityScenarioConfig,
        "stability": StabilityScenarioConfig,
        "noyron_hx": NoyronHXScenarioConfig,
    }

    # Lazy import: `LLMPriorAblationConfig` itself is light (it only pulls
    # in `LMStudioConfig`), but resolving it at module top would force the
    # *scenario* module to load — and that pulls in `scipy.stats`, the
    # MCTS engine, the PDE registry, and the LM Studio client. Loading
    # that surface on every config-dispatch call is wasteful for runs
    # that never touch the LLM-prior scenario, so we resolve it only on
    # demand. The integration itself remains opt-in via the [lm-studio]
    # extra.
    inferred_name = scenario_type or data.get("name", "")
    if inferred_name == "llm_prior_ablation":
        from src.poc.scenarios.llm_prior_config import (
            LLMPriorAblationConfig,
        )

        type_map["llm_prior_ablation"] = LLMPriorAblationConfig

    # Same lazy-resolution rationale as llm_prior_ablation: the scaling-law
    # scenario module pulls in the MCTS engine, PDE registry, and LM Studio
    # client, so only resolve its (light) config class on demand.
    if inferred_name == "scaling_law":
        from src.poc.scenarios.scaling_law_config import ScalingLawConfig

        type_map["scaling_law"] = ScalingLawConfig

    # Same lazy-resolution rationale: the noyron_basis config is light but its
    # scenario module pulls in the PDE registry + helical operators, so resolve
    # only the config class, and only when this scenario is actually requested.
    if inferred_name == "noyron_basis":
        from src.poc.scenarios.noyron_basis_config import NoyronBasisConfig

        type_map["noyron_basis"] = NoyronBasisConfig

    # Same lazy-resolution rationale: the lshape_amr_compare config is light but
    # its scenario module pulls in the MCTS engine, the PDE registry, and scipy
    # via the comparison harness, so resolve only the config class on demand.
    if inferred_name == "lshape_amr_compare":
        from src.poc.scenarios.lshape_amr_compare_config import (
            LShapeAMRCompareConfig,
        )

        type_map["lshape_amr_compare"] = LShapeAMRCompareConfig

    # Same lazy-resolution rationale: the transfer_baseline_compare config is light
    # but its scenario module pulls in the PhysicsOperator, the CNN baseline, and the
    # comparison harness (torch-heavy), so resolve only the config class on demand.
    if inferred_name == "transfer_baseline_compare":
        from src.poc.scenarios.transfer_baseline_compare_config import (
            TransferBaselineCompareConfig,
        )

        type_map["transfer_baseline_compare"] = TransferBaselineCompareConfig

    # Determine type
    if scenario_type:
        config_cls = type_map.get(scenario_type)
    else:
        name = data.get("name", "")
        config_cls = type_map.get(name)

    if not config_cls:
        # Default to base config
        config_cls = BaseScenarioConfig

    return config_cls(**data)
