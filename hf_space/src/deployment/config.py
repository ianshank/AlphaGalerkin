"""Configuration schemas for model deployment.

This module defines Pydantic models for ONNX export and deployment
configuration, ensuring type safety and validation.

Design Principles:
    - No hardcoded values: All constants are configurable with sensible defaults
    - Backwards compatible: New fields have defaults
    - Validated: Pydantic enforces types and constraints at runtime
    - Serializable: All configs can be saved/loaded from YAML/JSON
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuantizationMode(str, Enum):
    """Quantization mode options."""

    DYNAMIC = "dynamic"  # Dynamic quantization (weights only)
    STATIC = "static"  # Static quantization (weights + activations)
    QAT = "qat"  # Quantization-aware training


class ExecutionProvider(str, Enum):
    """ONNX Runtime execution providers."""

    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    COREML = "CoreMLExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    DIRECTML = "DmlExecutionProvider"


class ExportConfig(BaseModel):
    """Configuration for ONNX export.

    Attributes:
        opset_version: ONNX opset version to use.
        input_names: Names for input tensors.
        output_names: Names for output tensors.
        dynamic_axes: Dynamic axes for variable-size inputs.
        do_constant_folding: Apply constant folding optimization.
        export_params: Export model parameters.
        verbose: Enable verbose logging during export.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # ONNX version
    opset_version: int = Field(
        default=17,
        ge=9,
        le=20,
        description="ONNX opset version (9-20)",
    )

    # Input/output specification
    input_names: list[str] = Field(
        default_factory=lambda: ["board_state"],
        description="Names for model inputs",
    )
    output_names: list[str] = Field(
        default_factory=lambda: ["policy", "value"],
        description="Names for model outputs",
    )

    # Dynamic axes for variable batch/resolution
    dynamic_axes: dict[str, dict[int, str]] = Field(
        default_factory=lambda: {
            "board_state": {0: "batch", 2: "height", 3: "width"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        description="Dynamic axes mapping for variable-size inputs",
    )

    # Export options
    do_constant_folding: bool = Field(
        default=True,
        description="Apply constant folding optimization",
    )
    export_params: bool = Field(
        default=True,
        description="Include model parameters in export",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose export logging",
    )

    # Tracing vs scripting
    export_method: Literal["trace", "script", "dynamo"] = Field(
        default="trace",
        description="Export method: trace, script, or dynamo",
    )

    # Optimization level
    optimization_level: Literal["none", "basic", "extended", "full"] = Field(
        default="full",
        description="Graph optimization level",
    )

    # Model metadata
    model_name: str = Field(
        default="alphagalerkin",
        description="Model name for ONNX metadata",
    )
    model_version: str = Field(
        default="1.0.0",
        description="Model version string",
    )

    @field_validator("input_names", "output_names")
    @classmethod
    def validate_names(cls, v: list[str]) -> list[str]:
        """Ensure names are valid identifiers."""
        for name in v:
            if not name.isidentifier():
                raise ValueError(f"Invalid name '{name}': must be valid Python identifier")
        return v


class QuantizationConfig(BaseModel):
    """Configuration for model quantization.

    Attributes:
        enabled: Whether to apply quantization.
        mode: Quantization mode (dynamic, static, qat).
        weight_type: Data type for quantized weights.
        activation_type: Data type for activations (static only).
        per_channel: Use per-channel quantization.
        reduce_range: Reduce range for int8 (7-bit effective).
        calibration_samples: Number of samples for static calibration.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Enable/disable
    enabled: bool = Field(
        default=True,
        description="Enable model quantization",
    )

    # Quantization mode
    mode: QuantizationMode = Field(
        default=QuantizationMode.DYNAMIC,
        description="Quantization mode",
    )

    # Data types
    weight_type: Literal["int8", "uint8", "int4"] = Field(
        default="int8",
        description="Quantized weight data type",
    )
    activation_type: Literal["int8", "uint8"] = Field(
        default="int8",
        description="Quantized activation data type (static mode)",
    )

    # Quantization options
    per_channel: bool = Field(
        default=True,
        description="Use per-channel weight quantization",
    )
    reduce_range: bool = Field(
        default=False,
        description="Reduce quantization range (7-bit effective)",
    )
    symmetric_weight: bool = Field(
        default=True,
        description="Use symmetric quantization for weights",
    )

    # Calibration (for static quantization)
    calibration_samples: int = Field(
        default=100,
        ge=10,
        description="Number of samples for calibration",
    )
    calibration_method: Literal["minmax", "entropy", "percentile"] = Field(
        default="entropy",
        description="Calibration method for static quantization",
    )

    # Model input configuration
    input_name: str = Field(
        default="board_state",
        description="Model input tensor name for calibration data",
    )

    # Operator selection
    operators_to_quantize: list[str] | None = Field(
        default=None,
        description="Specific operators to quantize (None = all supported)",
    )
    nodes_to_exclude: list[str] = Field(
        default_factory=list,
        description="Node names to exclude from quantization",
    )


class RuntimeConfig(BaseModel):
    """Configuration for ONNX Runtime inference.

    Attributes:
        execution_providers: Ordered list of execution providers.
        intra_op_threads: Threads for intra-operator parallelism.
        inter_op_threads: Threads for inter-operator parallelism.
        graph_optimization_level: Runtime graph optimization level.
        enable_profiling: Enable runtime profiling.
        memory_pattern: Enable memory pattern optimization.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Execution providers (in priority order)
    execution_providers: list[ExecutionProvider] = Field(
        default_factory=lambda: [
            ExecutionProvider.CUDA,
            ExecutionProvider.CPU,
        ],
        description="Execution providers in priority order",
    )

    # Threading
    intra_op_threads: int = Field(
        default=0,
        ge=0,
        description="Threads for intra-op parallelism (0 = auto)",
    )
    inter_op_threads: int = Field(
        default=0,
        ge=0,
        description="Threads for inter-op parallelism (0 = auto)",
    )

    # Optimization
    graph_optimization_level: Literal["disable", "basic", "extended", "all"] = Field(
        default="all",
        description="Graph optimization level",
    )

    # Memory
    enable_mem_pattern: bool = Field(
        default=True,
        description="Enable memory pattern optimization",
    )
    enable_cpu_mem_arena: bool = Field(
        default=True,
        description="Enable CPU memory arena",
    )

    # Profiling
    enable_profiling: bool = Field(
        default=False,
        description="Enable runtime profiling",
    )
    profile_output_path: str | None = Field(
        default=None,
        description="Path for profiling output",
    )

    # Session options
    log_severity_level: int = Field(
        default=2,
        ge=0,
        le=4,
        description="Logging severity level (0=verbose, 4=fatal)",
    )

    # CUDA-specific options
    cuda_device_id: int = Field(
        default=0,
        ge=0,
        description="CUDA device ID to use",
    )
    cuda_mem_limit: int = Field(
        default=0,
        ge=0,
        description="CUDA memory limit in bytes (0 = unlimited)",
    )
    cuda_arena_extend_strategy: Literal["kNextPowerOfTwo", "kSameAsRequested"] = Field(
        default="kNextPowerOfTwo",
        description="CUDA arena extend strategy",
    )


class DeploymentConfig(BaseModel):
    """Combined configuration for model deployment.

    Attributes:
        export: Export configuration.
        quantization: Quantization configuration.
        runtime: Runtime configuration.
        output_dir: Directory for deployment artifacts.
        validate_export: Validate exported model against PyTorch.
        validation_tolerance: Tolerance for validation comparison.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Sub-configurations
    export: ExportConfig = Field(default_factory=ExportConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # Output
    output_dir: str = Field(
        default="outputs/deployment",
        description="Directory for deployment artifacts",
    )

    # Validation
    validate_export: bool = Field(
        default=True,
        description="Validate exported model against PyTorch",
    )
    validation_tolerance: float = Field(
        default=1e-5,
        gt=0,
        description="Tolerance for output comparison",
    )
    validation_samples: int = Field(
        default=10,
        ge=1,
        description="Number of samples for validation",
    )

    # Target platforms
    target_platforms: list[str] = Field(
        default_factory=lambda: ["cpu", "cuda"],
        description="Target deployment platforms",
    )


def create_export_config(**kwargs: Any) -> ExportConfig:
    """Factory function for export configuration.

    Args:
        **kwargs: Configuration options.

    Returns:
        Configured ExportConfig instance.

    """
    return ExportConfig(**kwargs)


def create_quantization_config(
    mode: str = "dynamic",
    **kwargs: Any,
) -> QuantizationConfig:
    """Factory function for quantization configuration.

    Args:
        mode: Quantization mode.
        **kwargs: Additional options.

    Returns:
        Configured QuantizationConfig instance.

    """
    return QuantizationConfig(mode=QuantizationMode(mode), **kwargs)
