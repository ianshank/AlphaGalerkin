"""Backend configuration using Pydantic for validation.

All backend-specific settings (device, precision, JIT flags, etc.)
flow through this config — no hardcoded values.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from src.backend.types import BackendType, DeviceType, Precision
from src.templates.config import BaseModuleConfig

__all__ = ["BackendConfig"]


class BackendConfig(BaseModuleConfig):
    """Configuration for the compute backend.

    Controls which framework (PyTorch or JAX) is used, along with
    device selection, precision, and framework-specific options.

    Attributes:
        backend: Which framework to use.
        device: Device selection ("auto" detects GPU/TPU availability).
        precision: Default floating-point precision.
        rng_seed: Random seed for reproducibility.
        jax_jit_enabled: Whether to enable JAX JIT compilation.
        jax_debug_nans: Whether to check for NaNs in JAX computations.
        jax_log_compiles: Whether to log JAX JIT compilation events.
        jax_platform: JAX platform override (None = auto-detect).

    """

    name: str = Field(default="backend", description="Configuration name")

    # Core settings
    backend: BackendType = Field(
        default=BackendType.TORCH,
        description="Compute backend framework",
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Device selection",
    )
    precision: Precision = Field(
        default=Precision.FLOAT32,
        description="Default floating-point precision",
    )
    rng_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )

    # JAX-specific settings
    jax_jit_enabled: bool = Field(
        default=True,
        description="Enable JAX JIT compilation",
    )
    jax_debug_nans: bool = Field(
        default=False,
        description="Enable NaN checking in JAX (slower but safer for debugging)",
    )
    jax_log_compiles: bool = Field(
        default=False,
        description="Log JAX JIT compilation events",
    )
    jax_platform: str | None = Field(
        default=None,
        description="JAX platform override (cpu, gpu, tpu). None = auto-detect",
    )

    # Torch-specific settings
    torch_cudnn_benchmark: bool = Field(
        default=True,
        description="Enable cuDNN benchmark mode for torch",
    )
    torch_deterministic: bool = Field(
        default=False,
        description="Enable deterministic mode for torch (slower)",
    )

    @model_validator(mode="after")
    def validate_jax_platform(self) -> BackendConfig:
        """Validate JAX platform if specified."""
        valid_platforms = {"cpu", "gpu", "tpu", None}
        if self.jax_platform not in valid_platforms:
            msg = f"jax_platform must be one of {valid_platforms}, got '{self.jax_platform}'"
            raise ValueError(msg)
        return self
