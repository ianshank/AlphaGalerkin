"""Backend-agnostic type aliases for AlphaGalerkin.

Provides unified type system that works with both PyTorch and JAX tensors.
These types are used throughout the backend abstraction layer to avoid
hard-coding tensor types.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

__all__ = [
    "Array",
    "Shape",
    "ShapeLike",
    "DTypeLike",
    "Precision",
    "DeviceType",
    "BackendType",
]

# Array is the universal tensor type - can be torch.Tensor or jax.Array
# We use Any to avoid importing either framework at module load time
Array = Any

# Shape types
Shape = tuple[int, ...]
ShapeLike = Sequence[int] | Shape

# DType mapping - unified precision specification
DTypeLike = Any  # torch.dtype or jnp.dtype


class Precision(str, Enum):
    """Supported precision levels for computation."""

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class DeviceType(str, Enum):
    """Supported device types."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    AUTO = "auto"


class BackendType(str, Enum):
    """Supported backend frameworks."""

    TORCH = "torch"
    JAX = "jax"
