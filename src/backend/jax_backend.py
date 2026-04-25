"""JAX backend implementation for AlphaGalerkin.

Implements the BackendInterface protocol using JAX and jax.numpy,
providing GPU/TPU-accelerated tensor operations with explicit PRNG
key management, JIT compilation, and functional differentiation.

JAX-specific design notes:
- PRNG keys are managed internally; each random call splits the key.
- ``grad`` and ``value_and_grad`` delegate directly to ``jax.grad``
  and ``jax.value_and_grad``.
- Device placement uses ``jax.devices()`` for auto-detection and
  ``jax.device_put`` for explicit placement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import structlog

from src.backend.config import BackendConfig
from src.backend.types import Array, BackendType, DTypeLike, Precision, Shape, ShapeLike

__all__ = ["JaxBackend"]

logger = structlog.get_logger(__name__)


class JaxBackend:
    """JAX implementation of the AlphaGalerkin BackendInterface.

    Manages JAX configuration (debug NaNs, log compiles, platform),
    explicit PRNG key state, and dtype mapping.  All tensor operations
    are thin wrappers around ``jax.numpy``.

    Args:
        config: Backend configuration with JAX-specific settings.

    """

    def __init__(self, config: BackendConfig) -> None:
        self._config = config

        # Configure JAX global settings before any computation.
        if config.jax_platform is not None:
            jax.config.update("jax_platform_name", config.jax_platform)

        jax.config.update("jax_debug_nans", config.jax_debug_nans)
        jax.config.update("jax_log_compiles", config.jax_log_compiles)

        if not config.jax_jit_enabled:
            jax.config.update("jax_disable_jit", True)

        # Resolve the default dtype from the Precision enum.
        self._default_dtype: jnp.dtype = self.get_dtype(config.precision)

        # Initialize the PRNG key from the configured seed.
        self._rng_key: jax.Array = jax.random.PRNGKey(config.rng_seed)

        logger.info(
            "backend_initialized",
            device=self.get_default_device(),
            precision=config.precision.value,
            seed=config.rng_seed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dtype(self, dtype: DTypeLike | None) -> jnp.dtype:
        """Return *dtype* if given, otherwise the configured default."""
        return dtype if dtype is not None else self._default_dtype

    def _next_key(self) -> jax.Array:
        """Split the internal PRNG key and return a fresh sub-key."""
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> BackendType:
        """Return the backend type identifier."""
        return BackendType.JAX

    @property
    def default_dtype(self) -> DTypeLike:
        """Return the default floating-point dtype."""
        return self._default_dtype

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    def zeros(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor of zeros."""
        return jnp.zeros(shape, dtype=self._resolve_dtype(dtype))

    def ones(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor of ones."""
        return jnp.ones(shape, dtype=self._resolve_dtype(dtype))

    def full(
        self,
        shape: ShapeLike,
        fill_value: float,
        dtype: DTypeLike | None = None,
    ) -> Array:
        """Create a tensor filled with a constant value."""
        return jnp.full(shape, fill_value, dtype=self._resolve_dtype(dtype))

    def randn(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor with standard normal random values.

        Splits the internal PRNG key on every call.
        """
        key = self._next_key()
        return jax.random.normal(key, shape, dtype=self._resolve_dtype(dtype))

    def rand(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor with uniform random values in [0, 1)."""
        key = self._next_key()
        return jax.random.uniform(key, shape, dtype=self._resolve_dtype(dtype))

    def arange(
        self,
        start: float,
        stop: float | None = None,
        step: float = 1.0,
        dtype: DTypeLike | None = None,
    ) -> Array:
        """Create a 1-D tensor with evenly spaced values."""
        if stop is None:
            return jnp.arange(start, dtype=self._resolve_dtype(dtype))
        return jnp.arange(start, stop, step, dtype=self._resolve_dtype(dtype))

    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: DTypeLike | None = None,
    ) -> Array:
        """Create a 1-D tensor with linearly spaced values."""
        return jnp.linspace(start, stop, num, dtype=self._resolve_dtype(dtype))

    def from_numpy(self, array: Any) -> Array:
        """Convert a numpy array to a JAX array."""
        return jnp.asarray(array)

    def to_numpy(self, array: Array) -> Any:
        """Convert a JAX array to a numpy array."""
        return np.asarray(jax.device_get(array))

    def tensor(self, data: Any, dtype: DTypeLike | None = None) -> Array:
        """Create a JAX array from data (list, tuple, numpy array, etc.)."""
        return jnp.asarray(data, dtype=self._resolve_dtype(dtype))

    # ------------------------------------------------------------------
    # Tensor properties
    # ------------------------------------------------------------------

    def shape(self, x: Array) -> Shape:
        """Return the shape of a tensor."""
        return tuple(x.shape)

    def dtype(self, x: Array) -> DTypeLike:
        """Return the dtype of a tensor."""
        return x.dtype

    def numel(self, x: Array) -> int:
        """Return the total number of elements."""
        return int(x.size)

    # ------------------------------------------------------------------
    # Tensor manipulation
    # ------------------------------------------------------------------

    def reshape(self, x: Array, shape: ShapeLike) -> Array:
        """Reshape a tensor."""
        return jnp.reshape(x, shape)

    def transpose(self, x: Array, axes: tuple[int, ...] | None = None) -> Array:
        """Transpose a tensor."""
        return jnp.transpose(x, axes=axes)

    def expand_dims(self, x: Array, axis: int) -> Array:
        """Add a dimension at the specified axis."""
        return jnp.expand_dims(x, axis=axis)

    def squeeze(self, x: Array, axis: int | None = None) -> Array:
        """Remove dimensions of size 1."""
        return jnp.squeeze(x, axis=axis)

    def cat(self, arrays: list[Array], axis: int = 0) -> Array:
        """Concatenate tensors along an axis."""
        return jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        """Stack tensors along a new axis."""
        return jnp.stack(arrays, axis=axis)

    def split(
        self,
        x: Array,
        num_or_sections: int | list[int],
        axis: int = 0,
    ) -> list[Array]:
        """Split a tensor along an axis."""
        parts = jnp.split(x, num_or_sections, axis=axis)
        return list(parts)

    def pad(
        self,
        x: Array,
        pad_width: list[tuple[int, int]],
        value: float = 0.0,
    ) -> Array:
        """Pad a tensor with a constant value.

        Args:
            x: Input tensor.
            pad_width: List of (before, after) pad widths per dimension.
            value: Padding value.

        """
        return jnp.pad(x, pad_width, mode="constant", constant_values=value)

    # ------------------------------------------------------------------
    # Math operations
    # ------------------------------------------------------------------

    def add(self, x: Array, y: Array) -> Array:
        """Element-wise addition."""
        return jnp.add(x, y)

    def mul(self, x: Array, y: Array) -> Array:
        """Element-wise multiplication."""
        return jnp.multiply(x, y)

    def matmul(self, x: Array, y: Array) -> Array:
        """Matrix multiplication."""
        return jnp.matmul(x, y)

    def einsum(self, subscripts: str, *operands: Array) -> Array:
        """Einstein summation."""
        return jnp.einsum(subscripts, *operands)

    def sum(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Sum along axis."""
        return jnp.sum(x, axis=axis, keepdims=keepdims)

    def mean(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Mean along axis."""
        return jnp.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x: Array, axis: int | None = None) -> Array:
        """Maximum along axis."""
        return jnp.max(x, axis=axis)

    def min(self, x: Array, axis: int | None = None) -> Array:
        """Minimum along axis."""
        return jnp.min(x, axis=axis)

    def abs(self, x: Array) -> Array:
        """Absolute value."""
        return jnp.abs(x)

    def sqrt(self, x: Array) -> Array:
        """Element-wise square root."""
        return jnp.sqrt(x)

    def exp(self, x: Array) -> Array:
        """Element-wise exponential."""
        return jnp.exp(x)

    def log(self, x: Array) -> Array:
        """Element-wise natural logarithm."""
        return jnp.log(x)

    def cos(self, x: Array) -> Array:
        """Element-wise cosine."""
        return jnp.cos(x)

    def sin(self, x: Array) -> Array:
        """Element-wise sine."""
        return jnp.sin(x)

    def clamp(
        self,
        x: Array,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> Array:
        """Clamp values to a range.

        Uses positional arguments so this works on both legacy JAX
        (``a_min``/``a_max`` kwargs) and JAX 0.5+ (``min``/``max``
        kwargs).  ``None`` on either bound is the documented sentinel
        for "no clipping on that side" in both versions.
        """
        return jnp.clip(x, min_val, max_val)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        """Element-wise conditional selection."""
        return jnp.where(condition, x, y)

    def pow(self, x: Array, exponent: float | Array) -> Array:
        """Element-wise power."""
        return jnp.power(x, exponent)

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def softmax(self, x: Array, axis: int = -1) -> Array:
        """Softmax activation."""
        return jnn.softmax(x, axis=axis)

    def log_softmax(self, x: Array, axis: int = -1) -> Array:
        """Log-softmax activation."""
        return jnn.log_softmax(x, axis=axis)

    def relu(self, x: Array) -> Array:
        """ReLU activation."""
        return jnn.relu(x)

    def gelu(self, x: Array) -> Array:
        """GELU activation."""
        return jnn.gelu(x)

    def sigmoid(self, x: Array) -> Array:
        """Sigmoid activation."""
        return jnn.sigmoid(x)

    def tanh(self, x: Array) -> Array:
        """Hyperbolic tangent."""
        return jnp.tanh(x)

    # ------------------------------------------------------------------
    # FFT operations
    # ------------------------------------------------------------------

    def fft2(self, x: Array) -> Array:
        """2D FFT (complex-to-complex)."""
        return jnp.fft.fft2(x)

    def ifft2(self, x: Array, s: tuple[int, int] | None = None) -> Array:
        """2D inverse FFT."""
        return jnp.fft.ifft2(x, s=s)

    def rfft2(self, x: Array) -> Array:
        """2D real-to-complex FFT."""
        return jnp.fft.rfft2(x)

    def irfft2(self, x: Array, s: tuple[int, int] | None = None) -> Array:
        """2D complex-to-real inverse FFT."""
        return jnp.fft.irfft2(x, s=s)

    def fftfreq(self, n: int, d: float = 1.0) -> Array:
        """DFT sample frequencies."""
        return jnp.fft.fftfreq(n, d=d)

    def rfftfreq(self, n: int, d: float = 1.0) -> Array:
        """DFT sample frequencies for rfft."""
        return jnp.fft.rfftfreq(n, d=d)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def svdvals(self, x: Array) -> Array:
        """Compute singular values of a matrix."""
        return jnp.linalg.svd(x, compute_uv=False)

    def norm(
        self,
        x: Array,
        ord: int | float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
    ) -> Array:
        """Compute tensor norm."""
        return jnp.linalg.norm(x, ord=ord, axis=axis)

    # ------------------------------------------------------------------
    # Autograd / differentiation
    # ------------------------------------------------------------------

    def grad(
        self,
        fn: Callable[..., Array],
        argnums: int | tuple[int, ...] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        """Return a function that computes gradients.

        Wraps ``jax.grad`` with the specified argument indices and
        auxiliary-output flag.
        """
        return jax.grad(fn, argnums=argnums, has_aux=has_aux)

    def value_and_grad(
        self,
        fn: Callable[..., Array],
        argnums: int | tuple[int, ...] = 0,
        has_aux: bool = False,
    ) -> Callable[..., tuple[Any, ...]]:
        """Return a function that computes both value and gradients."""
        return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to_device(self, x: Array, device: str) -> Array:
        """Move tensor to a device.

        Uses ``jax.device_put`` to place the array on the named device.
        """
        target_devices = jax.devices(device)
        if not target_devices:
            raise RuntimeError(f"No JAX devices found for backend '{device}'")
        return jax.device_put(x, target_devices[0])

    def get_default_device(self) -> str:
        """Get the default device string.

        Inspects ``jax.devices()`` and returns 'tpu', 'gpu', or 'cpu'
        depending on which accelerator is available.
        """
        devices = jax.devices()
        if not devices:
            return "cpu"

        platform = devices[0].platform
        if platform == "tpu":
            return "tpu"
        if platform == "gpu":
            return "gpu"
        return "cpu"

    # ------------------------------------------------------------------
    # Dtype management
    # ------------------------------------------------------------------

    def get_dtype(self, precision: Precision) -> DTypeLike:
        """Convert a Precision enum to the corresponding ``jnp`` dtype."""
        mapping: dict[Precision, jnp.dtype] = {
            Precision.FLOAT16: jnp.float16,
            Precision.BFLOAT16: jnp.bfloat16,
            Precision.FLOAT32: jnp.float32,
            Precision.FLOAT64: jnp.float64,
        }
        dtype = mapping.get(precision)
        if dtype is None:
            msg = f"Unsupported precision: {precision}"
            raise ValueError(msg)
        return dtype

    def cast(self, x: Array, dtype: DTypeLike) -> Array:
        """Cast a tensor to a different dtype."""
        return x.astype(dtype)

    # ------------------------------------------------------------------
    # Random state
    # ------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """Set the random seed by re-initializing the PRNG key."""
        self._rng_key = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Mesh grid
    # ------------------------------------------------------------------

    def meshgrid(self, *arrays: Array, indexing: str = "ij") -> tuple[Array, ...]:
        """Create coordinate matrices from coordinate vectors."""
        return tuple(jnp.meshgrid(*arrays, indexing=indexing))

    # ------------------------------------------------------------------
    # Comparison / boolean helpers
    # ------------------------------------------------------------------

    def ones_like(self, x: Array) -> Array:
        """Create a tensor of ones with the same shape and dtype."""
        return jnp.ones_like(x)

    def zeros_like(self, x: Array) -> Array:
        """Create a tensor of zeros with the same shape and dtype."""
        return jnp.zeros_like(x)

    def float_scalar(self, x: Array) -> float:
        """Extract a scalar float from a 0-d tensor."""
        return float(x)
