"""Backend interface protocol defining all operations needed by AlphaGalerkin.

This protocol defines the contract that both PyTorch and JAX backends must
implement. Using Protocol (structural subtyping) instead of ABC allows
backends to satisfy the interface without explicit inheritance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from src.backend.types import Array, BackendType, DTypeLike, Precision, Shape, ShapeLike


@runtime_checkable
class BackendInterface(Protocol):
    """Protocol defining the operations required from a compute backend.

    Both TorchBackend and JaxBackend must implement all these methods.
    The interface covers tensor creation, math ops, FFT, linear algebra,
    autograd, and device management.
    """

    @property
    def name(self) -> BackendType:
        """Return the backend type identifier."""
        ...

    @property
    def default_dtype(self) -> DTypeLike:
        """Return the default floating-point dtype."""
        ...

    # ----------------------------------------------------------------
    # Tensor creation
    # ----------------------------------------------------------------

    def zeros(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor of zeros."""
        ...

    def ones(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor of ones."""
        ...

    def full(self, shape: ShapeLike, fill_value: float, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor filled with a constant value."""
        ...

    def randn(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor with standard normal random values.

        For JAX, uses the internal PRNG key manager.
        For PyTorch, uses the global generator (seeded via set_seed).
        """
        ...

    def rand(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor with uniform random values in [0, 1)."""
        ...

    def arange(
        self, start: float, stop: float | None = None, step: float = 1.0,
        dtype: DTypeLike | None = None,
    ) -> Array:
        """Create a 1-D tensor with evenly spaced values."""
        ...

    def linspace(
        self, start: float, stop: float, num: int, dtype: DTypeLike | None = None,
    ) -> Array:
        """Create a 1-D tensor with linearly spaced values."""
        ...

    def from_numpy(self, array: Any) -> Array:
        """Convert a numpy array to a backend tensor."""
        ...

    def to_numpy(self, array: Array) -> Any:
        """Convert a backend tensor to a numpy array."""
        ...

    def tensor(self, data: Any, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor from data (list, tuple, numpy array, etc.)."""
        ...

    # ----------------------------------------------------------------
    # Tensor properties
    # ----------------------------------------------------------------

    def shape(self, x: Array) -> Shape:
        """Return the shape of a tensor."""
        ...

    def dtype(self, x: Array) -> DTypeLike:
        """Return the dtype of a tensor."""
        ...

    def numel(self, x: Array) -> int:
        """Return the total number of elements."""
        ...

    # ----------------------------------------------------------------
    # Tensor manipulation
    # ----------------------------------------------------------------

    def reshape(self, x: Array, shape: ShapeLike) -> Array:
        """Reshape a tensor."""
        ...

    def transpose(self, x: Array, axes: tuple[int, ...] | None = None) -> Array:
        """Transpose a tensor."""
        ...

    def expand_dims(self, x: Array, axis: int) -> Array:
        """Add a dimension at the specified axis."""
        ...

    def squeeze(self, x: Array, axis: int | None = None) -> Array:
        """Remove dimensions of size 1."""
        ...

    def cat(self, arrays: list[Array], axis: int = 0) -> Array:
        """Concatenate tensors along an axis."""
        ...

    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        """Stack tensors along a new axis."""
        ...

    def split(self, x: Array, num_or_sections: int | list[int], axis: int = 0) -> list[Array]:
        """Split a tensor along an axis."""
        ...

    def pad(
        self, x: Array, pad_width: list[tuple[int, int]], value: float = 0.0,
    ) -> Array:
        """Pad a tensor with a constant value.

        Args:
            x: Input tensor.
            pad_width: List of (before, after) pad widths per dimension.
            value: Padding value.
        """
        ...

    # ----------------------------------------------------------------
    # Math operations
    # ----------------------------------------------------------------

    def add(self, x: Array, y: Array) -> Array:
        """Element-wise addition."""
        ...

    def mul(self, x: Array, y: Array) -> Array:
        """Element-wise multiplication."""
        ...

    def matmul(self, x: Array, y: Array) -> Array:
        """Matrix multiplication."""
        ...

    def einsum(self, subscripts: str, *operands: Array) -> Array:
        """Einstein summation."""
        ...

    def sum(self, x: Array, axis: int | tuple[int, ...] | None = None,
            keepdims: bool = False) -> Array:
        """Sum along axis."""
        ...

    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None,
             keepdims: bool = False) -> Array:
        """Mean along axis."""
        ...

    def max(self, x: Array, axis: int | None = None) -> Array:
        """Maximum along axis."""
        ...

    def min(self, x: Array, axis: int | None = None) -> Array:
        """Minimum along axis."""
        ...

    def abs(self, x: Array) -> Array:
        """Absolute value."""
        ...

    def sqrt(self, x: Array) -> Array:
        """Element-wise square root."""
        ...

    def exp(self, x: Array) -> Array:
        """Element-wise exponential."""
        ...

    def log(self, x: Array) -> Array:
        """Element-wise natural logarithm."""
        ...

    def cos(self, x: Array) -> Array:
        """Element-wise cosine."""
        ...

    def sin(self, x: Array) -> Array:
        """Element-wise sine."""
        ...

    def clamp(self, x: Array, min_val: float | None = None,
              max_val: float | None = None) -> Array:
        """Clamp values to a range."""
        ...

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        """Element-wise conditional selection."""
        ...

    def pow(self, x: Array, exponent: float | Array) -> Array:
        """Element-wise power."""
        ...

    # ----------------------------------------------------------------
    # Activation functions
    # ----------------------------------------------------------------

    def softmax(self, x: Array, axis: int = -1) -> Array:
        """Softmax activation."""
        ...

    def log_softmax(self, x: Array, axis: int = -1) -> Array:
        """Log-softmax activation."""
        ...

    def relu(self, x: Array) -> Array:
        """ReLU activation."""
        ...

    def gelu(self, x: Array) -> Array:
        """GELU activation."""
        ...

    def sigmoid(self, x: Array) -> Array:
        """Sigmoid activation."""
        ...

    def tanh(self, x: Array) -> Array:
        """Hyperbolic tangent."""
        ...

    # ----------------------------------------------------------------
    # FFT operations
    # ----------------------------------------------------------------

    def fft2(self, x: Array) -> Array:
        """2D FFT (complex-to-complex)."""
        ...

    def ifft2(self, x: Array, s: tuple[int, int] | None = None) -> Array:
        """2D inverse FFT."""
        ...

    def rfft2(self, x: Array) -> Array:
        """2D real-to-complex FFT."""
        ...

    def irfft2(self, x: Array, s: tuple[int, int] | None = None) -> Array:
        """2D complex-to-real inverse FFT."""
        ...

    def fftfreq(self, n: int, d: float = 1.0) -> Array:
        """DFT sample frequencies."""
        ...

    def rfftfreq(self, n: int, d: float = 1.0) -> Array:
        """DFT sample frequencies for rfft."""
        ...

    # ----------------------------------------------------------------
    # Linear algebra
    # ----------------------------------------------------------------

    def svdvals(self, x: Array) -> Array:
        """Compute singular values of a matrix."""
        ...

    def norm(self, x: Array, ord: int | float | str | None = None,
             axis: int | tuple[int, ...] | None = None) -> Array:
        """Compute tensor norm."""
        ...

    # ----------------------------------------------------------------
    # Autograd / differentiation
    # ----------------------------------------------------------------

    def grad(
        self, fn: Callable[..., Array], argnums: int | tuple[int, ...] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        """Return a function that computes gradients.

        For JAX: wraps jax.grad.
        For PyTorch: wraps a custom autograd-based gradient function.
        """
        ...

    def value_and_grad(
        self, fn: Callable[..., Array], argnums: int | tuple[int, ...] = 0,
        has_aux: bool = False,
    ) -> Callable[..., tuple[Any, ...]]:
        """Return a function that computes both value and gradients."""
        ...

    # ----------------------------------------------------------------
    # Device management
    # ----------------------------------------------------------------

    def to_device(self, x: Array, device: str) -> Array:
        """Move tensor to a device."""
        ...

    def get_default_device(self) -> str:
        """Get the default device string."""
        ...

    # ----------------------------------------------------------------
    # Dtype management
    # ----------------------------------------------------------------

    def get_dtype(self, precision: Precision) -> DTypeLike:
        """Convert a Precision enum to a framework-specific dtype."""
        ...

    def cast(self, x: Array, dtype: DTypeLike) -> Array:
        """Cast a tensor to a different dtype."""
        ...

    # ----------------------------------------------------------------
    # Random state
    # ----------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        ...

    # ----------------------------------------------------------------
    # Mesh grid
    # ----------------------------------------------------------------

    def meshgrid(self, *arrays: Array, indexing: str = "ij") -> tuple[Array, ...]:
        """Create coordinate matrices from coordinate vectors."""
        ...

    # ----------------------------------------------------------------
    # Comparison / boolean
    # ----------------------------------------------------------------

    def ones_like(self, x: Array) -> Array:
        """Create a tensor of ones with the same shape and dtype."""
        ...

    def zeros_like(self, x: Array) -> Array:
        """Create a tensor of zeros with the same shape and dtype."""
        ...

    def float_scalar(self, x: Array) -> float:
        """Extract a scalar float from a 0-d tensor."""
        ...
