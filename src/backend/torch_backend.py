"""PyTorch backend implementation for AlphaGalerkin.

Implements the BackendInterface protocol using PyTorch, providing
GPU-accelerated tensor operations with reproducible random number
generation via ``torch.Generator``, functional autograd wrappers,
and automatic CUDA device detection.

PyTorch-specific design notes:
- Random number generation uses a dedicated ``torch.Generator`` seeded
  from ``BackendConfig.rng_seed`` for reproducibility.
- ``grad`` and ``value_and_grad`` wrap ``torch.autograd.grad`` in a
  functional style, temporarily enabling ``requires_grad`` on the
  target arguments, computing the forward pass, and extracting
  gradients without mutating the caller's tensors.
- Device auto-detection checks CUDA availability and falls back to CPU.
- cuDNN benchmark and deterministic modes are configured from
  ``BackendConfig`` at construction time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.backend.config import BackendConfig
from src.backend.types import Array, BackendType, DTypeLike, Precision, Shape, ShapeLike

logger = logging.getLogger(__name__)


class TorchBackend:
    """PyTorch implementation of the AlphaGalerkin BackendInterface.

    Manages PyTorch configuration (cuDNN benchmark, deterministic mode),
    a dedicated ``torch.Generator`` for reproducible randomness, and
    dtype mapping.  All tensor operations are thin wrappers around
    ``torch`` functions.

    Args:
        config: Backend configuration with PyTorch-specific settings.
    """

    def __init__(self, config: BackendConfig) -> None:
        self._config = config

        # Configure PyTorch global settings.
        torch.backends.cudnn.benchmark = config.torch_cudnn_benchmark
        torch.backends.cudnn.deterministic = config.torch_deterministic
        if config.torch_deterministic:
            torch.use_deterministic_algorithms(True)

        # Resolve the default dtype from the Precision enum.
        self._default_dtype: torch.dtype = self.get_dtype(config.precision)

        # Resolve the default device.
        self._default_device: str = self._resolve_device(config)

        # Create a dedicated generator for reproducible random ops.
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(config.rng_seed)

        # Also set the global seed for modules that rely on it.
        torch.manual_seed(config.rng_seed)

        logger.info(
            "TorchBackend initialized: device=%s, precision=%s, seed=%d",
            self._default_device,
            config.precision.value,
            config.rng_seed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(config: BackendConfig) -> str:
        """Determine the concrete device string from the config.

        If the config specifies ``DeviceType.AUTO``, CUDA availability
        is checked.  Otherwise the enum value is mapped to a PyTorch
        device string.
        """
        from src.backend.types import DeviceType

        if config.device == DeviceType.AUTO:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if config.device == DeviceType.GPU:
            return "cuda"
        if config.device == DeviceType.CPU:
            return "cpu"
        # TPU not natively supported by PyTorch (would need torch_xla).
        return "cpu"

    def _resolve_dtype(self, dtype: DTypeLike | None) -> torch.dtype:
        """Return *dtype* if given, otherwise the configured default."""
        return dtype if dtype is not None else self._default_dtype

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> BackendType:
        """Return the backend type identifier."""
        return BackendType.TORCH

    @property
    def default_dtype(self) -> DTypeLike:
        """Return the default floating-point dtype."""
        return self._default_dtype

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    def zeros(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor of zeros."""
        return torch.zeros(
            *shape, dtype=self._resolve_dtype(dtype), device=self._default_device,
        )

    def ones(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor of ones."""
        return torch.ones(
            *shape, dtype=self._resolve_dtype(dtype), device=self._default_device,
        )

    def full(
        self,
        shape: ShapeLike,
        fill_value: float,
        dtype: DTypeLike | None = None,
    ) -> Array:
        """Create a tensor filled with a constant value."""
        return torch.full(
            tuple(shape),
            fill_value,
            dtype=self._resolve_dtype(dtype),
            device=self._default_device,
        )

    def randn(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor with standard normal random values.

        Uses the internal ``torch.Generator`` for reproducibility.
        The tensor is generated on CPU (generator device) and moved
        to the default device.
        """
        t = torch.randn(
            *shape,
            dtype=self._resolve_dtype(dtype),
            generator=self._generator,
        )
        return t.to(device=self._default_device)

    def rand(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Array:
        """Create a tensor with uniform random values in [0, 1).

        Uses the internal ``torch.Generator`` for reproducibility.
        """
        t = torch.rand(
            *shape,
            dtype=self._resolve_dtype(dtype),
            generator=self._generator,
        )
        return t.to(device=self._default_device)

    def arange(
        self,
        start: float,
        stop: float | None = None,
        step: float = 1.0,
        dtype: DTypeLike | None = None,
    ) -> Array:
        """Create a 1-D tensor with evenly spaced values."""
        if stop is None:
            return torch.arange(
                start,
                dtype=self._resolve_dtype(dtype),
                device=self._default_device,
            )
        return torch.arange(
            start,
            stop,
            step,
            dtype=self._resolve_dtype(dtype),
            device=self._default_device,
        )

    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: DTypeLike | None = None,
    ) -> Array:
        """Create a 1-D tensor with linearly spaced values."""
        return torch.linspace(
            start,
            stop,
            num,
            dtype=self._resolve_dtype(dtype),
            device=self._default_device,
        )

    def from_numpy(self, array: Any) -> Array:
        """Convert a numpy array to a PyTorch tensor on the default device."""
        return torch.from_numpy(np.asarray(array)).to(device=self._default_device)

    def to_numpy(self, array: Array) -> Any:
        """Convert a PyTorch tensor to a numpy array."""
        return array.detach().cpu().numpy()

    def tensor(self, data: Any, dtype: DTypeLike | None = None) -> Array:
        """Create a PyTorch tensor from data (list, tuple, numpy array, etc.)."""
        return torch.as_tensor(
            data, dtype=self._resolve_dtype(dtype),
        ).to(device=self._default_device)

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
        return int(x.numel())

    # ------------------------------------------------------------------
    # Tensor manipulation
    # ------------------------------------------------------------------

    def reshape(self, x: Array, shape: ShapeLike) -> Array:
        """Reshape a tensor."""
        return x.reshape(tuple(shape))

    def transpose(self, x: Array, axes: tuple[int, ...] | None = None) -> Array:
        """Transpose a tensor.

        For 2-D tensors with ``axes=None``, uses ``torch.t()``.
        For N-D tensors, ``axes`` must be provided as a full
        permutation and ``torch.permute()`` is used.
        """
        if axes is None:
            # Default transpose: reverse dimensions.
            return x.permute(*reversed(range(x.ndim)))
        return x.permute(*axes)

    def expand_dims(self, x: Array, axis: int) -> Array:
        """Add a dimension at the specified axis."""
        return x.unsqueeze(axis)

    def squeeze(self, x: Array, axis: int | None = None) -> Array:
        """Remove dimensions of size 1."""
        if axis is None:
            return x.squeeze()
        return x.squeeze(axis)

    def cat(self, arrays: list[Array], axis: int = 0) -> Array:
        """Concatenate tensors along an axis."""
        return torch.cat(arrays, dim=axis)

    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        """Stack tensors along a new axis."""
        return torch.stack(arrays, dim=axis)

    def split(
        self,
        x: Array,
        num_or_sections: int | list[int],
        axis: int = 0,
    ) -> list[Array]:
        """Split a tensor along an axis.

        ``torch.split`` expects the split size (or list of sizes),
        whereas ``numpy``/``jax`` ``split`` accept indices. Here we
        follow the interface convention where an ``int`` means
        "split into chunks of this size" and a ``list`` gives the
        size of each section.
        """
        parts = torch.split(x, num_or_sections, dim=axis)
        return list(parts)

    def pad(
        self,
        x: Array,
        pad_width: list[tuple[int, int]],
        value: float = 0.0,
    ) -> Array:
        """Pad a tensor with a constant value.

        ``torch.nn.functional.pad`` expects padding in reverse
        dimension order as a flat tuple: (last_dim_left, last_dim_right,
        second_last_left, second_last_right, ...).

        Args:
            x: Input tensor.
            pad_width: List of (before, after) pad widths per dimension,
                       ordered from the first to the last dimension.
            value: Padding value.
        """
        # Reverse the dimension order and flatten to match torch convention.
        torch_pad: list[int] = []
        for before, after in reversed(pad_width):
            torch_pad.extend([before, after])
        return F.pad(x, torch_pad, mode="constant", value=value)

    # ------------------------------------------------------------------
    # Math operations
    # ------------------------------------------------------------------

    def add(self, x: Array, y: Array) -> Array:
        """Element-wise addition."""
        return torch.add(x, y)

    def mul(self, x: Array, y: Array) -> Array:
        """Element-wise multiplication."""
        return torch.mul(x, y)

    def matmul(self, x: Array, y: Array) -> Array:
        """Matrix multiplication."""
        return torch.matmul(x, y)

    def einsum(self, subscripts: str, *operands: Array) -> Array:
        """Einstein summation."""
        return torch.einsum(subscripts, *operands)

    def sum(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Sum along axis."""
        if axis is None:
            return x.sum()
        return x.sum(dim=axis, keepdim=keepdims)

    def mean(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Mean along axis."""
        if axis is None:
            return x.mean()
        return x.mean(dim=axis, keepdim=keepdims)

    def max(self, x: Array, axis: int | None = None) -> Array:
        """Maximum along axis.

        When *axis* is provided, ``torch.max`` returns a named tuple
        ``(values, indices)``.  We return only the values to match
        the interface contract.
        """
        if axis is None:
            return x.max()
        return x.max(dim=axis).values

    def min(self, x: Array, axis: int | None = None) -> Array:
        """Minimum along axis.

        When *axis* is provided, ``torch.min`` returns a named tuple
        ``(values, indices)``.  We return only the values to match
        the interface contract.
        """
        if axis is None:
            return x.min()
        return x.min(dim=axis).values

    def abs(self, x: Array) -> Array:
        """Absolute value."""
        return torch.abs(x)

    def sqrt(self, x: Array) -> Array:
        """Element-wise square root."""
        return torch.sqrt(x)

    def exp(self, x: Array) -> Array:
        """Element-wise exponential."""
        return torch.exp(x)

    def log(self, x: Array) -> Array:
        """Element-wise natural logarithm."""
        return torch.log(x)

    def cos(self, x: Array) -> Array:
        """Element-wise cosine."""
        return torch.cos(x)

    def sin(self, x: Array) -> Array:
        """Element-wise sine."""
        return torch.sin(x)

    def clamp(
        self,
        x: Array,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> Array:
        """Clamp values to a range."""
        return torch.clamp(x, min=min_val, max=max_val)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        """Element-wise conditional selection."""
        return torch.where(condition, x, y)

    def pow(self, x: Array, exponent: float | Array) -> Array:
        """Element-wise power."""
        return torch.pow(x, exponent)

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def softmax(self, x: Array, axis: int = -1) -> Array:
        """Softmax activation."""
        return F.softmax(x, dim=axis)

    def log_softmax(self, x: Array, axis: int = -1) -> Array:
        """Log-softmax activation."""
        return F.log_softmax(x, dim=axis)

    def relu(self, x: Array) -> Array:
        """ReLU activation."""
        return F.relu(x)

    def gelu(self, x: Array) -> Array:
        """GELU activation."""
        return F.gelu(x)

    def sigmoid(self, x: Array) -> Array:
        """Sigmoid activation."""
        return torch.sigmoid(x)

    def tanh(self, x: Array) -> Array:
        """Hyperbolic tangent."""
        return torch.tanh(x)

    # ------------------------------------------------------------------
    # FFT operations
    # ------------------------------------------------------------------

    def fft2(self, x: Array) -> Array:
        """2D FFT (complex-to-complex)."""
        return torch.fft.fft2(x)

    def ifft2(self, x: Array, s: tuple[int, int] | None = None) -> Array:
        """2D inverse FFT."""
        return torch.fft.ifft2(x, s=s)

    def rfft2(self, x: Array) -> Array:
        """2D real-to-complex FFT."""
        return torch.fft.rfft2(x)

    def irfft2(self, x: Array, s: tuple[int, int] | None = None) -> Array:
        """2D complex-to-real inverse FFT."""
        return torch.fft.irfft2(x, s=s)

    def fftfreq(self, n: int, d: float = 1.0) -> Array:
        """DFT sample frequencies."""
        return torch.fft.fftfreq(n, d=d, device=self._default_device)

    def rfftfreq(self, n: int, d: float = 1.0) -> Array:
        """DFT sample frequencies for rfft."""
        return torch.fft.rfftfreq(n, d=d, device=self._default_device)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def svdvals(self, x: Array) -> Array:
        """Compute singular values of a matrix."""
        return torch.linalg.svdvals(x)

    def norm(
        self,
        x: Array,
        ord: int | float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
    ) -> Array:
        """Compute tensor norm."""
        return torch.linalg.norm(x, ord=ord, dim=axis)

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

        Wraps ``torch.autograd.grad`` in a functional style analogous
        to ``jax.grad``.  The returned callable:

        1. Clones the target argument(s) (identified by *argnums*)
           with ``requires_grad=True``.
        2. Calls *fn* with the cloned arguments.
        3. Computes gradients of the scalar output w.r.t. the cloned
           arguments via ``torch.autograd.grad``.
        4. Returns the gradient(s).  When *has_aux* is ``True``, *fn*
           is expected to return ``(loss, aux)`` and the returned
           callable yields ``(grads, aux)``.

        Args:
            fn: A function returning a scalar tensor (or ``(scalar, aux)``
                when *has_aux* is True).
            argnums: Index or tuple of indices identifying which
                positional arguments to differentiate w.r.t.
            has_aux: If True, *fn* returns ``(output, aux)`` and the
                wrapper returns ``(grads, aux)``.

        Returns:
            A callable with the same signature as *fn* that returns
            gradients instead of the function value.
        """
        indices = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        def grad_fn(*args: Any, **kwargs: Any) -> Any:
            args_list = list(args)
            cloned: list[torch.Tensor] = []
            for idx in indices:
                arg = args_list[idx]
                c = arg.detach().clone().requires_grad_(True)
                args_list[idx] = c
                cloned.append(c)

            output = fn(*args_list, **kwargs)

            if has_aux:
                loss, aux = output
            else:
                loss = output

            grads = torch.autograd.grad(
                loss,
                cloned,
                create_graph=False,
            )

            if isinstance(argnums, int):
                result_grads = grads[0]
            else:
                result_grads = grads

            if has_aux:
                return result_grads, aux
            return result_grads

        return grad_fn

    def value_and_grad(
        self,
        fn: Callable[..., Array],
        argnums: int | tuple[int, ...] = 0,
        has_aux: bool = False,
    ) -> Callable[..., tuple[Any, ...]]:
        """Return a function that computes both value and gradients.

        Wraps ``torch.autograd.grad`` in a functional style analogous
        to ``jax.value_and_grad``.  The returned callable:

        1. Clones the target argument(s) with ``requires_grad=True``.
        2. Calls *fn* and retains the output value.
        3. Computes gradients of the scalar output.
        4. Returns ``(value, grads)`` or ``((value, aux), grads)``
           depending on *has_aux*.

        Args:
            fn: A function returning a scalar tensor (or ``(scalar, aux)``
                when *has_aux* is True).
            argnums: Index or tuple of indices identifying which
                positional arguments to differentiate w.r.t.
            has_aux: If True, *fn* returns ``(output, aux)`` and the
                wrapper returns ``((output, aux), grads)``.

        Returns:
            A callable with the same signature as *fn* that returns
            ``(value, grads)``.
        """
        indices = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        def val_and_grad_fn(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            args_list = list(args)
            cloned: list[torch.Tensor] = []
            for idx in indices:
                arg = args_list[idx]
                c = arg.detach().clone().requires_grad_(True)
                args_list[idx] = c
                cloned.append(c)

            output = fn(*args_list, **kwargs)

            if has_aux:
                loss, aux = output
            else:
                loss = output

            grads = torch.autograd.grad(
                loss,
                cloned,
                create_graph=False,
            )

            if isinstance(argnums, int):
                result_grads = grads[0]
            else:
                result_grads = grads

            if has_aux:
                return (loss, aux), result_grads
            return loss, result_grads

        return val_and_grad_fn

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to_device(self, x: Array, device: str) -> Array:
        """Move tensor to a device."""
        return x.to(device=device)

    def get_default_device(self) -> str:
        """Get the default device string.

        Returns the device resolved at construction time (e.g.
        ``'cuda'`` if a GPU is available and ``DeviceType.AUTO``
        was configured, otherwise ``'cpu'``).
        """
        return self._default_device

    # ------------------------------------------------------------------
    # Dtype management
    # ------------------------------------------------------------------

    def get_dtype(self, precision: Precision) -> DTypeLike:
        """Convert a Precision enum to the corresponding ``torch`` dtype."""
        mapping: dict[Precision, torch.dtype] = {
            Precision.FLOAT16: torch.float16,
            Precision.BFLOAT16: torch.bfloat16,
            Precision.FLOAT32: torch.float32,
            Precision.FLOAT64: torch.float64,
        }
        dtype = mapping.get(precision)
        if dtype is None:
            msg = f"Unsupported precision: {precision}"
            raise ValueError(msg)
        return dtype

    def cast(self, x: Array, dtype: DTypeLike) -> Array:
        """Cast a tensor to a different dtype."""
        return x.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Random state
    # ------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility.

        Re-seeds both the internal ``torch.Generator`` and the global
        PyTorch random state (``torch.manual_seed``).
        """
        self._generator.manual_seed(seed)
        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Mesh grid
    # ------------------------------------------------------------------

    def meshgrid(self, *arrays: Array, indexing: str = "ij") -> tuple[Array, ...]:
        """Create coordinate matrices from coordinate vectors."""
        return tuple(torch.meshgrid(*arrays, indexing=indexing))

    # ------------------------------------------------------------------
    # Comparison / boolean helpers
    # ------------------------------------------------------------------

    def ones_like(self, x: Array) -> Array:
        """Create a tensor of ones with the same shape and dtype."""
        return torch.ones_like(x)

    def zeros_like(self, x: Array) -> Array:
        """Create a tensor of zeros with the same shape and dtype."""
        return torch.zeros_like(x)

    def float_scalar(self, x: Array) -> float:
        """Extract a scalar float from a 0-d tensor."""
        return float(x.item())
