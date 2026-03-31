"""Backend-aware structured logging for AlphaGalerkin.

Extends :class:`~src.templates.logging.BaseModuleLogger` with
backend-specific context (device, precision, backend type) and
convenience methods for tensor inspection and computation timing.

Example:
    from src.backend.logging import BackendLogger, create_backend_logger
    from src.backend.config import BackendConfig

    # Direct construction
    logger = BackendLogger("encoder", backend_type="torch", device="cuda")
    logger.log_tensor("weights", model.weight)

    # Factory from config
    config = BackendConfig(backend="torch", device="gpu")
    logger = create_backend_logger("decoder", config=config)

    # Log a computation with timing and shape tracking
    result = logger.log_computation("matmul", torch.matmul, a, b)

"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from src.backend.types import Array
from src.templates.logging import BaseModuleLogger

__all__ = ["BackendLogger", "create_backend_logger"]


class BackendLogger(BaseModuleLogger):
    """Logger with backend-aware context (device, precision, backend type).

    Inherits all standard logging methods from
    :class:`~src.templates.logging.BaseModuleLogger` and adds
    tensor-level inspection and computation profiling helpers.
    """

    _module_name: str = "backend"

    def __init__(
        self,
        component: str,
        backend_type: str | None = None,
        device: str | None = None,
        precision: str | None = None,
        run_id: str | None = None,
        **context: Any,
    ) -> None:
        """Initialize the backend logger.

        Args:
            component: Component name within the backend module.
            backend_type: Backend framework identifier (e.g. "torch", "jax").
            device: Device string (e.g. "cpu", "cuda:0", "tpu").
            precision: Floating-point precision (e.g. "float32").
            run_id: Optional unique run identifier.
            **context: Additional context key-value pairs to bind.

        """
        extra: dict[str, Any] = {}
        if backend_type is not None:
            extra["backend_type"] = backend_type
        if device is not None:
            extra["device"] = device
        if precision is not None:
            extra["precision"] = precision
        extra.update(context)

        super().__init__(component=component, run_id=run_id, **extra)

    # ------------------------------------------------------------------
    # Tensor inspection
    # ------------------------------------------------------------------

    def log_tensor(
        self,
        name: str,
        array: Array,
        level: str = "debug",
    ) -> None:
        """Log summary statistics of a tensor.

        Captures shape, dtype, and basic statistics (min, max, mean)
        without materializing the full tensor contents.

        Args:
            name: Human-readable name for the tensor.
            array: Tensor-like object (torch.Tensor or jax.Array).
            level: Log level to emit the message at.

        """
        log_method = getattr(self._logger, level)
        info: dict[str, Any] = {"tensor_name": name}

        try:
            info["shape"] = tuple(array.shape)
        except (AttributeError, TypeError):
            info["shape"] = "unknown"

        try:
            info["dtype"] = str(array.dtype)
        except (AttributeError, TypeError):
            info["dtype"] = "unknown"

        # Compute scalar stats via duck-typing.  Both torch and jax
        # tensors support .min(), .max(), .mean() and can be converted
        # to Python floats with float().
        try:
            info["min"] = float(array.min())
            info["max"] = float(array.max())
            # .mean() may fail on integer dtypes in some frameworks.
            info["mean"] = float(array.mean())
        except (AttributeError, TypeError, RuntimeError):
            pass

        log_method("tensor_stats", **info)

    # ------------------------------------------------------------------
    # Computation profiling
    # ------------------------------------------------------------------

    def log_computation(
        self,
        name: str,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *fn* and log its timing plus input/output shapes.

        Args:
            name: Human-readable name for the computation.
            fn: Callable to execute.
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            The return value of *fn(*args, **kwargs)*.

        """
        # Collect input shapes where available.
        input_shapes: list[tuple[Any, ...] | None] = []
        for arg in args:
            try:
                input_shapes.append(tuple(arg.shape))
            except (AttributeError, TypeError):
                input_shapes.append(None)

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        duration = time.perf_counter() - start

        info: dict[str, Any] = {
            "computation": name,
            "duration_seconds": round(duration, 6),
            "input_shapes": input_shapes,
        }

        try:
            info["output_shape"] = tuple(result.shape)
        except (AttributeError, TypeError):
            info["output_shape"] = None

        self._logger.debug("computation_profile", **info)
        return result


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def create_backend_logger(
    component: str,
    config: Any | None = None,
    **extra_context: Any,
) -> BackendLogger:
    """Create a :class:`BackendLogger` optionally populated from a config.

    If *config* is a :class:`~src.backend.config.BackendConfig` (or
    duck-types with ``.backend``, ``.device``, and ``.precision``
    attributes), the logger context is filled automatically.

    Args:
        component: Component name within the backend module.
        config: Optional :class:`BackendConfig` instance.
        **extra_context: Additional context key-value pairs.

    Returns:
        Configured :class:`BackendLogger` instance.

    """
    backend_type: str | None = None
    device: str | None = None
    precision: str | None = None

    if config is not None:
        try:
            backend_type = str(config.backend.value)
        except AttributeError:
            with contextlib.suppress(AttributeError):
                backend_type = str(config.backend)

        try:
            device = str(config.device.value)
        except AttributeError:
            with contextlib.suppress(AttributeError):
                device = str(config.device)

        try:
            precision = str(config.precision.value)
        except AttributeError:
            with contextlib.suppress(AttributeError):
                precision = str(config.precision)

    return BackendLogger(
        component=component,
        backend_type=backend_type,
        device=device,
        precision=precision,
        **extra_context,
    )
