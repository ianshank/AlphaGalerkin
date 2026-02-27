"""Debug utilities for tensor inspection across PyTorch and JAX backends.

All functions work with both ``torch.Tensor`` and ``jax.Array`` via
duck-typing.  Neither framework is imported at the top level -- only
at the point of use when framework-specific behaviour is needed.

Example:
    import torch
    from src.backend.debug import assert_shape, assert_no_nans, log_tensor_stats

    x = torch.randn(4, 8, 16)
    assert_shape(x, (4, -1, 16), name="encoder_output")
    assert_no_nans(x, name="encoder_output")
    log_tensor_stats(logger, "encoder_output", x)

"""

from __future__ import annotations

import math
from typing import Any


# ------------------------------------------------------------------
# Shape / dtype assertions
# ------------------------------------------------------------------


def assert_shape(
    array: Any,
    expected_shape: tuple[int, ...],
    name: str = "",
) -> None:
    """Raise ``ValueError`` if the array shape does not match *expected_shape*.

    A value of ``-1`` in *expected_shape* acts as a wildcard that matches
    any size in that dimension.

    Args:
        array: Tensor-like object with a ``.shape`` attribute.
        expected_shape: Tuple of expected sizes (``-1`` = any).
        name: Optional human-readable name for error messages.

    Raises:
        ValueError: If the shapes are incompatible.

    """
    actual_shape = tuple(array.shape)
    label = f" '{name}'" if name else ""

    if len(actual_shape) != len(expected_shape):
        msg = (
            f"Tensor{label} has {len(actual_shape)} dimensions, "
            f"expected {len(expected_shape)}. "
            f"Actual shape: {actual_shape}, expected: {expected_shape}"
        )
        raise ValueError(msg)

    for dim_idx, (actual, expected) in enumerate(
        zip(actual_shape, expected_shape)
    ):
        if expected == -1:
            continue
        if actual != expected:
            msg = (
                f"Tensor{label} has size {actual} at dimension {dim_idx}, "
                f"expected {expected}. "
                f"Actual shape: {actual_shape}, expected: {expected_shape}"
            )
            raise ValueError(msg)


def _normalize_dtype_str(dtype: Any) -> str:
    """Normalize a dtype to a comparable string representation.

    Handles framework dtypes (``torch.float32``), numpy dtype objects
    (``np.dtype('float32')``), numpy type classes (``np.float64``),
    and plain strings.
    """
    # If it has a .name attribute (numpy dtype, torch dtype), use it.
    if hasattr(dtype, "name"):
        return str(dtype.name)

    # Fallback: convert to string and strip module prefixes.
    s = str(dtype)

    # Handle numpy type classes like "<class 'numpy.float64'>"
    if s.startswith("<class '") and s.endswith("'>"):
        s = s[len("<class '"):-len("'>")]
        # Strip module prefix: "numpy.float64" -> "float64"
        if "." in s:
            s = s.rsplit(".", 1)[-1]

    # Strip framework prefixes: "torch.float32" -> "float32"
    for prefix in ("torch.", "jnp.", "jax.numpy."):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    return s


def assert_dtype(
    array: Any,
    expected_dtype: Any,
    name: str = "",
) -> None:
    """Raise ``ValueError`` if the array dtype does not match.

    The comparison uses string representations so that it works
    uniformly across frameworks (e.g. ``torch.float32`` vs
    ``jnp.float32``).

    Args:
        array: Tensor-like object with a ``.dtype`` attribute.
        expected_dtype: Expected dtype (framework dtype or string).
        name: Optional human-readable name for error messages.

    Raises:
        ValueError: If the dtypes differ.

    """
    actual_str = _normalize_dtype_str(array.dtype)
    expected_str = _normalize_dtype_str(expected_dtype)

    if actual_str != expected_str:
        label = f" '{name}'" if name else ""
        msg = (
            f"Tensor{label} has dtype {actual_str}, "
            f"expected {expected_str}"
        )
        raise ValueError(msg)


# ------------------------------------------------------------------
# Numeric health assertions
# ------------------------------------------------------------------


def _has_nans(array: Any) -> bool:
    """Return ``True`` if *array* contains any NaN values.

    Uses framework-specific ``isnan`` when available, falling back
    to a Python-level check for scalars.
    """
    try:
        # Both torch and jax/jnp expose isnan + any.
        nan_mask = _call_isnan(array)
        return bool(nan_mask.any())
    except (AttributeError, TypeError):
        # Scalar fallback.
        try:
            return bool(math.isnan(float(array)))
        except (TypeError, ValueError):
            return False


def _has_infs(array: Any) -> bool:
    """Return ``True`` if *array* contains any Inf values."""
    try:
        inf_mask = _call_isinf(array)
        return bool(inf_mask.any())
    except (AttributeError, TypeError):
        try:
            return bool(math.isinf(float(array)))
        except (TypeError, ValueError):
            return False


def _call_isnan(array: Any) -> Any:
    """Call the appropriate ``isnan`` for the array's framework."""
    # Try torch first (torch.Tensor has .isnan() method).
    if hasattr(array, "isnan") and callable(array.isnan):
        return array.isnan()

    # Try numpy (numpy arrays don't have .isnan() but numpy.isnan works).
    try:
        import numpy as _np  # noqa: WPS433

        if isinstance(array, _np.ndarray):
            return _np.isnan(array)
    except ImportError:
        pass

    # Try jnp.isnan via the array's module.
    try:
        import jax.numpy as jnp  # noqa: WPS433

        return jnp.isnan(array)
    except ImportError:
        pass

    try:
        import torch  # noqa: WPS433

        return torch.isnan(array)
    except ImportError:
        pass

    msg = "Cannot determine framework for isnan check"
    raise TypeError(msg)


def _call_isinf(array: Any) -> Any:
    """Call the appropriate ``isinf`` for the array's framework."""
    if hasattr(array, "isinf") and callable(array.isinf):
        return array.isinf()

    # Try numpy.
    try:
        import numpy as _np  # noqa: WPS433

        if isinstance(array, _np.ndarray):
            return _np.isinf(array)
    except ImportError:
        pass

    try:
        import jax.numpy as jnp  # noqa: WPS433

        return jnp.isinf(array)
    except ImportError:
        pass

    try:
        import torch  # noqa: WPS433

        return torch.isinf(array)
    except ImportError:
        pass

    msg = "Cannot determine framework for isinf check"
    raise TypeError(msg)


def assert_no_nans(array: Any, name: str = "") -> None:
    """Raise ``ValueError`` if the array contains NaN values.

    Args:
        array: Tensor-like object.
        name: Optional human-readable name for error messages.

    Raises:
        ValueError: If any element is NaN.

    """
    if _has_nans(array):
        label = f" '{name}'" if name else ""
        shape_str = ""
        try:
            shape_str = f" (shape={tuple(array.shape)})"
        except (AttributeError, TypeError):
            pass
        msg = f"Tensor{label}{shape_str} contains NaN values"
        raise ValueError(msg)


def assert_finite(array: Any, name: str = "") -> None:
    """Raise ``ValueError`` if the array contains NaN or Inf values.

    Args:
        array: Tensor-like object.
        name: Optional human-readable name for error messages.

    Raises:
        ValueError: If any element is NaN or Inf.

    """
    label = f" '{name}'" if name else ""
    shape_str = ""
    try:
        shape_str = f" (shape={tuple(array.shape)})"
    except (AttributeError, TypeError):
        pass

    if _has_nans(array):
        msg = f"Tensor{label}{shape_str} contains NaN values"
        raise ValueError(msg)

    if _has_infs(array):
        msg = f"Tensor{label}{shape_str} contains Inf values"
        raise ValueError(msg)


# ------------------------------------------------------------------
# Tensor statistics logging
# ------------------------------------------------------------------


def log_tensor_stats(
    logger: Any,
    name: str,
    array: Any,
) -> dict[str, Any]:
    """Log detailed statistics for a tensor and return them as a dict.

    Logs: shape, dtype, min, max, mean, std, has_nan, has_inf.

    Args:
        logger: Any logger with ``.debug(event, **kw)`` method
            (e.g. :class:`~src.backend.logging.BackendLogger`).
        name: Human-readable name for the tensor.
        array: Tensor-like object.

    Returns:
        Dictionary of computed statistics.

    """
    stats: dict[str, Any] = {"tensor_name": name}

    # Shape
    try:
        stats["shape"] = tuple(array.shape)
    except (AttributeError, TypeError):
        stats["shape"] = "unknown"

    # Dtype
    try:
        stats["dtype"] = str(array.dtype)
    except (AttributeError, TypeError):
        stats["dtype"] = "unknown"

    # Scalar statistics
    try:
        stats["min"] = float(array.min())
    except (AttributeError, TypeError, RuntimeError, ValueError):
        stats["min"] = None

    try:
        stats["max"] = float(array.max())
    except (AttributeError, TypeError, RuntimeError, ValueError):
        stats["max"] = None

    try:
        stats["mean"] = float(array.mean())
    except (AttributeError, TypeError, RuntimeError, ValueError):
        stats["mean"] = None

    try:
        stats["std"] = float(array.std())
    except (AttributeError, TypeError, RuntimeError, ValueError):
        stats["std"] = None

    # Health checks
    stats["has_nan"] = _has_nans(array)
    stats["has_inf"] = _has_infs(array)

    logger.debug("tensor_stats_detailed", **stats)
    return stats


# ------------------------------------------------------------------
# Gradient health checking
# ------------------------------------------------------------------


def _flatten_grads(grads: Any) -> list[tuple[str, Any]]:
    """Flatten a gradient structure into ``(name, tensor)`` pairs.

    Supports:
    - ``dict[str, Tensor]``
    - ``list / tuple`` of tensors
    - JAX pytrees (via ``jax.tree_util.tree_leaves``)
    - Single tensors

    Returns:
        List of (name, tensor) pairs.

    """
    if isinstance(grads, dict):
        return list(grads.items())

    if isinstance(grads, (list, tuple)):
        return [(f"grad_{i}", g) for i, g in enumerate(grads)]

    # Try JAX pytree flattening.
    try:
        import jax.tree_util  # noqa: WPS433

        leaves = jax.tree_util.tree_leaves(grads)
        if len(leaves) > 1 or (len(leaves) == 1 and leaves[0] is not grads):
            return [(f"leaf_{i}", leaf) for i, leaf in enumerate(leaves)]
    except ImportError:
        pass

    # Single tensor.
    return [("grad", grads)]


def _compute_norm(array: Any) -> float:
    """Compute the L2 norm of *array* as a Python float.

    Tries framework-specific norm functions, falling back to a
    manual sqrt(sum(x^2)) computation.
    """
    # Try numpy first (most common in testing, always available).
    try:
        import numpy as _np  # noqa: WPS433

        if isinstance(array, _np.ndarray):
            return float(_np.linalg.norm(array.ravel()))
    except ImportError:
        pass

    try:
        # PyTorch: torch.linalg.norm
        import torch  # noqa: WPS433

        if isinstance(array, torch.Tensor):
            return float(torch.linalg.norm(array.float()))
    except ImportError:
        pass

    try:
        import jax.numpy as jnp  # noqa: WPS433

        return float(jnp.linalg.norm(array))
    except ImportError:
        pass

    # Generic fallback.
    try:
        return float((array * array).sum() ** 0.5)
    except (AttributeError, TypeError):
        return float("nan")


def check_gradient_health(
    grads: Any,
    logger: Any | None = None,
    max_norm: float | None = None,
) -> dict[str, Any]:
    """Check a gradient dict/pytree for health issues.

    Inspects every leaf tensor for NaN and Inf values, computes
    per-parameter and total gradient norms, and optionally checks
    against a maximum norm threshold.

    Args:
        grads: Gradient structure (dict, list, pytree, or single tensor).
        logger: Optional logger with ``.warning()`` and ``.debug()``
            methods.  If ``None``, issues are recorded but not logged.
        max_norm: Optional maximum L2 norm.  If any parameter's gradient
            norm exceeds this, it is flagged as exploding.

    Returns:
        Dictionary with keys:

        - ``healthy`` (bool): ``True`` if no issues detected.
        - ``total_norm`` (float): Global gradient L2 norm.
        - ``num_params`` (int): Number of gradient tensors inspected.
        - ``has_nan`` (bool): ``True`` if any gradient contains NaN.
        - ``has_inf`` (bool): ``True`` if any gradient contains Inf.
        - ``max_param_norm`` (float): Largest per-parameter norm.
        - ``issues`` (list[str]): Human-readable issue descriptions.

    """
    flat = _flatten_grads(grads)

    issues: list[str] = []
    has_nan = False
    has_inf = False
    param_norms: list[float] = []
    total_sq: float = 0.0

    for param_name, grad_tensor in flat:
        if grad_tensor is None:
            continue

        # NaN check
        if _has_nans(grad_tensor):
            has_nan = True
            issue = f"NaN detected in gradient '{param_name}'"
            issues.append(issue)
            if logger is not None:
                logger.warning("gradient_nan", param=param_name)

        # Inf check
        if _has_infs(grad_tensor):
            has_inf = True
            issue = f"Inf detected in gradient '{param_name}'"
            issues.append(issue)
            if logger is not None:
                logger.warning("gradient_inf", param=param_name)

        # Norm
        norm = _compute_norm(grad_tensor)
        param_norms.append(norm)

        if not math.isnan(norm):
            total_sq += norm * norm

        # Per-parameter explosion check
        if max_norm is not None and norm > max_norm:
            issue = (
                f"Gradient '{param_name}' norm {norm:.4f} "
                f"exceeds max_norm {max_norm:.4f}"
            )
            issues.append(issue)
            if logger is not None:
                logger.warning(
                    "gradient_explosion",
                    param=param_name,
                    norm=round(norm, 6),
                    max_norm=max_norm,
                )

    total_norm = math.sqrt(total_sq)
    max_param_norm = max(param_norms) if param_norms else 0.0
    healthy = len(issues) == 0

    result = {
        "healthy": healthy,
        "total_norm": round(total_norm, 6),
        "num_params": len(flat),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "max_param_norm": round(max_param_norm, 6),
        "issues": issues,
    }

    if logger is not None:
        if healthy:
            logger.debug(
                "gradient_health_ok",
                total_norm=result["total_norm"],
                num_params=result["num_params"],
                max_param_norm=result["max_param_norm"],
            )
        else:
            logger.warning(
                "gradient_health_issues",
                total_norm=result["total_norm"],
                num_issues=len(issues),
                issues=issues,
            )

    return result
