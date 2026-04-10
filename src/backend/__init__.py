"""Backend abstraction layer for AlphaGalerkin.

Provides a unified interface for compute operations that can be backed by
either PyTorch or JAX. During the migration from PyTorch to JAX, both
backends coexist. Once migration is complete, the PyTorch backend will
be removed and JAX becomes the sole backend.

Usage:
    from src.backend import get_backend, BackendConfig

    # Create backend from config (no hardcoded values)
    config = BackendConfig(backend="jax", precision="float32")
    backend = get_backend(config=config)

    # Or use shorthand with backend name
    backend = get_backend("torch")

    # Use the backend
    x = backend.randn((4, 10))
    y = backend.softmax(x, axis=-1)

    # Access the global default backend
    from src.backend import default_backend
    z = default_backend().zeros((3, 3))
"""

from __future__ import annotations

import threading
from typing import overload

import structlog

from src.backend.config import BackendConfig
from src.backend.interface import BackendInterface
from src.backend.types import Array, BackendType, DeviceType, Precision, Shape, ShapeLike

logger = structlog.get_logger(__name__)

# Module-level cache for backend singletons
_backends: dict[str, BackendInterface] = {}
_lock = threading.Lock()
_default_backend: BackendInterface | None = None


def _create_backend(config: BackendConfig) -> BackendInterface:
    """Create a backend instance from config.

    Args:
        config: Backend configuration.

    Returns:
        Configured backend instance.

    Raises:
        ImportError: If the requested backend's dependencies are not installed.
        ValueError: If the backend type is not recognized.

    """
    if config.backend == BackendType.TORCH:
        try:
            from src.backend.torch_backend import TorchBackend
        except ImportError as e:
            msg = "PyTorch is required for the 'torch' backend. Install it with: pip install torch"
            raise ImportError(msg) from e
        return TorchBackend(config)

    elif config.backend == BackendType.JAX:
        try:
            from src.backend.jax_backend import JaxBackend
        except ImportError as e:
            msg = (
                "JAX is required for the 'jax' backend. "
                "Install it with: pip install 'alphagalerkin[jax]'"
            )
            raise ImportError(msg) from e
        return JaxBackend(config)

    else:
        msg = f"Unknown backend: {config.backend}. Supported: {list(BackendType)}"
        raise ValueError(msg)


@overload
def get_backend(backend_or_config: str) -> BackendInterface: ...


@overload
def get_backend(backend_or_config: BackendConfig) -> BackendInterface: ...


@overload
def get_backend(*, config: BackendConfig) -> BackendInterface: ...


@overload
def get_backend() -> BackendInterface: ...


def get_backend(
    backend_or_config: str | BackendConfig | None = None,
    *,
    config: BackendConfig | None = None,
) -> BackendInterface:
    """Get or create a backend instance.

    Backends are cached by their config hash so repeated calls with
    the same configuration return the same instance.

    Args:
        backend_or_config: Either a backend name ("torch"/"jax") or a BackendConfig.
        config: Explicit config (alternative to positional arg).

    Returns:
        A configured BackendInterface instance.

    Examples:
        backend = get_backend("torch")
        backend = get_backend(BackendConfig(backend="jax"))
        backend = get_backend(config=BackendConfig(backend="jax"))
        backend = get_backend()  # returns default (torch)

    """
    # Resolve the config
    if config is not None:
        resolved_config = config
    elif isinstance(backend_or_config, BackendConfig):
        resolved_config = backend_or_config
    elif isinstance(backend_or_config, str):
        resolved_config = BackendConfig(backend=BackendType(backend_or_config))
    elif backend_or_config is None:
        resolved_config = BackendConfig()
    else:
        msg = f"Expected str, BackendConfig, or None, got {type(backend_or_config)}"
        raise TypeError(msg)

    cache_key = resolved_config.compute_hash()

    with _lock:
        if cache_key not in _backends:
            backend = _create_backend(resolved_config)
            _backends[cache_key] = backend
            logger.info(
                "backend.created",
                backend=resolved_config.backend.value,
                device=resolved_config.device.value,
                precision=resolved_config.precision.value,
                cache_key=cache_key,
            )
        return _backends[cache_key]


def default_backend() -> BackendInterface:
    """Get the default backend (torch, with default config).

    Returns:
        The default BackendInterface instance.

    """
    global _default_backend
    if _default_backend is None:
        _default_backend = get_backend()
    return _default_backend


def set_default_backend(backend: BackendInterface | BackendConfig | str) -> None:
    """Set the default backend.

    Args:
        backend: A BackendInterface, BackendConfig, or backend name string.

    """
    global _default_backend
    if isinstance(backend, str | BackendConfig):
        _default_backend = get_backend(backend)
    else:
        _default_backend = backend


def clear_cache() -> None:
    """Clear the backend cache. Primarily for testing."""
    global _default_backend
    with _lock:
        _backends.clear()
        _default_backend = None


def get_device(device: str = "auto") -> torch.device:
    """Return a ``torch.device``, auto-selecting GPU when available.

    Centralises the ``torch.device("cuda" if … else "cpu")`` pattern that
    was previously duplicated across training, POC, and experiment modules.

    Args:
        device: One of ``"auto"`` (GPU if available, else CPU), ``"cuda"``,
            or ``"cpu"``.  Unrecognised strings are forwarded to
            :class:`torch.device` directly so that ``"cuda:1"`` etc. work.

    Returns:
        A :class:`torch.device` instance.

    Examples:
        >>> dev = get_device()           # cuda if available else cpu
        >>> dev = get_device("cpu")      # always CPU
        >>> dev = get_device("cuda:1")   # second GPU

    """
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


__all__ = [
    "Array",
    "BackendConfig",
    "BackendInterface",
    "BackendType",
    "DeviceType",
    "Precision",
    "Shape",
    "ShapeLike",
    "clear_cache",
    "default_backend",
    "get_backend",
    "get_device",
    "set_default_backend",
]
