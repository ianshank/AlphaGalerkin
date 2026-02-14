"""Architecture registry for hot-swapping backbones."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger("nn.registry")

_ARCHITECTURE_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_architecture(name: str) -> Callable[[type], type]:
    """Decorator to register a backbone architecture."""

    def decorator(cls: type) -> type:
        if name in _ARCHITECTURE_REGISTRY:
            msg = f"Architecture '{name}' already registered"
            raise ValueError(msg)
        _ARCHITECTURE_REGISTRY[name] = cls
        logger.info("nn.registry.registered", name=name)
        return cls

    return decorator


def get_architecture(name: str, **kwargs: Any) -> Any:
    """Get a registered architecture by name."""
    if name not in _ARCHITECTURE_REGISTRY:
        available = ", ".join(
            sorted(_ARCHITECTURE_REGISTRY.keys()),
        )
        msg = f"Architecture '{name}' not found. Available: {available}"
        raise KeyError(msg)
    return _ARCHITECTURE_REGISTRY[name](**kwargs)


def list_architectures() -> list[str]:
    """List all registered architecture names."""
    return sorted(_ARCHITECTURE_REGISTRY.keys())
