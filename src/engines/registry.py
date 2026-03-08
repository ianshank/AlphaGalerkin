"""Engine protocol registry.

Provides discovery and factory creation of engine implementations
using the template registry pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.engines.protocol import BaseEngine
from src.templates.registry import create_registry

if TYPE_CHECKING:
    from src.engines.config import EngineConfig

logger = structlog.get_logger(__name__)

# Create the registry and decorator using the template pattern
EngineRegistry, register_engine = create_registry("Engine", BaseEngine)


def create_engine(config: EngineConfig) -> BaseEngine:
    """Factory function to create an engine from configuration.

    Uses the protocol field to look up the registered engine class,
    then instantiates it with the provided configuration.

    Args:
        config: Engine configuration with protocol and path.

    Returns:
        Engine instance (not yet started).

    Raises:
        KeyError: If the protocol is not registered.

    """
    protocol_name = config.protocol.value
    engine_cls = EngineRegistry().get_or_raise(protocol_name)

    logger.info(
        "engine_created",
        protocol=protocol_name,
        engine_cls=engine_cls.__name__,
        path=str(config.engine_path),
    )

    return engine_cls(config)  # type: ignore[call-arg]


def _register_builtin_engines() -> None:
    """Register built-in engine protocol implementations."""
    registry = EngineRegistry()

    if not registry.is_registered("uci"):
        from src.engines.uci import UCIEngine

        register_engine("uci")(UCIEngine)


# Register on import
_register_builtin_engines()
