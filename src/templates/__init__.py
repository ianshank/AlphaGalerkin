"""Reusable templates for AlphaGalerkin module development.

This package provides base classes and utilities that enforce best practices:
- Pydantic-based configuration with validation
- Thread-safe singleton registries with decorator registration
- Structured logging with context binding
- Debug utilities with timing and memory tracking

Usage:
    from src.templates import (
        BaseModuleConfig,
        create_registry,
        create_logger_class,
        DebugContext,
    )

    # Create module-specific config
    class MyModuleConfig(BaseModuleConfig):
        my_param: int = Field(default=100, ge=1)

    # Create module-specific registry
    MyRegistry, register_my_module = create_registry("MyModule", BaseMyClass)

    # Create module-specific logger
    MyLogger = create_logger_class("MyModule")
"""

from src.templates.config import (
    BaseModuleConfig,
    MetricDefinition,
    ThresholdOperator,
    create_config_class,
)
from src.templates.registry import (
    BaseRegistry,
    create_registry,
)
from src.templates.logging import (
    BaseModuleLogger,
    DebugContext,
    configure_module_logging,
    create_logger_class,
    log_timing,
    log_call,
)
from src.templates.base import (
    BaseExecutable,
    ExecutionResult,
    ExecutionStatus,
)

__all__ = [
    # Config
    "BaseModuleConfig",
    "MetricDefinition",
    "ThresholdOperator",
    "create_config_class",
    # Registry
    "BaseRegistry",
    "create_registry",
    # Logging
    "BaseModuleLogger",
    "DebugContext",
    "configure_module_logging",
    "create_logger_class",
    "log_timing",
    "log_call",
    # Base
    "BaseExecutable",
    "ExecutionResult",
    "ExecutionStatus",
]
