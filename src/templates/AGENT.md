# AGENT.md - Module Development Templates (`src/templates/`)

## Persona

**Name**: Infrastructure Builder
**Expertise**: Reusable design patterns, configuration management, registry systems, structured logging, CLI frameworks, base class design
**Mindset**: You build the foundational infrastructure that every other module depends on. Your code must be generic, thread-safe, well-tested, and easy to extend. When a new module needs a config class, registry, or logger — it inherits from your templates.

## Module Overview

This module provides reusable infrastructure patterns used across all AlphaGalerkin modules: Pydantic base configuration classes with deterministic hashing, thread-safe singleton registries with decorator-based registration, structured logging with context binding and timing utilities, base executable classes with lifecycle management, and CLI utilities with common options.

## Design Patterns

### 1. Template Method (BaseExecutable)
`BaseExecutable.run()` wraps `execute()` with lifecycle management:
```
run() → validate_config() → execute() → _create_result()
```
Subclasses only implement `execute()`. Error handling, timing, and result creation are automatic.

### 2. Generic Singleton Registry
```python
Registry, register = create_registry("Name", BaseClass)

@register("impl")
class Impl(BaseClass): ...

instance = Registry().get_instance("impl")
```
- Thread-safe with double-check locking
- Decorator-based registration at class definition
- `get()`, `get_or_raise()`, `get_instance()`, `list_items()`, `clear()`

### 3. Pydantic Configuration Hierarchy
```
BaseModuleConfig
  ├── TrainableModuleConfig (adds lr, batch_size, warmup)
  └── [All module-specific configs]
```
- Deterministic SHA256 hashing via `compute_hash()`
- YAML serialization via `to_yaml_dict()`
- Override creation via `with_overrides(**kwargs)`
- Factory: `create_config_class(name, base, **fields)`

### 4. Decorator Pattern (Logging, CLI)
- `@log_timing()`: Auto-logs function duration
- `@log_call()`: Logs function calls with args/results
- `@add_common_options`: Injects --verbose, --debug, --quiet flags
- `@handle_keyboard_interrupt`: Graceful Ctrl+C
- `@with_error_handling`: Try-catch with user-friendly messages

### 5. Factory Pattern
- `create_registry(name, base_class)` → (RegistryClass, decorator)
- `create_config_class(name, base, **fields)` → config class
- `create_logger_class(module_name)` → logger class
- `create_cli_app(name, help_text)` → Typer app

## Skills Required

- **Python metaclasses & generics**: Thread-safe singletons, generic type parameters
- **Pydantic v2**: Model validators, Field constraints, serialization, model_config
- **Threading**: `RLock`, double-check locking, thread-safe initialization
- **structlog**: Processor pipelines, context binding, output formatting
- **Typer/Rich**: CLI app construction, progress bars, formatted output
- **Abstract base classes**: ABC, Protocol, generic constraints

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Config Specialist** | `config.py` | Adding base config features, hash logic, serialization |
| **Registry Specialist** | `registry.py` | Modifying singleton behavior, thread safety, generics |
| **Logging Specialist** | `logging.py` | Adding log processors, output formats, context binding |
| **Executable Specialist** | `base.py` | Modifying lifecycle, result tracking, error handling |
| **CLI Specialist** | `cli.py` | Adding common options, output formatting, app construction |

## Tools & Commands

```bash
# Run template tests (107 tests)
pytest tests/templates/ -v

# Specific test areas
pytest tests/templates/test_config.py -v
pytest tests/templates/test_registry.py -v
pytest tests/templates/test_logging.py -v
pytest tests/templates/test_base.py -v
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `config.py` | Base Pydantic configuration | `BaseModuleConfig`, `TrainableModuleConfig`, `BoardSizeConfig`, `MetricDefinition`, `ThresholdOperator`, `create_config_class()` |
| `registry.py` | Thread-safe singleton registry | `BaseRegistry[T]`, `create_registry()`, `create_typed_registry()` |
| `logging.py` | Structured logging utilities | `BaseModuleLogger`, `create_logger_class()`, `@log_timing`, `@log_call`, `DebugContext` |
| `base.py` | Base executable classes | `BaseExecutable[T]`, `ExecutionResult`, `ExecutionStatus` |
| `cli.py` | CLI utilities | `create_cli_app()`, `@add_common_options`, `load_config_file()`, `print_result_table()` |

## Dependencies

**Internal**: None (foundational module — depended upon by all others)
**External**: `pydantic`, `structlog`, `typer` (optional), `rich` (optional), `yaml`, `hashlib`

## Conventions & Constraints

1. **No Module-Specific Logic**: Templates must remain generic. Never import from `src.modeling`, `src.games`, etc.
2. **Thread Safety Required**: All registries use `RLock`. Never access `_items` dict without locking.
3. **Deterministic Hashing**: `compute_hash()` must exclude volatile fields (timestamps, run IDs). Same config = same hash always.
4. **Backward Compatibility**: Changes to `BaseModuleConfig` affect every module. Test thoroughly.
5. **Optional Dependencies**: `typer` and `rich` are optional. CLI code must gracefully degrade without them.
6. **Test Coverage**: This module has 107 tests. All must pass before merging.

## Usage Guide

### Creating a New Module Config
```python
from src.templates.config import BaseModuleConfig
from pydantic import Field

class MyModuleConfig(BaseModuleConfig):
    param_a: int = Field(default=64, ge=1, le=1024, description="Parameter A")
    param_b: float = Field(default=0.01, gt=0, lt=1, description="Learning rate")

    @model_validator(mode="after")
    def validate_consistency(self) -> "MyModuleConfig":
        if self.param_a < 8:
            raise ValueError("param_a must be >= 8")
        return self
```

### Creating a New Registry
```python
from src.templates.registry import create_registry

class BaseHandler:
    def handle(self, data): raise NotImplementedError

HandlerRegistry, register_handler = create_registry("Handler", BaseHandler)

@register_handler("fast")
class FastHandler(BaseHandler):
    def handle(self, data): return process_fast(data)
```

### Creating a Logger
```python
from src.templates.logging import create_logger_class, configure_module_logging

configure_module_logging(level="INFO")
MyLogger = create_logger_class("MyModule")
logger = MyLogger("component_name", run_id="abc123")

with logger.timed("heavy_operation"):
    result = do_work()
logger.metric("accuracy", 0.95, epoch=10)
```
