---
name: python-standards
description: Prescriptive Python house standards covering Protocol-based dependency injection, Pydantic v2 configuration schemas, asyncio discipline, structlog structured logging, and strict typing/lint gates. Use when writing or reviewing any Python module, class, config, or async code — trigger phrases include "new module", "add a config", "wire a dependency", "add logging", "make this async", "fix mypy errors", or "does this meet standards".
---

# Python Standards

## Dependency Injection

- Define dependencies as `typing.Protocol` interfaces, not ABC inheritance. Implementations satisfy Protocols structurally — no base-class import coupling.
- Inject via constructor parameters typed against the Protocol. Never instantiate concrete collaborators inside a class.
- No service locators, no global registries reached into at call time, no module-level singletons as hidden dependencies.
- Default arguments may provide a production implementation only when it is stateless and side-effect free; otherwise require explicit injection.

## Configuration: Pydantic v2, No Hardcoded Values

- Every config schema is a Pydantic v2 `BaseModel`. Every tunable is a typed `Field` with a default and a `description`.
- No numeric or string literals with behavioral meaning inline in code. Surface them as either:
  - a named module constant (`UPPER_SNAKE_CASE`, with a comment stating what it controls), or
  - a config field (preferred when a user might tune it).
- Validate at the boundary: parse external input (YAML, JSON, env, CLI) into a model once; pass typed models inward, never raw dicts.
- Use `model_config = ConfigDict(...)`; set `extra="forbid"` for user-authored configs, `extra="ignore"` for forward-compatible persisted documents.
- Use `field_validator` / `model_validator` for cross-field invariants; raise `ValueError` with an actionable message.

## Asyncio Discipline

- No blocking calls (`time.sleep`, sync I/O, sync HTTP, CPU-heavy loops) inside `async def`. Use `asyncio.sleep`, async clients, or `asyncio.to_thread`.
- Every await on external I/O gets an explicit timeout (`asyncio.timeout(...)` or the client's timeout parameter). No unbounded awaits.
- Create tasks with `asyncio.TaskGroup` (or track and await them); never fire-and-forget with bare `create_task`.
- Do not mix event loops: no `asyncio.run` inside library code, only at entry points.

## Structured Logging: structlog

- Use `structlog.get_logger()`; bind stable context once (`logger = logger.bind(run_id=..., component=...)`) and reuse the bound logger.
- Event-style messages: a short snake_case event name plus key-value fields. `logger.info("checkpoint_saved", path=str(p), step=step)`.
- Never interpolate values into the message string — no f-strings, no `%` formatting in log calls. Values go in fields so they stay queryable.
- Log at the boundary of an operation (start/success/failure), not every line. Failures log `exc_info=True` or use `logger.exception`.

## Typing and Lint Gates

- Full type hints on all public functions, methods, and module-level variables. No implicit `Any`; avoid `Any` in signatures — use Protocols, `TypeVar`, or unions.
- Code must pass `mypy --strict` and `ruff check` clean. Run both before declaring work done. `# type: ignore` requires an error code and a justifying comment.
- Prefer `dataclass(frozen=True)` or Pydantic models over tuples/dicts for structured data crossing function boundaries.

See references/patterns.md for full worked examples of each pattern (Protocol + injection site, Pydantic config with validators, structlog setup, magic-number surfacing before/after).
