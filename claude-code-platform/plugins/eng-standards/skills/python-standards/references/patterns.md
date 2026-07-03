# Python Standards — Worked Examples

## 1. Protocol + Implementation + Injection Site

```python
from typing import Protocol


class CheckpointStore(Protocol):
    """Structural interface — implementations need not import this."""

    def save(self, key: str, payload: bytes) -> None: ...
    def load(self, key: str) -> bytes: ...


class FilesystemCheckpointStore:
    """Satisfies CheckpointStore structurally; no inheritance."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, key: str, payload: bytes) -> None:
        (self._root / key).write_bytes(payload)

    def load(self, key: str) -> bytes:
        return (self._root / key).read_bytes()


class Trainer:
    """Constructor injection: depends on the Protocol, never a concrete class."""

    def __init__(self, store: CheckpointStore, config: TrainerConfig) -> None:
        self._store = store
        self._config = config
```

Anti-patterns this replaces:

```python
# BAD: concrete dependency constructed inside the class
class Trainer:
    def __init__(self) -> None:
        self._store = FilesystemCheckpointStore(Path("checkpoints"))  # hidden coupling

# BAD: service locator
class Trainer:
    def save(self) -> None:
        store = Registry.get("checkpoint_store")  # untyped, untestable
```

Tests inject a fake with zero mocking framework:

```python
class InMemoryStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def save(self, key: str, payload: bytes) -> None:
        self.data[key] = payload

    def load(self, key: str) -> bytes:
        return self.data[key]


def test_trainer_saves_checkpoint() -> None:
    store = InMemoryStore()
    trainer = Trainer(store=store, config=TrainerConfig())
    trainer.checkpoint("step_10")
    assert "step_10" in store.data
```

## 2. Pydantic v2 Config with Validators

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_retries: int = Field(
        default=3, ge=0, le=20,
        description="Maximum retry attempts before giving up.",
    )
    backoff_base_s: float = Field(
        default=0.5, gt=0.0,
        description="Base delay for exponential backoff, in seconds.",
    )
    timeout_s: float = Field(
        default=30.0, gt=0.0,
        description="Per-attempt timeout, in seconds.",
    )

    @field_validator("timeout_s")
    @classmethod
    def timeout_reasonable(cls, v: float) -> float:
        if v > 600.0:
            raise ValueError(f"timeout_s={v} exceeds 600s ceiling; split the operation")
        return v

    @model_validator(mode="after")
    def backoff_fits_timeout(self) -> "RetryConfig":
        worst_case = self.backoff_base_s * (2**self.max_retries)
        if worst_case > self.timeout_s * (self.max_retries + 1):
            raise ValueError("backoff schedule cannot fit inside total timeout budget")
        return self
```

Boundary parsing — once, at the edge:

```python
def load_config(path: Path) -> RetryConfig:
    raw = yaml.safe_load(path.read_text())
    return RetryConfig.model_validate(raw)  # everything inward is typed
```

## 3. structlog Setup with Processors

```python
import structlog


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level)
        ),
    )


# Bind context once; reuse the bound logger.
logger = structlog.get_logger().bind(component="trainer", run_id=run_id)

logger.info("training_started", n_epochs=config.n_epochs, device=device)
try:
    result = train(...)
    logger.info("training_finished", final_loss=result.loss, wallclock_s=result.wallclock_s)
except Exception:
    logger.exception("training_failed")
    raise
```

Anti-patterns:

```python
# BAD: value interpolated into the message — not queryable, breaks aggregation
logger.info(f"Training finished with loss {result.loss}")

# BAD: rebinding per call / passing context in the message
logger.info("finished run " + run_id)
```

## 4. Surfacing a Magic Number — Before / After

Before:

```python
def project_to_surface(self, points: Tensor) -> Tensor:
    for _ in range(24):                      # what is 24?
        grad = self._gradient(points, 1e-6)  # what is 1e-6?
        points = points - self._sdf(points) * grad
    return points
```

After:

```python
# Newton projection: iteration cap and finite-difference step for SDF gradients.
DEFAULT_NEWTON_MAX_ITERS = 24
DEFAULT_GRADIENT_EPS = 1e-6


class SurfaceProjectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    newton_max_iters: int = Field(
        default=DEFAULT_NEWTON_MAX_ITERS, ge=1,
        description="Maximum Newton iterations for surface projection.",
    )
    gradient_eps: float = Field(
        default=DEFAULT_GRADIENT_EPS, gt=0.0,
        description="Finite-difference step for SDF gradient estimation.",
    )


def project_to_surface(self, points: Tensor) -> Tensor:
    for _ in range(self._config.newton_max_iters):
        grad = self._gradient(points, self._config.gradient_eps)
        points = points - self._sdf(points) * grad
    return points
```

Rule of thumb: if a reviewer could ask "why this value?", it must be a named constant; if a user could reasonably tune it, it must also be a config field defaulting to that constant.
