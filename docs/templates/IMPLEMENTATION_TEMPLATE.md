# Agentic Coding System Prompt Template for AlphaGalerkin

> **Design Principle:** This template treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

> **AlphaGalerkin Integration:** All templates leverage existing infrastructure patterns from the codebase for consistency and reusability.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: [System type in one sentence]

Example: I am building a curriculum learning system that progressively trains on increasing board sizes.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All unit tests pass: `pytest tests/[module]/ -v`
- [ ] Type checking passes: `mypy src/[module]/ --strict`
- [ ] Linting passes: `ruff check src/[module]/`
- [ ] Integration tests pass: `pytest tests/integration/ -v -k [feature]`
- [ ] [Domain-specific criterion: e.g., "Zero-shot transfer MSE < 0.05"]
- [ ] [Performance criterion: e.g., "Throughput >= 1000 games/hour"]
- [ ] C4 architecture documentation exists in `docs/architecture/`
```

### 1.3 Problem Description (The "Three Paragraphs")

```
[Paragraph 1: What problem does this solve? What's the core coordination logic?]
Example: The curriculum learning system addresses the challenge of efficiently training
a resolution-independent Go AI by starting with smaller board sizes (9x9) where learning
is faster, then progressively transferring to larger boards (13x13, 19x19).

[Paragraph 2: What are the key data flows? What state must be maintained and synchronized?]
Example: Training data flows from self-play through the replay buffer. The curriculum
scheduler maintains state about current difficulty level, performance metrics per level,
and triggers for progression. Model checkpoints are synchronized with curriculum stage.

[Paragraph 3: What are the failure modes? What invariants must hold under adversarial conditions?]
Example: Failure modes include premature progression (insufficient mastery), catastrophic
forgetting when scaling up, and metric manipulation. Invariants: win rate against baseline
must meet threshold before progression; model must maintain performance on previous sizes.
```

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
AlphaGalerkin Standard Constraints:
- Language/Runtime: Python 3.11+
- Required Dependencies:
  - PyTorch 2.0+ (neural network operations)
  - Pydantic v2 (configuration validation)
  - structlog (structured logging)
  - pytest (testing)
  - einops (tensor operations)
- Security: No hardcoded secrets, all inputs validated via Pydantic
- Compatibility: Must run on both CPU and CUDA, support variable board sizes

Module-Specific Constraints:
- [e.g., Must integrate with existing ModelZoo in src/distributed/model_zoo.py]
- [e.g., Must use GameInterface from src/games/interface.py]
- [e.g., Configuration must extend BaseScenarioConfig pattern]
```

### 2.2 Soft Constraints (Preferences)

```
AlphaGalerkin Style Guide:
- Use Google Style Guide for Python
- 100 character line limit (enforced by ruff)
- Type hints required for all public functions
- Docstrings required for all public classes and functions
- Prefer composition over inheritance
- Use dataclasses/Pydantic models over raw dicts
- Use einops for tensor dimension manipulation
- Prefer async I/O for external operations

Testing Requirements:
- pytest with fixtures for all tests
- >80% coverage on core logic
- Property-based tests for mathematical operations
- Integration tests for end-to-end flows
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure existing file organization within the module
- Add new dependencies via `pip install` (document in pyproject.toml)
- Refactor adjacent code for consistency
- Create new configuration schemas extending base patterns
- Add new scenarios to the PoC framework
- Modify test fixtures in module-specific conftest.py
- Choose implementation patterns not specified (document in ADR)
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE (modify freely):
- src/[new_module]/          # New module directory
- tests/[new_module]/        # Module tests
- config/[module].yaml       # Module configuration
- docs/architecture/         # C4 documentation

EXTEND ONLY (add, don't break existing):
- src/poc/registry.py        # Add new scenario registration
- src/games/registry.py      # Add new game registration
- config/scenarios/          # Add new scenario configs
- tests/conftest.py          # Add shared fixtures

OUT OF SCOPE (do not modify):
- src/math_kernel/           # Core mathematical operators
- src/modeling/galerkin.py   # Core Galerkin attention
- vendor/ or third_party/    # External code
- .github/                   # CI/CD configuration
- Files marked with # DO NOT MODIFY
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation/deletion within IN SCOPE directories
- Dependency installation (add to pyproject.toml)
- Running tests: pytest, mypy, ruff
- Creating new Pydantic config schemas
- Adding registry entries via decorators
- Writing C4 documentation
- Refactoring for consistency within module

CONFIRM FIRST (ask before proceeding):
- Architectural changes affecting >3 existing modules
- Breaking API changes to public interfaces
- Deletions of >100 lines from existing files
- Changes to core mathematical operators
- Modifications to training loss functions

PROHIBITED (do not attempt):
- Commits to main branch directly
- External API calls with side effects in production
- Modifications outside IN SCOPE/EXTEND ONLY
- Hardcoding configuration values
- Skipping test implementation
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5
- Max files to modify in single pass: 20
- Max new files to create: 15
- Time-boxed exploration: spend ≤10 min on research before asking
- Test coverage minimum: 80% for new code
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:

# 1. Lint check
ruff check src/[module]/

# 2. Type check (strict mode)
mypy src/[module]/ --strict

# 3. Unit tests for new module
pytest tests/[module]/ -v

# 4. Integration tests
pytest tests/integration/ -v -k [feature]

# 5. Full test suite (ensure no regressions)
pytest tests/ -v --tb=short

# 6. Coverage report
pytest tests/[module]/ --cov=src/[module] --cov-report=term-missing
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
  → Run `ruff check src/[module]/ --fix` for auto-fixable issues
  → Manually fix remaining issues
  → Re-run lint check

ON TYPE ERROR:
  → Analyze error message carefully
  → Check if type stubs needed (add to pyproject.toml)
  → Fix type annotations
  → Re-run type check

ON TEST FAILURE:
  → Read full failure output including traceback
  → Identify root cause (implementation bug vs test bug)
  → Fix implementation first (tests are specification)
  → If test is clearly wrong, fix test with explanation
  → Re-run failed test in isolation: pytest tests/path/test_file.py::test_name -v

ON REPEATED FAILURE (same error 3x):
  → Stop and document analysis
  → List hypotheses for root cause
  → Request human guidance with context

ON IMPORT ERROR:
  → Check if dependency installed: pip show [package]
  → Install if missing: pip install [package]
  → Add to pyproject.toml dependencies
  → Re-run
```

### 4.3 Success Verification

```
Before reporting completion:
1. All verification commands pass (lint, type, test)
2. Coverage meets minimum (80% for new code)
3. C4 architecture documentation exists
4. CLAUDE.md updated with new commands/decisions
5. Manual smoke test if applicable:
   python -c "from src.[module] import [main_class]; print([main_class])"
6. Generate brief summary of changes made
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory (CLAUDE.md Updates)

```markdown
# Additions to CLAUDE.md after implementation

## [Module Name] Commands
```bash
# [Command description]
python -m src.[module].[entry] [args]

# Test command
pytest tests/[module]/ -v
```

## Architecture Decisions
- [YYYY-MM-DD]: [Decision]: [Rationale]

## Known Issues
- [Issue description]: [Workaround if any]
```

### 5.2 Information to Preserve Across Sessions

```
MUST PRESERVE:
- Build/test commands that work
- Non-obvious environment setup steps
- Architectural decisions and their rationale (ADRs)
- Gotchas discovered during implementation
- Configuration schema changes
- Registry entries added

CAN RE-DERIVE:
- File structure (can be scanned)
- Dependency versions (in pyproject.toml)
- Current test status (can re-run)
- Type annotations (can infer from code)
```

### 5.3 Documentation Requirements

```
REQUIRED for each new module:
1. Module docstring in __init__.py
2. Class/function docstrings (Google style)
3. C4 architecture update in docs/architecture/
4. Configuration schema documentation
5. Usage examples in docstrings or README
```

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```bash
# 1. Read project context
cat CLAUDE.md

# 2. Understand existing structure
find src/ -type f -name "*.py" | head -50

# 3. Check existing patterns in similar modules
ls -la src/poc/
ls -la src/games/

# 4. Run existing tests to establish baseline
pytest tests/ -v --tb=short -q

# 5. Identify entry points and dependencies
grep -r "def main" src/ | head -10
```

### 6.2 Implementation Order

```
1. UNDERSTAND: Read existing patterns before writing
   - Review similar modules (src/poc/, src/games/)
   - Understand configuration patterns (src/poc/config.py)
   - Review registry patterns (src/poc/registry.py)
   - Check logging patterns (src/poc/logging.py)

2. DESIGN: Create C4 architecture first
   - Level 1: System context (how module fits)
   - Level 2: Container view (internal components)
   - Level 3: Component details
   - Level 4: Code-level decisions (ADRs)

3. CONFIGURE: Create Pydantic configuration schemas
   - Extend BaseScenarioConfig or similar base
   - Define all parameters with Field() and constraints
   - No hardcoded values - all from config

4. IMPLEMENT: Core logic (smallest working version)
   - Start with interface/protocol definitions
   - Implement core business logic
   - Add error handling with structured logging
   - Use dependency injection for testability

5. REGISTER: Add to appropriate registry
   - Use decorator pattern: @scenario(), @register_game()
   - Ensure automatic discovery works

6. TEST: Write comprehensive tests
   - Unit tests for each component
   - Integration tests for workflows
   - Property-based tests for mathematical operations
   - Fixture-based test isolation

7. VERIFY: Run full verification loop
   - Lint → Type → Unit → Integration → Coverage

8. DOCUMENT: Update all documentation
   - CLAUDE.md with new commands
   - C4 architecture diagrams
   - Configuration schema docs

9. REFACTOR: Clean up if needed
   - Remove dead code
   - Improve naming
   - Add missing type hints
```

### 6.3 Completion Checklist

```
□ All success criteria from Section 1.2 met
□ All verification commands pass (Section 4.1)
□ No hardcoded values - all configurable
□ Configuration uses Pydantic with validation
□ Structured logging implemented
□ Registry integration complete
□ C4 architecture documentation exists
□ CLAUDE.md updated with new commands/decisions
□ Test coverage ≥80% for new code
□ Summary of changes provided
□ Known limitations documented
```

---

## SECTION 7: REUSABLE COMPONENT TEMPLATES

### 7.1 Configuration Template

```python
# Use: src/[module]/config.py
# Pattern: Extend from base, add domain-specific fields

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal
from enum import Enum

class [Module]Status(str, Enum):
    """Status enumeration for [module]."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class [Module]Config(BaseModel):
    """Configuration for [module].

    All parameters must have:
    - Type annotations
    - Field() with description
    - Sensible defaults (no hardcoding in code)
    - Constraints where applicable (ge, le, gt, lt)
    """

    model_config = ConfigDict(
        extra="forbid",  # Catch typos
        validate_assignment=True,  # Re-validate on change
    )

    # Required fields (no default)
    name: str = Field(..., description="Unique identifier")

    # Optional fields with defaults
    param_int: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Integer parameter with bounds",
    )
    param_float: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Float parameter with bounds",
    )
    param_list: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="List parameter with factory default",
    )

    @field_validator("param_list")
    @classmethod
    def validate_param_list(cls, v: list[int]) -> list[int]:
        """Validate and normalize list."""
        if not v:
            raise ValueError("param_list cannot be empty")
        return sorted(set(v))
```

### 7.2 Registry Template

```python
# Use: src/[module]/registry.py
# Pattern: Thread-safe singleton with decorator registration

from __future__ import annotations
import threading
from typing import TypeVar, Callable, TYPE_CHECKING
import structlog

if TYPE_CHECKING:
    from .[base] import Base[Module]

logger = structlog.get_logger(__name__)
T = TypeVar("T", bound="Base[Module]")

class [Module]Registry:
    """Thread-safe singleton registry for [module] components."""

    _instance: [Module]Registry | None = None
    _lock: threading.Lock = threading.Lock()
    _items: dict[str, type[Base[Module]]]

    def __new__(cls) -> [Module]Registry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._items = {}
        return cls._instance

    def register(self, name: str, item_cls: type[Base[Module]]) -> None:
        """Register item with validation."""
        if not name or not name.strip():
            raise ValueError("Name cannot be empty")

        with self._lock:
            if name in self._items:
                raise ValueError(f"'{name}' already registered")
            self._items[name] = item_cls
            logger.debug("item_registered", name=name, cls=item_cls.__name__)

    def get(self, name: str) -> type[Base[Module]] | None:
        """Thread-safe lookup."""
        with self._lock:
            return self._items.get(name)

    def list_items(self) -> list[str]:
        """Return copy of registered names."""
        with self._lock:
            return list(self._items.keys())

    def clear(self) -> None:
        """Clear registry (for testing only)."""
        with self._lock:
            logger.warning("registry_cleared", count=len(self._items))
            self._items.clear()


def register_[module](name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a [module] class."""
    def decorator(cls: type[T]) -> type[T]:
        [Module]Registry().register(name, cls)
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls
    return decorator
```

### 7.3 Logging Template

```python
# Use: src/[module]/logging.py
# Pattern: Context-aware logger with timing support

from __future__ import annotations
import time
import functools
from contextlib import contextmanager
from typing import Any, Generator, Callable, TypeVar, ParamSpec
import structlog

P = ParamSpec("P")
R = TypeVar("R")

class [Module]Logger:
    """Context-aware logger for [module]."""

    def __init__(
        self,
        name: str,
        run_id: str | None = None,
        **context: Any,
    ) -> None:
        self._base = structlog.get_logger(__name__)
        self._context = {
            "module": name,
            **({"run_id": run_id} if run_id else {}),
            **context,
        }
        self._logger = self._base.bind(**self._context)

    def bind(self, **context: Any) -> [Module]Logger:
        """Create new logger with additional context."""
        new_logger = [Module]Logger.__new__([Module]Logger)
        new_logger._base = self._base
        new_logger._context = {**self._context, **context}
        new_logger._logger = self._base.bind(**new_logger._context)
        return new_logger

    def info(self, event: str, **kw: Any) -> None:
        self._logger.info(event, **kw)

    def debug(self, event: str, **kw: Any) -> None:
        self._logger.debug(event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._logger.warning(event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._logger.error(event, **kw)

    def metric(self, name: str, value: float, **tags: Any) -> None:
        """Log a metric with tags."""
        self._logger.info("metric", metric_name=name, metric_value=value, **tags)

    @contextmanager
    def timed(self, operation: str) -> Generator[dict[str, float], None, None]:
        """Context manager for timing operations."""
        timing: dict[str, float] = {}
        start = time.perf_counter()
        self._logger.debug(f"{operation}_start")
        try:
            yield timing
        finally:
            duration = time.perf_counter() - start
            timing["duration_seconds"] = duration
            self._logger.debug(f"{operation}_complete", duration_seconds=duration)


def log_timing(
    logger: structlog.stdlib.BoundLogger | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function execution time."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal logger
        if logger is None:
            logger = structlog.get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                logger.debug(  # type: ignore[union-attr]
                    "function_timing",
                    function=func.__name__,
                    duration_seconds=duration,
                )
        return wrapper
    return decorator
```

### 7.4 Test Template

```python
# Use: tests/[module]/test_[component].py
# Pattern: Class-based organization with fixtures

from __future__ import annotations
import pytest
from pydantic import ValidationError

from src.[module].config import [Module]Config
from src.[module].registry import [Module]Registry, register_[module]
from src.[module].[base] import Base[Module]


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clean registry before each test."""
    [Module]Registry().clear()


class Test[Module]Config:
    """Tests for [Module]Config."""

    def test_default_values(self) -> None:
        """Test that defaults are applied."""
        config = [Module]Config(name="test")
        assert config.param_int == 100
        assert config.param_float == 0.01

    def test_required_fields(self) -> None:
        """Test that required fields raise on missing."""
        with pytest.raises(ValidationError):
            [Module]Config()  # type: ignore[call-arg]

    def test_constraint_validation(self) -> None:
        """Test Field constraints are enforced."""
        with pytest.raises(ValidationError):
            [Module]Config(name="test", param_int=0)  # ge=1

    def test_extra_fields_forbidden(self) -> None:
        """Test extra fields raise errors."""
        with pytest.raises(ValidationError):
            [Module]Config(name="test", unknown="value")  # type: ignore[call-arg]


class Test[Module]Registry:
    """Tests for [Module]Registry."""

    def test_singleton(self) -> None:
        """Test registry is a singleton."""
        reg1 = [Module]Registry()
        reg2 = [Module]Registry()
        assert reg1 is reg2

    def test_register_and_get(self) -> None:
        """Test registration and retrieval."""
        registry = [Module]Registry()

        class Test[Module](Base[Module]):
            pass

        registry.register("test", Test[Module])
        assert registry.get("test") is Test[Module]

    def test_duplicate_raises(self) -> None:
        """Test duplicate registration raises."""
        registry = [Module]Registry()

        class Test[Module](Base[Module]):
            pass

        registry.register("dup", Test[Module])
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup", Test[Module])


class TestRegisterDecorator:
    """Tests for @register_[module] decorator."""

    def test_decorator_registers(self) -> None:
        """Test decorator performs registration."""

        @register_[module]("decorated")
        class Decorated[Module](Base[Module]):
            pass

        assert [Module]Registry().get("decorated") is Decorated[Module]
```

---

## SECTION 8: C4 ARCHITECTURE TEMPLATE

See `docs/templates/C4_TEMPLATE.md` for the full C4 architecture template with Mermaid diagrams.

---

## SECTION 9: INSTANTIATION EXAMPLES

### Example 1: Curriculum Learning System

```markdown
## OBJECTIVE FUNCTION

### System Intent
I am building a curriculum learning system that progressively trains the AlphaGalerkin
model on increasing board sizes (9x9 → 13x13 → 19x19) to accelerate convergence.

### Success Criteria
- [ ] All tests pass: `pytest tests/curriculum/ -v`
- [ ] Type checking: `mypy src/curriculum/ --strict`
- [ ] Win rate ≥ 60% against random baseline before progression
- [ ] No catastrophic forgetting (maintain ≥ 50% on previous sizes)
- [ ] Training time reduced by ≥ 30% compared to direct 19x19 training

### Problem Description
The curriculum system orchestrates training across multiple board sizes. It starts
with 9x9 games where episodes are short and learning signal is dense, then progresses
to larger boards as competency is demonstrated.

Data flows from self-play → replay buffer → trainer → checkpoint. The curriculum
scheduler maintains: current stage, performance metrics per stage, progression
thresholds, and model checkpoints per stage. The ModelZoo provides opponent selection.

Failure modes: premature progression (model not ready), catastrophic forgetting
(loses ability on smaller boards), oscillation (bouncing between stages).
Invariants: win rate against stage-appropriate baseline must exceed threshold;
performance on all previous stages must remain above minimum.

## FEASIBLE REGION

### Hard Constraints
- Must integrate with existing Trainer in src/training/trainer.py
- Must use ModelZoo for curriculum opponent selection
- Configuration via Pydantic extending existing patterns
- All board size parameters from config (no hardcoding)

### Soft Constraints
- Prefer composition: CurriculumScheduler wraps Trainer
- Use existing replay buffer infrastructure
- Log all stage transitions with structured logging

## PERMISSION ARCHITECTURE

### Scope
IN SCOPE: src/curriculum/, tests/curriculum/, config/curriculum.yaml
EXTEND: src/training/trainer.py (add hooks), src/distributed/model_zoo.py

## VERIFICATION

### Commands
pytest tests/curriculum/ -v
mypy src/curriculum/ --strict
python -m src.curriculum.cli --dry-run  # Smoke test
```

### Example 2: SGF Game Analysis

```markdown
## OBJECTIVE FUNCTION

### System Intent
I am building an SGF (Smart Game Format) parsing and analysis system for reviewing
professional Go games with AlphaGalerkin evaluation.

### Success Criteria
- [ ] Parse 100% of standard SGF files from GoGoD database
- [ ] Round-trip SGF → internal → SGF preserves all data
- [ ] Position evaluation produces policy + value in < 100ms
- [ ] Export analysis results to annotated SGF

### Problem Description
The SGF system provides import/export of Go game records in the standard format.
It enables analysis of professional games by stepping through positions and
evaluating each with the neural network.

Data flows from SGF file → parser → GameState sequence → evaluation → annotated SGF.
State includes: game tree (with variations), move comments, evaluation annotations,
and board position at each node.

Failure modes: malformed SGF (graceful degradation), encoding issues (UTF-8 handling),
large game trees (memory limits). Invariants: move sequence must be legal according
to Go rules; evaluation annotations must correspond to evaluated position.
```

---

## USAGE NOTES

1. **Start with C4 Architecture**: Design the system context and components before coding. This catches structural issues early.

2. **Configuration First**: Define all parameters in Pydantic configs before implementing logic. This prevents hardcoding.

3. **Test-Driven for Math**: For mathematical operations, write property-based tests first (e.g., "attention output same resolution as input").

4. **Registry Pattern for Extensibility**: Use the registry pattern for any component that might have multiple implementations.

5. **Structured Logging Throughout**: Add logging at entry/exit of major functions and around external calls.

6. **Verify Incrementally**: Run the verification loop after each significant change, not just at the end.

7. **Document Decisions**: When you make a non-obvious choice, add an ADR to the C4 documentation.
