# AlphaGalerkin - Agentic Coding System Prompt Template

> **Design Principle:** This template treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: A resolution-independent Go AI using Continuous Operator Learning
(Galerkin Transformers & FNet) that enables zero-shot transfer between board sizes.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All unit tests pass: `pytest tests/math_kernel/ tests/poc/ -v`
- [ ] All integration tests pass: `pytest tests/integration/ -v`
- [ ] Type checking passes: `mypy src/ --strict`
- [ ] Linting passes: `ruff check src/`
- [ ] Zero-shot transfer: MSE < 0.05 on 19x19 when trained on 9x9
- [ ] LBB stability: β > 1e-6 throughout training
- [ ] FNet speedup: >1.5x over Softmax at 625 tokens
```

### 1.3 Problem Description (The "Three Paragraphs")

```
The system implements a neural operator that treats Go board positions as samples
from a continuous influence field, discretized at different resolutions. The core
innovation is Galerkin attention with Monte Carlo normalization (1/n), enabling
O(N) complexity and resolution-independent outputs. The tactical head uses standard
softmax attention to preserve injectivity for local reading sequences.

Data flows from board state → Fourier feature encoding → Galerkin body layers →
FNet mixing → Softmax tactical head → policy/value outputs. State maintained:
model weights, LBB constant monitoring, training metrics. The physics PoC uses
Poisson equation data as a supervised learning proxy for validating the core
mathematical claims before full Go integration.

Failure modes: LBB constant collapse (σ_min → 0), numerical instability in FFT
at large resolutions, transfer gap between physics proxy and actual Go strategy.
Invariants: (1) Output scale O(1) regardless of N due to 1/n normalization,
(2) LBB constant β > threshold for well-posedness, (3) Policy sums to 1.0.
```

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: Python 3.11+, PyTorch 2.0+
- Required Dependencies: pydantic, structlog, numpy, einops, hypothesis
- Security: No hardcoded secrets, all inputs validated via Pydantic
- Mathematical: dim(Key) >= dim(Query) for LBB condition
- Normalization: Must use 1/n Monte Carlo normalization (NOT 1/sqrt(d))
- Compatibility: Must run on CPU (GPU optional for performance)
```

### 2.2 Soft Constraints (Preferences)

```
- Style: Google Python Style Guide, 88 char line limit (black compatible)
- Architecture: Prefer composition over inheritance, use dataclasses/Pydantic
- Performance: Prefer einops for tensor operations, async I/O where applicable
- Testing: pytest + hypothesis for property-based tests, >80% coverage on math_kernel
- Logging: Use structlog for structured logging throughout
- Config: Use Pydantic models for all configuration, YAML for external config files
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure test organization if it improves clarity
- Add PyPI dependencies for testing/development (not core math)
- Refactor logging calls for consistency
- Choose batch sizes and learning rates for experiments
- Add new scenario types to the PoC framework
- Create helper functions for complex tensor operations
- Optimize critical paths with torch.compile or similar
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE:
- src/poc/           - PoC scenario framework (primary focus)
- src/modeling/      - Neural network architectures
- src/math_kernel/   - Mathematical primitives
- src/experiments/   - Experiment scripts
- src/physics/       - Physics data generation
- tests/             - All test files
- config/            - Configuration files
- docs/              - Documentation

OUT OF SCOPE:
- .git/              - Git internals
- outputs/           - Generated artifacts (read-only reference)
- checkpoints/       - Trained models (read-only reference)
- __pycache__/       - Python bytecode
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation/deletion within src/poc/, tests/poc/, config/scenarios/
- Running tests and linting
- Fixing type errors and lint warnings
- Adding new scenario configurations
- Refactoring for consistency within scope
- Creating documentation

CONFIRM FIRST (ask before proceeding):
- Changing mathematical operators (attention, integral) without tests
- Modifying existing scenario success criteria/thresholds
- Adding new external dependencies
- Changes affecting >5 files simultaneously
- Removing existing functionality

PROHIBITED (do not attempt):
- Committing directly to main branch
- Modifying .github/ workflows
- Changing fundamental mathematical claims without evidence
- Disabling or removing existing tests
- Hardcoding values that should be configurable
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5
- Max files to modify in single pass: 20
- Time-boxed exploration: Spend ≤10 minutes on research before asking
- Max test runtime for CI: 10 minutes total
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:
1. ruff check src/
2. mypy src/ --strict
3. pytest tests/poc/ -v --tb=short
4. pytest tests/math_kernel/ -v --tb=short
5. pytest tests/integration/ -v --tb=short (if time permits)
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
  → Run `ruff check --fix src/` for auto-fixable issues
  → Manually fix remaining issues
  → Re-run lint check

ON TYPE ERROR:
  → Analyze error message
  → Fix type annotations or add type: ignore with comment explaining why
  → Re-run mypy

ON TEST FAILURE:
  → Read failure output and traceback
  → Identify root cause (implementation bug vs test bug)
  → Fix implementation if test is correct
  → Fix test if test is incorrect (with justification)
  → Re-run failing test

ON REPEATED FAILURE (same error 3x):
  → Stop and report analysis
  → Include: error message, attempted fixes, hypothesis about root cause
  → Request human guidance
```

### 4.3 Success Verification

```
Before reporting completion:
1. All verification commands pass
2. No new warnings introduced
3. Coverage maintained or improved
4. New code has corresponding tests
5. CLAUDE.md updated if new commands/decisions made
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory (CLAUDE.md)

```markdown
# CLAUDE.md (already exists, update as needed)

## Build Commands
- ruff check src/: Lint Python code
- mypy src/ --strict: Type check
- pytest tests/ -v: Run all tests

## Test Commands
- pytest tests/poc/ -v: PoC framework tests
- pytest tests/math_kernel/ -v: Mathematical property tests
- python -m src.experiments.verify_transfer: Zero-shot transfer validation

## Architecture Decisions
- [2026-01-26]: Galerkin attention for O(N) complexity
- [2026-01-26]: FNet mixing for O(N log N) rollouts
- [2026-01-26]: Physics PoC as supervised learning proxy

## Known Issues
- (none currently)
```

### 5.2 Information to Preserve Across Sessions

```
- Build/test commands that work
- Scenario configurations and their results
- Architectural decisions and their rationale
- Mathematical invariants and their verification status
- Gotchas discovered during implementation
```

### 5.3 Information That Can Be Re-derived

```
- File structure (can be scanned with ls/find)
- Dependency versions (in pyproject.toml or requirements.txt)
- Current test status (can re-run pytest)
- Type error status (can re-run mypy)
```

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```
1. Read CLAUDE.md for project context
2. Run `ruff check src/` and `mypy src/` to establish baseline
3. Run `pytest tests/poc/ -v` to verify test infrastructure works
4. Identify entry points: src/poc/__init__.py, tests/poc/
```

### 6.2 Implementation Order

```
1. Understand existing patterns (read relevant files first)
2. Write or update Pydantic config models (if config changes needed)
3. Implement core logic with type hints
4. Add structured logging at key points
5. Write tests (unit first, then integration)
6. Run verification loop (lint → type → test)
7. Refactor if needed (keeping tests passing)
8. Update CLAUDE.md with new commands/decisions
```

### 6.3 Completion Checklist

```
□ All success criteria met
□ All verification commands pass
□ New code has docstrings
□ New public APIs have type hints
□ Tests cover happy path and edge cases
□ CLAUDE.md updated with new commands/decisions
□ Summary of changes provided
□ Known limitations documented
```

---

## SECTION 7: POC SCENARIO FRAMEWORK USAGE

### 7.1 Running Scenarios

```bash
# Run all registered scenarios
python -c "
from src.poc import ScenarioRunner
from src.poc.scenarios import *  # Register built-in scenarios
runner = ScenarioRunner()
runner.run_all()
"

# Run specific scenario
python -c "
from src.poc import ScenarioRunner
from src.poc.scenarios.transfer import TransferScenario
runner = ScenarioRunner()
runner.run('transfer', train_resolution=9, eval_resolutions=[9, 13, 19])
"

# Run from config file
python -c "
from src.poc import ScenarioRunner
from src.poc.scenarios import *
runner = ScenarioRunner()
runner.run_from_config('config/scenarios/poc_full.yaml')
"
```

### 7.2 Creating New Scenarios

```python
from src.poc import scenario, BaseScenario, ScenarioResult
from src.poc.config import BaseScenarioConfig, ScenarioStatus

class MyScenarioConfig(BaseScenarioConfig):
    """Configuration for my custom scenario."""
    name: str = "my_scenario"
    description: str = "Validates my custom claim"
    custom_param: int = 42

@scenario("my_scenario")
class MyScenario(BaseScenario):
    config_class = MyScenarioConfig

    def setup(self) -> None:
        # Initialize resources
        pass

    def execute(self) -> ScenarioResult:
        # Run validation logic
        self.record_metric("my_metric", 0.95)
        return self._create_result(ScenarioStatus.PASSED)

    def teardown(self) -> None:
        # Cleanup resources
        pass
```

### 7.3 Scenario Configuration Files

```yaml
# config/scenarios/transfer.yaml
name: transfer
description: Zero-shot transfer from 9x9 to 19x19
tier: integration
enabled: true

train_resolution: 9
eval_resolutions: [9, 13, 19]
primary_eval_resolution: 19

n_train_samples: 5000
n_eval_samples: 500
n_epochs: 100

mse_threshold: 0.05
```

---

## SECTION 8: TOOL SELECTION GUIDE

### 8.1 Sub-Agent Selection

| Task Type | Sub-Agent | When to Use |
|-----------|-----------|-------------|
| **Code exploration** | Explore | Finding files, understanding codebase structure |
| **Implementation planning** | Plan | Designing multi-file changes |
| **Git operations** | Bash | Commits, branches, status checks |
| **File search by pattern** | Glob | `**/*.py`, `src/**/*.yaml` |
| **Content search** | Grep | Finding usages, implementations |
| **Reading code** | Read | Understanding existing code |
| **Writing code** | Edit/Write | Implementing changes |
| **Running tests** | Bash | `pytest`, `mypy`, `ruff` |

### 8.2 Parallel vs Sequential Operations

```
PARALLEL (run simultaneously):
- Multiple independent file reads
- Lint + type check (if both needed)
- Independent test suites

SEQUENTIAL (run in order):
- Edit file → Run tests (dependency)
- Create file → Import it (dependency)
- Fix error → Verify fix (dependency)
```

---

## USAGE NOTES

1. **Don't over-specify implementation**: If you find yourself writing pseudocode in the problem description, you're constraining too much. Describe the *what* and *why*, not the *how*.

2. **Make success criteria executable**: "Works correctly" is not verifiable. "All tests pass" is. "Handles edge cases" is not verifiable. "Returns 400 on malformed input" is.

3. **Anti-constraints matter**: Explicitly stating what the agent CAN do is as important as stating what it can't. Unlisted permissions default to "ask first" which slows everything down.

4. **Calibrate autonomy to trust**: Start with more confirmations, expand autonomy as you verify the agent's judgment matches yours.

5. **The feedback loop is the objective function**: If you can't specify how to verify success, you can't prompt for it effectively. Write the test specification before the implementation prompt.

6. **Use the PoC framework**: All new validation scenarios should be implemented using the `src/poc/` framework for consistency and reproducibility.

7. **Configuration over code**: Prefer adding parameters to Pydantic configs over hardcoding values. This enables experimentation without code changes.
