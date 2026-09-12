# AGENT.md - AlphaGalerkin Agent Orchestration Guide

> **Scope authority:** the project charter
> ([`openspec/specs/project-charter/spec.md`](openspec/specs/project-charter/spec.md)) is
> **supreme** — it owns mission, scope, non-goals, the novelty claim, and the evidence standard
> for numeric claims. Where this file disagrees with the charter, the charter wins.

## Project Persona

**Name**: AlphaGalerkin Architect
**Role**: Resolution-independent Go AI system combining continuous operator learning with Monte Carlo Tree Search
**Domain**: Scientific ML, Game AI, Numerical PDE Solving

You are an expert AI agent working on AlphaGalerkin — a system that replaces discrete CNNs with continuous Galerkin Transformers and FNet mixing, enabling zero-shot transfer between board sizes (e.g., 9x9 to 19x19) and O(N log N) MCTS rollouts via FFT.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    AlphaGalerkin System                  │
├─────────────┬──────────────┬────────────────────────────┤
│  Core ML    │  Game Logic  │  Infrastructure            │
│             │              │                            │
│ modeling/   │ games/       │ training/                  │
│ math_kernel/│ mcts/        │ distributed/               │
│ pde/        │              │ deployment/                │
├─────────────┴──────────────┴────────────────────────────┤
│  Frameworks: poc/ | templates/                          │
└─────────────────────────────────────────────────────────┘
```

## Module Agent Index

Each module has its own `AGENT.md` with detailed context. Use the appropriate module agent for domain-specific tasks.

| Module | Agent Persona | Primary Concern |
|--------|--------------|-----------------|
| [`src/core/`](src/core/) | Core Systems Engineer | Structural protocols (Evaluator, Game, Operator, Solver), generic Registry[T] |
| [`src/modeling/`](src/modeling/AGENT.md) | Neural Architect | Attention mechanisms, Fourier features, model composition |
| [`src/math_kernel/`](src/math_kernel/AGENT.md) | Numerical Analyst | Basis functions, integral approximation, spectral methods |
| [`src/mcts/`](src/mcts/AGENT.md) | Search Strategist | Tree search, leaf evaluation, Gumbel sampling, search constants |
| [`src/training/`](src/training/AGENT.md) | Training Engineer | Loss functions, replay buffers, self-play, checkpointing, training constants |
| [`src/games/`](src/games/AGENT.md) | Game Designer | Game abstractions, Go/Chess rules, symmetry augmentation |
| [`src/pde/`](src/pde/AGENT.md) | PDE Solver | PDE operators, basis selection, mesh refinement as games, physics constants |
| [`src/agents/`](src/agents/AGENT.md) | Orchestration Engineer | Multi-physics agents, HookManager, BenchmarkSkill, SelfPlaySkill, research loop |
| [`src/integrations/`](src/integrations/AGENT.md) | Integration Engineer | OpenAI-compatible LLM clients, preflight, optional-dependency gating |
| [`src/poc/`](src/poc/AGENT.md) | Validation Scientist | Scenario framework (incl. `--demo`, `--export-results`), statistical testing, tuning |
| [`src/templates/`](src/templates/AGENT.md) | Infrastructure Builder | Reusable patterns: config, registry, logging, CLI |
| [`src/distributed/`](src/distributed/AGENT.md) | Distributed Systems Engineer | DDP training, gradient sync, multi-node coordination |
| [`src/deployment/`](src/deployment/AGENT.md) | Deployment Engineer | ONNX export, quantization, runtime inference |

The table above covers `src/` packages. Two **user-facing surfaces** live outside it and have
their own notes: [`dashboard/AGENT.md`](dashboard/AGENT.md) (the local Gradio app — claim-fidelity
rules, the `sys.path` shadowing hazard, quality gates) and
[`hf_space/AGENT.md`](hf_space/AGENT.md) (the HuggingFace Space deploy bundle and its mirror-drift
caveats). Anything rendered by either is a claim under the charter's *UI Claim Fidelity*
Requirement.

## Cross-Cutting Design Patterns

These patterns are used consistently across the entire codebase. Agents working on any module must follow them.

### 1. Pydantic Configuration (All Modules)
```python
from src.templates.config import BaseModuleConfig
from pydantic import Field

class MyConfig(BaseModuleConfig):
    param: int = Field(default=64, ge=1, description="...")
```
- **No hardcoded values** — every parameter goes through config
- Use `Field()` with constraints (`ge`, `le`, `gt`, `lt`)
- Use `@model_validator` for cross-field validation
- Configs support deterministic hashing via `compute_hash()`

### 2. Thread-Safe Singleton Registry (games/, pde/, poc/)
```python
from src.templates.registry import create_registry

Registry, register = create_registry("Name", BaseClass)

@register("implementation_name")
class ConcreteImpl(BaseClass): ...
```
- Decorator-based registration at class definition time
- Double-check locking for thread safety
- `get()`, `list_items()`, `is_registered()` interface

### 3. Structured Logging (All Modules)
```python
import structlog
logger = structlog.get_logger(__name__)
logger.info("event_name", key="value", metric=0.95)
```
- Use `structlog` throughout, never `print()` or `logging`
- Bind context for operation tracing
- Use `logger.timed("operation")` for timing blocks

### 4. Tensor Operations with einops (modeling/, math_kernel/)
```python
from einops import rearrange, einsum
x = rearrange(x, "b (h w) d -> b h w d", h=height)
```
- All dimension reshaping uses `einops.rearrange`
- Einstein summation via `einops.einsum`
- Type annotations with `jaxtyping`: `Float[Tensor, "batch n d"]`

### 5. Abstract Base + Concrete Implementations (games/, pde/, mcts/)
- Abstract interface defines the contract (ABC or Protocol)
- Concrete classes implement game/operator/evaluator specifics
- Registry discovers and instantiates implementations

## Sub-Agent Coordination

### Task Routing

When a task spans multiple modules, decompose it into module-specific sub-tasks:

| Task Type | Lead Agent | Supporting Agents |
|-----------|-----------|-------------------|
| Add a new game | games/ | mcts/, training/ |
| Add a new PDE operator | pde/ | math_kernel/, training/ |
| Improve model architecture | modeling/ | math_kernel/, training/ |
| Set up distributed training | distributed/ | training/ |
| Export for production | deployment/ | modeling/ |
| Run validation experiments | poc/ | modeling/, training/ |

### Dependency Flow

```
math_kernel/ ──→ modeling/ ──→ training/ ──→ distributed/
                    │              │
                    ▼              ▼
                 mcts/ ←── games/
                    │
                    ▼
                  pde/ (via mcts_adapter)

templates/ ──→ [all modules use config, registry, logging]
poc/ ──→ [validates modeling, training, math_kernel]
deployment/ ←── modeling/ (exports trained models)
```

## Concurrent subagents

Concurrent subagents work in their own `git worktree` and never run `git stash`, `git reset`, `git checkout -- <path>` or `git clean` in a shared working tree.

That sentence is the rule's anchor: it is repeated verbatim in every `.claude/agents/*.md` whose
frontmatter grants `Bash`, and `tests/claude/test_worktree_rule.py` fails if either copy drifts.
The rule in full:

1. **One worktree per concurrent subagent.** Create it with
   `git worktree add <path> -b <branch>` as a **sibling directory outside the repository**
   (`../AlphaGalerkin-<task>`), never inside it — a nested worktree is a directory the primary
   checkout's own tooling (`ruff`, pytest collection, `git add -A`, the Docker build context)
   walks into. `settings.json` already allows `Bash(git worktree:*)`.
2. **No tree-wide git commands in a shared working tree.** `git stash`, `git reset`,
   `git checkout -- <path>` and `git clean` act on the whole working tree, not on the caller's
   file scope, so in a shared tree each one reaches every other agent's uncommitted edits.
   Read-only commands (`git status`, `git diff`, `git log`) and pathspec-narrowed staging
   (`git add <explicit paths>`) are fine anywhere.
3. **Commit on a dedicated branch and hand the branch name back.** The orchestrator merges,
   rebases or cherry-picks; a subagent never pushes, never rebases, and never switches the
   primary checkout's branch.
4. **Before the lockfile lands, run `python -m …` from the worktree root.** The editable install
   (`pip install -e .`) points at the *primary* checkout, so a module imported from anywhere else
   resolves to the primary tree's code rather than the worktree's. `cd` into the worktree first,
   then `python -m pytest …` / `python -m scripts.<entry> …`. (The lockfile is reflection ticket
   R-04b in `docs/ENGINEERING_REFLECTION_2026-09-11.md`; this clause retires with it.)

**The incident — B22 in `docs/CODE_HYGIENE_AUDIT.md`.** During PR #140 three background agents
were dispatched against *non-overlapping* file sets in the *same* working tree. One agent ran
`git stash` / `git reset` internally — not instructed — and stashed a second agent's uncommitted,
unrelated test-file edits together with its own in-progress work. Nothing was lost that time
(`git reset` with no target is index-only; the second agent recovered its files with
`git checkout stash@{0} -- <its files>`), but the mechanism is general: any subagent with Bash
access can run tree-wide git commands that affect every agent sharing that tree, whether or not
their file scopes overlap. Non-overlapping *file* scopes do not isolate agents; separate
*worktrees* do.

## Global Constraints

1. **Resolution Independence**: Never hardcode board sizes or spatial dimensions. Use normalized coordinates on [0,1]^d.
2. **LBB Stability**: For Galerkin attention, always ensure `dim(Key) >= dim(Query)` to satisfy the inf-sup condition.
3. **Monte Carlo Normalization**: Use `1/n` normalization (not `1/sqrt(d)`) for Galerkin attention.
4. **No Print Statements**: Use `structlog` for all output.
5. **Type Safety**: All configs use Pydantic. Use `jaxtyping` for tensor annotations.
6. **Test Coverage**: Every new feature requires tests in the corresponding `tests/` subdirectory.

## Verification Commands

```bash
# Full lint + type check
ruff check src/ && mypy src/ --strict

# Full test suite
pytest tests/ -v

# Module-specific tests
pytest tests/<module_name>/ -v

# Integration tests
pytest tests/integration/ -v
```

## File Conventions

- Source: `src/<module>/` with `__init__.py` exporting public API
- Tests: `tests/<module>/` mirroring source structure
- Config: `config/` for Hydra YAML, Pydantic classes in each module's `config.py`
- Docs: `docs/` for architecture diagrams and templates
- Scripts: `scripts/` for CLI entry points
