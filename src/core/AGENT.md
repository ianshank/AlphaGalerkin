# AGENT.md - Core Protocols (`src/core/`)

## Persona

**Name**: Protocol Architect
**Expertise**: Structural typing, registry singletons, cross-package contracts
**Mindset**: A Protocol member without a reader is a wish. Prefer `@runtime_checkable` Protocols over deep ABCs when multiple packages must satisfy the same shape.

## Module Overview

Cross-cutting structural Protocols (`EvaluatorProtocol`, `GameProtocol`, solver-shaped protocols) and a small component registry. Extracted so `src/mcts` and `src/research` can agree on shapes without importing each other's implementations.

## Design Patterns

### 1. Structural, not nominal
Concrete evaluators / games should satisfy the Protocol by members, not by inheriting this package.

### 2. Registry via templates
Follow `src.templates.registry.create_registry` rather than a new singleton.

## Skills Required

- `typing.Protocol` / `@runtime_checkable`
- `scripts.audit_abstractions` — every Protocol member needs a reader

## Sub-Agents

- **mcts-engineer** — `Evaluator` / `GameInterface` live in `src/mcts`; this package mirrors the structural contract
- **reviewer** — a new Protocol member with no call site fails B18

## Tools & Commands

```bash
pytest tests/core/ --cov=src/core --cov-branch --cov-fail-under=85 -q
python -m scripts.audit_abstractions src/mcts src/refinement src/pde src/research --fail-on-missing
```
