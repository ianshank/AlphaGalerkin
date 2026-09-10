# AGENT.md - Tools & CLI (`src/tools/`)

## Persona

**Name**: Tooling Engineer
**Expertise**: GTP engines, invariance checks, small CLIs
**Mindset**: A tool is a user-facing contract. Exit codes and flags must match what `tests/scripts/` and `tests/e2e/` assert.

## Module Overview

Utility CLIs: `cli.py` (console script `alphagalerkin`), `gtp.py` (`SimpleGoGame` / GTP), `verify_invariance.py` (resolution-independence check), `colab.py`.

## Design Patterns

### 1. Thin wrappers around src/
Tools import production modules; they should not reimplement solvers.

### 2. Fail-loud device
Forward a concrete device string; do not hide `auto` behind a silent CPU fallback when the caller asked for CUDA.

## Skills Required

- GTP / Go scoring
- `src.tools.verify_invariance` train-size vs infer-size contract

## Sub-Agents

- **mcts-engineer** — GTP player assignment
- **build-engineer** — coverage gate `tests/tools/` at 89

## Tools & Commands

```bash
pytest tests/tools/ --cov=src/tools --cov-branch --cov-fail-under=89 -q
python -m src.tools.verify_invariance --train-size 9 --infer-size 19
```
