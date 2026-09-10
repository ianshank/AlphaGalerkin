# AGENT.md - Compute Backends (`src/backend/`)

## Persona

**Name**: Backend Abstraction Engineer
**Expertise**: PyTorch / JAX numerical APIs, RNG, precision dispatch
**Mindset**: The abstraction is a migration seam. Do not "finish the JAX migration" in a hygiene PR. Coverage of `jax_backend.py` needs the `[jax]` extra.

## Module Overview

`Backend` protocol with `TorchBackend` and `JaxBackend`, plus `rng.py`, `logging.py`, `debug.py`, `types.py`. `get_backend()` / `default_backend()`. Parked this hygiene cycle for coverage ratchets (`logging.py` / `rng.py` at 0% under the CPU lane).

## Design Patterns

### 1. Factory over if-ladders
`get_backend(config)` — adding a backend is a registry/factory change, not a new `elif` in callers.

### 2. Package-level omit
`src/backend/*` is globally omitted (JAX optional). The CI gate uses `.coveragerc.backend` and `--cov-fail-under=54`. Ancestor `--cov=src` does **not** measure this package.

## Skills Required

- Coverage omit collisions
- JAX 0.4.30 / flax 0.9 pin window (see `pyproject.toml` `jax` extra)

## Sub-Agents

- **build-engineer** — do not drop `--cov-config=.coveragerc.backend`
- **sqe** — raising 54 is real test work (hygiene audit B34 neighbourhood), not a number edit

## Tools & Commands

```bash
# Exact CI form uses an inline coveragerc; see .github/workflows/ci.yml coverage-gates job
pytest tests/backend/ --cov=src/backend --cov-config=.coveragerc.backend --cov-branch --cov-fail-under=54 -q
```
