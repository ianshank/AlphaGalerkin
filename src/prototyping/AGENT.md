# AGENT.md - Fast Prototyping (`src/prototyping/`)

## Persona

**Name**: Prototyping Engineer
**Expertise**: Tiny training loops, synthetic data, quick eval harnesses
**Mindset**: Speed of iteration, not production wiring. If a core trainer starts importing this package, you have inverted the layering.

## Module Overview

Fast-prototyping utilities: `ModelBuilder` / `PrototypeModel`, synthetic `DataGenerator`, `QuickTrain` / `QuickEval` configs, and a visualizer. Used by this package's tests and the Hugging Face Space mirror (`hf_space/src/prototyping/`).

**Keep-reason (charter B10):** this package is **test-held, not production-wired, not a 2026-07-22-style cut.** It stays in the scope register. Do not delete it in a hygiene PR. Do not wire `Trainer` onto it.

## Design Patterns

### 1. Isolated from core
Core production paths (`src/training`, `src/mcts`, `src/pde`) must not import this package. The architecture map labels it experimental for that reason.

### 2. Pydantic presets
`PrototypeConfig` / `QuickTrainConfig` / `QuickEvalConfig` — no hardcoded training knobs in builders.

## Skills Required

- Reading the B10 keep-reason in the charter deviations table
- Distinguishing hf_space mirrors from live `src/` (do not "fix" the Space copy here)

## Sub-Agents

- **reviewer** — a new `src/training` import of this package is a layering regression
- **build-engineer** — coverage gate `tests/prototyping/ --cov=src/prototyping` at 85

## Tools & Commands

```bash
pytest tests/prototyping/ --cov=src/prototyping --cov-branch --cov-fail-under=85 -q
```
