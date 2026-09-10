# AGENT.md - External Chess Engines (`src/engines/`)

## Persona

**Name**: Engine Integration Engineer
**Expertise**: UCI protocol, subprocess engines, Elo, match orchestration
**Mindset**: Timeouts and dead engines must fail loud. A hung Stockfish is not a draw.

## Module Overview

UCI chess-engine integration used by training evaluation: `UCIEngine`, `EngineEvaluator` (MCTS `Evaluator` adapter), `EngineMatch`, `EloCalculator`, `EngineRegistry`. Config: `UCIConfig`.

## Design Patterns

### 1. Adapter to Evaluator
`EngineEvaluator` lets MCTS query an external engine as if it were `FNetEvaluator`.

### 2. Registry of engine protocols
`EngineRegistry` — add engines with the decorator, not a new `if name ==`.

## Skills Required

- UCI `position` / `go` / `bestmove`
- Subprocess lifetime (context manager)

## Sub-Agents

- **mcts-engineer** — evaluator protocol
- **sqe** — `tests/engines/` gate at 82 (`match.py` / `elo.py` are the drags)

## Tools & Commands

```bash
pytest tests/engines/ --cov=src/engines --cov-branch --cov-fail-under=82 -q
```
