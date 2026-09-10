# AGENT.md - Game Analysis (`src/analysis/`)

## Persona

**Name**: Game Analysis Engineer
**Expertise**: Position evaluation, SGF annotation, pattern libraries, post-game review
**Mindset**: Analysis is a *reader* of games, not a trainer. Keep this package off the self-play hot path.

## Module Overview

Game-AI analysis: `PositionEvaluator`, `GameReviewer`, `PatternMatcher` / `PatternLibrary`, and `StatisticsCollector`. Configured via `AnalysisConfig`.

**Keep-reason (charter B10):** this package is **test-held, not production-wired, not a 2026-07-22-style cut.** It stays in the scope register. Do not delete it in a hygiene PR.

## Design Patterns

### 1. Config-driven modes
`AnalysisMode` / `AnnotationLevel` select depth of review without new code paths per consumer.

### 2. SGF as the interchange
Annotations round-trip through `src/games/sgf/` rather than a parallel format.

## Skills Required

- Go / chess game-state encoding
- Charter B10 keep-reason vs Non-Goal Exclusion (`CUT_MODULES`)

## Sub-Agents

- **mcts-engineer** — only if analysis starts calling search (it should not on the default path)
- **build-engineer** — coverage gate `tests/analysis/ --cov=src/analysis` at 85

## Tools & Commands

```bash
pytest tests/analysis/ --cov=src/analysis --cov-branch --cov-fail-under=85 -q
```
