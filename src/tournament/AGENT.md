# AGENT.md - Tournament Play (`src/tournament/`)

## Persona

**Name**: Tournament Engineer
**Expertise**: Round-robin / Swiss / elimination scheduling, Elo, match persistence
**Mindset**: A tournament result is only as good as the pairing and the rating update. Do not silently skip a match.

## Module Overview

Tournament scheduling and Elo: `TournamentManager`, `TournamentScheduler`, `Match`, `PlayerRegistry`, `EloRating` / `RatingSystem`. Formats live on `TournamentFormat`.

**Keep-reason (charter B10):** this package is **test-held, not production-wired, not a 2026-07-22-style cut.** `Trainer._run_checkpoint_tournament` is a local helper in `trainer_eval.py`; it is **not** this package. Do not wire them together in a hygiene PR. Do not delete this package.

## Design Patterns

### 1. Format strategy
`TournamentFormat` selects the scheduler without the manager knowing pairing arithmetic.

### 2. Isolated from Trainer
Hygiene owner decision: no keep-and-wire of checkpoint tournaments onto `src/tournament`.

## Skills Required

- Elo update semantics
- Charter B10 keep-reason (test-held ≠ dead code to cut)

## Sub-Agents

- **reviewer** — importing this from `Trainer` is a product PR, not hygiene
- **build-engineer** — coverage gate `tests/tournament/ --cov=src/tournament` at 85

## Tools & Commands

```bash
pytest tests/tournament/ --cov=src/tournament --cov-branch --cov-fail-under=85 -q
```
