# AGENT.md - Curriculum Learning (`src/curriculum/`)

## Persona

**Name**: Curriculum Engineer
**Expertise**: Board-size progression, stage transitions, opponent selection
**Mindset**: A stage transition is a measured event, not a silent field mutation. Log the criterion that fired.

## Module Overview

Curriculum scheduler for board-size progression (9→13→19): `CurriculumConfig`, `CurriculumStage`, `CurriculumScheduler`, `CurriculumManager`.

**Keep-reason (charter B10):** this package is **test-held, not production-wired, not a 2026-07-22-style cut.** `Trainer` has its own curriculum fields; this package is not the production driver. Do not delete it in a hygiene PR. Do not "keep-and-wire" it onto `Trainer` in a hygiene PR.

## Design Patterns

### 1. Explicit stage machine
`StageStatus` + `ProgressionCriterion` / `ProgressionOperator` — transitions are typed, not ad-hoc `if board_size`.

### 2. Separate from Trainer
`src/training/trainer.py` must not grow a dependency on this package without a dedicated change (B10 retirement criterion).

## Skills Required

- Charter B10 keep-reason vs a 2026-07-22-style cut
- Pydantic config hashing for reproducible schedules

## Sub-Agents

- **reviewer** — a new `Trainer` import of this package is a product decision, not hygiene
- **build-engineer** — coverage gate `tests/curriculum/ --cov=src/curriculum` at 87

## Tools & Commands

```bash
pytest tests/curriculum/ --cov=src/curriculum --cov-branch --cov-fail-under=87 -q
```
