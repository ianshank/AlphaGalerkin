---
name: spec-new
description: Scaffold a new spec-driven feature in AlphaGalerkin — creates a specs/<feature>.spec.md from specs/TEMPLATE.spec.md and the mirrored src/ module + tests/ stubs. Use when starting any new PoC scenario, PDE operator, agent type, or module so the spec (data contract + acceptance criteria + MetricThreshold thresholds) is written before the code.
---

# spec-new — scaffold a spec-driven feature

AlphaGalerkin uses **spec-driven development**: write the contract before the code. This skill
bootstraps that flow for a new feature named `$FEATURE`.

## Steps

1. **Copy the template**: create `specs/$FEATURE.spec.md` from `specs/TEMPLATE.spec.md`. Fill in
   Context (link the CLAUDE.md Next-Steps row / ROI tier), User Story, Data Contract, Acceptance
   Criteria, and the Thresholds table.
2. **Thresholds = existing type**: express every pass/fail criterion as a
   `src.poc.config.MetricThreshold(name, operator, value)`. Do **not** invent a new threshold
   schema — the feature's Pydantic config `get_default_thresholds()` is the single source of truth.
3. **Mirror the layout**: create the module under `src/<pkg>/` and its test under `tests/<pkg>/`
   (identical relative path). Follow the closest existing precedent
   (`src/poc/scenarios/scaling_law.py` + `scaling_law_config.py` for scenarios;
   `src/pde/operators.py` for PDE operators; `src/agents/*.py` for agents).
4. **Config first**: every tunable is a typed `Field(default=…, <bounds>, description=…)`.
   Surface numerical-stability literals as named module constants.
5. **AQA test**: add a test asserting the config's `get_default_thresholds()` returns exactly the
   metrics/operators the spec's Thresholds table documents (spec ↔ config agreement).
6. **Register the regression surface**: add the feature's test command block to the
   *Regression Surface* table in `CLAUDE.md`.

## Guardrails

- Additive/backwards-compatible only: new files, new optional fields, new registry keys.
- `ruff` + `mypy --strict` clean on the changed surface; tests to the 85% coverage gate.
- Read `specs/README.md` for the full workflow.

## When this is the wrong skill

`specs/` sits *below* the charter (`openspec/project.md`: charter → `ARCHITECTURE.md` →
`CLAUDE.md` → `specs/`). If the change touches a charter Requirement — scope, a numeric claim,
the capability register, a coverage gate, or an accepted deviation — it needs an **OpenSpec
change package** as well, and the `openspec-change` skill scaffolds it. A feature that adds a
`src/` package or registers a PoC scenario touches *Scope Integrity* and *Capability Register
Accuracy*, so it usually needs both.

Also add the new spec to `specs/README.md`'s index. One shipped spec sat outside that table
from the day it was written.
