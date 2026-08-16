---
name: regression-surface
description: Run the correct AlphaGalerkin regression-surface test command block for a changed code path. Use after editing solver/evaluator/PDE/scenario/agent code to run exactly the guarding test suites the CLAUDE.md Regression Surface table prescribes, instead of guessing which tests to run.
---

# regression-surface — run the guarding tests for a change

`CLAUDE.md` maintains a **Regression Surface** table mapping each subsystem to the exact test
command(s) that must stay green when it changes. This skill selects and runs the right block.

## Steps

1. **Identify the changed surface** from the edited files, then open the *Regression Surface*
   table in `CLAUDE.md` and find the matching row(s). Key mappings:
   - **`src/mcts/**` → MCTS backup semantics (F0) row** — sign inversion per `SearchMode`, the
     `test_single_agent_search_prefers_higher_value_at_all_depths` anchor, `BatchMCTS` leaf
     mapping. Gates at **90** branch, the strictest in the repo. Also run *Reward reachability
     (F1) + clone isolation (F3)*, which exercises the adapter against this engine.
   - `src/refinement/**` → Domain-free refinement engine (WS1, BC) row. Gates at 85 branch.
   - `src/alphagalerkin/**` → Solver wiring + Trained evaluator + per-module coverage rows.
   - `src/pde/**`, `src/poc/scenarios/_centaur_common.py` → PDE e2e + LLM-prior + scaling-law +
     research-loop rows (the shared centaur primitives fan out to all three).
   - `src/pde/stochastic/**` → Stochastic Galerkin NKE layer row. Note its **import-isolation
     allowlist** (`tests/pde/stochastic/test_import_isolation.py`): any *new* `src.` import in
     that layer fails the guard by design.
   - `src/poc/scenarios/noyron_basis*` , `src/pde/operators_picogk.py`, `src/pde/sdf.py`,
     `src/pde/geometry_picogk.py` → Noyron HX scenario row.
   - `src/integrations/**` → LLM-prior (mocked CPU) + backend-registry rows.
   - `src/agents/**` → Centaur research-loop + agents coverage rows.
   - `src/research/transfer_baseline_compare.py`, `src/experiments/cnn_baseline.py`,
     `src/poc/scenarios/transfer_baseline_compare*` → Honest zero-shot transfer row.
   - **`src/research/baselines.py` → BOTH the L-shape AMR row AND the SBIR P40 row.** The table
     carries an explicit shared-code warning here: the AMR comparison reuses the same masked
     solver, so an edit that looks AMR-local silently moves the SBIR benchmark too.
   - `src/research/**` (other) → SBIR P40 hardening surface row.
   - `dashboard/**` → Dashboard quality gates (WS6) row (lint + 84 branch coverage).
   - `openspec/**`, `ARCHITECTURE.md`, `CLAUDE.md`, `docs/**` → Project charter alignment row.
   - `src/seeding.py`, `src/constants.py` → no dedicated row; run the suites of the packages that
     import them (`tests/test_seeding.py`, `tests/test_constants.py`, plus every scenario config
     test asserting seed derivation).

   The table grows with every milestone (count it rather than trusting a number here:
   `awk '/^\| Surface \|/,/^$/' CLAUDE.md | grep -c '^| '`); the list above is the high-traffic subset, not a complete
   index. When the changed path is not listed, read the table directly rather than assuming
   no row applies.
2. **Run the block** with `-m "not gpu_required"` unless CUDA is available; the root `conftest.py`
   auto-skips GPU tests on CPU hosts.
3. **Run the coverage gate** row if the change is non-trivial (`--cov-fail-under` per module).
4. **Report** pass/skip/fail faithfully — never claim green on a suite you did not run.

## Notes

- If the change adds a *new* surface, add a new row to the table (see the `spec-new` skill).
- The dependency notes under the table list cross-surface fan-out — honor them.
