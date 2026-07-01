---
name: regression-surface
description: Run the correct AlphaGalerkin regression-surface test command block for a changed code path. Use after editing solver/evaluator/PDE/scenario/agent/codec code to run exactly the guarding test suites the CLAUDE.md Regression Surface table prescribes, instead of guessing which tests to run.
---

# regression-surface — run the guarding tests for a change

`CLAUDE.md` maintains a **Regression Surface** table mapping each subsystem to the exact test
command(s) that must stay green when it changes. This skill selects and runs the right block.

## Steps

1. **Identify the changed surface** from the edited files, then open the *Regression Surface*
   table in `CLAUDE.md` and find the matching row(s). Key mappings:
   - `src/alphagalerkin/**` → Solver wiring + Trained evaluator + per-module coverage rows.
   - `src/pde/**`, `src/poc/scenarios/_centaur_common.py` → PDE e2e + LLM-prior + scaling-law +
     research-loop rows (the shared centaur primitives fan out to all three).
   - `src/poc/scenarios/noyron_basis*` , `src/pde/operators_picogk.py`, `src/pde/sdf.py`,
     `src/pde/geometry_picogk.py` → Noyron HX scenario row.
   - `src/integrations/**` → LLM-prior (mocked CPU) + backend-registry rows.
   - `src/agents/**` → Centaur research-loop + agents coverage rows.
   - `src/video_compression/**` → the relevant Phase 0/1/2 codec rows (separate coverage workflow).
2. **Run the block** with `-m "not gpu_required"` unless CUDA is available; the root `conftest.py`
   auto-skips GPU tests on CPU hosts.
3. **Run the coverage gate** row if the change is non-trivial (`--cov-fail-under` per module).
4. **Report** pass/skip/fail faithfully — never claim green on a suite you did not run.

## Notes

- If the change adds a *new* surface, add a new row to the table (see the `spec-new` skill).
- The dependency notes under the table list cross-surface fan-out — honor them.
