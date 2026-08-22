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
     mapping. Gates at **90** branch, the strictest in the repo.
   - `src/refinement/**` → Domain-free refinement engine (WS1, BC) row. Gates at 85 branch.
   - `src/alphagalerkin/**` → Solver wiring + Trained evaluator + per-module coverage rows.
   - `src/pde/**`, `src/poc/scenarios/_centaur_common.py` → PDE e2e + LLM-prior + scaling-law +
     research-loop rows (the shared centaur primitives fan out to all three).
   - `src/pde/stochastic/**` → Stochastic Galerkin NKE layer row. Note its **import-isolation
     allowlist** (`tests/pde/stochastic/test_import_isolation.py`): any *new* `src.` import in
     that layer fails the guard by design.
   - `src/poc/scenarios/noyron_basis*` → **Noyron basis selection (v2.2)** row
     (`tests/poc/test_noyron_basis_config.py tests/poc/test_noyron_basis_scenario.py`).
   - `src/pde/operators_picogk.py`, `src/pde/sdf.py`, `src/pde/geometry_picogk.py` →
     **Noyron HX scenario** row
     (`tests/pde/test_sdf.py tests/pde/test_picogk_domain.py tests/poc/test_noyron_hx_scenario.py`).
   - `src/integrations/**` → LLM-prior (mocked CPU) + backend-registry rows.
   - `src/agents/**` → Centaur research-loop + agents coverage rows.
   - `src/research/transfer_baseline_compare.py`, `src/experiments/cnn_baseline.py`,
     `src/poc/scenarios/transfer_baseline_compare*` → Honest zero-shot transfer row.
   - **`src/research/baselines.py` → BOTH the L-shape AMR row AND the SBIR P40 row.**
   - `src/research/**` (other) → SBIR P40 hardening surface row.
   - `dashboard/**` → Dashboard quality gates (WS6) row (lint + 84 branch coverage).
   - `openspec/**`, `ARCHITECTURE.md`, `CLAUDE.md`, `docs/**` → Project charter alignment row.
2. **Run the block** with `-m "not gpu_required"` unless CUDA is available.
3. **Run the coverage gate** row if the change is non-trivial (`--cov-fail-under` per module).
4. **Report** pass/skip/fail faithfully — never claim green on a suite you did not run.
