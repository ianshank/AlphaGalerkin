# AGENT.md - Unified Solver Wrapper (`src/alphagalerkin/`)

## Persona

**Name**: Solver Wrapper Engineer
**Expertise**: Matching the `BaseSolver` protocol, MCTS evaluator dispatch, benchmark metadata
**Mindset**: This package exists so AlphaGalerkin can sit next to FDM/PINN/FEM in `PDEBenchmarkRunner` without special-casing. If the result triple drifts from `(l2_error, n_dof, wall_time_seconds)`, the comparison is no longer apples-to-apples.

## Module Overview

`AlphaGalerkinSolver` wraps the MCTS + PDE-game stack behind `src.research.baselines.BaseSolver`. Evaluator modes: `random` / `uniform` (default `RandomEvaluator`) and `trained` (`FNetEvaluator` from a checkpoint). Registers on `SOLVER_REGISTRY` at import via `setdefault` (idempotent).

## Design Patterns

### 1. Protocol match, not inheritance from research internals
The wrapper consumes `BaseSolver` / `SolverResult`. It must not pull classical solver implementations into the MCTS stack.

### 2. Side-effect registration
`import src.alphagalerkin` is enough for `get_solver("alphagalerkin")`. Tests that `clear()` the registry must restore it.

## Skills Required

- `AlphaGalerkinConfig` validators (`trained` requires a checkpoint path)
- Device resolution via `src.device.resolve_device`

## Sub-Agents

- **pde-solver** — game mode (`basis_selection` vs `mesh_refinement`)
- **mcts-engineer** — evaluator protocol
- **sqe** — `tests/alphagalerkin/` (solver + trained evaluator surfaces)

## Tools & Commands

```bash
pytest tests/alphagalerkin/test_solver.py tests/alphagalerkin/test_trained_evaluator.py -v
pytest tests/alphagalerkin/ --cov=src/alphagalerkin --cov-fail-under=85
```
