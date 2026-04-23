"""Shared constants for ``SolverResult.metadata`` contracts.

Centralises the string keys that both solver implementations (producers)
and benchmark/report consumers (readers) use so renames are a single-file
change and typos fail loudly at import rather than at runtime as
``KeyError`` on a misspelled literal.

This module intentionally has no runtime imports beyond ``__future__`` to
stay at the bottom of the dependency graph — any module can import from
here without creating a cycle.
"""

from __future__ import annotations

# Solver-agnostic keys surfaced in ``SolverResult.metadata`` and consumed
# by ``PDEBenchmarkRunner.export_csv`` plus downstream CSV/JSON consumers.
METADATA_KEY_SEED = "seed"
METADATA_KEY_REFINEMENT_LEVEL = "refinement_level"
METADATA_KEY_METHOD = "method"

# AlphaGalerkin-specific provenance keys.  Kept here so dashboards / CSV
# post-processors can pick them up without importing the solver package.
METADATA_KEY_SOLVER = "solver"
METADATA_KEY_GAME_MODE = "game_mode"
METADATA_KEY_N_MCTS_SIMULATIONS = "n_mcts_simulations"
METADATA_KEY_N_ACTIONS_TAKEN = "n_actions_taken"
METADATA_KEY_MAX_STEPS = "max_steps"
METADATA_KEY_TARGET_TOLERANCE = "target_tolerance"
METADATA_KEY_EVALUATOR = "evaluator"
METADATA_KEY_MIN_GAME_DOF = "min_game_dof"
METADATA_KEY_ERROR_HISTORY = "error_history"
METADATA_KEY_SOLUTION_AVAILABLE = "solution_available"
METADATA_KEY_TERMINATION_REASON = "termination_reason"
