"""The callable target runs a real random-arm cell (requires the torch stack)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.integrations.eval_harness.target import run_basis_cell  # noqa: E402

_CELL = {
    "pde_family": "poisson",
    "seed": 0,
    "arm": "random",
    "max_basis_functions": 3,
    "n_candidate_bases": 8,
    "target_residual": 1e-3,
    "max_rollouts": 32,
    "n_simulations": 8,
    "topk": 3,
}


def test_run_basis_cell_random_arm_shape() -> None:
    out = run_basis_cell(dict(_CELL))
    assert {
        "final_residual",
        "rollouts_used",
        "chosen_action",
        "topk_actions",
        "value",
        "pde_family",
        "arm",
        "seed",
    } <= set(out)
    assert out["arm"] == "random"
    assert out["pde_family"] == "poisson"
    assert isinstance(out["final_residual"], float)
    assert out["final_residual"] == out["final_residual"]  # not NaN
    assert out["rollouts_used"] >= 0
    assert out["chosen_action"] is None or isinstance(out["chosen_action"], int)
    assert isinstance(out["topk_actions"], list)
    assert len(out["topk_actions"]) <= 3


def test_run_basis_cell_is_deterministic_per_seed() -> None:
    first = run_basis_cell(dict(_CELL))
    second = run_basis_cell(dict(_CELL))
    assert first["final_residual"] == pytest.approx(second["final_residual"])
    assert first["chosen_action"] == second["chosen_action"]
    assert first["rollouts_used"] == second["rollouts_used"]
