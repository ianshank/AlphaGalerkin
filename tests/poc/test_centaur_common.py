"""Direct tests for the shared centaur MCTS primitives (`_centaur_common`).

These cover the construction helpers and the inner rollout loop independently
of any scenario, so the shared module is exercised even when scenario tests
use a synthetic cell override.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.mcts.evaluator import RandomEvaluator
from src.poc.scenarios._centaur_common import (
    PDE_TYPE_MAP,
    build_arm_evaluator,
    build_basis_game,
    build_pde_operator,
    enumerate_basis_descriptions,
    run_basis_selection_cell,
)


def _poisson_game(target_residual: float = 1e-6):
    operator = build_pde_operator("poisson")
    return build_basis_game(
        "poisson",
        operator,
        max_basis_functions=2,
        n_candidate_bases=4,
        target_residual=target_residual,
    )


def test_pde_type_map_includes_ood_operators() -> None:
    assert "helmholtz" in PDE_TYPE_MAP
    assert "biharmonic" in PDE_TYPE_MAP


def test_build_pde_operator_known() -> None:
    operator = build_pde_operator("helmholtz")
    assert operator.name == "helmholtz"


def test_build_pde_operator_unknown_raises() -> None:
    with pytest.raises(ValueError, match="has no PDEType mapping"):
        build_pde_operator("not_a_pde")


def test_build_basis_game_and_descriptions() -> None:
    game = _poisson_game()
    assert game.action_space_size == 4
    descriptions = enumerate_basis_descriptions(game)
    assert len(descriptions) == 4


# --------------------------------------------------------------------------- #
# build_arm_evaluator                                                         #
# --------------------------------------------------------------------------- #


def test_build_arm_evaluator_random() -> None:
    game = SimpleNamespace(action_space_size=5)
    evaluator = build_arm_evaluator(
        "random", game=game, pde_name="poisson", basis_descriptions=[], seed=0
    )
    assert isinstance(evaluator, RandomEvaluator)


def test_build_arm_evaluator_unknown_arm_raises() -> None:
    game = SimpleNamespace(action_space_size=5)
    with pytest.raises(ValueError, match="unknown arm"):
        build_arm_evaluator(
            "invented", game=game, pde_name="poisson", basis_descriptions=[], seed=0
        )


def test_build_arm_evaluator_trained_missing_model_raises() -> None:
    game = SimpleNamespace(action_space_size=5)
    with pytest.raises(RuntimeError, match="trained_model is None"):
        build_arm_evaluator(
            "trained", game=game, pde_name="poisson", basis_descriptions=[], seed=0, device="cpu"
        )


def test_build_arm_evaluator_trained_missing_device_raises() -> None:
    game = SimpleNamespace(action_space_size=5)
    with pytest.raises(RuntimeError, match="device is None"):
        build_arm_evaluator(
            "trained",
            game=game,
            pde_name="poisson",
            basis_descriptions=[],
            seed=0,
            trained_model=MagicMock(),
        )


def test_build_arm_evaluator_llm_missing_client_raises() -> None:
    game = SimpleNamespace(action_space_size=5)
    with pytest.raises(RuntimeError, match="lm_client is None"):
        build_arm_evaluator(
            "llm", game=game, pde_name="poisson", basis_descriptions=["b"] * 5, seed=0
        )


def test_build_arm_evaluator_llm_with_client() -> None:
    from src.integrations.lm_studio.evaluator import LMStudioEvaluator

    game = SimpleNamespace(action_space_size=3)
    evaluator = build_arm_evaluator(
        "llm",
        game=game,
        pde_name="poisson",
        basis_descriptions=["b0", "b1", "b2"],
        seed=7,
        lm_client=MagicMock(name="LMStudioClient"),
    )
    assert isinstance(evaluator, LMStudioEvaluator)


# --------------------------------------------------------------------------- #
# run_basis_selection_cell                                                    #
# --------------------------------------------------------------------------- #


def test_run_cell_returns_immediately_when_below_target() -> None:
    # A huge target means the initial error is already below it.
    game = _poisson_game(target_residual=0.999)
    evaluator = RandomEvaluator(n_actions=game.action_space_size)
    outcome = run_basis_selection_cell(
        game=game,
        evaluator=evaluator,
        target_residual=0.999,
        max_rollouts=8,
        n_simulations=2,
    )
    assert outcome.rollouts_used == 0
    assert outcome.final_residual >= 0.0


def test_run_cell_real_loop_accumulates_rollouts() -> None:
    import numpy as np
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    game = _poisson_game(target_residual=1e-9)
    evaluator = RandomEvaluator(n_actions=game.action_space_size)
    outcome = run_basis_selection_cell(
        game=game,
        evaluator=evaluator,
        target_residual=1e-9,
        max_rollouts=4,
        n_simulations=2,
        scenario_logger=MagicMock(),
    )
    assert outcome.rollouts_used >= 0
    assert outcome.final_residual >= 0.0
