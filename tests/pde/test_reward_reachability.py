"""Tests that ``PDEGame.get_reward`` is reachable through MCTS (F1).

Historically ``get_reward`` was abstract and overridden by every PDE game but
had zero call sites in ``src/`` — MCTS backed up only the terminal
``{-1, 0, 1}`` winner. The intermediate-reward wiring makes it reachable, but
only behind the opt-in ``use_intermediate_rewards`` flag so the default search
behaviour is byte-for-byte unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mcts.evaluator import RandomEvaluator
from src.mcts.search import MCTS, SearchMode
from src.pde.config import BasisSelectionConfig, PDEConfig, PDEGameConfig, PDEType
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.operators import PoissonOperator


def _adapter() -> PDEGameAdapter:
    pde_config = PDEConfig(
        name="reward_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )
    game_config = PDEGameConfig(
        name="reward_basis",
        pde_config=pde_config,
        game_mode="basis_selection",
        max_steps=6,
        error_tolerance=1e-6,
        computational_budget=1e4,
        basis_config=BasisSelectionConfig(
            name="reward_basis_selection",
            max_basis_functions=8,
            basis_type="fourier",
            max_frequency=3,
        ),
    )
    operator = PoissonOperator(pde_config)
    # BasisSelectionGame is stateless: clone() returns self, so an instance
    # attribute spy on get_reward survives the per-simulation clones.
    game = BasisSelectionGame(operator, game_config)
    return PDEGameAdapter(game)


class _RewardSpy:
    def __init__(self, adapter: PDEGameAdapter) -> None:
        self.count = 0
        self._orig = adapter.pde_game.get_reward

        def counting(state, prev_state):  # noqa: ANN001, ANN202
            self.count += 1
            return self._orig(state, prev_state)

        adapter.pde_game.get_reward = counting  # type: ignore[method-assign]


def test_get_reward_not_called_when_disabled() -> None:
    np.random.seed(0)
    adapter = _adapter()
    spy = _RewardSpy(adapter)
    mcts = MCTS(
        evaluator=RandomEvaluator(n_actions=adapter.pde_game.action_space_size),
        n_simulations=16,
        search_mode=SearchMode.SINGLE_AGENT,
    )
    mcts.search(adapter, add_noise=False)
    assert spy.count == 0


def test_get_reward_called_when_enabled() -> None:
    np.random.seed(0)
    adapter = _adapter()
    spy = _RewardSpy(adapter)
    mcts = MCTS(
        evaluator=RandomEvaluator(n_actions=adapter.pde_game.action_space_size),
        n_simulations=16,
        search_mode=SearchMode.SINGLE_AGENT,
        use_intermediate_rewards=True,
    )
    mcts.search(adapter, add_noise=False)
    assert spy.count >= 1


def test_adapter_get_last_reward_zero_before_action() -> None:
    """No transition yet → zero reward, not an error."""
    adapter = _adapter()
    assert adapter.get_last_reward() == pytest.approx(0.0)


def test_adapter_get_last_reward_matches_get_reward() -> None:
    """get_last_reward mirrors get_reward on the most recent transition."""
    adapter = _adapter()
    prev = adapter.state
    action = adapter.get_legal_actions()[0]
    adapter.apply_action(action)
    expected = adapter.pde_game.get_reward(adapter.state, prev)
    assert adapter.get_last_reward() == pytest.approx(expected)
