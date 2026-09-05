"""Slice E: SubstrateRefinementGame purity, registry, adapter smoke."""

from __future__ import annotations

import numpy as np
import pytest

import src.pde.register_refinement_games  # noqa: F401
from src.pde.games.substrate_refinement import (
    GAME_REGISTRY_NAME,
    SubstrateEpisodeState,
    SubstrateRefinementGame,
)
from src.pde.games.substrate_refinement_config import SubstrateRefinementConfig
from src.refinement.adapter import RefinementGameAdapter
from src.refinement.registry import RefinementGameRegistry
from src.research.substrates.config import SubstrateConfig


@pytest.fixture
def game() -> SubstrateRefinementGame:
    config = SubstrateRefinementConfig(
        name="test_game",
        substrate=SubstrateConfig(
            name="test_tg",
            kind="tensor_grid",
            initial_side=4,
            solve_cache_max_entries=32,
        ),
        operator_name="poisson",
        max_steps=3,
        max_action_space=64,
        computational_budget=100.0,
        refine_cost=1.0,
        error_tolerance=1e-12,
    )
    return SubstrateRefinementGame(config=config)


class TestRegistry:
    def test_production_registrant_present(self) -> None:
        cls = RefinementGameRegistry().get_or_raise(GAME_REGISTRY_NAME)
        assert cls is SubstrateRefinementGame


class TestPurity:
    def test_apply_action_does_not_mutate_instance_or_prior_state(
        self, game: SubstrateRefinementGame
    ) -> None:
        state = game.get_initial_state()
        assert isinstance(state, SubstrateEpisodeState)
        actions = game.get_valid_actions(state)
        assert actions, "expected at least one refinable unit"
        action = actions[0]
        before_error = state.error_estimate
        before_step = state.step
        before_history = list(state.history)
        # Capture instance identity markers — game must stay episode-stateless.
        cache_id = id(game.solve_cache)
        new_state = game.apply_action(state, action)
        assert state.error_estimate == before_error
        assert state.step == before_step
        assert state.history == before_history
        assert new_state is not state
        assert new_state.step == before_step + 1
        assert new_state.history == [*before_history, action]
        assert id(game.solve_cache) == cache_id

    def test_default_clone_shares_stateless_game(self, game: SubstrateRefinementGame) -> None:
        assert game.clone() is game


class TestSolveCacheIntegration:
    def test_replaying_same_mesh_hits_cache(self, game: SubstrateRefinementGame) -> None:
        state = game.get_initial_state()
        misses_after_init = game.solve_cache.misses
        # Re-solve initial mesh via cache directly.
        game.solve_cache.get_or_solve(game.substrate, state.mesh)
        assert game.solve_cache.hits >= 1
        assert game.solve_cache.misses == misses_after_init


class TestTerminalAndTensor:
    def test_to_tensor_fixed_width(self, game: SubstrateRefinementGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.shape == (game.action_space_size,)
        assert tensor.dtype == np.float32

    def test_invalid_action_raises(self, game: SubstrateRefinementGame) -> None:
        state = game.get_initial_state()
        with pytest.raises(ValueError, match="not valid"):
            game.apply_action(state, game.action_space_size - 1)


class TestAdapterSmoke:
    def test_adapter_applies_one_action(self, game: SubstrateRefinementGame) -> None:
        adapter = RefinementGameAdapter(game)
        legal = adapter.get_legal_actions()
        assert legal
        adapter.apply_action(legal[0])
        assert adapter.state.step == 1
        assert len(adapter.error_history) == 2
