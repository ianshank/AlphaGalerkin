"""Slice E: SubstrateRefinementGame purity, registry, real MCTS smoke."""

from __future__ import annotations

import numpy as np
import pytest

import src.pde.register_refinement_games  # noqa: F401
from src.mcts.search import MCTS, SearchMode
from src.pde.games.lshape_amr import EncodedValueEvaluator
from src.pde.games.substrate_refinement import (
    GAME_REGISTRY_NAME,
    VALUE_CHANNEL_SIZE,
    SubstrateEpisodeState,
    SubstrateRefinementGame,
)
from src.pde.games.substrate_refinement_config import SubstrateRefinementConfig
from src.refinement.adapter import RefinementGameAdapter
from src.refinement.registry import RefinementGameRegistry
from src.research.substrates.config import SubstrateConfig
from src.research.substrates.residual_evaluator import ResidualPriorErrorValueEvaluator


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
    def test_to_tensor_action_aligned_plus_value_slot(self, game: SubstrateRefinementGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.shape == (game.action_space_size + VALUE_CHANNEL_SIZE,)
        assert tensor.dtype == np.float32
        assert tensor[-1] != tensor[0]

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


class TestResidualEvaluatorAndMcts:
    """Task 4.1 closeout: a real ``MCTS.get_action`` call, not one adapter apply."""

    def test_value_slot_is_last_not_first(self, game: SubstrateRefinementGame) -> None:
        """``EncodedValueEvaluator`` would read ``state[0]`` (an indicator)."""
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        n_actions = game.action_space_size
        encoded = EncodedValueEvaluator(n_actions=n_actions)
        residual = ResidualPriorErrorValueEvaluator(n_actions=n_actions)
        legal = game.get_valid_actions(state)
        _, encoded_value = encoded.evaluate(tensor, legal)
        _, residual_value = residual.evaluate(tensor, legal)
        # EncodedValueEvaluator clamps to [-1, 1]; residual indicators are not
        # a leaf value and often sit outside that range (the clamp is the bug).
        first_slot = float(tensor[0])
        encoded_expected = max(-1.0, min(1.0, first_slot))
        assert encoded_value == pytest.approx(encoded_expected, abs=1e-6)
        assert residual_value == pytest.approx(float(tensor[-1]), abs=1e-6)
        assert encoded_value != residual_value
        assert first_slot != float(tensor[-1])

    def test_mcts_search_and_get_action_on_tensor_grid(self, game: SubstrateRefinementGame) -> None:
        adapter = RefinementGameAdapter(game)
        evaluator = ResidualPriorErrorValueEvaluator(n_actions=adapter.action_space_size)
        mcts = MCTS(
            evaluator=evaluator,
            n_simulations=4,
            search_mode=adapter.search_mode,
            use_intermediate_rewards=False,
        )
        assert adapter.search_mode is SearchMode.SINGLE_AGENT
        action = mcts.get_action(adapter, temperature=0.0, add_noise=False)
        legal = adapter.get_legal_actions()
        assert action in legal
        adapter.apply_action(action)
        assert adapter.state.step == 1

    def test_top_k_caps_legal_set(self) -> None:
        game = SubstrateRefinementGame(
            SubstrateRefinementConfig(
                name="topk",
                substrate=SubstrateConfig(
                    name="topk_tg",
                    kind="tensor_grid",
                    initial_side=4,
                    solve_cache_max_entries=32,
                ),
                operator_name="poisson",
                max_steps=3,
                max_action_space=64,
                top_k_actions=2,
            )
        )
        legal = game.get_valid_actions(game.get_initial_state())
        assert len(legal) == 2


@pytest.mark.fem_required
class TestSkfemTriMctsSmoke:
    def test_skfem_lshape_search_one_step(self) -> None:
        from src.research.substrates.skfem_tri import SkfemTriSubstrate

        game = SubstrateRefinementGame(
            SubstrateRefinementConfig(
                name="skfem-smoke",
                substrate=SubstrateConfig(
                    name="skfem-smoke-sub",
                    kind="skfem_tri",
                    initial_refinements=1,
                    solve_cache_max_entries=64,
                ),
                operator_name="lshape_poisson",
                max_action_space=256,
                max_steps=2,
                top_k_actions=4,
            )
        )
        assert isinstance(game.substrate, SkfemTriSubstrate)
        adapter = RefinementGameAdapter(game)
        evaluator = ResidualPriorErrorValueEvaluator(n_actions=adapter.action_space_size)
        mcts = MCTS(
            evaluator=evaluator,
            n_simulations=2,
            search_mode=adapter.search_mode,
            use_intermediate_rewards=False,
        )
        action = mcts.get_action(adapter, temperature=0.0, add_noise=False)
        adapter.apply_action(action)
        assert adapter.state.step == 1
