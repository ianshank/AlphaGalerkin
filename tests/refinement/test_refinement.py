"""Tests for the domain-free refinement engine (WS1 extraction).

Covers the RefinementState converters (both directions with PDEState),
the generic RefinementGameConfig round-trip (zero field loss), the
RefinementGameAdapter protocol + a real single-agent MCTS micro-run, the
RefinementGame ABC clone contract, and the registry.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import Field

from src.refinement import (
    RefinementGame,
    RefinementGameAdapter,
    RefinementGameConfig,
    RefinementGameRegistry,
    RefinementLike,
    RefinementState,
    register_refinement_game,
)
from src.templates.config import BaseModuleConfig

# --------------------------------------------------------------------------- #
# A concrete toy game                                                          #
# --------------------------------------------------------------------------- #


class _ToyRefinementGame(RefinementGame):
    """Deterministic refinement game: each action halves the error."""

    def __init__(self, n_actions: int = 4, max_steps: int = 5) -> None:
        self._n = n_actions
        self._max_steps = max_steps

    @property
    def action_space_size(self) -> int:
        return self._n

    def get_initial_state(self) -> RefinementState:
        return RefinementState(
            values=np.zeros(self._n, dtype=np.float32),
            indicators=np.ones(self._n, dtype=np.float32),
            error_estimate=1.0,
            dof=self._n,
            budget_remaining=100.0,
        )

    def get_valid_actions(self, state: RefinementState) -> list[int]:
        return [] if self.is_terminal(state) else list(range(self._n))

    def apply_action(self, state: RefinementState, action: int) -> RefinementState:
        ns = state.clone()
        ns.values[action] += 1.0
        ns.error_estimate = state.error_estimate * 0.5
        ns.dof = state.dof + 1
        ns.step = state.step + 1
        ns.budget_remaining = state.budget_remaining - 1.0
        ns.history.append(action)
        return ns

    def is_terminal(self, state: RefinementState) -> bool:
        return (
            state.step >= self._max_steps
            or state.error_estimate < 1e-3
            or state.budget_remaining <= 0.0
        )

    def get_reward(self, state: RefinementState, prev_state: RefinementState) -> float:
        return (prev_state.error_estimate - state.error_estimate) - 0.1

    def get_winner(self, state: RefinementState) -> int:
        return 1 if state.error_estimate < 0.1 else -1

    def to_tensor(self, state: RefinementState) -> np.ndarray:
        return state.values.astype(np.float32)


class _StatefulToyGame(_ToyRefinementGame):
    """Holds mutable per-episode state on the instance; overrides clone."""

    def __init__(self, n_actions: int = 4, max_steps: int = 5) -> None:
        super().__init__(n_actions, max_steps)
        self.touched: list[int] = []

    def apply_action(self, state: RefinementState, action: int) -> RefinementState:
        self.touched.append(action)  # instance mutation
        return super().apply_action(state, action)

    def clone(self) -> RefinementGame:
        cloned = _StatefulToyGame(self._n, self._max_steps)
        cloned.touched = list(self.touched)
        return cloned


# --------------------------------------------------------------------------- #
# RefinementState                                                             #
# --------------------------------------------------------------------------- #


class TestRefinementState:
    def test_clone_is_independent(self) -> None:
        s = RefinementState(
            values=np.array([1.0, 2.0], dtype=np.float32),
            indicators=np.array([0.5, 0.5], dtype=np.float32),
            error_estimate=0.9,
            dof=2,
            history=[1],
        )
        c = s.clone()
        c.values[0] = 99.0
        c.history.append(3)
        assert s.values[0] == 1.0
        assert s.history == [1]

    def test_dict_round_trip(self) -> None:
        s = RefinementState(
            values=np.array([1.0, 2.0], dtype=np.float32),
            indicators=np.array([0.1, 0.2], dtype=np.float32),
            error_estimate=0.3,
            dof=5,
            step=2,
            budget_remaining=42.0,
            history=[0, 1],
        )
        r = RefinementState.from_dict(s.to_dict())
        assert np.allclose(r.values, s.values)
        assert np.allclose(r.indicators, s.indicators)
        assert (r.error_estimate, r.dof, r.step, r.budget_remaining) == (
            s.error_estimate,
            s.dof,
            s.step,
            s.budget_remaining,
        )
        assert r.history == s.history

    def test_satisfies_refinement_like(self) -> None:
        s = RefinementState(
            values=np.zeros(1, dtype=np.float32),
            indicators=np.zeros(1, dtype=np.float32),
        )
        assert isinstance(s, RefinementLike)


# --------------------------------------------------------------------------- #
# PDEState <-> RefinementState converters                                     #
# --------------------------------------------------------------------------- #


class TestPDEStateConverters:
    def test_to_refinement_projects_fields(self) -> None:
        from src.pde.game import PDEState

        pde = PDEState(
            coords=np.zeros((3, 2), dtype=np.float32),
            solution=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            residuals=np.array([-0.5, 0.5, -0.25], dtype=np.float32),
            error_estimate=0.7,
            dof=3,
            step=1,
            budget_remaining=10.0,
            history=[2],
        )
        r = pde.to_refinement()
        assert np.allclose(r.values, [1.0, 2.0, 3.0])
        # indicators are |residuals|
        assert np.allclose(r.indicators, [0.5, 0.5, 0.25])
        assert r.error_estimate == pytest.approx(0.7)
        assert r.dof == 3
        assert r.step == 1
        assert r.budget_remaining == pytest.approx(10.0)
        assert isinstance(r, RefinementLike)

    def test_pde_state_satisfies_refinement_like(self) -> None:
        from src.pde.game import PDEState

        pde = PDEState(
            coords=np.zeros((1, 1), dtype=np.float32),
            solution=np.zeros(1, dtype=np.float32),
            residuals=np.zeros(1, dtype=np.float32),
        )
        assert isinstance(pde, RefinementLike)

    def test_from_refinement_synthesises_coords(self) -> None:
        from src.pde.game import PDEState

        r = RefinementState(
            values=np.array([1.0, 2.0], dtype=np.float32),
            indicators=np.array([0.3, 0.4], dtype=np.float32),
            error_estimate=0.2,
            dof=2,
        )
        pde = PDEState.from_refinement(r)
        assert isinstance(pde, PDEState)
        assert pde.coords.shape == (2, 1)
        assert np.allclose(pde.solution, [1.0, 2.0])
        assert np.allclose(pde.residuals, [0.3, 0.4])

    def test_from_refinement_uses_supplied_coords(self) -> None:
        from src.pde.game import PDEState

        r = RefinementState(
            values=np.array([1.0], dtype=np.float32),
            indicators=np.array([0.1], dtype=np.float32),
        )
        coords = np.array([[0.5, 0.5]], dtype=np.float32)
        pde = PDEState.from_refinement(r, coords=coords)
        assert np.allclose(pde.coords, coords)

    def test_from_refinement_empty_values(self) -> None:
        """Zero-length values synthesise an empty (0, 1) coord grid, no crash."""
        from src.pde.game import PDEState

        r = RefinementState(
            values=np.zeros(0, dtype=np.float32),
            indicators=np.zeros(0, dtype=np.float32),
        )
        pde = PDEState.from_refinement(r)
        assert pde.coords.shape == (0, 1)


# --------------------------------------------------------------------------- #
# RefinementGameConfig (generic; zero field loss)                             #
# --------------------------------------------------------------------------- #


class _DomainCfg(BaseModuleConfig):
    knob_a: float = Field(default=1.0)
    knob_b: int = Field(default=2)


class TestRefinementGameConfig:
    def test_defaults(self) -> None:
        cfg = RefinementGameConfig(name="rg", domain_config=_DomainCfg(name="d"))
        assert cfg.max_steps == 30
        assert cfg.use_intermediate_rewards is False
        assert cfg.reward_discount == 1.0

    def test_reward_discount_bounds(self) -> None:
        with pytest.raises(ValueError):
            RefinementGameConfig(name="rg", domain_config=_DomainCfg(name="d"), reward_discount=1.5)

    @given(a=st.floats(-100, 100), b=st.integers(-1000, 1000))
    def test_domain_config_roundtrip_no_field_loss(self, a: float, b: int) -> None:
        cfg = RefinementGameConfig[_DomainCfg](
            name="rg",
            domain_config=_DomainCfg(name="d", knob_a=a, knob_b=b),
        )
        dumped = cfg.model_dump()
        assert dumped["domain_config"]["knob_a"] == pytest.approx(a)
        assert dumped["domain_config"]["knob_b"] == b
        # The typed accessor keeps the concrete type + values.
        assert cfg.domain_config.knob_a == pytest.approx(a)
        assert cfg.domain_config.knob_b == b


# --------------------------------------------------------------------------- #
# RefinementGameAdapter + real single-agent MCTS                              #
# --------------------------------------------------------------------------- #


class TestRefinementGameAdapter:
    def test_protocol_surface(self) -> None:
        adapter = RefinementGameAdapter(_ToyRefinementGame())
        assert isinstance(adapter.get_state(), np.ndarray)
        assert adapter.get_legal_actions() == [0, 1, 2, 3]
        assert adapter.is_terminal() is False
        assert adapter.get_last_reward() == 0.0  # no action yet
        adapter.apply_action(0)
        assert adapter.state.step == 1
        assert adapter.get_last_reward() != 0.0

    def test_search_mode_is_single_agent(self) -> None:
        from src.mcts.search import SearchMode

        adapter = RefinementGameAdapter(_ToyRefinementGame())
        assert adapter.search_mode is SearchMode.SINGLE_AGENT

    def test_get_state_converts_torch_tensor(self) -> None:
        """A game whose to_tensor returns a torch.Tensor is converted to numpy."""
        import torch

        class _TorchTensorGame(_ToyRefinementGame):
            def to_tensor(self, state: RefinementState) -> object:  # type: ignore[override]
                return torch.zeros(self._n, dtype=torch.float32)

        adapter = RefinementGameAdapter(_TorchTensorGame())
        state = adapter.get_state()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32

    def test_action_space_size_delegates(self) -> None:
        adapter = RefinementGameAdapter(_ToyRefinementGame(n_actions=7))
        assert adapter.action_space_size == 7

    def test_clone_isolates_state(self) -> None:
        adapter = RefinementGameAdapter(_ToyRefinementGame())
        clone = adapter.clone()
        clone.apply_action(0)
        assert adapter.state.step == 0
        assert clone.state.step == 1

    def test_clone_isolates_stateful_game(self) -> None:
        adapter = RefinementGameAdapter(_StatefulToyGame())
        clone = adapter.clone()
        assert clone.game is not adapter.game
        clone.apply_action(1)
        assert adapter.game.touched == []
        assert clone.game.touched == [1]

    def test_reset_and_error_reduction(self) -> None:
        adapter = RefinementGameAdapter(_ToyRefinementGame())
        adapter.apply_action(0)
        assert adapter.error_reduction > 0.0
        adapter.reset()
        assert adapter.state.step == 0
        assert adapter.current_error == 1.0

    def test_real_single_agent_mcts_micro_run(self) -> None:
        from src.mcts.evaluator import RandomEvaluator
        from src.mcts.search import MCTS

        np.random.seed(0)
        game = _ToyRefinementGame(n_actions=4, max_steps=4)
        adapter = RefinementGameAdapter(game)
        mcts = MCTS(
            evaluator=RandomEvaluator(n_actions=game.action_space_size),
            n_simulations=8,
            search_mode=adapter.search_mode,
        )
        steps = 0
        while not adapter.is_terminal() and adapter.get_legal_actions():
            action = mcts.get_action(adapter, temperature=0.0, add_noise=False)
            assert action in adapter.get_legal_actions()
            adapter.apply_action(action)
            mcts.advance(action)
            steps += 1
            assert steps <= 4  # budget respected → terminates
        assert adapter.is_terminal()


# --------------------------------------------------------------------------- #
# RefinementGame ABC + registry                                               #
# --------------------------------------------------------------------------- #


class TestRefinementGameABC:
    def test_default_clone_returns_self(self) -> None:
        game = _ToyRefinementGame()
        assert game.clone() is game

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            RefinementGame()  # type: ignore[abstract]


class TestRegistry:
    def test_register_and_retrieve(self) -> None:
        @register_refinement_game("toy_reg_test")
        class _Registered(_ToyRefinementGame):
            pass

        cls = RefinementGameRegistry().get("toy_reg_test")
        assert issubclass(cls, RefinementGame)
