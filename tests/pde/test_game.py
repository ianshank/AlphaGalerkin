"""Tests for PDE game abstraction layer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import PDEConfig, PDEGameConfig, PDEType
from src.pde.game import GamePhase, PDEGame, PDEResult, PDEState
from src.pde.operators import PoissonOperator


class TestGamePhase:
    """Tests for PDE GamePhase enum."""

    def test_phase_values(self) -> None:
        assert GamePhase.INITIAL == "initial"
        assert GamePhase.EXPLORING == "exploring"
        assert GamePhase.REFINING == "refining"
        assert GamePhase.CONVERGED == "converged"
        assert GamePhase.BUDGET_EXHAUSTED == "budget_exhausted"

    def test_phase_is_string(self) -> None:
        assert isinstance(GamePhase.INITIAL, str)

    def test_all_phases_exist(self) -> None:
        phases = list(GamePhase)
        assert len(phases) == 5

    def test_phase_from_value(self) -> None:
        assert GamePhase("initial") == GamePhase.INITIAL
        assert GamePhase("converged") == GamePhase.CONVERGED


class TestPDEState:
    """Tests for PDEState dataclass."""

    @pytest.fixture
    def sample_state(self) -> PDEState:
        n = 16
        rng = np.random.default_rng(42)
        return PDEState(
            coords=rng.random((n, 2)).astype(np.float32),
            solution=np.zeros(n, dtype=np.float32),
            residuals=rng.random(n).astype(np.float32),
            error_estimate=0.5,
            dof=10,
            step=3,
            budget_remaining=100.0,
        )

    def test_create_state(self) -> None:
        state = PDEState(
            coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            solution=np.array([0.0, 0.0], dtype=np.float32),
            residuals=np.array([0.1, 0.2], dtype=np.float32),
        )
        assert state.n_points == 2
        assert state.dim == 2

    def test_n_points(self, sample_state: PDEState) -> None:
        assert sample_state.n_points == 16

    def test_dim(self, sample_state: PDEState) -> None:
        assert sample_state.dim == 2

    def test_dim_1d(self) -> None:
        state = PDEState(
            coords=np.zeros((4, 1), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.zeros(4, dtype=np.float32),
        )
        assert state.dim == 1

    def test_dim_3d(self) -> None:
        state = PDEState(
            coords=np.zeros((4, 3), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.zeros(4, dtype=np.float32),
        )
        assert state.dim == 3

    def test_n_basis_empty(self, sample_state: PDEState) -> None:
        assert sample_state.n_basis == 0

    def test_n_basis_with_coefficients(self) -> None:
        state = PDEState(
            coords=np.zeros((4, 2), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.zeros(4, dtype=np.float32),
            basis_coefficients=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
        assert state.n_basis == 3

    def test_n_basis_none(self) -> None:
        state = PDEState(
            coords=np.zeros((4, 2), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.zeros(4, dtype=np.float32),
            basis_coefficients=None,
        )
        assert state.n_basis == 0

    def test_clone(self, sample_state: PDEState) -> None:
        clone = sample_state.clone()
        assert clone is not sample_state
        assert np.array_equal(clone.coords, sample_state.coords)
        assert np.array_equal(clone.solution, sample_state.solution)
        assert np.array_equal(clone.residuals, sample_state.residuals)
        assert clone.error_estimate == sample_state.error_estimate
        assert clone.dof == sample_state.dof
        assert clone.step == sample_state.step
        assert clone.budget_remaining == sample_state.budget_remaining
        assert clone.phase == sample_state.phase

    def test_clone_deep_copy(self, sample_state: PDEState) -> None:
        clone = sample_state.clone()
        clone.coords[0, 0] = 999.0
        assert sample_state.coords[0, 0] != 999.0

    def test_clone_deep_copy_history(self, sample_state: PDEState) -> None:
        sample_state.history = [1, 2, 3]
        clone = sample_state.clone()
        clone.history.append(4)
        assert len(sample_state.history) == 3

    def test_clone_with_optional_fields(self) -> None:
        state = PDEState(
            coords=np.zeros((4, 2), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.zeros(4, dtype=np.float32),
            basis_coefficients=np.array([0.1, 0.2], dtype=np.float32),
            mesh_levels=np.array([0, 1, 2, 3], dtype=np.int32),
            polynomial_degrees=np.array([1, 2, 1, 1], dtype=np.int32),
        )
        clone = state.clone()
        assert np.array_equal(clone.basis_coefficients, state.basis_coefficients)
        assert np.array_equal(clone.mesh_levels, state.mesh_levels)
        assert np.array_equal(clone.polynomial_degrees, state.polynomial_degrees)

    def test_clone_none_optional_fields(self) -> None:
        state = PDEState(
            coords=np.zeros((4, 2), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.zeros(4, dtype=np.float32),
        )
        clone = state.clone()
        assert clone.basis_coefficients is None
        assert clone.mesh_levels is None
        assert clone.polynomial_degrees is None

    def test_to_dict(self, sample_state: PDEState) -> None:
        d = sample_state.to_dict()
        assert "coords" in d
        assert "solution" in d
        assert "residuals" in d
        assert "error_estimate" in d
        assert d["error_estimate"] == 0.5
        assert d["dof"] == 10
        assert d["step"] == 3
        assert d["phase"] == "initial"

    def test_from_dict(self, sample_state: PDEState) -> None:
        d = sample_state.to_dict()
        restored = PDEState.from_dict(d)
        assert restored.n_points == sample_state.n_points
        assert restored.error_estimate == sample_state.error_estimate
        assert restored.dof == sample_state.dof
        assert restored.step == sample_state.step
        assert np.allclose(restored.coords, sample_state.coords)

    def test_roundtrip_dict(self, sample_state: PDEState) -> None:
        d = sample_state.to_dict()
        restored = PDEState.from_dict(d)
        d2 = restored.to_dict()
        assert d["error_estimate"] == d2["error_estimate"]
        assert d["dof"] == d2["dof"]
        assert d["step"] == d2["step"]
        assert d["phase"] == d2["phase"]

    def test_from_dict_with_optional(self) -> None:
        state = PDEState(
            coords=np.zeros((4, 2), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.zeros(4, dtype=np.float32),
            basis_coefficients=np.array([0.5], dtype=np.float32),
            mesh_levels=np.array([1, 2, 3, 4], dtype=np.int32),
        )
        d = state.to_dict()
        restored = PDEState.from_dict(d)
        assert restored.basis_coefficients is not None
        assert restored.mesh_levels is not None
        assert np.allclose(restored.basis_coefficients, state.basis_coefficients)

    def test_from_dict_none_optional(self) -> None:
        d = {
            "coords": [[0, 0], [1, 1]],
            "solution": [0, 0],
            "residuals": [0.1, 0.2],
            "basis_coefficients": None,
            "mesh_levels": None,
            "polynomial_degrees": None,
            "error_estimate": 0.5,
            "dof": 0,
            "step": 0,
            "budget_remaining": 100.0,
            "phase": "initial",
            "history": [],
        }
        state = PDEState.from_dict(d)
        assert state.basis_coefficients is None
        assert state.mesh_levels is None

    def test_default_values(self) -> None:
        state = PDEState(
            coords=np.zeros((4, 2), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.zeros(4, dtype=np.float32),
        )
        assert state.error_estimate == 1.0
        assert state.dof == 0
        assert state.step == 0
        assert state.budget_remaining == 1e6
        assert state.phase == GamePhase.INITIAL
        assert state.history == []


class TestPDEResult:
    """Tests for PDEResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> PDEResult:
        return PDEResult(
            final_error=0.001,
            final_dof=50,
            n_steps=10,
            converged=True,
            l2_error=0.001,
            h1_error=0.002,
            linf_error=0.005,
            residual_norm=0.0001,
            error_reduction_rate=0.01,
            dof_efficiency=0.0002,
            compute_efficiency=0.001,
            initial_error=1.0,
            best_error=0.001,
            average_error=0.1,
            error_history=[1.0, 0.5, 0.1, 0.001],
            termination_reason="converged",
            budget_used=10.0,
        )

    def test_create_result(self, sample_result: PDEResult) -> None:
        assert sample_result.converged is True
        assert sample_result.final_error == 0.001
        assert sample_result.final_dof == 50
        assert sample_result.n_steps == 10

    def test_result_fields(self, sample_result: PDEResult) -> None:
        assert sample_result.l2_error == 0.001
        assert sample_result.h1_error == 0.002
        assert sample_result.linf_error == 0.005
        assert sample_result.residual_norm == 0.0001
        assert sample_result.termination_reason == "converged"

    def test_to_dict(self, sample_result: PDEResult) -> None:
        d = sample_result.to_dict()
        assert d["final_error"] == 0.001
        assert d["converged"] is True
        assert d["termination_reason"] == "converged"
        assert len(d["error_history"]) == 4
        assert d["budget_used"] == 10.0

    def test_to_dict_not_converged(self) -> None:
        result = PDEResult(
            final_error=0.01,
            final_dof=20,
            n_steps=5,
            converged=False,
            l2_error=0.01,
            h1_error=0.02,
            linf_error=0.05,
            residual_norm=0.001,
            error_reduction_rate=0.1,
            dof_efficiency=0.005,
            compute_efficiency=0.01,
            initial_error=1.0,
            best_error=0.01,
            average_error=0.3,
            error_history=[1.0, 0.5, 0.01],
            termination_reason="max_steps",
            budget_used=5.0,
        )
        d = result.to_dict()
        assert d["converged"] is False
        assert d["termination_reason"] == "max_steps"


class TestPDEGameAbstract:
    """Tests for PDEGame abstract interface."""

    def test_cannot_instantiate_abstract(self) -> None:
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_config = PDEGameConfig(name="test", pde_config=pde_config)
        operator = PoissonOperator(pde_config)
        with pytest.raises(TypeError):
            PDEGame(operator, game_config)  # type: ignore[abstract]

    def test_abstract_methods_exist(self) -> None:
        """Verify required abstract methods are defined."""
        abstracts = set()
        for name in dir(PDEGame):
            obj = getattr(PDEGame, name, None)
            if getattr(obj, "__isabstractmethod__", False):
                abstracts.add(name)
        expected = {
            "action_space_size",
            "state_channels",
            "get_initial_state",
            "get_valid_actions",
            "get_action_mask",
            "apply_action",
            "get_reward",
            "is_terminal",
            "get_result",
            "compute_exact_error",
            "to_tensor",
        }
        assert expected.issubset(abstracts)

    def test_class_attributes(self) -> None:
        assert PDEGame.name == "pde_game"
        assert PDEGame.description == "Abstract PDE game"


class ConcretePDEGame(PDEGame):
    """Minimal concrete PDEGame for testing base class methods."""

    name = "test_pde_game"

    @property
    def action_space_size(self) -> int:
        return 10

    @property
    def state_channels(self) -> int:
        return 3

    def get_initial_state(self) -> PDEState:
        return PDEState(
            coords=np.zeros((4, 2), dtype=np.float32),
            solution=np.zeros(4, dtype=np.float32),
            residuals=np.ones(4, dtype=np.float32),
            error_estimate=1.0,
        )

    def get_valid_actions(self, state: PDEState) -> list[int]:
        return list(range(self.action_space_size))

    def get_action_mask(self, state: PDEState) -> NDArray[np.bool_]:
        return np.ones(self.action_space_size, dtype=bool)

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        new = state.clone()
        new.step += 1
        new.error_estimate *= 0.5
        new.history.append(action)
        return new

    def get_reward(self, state: PDEState, prev_state: PDEState) -> float:
        return prev_state.error_estimate - state.error_estimate

    def is_terminal(self, state: PDEState) -> bool:
        return state.step >= 5 or state.error_estimate < 0.01

    def get_result(self, state: PDEState, error_history: list[float]) -> PDEResult:
        return PDEResult(
            final_error=state.error_estimate,
            final_dof=state.dof,
            n_steps=state.step,
            converged=state.error_estimate < 0.01,
            l2_error=state.error_estimate,
            h1_error=state.error_estimate,
            linf_error=state.error_estimate,
            residual_norm=0.0,
            error_reduction_rate=0.0,
            dof_efficiency=0.0,
            compute_efficiency=0.0,
            initial_error=1.0,
            best_error=state.error_estimate,
            average_error=state.error_estimate,
            error_history=error_history,
            termination_reason="test",
            budget_used=0.0,
        )

    def compute_exact_error(self, state: PDEState) -> dict[str, float]:
        return {
            "l2": state.error_estimate,
            "h1": state.error_estimate,
            "linf": state.error_estimate,
            "residual": 0.0,
        }

    def to_tensor(self, state: PDEState) -> Float[Tensor, "channels height width"]:
        return torch.zeros(self.state_channels, 2, 2)


class TestPDEGameConcreteMethods:
    """Tests for concrete (non-abstract) methods on PDEGame."""

    @pytest.fixture
    def game(self) -> ConcretePDEGame:
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_config = PDEGameConfig(name="test", pde_config=pde_config)
        operator = PoissonOperator(pde_config)
        return ConcretePDEGame(operator, game_config)

    def test_validate_action_valid(self, game: ConcretePDEGame) -> None:
        state = game.get_initial_state()
        assert game.validate_action(state, 0) is True
        assert game.validate_action(state, 9) is True

    def test_validate_action_negative(self, game: ConcretePDEGame) -> None:
        state = game.get_initial_state()
        assert game.validate_action(state, -1) is False

    def test_validate_action_out_of_range(self, game: ConcretePDEGame) -> None:
        state = game.get_initial_state()
        assert game.validate_action(state, 100) is False

    def test_action_to_string(self, game: ConcretePDEGame) -> None:
        s = game.action_to_string(5)
        assert s == "action_5"

    def test_get_symmetries_default(self, game: ConcretePDEGame) -> None:
        state = game.get_initial_state()
        policy = np.ones(10, dtype=np.float32) / 10
        syms = game.get_symmetries(state, policy)
        assert len(syms) == 1
        assert syms[0][0] is state
        assert np.array_equal(syms[0][1], policy)

    def test_batch_to_tensor(self, game: ConcretePDEGame) -> None:
        states = [game.get_initial_state() for _ in range(3)]
        batch = game.batch_to_tensor(states, device="cpu")
        assert batch.shape == (3, 3, 2, 2)
        assert batch.device.type == "cpu"

    def test_batch_to_tensor_single(self, game: ConcretePDEGame) -> None:
        states = [game.get_initial_state()]
        batch = game.batch_to_tensor(states, device="cpu")
        assert batch.shape == (1, 3, 2, 2)

    def test_clone(self, game: ConcretePDEGame) -> None:
        clone = game.clone()
        assert isinstance(clone, ConcretePDEGame)
        assert clone is not game

    def test_repr(self, game: ConcretePDEGame) -> None:
        r = repr(game)
        assert "test_pde_game" in r

    def test_get_phase_terminal_converged(self, game: ConcretePDEGame) -> None:
        state = game.get_initial_state()
        state.error_estimate = 1e-6
        phase = game.get_phase(state)
        assert phase == GamePhase.CONVERGED

    def test_get_phase_terminal_budget(self, game: ConcretePDEGame) -> None:
        state = game.get_initial_state()
        state.step = 10  # triggers is_terminal
        phase = game.get_phase(state)
        assert phase == GamePhase.BUDGET_EXHAUSTED

    def test_get_phase_initial(self, game: ConcretePDEGame) -> None:
        state = game.get_initial_state()
        state.step = 0
        state.error_estimate = 1.0
        # Not terminal, step < early_phase_step_threshold
        phase = game.get_phase(state)
        assert phase == GamePhase.INITIAL

    def test_game_loop(self, game: ConcretePDEGame) -> None:
        """Test a basic game loop."""
        state = game.get_initial_state()
        error_history = [state.error_estimate]

        while not game.is_terminal(state):
            actions = game.get_valid_actions(state)
            assert len(actions) > 0
            prev = state
            state = game.apply_action(state, actions[0])
            reward = game.get_reward(state, prev)
            error_history.append(state.error_estimate)
            assert reward >= 0  # error should decrease

        result = game.get_result(state, error_history)
        assert result.n_steps > 0
        assert result.final_error < 1.0

    def test_game_loop_terminates(self, game: ConcretePDEGame) -> None:
        """Ensure the game eventually terminates."""
        state = game.get_initial_state()
        steps = 0
        while not game.is_terminal(state) and steps < 100:
            state = game.apply_action(state, 0)
            steps += 1
        assert game.is_terminal(state)
