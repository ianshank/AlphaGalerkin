"""Coverage tests for basis selection game.

Tests cover:
- BasisFunction: Evaluation for fourier, polynomial, rbf types
- BasisSelectionGame: Initialization, state creation, actions, terminal conditions
- Game flow: Multiple steps with error tracking
"""

from __future__ import annotations

import numpy as np
import pytest

import torch
from torch import Tensor

from src.pde.config import BasisSelectionConfig, PDEConfig, PDEGameConfig, PDEType
from src.pde.game import GamePhase, PDEState
from src.pde.games.basis_selection import BasisFunction, BasisSelectionGame
from src.pde.operators import PDEResidual, PoissonOperator

SEED = 42


class SafePoissonOperator(PoissonOperator):
    """PoissonOperator that handles non-grad tensors in residual computation."""

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute residual without requiring autograd."""
        source = self.source_term(coords)
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)
        # Approximate residual as -source (laplacian is zero for non-grad tensors)
        residual_values = -source
        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())
        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives={},
        )
N_COLLOCATION = 50
N_BOUNDARY_PER_FACE = 10
N_CANDIDATES = 8
MAX_BASIS = 5


@pytest.fixture
def pde_config() -> PDEConfig:
    """Minimal PDE config."""
    return PDEConfig(
        name="test_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )


@pytest.fixture
def poisson_operator(pde_config: PDEConfig) -> SafePoissonOperator:
    return SafePoissonOperator(pde_config)


def _make_game_config(
    pde_config: PDEConfig,
    basis_type: str = "fourier",
) -> PDEGameConfig:
    """Create a game config with small sizes for testing."""
    basis_config = BasisSelectionConfig(
        name="test_basis",
        basis_type=basis_type,
        max_basis_functions=MAX_BASIS,
        n_candidate_bases=N_CANDIDATES,
        max_frequency=2,
        n_collocation_points=N_COLLOCATION,
        n_boundary_points_per_face=N_BOUNDARY_PER_FACE,
        seed=SEED,
    )
    return PDEGameConfig(
        name="test_game",
        pde_config=pde_config,
        game_mode="basis_selection",
        basis_config=basis_config,
        max_steps=10,
        computational_budget=20.0,
        error_tolerance=1e-6,
    )


class TestBasisFunction:
    """Tests for BasisFunction evaluation."""

    @pytest.fixture
    def coords_2d(self) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        return rng.uniform(0, 1, (20, 2)).astype(np.float32)

    def test_fourier_evaluation(self, coords_2d: np.ndarray) -> None:
        bf = BasisFunction(type="fourier", params={"k_x": 1, "k_y": 0, "phase": 0.0}, index=0)
        values = bf.evaluate(coords_2d)
        assert values.shape == (20,)
        assert values.dtype == np.float32
        # Fourier values should be in [-1, 1]
        assert np.all(np.abs(values) <= 1.0 + 1e-6)

    def test_fourier_with_phase(self, coords_2d: np.ndarray) -> None:
        bf = BasisFunction(
            type="fourier", params={"k_x": 1, "k_y": 1, "phase": np.pi / 2}, index=0
        )
        values = bf.evaluate(coords_2d)
        assert values.shape == (20,)

    def test_polynomial_evaluation(self, coords_2d: np.ndarray) -> None:
        bf = BasisFunction(
            type="polynomial", params={"degree_x": 2, "degree_y": 1}, index=0
        )
        values = bf.evaluate(coords_2d)
        assert values.shape == (20,)
        # x^2 * y for coords in [0,1]
        expected = coords_2d[:, 0] ** 2 * coords_2d[:, 1]
        np.testing.assert_allclose(values, expected, atol=1e-5)

    def test_rbf_evaluation(self, coords_2d: np.ndarray) -> None:
        bf = BasisFunction(
            type="rbf",
            params={"center_x": 0.5, "center_y": 0.5, "sigma": 0.2},
            index=0,
        )
        values = bf.evaluate(coords_2d)
        assert values.shape == (20,)
        # RBF values should be positive (Gaussian)
        assert np.all(values > 0)
        assert np.all(values <= 1.0 + 1e-6)

    def test_unknown_type_raises(self, coords_2d: np.ndarray) -> None:
        bf = BasisFunction(type="unknown", params={}, index=0)
        with pytest.raises(ValueError, match="Unknown basis type"):
            bf.evaluate(coords_2d)

    def test_fourier_1d_fallback(self) -> None:
        """Test Fourier basis with 1D coords (no y dimension)."""
        coords_1d = np.random.rand(10, 1).astype(np.float32)
        bf = BasisFunction(type="fourier", params={"k_x": 1, "k_y": 0, "phase": 0.0}, index=0)
        values = bf.evaluate(coords_1d)
        assert values.shape == (10,)


class TestBasisSelectionGame:
    """Tests for BasisSelectionGame."""

    @pytest.fixture
    def game(self, pde_config: PDEConfig, poisson_operator: PoissonOperator) -> BasisSelectionGame:
        config = _make_game_config(pde_config, basis_type="fourier")
        return BasisSelectionGame(poisson_operator, config)

    def test_initialization(self, game: BasisSelectionGame) -> None:
        assert game.name == "basis_selection"
        assert game.action_space_size == N_CANDIDATES
        assert game.state_channels == 3 + MAX_BASIS

    def test_get_initial_state(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert isinstance(state, PDEState)
        assert state.step == 0
        assert state.dof == 0
        assert state.error_estimate > 0
        assert state.phase == GamePhase.INITIAL
        assert len(state.history) == 0
        assert state.solution.shape[0] == N_COLLOCATION

    def test_get_valid_actions_initial(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        assert len(valid) == N_CANDIDATES

    def test_get_action_mask(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        mask = game.get_action_mask(state)
        assert mask.shape == (N_CANDIDATES,)
        assert mask.all()  # All actions valid initially

    def test_apply_action(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        valid_actions = game.get_valid_actions(state)
        action = valid_actions[0]
        new_state = game.apply_action(state, action)

        assert new_state.step == 1
        assert new_state.dof == 1
        assert action in new_state.history
        assert new_state.solution is not None

    def test_apply_duplicate_action_raises(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        with pytest.raises(ValueError, match="already selected"):
            game.apply_action(new_state, 0)

    def test_apply_invalid_action_raises(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(state, -1)
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(state, game.action_space_size + 10)

    def test_is_terminal_initial(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert not game.is_terminal(state)

    def test_is_terminal_max_basis(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        # Exhaust all basis selections up to max
        for i in range(min(MAX_BASIS, N_CANDIDATES)):
            if game.is_terminal(state):
                break
            valid = game.get_valid_actions(state)
            if not valid:
                break
            state = game.apply_action(state, valid[0])
        # Should eventually terminate
        assert game.is_terminal(state) or state.dof == MAX_BASIS

    def test_get_reward(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        reward = game.get_reward(new_state, state)
        assert isinstance(reward, float)

    def test_compute_exact_error(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        errors = game.compute_exact_error(state)
        assert "l2" in errors
        assert "h1" in errors
        assert "linf" in errors
        assert "residual" in errors
        assert errors["l2"] >= 0

    def test_action_to_string(self, game: BasisSelectionGame) -> None:
        s = game.action_to_string(0)
        assert isinstance(s, str)
        assert "fourier" in s

    def test_action_to_string_invalid(self, game: BasisSelectionGame) -> None:
        s = game.action_to_string(-1)
        assert "invalid" in s

    def test_to_tensor(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.ndim >= 2
        assert tensor.shape[0] == game.state_channels

    def test_get_result(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        initial_error = state.error_estimate
        error_history = [initial_error]

        new_state = game.apply_action(state, 0)
        error_history.append(new_state.error_estimate)

        result = game.get_result(new_state, error_history)
        assert result.final_error >= 0
        assert result.final_dof == 1
        assert result.n_steps == 1
        assert result.termination_reason in ("converged", "max_basis", "budget_exhausted", "max_steps")

    def test_valid_actions_shrink(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        initial_valid = len(game.get_valid_actions(state))
        new_state = game.apply_action(state, 0)
        assert len(game.get_valid_actions(new_state)) == initial_valid - 1


class TestBasisSelectionPolynomial:
    """Tests with polynomial basis type."""

    def test_polynomial_game(self, pde_config: PDEConfig, poisson_operator: PoissonOperator) -> None:
        config = _make_game_config(pde_config, basis_type="polynomial")
        game = BasisSelectionGame(poisson_operator, config)
        state = game.get_initial_state()
        assert game.action_space_size > 0
        valid = game.get_valid_actions(state)
        assert len(valid) > 0
        new_state = game.apply_action(state, valid[0])
        assert new_state.dof == 1


class TestBasisSelectionRBF:
    """Tests with RBF basis type."""

    def test_rbf_game(self, pde_config: PDEConfig, poisson_operator: PoissonOperator) -> None:
        config = _make_game_config(pde_config, basis_type="rbf")
        game = BasisSelectionGame(poisson_operator, config)
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        assert len(valid) > 0
        s = game.action_to_string(valid[0])
        assert "rbf" in s
