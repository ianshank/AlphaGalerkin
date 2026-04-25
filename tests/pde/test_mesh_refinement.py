"""Tests for MeshRefinementGame."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pde.config import (
    MeshRefinementConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
    RefinementStrategy,
)
from src.pde.game import GamePhase, PDEState
from src.pde.games.mesh_refinement import Mesh, MeshElement, MeshRefinementGame
from src.pde.operators import PoissonOperator

# ---- Mesh tests ----


class TestMeshElement:
    """Tests for MeshElement dataclass."""

    def test_create_element(self) -> None:
        vertices = np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1]],
            dtype=np.float32,
        )
        elem = MeshElement(
            index=0,
            vertices=vertices,
            center=np.array([0.5, 0.5], dtype=np.float32),
            size=1.414,
        )
        assert elem.index == 0
        assert elem.level == 0
        assert elem.polynomial_degree == 1
        assert elem.is_leaf is True
        assert elem.parent is None

    def test_is_leaf_true(self) -> None:
        elem = MeshElement(
            index=0,
            vertices=np.zeros((4, 2), dtype=np.float32),
            center=np.zeros(2, dtype=np.float32),
            size=1.0,
        )
        assert elem.is_leaf is True

    def test_is_leaf_with_children(self) -> None:
        elem = MeshElement(
            index=0,
            vertices=np.zeros((4, 2), dtype=np.float32),
            center=np.zeros(2, dtype=np.float32),
            size=1.0,
            children=[1, 2, 3, 4],
        )
        assert elem.is_leaf is False

    def test_default_values(self) -> None:
        elem = MeshElement(
            index=5,
            vertices=np.zeros((4, 2), dtype=np.float32),
            center=np.zeros(2, dtype=np.float32),
            size=0.5,
        )
        assert elem.level == 0
        assert elem.polynomial_degree == 1
        assert elem.parent is None
        assert elem.children == []

    def test_with_parent(self) -> None:
        elem = MeshElement(
            index=5,
            vertices=np.zeros((4, 2), dtype=np.float32),
            center=np.zeros(2, dtype=np.float32),
            size=0.5,
            parent=0,
            level=1,
        )
        assert elem.parent == 0
        assert elem.level == 1


class TestMesh:
    """Tests for Mesh class."""

    def test_create_2d_mesh(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=4,
        )
        assert mesh.n_elements == 16  # 4x4
        assert mesh.dim == 2

    def test_create_1d_mesh(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0], dtype=np.float32),
            domain_max=np.array([1.0], dtype=np.float32),
            initial_resolution=4,
        )
        assert mesh.n_elements == 4
        assert mesh.dim == 1

    def test_create_3d_mesh(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        assert mesh.n_elements == 8  # 2x2x2
        assert mesh.dim == 3

    def test_unsupported_dimension(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            Mesh(
                domain_min=np.zeros(5, dtype=np.float32),
                domain_max=np.ones(5, dtype=np.float32),
                initial_resolution=2,
            )

    def test_zero_dimension(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            Mesh(
                domain_min=np.array([], dtype=np.float32),
                domain_max=np.array([], dtype=np.float32),
                initial_resolution=2,
            )

    def test_leaf_elements_initial(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        leaves = mesh.leaf_elements
        assert len(leaves) == 4
        for elem in leaves:
            assert elem.is_leaf is True

    def test_n_dof(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        # Each element: polynomial_degree=1, dim=2 => (1+1)^2 = 4 DOF
        # 4 elements => 16 DOF
        assert mesh.n_dof == 16

    def test_n_dof_1d(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0], dtype=np.float32),
            domain_max=np.array([1.0], dtype=np.float32),
            initial_resolution=4,
        )
        # Each element: (1+1)^1 = 2 DOF, 4 elements => 8 DOF
        assert mesh.n_dof == 8

    def test_get_element_centers(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        centers = mesh.get_element_centers()
        assert centers.shape == (4, 2)
        # All centers should be within domain
        assert np.all(centers >= 0.0)
        assert np.all(centers <= 1.0)

    def test_get_element_centers_1d(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0], dtype=np.float32),
            domain_max=np.array([1.0], dtype=np.float32),
            initial_resolution=4,
        )
        centers = mesh.get_element_centers()
        assert centers.shape == (4, 1)

    def test_get_element_sizes(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        sizes = mesh.get_element_sizes()
        assert sizes.shape == (4,)
        assert np.all(sizes > 0)
        # All sizes should be equal for uniform mesh
        np.testing.assert_allclose(sizes, sizes[0], atol=1e-6)

    def test_h_refinement_2d(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        initial_count = mesh.n_elements
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        assert len(children) == 4  # 2^dim children
        assert mesh.n_elements == initial_count + 4
        # Original element should no longer be a leaf
        assert mesh.elements[0].is_leaf is False

    def test_h_refinement_1d(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0], dtype=np.float32),
            domain_max=np.array([1.0], dtype=np.float32),
            initial_resolution=2,
        )
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        assert len(children) == 2  # 2^1 children in 1D

    def test_h_refinement_3d(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        assert len(children) == 8  # 2^3 children in 3D

    def test_p_refinement(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        original_degree = mesh.elements[0].polynomial_degree
        mesh.refine_element(0, RefinementStrategy.P_REFINEMENT)
        assert mesh.elements[0].polynomial_degree == original_degree + 1
        # Element count should not change for p-refinement
        assert mesh.n_elements == 4

    def test_p_refinement_increases_dof(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        dof_before = mesh.n_dof
        mesh.refine_element(0, RefinementStrategy.P_REFINEMENT)
        dof_after = mesh.n_dof
        assert dof_after > dof_before

    def test_hp_refinement(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        # Level 0 element -> h-refinement (level < 2)
        result = mesh.refine_element(0, RefinementStrategy.HP_REFINEMENT)
        assert len(result) == 4  # h-refinement for level < 2

    def test_h_refinement_updates_level(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        for child_idx in children:
            assert mesh.elements[child_idx].level == 1

    def test_h_refinement_child_size(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        parent_size = mesh.elements[0].size
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        for child_idx in children:
            assert mesh.elements[child_idx].size == pytest.approx(parent_size / 2, abs=1e-6)

    def test_h_refinement_parent_child_relationship(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        for child_idx in children:
            assert mesh.elements[child_idx].parent == 0
        assert mesh.elements[0].children == children

    def test_leaf_elements_after_refinement(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        leaves = mesh.leaf_elements
        # 3 original leaves + 4 new children = 7
        assert len(leaves) == 7

    def test_nested_refinement(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        # Refine element 0
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        # Refine first child
        grandchildren = mesh.refine_element(children[0], RefinementStrategy.H_REFINEMENT)
        for gc_idx in grandchildren:
            assert mesh.elements[gc_idx].level == 2


# ---- MeshRefinementGame tests ----


@pytest.fixture
def poisson_operator() -> PoissonOperator:
    config = PDEConfig(name="test_poisson", pde_type=PDEType.POISSON)
    return PoissonOperator(config)


@pytest.fixture
def mesh_config() -> MeshRefinementConfig:
    return MeshRefinementConfig(
        name="test_mesh",
        initial_resolution=2,
        max_refinement_level=3,
        refinement_strategy=RefinementStrategy.H_REFINEMENT,
        n_candidate_elements=64,
    )


@pytest.fixture
def game_config(mesh_config: MeshRefinementConfig) -> PDEGameConfig:
    pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
    return PDEGameConfig(
        name="test_game",
        pde_config=pde_config,
        game_mode="mesh_refinement",
        mesh_config=mesh_config,
        max_steps=10,
        max_dof=500,
        error_tolerance=1e-4,
    )


@pytest.fixture
def game(poisson_operator: PoissonOperator, game_config: PDEGameConfig) -> MeshRefinementGame:
    return MeshRefinementGame(poisson_operator, game_config)


class TestMeshRefinementGameInit:
    """Tests for game initialization."""

    def test_creation(self, game: MeshRefinementGame) -> None:
        assert game.name == "mesh_refinement"

    def test_description(self, game: MeshRefinementGame) -> None:
        assert game.description == "Adaptive mesh refinement game"

    def test_action_space_size(self, game: MeshRefinementGame) -> None:
        assert game.action_space_size > 0

    def test_state_channels(self, game: MeshRefinementGame) -> None:
        assert game.state_channels == 5

    def test_mesh_initialized(self, game: MeshRefinementGame) -> None:
        assert game.mesh is not None
        assert game.mesh.dim == 2


class TestMeshRefinementGameInitialState:
    """Tests for initial state."""

    def test_initial_state(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert isinstance(state, PDEState)
        assert state.step == 0
        assert state.phase == GamePhase.INITIAL

    def test_initial_state_has_coords(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert state.coords is not None
        assert state.coords.ndim == 2
        assert state.coords.shape[1] == 2

    def test_initial_state_zero_solution(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        np.testing.assert_array_equal(state.solution, np.zeros_like(state.solution))

    def test_initial_state_has_residuals(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert state.residuals is not None
        assert len(state.residuals) == state.n_points

    def test_initial_state_has_mesh_levels(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert state.mesh_levels is not None
        np.testing.assert_array_equal(state.mesh_levels, np.zeros_like(state.mesh_levels))

    def test_initial_state_positive_error(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert state.error_estimate > 0

    def test_initial_state_has_dof(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert state.dof > 0

    def test_initial_state_has_budget(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert state.budget_remaining > 0

    def test_initial_state_empty_history(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert state.history == []

    def test_initial_state_resets_mesh(self, game: MeshRefinementGame) -> None:
        """Calling get_initial_state should reset the mesh."""
        state1 = game.get_initial_state()
        n_elements_1 = len(game.mesh.elements)
        actions = game.get_valid_actions(state1)
        if actions:
            game.apply_action(state1, actions[0])
        state2 = game.get_initial_state()
        n_elements_2 = len(game.mesh.elements)
        assert n_elements_2 == n_elements_1


class TestMeshRefinementGameActions:
    """Tests for action handling."""

    def test_valid_actions(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        assert len(actions) > 0

    def test_valid_actions_bounded(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        for a in actions:
            assert a >= 0

    def test_action_mask(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        mask = game.get_action_mask(state)
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_action_mask_shape(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        mask = game.get_action_mask(state)
        assert mask.shape == (game.action_space_size,)

    def test_apply_action(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        new_state = game.apply_action(state, actions[0])
        assert new_state.step == 1
        assert actions[0] in new_state.history

    def test_apply_action_increases_dof(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        initial_dof = state.dof
        actions = game.get_valid_actions(state)
        new_state = game.apply_action(state, actions[0])
        assert new_state.dof > initial_dof

    def test_apply_action_decreases_budget(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        initial_budget = state.budget_remaining
        actions = game.get_valid_actions(state)
        new_state = game.apply_action(state, actions[0])
        assert new_state.budget_remaining < initial_budget

    def test_apply_action_updates_coords(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        initial_n_points = state.n_points
        actions = game.get_valid_actions(state)
        new_state = game.apply_action(state, actions[0])
        # After h-refinement, should have more points
        assert new_state.n_points > initial_n_points

    def test_apply_action_updates_mesh_levels(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        new_state = game.apply_action(state, actions[0])
        assert new_state.mesh_levels is not None
        assert np.max(new_state.mesh_levels) >= 0

    def test_apply_invalid_action_raises(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        with pytest.raises(ValueError):
            game.apply_action(state, 9999)

    def test_apply_multiple_actions(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        for _ in range(3):
            if game.is_terminal(state):
                break
            actions = game.get_valid_actions(state)
            if not actions:
                break
            state = game.apply_action(state, actions[0])
        assert state.step <= 3


class TestMeshRefinementGameReward:
    """Tests for reward computation."""

    def test_reward_basic(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        new_state = game.apply_action(state, actions[0])
        reward = game.get_reward(new_state, state)
        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_reward_accounts_for_dof_cost(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        new_state = game.apply_action(state, actions[0])
        reward = game.get_reward(new_state, state)
        assert np.isfinite(reward)

    def test_reward_convergence_bonus(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        prev_state = state.clone()
        state.error_estimate = 1e-6
        state.dof = prev_state.dof
        reward = game.get_reward(state, prev_state)
        assert reward > 0


class TestMeshRefinementGameTerminal:
    """Tests for terminal conditions."""

    def test_not_terminal_initial(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert game.is_terminal(state) is False

    def test_terminal_low_error(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        state.error_estimate = 1e-6  # below tolerance
        assert game.is_terminal(state) is True

    def test_terminal_max_dof(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        state.dof = 10000  # above max_dof (500)
        assert game.is_terminal(state) is True

    def test_terminal_budget_exhausted(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        state.budget_remaining = 0
        assert game.is_terminal(state) is True

    def test_terminal_max_steps(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        state.step = game.config.max_steps
        assert game.is_terminal(state) is True


class TestMeshRefinementGameResult:
    """Tests for game result generation."""

    def test_get_result(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        state = game.apply_action(state, actions[0])
        error_history = [1.0, state.error_estimate]
        result = game.get_result(state, error_history)
        assert result.n_steps == 1
        assert result.final_dof > 0
        assert len(result.error_history) == 2

    def test_get_result_converged(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        state.error_estimate = 1e-6
        result = game.get_result(state, [1.0, 1e-6])
        assert result.converged is True
        assert result.termination_reason == "converged"

    def test_get_result_max_dof(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        state.dof = 10000
        result = game.get_result(state, [1.0])
        assert result.termination_reason == "max_dof"

    def test_get_result_budget_exhausted(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        state.budget_remaining = 0
        result = game.get_result(state, [1.0])
        assert result.termination_reason == "budget_exhausted"

    def test_get_result_empty_history(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        result = game.get_result(state, [])
        assert result.error_reduction_rate == 0.0

    def test_get_result_efficiency_metrics(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        state = game.apply_action(state, actions[0])
        error_history = [1.0, state.error_estimate]
        result = game.get_result(state, error_history)
        assert isinstance(result.error_reduction_rate, float)
        assert isinstance(result.dof_efficiency, float)
        assert isinstance(result.compute_efficiency, float)


class TestMeshRefinementGameError:
    """Tests for error computation."""

    def test_compute_exact_error_keys(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        errors = game.compute_exact_error(state)
        assert "l2" in errors
        assert "h1" in errors
        assert "linf" in errors
        assert "residual" in errors

    def test_compute_exact_error_non_negative(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        errors = game.compute_exact_error(state)
        assert errors["l2"] >= 0
        assert errors["h1"] >= 0
        assert errors["linf"] >= 0
        assert errors["residual"] >= 0


class TestMeshRefinementGameTensor:
    """Tests for tensor conversion."""

    def test_to_tensor(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.ndim >= 2
        assert tensor.shape[0] == game.state_channels

    def test_to_tensor_dtype(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.dtype == torch.float32


class TestMeshRefinementGameMisc:
    """Tests for miscellaneous methods."""

    def test_action_to_string(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        s = game.action_to_string(actions[0])
        assert "refine_element" in s

    def test_action_to_string_invalid(self, game: MeshRefinementGame) -> None:
        s = game.action_to_string(99999)
        assert "invalid" in s

    def test_game_loop(self, game: MeshRefinementGame) -> None:
        """Run a short game loop to verify integration."""
        state = game.get_initial_state()
        error_history = [state.error_estimate]
        steps = 0

        while not game.is_terminal(state) and steps < 3:
            actions = game.get_valid_actions(state)
            if not actions:
                break
            prev = state
            state = game.apply_action(state, actions[0])
            reward = game.get_reward(state, prev)
            error_history.append(state.error_estimate)
            steps += 1
            assert isinstance(reward, float)

        result = game.get_result(state, error_history)
        assert result.n_steps == steps

    def test_mesh_quality_tracking(self, game: MeshRefinementGame) -> None:
        """Track mesh quality through refinement steps."""
        state = game.get_initial_state()
        initial_n_points = state.n_points
        initial_dof = state.dof

        actions = game.get_valid_actions(state)
        if actions:
            new_state = game.apply_action(state, actions[0])
            # After h-refinement: more points, more DOF
            assert new_state.n_points > initial_n_points
            assert new_state.dof > initial_dof
            assert new_state.mesh_levels is not None


class TestPRefinementGame:
    """Tests for p-refinement strategy."""

    def test_p_refinement_game(self, poisson_operator: PoissonOperator) -> None:
        mesh_config = MeshRefinementConfig(
            name="p_mesh",
            initial_resolution=2,
            max_refinement_level=3,
            refinement_strategy=RefinementStrategy.P_REFINEMENT,
            n_candidate_elements=16,
            max_polynomial_degree=5,
        )
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_config = PDEGameConfig(
            name="test",
            pde_config=pde_config,
            game_mode="mesh_refinement",
            mesh_config=mesh_config,
            max_steps=5,
            max_dof=200,
        )
        game = MeshRefinementGame(poisson_operator, game_config)
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        assert len(actions) > 0
        new_state = game.apply_action(state, actions[0])
        assert new_state.dof >= state.dof
