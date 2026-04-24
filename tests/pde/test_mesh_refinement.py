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


class TestMeshRefinementGameInterpolation:
    """Tests for ``_interpolate_solution`` (linear projection between meshes)."""

    def test_smooth_function_interpolated_accurately(self, game: MeshRefinementGame) -> None:
        """Beat nearest-neighbor on a smooth known field after one refinement.

        Interpolating a smooth field onto a refined mesh should be close to
        sampling the field directly and strictly at least as accurate as a
        nearest-neighbor baseline.
        """
        initial = game.get_initial_state()

        # Populate the initial state's solution with a smooth known field.
        def u(points: np.ndarray) -> np.ndarray:
            return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])

        old_state = initial.clone()
        old_state.solution = u(old_state.coords).astype(np.float32)

        # Refine the first leaf to produce a denser mesh, then interpolate.
        actions = game.get_valid_actions(initial)
        new_state = game.apply_action(initial, actions[0])
        new_coords = new_state.coords
        interpolated = game._interpolate_solution(old_state, new_coords)
        truth = u(new_coords).astype(np.float32)

        # Nearest-neighbor baseline for comparison.
        from scipy.spatial import cKDTree

        tree = cKDTree(old_state.coords)
        _, idx = tree.query(new_coords, k=1)
        nearest_baseline = old_state.solution[idx].astype(np.float32)

        linear_err = float(np.sqrt(np.mean((interpolated - truth) ** 2)))
        nn_err = float(np.sqrt(np.mean((nearest_baseline - truth) ** 2)))
        assert linear_err <= nn_err + 1e-9
        assert np.all(np.isfinite(interpolated))

    def test_interpolation_handles_empty_old_state(self, game: MeshRefinementGame) -> None:
        """Empty old coords produces a zero-filled result of the right shape."""
        initial = game.get_initial_state()
        empty_state = PDEState(
            coords=np.zeros((0, game.mesh.dim), dtype=np.float32),
            solution=np.zeros(0, dtype=np.float32),
            residuals=np.zeros(0, dtype=np.float32),
            mesh_levels=None,
            error_estimate=0.0,
            dof=0,
            step=0,
            budget_remaining=0.0,
            phase=GamePhase.INITIAL,
            history=[],
        )
        values = game._interpolate_solution(empty_state, initial.coords)
        assert values.shape == (len(initial.coords),)
        assert np.all(values == 0.0)


class TestMeshRefinementGameLogReward:
    """Tests for the proposal-form log reward on the mesh game."""

    @pytest.fixture
    def log_reward_game(self, poisson_operator: PoissonOperator) -> MeshRefinementGame:
        mesh_cfg = MeshRefinementConfig(
            name="log_reward_mesh",
            initial_resolution=2,
            max_refinement_level=3,
            refinement_strategy=RefinementStrategy.H_REFINEMENT,
            n_candidate_elements=16,
        )
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_cfg = PDEGameConfig(
            name="log_reward_game",
            pde_config=pde_config,
            game_mode="mesh_refinement",
            mesh_config=mesh_cfg,
            max_steps=10,
            max_dof=500,
            error_tolerance=1e-4,
            reward_form="log",
        )
        return MeshRefinementGame(poisson_operator, game_cfg)

    def test_log_reward_finite_and_state_dependent(
        self, log_reward_game: MeshRefinementGame
    ) -> None:
        state = log_reward_game.get_initial_state()
        new_state = log_reward_game.apply_action(state, log_reward_game.get_valid_actions(state)[0])
        reward = log_reward_game.get_reward(new_state, state)
        assert np.isfinite(reward)
        # Log-form depends on new_state only; sanity-check monotonicity in error
        halved = new_state.clone()
        halved.error_estimate = new_state.error_estimate * 0.5
        halved_reward = log_reward_game.get_reward(halved, state)
        assert halved_reward >= reward - 1e-9


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


class TestMeshRefinementGameCloneSafety:
    """MCTS tree-search depends on ``clone()`` producing independent meshes."""

    def test_clone_deep_copies_mesh(self, game: MeshRefinementGame) -> None:
        """``clone()`` returns a new game whose mesh is a deep copy."""
        cloned = game.clone()
        assert cloned is not game
        assert cloned.mesh is not game.mesh
        assert cloned.mesh.n_elements == game.mesh.n_elements
        # Shared immutables remain shared (no wasted copy of heavy state).
        assert cloned.pde_operator is game.pde_operator
        assert cloned.config is game.config
        assert cloned.mesh_config is game.mesh_config

    def test_clone_isolates_refinement_mutation(self, game: MeshRefinementGame) -> None:
        """Refining on the clone must not change the original's mesh."""
        state = game.get_initial_state()
        original_n_elements = game.mesh.n_elements

        cloned = game.clone()
        cloned.apply_action(state, cloned.get_valid_actions(state)[0])

        assert game.mesh.n_elements == original_n_elements
        assert cloned.mesh.n_elements > original_n_elements

    def test_clone_isolates_coarsen_mutation(self, poisson_operator: PoissonOperator) -> None:
        """Coarsening on the clone must not collapse the original's tree."""
        mesh_cfg = MeshRefinementConfig(
            name="clone_coarsen_mesh",
            initial_resolution=2,
            max_refinement_level=3,
            refinement_strategy=RefinementStrategy.H_REFINEMENT,
            n_candidate_elements=32,
            allow_coarsening=True,
        )
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_cfg = PDEGameConfig(
            name="clone_coarsen_game",
            pde_config=pde_config,
            game_mode="mesh_refinement",
            mesh_config=mesh_cfg,
            max_steps=20,
            max_dof=2000,
        )
        game = MeshRefinementGame(poisson_operator, game_cfg)

        # Refine the original so a coarsen action is available.
        state = game.get_initial_state()
        state = game.apply_action(state, game.get_valid_actions(state)[0])
        original_leaf_count = len(game.mesh.leaf_elements)

        cloned = game.clone()
        slots = cloned._refine_slot_count
        coarsen_actions = [a for a in cloned.get_valid_actions(state) if a >= slots]
        assert coarsen_actions, "expected a coarsen action on the cloned game"
        cloned.apply_action(state, coarsen_actions[0])

        assert len(game.mesh.leaf_elements) == original_leaf_count
        assert len(cloned.mesh.leaf_elements) < original_leaf_count


class TestMeshCoarsen:
    """Unit tests for Mesh.coarsen_element (independent of the game)."""

    def _make_mesh(self) -> Mesh:
        return Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )

    def test_unrefined_leaf_is_not_coarsenable(self) -> None:
        """Top-level leaves have no parent and cannot be coarsened."""
        mesh = self._make_mesh()
        for element in mesh.elements:
            assert mesh.can_coarsen_element(element.index) is False

    def test_coarsen_undoes_h_refinement(self) -> None:
        """Refining then coarsening restores the original leaf count and DOF."""
        mesh = self._make_mesh()
        initial_leaves = len(mesh.leaf_elements)
        initial_dof = mesh.n_dof

        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        assert len(mesh.leaf_elements) == initial_leaves + len(children) - 1

        parent_idx = mesh.coarsen_element(children[0])
        assert parent_idx == 0
        assert len(mesh.leaf_elements) == initial_leaves
        assert mesh.n_dof == initial_dof
        assert all(not mesh.elements[c].active for c in children)
        assert mesh.elements[0].is_leaf is True

    def test_coarsen_all_siblings_equivalent(self) -> None:
        """Any sibling can drive the coarsen — same parent collapses either way."""
        mesh1 = self._make_mesh()
        mesh2 = self._make_mesh()
        ch1 = mesh1.refine_element(1, RefinementStrategy.H_REFINEMENT)
        ch2 = mesh2.refine_element(1, RefinementStrategy.H_REFINEMENT)

        assert mesh1.coarsen_element(ch1[0]) == 1
        assert mesh2.coarsen_element(ch2[-1]) == 1
        assert len(mesh1.leaf_elements) == len(mesh2.leaf_elements)

    def test_cannot_coarsen_when_sibling_refined(self) -> None:
        """Coarsening is rejected when any sibling has been refined further."""
        mesh = self._make_mesh()
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        mesh.refine_element(children[0], RefinementStrategy.H_REFINEMENT)
        # Remaining child siblings should report not coarsenable because
        # their sibling children[0] is no longer a leaf.
        for idx in children[1:]:
            assert mesh.can_coarsen_element(idx) is False
        with pytest.raises(ValueError):
            mesh.coarsen_element(children[1])

    def test_coarsen_then_refine_round_trip(self) -> None:
        """Re-refining a coarsened parent yields fresh leaf count."""
        mesh = self._make_mesh()
        children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        mesh.coarsen_element(children[0])
        leaves_after_coarsen = len(mesh.leaf_elements)
        new_children = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        # fresh_children are freshly allocated (different indices from the
        # inactive original children) because the mesh stores all elements
        # ever created.
        assert set(new_children).isdisjoint(set(children))
        assert len(mesh.leaf_elements) == leaves_after_coarsen + len(new_children) - 1


class TestMeshRefinementGameCoarsen:
    """Integration tests for the coarsen half of the action space."""

    @pytest.fixture
    def coarsen_game(self, poisson_operator: PoissonOperator) -> MeshRefinementGame:
        mesh_cfg = MeshRefinementConfig(
            name="coarsen_mesh",
            initial_resolution=2,
            max_refinement_level=3,
            refinement_strategy=RefinementStrategy.H_REFINEMENT,
            # Match the default `game` fixture's candidate count so
            # action_space_size comparisons are meaningful.
            n_candidate_elements=64,
            allow_coarsening=True,
        )
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_cfg = PDEGameConfig(
            name="coarsen_game",
            pde_config=pde_config,
            game_mode="mesh_refinement",
            mesh_config=mesh_cfg,
            max_steps=20,
            max_dof=2000,
            error_tolerance=1e-4,
        )
        return MeshRefinementGame(poisson_operator, game_cfg)

    def test_action_space_doubles_when_coarsen_enabled(
        self, game: MeshRefinementGame, coarsen_game: MeshRefinementGame
    ) -> None:
        """Enabling allow_coarsening doubles action_space_size."""
        assert coarsen_game.action_space_size == 2 * game.action_space_size

    def test_initial_valid_actions_are_all_refine(self, coarsen_game: MeshRefinementGame) -> None:
        """On a fresh mesh every leaf is refinable, none is coarsenable."""
        state = coarsen_game.get_initial_state()
        valid = coarsen_game.get_valid_actions(state)
        slots = coarsen_game.mesh_config.n_candidate_elements
        assert all(a < slots for a in valid)

    def test_coarsen_action_exposed_after_refinement(
        self, coarsen_game: MeshRefinementGame
    ) -> None:
        """After one h-refinement, coarsen slots appear in the valid list."""
        state = coarsen_game.get_initial_state()
        refine_actions = coarsen_game.get_valid_actions(state)
        state = coarsen_game.apply_action(state, refine_actions[0])

        valid = coarsen_game.get_valid_actions(state)
        slots = coarsen_game.mesh_config.n_candidate_elements
        assert any(a >= slots for a in valid), "coarsen slot should be valid"

    def test_coarsen_action_reduces_dof(self, coarsen_game: MeshRefinementGame) -> None:
        """Applying a coarsen action strictly reduces the DOF count."""
        state = coarsen_game.get_initial_state()
        state = coarsen_game.apply_action(state, coarsen_game.get_valid_actions(state)[0])
        refined_dof = state.dof

        slots = coarsen_game.mesh_config.n_candidate_elements
        coarsen_actions = [a for a in coarsen_game.get_valid_actions(state) if a >= slots]
        assert coarsen_actions, "expected at least one coarsen action after refinement"
        state = coarsen_game.apply_action(state, coarsen_actions[0])
        assert state.dof < refined_dof

    def test_action_mask_has_doubled_length(self, coarsen_game: MeshRefinementGame) -> None:
        """Mask length matches the doubled action_space_size."""
        state = coarsen_game.get_initial_state()
        mask = coarsen_game.get_action_mask(state)
        assert mask.shape[0] == coarsen_game.action_space_size

    def test_action_to_string_distinguishes_kinds(self, coarsen_game: MeshRefinementGame) -> None:
        """action_to_string labels refine vs coarsen correctly."""
        state = coarsen_game.get_initial_state()
        state = coarsen_game.apply_action(state, coarsen_game.get_valid_actions(state)[0])

        slots = coarsen_game.mesh_config.n_candidate_elements
        refine_actions = [a for a in coarsen_game.get_valid_actions(state) if a < slots]
        coarsen_actions = [a for a in coarsen_game.get_valid_actions(state) if a >= slots]
        assert refine_actions and coarsen_actions
        assert "refine_element" in coarsen_game.action_to_string(refine_actions[0])
        assert "coarsen_element" in coarsen_game.action_to_string(coarsen_actions[0])

    def test_invalid_coarsen_action_raises(self, coarsen_game: MeshRefinementGame) -> None:
        """Dispatching a coarsen action on an ineligible leaf raises ValueError."""
        state = coarsen_game.get_initial_state()
        slots = coarsen_game.mesh_config.n_candidate_elements
        # Index 0 is a fresh leaf with no parent — coarsening is impossible.
        with pytest.raises(ValueError):
            coarsen_game.apply_action(state, slots)

    def test_out_of_range_action_raises(self, coarsen_game: MeshRefinementGame) -> None:
        """Actions outside [0, action_space_size) are rejected."""
        state = coarsen_game.get_initial_state()
        with pytest.raises(ValueError):
            coarsen_game.apply_action(state, coarsen_game.action_space_size + 1)

    def test_backwards_compat_no_coarsen(self, game: MeshRefinementGame) -> None:
        """With allow_coarsening=False (default), action layout is unchanged."""
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        slots = game.mesh_config.n_candidate_elements
        assert game.action_space_size == slots
        assert all(a < slots for a in valid)
