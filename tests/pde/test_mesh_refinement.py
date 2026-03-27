"""Coverage tests for mesh refinement game.

Tests cover:
- MeshElement: Properties and data
- Mesh: Initialization, refinement (h, p, hp), properties
- MeshRefinementGame: State creation, actions, terminal conditions, rewards
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from src.pde.config import (
    MeshRefinementConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
    RefinementStrategy,
)
from src.pde.game import GamePhase, PDEState
from src.pde.games.mesh_refinement import Mesh, MeshElement, MeshRefinementGame
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
        source = self.source_term(coords)
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)
        residual_values = -source
        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())
        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives={},
        )
INITIAL_RESOLUTION = 2  # Keep small for fast tests


@pytest.fixture
def pde_config() -> PDEConfig:
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
    strategy: RefinementStrategy = RefinementStrategy.H_REFINEMENT,
) -> PDEGameConfig:
    mesh_config = MeshRefinementConfig(
        name="test_mesh",
        initial_resolution=INITIAL_RESOLUTION,
        max_refinement_level=3,
        max_polynomial_degree=3,
        n_candidate_elements=16,
        refinement_strategy=strategy,
    )
    return PDEGameConfig(
        name="test_mesh_game",
        pde_config=pde_config,
        game_mode="mesh_refinement",
        mesh_config=mesh_config,
        max_steps=10,
        computational_budget=20.0,
        error_tolerance=1e-8,
        max_dof=500,
    )


class TestMeshElement:
    """Tests for MeshElement dataclass."""

    def test_creation(self) -> None:
        elem = MeshElement(
            index=0,
            vertices=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
            center=np.array([0.5, 0.5], dtype=np.float32),
            size=1.0,
            level=0,
            polynomial_degree=1,
        )
        assert elem.index == 0
        assert elem.is_leaf
        assert elem.level == 0
        assert elem.polynomial_degree == 1

    def test_is_leaf_with_children(self) -> None:
        elem = MeshElement(
            index=0,
            vertices=np.zeros((4, 2), dtype=np.float32),
            center=np.zeros(2, dtype=np.float32),
            size=1.0,
            children=[1, 2, 3, 4],
        )
        assert not elem.is_leaf


class TestMesh:
    """Tests for Mesh class."""

    def test_2d_initialization(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=INITIAL_RESOLUTION,
        )
        # 2x2 grid = 4 elements
        assert mesh.n_elements == INITIAL_RESOLUTION**2
        assert mesh.dim == 2
        assert len(mesh.leaf_elements) == INITIAL_RESOLUTION**2

    def test_1d_initialization(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0], dtype=np.float32),
            domain_max=np.array([1.0], dtype=np.float32),
            initial_resolution=4,
        )
        assert mesh.n_elements == 4
        assert mesh.dim == 1

    def test_3d_initialization(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        assert mesh.n_elements == 8  # 2^3
        assert mesh.dim == 3

    def test_invalid_dimension_too_high(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            Mesh(
                domain_min=np.zeros(5, dtype=np.float32),
                domain_max=np.ones(5, dtype=np.float32),
                initial_resolution=2,
            )

    def test_invalid_dimension_zero(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            Mesh(
                domain_min=np.array([], dtype=np.float32),
                domain_max=np.array([], dtype=np.float32),
                initial_resolution=2,
            )

    def test_n_dof(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=INITIAL_RESOLUTION,
        )
        # Each element has polynomial_degree=1, dim=2, DOF = (1+1)^2 = 4
        assert mesh.n_dof == INITIAL_RESOLUTION**2 * 4

    def test_h_refinement(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=INITIAL_RESOLUTION,
        )
        initial_elements = mesh.n_elements
        new_indices = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        # Should create 4 children (2^dim)
        assert len(new_indices) == 4
        assert mesh.n_elements == initial_elements + 4
        # Original element should no longer be a leaf
        assert not mesh.elements[0].is_leaf

    def test_p_refinement(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=INITIAL_RESOLUTION,
        )
        initial_degree = mesh.elements[0].polynomial_degree
        new_indices = mesh.refine_element(0, RefinementStrategy.P_REFINEMENT)
        assert new_indices == [0]
        assert mesh.elements[0].polynomial_degree == initial_degree + 1

    def test_hp_refinement_low_level(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=INITIAL_RESOLUTION,
        )
        # Level 0 element should be h-refined (level < 2)
        new_indices = mesh.refine_element(0, RefinementStrategy.HP_REFINEMENT)
        assert len(new_indices) == 4  # h-refinement

    def test_hp_refinement_high_level(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=INITIAL_RESOLUTION,
        )
        # Refine twice to get to level 2
        children1 = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        children2 = mesh.refine_element(children1[0], RefinementStrategy.H_REFINEMENT)
        # Now refine a level-2 element with HP -> should p-refine
        initial_degree = mesh.elements[children2[0]].polynomial_degree
        mesh.refine_element(children2[0], RefinementStrategy.HP_REFINEMENT)
        assert mesh.elements[children2[0]].polynomial_degree == initial_degree + 1

    def test_get_element_centers(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=INITIAL_RESOLUTION,
        )
        centers = mesh.get_element_centers()
        assert centers.shape == (INITIAL_RESOLUTION**2, 2)
        # All centers should be inside domain
        assert np.all(centers >= 0.0)
        assert np.all(centers <= 1.0)

    def test_get_element_sizes(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=INITIAL_RESOLUTION,
        )
        sizes = mesh.get_element_sizes()
        assert sizes.shape == (INITIAL_RESOLUTION**2,)
        assert np.all(sizes > 0)


class TestMeshRefinementGame:
    """Tests for MeshRefinementGame."""

    @pytest.fixture
    def game(self, pde_config: PDEConfig, poisson_operator: PoissonOperator) -> MeshRefinementGame:
        config = _make_game_config(pde_config)
        return MeshRefinementGame(poisson_operator, config)

    def test_initialization(self, game: MeshRefinementGame) -> None:
        assert game.name == "mesh_refinement"
        assert game.action_space_size > 0
        assert game.state_channels == 5

    def test_get_initial_state(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert isinstance(state, PDEState)
        assert state.step == 0
        assert state.error_estimate > 0
        assert state.phase == GamePhase.INITIAL
        assert state.mesh_levels is not None

    def test_get_valid_actions(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        assert len(valid) > 0

    def test_get_action_mask(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        mask = game.get_action_mask(state)
        assert mask.shape == (game.action_space_size,)
        assert mask.any()

    def test_apply_action(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        new_state = game.apply_action(state, valid[0])
        assert new_state.step == 1
        assert new_state.dof >= state.dof

    def test_apply_invalid_action_raises(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(state, 10000)

    def test_is_terminal_initial(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert not game.is_terminal(state)

    def test_get_reward(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        new_state = game.apply_action(state, valid[0])
        reward = game.get_reward(new_state, state)
        assert isinstance(reward, float)

    def test_compute_exact_error(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        errors = game.compute_exact_error(state)
        assert "l2" in errors
        assert "h1" in errors
        assert "linf" in errors
        assert "residual" in errors

    def test_get_result(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        error_history = [state.error_estimate]
        valid = game.get_valid_actions(state)
        new_state = game.apply_action(state, valid[0])
        error_history.append(new_state.error_estimate)

        result = game.get_result(new_state, error_history)
        assert result.final_error >= 0
        assert result.n_steps == 1
        assert result.termination_reason in (
            "converged", "max_dof", "budget_exhausted", "max_steps"
        )

    def test_action_to_string(self, game: MeshRefinementGame) -> None:
        s = game.action_to_string(0)
        assert "refine_element" in s

    def test_action_to_string_invalid(self, game: MeshRefinementGame) -> None:
        s = game.action_to_string(99999)
        assert "invalid" in s

    def test_to_tensor(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.ndim >= 2
        assert tensor.shape[0] == game.state_channels

    def test_multiple_refinements(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        for _ in range(3):
            if game.is_terminal(state):
                break
            valid = game.get_valid_actions(state)
            if not valid:
                break
            state = game.apply_action(state, valid[0])
        assert state.step >= 1


class TestMeshRefinementPRefinement:
    """Tests with p-refinement strategy."""

    def test_p_refinement_game(
        self, pde_config: PDEConfig, poisson_operator: PoissonOperator
    ) -> None:
        config = _make_game_config(pde_config, strategy=RefinementStrategy.P_REFINEMENT)
        game = MeshRefinementGame(poisson_operator, config)
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        assert len(valid) > 0
        new_state = game.apply_action(state, valid[0])
        assert new_state.step == 1
