"""Tests for MeshRefinementGame and Mesh."""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import (
    MeshRefinementConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
    RefinementStrategy,
)
from src.pde.games.mesh_refinement import Mesh, MeshElement, MeshRefinementGame
from src.pde.operators import PoissonOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    )


@pytest.fixture
def game_config(mesh_config: MeshRefinementConfig) -> PDEGameConfig:
    return PDEGameConfig(
        name="test_game",
        pde_config=PDEConfig(name="test_poisson", pde_type=PDEType.POISSON),
        game_mode="mesh_refinement",
        mesh_config=mesh_config,
        max_steps=10,
    )


@pytest.fixture
def game(
    poisson_operator: PoissonOperator, game_config: PDEGameConfig
) -> MeshRefinementGame:
    return MeshRefinementGame(poisson_operator, game_config)


# ---------------------------------------------------------------------------
# MeshElement tests
# ---------------------------------------------------------------------------


class TestMeshElement:
    def test_is_leaf_default(self) -> None:
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

    def test_default_polynomial_degree(self) -> None:
        elem = MeshElement(
            index=0,
            vertices=np.zeros((4, 2), dtype=np.float32),
            center=np.zeros(2, dtype=np.float32),
            size=1.0,
        )
        assert elem.polynomial_degree == 1

    def test_default_level(self) -> None:
        elem = MeshElement(
            index=0,
            vertices=np.zeros((4, 2), dtype=np.float32),
            center=np.zeros(2, dtype=np.float32),
            size=1.0,
        )
        assert elem.level == 0


# ---------------------------------------------------------------------------
# Mesh tests
# ---------------------------------------------------------------------------


class TestMesh:
    def test_2d_init(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        assert mesh.n_elements == 4  # 2x2
        assert mesh.dim == 2

    def test_1d_init(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0], dtype=np.float32),
            domain_max=np.array([1.0], dtype=np.float32),
            initial_resolution=4,
        )
        assert mesh.n_elements == 4
        assert mesh.dim == 1

    def test_all_elements_are_leaves_initially(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        assert len(mesh.leaf_elements) == mesh.n_elements

    def test_n_dof_initial(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        # 4 elements, each with p=1, dim=2: DOF per elem = (1+1)^2 = 4
        assert mesh.n_dof == 4 * 4

    def test_h_refinement(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        initial_leaves = len(mesh.leaf_elements)
        mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        # One element replaced by 4 children in 2D
        assert len(mesh.leaf_elements) == initial_leaves - 1 + 4

    def test_p_refinement(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        mesh.refine_element(0, RefinementStrategy.P_REFINEMENT)
        assert mesh.elements[0].polynomial_degree == 2
        # Element count unchanged
        assert len(mesh.leaf_elements) == 4

    def test_invalid_high_dimension(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            Mesh(
                domain_min=np.zeros(5, dtype=np.float32),
                domain_max=np.ones(5, dtype=np.float32),
                initial_resolution=2,
            )

    def test_element_vertices_in_domain(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=3,
        )
        for elem in mesh.elements:
            assert np.all(elem.vertices >= -1e-6)
            assert np.all(elem.vertices <= 1.0 + 1e-6)

    def test_element_centers_in_domain(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=3,
        )
        for elem in mesh.elements:
            assert np.all(elem.center >= 0.0)
            assert np.all(elem.center <= 1.0)

    def test_children_have_higher_level(self) -> None:
        mesh = Mesh(
            domain_min=np.array([0.0, 0.0], dtype=np.float32),
            domain_max=np.array([1.0, 1.0], dtype=np.float32),
            initial_resolution=2,
        )
        new_indices = mesh.refine_element(0, RefinementStrategy.H_REFINEMENT)
        for idx in new_indices:
            assert mesh.elements[idx].level == 1


# ---------------------------------------------------------------------------
# MeshRefinementGame tests
# ---------------------------------------------------------------------------


class TestMeshRefinementGame:
    def test_initialization(self, game: MeshRefinementGame) -> None:
        assert game.name == "mesh_refinement"
        assert game.mesh is not None

    def test_action_space_positive(self, game: MeshRefinementGame) -> None:
        assert game.action_space_size > 0

    def test_state_channels(self, game: MeshRefinementGame) -> None:
        assert game.state_channels == 5

    def test_get_initial_state(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        assert state is not None
        assert state.step == 0
        assert state.error_estimate >= 0

    def test_valid_actions(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        assert isinstance(valid, list)
        assert len(valid) > 0

    def test_apply_action(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        valid = game.get_valid_actions(state)
        assert len(valid) > 0
        action = valid[0]
        new_state = game.apply_action(state, action)
        assert new_state.step == state.step + 1

    def test_mesh_grows_after_action(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        initial_n_dof = game.mesh.n_dof
        valid = game.get_valid_actions(state)
        action = valid[0]
        game.apply_action(state, action)
        assert game.mesh.n_dof >= initial_n_dof

    def test_to_tensor(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor is not None
        assert tensor.ndim >= 2

    def test_get_result(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        result = game.get_result(state, error_history=[1.0, 0.5, 0.2])
        assert result is not None
        assert hasattr(result, "final_error")
        assert hasattr(result, "final_dof")

    def test_multiple_refinements(self, game: MeshRefinementGame) -> None:
        state = game.get_initial_state()
        for _ in range(3):
            valid = game.get_valid_actions(state)
            if len(valid) == 0:
                break
            action = valid[0]
            state = game.apply_action(state, action)
        assert state.step >= 1


class TestMeshRefinementConfig:
    def test_default_config(self) -> None:
        config = MeshRefinementConfig(name="test")
        assert config.initial_resolution >= 1
        assert config.max_refinement_level >= 1

    def test_h_refinement_strategy(self) -> None:
        config = MeshRefinementConfig(
            name="test", refinement_strategy=RefinementStrategy.H_REFINEMENT
        )
        assert config.refinement_strategy == RefinementStrategy.H_REFINEMENT

    def test_p_refinement_strategy(self) -> None:
        config = MeshRefinementConfig(
            name="test", refinement_strategy=RefinementStrategy.P_REFINEMENT
        )
        assert config.refinement_strategy == RefinementStrategy.P_REFINEMENT
