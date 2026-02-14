"""Shared test fixtures for AlphaGalerkin tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.alphagalerkin.core.config import (
    AlphaGalerkinConfig,
    EnvironmentConfig,
    MCTSConfig,
    ReplayConfig,
    TrainingConfig,
)
from src.alphagalerkin.core.types import ActionType
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.mesh_graph import MeshGraph
from src.alphagalerkin.env.state import DiscretizationState


@pytest.fixture
def default_config() -> AlphaGalerkinConfig:
    """Minimal valid configuration for testing."""
    return AlphaGalerkinConfig(
        mcts=MCTSConfig(
            num_simulations=5,
            max_tree_depth=3,
            action_topk=3,
        ),
        training=TrainingConfig(
            batch_size=4,
            total_steps=2,
            self_play_games_per_step=2,
            replay=ReplayConfig(
                capacity=1000,
                min_size_to_train=10,
            ),
        ),
        environment=EnvironmentConfig(
            max_steps=5,
            max_dof=500,
        ),
        device="cpu",
    )


@pytest.fixture
def quad_mesh_2x2() -> MeshGraph:
    """A 2x2 uniform quadrilateral mesh on the unit square."""
    return MeshGraph.create_uniform_quad(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        num_elements=(2, 2),
    )


@pytest.fixture
def tri_mesh_small() -> MeshGraph:
    """A 2x2 uniform triangular mesh (8 elements) on unit square."""
    return MeshGraph.create_uniform_tri(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        num_elements=(2, 2),
    )


@pytest.fixture
def initial_state(quad_mesh_2x2: MeshGraph) -> DiscretizationState:
    """Initial discretization state on a 2x2 quad mesh, p=1."""
    return DiscretizationState.from_mesh(
        mesh=quad_mesh_2x2,
        initial_polynomial_order=1,
    )


@pytest.fixture
def valid_actions(
    initial_state: DiscretizationState,
) -> list[Action]:
    """A small set of valid actions for the initial state."""
    actions: list[Action] = []
    for eid in initial_state.mesh.element_ids[:2]:
        for atype in [ActionType.H_REFINE, ActionType.P_REFINE]:
            actions.append(
                Action(
                    element_id=eid,
                    action_type=atype,
                    params={},
                )
            )
    return actions


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    """Seed all random number generators for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
