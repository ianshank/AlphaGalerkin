"""Clone-isolation tests across every concrete PDEGame subclass (F3).

``PDEGame.clone()`` returns ``self`` by default; stateful games (which mutate
instance attributes in ``apply_action``) must override it or a cloned MCTS
branch will silently corrupt its sibling. These tests assert, for each
concrete game, that applying actions on a clone leaves the original adapter's
observable state untouched.

Note: ``SwarmPlanningGame`` is intentionally excluded — it does not inherit
``PDEGame`` and uses its own ``SwarmState``. The reflection-over-subclasses
form of this test lands with ``RefinementGame`` in the WS1 extraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import (
    BasisSelectionConfig,
    MeshRefinementConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
    RefinementStrategy,
)
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.games.lshape_amr import GridSolveResult, LShapeAMRGame
from src.pde.games.mesh_refinement import MeshRefinementGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.operators import PoissonOperator


def _pde_config(name: str) -> PDEConfig:
    return PDEConfig(
        name=name,
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )


def _basis_adapter() -> PDEGameAdapter:
    pde_cfg = _pde_config("clone_basis_poisson")
    game_cfg = PDEGameConfig(
        name="clone_basis_game",
        pde_config=pde_cfg,
        game_mode="basis_selection",
        max_steps=10,
        error_tolerance=1e-6,
        computational_budget=1e4,
        basis_config=BasisSelectionConfig(
            name="clone_basis_selection",
            max_basis_functions=8,
            basis_type="fourier",
            max_frequency=3,
        ),
    )
    return PDEGameAdapter(BasisSelectionGame(PoissonOperator(pde_cfg), game_cfg))


def _mesh_adapter() -> PDEGameAdapter:
    pde_cfg = _pde_config("clone_mesh_poisson")
    game_cfg = PDEGameConfig(
        name="clone_mesh_game",
        pde_config=pde_cfg,
        game_mode="mesh_refinement",
        mesh_config=MeshRefinementConfig(
            name="clone_mesh_cfg",
            initial_resolution=2,
            max_refinement_level=3,
            refinement_strategy=RefinementStrategy.H_REFINEMENT,
            n_candidate_elements=32,
        ),
        max_steps=10,
        max_dof=2000,
    )
    return PDEGameAdapter(MeshRefinementGame(PoissonOperator(pde_cfg), game_cfg))


def _lshape_adapter() -> PDEGameAdapter:
    def solve(xs: np.ndarray, ys: np.ndarray) -> GridSolveResult:
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)
        n_nodes = grid.shape[0]
        nx, ny = len(xs) - 1, len(ys) - 1
        return GridSolveResult(
            solution=np.zeros(n_nodes, dtype=np.float64),
            grid=grid,
            l2_error=1.0 / n_nodes,
            n_dof=n_nodes,
            indicators=np.ones((nx, ny), dtype=np.float64),
        )

    pde_cfg = _pde_config("clone_lshape_poisson")
    game_cfg = PDEGameConfig(
        name="clone_lshape_game",
        pde_config=pde_cfg,
        game_mode="mesh_refinement",
        max_steps=10,
        max_dof=5000,
    )
    game = LShapeAMRGame(
        PoissonOperator(pde_cfg),
        game_cfg,
        solve_fn=solve,
        initial_side=4,
        n_candidate_elements=6,
    )
    return PDEGameAdapter(game)


_FACTORIES = {
    "basis_selection": _basis_adapter,
    "mesh_refinement": _mesh_adapter,
    "lshape_amr": _lshape_adapter,
}


@pytest.mark.parametrize("name", sorted(_FACTORIES))
def test_clone_isolates_state(name: str) -> None:
    """Applying actions on a clone must not change the original adapter."""
    adapter = _FACTORIES[name]()

    original_error = adapter.state.error_estimate
    original_step = adapter.state.step
    original_history_len = len(adapter.error_history)

    cloned = adapter.clone()
    # The cloned game instance must be independent for stateful games.
    for _ in range(3):
        actions = cloned.get_legal_actions()
        if not actions:
            break
        cloned.apply_action(actions[0])

    # The original is unchanged.
    assert adapter.state.error_estimate == pytest.approx(original_error)
    assert adapter.state.step == original_step
    assert len(adapter.error_history) == original_history_len


@pytest.mark.parametrize("name", sorted(_FACTORIES))
def test_clone_underlying_game_independent_for_stateful(name: str) -> None:
    """Stateful games must not share the mutable instance with their clone."""
    adapter = _FACTORIES[name]()
    cloned = adapter.clone()

    if name == "mesh_refinement":
        assert cloned.pde_game is not adapter.pde_game
        assert cloned.pde_game.mesh is not adapter.pde_game.mesh
    elif name == "lshape_amr":
        assert cloned.pde_game is not adapter.pde_game
        # Node coordinate arrays are per-episode mutable state.
        assert cloned.pde_game._xs is not adapter.pde_game._xs
