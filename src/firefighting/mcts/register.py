"""Game registration for firefighting MCTS integration.

Registers FireSpreadGame with the PDEGameAdapter for
MCTS-guided mesh adaptation during fire predictions.
"""

from __future__ import annotations

import structlog

from src.pde.config import PDEConfig, PDEGameConfig, PDEType
from src.pde.mcts_adapter import PDEGameAdapter

logger = structlog.get_logger(__name__)


def create_fire_spread_adapter(
    n_regions: int = 16,
    max_budget: int = 300,
    max_steps: int = 15,
    convergence_tolerance: float = 0.02,
) -> PDEGameAdapter:
    """Create a MCTS-compatible adapter for fire spread mesh refinement.

    Args:
        n_regions: Number of refinement regions.
        max_budget: Maximum DOF budget.
        max_steps: Maximum refinement steps.
        convergence_tolerance: Error threshold.

    Returns:
        PDEGameAdapter wrapping the fire spread game.

    """
    from src.firefighting.mcts.game import FireSpreadGame
    from src.pde.operators import PoissonOperator

    pde_config = PDEConfig(
        name="fire_spread",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )

    game_config = PDEGameConfig(
        name="fire_spread_game",
        pde_config=pde_config,
        game_mode="mesh_refinement",
        max_budget=max_budget,
        max_steps=max_steps,
        convergence_tolerance=convergence_tolerance,
    )

    operator = PoissonOperator(pde_config)
    game = FireSpreadGame(operator, game_config, n_regions=n_regions)

    return PDEGameAdapter(game)
