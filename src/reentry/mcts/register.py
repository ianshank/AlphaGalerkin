"""Game registration for reentry compressible flow MCTS integration.

Registers CompressibleFlowGame with the GameRegistry and
PDEGameAdapter so it can be used by the MCTS search engine.
"""

from __future__ import annotations

import structlog

from src.pde.config import PDEConfig, PDEGameConfig, PDEType
from src.pde.mcts_adapter import PDEGameAdapter

logger = structlog.get_logger(__name__)


def create_compressible_flow_adapter(
    n_regions: int = 16,
    max_budget: int = 500,
    max_steps: int = 20,
    convergence_tolerance: float = 0.01,
) -> PDEGameAdapter:
    """Create a MCTS-compatible adapter for compressible flow refinement.

    This factory creates the full chain:
    PDEConfig → PDEOperator → CompressibleFlowGame → PDEGameAdapter

    The adapter satisfies the GameInterface protocol expected by MCTS.

    Args:
        n_regions: Number of refinement regions (action space size).
        max_budget: Maximum DOF budget.
        max_steps: Maximum refinement steps per episode.
        convergence_tolerance: Error threshold for convergence.

    Returns:
        PDEGameAdapter wrapping the compressible flow game.

    """
    from src.pde.operators import PoissonOperator
    from src.reentry.mcts.game import CompressibleFlowGame

    # Use a Poisson operator as placeholder for domain geometry
    # (the actual compressible solver runs outside MCTS; MCTS only
    # guides mesh refinement decisions)
    pde_config = PDEConfig(
        name="compressible_flow",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )

    game_config = PDEGameConfig(
        name="compressible_flow_game",
        pde_config=pde_config,
        game_mode="mesh_refinement",
        max_budget=max_budget,
        max_steps=max_steps,
        convergence_tolerance=convergence_tolerance,
    )

    operator = PoissonOperator(pde_config)
    game = CompressibleFlowGame(operator, game_config, n_refinement_regions=n_regions)

    return PDEGameAdapter(game)


def create_evaluator_for_game(
    action_space_size: int,
) -> object:
    """Create a random evaluator for MCTS (for testing/bootstrap).

    Returns an evaluator satisfying the MCTS Evaluator protocol
    that returns random policy and neutral value.

    Args:
        action_space_size: Size of the action space.

    Returns:
        Evaluator instance for MCTS.

    """
    from src.mcts.evaluator import RandomEvaluator

    return RandomEvaluator(action_size=action_space_size)
