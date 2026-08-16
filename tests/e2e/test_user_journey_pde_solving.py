import pytest

from src.alphagalerkin.solver import AlphaGalerkinConfig, AlphaGalerkinSolver
from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SolverResult


@pytest.mark.e2e
def test_user_journey_pde_solving() -> None:
    """End-to-end journey for PDE solving workflow."""
    # Step 1: Initialize PDE configuration
    pde_config = PDEConfig(name="test_pde", pde_type=PDEType.POISSON)
    operator = PoissonOperator(config=pde_config)

    # Step 2: Set up AlphaGalerkinConfig with light parameters
    ag_config = AlphaGalerkinConfig(game_mode="basis_selection", n_mcts_simulations=10, max_steps=2)

    # Step 3: Run AlphaGalerkinSolver.solve()
    solver = AlphaGalerkinSolver(config=ag_config)
    result: SolverResult = solver.solve(operator, n_dof=10)

    # Step 4: Verify SolverResult
    assert hasattr(result, "l2_error")
    assert hasattr(result, "n_dof")
    assert hasattr(result, "wall_time_seconds")
    assert hasattr(result, "metadata")

    assert result.n_dof > 0
    assert result.l2_error >= 0.0
    assert result.wall_time_seconds >= 0.0
    assert isinstance(result.metadata, dict)
