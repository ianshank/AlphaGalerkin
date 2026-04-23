"""AlphaGalerkin unified solver wrapper.

Provides the ``AlphaGalerkinSolver`` class that matches the
``src.research.baselines.BaseSolver`` protocol, enabling apples-to-apples
benchmarking of MCTS-guided Galerkin methods against classical baselines
(FDM, Dorfler AMR, PINN, scikit-fem hp-adaptive).

The solver drives a ``PDEGame`` (either ``BasisSelectionGame`` or
``MeshRefinementGame``) with an MCTS search loop and returns a
``SolverResult`` with the canonical ``(error, n_dof, wall_time)`` triple.

Example::

    from src.alphagalerkin import AlphaGalerkinSolver, AlphaGalerkinConfig
    from src.pde.operators import PoissonOperator
    from src.pde.config import PDEConfig, PDEType

    pde_cfg = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
    operator = PoissonOperator(pde_cfg)
    solver = AlphaGalerkinSolver(AlphaGalerkinConfig(game_mode="basis_selection"))
    result = solver.solve(operator, n_dof=64)
"""

from __future__ import annotations

from src.alphagalerkin.solver import AlphaGalerkinConfig, AlphaGalerkinSolver

__all__ = ["AlphaGalerkinConfig", "AlphaGalerkinSolver"]
