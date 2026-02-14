"""Planning frameworks for physics-informed machine learning.

This package provides three MCTS-style planning frameworks:

- **PINN Planner**: Treats PINN training as a sequential decision
  problem, using look-ahead to optimise collocation point placement,
  loss weighting, and optimizer switching.

- **Multi-Fidelity Manager**: Plans optimal allocation of computational
  budget across high-, medium-, low-fidelity simulations and neural
  surrogates.

- **Symbolic Discovery**: Discovers symbolic equations from data by
  searching over expression trees with UCB-based exploration.
"""
from __future__ import annotations

from src.alphagalerkin.planning.multi_fidelity import (
    FidelityAction,
    FidelityActionType,
    FidelityLevel,
    MultiFidelityManager,
    MultiFidelityState,
    SimulationPoint,
)
from src.alphagalerkin.planning.pinn_planner import (
    PINNAction,
    PINNActionType,
    PINNPlanner,
    PINNTrainingState,
)
from src.alphagalerkin.planning.symbolic_discovery import (
    ExpressionNode,
    SymbolicAction,
    SymbolicActionType,
    SymbolicDiscovery,
    SymbolicState,
    SymbolType,
)

__all__ = [
    # PINN Planner
    "PINNActionType",
    "PINNAction",
    "PINNPlanner",
    "PINNTrainingState",
    # Multi-Fidelity
    "FidelityAction",
    "FidelityActionType",
    "FidelityLevel",
    "MultiFidelityManager",
    "MultiFidelityState",
    "SimulationPoint",
    # Symbolic Discovery
    "ExpressionNode",
    "SymbolicAction",
    "SymbolicActionType",
    "SymbolicDiscovery",
    "SymbolicState",
    "SymbolType",
]
