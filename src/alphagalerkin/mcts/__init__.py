"""MCTS engine for discretization games.

This package provides the Monte Carlo Tree Search implementation
for exploring discretization action spaces.  Key components:

- :class:`MCTSNode` -- tree node with PUCT scoring and backup.
- :class:`TreeManager` -- orchestrates the search loop
  (select -> expand -> evaluate -> backup).
- :class:`DirichletNoise` -- exploration noise for root node.
- :class:`TemperatureSchedule` -- temperature annealing for action
  selection.
- :class:`ActionMasker` -- legal action masking for the environment.
- Selection strategies: PUCT, UCB1, RAVE.
- Interoperability protocols: :class:`MCTSSearchable`,
  :class:`MCTSEvaluable`, :class:`GameInterface`.
"""
from __future__ import annotations

from src.alphagalerkin.mcts.action_masking import ActionMasker
from src.alphagalerkin.mcts.backpropagation import backup
from src.alphagalerkin.mcts.game_adapter import DiscretizationGame
from src.alphagalerkin.mcts.node import MCTSNode
from src.alphagalerkin.mcts.noise import DirichletNoise
from src.alphagalerkin.mcts.protocol import (
    GameInterface,
    MCTSEvaluable,
    MCTSSearchable,
)
from src.alphagalerkin.mcts.selection import (
    SelectionFn,
    get_selection_fn,
    puct_score,
    rave_score,
    ucb1_score,
)
from src.alphagalerkin.mcts.temperature import TemperatureSchedule
from src.alphagalerkin.mcts.tree import TreeManager

__all__ = [
    # Core tree
    "MCTSNode",
    "TreeManager",
    # Backup
    "backup",
    # Noise
    "DirichletNoise",
    # Temperature
    "TemperatureSchedule",
    # Selection
    "SelectionFn",
    "get_selection_fn",
    "puct_score",
    "ucb1_score",
    "rave_score",
    # Action masking
    "ActionMasker",
    # Game adapter
    "DiscretizationGame",
    # Protocols
    "MCTSSearchable",
    "MCTSEvaluable",
    "GameInterface",
]
