"""Planning frameworks for physics-informed machine learning.

This package provides eight MCTS-style planning frameworks:

- **PINN Planner**: Treats PINN training as a sequential decision
  problem, using look-ahead to optimise collocation point placement,
  loss weighting, and optimizer switching.

- **Multi-Fidelity Manager**: Plans optimal allocation of computational
  budget across high-, medium-, low-fidelity simulations and neural
  surrogates.

- **Symbolic Discovery**: Discovers symbolic equations from data by
  searching over expression trees with UCB-based exploration.

- **Quantum Chemistry**: Active space selection for CASSCF and DMRG
  orbital ordering via look-ahead search over orbital configurations
  and permutations.

- **Plasma Physics**: Stellarator coil design and plasma model
  selection via look-ahead planning over coil geometry and
  multi-physics model assignment.

- **Neural Architecture Search**: Searches the combinatorial space
  of neural operator architectures (FNO, DeepONet, Wavelet, Galerkin)
  via MCTS-style look-ahead to find optimal operator compositions.

- **Inverse Problems**: Solves non-convex, multimodal inverse problems
  via MCTS-based measurement planning for optimal sensor placement
  and experimental design.

- **Numerical Relativity**: MCTS-based adaptive mesh refinement for
  numerical relativity simulations (binary black hole mergers, etc.),
  including gauge condition selection and GW extraction placement.
"""

from __future__ import annotations

from src.alphagalerkin.planning.inverse_problems import (
    InverseAction,
    InverseActionType,
    InverseProblemSolver,
    InverseProblemState,
    SensorConfig,
)
from src.alphagalerkin.planning.multi_fidelity import (
    FidelityAction,
    FidelityActionType,
    FidelityLevel,
    MultiFidelityManager,
    MultiFidelityState,
    SimulationPoint,
)
from src.alphagalerkin.planning.neural_arch_search import (
    ArchitectureState,
    LayerSpec,
    NASAction,
    NASActionType,
    NeuralOperatorNAS,
    OperatorBlockType,
)
from src.alphagalerkin.planning.numerical_relativity import (
    GaugeCondition,
    NRAction,
    NRActionType,
    NRMeshManager,
    NRMeshState,
    RefinementLevel,
)
from src.alphagalerkin.planning.pinn_planner import (
    PINNAction,
    PINNActionType,
    PINNPlanner,
    PINNTrainingState,
)
from src.alphagalerkin.planning.plasma_physics import (
    CoilAction,
    CoilActionType,
    CoilGeometry,
    ModelSelectionAction,
    ModelSelectionActionType,
    PlasmaModelSelector,
    PlasmaModelState,
    PlasmaModelType,
    PlasmaRegion,
    StellaratorOptimizer,
    StellaratorState,
)
from src.alphagalerkin.planning.quantum_chemistry import (
    ActiveSpaceAction,
    ActiveSpaceActionType,
    ActiveSpaceSelector,
    ActiveSpaceState,
    DMRGOrderingOptimizer,
    DMRGOrderingState,
    OrbitalInfo,
    OrderingAction,
    OrderingActionType,
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
    # Plasma Physics
    "CoilAction",
    "CoilActionType",
    "CoilGeometry",
    "ModelSelectionAction",
    "ModelSelectionActionType",
    "PlasmaModelSelector",
    "PlasmaModelState",
    "PlasmaModelType",
    "PlasmaRegion",
    "StellaratorOptimizer",
    "StellaratorState",
    # Quantum Chemistry
    "ActiveSpaceAction",
    "ActiveSpaceActionType",
    "ActiveSpaceSelector",
    "ActiveSpaceState",
    "DMRGOrderingOptimizer",
    "DMRGOrderingState",
    "OrderingAction",
    "OrderingActionType",
    "OrbitalInfo",
    # Neural Architecture Search
    "ArchitectureState",
    "LayerSpec",
    "NASAction",
    "NASActionType",
    "NeuralOperatorNAS",
    "OperatorBlockType",
    # Inverse Problems
    "InverseAction",
    "InverseActionType",
    "InverseProblemSolver",
    "InverseProblemState",
    "SensorConfig",
    # Numerical Relativity
    "GaugeCondition",
    "NRAction",
    "NRActionType",
    "NRMeshManager",
    "NRMeshState",
    "RefinementLevel",
]
