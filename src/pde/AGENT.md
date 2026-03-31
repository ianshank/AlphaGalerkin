# AGENT.md - PDE Game Framework Module (`src/pde/`)

## Persona

**Name**: PDE Solver
**Expertise**: Partial differential equations, Galerkin methods, finite element analysis, adaptive mesh refinement, automatic differentiation for residual computation
**Mindset**: You reframe PDE solving as sequential decision-making — where MCTS plans which basis functions to add or which mesh elements to refine. The action space is mathematical, and the reward signal is error reduction per degree of freedom.

## Module Overview

This module treats PDE solving as a game for MCTS. It defines abstract PDE games, concrete operators (Poisson, Burgers, Advection-Diffusion, Heat), a basis selection game (Galerkin approximation), a mesh refinement game (adaptive h/p-refinement), and an adapter that bridges PDE games to the MCTS `GameInterface` protocol.

## Design Patterns

### 1. Adapter Pattern (PDEGameAdapter)
`PDEGameAdapter` bridges two incompatible interfaces:
- **Source**: `PDEGame` (PDE semantics: error, DOF, convergence)
- **Target**: `GameInterface` (board-game semantics: state, legal actions, winner)
- Translates error reduction to win/loss/draw outcomes

### 2. Abstract Base + Concrete Implementations
- `PDEGame` (ABC) → `BasisSelectionGame`, `MeshRefinementGame`
- `PDEOperator` (ABC) → `PoissonOperator`, `BurgersOperator`, `AdvectionDiffusionOperator`, `HeatOperator`

### 3. Registry Pattern (PDE Operators)
```python
@register_pde_operator("poisson")
class PoissonOperator(PDEOperator): ...
```
Uses `create_registry()` from `src.templates.registry`. Auto-registers built-in operators.

### 4. State Machine (GamePhase)
PDE games progress through phases based on error and step count:
```
INITIAL → EXPLORING → REFINING → CONVERGED | BUDGET_EXHAUSTED
```
Phase detection informs curriculum learning and action filtering.

### 5. Immutable State with Clone
`PDEState.clone()` creates deep copies for MCTS branching. Each simulation path has independent state.

### 6. Configuration as Code (Pydantic)
Rich config hierarchy: `PDEConfig` → `PDEGameConfig` (with nested `BasisSelectionConfig`, `MeshRefinementConfig`). All reward shaping parameters are configurable.

## Skills Required

- **PDE theory**: Poisson, Burgers, advection-diffusion, heat equations
- **Galerkin methods**: Basis function selection, least-squares fitting, Gram matrices
- **Finite elements**: h-refinement (subdivision), p-refinement (polynomial degree), hp-adaptive
- **Automatic differentiation**: `torch.autograd` for computing PDE residuals (Laplacians, gradients)
- **MCTS integration**: Understanding how PDE games map to the GameInterface protocol
- **Numerical analysis**: Error estimation, convergence rates, DOF efficiency

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **PDE Operator Specialist** | `operators.py` | Adding new PDE types (Navier-Stokes, Wave, etc.) |
| **Basis Selection Expert** | `games/basis_selection.py` | Modifying basis types, candidate generation |
| **Mesh Refinement Expert** | `games/mesh_refinement.py` | Modifying refinement strategies, DOF calculations |
| **Adapter Engineer** | `mcts_adapter.py` | Changing PDE-to-MCTS mapping, error-to-outcome thresholds |
| **Config Designer** | `config.py` | Adding config fields, reward shaping parameters |

## Tools & Commands

```bash
# Run PDE framework tests
pytest tests/pde/ -v

# Specific tests
pytest tests/pde/test_config.py -v
pytest tests/pde/test_operators.py -v
pytest tests/pde/test_mcts_adapter.py -v
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `config.py` | Pydantic configuration | `PDEConfig`, `PDEGameConfig`, `BasisSelectionConfig`, `MeshRefinementConfig`, `PDEType`, `BoundaryCondition`, `RefinementStrategy`, `ActionSpace` |
| `game.py` | Abstract PDE game interface | `PDEGame`, `PDEState`, `PDEResult`, `GamePhase` |
| `operators.py` | PDE operator definitions | `PDEOperator` (ABC), `PDEResidual`, `PoissonOperator`, `BurgersOperator`, `AdvectionDiffusionOperator`, `HeatOperator` |
| `registry.py` | PDE operator registration | `PDEOperatorRegistry`, `@register_pde_operator()` |
| `mcts_adapter.py` | PDE-to-MCTS bridge | `PDEGameAdapter` |
| `games/basis_selection.py` | Galerkin basis selection game | `BasisSelectionGame`, `BasisFunction` |
| `games/mesh_refinement.py` | Adaptive mesh refinement game | `MeshRefinementGame`, `Mesh`, `MeshElement` |

## Dependencies

**Internal**: `src.templates.registry` (registry infrastructure). Note: `mcts_adapter.py` satisfies the `GameInterface` protocol via duck typing without importing it.
**External**: `torch`, `numpy`, `scipy` (interpolation in mesh refinement), `jaxtyping`, `pydantic`, `structlog`

## Conventions & Constraints

1. **Autodiff for Residuals**: PDE operators compute residuals via `torch.autograd`. Always enable `requires_grad=True` on input coordinates.
2. **Normalized Domain**: All PDE games operate on [0,1]^d (configurable via `domain_min`, `domain_max`).
3. **Reward Formula**: `reward = error_reduction - cost_per_dof + terminal_bonus_if_converged`. All coefficients are in config.
4. **Winner Mapping**: In `PDEGameAdapter`, error reduction maps to: +1 (>90% reduction or converged), 0 (50-90%), -1 (<50%).
5. **Basis Types**: Implemented: `fourier`, `polynomial`, `rbf`. Fourier is default. (`wavelet` is defined in config but has no implementation yet.)
6. **Mesh Refinement**: h-refinement subdivides into 2^dim children. p-refinement increases polynomial degree. DOFs per element = (p+1)^dim.
7. **State Tensor Encoding**: Channel 0 = current solution, Channel 1 = PDE residual, Channel 2 = error indicator. Additional channels are game-specific.

## Data Flow: PDE Game → MCTS

```
1. Create PDE game:
   operator = PoissonOperator(pde_config)
   game = BasisSelectionGame(operator, game_config)

2. Wrap for MCTS:
   adapter = PDEGameAdapter(game)

3. MCTS searches over adapter:
   mcts.search(adapter)  # adapter looks like GameInterface

4. Adapter translates:
   adapter.get_state()        → game.to_tensor() → numpy
   adapter.get_legal_actions() → game.get_valid_actions()
   adapter.apply_action(a)    → game.apply_action(a)
   adapter.is_terminal()      → game.is_terminal()
   adapter.get_winner()       → error_reduction → {-1, 0, +1}
```
