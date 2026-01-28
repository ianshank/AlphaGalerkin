# C4 Architecture: PDE Game Framework

This document describes the architecture of the PDE Game Framework using Simon Brown's C4 model.
The framework enables AlphaZero-style MCTS to solve PDEs by treating equation solving as sequential
decision-making.

## Overview

The PDE Game Framework extends AlphaGalerkin's capabilities from board games (Go) to partial
differential equations. This novel approach frames PDE solving as a game where:

- **State**: Current approximation quality (basis set, mesh, solution)
- **Actions**: Strategic decisions (add basis, refine mesh, place collocation points)
- **Reward**: Error reduction per computational cost
- **Terminal**: Converged or budget exhausted

This enables MCTS to **look ahead** multiple refinement steps—something classical error indicators cannot do.

---

## Level 1: System Context

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PDE GAME FRAMEWORK CONTEXT                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────┐         ┌───────────────────────────────────────┐                │
│   │  Researcher │────────>│                                       │                │
│   │  / User     │         │       PDE GAME FRAMEWORK              │                │
│   └─────────────┘         │                                       │                │
│                           │  ┌───────────────────────────────┐    │                │
│   ┌─────────────┐         │  │ • PDE Operators               │    │                │
│   │ AlphaGalerkin│<──────>│  │ • Game Abstractions          │    │                │
│   │ Core        │         │  │ • Physics-Informed Loss      │    │                │
│   └─────────────┘         │  │ • Adaptive Loss Balancing    │    │                │
│                           │  │ • Multi-Scale Fourier        │    │                │
│   ┌─────────────┐         │  └───────────────────────────────┘    │                │
│   │   MCTS      │<──────> │                                       │                │
│   │  Engine     │         └───────────────────────────────────────┘                │
│   └─────────────┘                          │                                        │
│                                            │                                        │
│   ┌─────────────┐                          ▼                                        │
│   │  Training   │<─────────────────────────┘                                        │
│   │  Pipeline   │                                                                   │
│   └─────────────┘                                                                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

```mermaid
C4Context
    title PDE Game Framework - System Context

    Person(researcher, "Researcher", "Solves PDEs using learned policies")

    System(pde_framework, "PDE Game Framework", "Treats PDE solving as sequential decision-making for MCTS")

    System_Ext(alphagalerkin, "AlphaGalerkin Core", "Neural operator with Galerkin attention")
    System_Ext(mcts, "MCTS Engine", "Monte Carlo Tree Search with neural guidance")
    System_Ext(training, "Training Pipeline", "Self-play and optimization")

    Rel(researcher, pde_framework, "Configures PDE problems", "Pydantic configs")
    Rel(pde_framework, alphagalerkin, "Uses continuous embeddings", "Fourier features")
    Rel(pde_framework, mcts, "Provides game interface", "PDEGame API")
    Rel(pde_framework, training, "Provides physics loss", "Residual + BC + IC")
```

### External Entities

| Entity | Type | Description | Interaction |
|--------|------|-------------|-------------|
| Researcher | Human Actor | PDE scientist or ML researcher | Configures PDEs, analyzes results |
| AlphaGalerkin Core | System | Resolution-independent neural operator | Provides embeddings, attention |
| MCTS Engine | System | Tree search with learned policy/value | Calls PDEGame interface |
| Training Pipeline | System | Self-play, replay buffer, trainer | Uses physics-informed losses |

---

## Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PDE GAME FRAMEWORK CONTAINERS                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐       │
│  │    PDE OPERATORS    │   │     GAME ENGINE     │   │   PHYSICS LOSSES    │       │
│  │    (src/pde/)       │   │  (src/pde/games/)   │   │ (src/training/)     │       │
│  ├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤       │
│  │ • PoissonOperator   │   │ • BasisSelectionGame│   │ • ResidualLoss      │       │
│  │ • BurgersOperator   │──>│ • MeshRefinementGame│──>│ • BoundaryLoss      │       │
│  │ • AdvDiffOperator   │   │ • PDEState          │   │ • PhysicsInformedLoss│      │
│  │ • HeatOperator      │   │ • PDEResult         │   │ • CombinedLoss      │       │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────┘       │
│            │                         │                         │                    │
│            │                         │                         │                    │
│            ▼                         ▼                         ▼                    │
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐       │
│  │   CONFIGURATION     │   │    FOURIER FEATURES │   │   LOSS BALANCING    │       │
│  │   (src/pde/)        │   │   (src/modeling/)   │   │  (src/training/)    │       │
│  ├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤       │
│  │ • PDEConfig         │   │ • MultiScaleFourier │   │ • ReLoBRaLo         │       │
│  │ • PDEGameConfig     │   │ • AdaptiveFourier   │   │ • GradNorm          │       │
│  │ • BasisSelConfig    │   │ • ProgressiveFourier│   │ • Uncertainty       │       │
│  │ • MeshRefineConfig  │   │ • SpatialPosEnc     │   │ • SoftAdapt         │       │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────┘       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

```mermaid
C4Container
    title PDE Game Framework - Container Diagram

    Container(pde_ops, "PDE Operators", "Python/PyTorch", "Defines PDEs with autodiff residuals")
    Container(game_engine, "Game Engine", "Python", "Implements PDEGame interface for MCTS")
    Container(physics_loss, "Physics Losses", "Python/PyTorch", "Physics-informed loss components")
    Container(config, "Configuration", "Pydantic", "Type-safe validated configs")
    Container(fourier, "Fourier Features", "Python/PyTorch", "Multi-scale positional encoding")
    Container(balancing, "Loss Balancing", "Python/PyTorch", "Adaptive multi-loss weighting")

    Rel(pde_ops, game_engine, "Provides operators", "PDEOperator interface")
    Rel(game_engine, physics_loss, "Computes residuals", "Tensor operations")
    Rel(config, pde_ops, "Configures", "PDEConfig")
    Rel(config, game_engine, "Configures", "PDEGameConfig")
    Rel(fourier, pde_ops, "Encodes positions", "Coordinates → Features")
    Rel(balancing, physics_loss, "Weights losses", "Dynamic weights")
```

### Container Descriptions

| Container | Technology | Responsibility | Key Files |
|-----------|------------|----------------|-----------|
| PDE Operators | PyTorch + NumPy | Define PDE equations, compute residuals via autodiff | `src/pde/operators.py` |
| Game Engine | Python | Implement PDEGame interface for MCTS integration | `src/pde/games/*.py` |
| Physics Losses | PyTorch | Physics-informed loss terms (residual, BC, IC, conservation) | `src/training/physics_loss.py` |
| Configuration | Pydantic | Type-safe configs with validation | `src/pde/config.py` |
| Fourier Features | PyTorch | Multi-scale Fourier encoding for spectral bias mitigation | `src/modeling/multiscale_fourier.py` |
| Loss Balancing | PyTorch | Adaptive loss weighting (ReLoBRaLo, GradNorm, etc.) | `src/training/loss_balancing.py` |

---

## Level 3: Component Diagrams

### 3.1 PDE Operators Component

```mermaid
C4Component
    title PDE Operators - Component Diagram

    Component(base, "PDEOperator", "ABC", "Abstract base with autodiff derivatives")
    Component(poisson, "PoissonOperator", "Class", "∇²u = f (Laplacian)")
    Component(burgers, "BurgersOperator", "Class", "u_t + u·∇u = ν∇²u")
    Component(advdiff, "AdvectionDiffusionOperator", "Class", "u_t + a·∇u = ν∇²u")
    Component(heat, "HeatOperator", "Class", "u_t = κ∇²u")
    Component(registry, "PDEOperatorRegistry", "Singleton", "Thread-safe operator discovery")
    Component(residual, "PDEResidual", "Dataclass", "Residual values and norms")

    Rel(poisson, base, "extends")
    Rel(burgers, base, "extends")
    Rel(advdiff, base, "extends")
    Rel(heat, base, "extends")
    Rel(registry, base, "registers")
    Rel(base, residual, "returns")
```

**Component Details:**

| Component | Responsibility | Complexity |
|-----------|----------------|------------|
| `PDEOperator` | Abstract interface with `residual()`, `source_term()`, `boundary_value()`, autodiff `compute_derivatives()` | O(N) per evaluation |
| `PoissonOperator` | Steady-state Poisson: `-∇²u = f` with DST-based exact solver for ground truth | O(N log N) spectral |
| `BurgersOperator` | Nonlinear Burgers with shock formation, viscosity parameter | Nonlinear iteration |
| `AdvectionDiffusionOperator` | Linear advection-diffusion with velocity field | O(N) evaluation |
| `HeatOperator` | Heat equation with thermal diffusivity | O(N) evaluation |
| `PDEOperatorRegistry` | Decorator-based registration, thread-safe singleton | O(1) lookup |

**Key Methods (PDEOperator):**

```python
class PDEOperator(ABC):
    @abstractmethod
    def residual(self, u: Tensor, coords: Tensor) -> PDEResidual:
        """Compute R = L(u) - f via automatic differentiation."""

    @abstractmethod
    def source_term(self, coords: Tensor) -> Tensor:
        """Compute forcing function f(x)."""

    @abstractmethod
    def boundary_value(self, coords: Tensor) -> Tensor:
        """Compute boundary condition g(x)."""

    def compute_derivatives(self, u: Tensor, coords: Tensor) -> dict[str, Tensor]:
        """Compute ∂u/∂x, ∂²u/∂x², ∇²u via torch.autograd."""
```

---

### 3.2 Game Engine Component

```mermaid
C4Component
    title Game Engine - Component Diagram

    Component(game_base, "PDEGame", "ABC", "Abstract game interface for MCTS")
    Component(state, "PDEState", "Dataclass", "Solution, residuals, budget, history")
    Component(result, "PDEResult", "Dataclass", "Final metrics and trajectories")
    Component(basis_game, "BasisSelectionGame", "Class", "Galerkin basis selection")
    Component(mesh_game, "MeshRefinementGame", "Class", "Adaptive h/p refinement")
    Component(basis_func, "BasisFunction", "Dataclass", "Fourier/polynomial/RBF basis")
    Component(mesh, "Mesh", "Class", "2D quad mesh with refinement")

    Rel(basis_game, game_base, "implements")
    Rel(mesh_game, game_base, "implements")
    Rel(game_base, state, "uses")
    Rel(game_base, result, "returns")
    Rel(basis_game, basis_func, "manages")
    Rel(mesh_game, mesh, "manages")
```

**PDEGame Interface:**

```python
class PDEGame(ABC):
    @property
    @abstractmethod
    def action_space_size(self) -> int: ...

    @abstractmethod
    def get_initial_state(self) -> PDEState: ...

    @abstractmethod
    def get_valid_actions(self, state: PDEState) -> list[int]: ...

    @abstractmethod
    def apply_action(self, state: PDEState, action: int) -> PDEState: ...

    @abstractmethod
    def get_reward(self, state: PDEState, prev_state: PDEState) -> float: ...

    @abstractmethod
    def is_terminal(self, state: PDEState) -> bool: ...

    @abstractmethod
    def to_tensor(self, state: PDEState) -> Tensor: ...
```

**Game Implementations:**

| Game | State Representation | Actions | Reward Shaping |
|------|---------------------|---------|----------------|
| BasisSelectionGame | Basis coefficients, solution, residuals | Add Fourier/poly/RBF basis | Error reduction - DOF cost |
| MeshRefinementGame | Mesh elements, refinement levels, solution | Refine element (h or p) | Error reduction - DOF cost |

---

### 3.3 Physics-Informed Loss Component

```mermaid
C4Component
    title Physics-Informed Loss - Component Diagram

    Component(residual_loss, "ResidualLoss", "Module", "||L(u) - f||² minimization")
    Component(boundary_loss, "BoundaryLoss", "Module", "||u - g||² on ∂Ω")
    Component(initial_loss, "InitialConditionLoss", "Module", "||u(0) - u₀||²")
    Component(conserv_loss, "ConservationLoss", "Module", "||∫u dx - C||²")
    Component(physics_loss, "PhysicsInformedLoss", "Module", "Combined with balancing")
    Component(combined, "CombinedAlphaGalerkinPhysicsLoss", "Module", "Policy + Value + Physics")
    Component(config, "PhysicsLossConfig", "Pydantic", "Weights and sampling")

    Rel(physics_loss, residual_loss, "combines")
    Rel(physics_loss, boundary_loss, "combines")
    Rel(physics_loss, initial_loss, "combines")
    Rel(physics_loss, conserv_loss, "combines")
    Rel(combined, physics_loss, "includes")
    Rel(config, physics_loss, "configures")
```

**Loss Formulation:**

```
L_total = w_r · L_residual + w_b · L_boundary + w_ic · L_initial + w_c · L_conservation

Where:
  L_residual = 1/N Σ ||L(u)(xᵢ) - f(xᵢ)||²     (PDE residual)
  L_boundary = 1/N_b Σ ||u(x_b) - g(x_b)||²    (Boundary conditions)
  L_initial  = 1/N₀ Σ ||u(x, 0) - u₀(x)||²     (Initial condition)
  L_conservation = |∫_Ω u dx - C|²              (Conservation laws)
```

---

### 3.4 Loss Balancing Component

```mermaid
C4Component
    title Loss Balancing - Component Diagram

    Component(balancer_base, "LossBalancer", "ABC", "Base balancing interface")
    Component(relobralo, "ReLoBRaLo", "Class", "Random lookback balancing")
    Component(gradnorm, "GradNorm", "Class", "Gradient normalization")
    Component(uncertainty, "UncertaintyWeighting", "Class", "Learned log-variance")
    Component(softadapt, "SoftAdapt", "Class", "Rate-based adaptation")
    Component(static, "StaticWeighting", "Class", "Fixed weights baseline")
    Component(factory, "create_loss_balancer", "Function", "Factory with config")
    Component(config, "LossBalancingConfig", "Pydantic", "Strategy and hyperparams")

    Rel(relobralo, balancer_base, "extends")
    Rel(gradnorm, balancer_base, "extends")
    Rel(uncertainty, balancer_base, "extends")
    Rel(softadapt, balancer_base, "extends")
    Rel(static, balancer_base, "extends")
    Rel(factory, balancer_base, "creates")
    Rel(config, factory, "configures")
```

**Balancing Strategies:**

| Strategy | Formula | Use Case |
|----------|---------|----------|
| ReLoBRaLo | `wᵢ = softmax(Lᵢ(t) / Lᵢ(τ))` with random τ | Physics-informed neural networks |
| GradNorm | Normalize gradients to achieve equal training rates | Multi-task learning |
| Uncertainty | `wᵢ = 1/(2σᵢ²)` with learned σ | Heteroscedastic multi-objective |
| SoftAdapt | Weight by loss improvement rate | Faster convergence |

---

### 3.5 Multi-Scale Fourier Features Component

```mermaid
C4Component
    title Multi-Scale Fourier Features - Component Diagram

    Component(multiscale, "MultiScaleFourierFeatures", "Module", "Multiple frequency bands")
    Component(adaptive, "AdaptiveFourierFeatures", "Module", "Attention-weighted banks")
    Component(progressive, "ProgressiveFourierFeatures", "Module", "Curriculum frequency gating")
    Component(spatial, "SpatialPositionalEncoding", "Module", "2D grid encoding")
    Component(positional, "PositionalEncoding", "Module", "Transformer-style 1D")
    Component(config, "FourierFeaturesConfig", "Pydantic", "Scales, learnable, etc.")

    Rel(multiscale, config, "uses")
    Rel(adaptive, multiscale, "extends")
    Rel(progressive, multiscale, "extends")
```

**Spectral Bias Mitigation:**

```
γ(x) = [x, sin(2πB₁x), cos(2πB₁x), ..., sin(2πBₖx), cos(2πBₖx)]

Where Bᵢ ~ N(0, σᵢ²) with σᵢ spanning orders of magnitude:
  - Low σ (1.0): Large-scale structure
  - Medium σ (10.0): Intermediate features
  - High σ (100.0): Fine details, sharp gradients
```

---

## Level 4: Code and ADRs

### 4.1 Key Class Diagram

```mermaid
classDiagram
    class PDEOperator {
        <<abstract>>
        +config: PDEConfig
        +dim: int
        +residual(u, coords) PDEResidual
        +source_term(coords) Tensor
        +boundary_value(coords) Tensor
        +compute_derivatives(u, coords) dict
        +generate_collocation_points(n) ndarray
    }

    class PDEGame {
        <<abstract>>
        +pde_operator: PDEOperator
        +config: PDEGameConfig
        +action_space_size: int
        +get_initial_state() PDEState
        +apply_action(state, action) PDEState
        +get_reward(state, prev) float
        +is_terminal(state) bool
        +to_tensor(state) Tensor
    }

    class PDEState {
        +coords: ndarray
        +solution: ndarray
        +residuals: ndarray
        +error_estimate: float
        +dof: int
        +step: int
        +budget_remaining: float
        +clone() PDEState
    }

    class LossBalancer {
        <<abstract>>
        +config: LossBalancingConfig
        +weights: dict
        +update(losses) dict
        +compute_weighted_loss(losses) LossTerms
    }

    PDEOperator <|-- PoissonOperator
    PDEOperator <|-- BurgersOperator
    PDEGame <|-- BasisSelectionGame
    PDEGame <|-- MeshRefinementGame
    PDEGame --> PDEOperator : uses
    PDEGame --> PDEState : manages
    LossBalancer <|-- ReLoBRaLo
    LossBalancer <|-- GradNorm
```

### 4.2 Sequence Diagram: MCTS with PDE Game

```mermaid
sequenceDiagram
    participant MCTS
    participant PDEGame
    participant PDEOperator
    participant NeuralNet

    MCTS->>PDEGame: get_initial_state()
    PDEGame->>PDEOperator: generate_collocation_points()
    PDEGame-->>MCTS: PDEState (zero solution)

    loop MCTS Simulations
        MCTS->>PDEGame: get_valid_actions(state)
        PDEGame-->>MCTS: [action indices]

        MCTS->>NeuralNet: evaluate(to_tensor(state))
        NeuralNet-->>MCTS: (policy, value)

        MCTS->>MCTS: select_action(puct)

        MCTS->>PDEGame: apply_action(state, action)
        PDEGame->>PDEOperator: residual(solution, coords)
        PDEOperator-->>PDEGame: PDEResidual
        PDEGame-->>MCTS: new PDEState

        MCTS->>PDEGame: get_reward(new, old)
        PDEGame-->>MCTS: error_reduction - cost

        MCTS->>PDEGame: is_terminal(new)
        PDEGame-->>MCTS: bool
    end

    MCTS->>PDEGame: get_result(final_state, history)
    PDEGame-->>MCTS: PDEResult
```

### 4.3 Architecture Decision Records

#### ADR-001: PDE Solving as Game

**Status:** Accepted

**Context:**
Traditional PDE solvers use local error indicators (Zienkiewicz-Zhu, Kelly) for adaptive refinement.
These indicators are myopic—they cannot look ahead multiple refinement steps.

**Decision:**
Frame PDE solving as a game compatible with AlphaZero MCTS:
- State = approximation quality
- Actions = refinement decisions
- Reward = error reduction / cost
- Value = expected final error

**Consequences:**
- (+) MCTS can plan ahead multiple steps
- (+) Reuses existing AlphaGalerkin infrastructure
- (+) Learned policy generalizes across PDE instances
- (-) Requires defining discrete action space
- (-) Training requires self-play or supervised curriculum

---

#### ADR-002: ReLoBRaLo for Physics Loss Balancing

**Status:** Accepted

**Context:**
Physics-informed losses have vastly different scales (residual ~1, BC ~0.01) and convergence rates.
Fixed weights require extensive tuning per problem.

**Decision:**
Implement ReLoBRaLo (Bischof & Kraus, 2022) as default balancing strategy:
- Track running loss history
- Compute relative change via random lookback
- Softmax to get adaptive weights

**Consequences:**
- (+) Automatic balancing without per-problem tuning
- (+) Robust to different PDE types
- (-) Introduces hyperparameters (β, τ, lookback)
- (-) Adds computational overhead for history tracking

**Alternatives Considered:**
- GradNorm: Requires shared layer, more complex
- Uncertainty: Adds learnable parameters

---

#### ADR-003: Multi-Scale Fourier Features

**Status:** Accepted

**Context:**
Neural networks exhibit spectral bias—they learn low frequencies first.
High-frequency solutions (shocks, boundary layers) are underrepresented.

**Decision:**
Implement multi-scale Fourier features with log-spaced frequency bands:
- Low σ (1.0): Captures large-scale structure
- High σ (100.0): Captures fine details
- Progressive gating for curriculum learning

**Consequences:**
- (+) Mitigates spectral bias
- (+) Enables learning of high-frequency features
- (+) Progressive features support curriculum
- (-) Increases embedding dimension
- (-) May need scale tuning for extreme problems

---

## Data Flow Diagrams

### Training Data Flow

```mermaid
flowchart TD
    subgraph "Self-Play Generation"
        A[Sample PDE Instance] --> B[Initialize PDEGame]
        B --> C[MCTS Simulation]
        C --> D{Terminal?}
        D -->|No| C
        D -->|Yes| E[Store Trajectory]
    end

    subgraph "Training"
        E --> F[Replay Buffer]
        F --> G[Sample Batch]
        G --> H[Forward Pass]
        H --> I[Compute Losses]
        I --> J[Policy Loss]
        I --> K[Value Loss]
        I --> L[Physics Loss]
        J --> M[ReLoBRaLo Balance]
        K --> M
        L --> M
        M --> N[Backward + Optimize]
    end

    subgraph "Physics Loss Computation"
        L --> L1[Residual Loss]
        L --> L2[Boundary Loss]
        L --> L3[Conservation Loss]
    end
```

### Inference Data Flow

```mermaid
flowchart LR
    subgraph "Input"
        A[PDE Config] --> B[PDEOperator]
        C[Game Config] --> D[PDEGame]
        B --> D
    end

    subgraph "MCTS Search"
        D --> E[Initial State]
        E --> F[MCTS with Neural Guide]
        F --> G[Best Action Sequence]
    end

    subgraph "Output"
        G --> H[Final Solution]
        G --> I[Error Metrics]
        G --> J[Trajectory Analysis]
    end
```

---

## Deployment Considerations

### Local Development
```
┌────────────────────────────────────────┐
│           Developer Machine            │
├────────────────────────────────────────┤
│  Python 3.11+ │ PyTorch │ CUDA (opt)  │
│  ──────────────────────────────────────│
│  src/pde/     │ Tests   │ Notebooks   │
└────────────────────────────────────────┘
```

### Distributed Training
```
┌────────────────────────────────────────────────────────────┐
│                    Training Cluster                         │
├─────────────────────┬────────────────────┬─────────────────┤
│  Self-Play Workers  │  Parameter Server  │  Trainer Node   │
│  (Ray / DDP)        │  (Model Zoo)       │  (PyTorch DDP)  │
├─────────────────────┴────────────────────┴─────────────────┤
│  Shared Storage: Checkpoints, Replay Buffer, Configs       │
└────────────────────────────────────────────────────────────┘
```

---

## Cross-Cutting Concerns

### Configuration Management
- All configs via Pydantic with validation
- No hardcoded values
- Deterministic hashing for reproducibility
- YAML/JSON serialization support

### Logging Strategy
- Structured logging via structlog
- Component-scoped loggers
- Metric logging for training curves
- Timed operations for profiling

### Testing Strategy
- Unit tests for all operators and games
- Property-based tests for mathematical invariants
- Integration tests for full game loops
- Coverage target: >90%

### Observability
- Loss component tracking (policy, value, physics)
- Weight evolution monitoring
- Error trajectory logging
- Checkpoint versioning

---

## Integration Points

### With AlphaGalerkin Core
```python
from src.modeling.embeddings import ContinuousEmbedding
from src.modeling.multiscale_fourier import MultiScaleFourierFeatures
from src.pde.games import BasisSelectionGame

# Integrate Fourier features with neural operator
embedding = ContinuousEmbedding(
    input_channels=17,
    d_model=256,
    n_fourier_features=128,
)
```

### With MCTS
```python
from src.mcts.search import MCTS
from src.pde.games import MeshRefinementGame

game = MeshRefinementGame(operator, config)
mcts = MCTS(
    game=game,  # PDEGame implements GameInterface-like API
    evaluator=neural_evaluator,
)
```

### With Training Pipeline
```python
from src.training.loss import AlphaGalerkinLoss
from src.training.physics_loss import CombinedAlphaGalerkinPhysicsLoss

loss_fn = CombinedAlphaGalerkinPhysicsLoss(
    pde_operator=operator,
    physics_weight=0.1,
)
```

---

## Quality Attributes

| Attribute | Approach |
|-----------|----------|
| **Performance** | O(N) operators, batched evaluations, GPU acceleration |
| **Scalability** | Distributed self-play, DDP training, model zoo |
| **Maintainability** | Pydantic configs, registry pattern, structured logging |
| **Testability** | Abstract interfaces, dependency injection, property tests |
| **Extensibility** | Registry-based operators, game abstraction |
| **Reliability** | Validation at boundaries, checkpoint recovery |

---

## References

1. Brown, S. (2018). The C4 Model for Software Architecture.
2. Raissi, M., et al. (2019). Physics-Informed Neural Networks.
3. Bischof, R., & Kraus, M. (2022). Multi-Objective Loss Balancing.
4. Tancik, M., et al. (2020). Fourier Features Let Networks Learn High Frequency Functions.
5. Silver, D., et al. (2017). Mastering the Game of Go without Human Knowledge.
