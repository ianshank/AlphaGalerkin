# AlphaGalerkin C4 Architecture

This document describes the architecture of AlphaGalerkin using the C4 model (Context, Containers, Components, Code).

## Level 1: System Context Diagram

The System Context diagram shows how AlphaGalerkin fits into the broader ecosystem.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM CONTEXT                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────────────────┐
    │              │         │                          │
    │  Go Research │ ───────▶│     AlphaGalerkin        │
    │  /Developer  │         │                          │
    │              │◀─────── │  Resolution-Independent  │
    └──────────────┘         │       Go AI Agent        │
         │                   │                          │
         │                   └────────────┬─────────────┘
         │                                │
         │ Configures                     │ Queries State
         │ Experiments                    │ Submits Moves
         │                                │
         ▼                                ▼
    ┌──────────────┐         ┌──────────────────────────┐
    │              │         │                          │
    │  Training    │         │    Go Rules Engine       │
    │  Pipeline    │         │    (Gym/PettingZoo)      │
    │              │         │                          │
    └──────────────┘         └──────────────────────────┘
                                          │
                                          │
                                          ▼
                             ┌──────────────────────────┐
                             │                          │
                             │    Go GUI / Analysis     │
                             │    (Sabaki, GoGui, etc.) │
                             │                          │
                             └──────────────────────────┘
```

### Actors and Systems

| Element | Type | Description |
|---------|------|-------------|
| Go Researcher/Developer | Person | Configures experiments, trains models, analyzes games |
| AlphaGalerkin | System | The main AI agent using continuous operator learning |
| Go Rules Engine | External System | Validates moves, manages game state (e.g., gym-go) |
| Training Pipeline | External System | Distributed training infrastructure |
| Go GUI | External System | Visual interface for playing/analyzing games |

---

## Level 2: Container Diagram

The Container diagram shows the high-level technical building blocks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ALPHAGALERKIN SYSTEM                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │                 │    │                 │    │                 │         │
│  │  Configuration  │───▶│  Neural Operator│───▶│  Search Engine  │         │
│  │    (Hydra/     │    │     Model       │    │     (MCTS)      │         │
│  │   Pydantic)    │    │   (PyTorch)     │    │    (Python)     │         │
│  │                 │    │                 │    │                 │         │
│  └─────────────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │                 │    │                 │    │                 │         │
│  │  Math Kernel    │    │  Training Loop  │    │  GTP Interface  │         │
│  │  (NumPy/Torch)  │    │   (PyTorch)     │    │    (Python)     │         │
│  │                 │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                      │
                    │                                      │
                    ▼                                      ▼
           ┌─────────────────┐                    ┌─────────────────┐
           │   Model Store   │                    │   Go Engine     │
           │  (Checkpoints)  │                    │  (External)     │
           └─────────────────┘                    └─────────────────┘
```

### Container Descriptions

| Container | Technology | Responsibility |
|-----------|------------|----------------|
| Configuration | Hydra + Pydantic | Manages hyperparameters, domain settings, model architecture |
| Neural Operator Model | PyTorch 2.0+ | Continuous operator learning for Go position evaluation |
| Search Engine (MCTS) | Python | Monte Carlo Tree Search with neural network guidance |
| Math Kernel | NumPy + PyTorch | Basis functions, integral approximations, spectral operations |
| Training Loop | PyTorch | Self-play generation, gradient optimization |
| GTP Interface | Python | Go Text Protocol for external engine communication |

---

## Level 3: Component Diagram - Neural Operator Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEURAL OPERATOR MODEL                                │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         Input Processing            │
                    │  ┌─────────────┐ ┌──────────────┐  │
  Board State ─────▶│  │ Coordinate  │ │   Fourier    │  │
  [B, C, H, W]      │  │   Mapper    │ │  Features    │  │
                    │  └──────┬──────┘ └──────┬───────┘  │
                    │         └───────┬───────┘          │
                    │                 ▼                  │
                    │  ┌─────────────────────────────┐   │
                    │  │   Continuous Embedding      │   │
                    │  │   [B, N, D_model]           │   │
                    │  └─────────────────────────────┘   │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STRATEGY BODY                                       │
│                    (Global Influence Modeling)                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Galerkin Transformer Stack                        │   │
│  │                                                                      │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│  │   │   Galerkin   │    │    FNet      │    │   Galerkin   │         │   │
│  │   │  Attention   │───▶│   Mixing     │───▶│   Norm       │  × N    │   │
│  │   │   O(N)       │    │  O(N log N)  │    │              │         │   │
│  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │ Stability Guard │ ◀── Monitors LBB condition (σ_min > β > 0)            │
│  └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TACTICAL HEAD                                       │
│                    (Local Life & Death Reading)                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Softmax Attention Stack                           │   │
│  │                                                                      │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│  │   │   Softmax    │    │     FFN      │    │  Layer       │         │   │
│  │   │  Attention   │───▶│   (GELU)     │───▶│   Norm       │  × M    │   │
│  │   │  (Injective) │    │              │    │              │         │   │
│  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                 ┌─────────────────┐     ┌─────────────────┐
                 │   Policy Head   │     │   Value Head    │
                 │                 │     │                 │
                 │  Per-position   │     │  Global Pool    │
                 │  + Pass move    │     │  → [-1, 1]      │
                 │  [B, N+1]       │     │  [B, 1]         │
                 └─────────────────┘     └─────────────────┘
```

### Component Descriptions

| Component | Responsibility | Complexity |
|-----------|----------------|------------|
| Coordinate Mapper | Maps discrete grid to continuous [0,1]² domain | O(N) |
| Fourier Features | Spectral positional encoding | O(N·F) |
| Continuous Embedding | Projects board features to model dimension | O(N·D) |
| Galerkin Attention | Global influence via Petrov-Galerkin projection | O(N) |
| FNet Mixing | FFT-based token mixing for speed | O(N log N) |
| Stability Guard | Monitors LBB condition for numerical stability | O(D²) |
| Softmax Attention | Standard attention for local precision | O(N²) |
| Policy Head | Move probability distribution | O(N·D) |
| Value Head | Position evaluation | O(D) |

---

## Level 3: Component Diagram - Search Engine (MCTS)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SEARCH ENGINE (MCTS)                               │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────┐
                         │    Search Root      │
      Game State ───────▶│   (MCTSNode)        │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌───────────┐   ┌───────────┐   ┌───────────┐
             │  Child 1  │   │  Child 2  │   │  Child N  │
             │  (Action) │   │  (Action) │   │  (Action) │
             └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                   │               │               │
                   └───────────────┼───────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MCTS LOOP                                           │
│                                                                             │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│    │              │    │              │    │              │                │
│    │   SELECT     │───▶│    EXPAND    │───▶│   EVALUATE   │                │
│    │   (PUCT)     │    │  (Add Node)  │    │  (Neural Net)│                │
│    │              │    │              │    │              │                │
│    └──────────────┘    └──────────────┘    └──────┬───────┘                │
│           ▲                                       │                         │
│           │                                       │                         │
│           │            ┌──────────────┐          │                         │
│           │            │              │          │                         │
│           └────────────│    BACKUP    │◀─────────┘                         │
│                        │  (Propagate) │                                    │
│                        │              │                                    │
│                        └──────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │         Evaluator            │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │    Standard Path       │  │
                    │  │  (Full Model Forward)  │  │
                    │  └────────────────────────┘  │
                    │              OR              │
                    │  ┌────────────────────────┐  │
                    │  │      Fast Path         │  │
                    │  │   (FNet-only Forward)  │  │
                    │  └────────────────────────┘  │
                    │                              │
                    └──────────────────────────────┘
```

### MCTS Components

| Component | Responsibility |
|-----------|----------------|
| MCTSNode | Tree node storing visit counts, Q-values, priors |
| PUCT Selection | UCB-based action selection with policy prior |
| Expansion | Creates child nodes for unexplored actions |
| Evaluator | Neural network policy/value inference |
| Backup | Propagates values up the tree |
| Batch Evaluator | Collects multiple leaves for batch GPU inference |

---

## Level 3: Component Diagram - Math Kernel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MATH KERNEL                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          BASIS FUNCTIONS                                     │
│                                                                             │
│   ┌─────────────────┐        ┌─────────────────┐        ┌───────────────┐  │
│   │                 │        │                 │        │               │  │
│   │ Fourier Basis   │        │ Chebyshev Basis │        │  Grid Coords  │  │
│   │                 │        │                 │        │               │  │
│   │ φₖ(x) = e^{ikx} │        │ Tₙ(x) = cos(nθ) │        │ [0,1]² mesh   │  │
│   │                 │        │                 │        │               │  │
│   └────────┬────────┘        └────────┬────────┘        └───────┬───────┘  │
│            │                          │                         │          │
│            └──────────────────────────┼─────────────────────────┘          │
│                                       ▼                                    │
│                        ┌─────────────────────────────┐                     │
│                        │    Positional Encoding      │                     │
│                        │    PE(x,y) ∈ ℝ^{2F}         │                     │
│                        └─────────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       INTEGRAL APPROXIMATION                                 │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Galerkin Projection                               │  │
│   │                                                                      │  │
│   │    Find u ∈ V such that: ⟨Lu, v⟩ = ⟨f, v⟩  ∀v ∈ V                   │  │
│   │                                                                      │  │
│   │    Discrete form:  K^T V / n  (Monte Carlo integral)                │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                   Petrov-Galerkin Projection                         │  │
│   │                                                                      │  │
│   │    Find u ∈ U such that: ⟨Lu, v⟩ = ⟨f, v⟩  ∀v ∈ V  (U ≠ V)         │  │
│   │                                                                      │  │
│   │    LBB Condition: inf_{u} sup_{v} ⟨Lu,v⟩/(‖u‖‖v‖) ≥ β > 0          │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPECTRAL OPERATIONS                                   │
│                                                                             │
│   ┌─────────────────┐        ┌─────────────────┐        ┌───────────────┐  │
│   │                 │        │                 │        │               │  │
│   │ Spectral Filter │        │   Resolution    │        │  Anti-Alias   │  │
│   │                 │        │    Adapter      │        │    Filter     │  │
│   │ H(f) low-pass   │        │                 │        │               │  │
│   │ Gaussian/Butter │        │ 9×9 → 19×19     │        │ Cutoff ratio  │  │
│   │                 │        │ Zero-shot       │        │               │  │
│   └─────────────────┘        └─────────────────┘        └───────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 4: Code Diagram - GalerkinAttention

```python
class GalerkinAttention(nn.Module):
    """
    Petrov-Galerkin projection approximating integral operators.

    Mathematical formulation:
        Context = K^T V / n    (Monte Carlo integral approximation)
        Output = Q @ Context   (Reconstruction in query basis)

    Complexity: O(N) vs O(N²) for standard attention
    """

    def __init__(self, d_model: int, n_heads: int, ...):
        # Key dimension ≥ Query dimension (LBB stability)
        self.d_key = d_model // n_heads
        self.d_value = d_model // n_heads

        # Projections
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape

        # Project to Q, K, V
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # Reshape for multi-head
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)

        # Galerkin projection (LINEAR complexity!)
        # Step 1: K^T @ V / n  (project values onto key basis)
        context = einsum('b h n k, b h n v -> b h k v', k, v) / N

        # Step 2: Q @ Context  (reconstruct in query basis)
        out = einsum('b h n q, b h q v -> b h n v', q, context)

        # Merge heads and project
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                          │
└─────────────────────────────────────────────────────────────────────────────┘

  TRAINING FLOW:
  ══════════════

  Self-Play Games    Model Inference     Loss Computation      Gradient Update
       │                   │                    │                     │
       ▼                   ▼                    ▼                     ▼
  ┌─────────┐        ┌─────────┐         ┌─────────┐           ┌─────────┐
  │  Game   │───────▶│  Board  │────────▶│ Policy  │──────────▶│  Adam   │
  │  State  │        │ Tensor  │         │  Loss   │           │Optimizer│
  └─────────┘        └─────────┘         └─────────┘           └─────────┘
       │                   │                    │                     │
       │                   │                    │                     │
       ▼                   ▼                    ▼                     ▼
  ┌─────────┐        ┌─────────┐         ┌─────────┐           ┌─────────┐
  │  MCTS   │───────▶│  Model  │────────▶│  Value  │──────────▶│ Updated │
  │ Policy  │        │ Forward │         │  Loss   │           │ Weights │
  └─────────┘        └─────────┘         └─────────┘           └─────────┘


  INFERENCE FLOW:
  ═══════════════

  ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
  │   GTP     │     │   Game    │     │   MCTS    │     │  Neural   │
  │  Command  │────▶│   State   │────▶│  Search   │────▶│  Network  │
  │           │     │           │     │           │     │           │
  └───────────┘     └───────────┘     └───────────┘     └───────────┘
                                            │                 │
                                            │                 │
                                            ▼                 ▼
                                      ┌───────────┐     ┌───────────┐
                                      │   Best    │◀────│  Policy   │
                                      │   Move    │     │  + Value  │
                                      │           │     │           │
                                      └───────────┘     └───────────┘


  RESOLUTION TRANSFER:
  ═════════════════════

  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
  │  9×9 Model  │        │  Spectral   │        │ 19×19 Model │
  │  (Trained)  │───────▶│  Adapter    │───────▶│ (Inference) │
  │             │        │             │        │             │
  └─────────────┘        └─────────────┘        └─────────────┘
        │                       │                      │
        │                       │                      │
        ▼                       ▼                      ▼
   81 positions          Anti-aliasing           361 positions
   d_model dims          Spectral filter         d_model dims
```

---

## Deployment Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT OPTIONS                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  OPTION 1: Local Development
  ════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        Developer Machine                                 │
  │                                                                         │
  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
  │   │   Python    │    │   PyTorch   │    │    CUDA     │                │
  │   │    3.10+    │    │    2.0+     │    │   (GPU)     │                │
  │   └─────────────┘    └─────────────┘    └─────────────┘                │
  │                                                                         │
  │   ┌─────────────────────────────────────────────────────────────────┐  │
  │   │                    AlphaGalerkin                                 │  │
  │   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │  │
  │   │   │  Model  │  │  MCTS   │  │   GTP   │  │  Tests  │           │  │
  │   │   └─────────┘  └─────────┘  └─────────┘  └─────────┘           │  │
  │   └─────────────────────────────────────────────────────────────────┘  │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘


  OPTION 2: Distributed Training
  ═══════════════════════════════

  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │   Self-Play     │     │   Self-Play     │     │   Self-Play     │
  │   Worker 1      │     │   Worker 2      │     │   Worker N      │
  │   (CPU/GPU)     │     │   (CPU/GPU)     │     │   (CPU/GPU)     │
  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Replay Buffer         │
                    │         (Redis/RAM)          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Training Server         │
                    │      (Multi-GPU Node)        │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Model Checkpoint        │
                    │      (S3/GCS/Local)          │
                    └──────────────────────────────┘


  OPTION 3: Production Inference
  ═══════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          Cloud Infrastructure                            │
  │                                                                         │
  │   ┌─────────────────┐          ┌─────────────────────────────────────┐ │
  │   │   Load Balancer │          │        Inference Cluster            │ │
  │   │                 │────────▶ │                                     │ │
  │   │   (GTP/HTTP)    │          │  ┌─────────┐  ┌─────────┐          │ │
  │   └─────────────────┘          │  │ Worker  │  │ Worker  │  ...     │ │
  │                                │  │  (GPU)  │  │  (GPU)  │          │ │
  │                                │  └─────────┘  └─────────┘          │ │
  │                                │                                     │ │
  │                                └─────────────────────────────────────┘ │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Galerkin Attention | O(N) complexity enables large boards | Less expressive than O(N²) softmax |
| Softmax Tactical Head | Preserves injectivity for precise reading | Higher compute for local regions |
| FNet Mixing | 5x+ faster MCTS rollouts | Slightly lower accuracy |
| Fourier Positional Encoding | Resolution-independent | Requires spectral filtering |
| Petrov-Galerkin Projection | Mathematical grounding (Green's function) | Must satisfy LBB condition |
| Monte Carlo Normalization (1/n) | Consistent with integral approximation | Different from standard attention |

---

## Level 3: Component Diagram - Core Protocols & Agent Framework (v0.4.0)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CORE PROTOCOLS & AGENT SYSTEM                         │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │                      Runtime Checkable Protocols                     │
    │                                                                     │
    │   ┌─────────────────────┐                 ┌─────────────────────┐   │
    │   │  EvaluatorProtocol  │                 │    GameProtocol     │   │
    │   │  evaluate(), batch  │                 │ get_state, clone()  │   │
    │   └──────────┬──────────┘                 └──────────┬──────────┘   │
    │              │                                       │              │
    │              ▼                                       ▼              │
    │   ┌─────────────────────┐                 ┌─────────────────────┐   │
    │   │   OperatorProtocol  │                 │    SolverProtocol   │   │
    │   │ forward, residual() │                 │ solve(), get_metrics│   │
    │   └─────────────────────┘                 └─────────────────────┘   │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Thread-Safe Generic Registry                     │
    │                                                                     │
    │   Registry[T]: thread locking, alias resolution, deprecation warnings │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      Agent Lifecycle & Skills                       │
    │                                                                     │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │ HookManager: on_init, pre_step, post_step, on_error, complete│   │
    │   │ Built-in: LoggingHook, MetricsCollectorHook, EarlyStoppingHook │
    │   └──────────────────────────────┬──────────────────────────────┘   │
    │                                  │                                  │
    │   ┌──────────────────────────────┴──────────────────────────────┐   │
    │   │ Declarative Skills: BenchmarkSkill, SelfPlaySkill           │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## Quality Attributes

| Attribute | Approach |
|-----------|----------|
| **Performance** | O(N) Galerkin attention, O(N log N) FNet mixing, batch MCTS |
| **Scalability** | Resolution-independent architecture, distributed training |
| **Maintainability** | Strict typing (jaxtyping), einops for dimension clarity, core protocols |
| **Testability** | 7-tier test pyramid: sanity, security, benchmarks, regression, unit, integration, e2e |
| **Reliability** | LBB stability monitoring, gradient clipping, spectral filtering, lifecycle hooks |
