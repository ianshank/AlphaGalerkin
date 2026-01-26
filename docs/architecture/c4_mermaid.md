# AlphaGalerkin C4 Architecture (Mermaid Format)

This document provides a comprehensive C4 architecture model for the AlphaGalerkin system using Mermaid diagrams.
The C4 model consists of four levels: System Context, Containers, Components, and Code.

---

## Level 1: System Context Diagram

The System Context diagram shows how AlphaGalerkin fits into the broader ecosystem, highlighting the key users and external systems.

```mermaid
C4Context
    title System Context - AlphaGalerkin

    Person(researcher, "Go Researcher", "ML researcher studying resolution-independent learning and continuous operators")
    Person(developer, "Developer", "Implements and experiments with Go AI algorithms")
    Person(player, "Go Player", "Uses the system to play games and analyze positions")

    System(alphagalerkin, "AlphaGalerkin", "Resolution-independent Go AI using Continuous Operator Learning (Galerkin Transformers & FNet). Enables zero-shot transfer between board sizes.")

    System_Ext(go_gui, "Go GUI", "Visual interface for playing and analyzing games (Sabaki, GoGui, Lizzie, KaTrain)")
    System_Ext(go_engine, "Go Rules Engine", "Validates moves and manages game state (gym-go, PettingZoo)")
    System_Ext(compute, "Compute Infrastructure", "GPU clusters for training (CUDA, distributed training)")

    Rel(researcher, alphagalerkin, "Runs PoC experiments, validates mathematical claims", "Python CLI")
    Rel(developer, alphagalerkin, "Trains models, implements features", "Python API")
    Rel(player, go_gui, "Plays games, analyzes positions", "GUI")
    Rel(go_gui, alphagalerkin, "Sends moves, receives evaluations", "GTP Protocol")
    Rel(alphagalerkin, go_engine, "Validates moves, queries legal actions", "Python API")
    Rel(alphagalerkin, compute, "Executes training, performs inference", "PyTorch/CUDA")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Key Interactions

- **Researchers** validate core mathematical claims (zero-shot transfer, O(N) complexity, LBB stability)
- **Developers** train models and implement new features using the Python API
- **Go Players** interact via GTP-compatible GUIs
- **External Systems** provide game rules validation and compute resources

---

## Level 2: Container Diagram

The Container diagram shows the high-level technical building blocks of AlphaGalerkin.

```mermaid
C4Container
    title Container Diagram - AlphaGalerkin System

    Person(user, "User", "Researcher, Developer, or Go Player")

    Container_Boundary(alphagalerkin, "AlphaGalerkin System") {
        Container(cli, "CLI Entrypoints", "Python Scripts", "Command-line interface for training, benchmarking, and experiments")
        Container(gtp_server, "GTP Server", "Python", "Go Text Protocol server for game playing and analysis")
        
        Container(neural_operator, "Neural Operator Model", "PyTorch", "Core continuous operator learning model with Galerkin attention and FNet mixing")
        
        Container(mcts_engine, "MCTS Search Engine", "Python", "Monte Carlo Tree Search with neural network guidance for move selection")
        
        Container(training_pipeline, "Training Pipeline", "PyTorch", "Self-play, replay buffer, loss computation, and checkpoint management")
        
        Container(math_kernel, "Math Kernel", "NumPy/PyTorch", "Mathematical primitives: Fourier basis, Galerkin projection, spectral filtering")
        
        Container(poc_framework, "PoC Framework", "Python/Pydantic", "Scenario-based validation of mathematical claims with reproducible experiments")
        
        Container(data_layer, "Data Layer", "PyTorch Dataset", "Board state preprocessing, variable-size batching, physics data generation")
        
        ContainerDb(checkpoint_store, "Model Checkpoints", "File System", "Stores trained model weights and training state")
        ContainerDb(results_store, "Experiment Results", "JSON/YAML", "Stores PoC scenario results and metrics")
    }

    System_Ext(go_gui, "Go GUI", "GTP Client")
    System_Ext(compute, "GPU Cluster", "CUDA Infrastructure")

    Rel(user, cli, "Runs commands", "CLI")
    Rel(go_gui, gtp_server, "Sends GTP commands", "GTP/TCP")
    
    Rel(cli, training_pipeline, "Initiates training")
    Rel(cli, poc_framework, "Runs experiments")
    Rel(gtp_server, neural_operator, "Gets policy/value")
    Rel(gtp_server, mcts_engine, "Performs search")
    
    Rel(mcts_engine, neural_operator, "Evaluates positions")
    Rel(training_pipeline, neural_operator, "Trains weights")
    Rel(neural_operator, math_kernel, "Uses basis functions, projections")
    
    Rel(training_pipeline, data_layer, "Loads training data")
    Rel(poc_framework, neural_operator, "Validates claims")
    Rel(poc_framework, data_layer, "Generates physics data")
    
    Rel(training_pipeline, checkpoint_store, "Saves/loads models")
    Rel(poc_framework, results_store, "Persists results")
    
    Rel(neural_operator, compute, "GPU execution")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Container Responsibilities

| Container | Responsibility | Key Technologies |
|-----------|----------------|------------------|
| **CLI Entrypoints** | User interface for training, benchmarking, and experiments | Python, argparse, Hydra |
| **GTP Server** | Game playing interface compatible with Go GUIs | Python, GTP Protocol |
| **Neural Operator Model** | Resolution-independent position evaluation | PyTorch, Galerkin Attention, FNet |
| **MCTS Search Engine** | Tree search with neural guidance | Python, NumPy |
| **Training Pipeline** | Model training via self-play and supervised learning | PyTorch, distributed training |
| **Math Kernel** | Mathematical foundations and operators | NumPy, SciPy, FFT |
| **PoC Framework** | Validates mathematical claims through experiments | Pydantic, structlog |
| **Data Layer** | Data loading and preprocessing | PyTorch Dataset, padding/masking |

---

## Level 3: Component Diagram - Neural Operator Model

This diagram shows the internal components of the Neural Operator Model container.

```mermaid
C4Component
    title Component Diagram - Neural Operator Model

    Container_Boundary(neural_operator, "Neural Operator Model") {
        Component(model, "AlphaGalerkinModel", "PyTorch nn.Module", "Main model orchestrating all components")
        
        Component(embedding, "Continuous Embedding", "PyTorch Layer", "Maps discrete board to continuous domain with Fourier features")
        
        Component(galerkin_attention, "Galerkin Attention", "PyTorch Layer", "O(N) attention via Petrov-Galerkin projection for global influence")
        
        Component(softmax_attention, "Softmax Attention", "PyTorch Layer", "Traditional attention for local tactical reading")
        
        Component(fnet_block, "FNet Mixing Block", "PyTorch Layer", "O(N log N) FFT-based mixing for fast rollouts")
        
        Component(stability_guard, "LBB Stability Guard", "PyTorch Module", "Monitors inf-sup condition during training")
        
        Component(policy_head, "Policy Head", "PyTorch Layer", "Outputs move probability distribution")
        
        Component(value_head, "Value Head", "PyTorch Layer", "Outputs position evaluation [-1, 1]")
        
        Component(adapter, "Resolution Adapter", "Python Module", "Spectral filtering for zero-shot transfer")
    }

    Component_Ext(math_kernel, "Math Kernel", "Basis functions, projections")
    Component_Ext(training_loss, "Training Loss", "Policy CE + Value MSE + LBB regularization")

    Rel(model, embedding, "Embeds input")
    Rel(embedding, galerkin_attention, "Feeds features")
    Rel(galerkin_attention, fnet_block, "Mixes features")
    Rel(fnet_block, softmax_attention, "Refines features")
    Rel(softmax_attention, policy_head, "Generates policy")
    Rel(softmax_attention, value_head, "Generates value")
    
    Rel(galerkin_attention, stability_guard, "Monitors LBB")
    Rel(model, adapter, "Adapts resolution")
    
    Rel(embedding, math_kernel, "Uses Fourier basis")
    Rel(galerkin_attention, math_kernel, "Uses Galerkin projection")
    Rel(adapter, math_kernel, "Uses spectral filtering")
    
    Rel(stability_guard, training_loss, "Adds regularization term")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Component Descriptions

| Component | Responsibility | Mathematical Foundation |
|-----------|----------------|-------------------------|
| **Continuous Embedding** | Maps discrete grid to Fourier features on [0,1]² | Fourier positional encoding |
| **Galerkin Attention** | O(N) global influence modeling | Petrov-Galerkin projection, Monte Carlo integral |
| **Softmax Attention** | Local tactical reading with injectivity | Standard attention mechanism |
| **FNet Mixing** | Fast feature mixing via FFT | Spectral methods, O(N log N) |
| **Stability Guard** | Ensures well-posed learning | LBB inf-sup condition: dim(K) ≥ dim(Q) |
| **Policy Head** | Move distribution prediction | Cross-entropy loss |
| **Value Head** | Position evaluation | MSE loss |
| **Resolution Adapter** | Zero-shot board size transfer | Anti-aliasing, frequency filtering |

---

## Level 3: Component Diagram - Training Pipeline

This diagram shows the internal components of the Training Pipeline container.

```mermaid
C4Component
    title Component Diagram - Training Pipeline

    Container_Boundary(training_pipeline, "Training Pipeline") {
        Component(trainer, "Trainer", "Python Class", "Main training loop orchestration")
        
        Component(self_play, "Self-Play Engine", "Python Module", "Generates training games using MCTS")
        
        Component(replay_buffer, "Experience Replay Buffer", "Python Class", "Stores and samples game experiences")
        
        Component(loss_fn, "AlphaGalerkin Loss", "PyTorch Module", "Combined loss: policy CE + value MSE + LBB regularization")
        
        Component(checkpoint_mgr, "Checkpoint Manager", "Python Module", "Saves/loads models with rotation")
        
        Component(evaluator, "Model Evaluator", "Python Module", "Win rate and policy agreement metrics")
        
        Component(optimizer, "Optimizer", "PyTorch", "Adam optimizer with learning rate scheduling")
    }

    Component_Ext(neural_operator, "Neural Operator Model", "Model being trained")
    Component_Ext(mcts, "MCTS Engine", "Used for self-play")
    Component_Ext(data_layer, "Data Layer", "Batching and collation")
    ComponentDb_Ext(checkpoint_store, "Checkpoint Store", "Model persistence")

    Rel(trainer, self_play, "Generates games")
    Rel(self_play, mcts, "Uses for move selection")
    Rel(self_play, replay_buffer, "Stores experiences")
    
    Rel(trainer, replay_buffer, "Samples batches")
    Rel(replay_buffer, data_layer, "Uses collation")
    
    Rel(trainer, neural_operator, "Forward pass")
    Rel(neural_operator, loss_fn, "Computes loss")
    Rel(loss_fn, optimizer, "Backpropagates")
    Rel(optimizer, neural_operator, "Updates weights")
    
    Rel(trainer, evaluator, "Evaluates periodically")
    Rel(evaluator, neural_operator, "Tests model")
    
    Rel(trainer, checkpoint_mgr, "Saves checkpoints")
    Rel(checkpoint_mgr, checkpoint_store, "Persists to disk")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Training Pipeline Components

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| **Trainer** | Main training loop with logging | Python class with Hydra config |
| **Self-Play Engine** | Generates training data via MCTS | Parallel game execution |
| **Replay Buffer** | Experience storage and sampling | Uniform and prioritized replay |
| **Loss Function** | Multi-objective optimization | Policy CE + Value MSE + LBB term |
| **Checkpoint Manager** | Model persistence with best tracking | File I/O with rotation policy |
| **Model Evaluator** | Performance metrics | Win rate, policy agreement |
| **Optimizer** | Weight updates | Adam with warmup and decay |

---

## Level 3: Component Diagram - PoC Framework

This diagram shows the internal components of the Proof-of-Concept Framework container.

```mermaid
C4Component
    title Component Diagram - PoC Framework

    Container_Boundary(poc_framework, "PoC Framework") {
        Component(cli_poc, "PoC CLI", "Python argparse", "Command-line interface: run, list, info, compare")
        
        Component(registry, "Scenario Registry", "Python Module", "Discovers and manages available scenarios")
        
        Component(runner, "Scenario Runner", "Python Module", "Executes scenarios with parallel support")
        
        Component(config_mgr, "Config Manager", "Pydantic", "Validates and loads scenario configurations")
        
        Component(results_collector, "Results Collector", "Python Module", "Aggregates and persists experiment results")
        
        Component(logger, "Structured Logger", "structlog", "High-signal logging with context")
        
        Component(scenario_transfer, "Transfer Scenario", "Python Class", "Validates zero-shot transfer (9x9 → 19x19)")
        
        Component(scenario_complexity, "Complexity Scenario", "Python Class", "Validates O(N) vs O(N²) complexity")
        
        Component(scenario_stability, "Stability Scenario", "Python Class", "Monitors LBB condition during training")
    }

    Component_Ext(neural_operator, "Neural Operator Model", "Model under test")
    Component_Ext(physics_data, "Physics Data Generator", "Synthetic Poisson equation data")
    ComponentDb_Ext(results_store, "Results Store", "JSON/YAML files")

    Rel(cli_poc, registry, "Lists scenarios")
    Rel(cli_poc, runner, "Executes scenarios")
    Rel(cli_poc, config_mgr, "Loads configs")
    
    Rel(registry, scenario_transfer, "Registers")
    Rel(registry, scenario_complexity, "Registers")
    Rel(registry, scenario_stability, "Registers")
    
    Rel(runner, scenario_transfer, "Runs")
    Rel(runner, scenario_complexity, "Runs")
    Rel(runner, scenario_stability, "Runs")
    
    Rel(scenario_transfer, neural_operator, "Tests transfer")
    Rel(scenario_complexity, neural_operator, "Benchmarks complexity")
    Rel(scenario_stability, neural_operator, "Monitors stability")
    
    Rel(scenario_transfer, physics_data, "Generates test data")
    
    Rel(runner, results_collector, "Collects results")
    Rel(results_collector, results_store, "Persists results")
    Rel(results_collector, logger, "Logs outcomes")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### PoC Framework Components

| Component | Responsibility | Purpose |
|-----------|----------------|---------|
| **PoC CLI** | User interface for running experiments | `run`, `list`, `info`, `compare` commands |
| **Scenario Registry** | Discovery and management of scenarios | Auto-registration, metadata tracking |
| **Scenario Runner** | Parallel execution of experiments | Worker pool, timeout handling |
| **Config Manager** | Configuration validation | Pydantic schemas, YAML/Python configs |
| **Results Collector** | Aggregation and persistence | JSON/YAML output, comparison tools |
| **Transfer Scenario** | Zero-shot transfer validation | Train 9x9 → eval 19x19, MSE < 0.05 |
| **Complexity Scenario** | O(N) complexity verification | Timing benchmarks, scaling analysis |
| **Stability Scenario** | LBB condition monitoring | Singular value tracking, β > 0 check |

---

## Level 3: Component Diagram - Math Kernel

This diagram shows the mathematical primitives that underpin the system.

```mermaid
C4Component
    title Component Diagram - Math Kernel

    Container_Boundary(math_kernel, "Math Kernel") {
        Component(fourier_basis, "Fourier Basis Functions", "NumPy/PyTorch", "Resolution-independent positional encoding")
        
        Component(galerkin_projection, "Galerkin Projection", "PyTorch", "Monte Carlo integral approximation for operators")
        
        Component(spectral_filter, "Spectral Filter", "PyTorch FFT", "Anti-aliasing for resolution transfer")
        
        Component(lbb_checker, "LBB Condition Checker", "NumPy/PyTorch", "Computes inf-sup constant β")
        
        Component(fredholm_kernel, "Fredholm Kernel", "Math Functions", "Green's function formulation for influence")
        
        Component(monte_carlo_integral, "Monte Carlo Integral", "NumPy", "Numerical integration with 1/n normalization")
    }

    Component_Ext(neural_operator, "Neural Operator Model", "Uses math primitives")
    Component_Ext(poc_framework, "PoC Framework", "Validates mathematical properties")

    Rel(neural_operator, fourier_basis, "Encodes positions")
    Rel(neural_operator, galerkin_projection, "Projects values")
    Rel(neural_operator, spectral_filter, "Adapts resolution")
    Rel(neural_operator, lbb_checker, "Monitors stability")
    
    Rel(galerkin_projection, monte_carlo_integral, "Approximates integral")
    Rel(galerkin_projection, fredholm_kernel, "Uses Green's function")
    
    Rel(poc_framework, lbb_checker, "Validates β > 0")
    Rel(poc_framework, spectral_filter, "Tests transfer quality")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Math Kernel Components

| Component | Mathematical Foundation | Purpose |
|-----------|------------------------|---------|
| **Fourier Basis** | $\phi_k(x) = e^{2\pi i k \cdot x}$ | Resolution-independent encoding |
| **Galerkin Projection** | $\langle Lu, v \rangle = \langle f, v \rangle$ | O(N) operator approximation |
| **Spectral Filter** | Low-pass filter in frequency domain | Anti-aliasing for transfer |
| **LBB Checker** | $\inf_u \sup_v \frac{\langle Lu, v \rangle}{\\|u\\| \\|v\\|} \geq \beta$ | Stability guarantee |
| **Fredholm Kernel** | $u(x) = \int K(x,y) f(y) dy$ | Influence field modeling |
| **Monte Carlo Integral** | $\frac{1}{n} \sum_{i=1}^n f(x_i)$ | Numerical integration |

---

## Level 4: Code Diagram - Galerkin Attention

This diagram shows the implementation details of the Galerkin Attention component.

```mermaid
classDiagram
    class GalerkinLinearAttention {
        +int d_model
        +int n_heads
        +int d_head
        +Linear query_proj
        +Linear key_proj
        +Linear value_proj
        +Linear output_proj
        +forward(x: Tensor) Tensor
        +_compute_context(K: Tensor, V: Tensor) Tensor
        +_check_lbb_stability(K: Tensor, Q: Tensor) float
    }

    class MultiHeadAttention {
        <<interface>>
        +forward(x: Tensor) Tensor
    }

    class FourierEncoding {
        +int n_features
        +Tensor freqs
        +encode(positions: Tensor) Tensor
    }

    class MonteCarloIntegral {
        +approximate(K: Tensor, V: Tensor) Tensor
        +_normalize(result: Tensor, n: int) Tensor
    }

    class LBBStabilityGuard {
        +float beta_threshold
        +check_stability(K: Tensor, Q: Tensor) bool
        +compute_singular_values(M: Tensor) Tensor
    }

    class GalerkinAttentionBlock {
        +GalerkinLinearAttention attention
        +LayerNorm norm1
        +MLP ffn
        +LayerNorm norm2
        +forward(x: Tensor) Tensor
    }

    MultiHeadAttention <|-- GalerkinLinearAttention : implements
    GalerkinLinearAttention --> MonteCarloIntegral : uses
    GalerkinLinearAttention --> LBBStabilityGuard : uses
    GalerkinLinearAttention --> FourierEncoding : uses
    GalerkinAttentionBlock *-- GalerkinLinearAttention : contains
```

### Key Implementation Details

**Galerkin Attention Algorithm:**
```python
# Step 1: Project to Query, Key, Value spaces
Q = query_proj(x)    # (batch, n, d_head)
K = key_proj(x)      # (batch, n, d_head)
V = value_proj(x)    # (batch, n, d_head)

# Step 2: Monte Carlo integral approximation
# Context = K^T V / n  (not K^T V / sqrt(d))
Context = einsum('bnd,bnm->bdm', K, V) / n

# Step 3: Reconstruct in Query basis
Output = einsum('bnd,bdm->bnm', Q, Context)

# Step 4: LBB stability check (training only)
if training:
    beta = compute_inf_sup_constant(K, Q)
    assert beta > beta_threshold
```

**Complexity Analysis:**
- Standard Attention: O(N² × d)
- Galerkin Attention: O(N × d²)
- For typical Go: N=361, d=32 → **10x speedup**

---

## Data Flow Diagrams

### Training Data Flow

```mermaid
flowchart TB
    subgraph SelfPlay["Self-Play Generation"]
        SP1[Start Position] --> SP2[MCTS Search]
        SP2 --> SP3[Select Move]
        SP3 --> SP4[Execute Move]
        SP4 --> SP5{Game Over?}
        SP5 -->|No| SP2
        SP5 -->|Yes| SP6[Record Game]
    end

    subgraph ReplayBuffer["Replay Buffer"]
        RB1[Store Experience]
        RB2[Sample Batch]
        RB3[Priority Update]
    end

    subgraph Training["Training Loop"]
        T1[Forward Pass]
        T2[Compute Loss]
        T3[Backpropagation]
        T4[Update Weights]
    end

    SP6 --> RB1
    RB1 --> RB2
    RB2 --> T1
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> RB3
    RB3 --> RB1
    T4 --> SP1

    style SelfPlay fill:#e1f5ff
    style ReplayBuffer fill:#fff5e1
    style Training fill:#ffe1f5
```

### Inference Data Flow

```mermaid
flowchart LR
    Input[Board State<br/>17×H×W] --> Embed[Continuous<br/>Embedding]
    Embed --> Galerkin[Galerkin<br/>Attention<br/>6 layers]
    Galerkin --> FNet[FNet<br/>Mixing]
    FNet --> Softmax[Softmax<br/>Attention<br/>2 layers]
    
    Softmax --> Policy[Policy Head<br/>361+1 moves]
    Softmax --> Value[Value Head<br/>[-1, 1]]
    
    Policy --> MCTS[MCTS<br/>Search]
    Value --> MCTS
    
    MCTS --> Move[Best Move]
    
    style Input fill:#e1f5ff
    style Policy fill:#ffe1e1
    style Value fill:#e1ffe1
    style Move fill:#ffe1f5
```

### Resolution Transfer Flow

```mermaid
flowchart TB
    subgraph Train["Training on 9×9"]
        T1[9×9 Board] --> T2[Fourier Encoding]
        T2 --> T3[Galerkin Model]
        T3 --> T4[Train Weights]
    end

    subgraph Adapt["Resolution Adaptation"]
        A1[Save Frequency<br/>Representations]
        A2[Apply Spectral<br/>Filter]
        A3[Adjust Monte Carlo<br/>Normalization]
    end

    subgraph Infer["Inference on 19×19"]
        I1[19×19 Board] --> I2[Fourier Encoding]
        I2 --> I3[Galerkin Model<br/>Same Weights]
        I3 --> I4[Policy/Value]
    end

    T4 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> I3

    style Train fill:#e1f5ff
    style Adapt fill:#ffe1e1
    style Infer fill:#e1ffe1
```

---

## Deployment Diagram

```mermaid
C4Deployment
    title Deployment Diagram - AlphaGalerkin

    Deployment_Node(dev_machine, "Developer Machine", "Linux/Mac/Windows") {
        Container(cli_dev, "CLI Tools", "Python 3.10+")
        Container(jupyter, "Jupyter Notebooks", "Interactive Analysis")
    }

    Deployment_Node(training_cluster, "Training Cluster", "GPU Cluster") {
        Deployment_Node(gpu_node1, "GPU Node 1", "8× A100 GPUs") {
            Container(trainer1, "Trainer Instance 1", "PyTorch Distributed")
        }
        Deployment_Node(gpu_node2, "GPU Node 2", "8× A100 GPUs") {
            Container(trainer2, "Trainer Instance 2", "PyTorch Distributed")
        }
        ContainerDb(shared_storage, "Shared Storage", "NFS/S3", "Model checkpoints, experiment results")
    }

    Deployment_Node(inference_server, "Inference Server", "Cloud/Edge Device") {
        Container(gtp_server_deploy, "GTP Server", "Python FastAPI")
        Container(model_inference, "Model Inference", "PyTorch + ONNX")
    }

    Deployment_Node(client_machine, "Client Machine", "Desktop/Laptop") {
        Container(go_gui_deploy, "Go GUI", "Sabaki/GoGui")
    }

    Rel(cli_dev, trainer1, "Submits training jobs", "SSH/SLURM")
    Rel(trainer1, shared_storage, "Saves checkpoints")
    Rel(trainer2, shared_storage, "Saves checkpoints")
    Rel(trainer1, trainer2, "Synchronizes gradients", "NCCL/Gloo")
    
    Rel(gtp_server_deploy, model_inference, "Gets predictions")
    Rel(model_inference, shared_storage, "Loads models")
    Rel(go_gui_deploy, gtp_server_deploy, "Sends GTP commands", "TCP")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

---

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Deep Learning** | PyTorch 2.0+ | Neural network implementation |
| **Numerical Computing** | NumPy, SciPy | Mathematical operations |
| **Configuration** | Hydra, Pydantic | Configuration management and validation |
| **Testing** | pytest, hypothesis | Unit and property-based testing |
| **Logging** | structlog | Structured logging |
| **Type Checking** | mypy, jaxtyping | Static type analysis |
| **Code Quality** | ruff | Linting and formatting |

### Key Libraries

```python
# Core dependencies
torch >= 2.0.0          # Deep learning framework
einops >= 0.7.0         # Tensor operations
jaxtyping >= 0.2.25     # Type annotations for arrays
pydantic >= 2.0.0       # Data validation
hydra-core >= 1.3.0     # Configuration management
structlog >= 23.0.0     # Structured logging
numpy >= 1.24.0         # Numerical computing
```

---

## Architecture Principles

### 1. Resolution Independence
- **Continuous Domain**: Treat board as Ω = [0,1]² rather than discrete grid
- **Fourier Encoding**: Position-independent frequency representation
- **Spectral Methods**: Proper anti-aliasing and frequency filtering

### 2. Mathematical Rigor
- **Galerkin Projection**: Well-founded operator approximation theory
- **LBB Stability**: Monitored inf-sup condition ensures convergence
- **Fredholm Operators**: Integral equation formulation for influence

### 3. Performance Optimization
- **O(N) Attention**: Linear complexity via Petrov-Galerkin projection
- **FFT Mixing**: O(N log N) spectral mixing for fast rollouts
- **CUDA Acceleration**: Full GPU utilization for training and inference

### 4. Testability
- **Property-Based Tests**: Mathematical properties verified with Hypothesis
- **PoC Framework**: Reproducible validation of core claims
- **Modular Design**: Independent testing of components

### 5. Configurability
- **Hydra Integration**: Hierarchical configuration management
- **Pydantic Schemas**: Runtime validation of parameters
- **Environment Variables**: Deployment-specific overrides

---

## Key Architectural Decisions

### Decision 1: Galerkin vs Standard Attention
- **Context**: Need O(N) complexity for large board sizes
- **Decision**: Use Petrov-Galerkin projection instead of softmax
- **Rationale**: Reduces complexity from O(N²d) to O(Nd²)
- **Trade-offs**: Requires careful normalization (1/n, not 1/√d)

### Decision 2: Hybrid Architecture (Galerkin + Softmax)
- **Context**: Balance global strategy and local tactics
- **Decision**: Galerkin layers for strategy, softmax for tactics
- **Rationale**: Galerkin captures long-range influence, softmax preserves injectivity for life/death
- **Trade-offs**: More complex than uniform architecture

### Decision 3: FNet for Fast Rollouts
- **Context**: MCTS requires thousands of neural evaluations
- **Decision**: FFT-based mixing as alternative to attention
- **Rationale**: 5× speedup for leaf evaluation
- **Trade-offs**: Slightly lower accuracy vs full attention

### Decision 4: PoC Framework for Validation
- **Context**: Need reproducible validation of mathematical claims
- **Decision**: Config-driven scenario framework
- **Rationale**: Ensures claims are testable and reproducible
- **Trade-offs**: Additional infrastructure complexity

### Decision 5: Pydantic for Configuration
- **Context**: Complex hyperparameter space with mathematical constraints
- **Decision**: Pydantic schemas with validators
- **Rationale**: Runtime validation, type safety, IDE support
- **Trade-offs**: More verbose than plain dicts

---

## Future Architecture Enhancements

### Planned Improvements

1. **Distributed Training**
   - Multi-node self-play generation
   - Gradient aggregation via NCCL
   - Model zoo for curriculum learning

2. **ONNX Export**
   - Convert PyTorch models to ONNX
   - Deploy on edge devices (Raspberry Pi, Jetson)
   - Quantization for int8 inference

3. **Multi-Game Support**
   - Abstract game interface
   - Support for Chess, Shogi, etc.
   - Shared continuous operator core

4. **Advanced MCTS**
   - Gumbel AlphaZero search
   - Value-based exploration
   - Policy improvement operators

5. **Enhanced PoC Framework**
   - Automated hyperparameter tuning
   - Statistical significance testing
   - Comparative visualizations

---

## References

- **C4 Model**: [c4model.com](https://c4model.com)
- **Galerkin Transformers**: Cao et al. (2021)
- **FNet**: Lee-Thorp et al. (2021)
- **AlphaZero**: Silver et al. (2017)
- **Fredholm Theory**: Classical operator theory

---

## Document Metadata

- **Version**: 1.0.0
- **Created**: 2026-01-26
- **Format**: Mermaid C4 Diagrams
- **Status**: Complete
- **Audience**: Developers, Researchers, Technical Stakeholders
